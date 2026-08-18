"""Parse the eLobbyist feed and match its clients to resolved DOE vendors.

Two jobs, and the second one is where the risk lives.

PARSING. The feed's ``lobbyist_targets`` and ``lobbyist_activities`` cells are
denormalised free text with heavy repetition -- one observed cell held the same
council member fourteen times across ten kilobytes. Both are semicolon-joined
and both need exploding and deduplicating before any count is meaningful.

MATCHING. Joining a lobbying client to a Checkbook payee is *cross-system*
entity resolution, which is a harder problem than the within-Checkbook matching
in ``resolve.py`` and carries a different kind of risk. A bad merge inside
Checkbook miscounts a concentration statistic. A bad merge here asserts that a
named firm lobbied the agency that then handed it a no-bid contract. That is a
defamation-shaped error, so every rule below is built to fail toward "no match"
and every accepted match carries an explicit confidence and method.

The variant classes that actually appear between the two systems:

    legal form      Pearson Education, Inc.    ~ PEARSON EDUCATION
    abbreviation    INTL BUSINESS MACHINES     ~ International Business Machines
    acronym         IBM                        ~ INTERNATIONAL BUSINESS MACHINES
    spelling        CENTRE FOR ...             ~ CENTER FOR ...
    plural drift    EDUCATIONAL NETWORK        ~ Educational Networks, Inc.
    beneficiary     Carahsoft ... for the benefit of Elastic

That last one is specific to this feed: the registered client is often a
lobbying firm or reseller acting for someone else, so the name that matters for
procurement is buried mid-string. Both sides are kept and matched separately.

Guards that run before any similarity test, inherited from ``resolve.py`` and
extended:

* **Ordinal guard.** Differing trailing ordinals never merge -- charter network
  siblings are distinct legal entities.
* **Embedded-number guard.** Differing digit runs anywhere in the name block a
  merge. ``P.S. 123`` is not ``P.S. 456``.
* **Distinctiveness floor.** A match carried entirely by generic words
  ("services", "group", "new york") is rejected. Two firms sharing only
  ``EDUCATIONAL SERVICES`` are not the same firm.

Pairs that are close but not close enough go to a review queue rather than
being silently dropped, because a missed match and a wrong match are both
errors and only one of them is visible in the output.
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from normalize import normalize_vendor  # noqa: E402
from resolve import ordinal_of  # noqa: E402

# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------

# "Education, Department of (DOE)  David Banks" -> agency, code, official.
_TARGET_CODED = re.compile(r"^(?P<agency>.+?)\s*\((?P<code>[A-Z0-9/&-]{2,10})\)\s*(?P<who>.*)$")

DOE_TARGET_PATTERNS = ("EDUCATION, DEPARTMENT OF", "(DOE)")

# The activity taxonomy the City Clerk actually emits. ``Procurement`` is a
# first-class category here, which is what makes this feed usable at all.
PROCUREMENT_CATEGORY = "Procurement"

_BENEFICIARY_RE = re.compile(
    r"\s+(?:on behalf of|for the benefit of|obo|o/b/o)\s+", re.IGNORECASE)


def split_beneficiary(client_name: str) -> tuple[str, str | None]:
    """``X on behalf of Y`` -> ``(X, Y)``. Returns ``(name, None)`` if absent."""
    if not client_name:
        return "", None
    parts = _BENEFICIARY_RE.split(client_name, maxsplit=1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return client_name.strip(), None


def parse_targets(blob: str | None) -> list[dict]:
    """Explode a targets cell into distinct ``{agency, code, official}`` records."""
    if not blob:
        return []
    seen, out = set(), []
    for chunk in str(blob).split(";"):
        # The feed separates agency from official with a double space, so
        # whitespace can be tidied but that separator has to survive: collapsing
        # everything to single spaces silently merges the two fields.
        chunk = re.sub(r"[^\S ]+", " ", chunk).strip()
        chunk = re.sub(r" {2,}", "  ", chunk)
        if not chunk:
            continue
        match = _TARGET_CODED.match(chunk)
        if match:
            agency = match.group("agency").strip().rstrip(",")
            code = match.group("code").strip()
            official = match.group("who").strip()
        else:
            # Uncoded targets ("NYC Council Members Ben Kallos - District No. 5")
            # split agency from person on the double space the feed uses.
            pieces = re.split(r"\s{2,}", chunk, maxsplit=1)
            agency, code = pieces[0].strip(), ""
            official = pieces[1].strip() if len(pieces) > 1 else ""
        key = (agency.upper(), code.upper(), official.upper())
        if key in seen:
            continue  # the same target repeats many times per cell
        seen.add(key)
        out.append({"agency": agency, "code": code, "official": official})
    return out


def parse_activities(blob: str | None) -> list[dict]:
    """Explode an activities cell into distinct ``{category, detail}`` records."""
    if not blob:
        return []
    seen, out = set(), []
    for chunk in str(blob).split(";"):
        chunk = " ".join(chunk.split())
        if not chunk:
            continue
        category, _, detail = chunk.partition(" - ")
        # Some cells leak a trailing fragment of the previous item before the
        # category, e.g. "...Human Services sector  , Procurement".
        category = category.split(",")[-1].strip() or category.strip()
        key = (category.upper(), detail.upper()[:120])
        if key in seen:
            continue
        seen.add(key)
        out.append({"category": category, "detail": detail})
    return out


def targets_doe(blob: str | None) -> bool:
    text = (blob or "").upper()
    return any(pattern in text for pattern in DOE_TARGET_PATTERNS)


def has_procurement(blob: str | None) -> bool:
    return PROCUREMENT_CATEGORY.upper() in (blob or "").upper()


# --------------------------------------------------------------------------
# cross-system name canonicalisation
# --------------------------------------------------------------------------

ABBREVIATIONS = {
    "INTL": "INTERNATIONAL", "NATL": "NATIONAL", "MGMT": "MANAGEMENT",
    "SVCS": "SERVICES", "SVC": "SERVICES", "SERV": "SERVICES", "SERVS": "SERVICES",
    "CTR": "CENTER", "CTRS": "CENTERS", "CNTR": "CENTER",
    "INST": "INSTITUTE", "UNIV": "UNIVERSITY", "ACAD": "ACADEMY",
    "DEPT": "DEPARTMENT", "BROS": "BROTHERS", "MFG": "MANUFACTURING",
    "AMER": "AMERICAN", "ASSOC": "ASSOCIATES", "ASSN": "ASSOCIATION",
    "TECH": "TECHNOLOGY", "TECHS": "TECHNOLOGIES", "SYS": "SYSTEMS",
    "SOLS": "SOLUTIONS", "GRP": "GROUP", "CONSULT": "CONSULTING",
    "CONSTR": "CONSTRUCTION", "TRANS": "TRANSPORTATION", "TRANSP": "TRANSPORTATION",
    "EDUC": "EDUCATION", "PROD": "PRODUCTS", "INDUS": "INDUSTRIES",
    "ENTERPR": "ENTERPRISES", "DEV": "DEVELOPMENT", "MED": "MEDICAL",
    "LABS": "LABORATORIES", "LAB": "LABORATORY", "ELEC": "ELECTRIC",
    "ENVIRO": "ENVIRONMENTAL", "ENV": "ENVIRONMENTAL",
    "COMM": "COMMUNICATIONS", "COMMS": "COMMUNICATIONS", "PUBL": "PUBLISHING",
    "MTN": "MOUNTAIN", "NY": "NEW YORK", "NYC": "NEW YORK CITY",
    "USA": "UNITED STATES", "MT": "MOUNT",
}
# Deliberately excluded: "ST" (Saint vs Street is genuinely ambiguous) and
# "CO" / "CORP" / "INC" (already stripped as legal suffixes upstream).

SPELLING = {
    "CENTRE": "CENTER", "CENTRES": "CENTERS", "ORGANISATION": "ORGANIZATION",
    "ORGANISATIONS": "ORGANIZATIONS", "BEHAVIOUR": "BEHAVIOR",
    "BEHAVIOURAL": "BEHAVIORAL", "PROGRAMME": "PROGRAM", "PROGRAMMES": "PROGRAMS",
    "LICENCE": "LICENSE", "DEFENCE": "DEFENSE", "CATALOGUE": "CATALOG",
    "COUNSELLING": "COUNSELING", "LABOUR": "LABOR", "THEATRE": "THEATER",
}

# Explicit aliases where an acronym is not derivable from the tokens, or where
# the derivation would be too risky to trust automatically.
ALIASES = {
    "IBM": "INTERNATIONAL BUSINESS MACHINES",
    "AMNH": "AMERICAN MUSEUM OF NATURAL HISTORY",
    "ATU": "AMALGAMATED TRANSIT UNION",
    "CUNY": "CITY UNIVERSITY OF NEW YORK",
    "SUNY": "STATE UNIVERSITY OF NEW YORK",
    "EY": "ERNST & YOUNG",
    "PWC": "PRICEWATERHOUSECOOPERS",
    "ADP": "AUTOMATIC DATA PROCESSING",
    "HP": "HEWLETT PACKARD",
    "GE": "GENERAL ELECTRIC",
    "AWS": "AMAZON WEB SERVICES",
    "SCA": "SCHOOL CONSTRUCTION AUTHORITY",
    "DOE": "DEPARTMENT OF EDUCATION",
}

# Words that carry no identifying power on their own. Used only to decide
# whether a match rests on something distinctive -- never removed before
# comparison, because "EDUCATION" is noise in isolation but real in "PEARSON
# EDUCATION".
GENERIC = {
    "SERVICES", "SERVICE", "GROUP", "SOLUTIONS", "SOLUTION", "SYSTEMS", "SYSTEM",
    "ASSOCIATES", "ASSOCIATION", "CONSULTING", "CONSULTANTS", "MANAGEMENT",
    "NEW", "YORK", "CITY", "AMERICA", "AMERICAN", "NATIONAL", "INTERNATIONAL",
    "EDUCATION", "EDUCATIONAL", "SCHOOL", "SCHOOLS", "LEARNING", "ACADEMY",
    "TECHNOLOGY", "TECHNOLOGIES", "CENTER", "CENTERS", "PARTNERS", "PARTNERSHIP",
    "ENTERPRISES", "INDUSTRIES", "PRODUCTS", "COMPANY", "HOLDINGS", "GLOBAL",
    "OF", "THE", "AND", "FOR", "&", "IN", "AT", "TO", "A", "INSTITUTE",
    "FOUNDATION", "PROGRAM", "PROGRAMS", "PROJECT", "PROJECTS", "COMMUNITY",
    "DEVELOPMENT", "MANAGEMENTS", "CORPORATION", "INCORPORATED",
}

STOPWORDS = {"OF", "THE", "AND", "FOR", "&", "IN", "AT", "TO", "A", "ON"}

_DIGITS_RE = re.compile(r"\d+")


def _singular(token: str) -> str:
    if len(token) > 4 and token.endswith("IES"):
        return token[:-3] + "Y"
    if len(token) > 4 and token.endswith("SES"):
        return token[:-2]
    if len(token) > 4 and token.endswith("S") and not token.endswith("SS"):
        return token[:-1]
    return token


def canonical_tokens(name: str) -> tuple[str, ...]:
    """Comparison form: normalised, abbreviations expanded, singularised."""
    key = normalize_vendor(ALIASES.get(normalize_vendor(name), name))
    tokens: list[str] = []
    for raw in key.split():
        token = ABBREVIATIONS.get(raw, raw)
        token = SPELLING.get(token, token)
        # An expansion can introduce multiple words ("NY" -> "NEW YORK").
        for piece in token.split():
            piece = _singular(SPELLING.get(piece, piece))
            # Stripping punctuation upstream turns "CHILDREN'S" into
            # "CHILDREN S", so a bare S is a possessive remnant, not a token.
            if piece == "S" and tokens:
                continue
            tokens.append(piece)
    return tuple(tokens)


def distinctive(tokens: tuple[str, ...]) -> set[str]:
    return {t for t in tokens if t not in GENERIC and len(t) > 2}


def acronym(tokens: tuple[str, ...]) -> str:
    return "".join(t[0] for t in tokens if t not in STOPWORDS)


def numbers_in(tokens: tuple[str, ...]) -> set[str]:
    out: set[str] = set()
    for token in tokens:
        out.update(_DIGITS_RE.findall(token))
    return out


def blocked(a_name: str, b_name: str,
            a_tokens: tuple[str, ...], b_tokens: tuple[str, ...]) -> bool:
    """Hard guards. True means these two may never merge, at any similarity."""
    if ordinal_of(normalize_vendor(a_name)) != ordinal_of(normalize_vendor(b_name)):
        return True
    if numbers_in(a_tokens) != numbers_in(b_tokens):
        return True
    return False


# --------------------------------------------------------------------------
# matching
# --------------------------------------------------------------------------

# Only *equivalence* relations may match automatically: two renderings of one
# name, differing by legal form, abbreviation, spelling, word order, or a
# truncated field. Every other relation -- containment, partial token overlap,
# a derived initialism -- goes to human review no matter how high it scores.
#
# That asymmetry is deliberate and was forced by the data. Containment scored
# well and was wrong about half the time: "Red Hat, Inc." is contained in
# "RED HAT DAY CARE CENTER", "Verizon Corporate Resources Group" in "CORPORATE
# RESOURCE DEVELOPMENT". Both would have asserted that a named firm lobbied the
# agency that then gave it a no-bid contract. A missed match costs a row in the
# review queue; a wrong one costs a great deal more.
ACCEPT_METHODS = {"exact", "canonical", "reordered", "truncation"}
ACCEPT_FLOOR = 0.80
REVIEW_FLOOR = 0.50


def score_pair(a_name: str, b_name: str) -> tuple[float, str]:
    """Confidence in ``a`` and ``b`` being the same firm, plus the method used."""
    a_key, b_key = normalize_vendor(a_name), normalize_vendor(b_name)
    if not a_key or not b_key:
        return 0.0, "empty"
    if a_key == b_key:
        return 1.00, "exact"

    a_tok, b_tok = canonical_tokens(a_name), canonical_tokens(b_name)
    if not a_tok or not b_tok:
        return 0.0, "empty"
    if blocked(a_name, b_name, a_tok, b_tok):
        return 0.0, "blocked"

    # --- equivalence tiers: auto-acceptable -------------------------------
    if a_tok == b_tok:
        return 0.96, "canonical"
    if sorted(a_tok) == sorted(b_tok):
        return 0.93, "reordered"

    # Field truncation, the documented failure mode in this data: one rendering
    # is a strict prefix of the other and long enough that a collision is
    # implausible. Compared on the joined form so a cut mid-word still lands.
    a_flat, b_flat = "".join(a_tok), "".join(b_tok)
    shorter, longer = sorted((a_flat, b_flat), key=len)
    if (longer.startswith(shorter) and len(shorter) >= 14
            and len(longer) - len(shorter) <= 6):
        return 0.90, "truncation"

    # --- suggestive tiers: review only ------------------------------------
    a_dist, b_dist = distinctive(a_tok), distinctive(b_tok)

    for short, long in ((a_tok, b_tok), (b_tok, a_tok)):
        if len(short) == 1 and len(short[0]) >= 3 and short[0] == acronym(long):
            return 0.78, "acronym-derived"

    if a_dist and b_dist and (a_dist <= b_dist or b_dist <= a_dist):
        overlap = min(len(a_dist), len(b_dist))
        return (0.72 if overlap >= 2 else 0.58), "containment"

    union = a_dist | b_dist
    if not union:
        return 0.0, "generic-only"
    jaccard = len(a_dist & b_dist) / len(union)
    shared = len(a_dist & b_dist)
    if shared >= 2 and jaccard >= 0.70:
        return 0.50 + 0.20 * jaccard, "token-overlap"
    if shared >= 1 and jaccard >= 0.40:
        return 0.35 + 0.20 * jaccard, "token-overlap-weak"
    return jaccard, "insufficient"


def match_clients(client_names: list[str], vendor_keys: list[str]) -> tuple[
        pd.DataFrame, pd.DataFrame]:
    """Match lobbying clients to vendor keys.

    Returns ``(matches, review)``. Blocking is on the first four characters of
    each distinctive token, which keeps this near-linear while still letting a
    truncated or abbreviated name find its counterpart.
    """
    index: dict[str, set[str]] = defaultdict(set)
    for key in vendor_keys:
        for token in distinctive(canonical_tokens(key)) or {"\x00"}:
            index[token[:4]].add(key)

    accepted, review = [], []
    for client in client_names:
        base, beneficiary = split_beneficiary(client)
        best: dict[str, tuple[float, str, str]] = {}

        for side, label in ((base, "client"), (beneficiary, "beneficiary")):
            if not side:
                continue
            candidates: set[str] = set()
            for token in distinctive(canonical_tokens(side)) or {"\x00"}:
                candidates |= index.get(token[:4], set())
            for vendor in candidates:
                confidence, method = score_pair(side, vendor)
                if confidence <= 0:
                    continue
                if vendor not in best or confidence > best[vendor][0]:
                    best[vendor] = (confidence, method, label)

        if not best:
            continue
        vendor, (confidence, method, side) = max(best.items(), key=lambda kv: kv[1][0])
        row = {"client_name": client, "matched_side": side, "vendor_key": vendor,
               "confidence": round(confidence, 3), "method": method}

        # A second candidate scoring nearly as well means the name is ambiguous
        # between two real vendors; that is a review case, not a match.
        runners = sorted((c for v, (c, _, _) in best.items() if v != vendor), reverse=True)
        row["ambiguous"] = bool(runners and runners[0] >= confidence - 0.02)
        row["runner_up"] = round(runners[0], 3) if runners else None

        if (method in ACCEPT_METHODS and confidence >= ACCEPT_FLOOR
                and not row["ambiguous"]):
            accepted.append(row)
        elif confidence >= REVIEW_FLOOR or row["ambiguous"]:
            review.append(row)

    return (pd.DataFrame(accepted).sort_values("confidence", ascending=False)
            if accepted else pd.DataFrame(),
            pd.DataFrame(review).sort_values("confidence", ascending=False)
            if review else pd.DataFrame())


def accepts(a_name: str, b_name: str) -> bool:
    """Would this pair match automatically, without human review?"""
    confidence, method = score_pair(a_name, b_name)
    return method in ACCEPT_METHODS and confidence >= ACCEPT_FLOOR


if __name__ == "__main__":
    cases = [
        # equivalence -- must auto-accept
        ("Pearson Education, Inc.", "PEARSON EDUCATION", True),
        ("The Gordian Group, Inc.", "GORDIAN GROUP", True),
        ("International Business Machines Corporation", "IBM", True),
        ("IBM", "INTERNATIONAL BUSINESS MACHINES", True),
        ("Intl Business Machines", "INTERNATIONAL BUSINESS MACHINES", True),
        ("Educational Networks, Inc.", "EDUCATIONAL NETWORK", True),
        ("Centre For Civic Education", "CENTER FOR CIVIC EDUCATION", True),
        ("Brooklyn Children's Museum", "BROOKLYN CHILDRENS MUSEUM", True),
        ("CENTER FOR CIVIC EDUCATIO", "CENTER FOR CIVIC EDUCATION", True),
        ("Logan Bus Co., Inc.", "LOGAN BUS", True),
        ("N2Y, LLC", "N2Y", True),
        ("Amalgamated Transit Union", "ATU", True),
        # distinct entities -- hard guards
        ("Manhattan Charter School", "MANHATTAN CHARTER SCHOOL II", False),
        ("P.S. 123 Partners", "PS 456 PARTNERS", False),
        ("Harlem Success Academy 3", "HARLEM SUCCESS ACADEMY", False),
        ("Acme Educational Services", "BETA EDUCATIONAL SERVICES", False),
        ("New York Educational Group", "NEW YORK EDUCATIONAL SOLUTIONS", False),
        # containment false positives observed in the real feed -- these scored
        # well under the old subset rule and are all wrong
        ("Red Hat, Inc.", "RED HAT DAY CARE CENTER", False),
        ("Verizon Corporate Resources Group LLC", "CORPORATE RESOURCE DEVELOPMENT", False),
        ("Hamaspik of Kings County, Inc.", "KINGS COUNTY", False),
        ("Motion Picture Association, Inc.", "SWANK MOTION PICTURES", False),
        ("First Student, Inc.", "STUDENTS FIRST SERVICES", False),
        ("New York Shakespeare Festival", "HUDSON VALLEY SHAKESPEARE FESTIVAL", False),
    ]
    width = max(len(a) for a, _, _ in cases)
    ok = True
    for a, b, expected in cases:
        confidence, method = score_pair(a, b)
        got = accepts(a, b)
        ok &= got == expected
        flag = "ok " if got == expected else "FAIL"
        verdict = "accept" if got else "review"
        print(f"  [{flag}] {confidence:.2f} {verdict} {method:<18} {a:<{width}}  ~  {b}")
    print("\nall passed" if ok else "\nFAILURES present")
