"""Vendor name normalization -- the least glamorous, most consequential step.

The same firm appears as "ACME CORP", "Acme Corp.", and "ACME CORPORATION". If
those stay separate, every concentration number comes out too low and the
entrenchment ranking is wrong. This module collapses them.

Two other classes of payee also have to be handled before any vendor ranking is
believable, and both are easy to miss:

* ``N/A (PRIVACY/SECURITY)`` is a *redacted* payee covering hundreds of
  thousands of rows. It is not a vendor. Left in, it ranks near the top.
* Sister public entities -- the School Construction Authority, NYC Transit, the
  retiree health trust, union welfare funds -- receive the largest transfers in
  the data, but those are intergovernmental transfers, not procurement. A
  concentration finding that counts them is measuring the wrong thing.

Both get flagged here rather than silently dropped, so the analysis can include
or exclude them explicitly and say which it did.
"""

from __future__ import annotations

import re

# Corporate suffixes stripped from the tail of a name, longest first so that
# "INCORPORATED" is not left as a stray "ED" by an earlier "INCORPORAT" match.
LEGAL_SUFFIXES = [
    "INCORPORATED", "CORPORATION", "COMPANY", "LIMITED", "PLLC", "LLC", "LLP",
    "LTD", "INC", "CORP", "INTL", "INTERNATIONAL", "INSTITUTE",
    "INSTITUTION", "PC", "PA", "LP", "CO", "DBA", "NA",
]

_SUFFIX_RE = re.compile(r"\b(" + "|".join(LEGAL_SUFFIXES) + r")\b\.?\s*$")
_PUNCT_RE = re.compile(r"[^A-Z0-9&\s]")
_WS_RE = re.compile(r"\s+")

REDACTED_PAYEES = {"N/A (PRIVACY/SECURITY)", "N/A", ""}

# Substrings marking a counterparty that is a public body or benefit trust
# rather than a procured vendor.
GOVERNMENTAL_PATTERNS = [
    "SCHOOL CONSTRUCTION AUTHORITY", "TRANSIT AUTHORITY", "RETIREE HEALTH",
    "WELFARE FUND", "CITY OF NEW YORK", "NYC DEPARTMENT", "DEPT OF",
    "DEPARTMENT OF", "OFFICE OF", "NEW YORK STATE", "NYS ", "MTA ",
    "HEALTH BENEFITS TRUST", "PENSION", "COMPTROLLER", "TREASURY",
    "BOARD OF EDUCATION", "CITY UNIVERSITY", "STATE UNIVERSITY",
]


def normalize_vendor(name: str | None) -> str:
    """Collapse spelling variants of one firm onto a single key.

    Uppercase, drop punctuation, normalize ``AND`` to ``&``, then strip trailing
    legal suffixes repeatedly so "ACME CO INC" and "ACME CORPORATION" meet.
    """
    if not name:
        return ""
    text = name.upper().strip()
    text = text.replace("&AMP;", "&").replace(" AND ", " & ")
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()

    previous = None
    while previous != text:
        previous = text
        text = _SUFFIX_RE.sub("", text).strip()

    # A leading "THE" carries no identity ("THE ACME CO" == "ACME CO").
    if text.startswith("THE "):
        text = text[4:]
    return _WS_RE.sub(" ", text).strip()


def is_redacted(name: str | None) -> bool:
    return (name or "").strip().upper() in {p.upper() for p in REDACTED_PAYEES}


def is_governmental(name: str | None) -> bool:
    text = (name or "").upper()
    return any(pattern in text for pattern in GOVERNMENTAL_PATTERNS)


def classify(name: str | None) -> str:
    """One of ``redacted``, ``governmental``, or ``vendor``."""
    if is_redacted(name):
        return "redacted"
    if is_governmental(name):
        return "governmental"
    return "vendor"


if __name__ == "__main__":
    samples = [
        "ACME CORP", "Acme Corp.", "ACME CORPORATION", "The Acme Company",
        "SCHOOL CONSTRUCTION AUTHORITY", "NYC SCHOOL CONSTRUCTION AUTHORITY",
        "N/A (PRIVACY/SECURITY)", "AMBOY BUS COMPANY", "AMBOY BUS CO., INC.",
        "TEMCO SERVICES INDUSTRIES INC", "Temco Services Industries",
    ]
    width = max(len(s) for s in samples)
    for sample in samples:
        print(f"{sample:<{width}}  ->  {normalize_vendor(sample):<32} [{classify(sample)}]")
