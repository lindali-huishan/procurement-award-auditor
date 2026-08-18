"""Second-pass entity resolution over normalized vendor keys.

Exact normalization in ``normalize.py`` merges "ACME CORP" with "ACME
CORPORATION" but leaves three classes of the same firm split apart, all of which
appear in the real DOE data:

    truncation      CENTER FOR CIVIC EDUCATIO   ~ CENTER FOR CIVIC EDUCATION
    injected space  TRUSTEES OF COLUMBIA UNIVER SITY ~ TRUSTEES OF COLUMBIA UNIVERSITY
    plural drift    PARTY RENTAL                ~ PARTY RENTALS

The trap is that a fuzzy matcher aggressive enough to catch those also merges

    MANHATTAN CHARTER SCHOOL    vs  MANHATTAN CHARTER SCHOOL II
    HARLEM SUCCESS ACADEMY ...  vs  HARLEM SUCCESS ACADEMY ... 3

which are *separate legal entities* in a charter network. Merging them would
manufacture concentration that does not exist -- the exact error this analysis
is meant to measure. So an explicit ordinal guard runs before any similarity
test: if two names carry different trailing ordinals, they never merge.
"""

from __future__ import annotations

import re
from collections import defaultdict

ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6,
         "VII": 7, "VIII": 8, "IX": 9, "X": 10}

_ORDINAL_RE = re.compile(r"\b(\d{1,3}|" + "|".join(ROMAN) + r")$")

# Minimum core length before a prefix match is trusted as a truncation. Short
# names ("BAY", "BAYSIDE") are too easily distinct firms.
MIN_PREFIX_LEN = 14
MAX_TRUNCATION_GAP = 6


def ordinal_of(key: str) -> int | None:
    """Trailing entity number, e.g. ``... SCHOOL II`` -> 2. None if absent."""
    match = _ORDINAL_RE.search(key.strip())
    if not match:
        return None
    token = match.group(1)
    return ROMAN.get(token) if token in ROMAN else int(token)


def core(key: str) -> str:
    """Comparison form: ordinal removed, spaces dropped, trailing S dropped."""
    stripped = _ORDINAL_RE.sub("", key.strip()).strip()
    squashed = stripped.replace(" ", "")
    if len(squashed) > 4 and squashed.endswith("S"):
        squashed = squashed[:-1]
    return squashed


def may_merge(a: str, b: str) -> bool:
    """True if two keys are the same entity under the rules above."""
    if ordinal_of(a) != ordinal_of(b):
        return False  # numbered siblings are distinct entities

    core_a, core_b = core(a), core(b)
    if not core_a or not core_b:
        return False
    if core_a == core_b:
        return True

    # Truncation: one is a strict prefix of the other, long enough to be safe.
    shorter, longer = sorted((core_a, core_b), key=len)
    if (longer.startswith(shorter)
            and len(shorter) >= MIN_PREFIX_LEN
            and len(longer) - len(shorter) <= MAX_TRUNCATION_GAP):
        return True
    return False


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, a: str, b: str) -> None:
        root_a, root_b = self.find(a), self.find(b)
        if root_a != root_b:
            self.parent[root_b] = root_a


def build_canonical_map(weights: dict[str, float]) -> dict[str, str]:
    """Map each vendor key to a cluster representative.

    ``weights`` (key -> spend or row count) picks the representative: the
    highest-weight spelling wins, so the canonical name is the one the data
    actually uses most.

    Candidate pairs are blocked by the first four characters of the core, which
    keeps this near-linear instead of comparing all pairs.
    """
    keys = [k for k in weights if k]
    blocks: dict[str, list[str]] = defaultdict(list)
    for key in keys:
        prefix = core(key)[:4]
        if prefix:
            blocks[prefix].append(key)

    union_find = _UnionFind()
    for key in keys:
        union_find.find(key)

    for group in blocks.values():
        if len(group) < 2:
            continue
        group.sort()
        for i, a in enumerate(group):
            for b in group[i + 1:]:
                if may_merge(a, b):
                    union_find.union(a, b)

    clusters: dict[str, list[str]] = defaultdict(list)
    for key in keys:
        clusters[union_find.find(key)].append(key)

    canonical: dict[str, str] = {}
    for members in clusters.values():
        representative = max(members, key=lambda k: (weights.get(k, 0.0), len(k)))
        for member in members:
            canonical[member] = representative
    return canonical


if __name__ == "__main__":
    cases = [
        ("CENTER FOR CIVIC EDUCATIO", "CENTER FOR CIVIC EDUCATION", True),
        ("TRUSTEES OF COLUMBIA UNIVER SITY", "TRUSTEES OF COLUMBIA UNIVERSITY", True),
        ("PARTY RENTAL", "PARTY RENTALS", True),
        ("LABOR & INDUSTRY FOR EDUCA TION", "LABOR & INDUSTRY FOR EDUCATION", True),
        ("MANHATTAN CHARTER SCHOOL", "MANHATTAN CHARTER SCHOOL II", False),
        ("HARLEM SUCCESS ACADEMY CHARTER SCHOOL", "HARLEM SUCCESS ACADEMY CHARTER SCHOOL 3", False),
        ("CASTLE DAY CARE", "CASTLE DAY CARE II", False),
        ("BAYSIDE SIGN", "BAYSIDE SIGNS", True),
        ("ACME TRUCKING", "ACME CATERING", False),
    ]
    ok = True
    for a, b, expected in cases:
        got = may_merge(a, b)
        flag = "ok " if got == expected else "FAIL"
        ok &= got == expected
        print(f"  [{flag}] merge={got!s:<5} expect={expected!s:<5}  {a}  ~  {b}")
    print("\nall passed" if ok else "\nFAILURES present")
