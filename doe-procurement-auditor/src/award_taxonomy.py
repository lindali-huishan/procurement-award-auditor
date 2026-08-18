"""The award-method taxonomy: five tiers of competitive intensity.

The original pipeline asked one binary question -- competitive or not -- and
could only answer it for 8,232 of 58,523 contracts. The other 86% were dropped
for having an award method that fit neither side, and **74% of all DOE
contracts sat in a single one of those dropped codes**:
``SMALL PURCHASE - WRITTEN``. A tool that says nothing about three quarters of
the contract file is not describing DOE procurement.

The fix is not to force those rows into the binary. It is to admit that
"competitively bid" is an *ordinal* property, and that some award methods are
not DOE's decision at all:

``formal``
    DOE ran a full competitive solicitation. Sealed bidding, RFP, PQVL-based
    RFP, multiple awards.

``informal``
    Competed, but under a lighter regime -- small purchases (written quotes),
    pre-qualified listings, demonstration authority. Discretion is real here
    but so is the rulebook, so these are neither "bid" nor "sole-sourced".

``discretionary``
    DOE chose the vendor without a competitive process it was free to run.
    Sole source, negotiated acquisition, emergency, assignment, council
    discretionary allocations. This is the tier oversight cares about.

``statutory``
    Non-competitive **by law or because another government competed it**:
    legal mandate, intergovernmental purchase, grants, leases. Flagging these
    says nothing about DOE's conduct, so they are scored out of the model --
    but reported, not silently dropped.

``renewal``
    Exercising an option on a contract that already exists. Not a new award
    decision, and cannot be "competitively bid" in the same sense.

The ``SMALL_PURCHASE_CAP`` matters structurally. ``SMALL PURCHASE - WRITTEN``
is hard-capped: all 43,418 of them have a maximum *and* median of exactly
$25,000, and 99.8% sit precisely at the cap. That makes the amount field an
almost perfect proxy for the award code, so any model given both amount and
these rows learns "is this $25,000" and reports a spectacular, meaningless AUC.
``src/award_model.py`` handles that by scoping above the cap. Note that only
``SMALL PURCHASE - WRITTEN`` is actually capped -- the other informal codes carry
no ceiling, and ``DEPT OF ED LISTING APPLICATION`` reaches $126.9M -- so the cap
is a scope boundary on amount, never a filter on award code.
"""

from __future__ import annotations

# NYC's small-purchase ceiling for written-quote procurement. Flat at $25,000
# across FY2010-FY2026 -- never indexed to inflation, so it has eroded in real
# terms while the order count roughly quintupled.
SMALL_PURCHASE_CAP = 25_000

FORMAL = (
    "COMPETITIVE SEALED BIDDING",
    "REQUEST FOR  PROPOSAL (RFP)",  # two spaces, as the feed spells it
    "RFP FROM A PQVL",
    "MULTIPLE AWARDS",
)

INFORMAL = (
    "SMALL PURCHASE - WRITTEN",
    "M/WBE SMALL PURCHASE",
    "SM PURCH GOODS SERVICES 100K",
    "SMALL PURCH - IT- 25 K TO 100K",
    "SMALL PURCHASE - INFO TECH",
    "DEPT OF ED LISTING APPLICATION",
    "INNOVATIVE PROCUREMENT",
)

DISCRETIONARY = (
    "SOLE SOURCE",
    "NEGOTIATED ACQUISITION AND DOE NEGOTIATED SERVICES",
    "NEG ACQUISITSION  EXTN AND DOE NEGOTIATED SERVICES",
    "NEG ACQUISITSION  EXTN AND DOE NEGOTIATED SERVICES EXTN",
    "EMERGENCY",
    "ASSIGNMENT",
    "BORO NEEDS/DISCRETIONARY FUND",
    "DEMONSTRATION PROJECT",
    "PROF. MEMBERSHIP NEGOTIATION",
    "CONTRACT CONVERSION",
)

STATUTORY = (
    "DETERMINED BY GOV'T MANDATE",
    "DETERMINED BY LEGAL MANDATE",
    "INTERGOVERNMENTAL PROCUREMENT",
    "INTERGOVERNMENTAL PROCUREMENT RENEWAL",
    "GOVERNMENT TO GOVERNMENT",
    "GRANTS",
    "GRANT RENEWAL",
    "LESSEE NEGOTIATION",
    "BUY AGAINST",
)

RENEWAL = ("RENEWAL OF CONTRACT",)

TIER_OF_METHOD = {
    **{m: "formal" for m in FORMAL},
    **{m: "informal" for m in INFORMAL},
    **{m: "discretionary" for m in DISCRETIONARY},
    **{m: "statutory" for m in STATUTORY},
    **{m: "renewal" for m in RENEWAL},
}

# Tiers where DOE chose how much competition to run. The model is scoped here;
# statutory and renewal are reported separately rather than scored.
DISCRETION_ELIGIBLE = ("formal", "informal", "discretionary")

TIER_LABELS = {
    "formal": "Formally competed",
    "informal": "Informally competed",
    "discretionary": "Discretionary award",
    "statutory": "Non-competitive by law",
    "renewal": "Renewal of existing contract",
    "unmapped": "Unmapped award code",
}


def tier_of(method: str | None) -> str:
    """Map a raw ``prime_contract_award_method`` to its tier.

    Unknown codes return ``"unmapped"`` rather than guessing. They are counted
    and surfaced so a new code appearing in the feed shows up as a gap instead
    of being quietly absorbed into a tier it may not belong to.
    """
    if method is None:
        return "unmapped"
    return TIER_OF_METHOD.get(str(method).strip(), "unmapped")
