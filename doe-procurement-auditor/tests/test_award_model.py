"""Tests for the award taxonomy, the model features, and the small-purchase count.

Three failure modes are worth guarding against here, and all three are ones the
previous version of this model actually had:

1. **A new award code silently absorbed into a tier.** The taxonomy returns
   ``"unmapped"`` rather than guessing, so a code appearing in a future feed
   pull surfaces as a gap instead of being quietly labelled competitive.
2. **The threshold trap.** ``SMALL PURCHASE - WRITTEN`` is hard-capped at
   $25,000 and 99.8% of those orders sit exactly at the cap. Let them into the
   model and ``log_amount`` alone reaches 0.982 AUC by recognising the ceiling.
   ``model_frame`` must keep them out.
3. **A flag threshold set so low it flags everything.** 92.6% of in-scope
   contracts are registered after their start date, so a "registered late" check
   at any-lateness returns 85% of the file and is useless for triage. The bar is
   guarded directly.

The model no longer builds any feature from the label, so the leakage tests that
used to live here are gone with the feature they guarded.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from award_taxonomy import (  # noqa: E402
    DISCRETION_ELIGIBLE, SMALL_PURCHASE_CAP, TIER_LABELS, tier_of,
)
from award_model import (  # noqa: E402
    CONTRACT_TYPE_LABELS, FEATURES, INDUSTRY_LABELS, STRUCTURAL,
    build_features, model_frame,
)
from review_queue import RETRO_FLAG_DAYS, build as rq_build  # noqa: E402


# --------------------------------------------------------------------------
# taxonomy
# --------------------------------------------------------------------------

@pytest.mark.parametrize("method,expected", [
    ("COMPETITIVE SEALED BIDDING", "formal"),
    ("REQUEST FOR  PROPOSAL (RFP)", "formal"),      # two spaces, as the feed has it
    ("RFP FROM A PQVL", "formal"),
    ("SMALL PURCHASE - WRITTEN", "informal"),
    ("DEPT OF ED LISTING APPLICATION", "informal"),
    ("SOLE SOURCE", "discretionary"),
    ("EMERGENCY", "discretionary"),
    ("ASSIGNMENT", "discretionary"),
    ("BORO NEEDS/DISCRETIONARY FUND", "discretionary"),
    ("DETERMINED BY GOV'T MANDATE", "statutory"),
    ("INTERGOVERNMENTAL PROCUREMENT", "statutory"),
    ("GRANTS", "statutory"),
    ("RENEWAL OF CONTRACT", "renewal"),
])
def test_tier_of_known_methods(method, expected):
    assert tier_of(method) == expected


def test_rfp_double_space_is_not_normalised_away():
    """The feed spells it with two spaces. Collapsing whitespace would silently
    drop 1,421 formally competed contracts out of the taxonomy."""
    assert tier_of("REQUEST FOR  PROPOSAL (RFP)") == "formal"
    assert tier_of("REQUEST FOR PROPOSAL (RFP)") == "unmapped"


def test_unknown_method_is_unmapped_not_guessed():
    assert tier_of("SOME NEW CODE THE FEED INVENTED") == "unmapped"
    assert tier_of(None) == "unmapped"
    assert tier_of("") == "unmapped"


def test_surrounding_whitespace_is_tolerated():
    assert tier_of("  SOLE SOURCE  ") == "discretionary"


def test_statutory_and_renewal_are_not_discretion_eligible():
    """DOE has no discretion over a legal mandate, and a renewal is not a new
    award decision. Scoring either would be a claim about conduct."""
    assert "statutory" not in DISCRETION_ELIGIBLE
    assert "renewal" not in DISCRETION_ELIGIBLE
    assert set(DISCRETION_ELIGIBLE) == {"formal", "informal", "discretionary"}


def test_every_tier_has_a_label():
    for tier in (*DISCRETION_ELIGIBLE, "statutory", "renewal", "unmapped"):
        assert TIER_LABELS[tier]


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

def _contract(vendor, fy, method, amount, start, end, registered,
              industry="Professional Services", ctype="PROGRAMS", mwbe="Non-M/WBE"):
    return {
        "prime_contract_id": f"{vendor}-{fy}-{amount}",
        "vendor_record_type": "Prime Vendor",
        "prime_vendor": vendor,
        "fiscal_year": fy,
        "prime_contract_award_method": method,
        "prime_contract_original_amount": float(amount),
        "prime_contract_start_date": start,
        "prime_contract_end_date": end,
        "prime_contract_registration_date": registered,
        "prime_contract_industry": industry,
        "prime_contract_type": ctype,
        "prime_vendor_mwbe_category": mwbe,
        "prime_contract_expense_category": "SUPPLIES",
    }


@pytest.fixture
def frame():
    rows = [
        # above-cap formal, five-year term, registered on time
        _contract("ALPHA", 2015, "COMPETITIVE SEALED BIDDING", 800_000,
                  "2014-07-01", "2019-06-30", "2014-07-01"),
        # above-cap discretionary, one-year term, registered 200 days late
        _contract("BETA", 2016, "SOLE SOURCE", 400_000,
                  "2015-07-01", "2016-06-30", "2016-01-17"),
        # at-cap small purchases -- must be excluded from the model
        _contract("GAMMA", 2017, "SMALL PURCHASE - WRITTEN", SMALL_PURCHASE_CAP,
                  "2016-09-01", "2017-02-01", "2016-09-01"),
        _contract("GAMMA", 2017, "SMALL PURCHASE - WRITTEN", SMALL_PURCHASE_CAP,
                  "2016-09-01", "2017-02-01", "2016-09-01"),
        # statutory -- excluded regardless of amount
        _contract("DELTA", 2018, "GRANTS", 5_000_000,
                  "2017-07-01", "2020-06-30", "2017-07-01"),
        # renewal -- excluded
        _contract("ALPHA", 2019, "RENEWAL OF CONTRACT", 900_000,
                  "2018-07-01", "2021-06-30", "2018-07-01"),
        # above-cap informal -- the population the old scope threw away
        _contract("EPSILON", 2019, "DEPT OF ED LISTING APPLICATION", 630_000,
                  "2018-07-01", "2019-06-30", "2018-07-01"),
    ]
    return pd.DataFrame(rows).assign(
        method=lambda d: d.prime_contract_award_method.str.strip(),
        tier=lambda d: d.prime_contract_award_method.map(tier_of),
        fy=lambda d: d.fiscal_year.astype(int),
        amount=lambda d: d.prime_contract_original_amount.astype(float),
    )


# --------------------------------------------------------------------------
# features
# --------------------------------------------------------------------------

def test_duration_bands_partition_the_term_axis(frame):
    """The bands must be mutually exclusive. Overlapping dummies would
    double-count the term and make the published coefficients unreadable."""
    built = build_features(frame)
    both = built.dur_under_1y + built.dur_1_to_2y + built.dur_over_4y
    assert (both <= 1).all()


def test_duration_band_boundaries():
    rows = [_contract("V", 2015, "SOLE SOURCE", 100_000, "2014-01-01", end, "2014-01-01")
            for end in ("2014-05-01", "2015-01-05", "2015-06-01", "2016-01-01", "2018-06-01")]
    built = build_features(pd.DataFrame(rows).assign(
        amount=lambda d: d.prime_contract_original_amount))
    days = built.duration_days.tolist()
    # 120d, 369d -> under 1y; 516d, 730d -> 1-2y; 1612d -> over 4y
    assert built.dur_under_1y.tolist() == [1, 1, 0, 0, 0], days
    assert built.dur_1_to_2y.tolist() == [0, 0, 1, 1, 0], days
    assert built.dur_over_4y.tolist() == [0, 0, 0, 0, 1], days


def test_retro_days_is_zero_when_registered_before_start(frame):
    built = build_features(frame)
    alpha = built[built.prime_vendor == "ALPHA"].iloc[0]
    beta = built[built.prime_vendor == "BETA"].iloc[0]
    assert alpha.retro_days == 0
    assert beta.retro_days == 200


def test_retro_days_is_capped(frame):
    """Capped at 730 so a single contract registered a decade late cannot
    dominate a standardised coefficient."""
    row = _contract("LATE", 2015, "SOLE SOURCE", 100_000,
                    "2010-01-01", "2011-01-01", "2020-01-01")
    built = build_features(pd.DataFrame([row]).assign(
        amount=lambda d: d.prime_contract_original_amount))
    assert built.retro_days.iloc[0] == 730
    assert built.registered_after_end.iloc[0] == 1


def test_registration_lag_is_computed_but_not_a_feature(frame):
    """The lag drives two review-queue flags and must keep being computed. It
    must NOT be in the fit: it happens after the award, so training on it teaches
    the model that "no-bid and registered very late" is normal and hides exactly
    the contracts the queue exists to surface."""
    built = build_features(frame)
    assert "retro_days" in built.columns
    assert "registered_after_end" in built.columns
    assert "retro_days" not in FEATURES
    assert "registered_after_end" not in FEATURES


def test_no_feature_is_knowable_only_after_the_award():
    """Guards the boundary generally, not just for the two known cases."""
    banned = ("retro", "registered_after", "spent_to_date", "current_amount")
    assert not [f for f in FEATURES if any(b in f for b in banned)]


def test_mwbe_is_not_a_feature():
    """Dropped for carrying no signal -- removing it improved rolling AUC."""
    assert not [f for f in FEATURES if "mwbe" in f]


def test_all_declared_features_are_built(frame):
    built = build_features(frame)
    missing = [f for f in FEATURES if f not in built.columns]
    assert not missing, missing


def test_structural_features_are_a_subset_of_features():
    assert set(STRUCTURAL) <= set(FEATURES)


# --------------------------------------------------------------------------
# leakage
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# scope -- the threshold trap
# --------------------------------------------------------------------------

def test_model_frame_excludes_at_cap_small_purchases(frame):
    scoped = model_frame(frame)
    assert "GAMMA" not in set(scoped.prime_vendor)
    assert (scoped.amount > SMALL_PURCHASE_CAP).all()


def test_model_frame_excludes_statutory_and_renewal(frame):
    scoped = model_frame(frame)
    assert "DELTA" not in set(scoped.prime_vendor)          # GRANTS
    assert "RENEWAL OF CONTRACT" not in set(scoped.method)


def test_model_frame_includes_above_cap_informal(frame):
    """The 1,193 above-cap informal contracts the old binary scope discarded."""
    scoped = model_frame(frame)
    assert "EPSILON" in set(scoped.prime_vendor)
    assert scoped.loc[scoped.prime_vendor == "EPSILON", "y"].iloc[0] == 1


def test_label_is_not_formally_competed(frame):
    scoped = model_frame(frame)
    by_vendor = scoped.set_index("prime_vendor").y
    assert by_vendor["ALPHA"] == 0     # competitive sealed bidding
    assert by_vendor["BETA"] == 1      # sole source
    assert by_vendor["EPSILON"] == 1   # informal


# --------------------------------------------------------------------------
# small-purchase aggregation
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# taxonomy -> feature mapping
# --------------------------------------------------------------------------

def test_real_industries_map_to_their_own_dummy(frame):
    built = build_features(frame)
    assert built.loc[built.prime_vendor == "ALPHA", "ind_professional_services"].iloc[0] == 1
    assert built.loc[built.prime_vendor == "ALPHA", "ind_goods"].iloc[0] == 0


def test_no_construction_feature_survives():
    """DOE has no construction industry at all -- the old is_construction flag
    fitted a zero coefficient. If it ever reappears, something regressed."""
    assert not any("construction" in f for f in FEATURES)


def test_industry_dummies_are_mutually_exclusive():
    rows = [_contract("V", 2015, "SOLE SOURCE", 100_000, "2014-01-01", "2015-01-01",
                      "2014-01-01", industry=ind)
            for ind in ("Goods", "Human Services", "Professional Services",
                        "Standardized Services", "Not Classified")]
    built = build_features(pd.DataFrame(rows).assign(
        amount=lambda d: d.prime_contract_original_amount))
    dummies = [f for f in FEATURES if f.startswith("ind_")]
    assert (built[dummies].sum(axis=1) <= 1).all()
    # "Not Classified" is the reference level: every dummy zero.
    assert built[dummies].iloc[-1].sum() == 0


def test_contract_type_dummies_are_mutually_exclusive_and_grouped():
    """Near-duplicate feed codes collapse onto one dummy rather than each
    getting their own sparse column."""
    rows = [_contract("V", 2015, "SOLE SOURCE", 100_000, "2014-01-01", "2015-01-01",
                      "2014-01-01", ctype=t)
            for t in ("PROGRAMS", "PROGRAMS (NOT TAX LEVY FUNDED)",
                      "DEPT OF ED-REQUIREMENT CONTRACT", "REQUIREMENTS-GOODS",
                      "CONSULTANT", "SOMETHING UNKNOWN")]
    built = build_features(pd.DataFrame(rows).assign(
        amount=lambda d: d.prime_contract_original_amount))
    dummies = [f for f in FEATURES if f.startswith("type_")]
    assert (built[dummies].sum(axis=1) <= 1).all()
    assert built.type_programs.tolist()[:2] == [1, 1]          # both PROGRAMS codes
    assert built.type_doe_requirements.tolist()[2:4] == [1, 1]  # both requirement codes
    assert built[dummies].iloc[-1].sum() == 0                   # unknown -> reference


def test_every_label_covers_every_shipped_dummy():
    for feature in FEATURES:
        if feature.startswith("ind_"):
            assert feature in INDUSTRY_LABELS
        if feature.startswith("type_"):
            assert feature in CONTRACT_TYPE_LABELS


# --------------------------------------------------------------------------
# review queue thresholds
# --------------------------------------------------------------------------

def test_retro_flag_bar_is_high_enough_to_be_informative():
    """A bar of "any lateness" fires on 92.6% of in-scope contracts and makes the
    queue useless for triage. This guards the decision, not the arithmetic."""
    assert RETRO_FLAG_DAYS >= 180


def test_retroactive_flag_needs_a_year(frame):
    scored = model_frame(frame)
    scored["p"] = 0.5
    queue = rq_build(scored, [], Path("/nonexistent"))
    beta = [c for c in queue["cases"] if c["vendor"] == "BETA"]
    # BETA is 200 days late -- late, but under the flag bar, and it has no other
    # trigger, so it should not appear as a case at all.
    assert not beta


def test_registered_after_expiry_always_flags():
    row = _contract("EXPIRED", 2016, "SOLE SOURCE", 500_000,
                    "2015-07-01", "2016-06-30", "2016-09-01")
    frame = pd.DataFrame([row]).assign(
        method=lambda d: d.prime_contract_award_method.str.strip(),
        tier=lambda d: d.prime_contract_award_method.map(tier_of),
        fy=lambda d: d.fiscal_year.astype(int),
        amount=lambda d: d.prime_contract_original_amount.astype(float))
    scored = model_frame(frame)
    scored["p"] = 0.9
    queue = rq_build(scored, [], Path("/nonexistent"))
    assert queue["cases"][0]["flags"] == ["registered_after_expiry"]


def test_queue_omits_contracts_with_no_flags():
    """The queue is a worklist, not a table dump."""
    row = _contract("CLEAN", 2016, "COMPETITIVE SEALED BIDDING", 800_000,
                    "2015-07-01", "2020-06-30", "2015-07-01")
    frame = pd.DataFrame([row]).assign(
        method=lambda d: d.prime_contract_award_method.str.strip(),
        tier=lambda d: d.prime_contract_award_method.map(tier_of),
        fy=lambda d: d.fiscal_year.astype(int),
        amount=lambda d: d.prime_contract_original_amount.astype(float))
    scored = model_frame(frame)
    scored["p"] = 0.1
    queue = rq_build(scored, [], Path("/nonexistent"))
    assert queue["cases"] == []
    assert queue["total"] == 0


def test_queue_sorts_multi_flag_cases_first():
    rows = [
        # one flag, huge
        _contract("BIG", 2016, "SOLE SOURCE", 90_000_000,
                  "2015-07-01", "2016-06-30", "2017-06-01"),
        # two flags, small: expired registration + long no-bid term
        _contract("MANY", 2016, "SOLE SOURCE", 200_000,
                  "2015-07-01", "2020-06-30", "2020-09-01"),
    ]
    frame = pd.DataFrame(rows).assign(
        method=lambda d: d.prime_contract_award_method.str.strip(),
        tier=lambda d: d.prime_contract_award_method.map(tier_of),
        fy=lambda d: d.fiscal_year.astype(int),
        amount=lambda d: d.prime_contract_original_amount.astype(float))
    scored = model_frame(frame)
    scored["p"] = 0.9
    queue = rq_build(scored, [], Path("/nonexistent"))
    assert queue["cases"][0]["vendor"] == "MANY", [c["vendor"] for c in queue["cases"]]
    assert len(queue["cases"][0]["flags"]) >= 2


def test_lobbying_absent_degrades_without_error():
    """The tool must build with no lobbying feed pulled -- the queue just has
    one fewer check."""
    row = _contract("V", 2016, "SOLE SOURCE", 500_000,
                    "2015-07-01", "2016-06-30", "2016-09-01")
    frame = pd.DataFrame([row]).assign(
        method=lambda d: d.prime_contract_award_method.str.strip(),
        tier=lambda d: d.prime_contract_award_method.map(tier_of),
        fy=lambda d: d.fiscal_year.astype(int),
        amount=lambda d: d.prime_contract_original_amount.astype(float))
    scored = model_frame(frame)
    scored["p"] = 0.5
    queue = rq_build(scored, [], Path("/nonexistent"))
    assert queue["lobbying_available"] is False
    assert all("lobbied" not in c["flags"] for c in queue["cases"])
