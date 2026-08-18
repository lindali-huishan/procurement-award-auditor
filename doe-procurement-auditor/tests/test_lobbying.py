"""Tests for the lobbying feed parser and the cross-system name matcher.

The matcher is the highest-risk code added by the lobbying join. Inside
Checkbook, a bad merge miscounts a statistic. Here it asserts that a *named*
firm lobbied the agency that then handed it a no-bid contract, so the tests
below weight false positives far more heavily than false negatives.

The containment cases are real pairs the first version of the matcher accepted
and got wrong. They are kept as regression tests because the failure was not
obvious from reading the rule -- it only showed up against the live feed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lobbying import (  # noqa: E402
    ACCEPT_METHODS, accepts, canonical_tokens, has_procurement, parse_activities,
    parse_targets, score_pair, split_beneficiary, targets_doe,
)


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------

def test_parse_targets_extracts_agency_code_and_official():
    parsed = parse_targets("Education, Department of (DOE)  David Banks")
    assert parsed == [{"agency": "Education, Department of",
                       "code": "DOE", "official": "David Banks"}]


def test_parse_targets_deduplicates_repeats():
    """One cell in the live feed repeated the same target fourteen times."""
    blob = "; ".join(["Education, Department of (DOE)  David Banks"] * 14)
    assert len(parse_targets(blob)) == 1


def test_parse_targets_handles_uncoded_entries():
    parsed = parse_targets("NYC Council Members  Ben Kallos - District No. 5")
    assert parsed[0]["agency"] == "NYC Council Members"
    assert parsed[0]["code"] == ""
    assert "Ben Kallos" in parsed[0]["official"]


def test_parse_targets_empty():
    assert parse_targets(None) == []
    assert parse_targets("") == []
    assert parse_targets("  ;  ; ") == []


def test_parse_activities_splits_category_from_detail():
    parsed = parse_activities("Procurement - Contract for tutoring services")
    assert parsed == [{"category": "Procurement",
                       "detail": "Contract for tutoring services"}]


def test_parse_activities_recovers_category_after_leaked_fragment():
    """Some cells leak the tail of the previous item before the category."""
    blob = "infrastructure, indirect and OTPS cost for the sector  , Procurement - X"
    assert parse_activities(blob)[0]["category"] == "Procurement"


def test_parse_activities_deduplicates():
    blob = "Budget - FY24 ask; Budget - FY24 ask; Procurement - RFP"
    assert len(parse_activities(blob)) == 2


@pytest.mark.parametrize("blob,expected", [
    ("Education, Department of (DOE)  David Banks", True),
    ("EDUCATION, DEPARTMENT OF (DOE)", True),
    ("City Planning, Department of (DCP)  Someone", False),
    ("", False),
    (None, False),
])
def test_targets_doe(blob, expected):
    assert targets_doe(blob) is expected


@pytest.mark.parametrize("blob,expected", [
    ("Procurement - contract award", True),
    ("PROCUREMENT - contract award", True),
    ("Budget - funding ask", False),
    (None, False),
])
def test_has_procurement(blob, expected):
    assert has_procurement(blob) is expected


@pytest.mark.parametrize("raw,client,beneficiary", [
    ("Carahsoft Technology Corp. for the benefit of Elastic",
     "Carahsoft Technology Corp.", "Elastic"),
    ("First Student, Inc. on behalf of Bella Bus Corp.",
     "First Student, Inc.", "Bella Bus Corp."),
    ("Pearson Education, Inc.", "Pearson Education, Inc.", None),
])
def test_split_beneficiary(raw, client, beneficiary):
    assert split_beneficiary(raw) == (client, beneficiary)


# --------------------------------------------------------------------------
# canonicalisation
# --------------------------------------------------------------------------

def test_possessive_remnant_is_dropped():
    """Stripping punctuation turns CHILDREN'S into CHILDREN S."""
    assert canonical_tokens("Children's Aid Society") == \
           canonical_tokens("CHILDRENS AID SOCIETY")


def test_abbreviations_expand():
    assert canonical_tokens("Intl Business Machines") == \
           canonical_tokens("International Business Machines")


def test_spelling_variants_fold():
    assert canonical_tokens("Centre for Civic Education") == \
           canonical_tokens("Center for Civic Education")


def test_plural_drift_folds():
    assert canonical_tokens("Educational Networks") == \
           canonical_tokens("Educational Network")


# --------------------------------------------------------------------------
# matching -- must accept
# --------------------------------------------------------------------------

@pytest.mark.parametrize("a,b", [
    ("Pearson Education, Inc.", "PEARSON EDUCATION"),
    ("The Gordian Group, Inc.", "GORDIAN GROUP"),
    ("Logan Bus Co., Inc.", "LOGAN BUS"),
    ("N2Y, LLC", "N2Y"),
    ("IBM Corporation", "INTERNATIONAL BUSINESS MACHINES"),
    ("Amalgamated Transit Union", "ATU"),
    ("Brooklyn Children's Museum", "BROOKLYN CHILDRENS MUSEUM"),
    ("Heart Share Human Services", "HEARTSHARE HUMAN SERVICES"),
    ("Nontraditional Employment for Women", "NON TRADITIONAL EMPLOYMENT FOR WOMEN"),
    ("CENTER FOR CIVIC EDUCATIO", "CENTER FOR CIVIC EDUCATION"),
])
def test_equivalent_names_match(a, b):
    assert accepts(a, b), f"expected a match: {a} ~ {b}"


def test_accepted_methods_are_equivalence_only():
    """Containment and partial overlap must never be auto-accept methods."""
    assert ACCEPT_METHODS == {"exact", "canonical", "reordered", "truncation"}


def test_match_is_symmetric():
    assert score_pair("IBM", "INTERNATIONAL BUSINESS MACHINES") == \
           score_pair("INTERNATIONAL BUSINESS MACHINES", "IBM")


# --------------------------------------------------------------------------
# matching -- must refuse
# --------------------------------------------------------------------------

@pytest.mark.parametrize("a,b", [
    # ordinal guard: numbered siblings in a charter network
    ("Manhattan Charter School", "MANHATTAN CHARTER SCHOOL II"),
    ("Harlem Success Academy 3", "HARLEM SUCCESS ACADEMY"),
    # embedded-number guard
    ("P.S. 123 Partners", "PS 456 PARTNERS"),
    # generic words carry no identity
    ("Acme Educational Services", "BETA EDUCATIONAL SERVICES"),
    ("New York Educational Group", "NEW YORK EDUCATIONAL SOLUTIONS"),
])
def test_distinct_entities_refused(a, b):
    assert not accepts(a, b), f"must not match: {a} ~ {b}"


@pytest.mark.parametrize("a,b", [
    ("Red Hat, Inc.", "RED HAT DAY CARE CENTER"),
    ("Verizon Corporate Resources Group LLC", "CORPORATE RESOURCE DEVELOPMENT"),
    ("Hamaspik of Kings County, Inc.", "KINGS COUNTY"),
    ("Motion Picture Association, Inc.", "SWANK MOTION PICTURES"),
    ("First Student, Inc.", "STUDENTS FIRST SERVICES"),
    ("New York Shakespeare Festival", "HUDSON VALLEY SHAKESPEARE FESTIVAL"),
    ("Museum of the City of New York, Inc.", "MUSEUM HACK"),
    ("Acacia Network, Inc.", "EDUCATION NETWORKS OF AMERICA"),
])
def test_containment_false_positives_go_to_review(a, b):
    """Real pairs the first matcher accepted and got wrong."""
    assert not accepts(a, b), f"must not auto-accept: {a} ~ {b}"


def test_empty_names_never_match():
    assert not accepts("", "ACME")
    assert not accepts("ACME", "")


def test_blocked_pairs_score_zero():
    confidence, method = score_pair("Manhattan Charter School",
                                    "MANHATTAN CHARTER SCHOOL II")
    assert confidence == 0.0
    assert method == "blocked"
