"""Tests for the two steps that silently corrupt results when wrong.

Vendor resolution is the highest-risk code in the project: too timid and
concentration is understated, too aggressive and it is manufactured. The
numbered-entity cases below are the ones that matter most -- charter schools in
a network share a name but are separate legal entities.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from normalize import classify, normalize_vendor  # noqa: E402
from resolve import build_canonical_map, core, may_merge, ordinal_of  # noqa: E402


@pytest.mark.parametrize("raw,expected", [
    ("ACME CORP", "ACME"),
    ("Acme Corp.", "ACME"),
    ("ACME CORPORATION", "ACME"),
    ("The Acme Company", "ACME"),
    ("ACME CO., INC.", "ACME"),
    ("Smith & Sons LLC", "SMITH & SONS"),
    ("Smith and Sons", "SMITH & SONS"),
])
def test_normalize_collapses_variants(raw, expected):
    assert normalize_vendor(raw) == expected


def test_normalize_handles_empty():
    assert normalize_vendor(None) == ""
    assert normalize_vendor("") == ""


@pytest.mark.parametrize("name,expected", [
    ("N/A (PRIVACY/SECURITY)", "redacted"),
    ("SCHOOL CONSTRUCTION AUTHORITY", "governmental"),
    ("NEW YORK CITY TRANSIT AUTHORITY", "governmental"),
    ("U F T WELFARE FUND", "governmental"),
    ("AMBOY BUS COMPANY", "vendor"),
    ("TEMCO SERVICES INDUSTRIES INC", "vendor"),
])
def test_classify(name, expected):
    assert classify(name) == expected


@pytest.mark.parametrize("key,expected", [
    ("MANHATTAN CHARTER SCHOOL II", 2),
    ("HARLEM SUCCESS ACADEMY 3", 3),
    ("ACME TRUCKING", None),
])
def test_ordinal_extraction(key, expected):
    assert ordinal_of(key) == expected


@pytest.mark.parametrize("a,b", [
    ("CENTER FOR CIVIC EDUCATIO", "CENTER FOR CIVIC EDUCATION"),
    ("TRUSTEES OF COLUMBIA UNIVER SITY", "TRUSTEES OF COLUMBIA UNIVERSITY"),
    ("LABOR & INDUSTRY FOR EDUCA TION", "LABOR & INDUSTRY FOR EDUCATION"),
    ("PARTY RENTAL", "PARTY RENTALS"),
])
def test_merges_real_variants(a, b):
    assert may_merge(a, b)


@pytest.mark.parametrize("a,b", [
    # Numbered siblings are distinct legal entities -- merging them would
    # invent concentration that does not exist.
    ("MANHATTAN CHARTER SCHOOL", "MANHATTAN CHARTER SCHOOL II"),
    ("CASTLE DAY CARE", "CASTLE DAY CARE II"),
    ("HARLEM SUCCESS ACADEMY CHARTER SCHOOL", "HARLEM SUCCESS ACADEMY CHARTER SCHOOL 3"),
    ("EXPLORE CHARTER SCHOOL II", "EXPLORE CHARTER SCHOOL III"),
    # Unrelated firms sharing a first token.
    ("ACME TRUCKING", "ACME CATERING"),
    # Short names must not merge on a prefix.
    ("BAY", "BAYSIDE"),
])
def test_refuses_unsafe_merges(a, b):
    assert not may_merge(a, b)


def test_core_strips_ordinal_spaces_and_plural():
    assert core("PARTY RENTALS") == core("PARTY RENTAL")
    assert core("MANHATTAN CHARTER SCHOOL II") == core("MANHATTAN CHARTER SCHOOL")


def test_canonical_map_picks_highest_weight_spelling():
    weights = {
        "CENTER FOR CIVIC EDUCATION": 900_000.0,
        "CENTER FOR CIVIC EDUCATIO": 1_000.0,
    }
    canonical = build_canonical_map(weights)
    assert len(set(canonical.values())) == 1
    assert set(canonical.values()) == {"CENTER FOR CIVIC EDUCATION"}


def test_canonical_map_keeps_numbered_entities_apart():
    weights = {
        "MANHATTAN CHARTER SCHOOL": 500.0,
        "MANHATTAN CHARTER SCHOOL II": 400.0,
        "MANHATTAN CHARTER SCHOOL III": 300.0,
    }
    canonical = build_canonical_map(weights)
    assert len(set(canonical.values())) == 3


def test_canonical_map_is_idempotent():
    weights = {"ACME": 10.0, "ACME SERVICE": 5.0, "ACME SERVICES": 4.0}
    once = build_canonical_map(weights)
    twice = build_canonical_map({once[k]: v for k, v in weights.items()})
    assert {twice.get(v, v) for v in once.values()} == set(once.values())
