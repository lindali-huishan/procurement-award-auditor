"""Join lobbying filings to DOE contracts and flag industry-aware convergence.

The question this answers is narrow on purpose: *did a vendor that received a
non-competitive award also pay to lobby DOE about procurement, in or just
before the year of that award?* Both halves are public record and neither is
evidence of anything by itself. Together they identify a contract file worth
reading.

Three design decisions carry the result.

**Industry-aware, not absolute.** Non-competitive award rates differ enormously
by line of work -- Human Services runs on negotiated acquisitions in a way that
Goods does not. Flagging raw "was sole-sourced" would return the entire Human
Services book and nothing useful. Every flag here is graded against the
non-competitive base rate of that contract's own industry, so the question is
always "unusual *for this kind of work*".

**Timing has to point the right way.** A filing is only counted when it lands
in the award fiscal year or the one before it. Lobbying that happens after an
award cannot have influenced it, and counting it would manufacture convergence
out of ordinary vendor-relations work.

**Coverage is stated, not assumed.** The eLobbyist feed begins at report year
2021. Spending runs from FY2010. Convergence is therefore computable only for
roughly the last six years, and every count this writes is scoped to that
window rather than being silently compared against a seventeen-year denominator.

    python3 src/build_lobbying.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from award_taxonomy import tier_of  # noqa: E402
from lobbying import (  # noqa: E402
    ACCEPT_FLOOR, has_procurement, match_clients, parse_activities,
    parse_targets, split_beneficiary, targets_doe,
)
from normalize import normalize_vendor  # noqa: E402

PANEL_DIR = ROOT / "data" / "panel"
LOBBYING = ROOT / "data" / "raw_lobbying" / "lobbying.parquet"
CONTRACTS = ROOT / "data" / "raw_contracts" / "contracts_registered.parquet"

# A filing counts toward an award only if it lands in the award year or the one
# before. Anything later cannot have preceded the decision.
LOOKBACK_YEARS = 1


def load_filings() -> pd.DataFrame:
    """Filings that target DOE, with targets and activities parsed out."""
    frame = pd.read_parquet(LOBBYING)

    targets = frame["lobbyist_targets"].fillna("") + " ; " + frame.get(
        "periodic_targets", pd.Series("", index=frame.index)).fillna("")
    activities = frame["lobbyist_activities"].fillna("") + " ; " + frame.get(
        "periodic_activities", pd.Series("", index=frame.index)).fillna("")

    frame = frame.assign(all_targets=targets, all_activities=activities)
    doe = frame[frame["all_targets"].map(targets_doe)].copy()

    doe["is_procurement"] = doe["all_activities"].map(has_procurement)
    doe["report_year"] = pd.to_numeric(doe["report_year"], errors="coerce")
    doe["compensation_total"] = pd.to_numeric(
        doe["compensation_total"], errors="coerce").fillna(0.0)

    # Officials named at DOE specifically -- the FOIL-able detail.
    def doe_officials(blob: str) -> str:
        names = {t["official"] for t in parse_targets(blob)
                 if "EDUCATION, DEPARTMENT OF" in t["agency"].upper()
                 and t["official"] and "UNKNOWN" not in t["official"].upper()}
        return "; ".join(sorted(names)[:6])

    def categories(blob: str) -> str:
        cats = {a["category"] for a in parse_activities(blob) if a["category"]}
        return "; ".join(sorted(cats)[:8])

    doe["doe_officials"] = doe["all_targets"].map(doe_officials)
    doe["activity_categories"] = doe["all_activities"].map(categories)
    doe[["client_base", "client_beneficiary"]] = doe["client_name"].apply(
        lambda n: pd.Series(split_beneficiary(n)))
    return doe


def load_contracts() -> pd.DataFrame:
    """Discretionary contracts with the same labelling the award model uses."""
    frame = pd.read_parquet(CONTRACTS)
    frame = frame[(frame["vendor_record_type"] == "Prime Vendor")
                  & (frame["prime_contract_original_amount"] > 0)].copy()

    # Labelled off the shared tier taxonomy, not a local regex, so this file and
    # the award model can never drift apart on what "non-competitive" means.
    # ``discretionary`` is DOE choosing not to compete; ``formal`` is a full
    # solicitation. Everything else -- informal, statutory, renewal -- is dropped
    # here, because a convergence flag is a claim about DOE's own discretion.
    frame["tier"] = frame["prime_contract_award_method"].map(tier_of)
    frame = frame[frame["tier"].isin(("formal", "discretionary"))].copy()
    frame["noncompetitive"] = (frame["tier"] == "discretionary").astype(int)

    frame["fiscal_year"] = pd.to_numeric(frame["fiscal_year"], errors="coerce")
    frame["industry"] = frame["prime_contract_industry"].fillna("Not Classified")
    frame["vendor_key"] = frame["prime_vendor"].map(normalize_vendor)
    return frame


def industry_base_rates(contracts: pd.DataFrame) -> pd.DataFrame:
    """Non-competitive rate per industry -- the yardstick every flag is graded on."""
    stats = contracts.groupby("industry").agg(
        n_contracts=("noncompetitive", "size"),
        n_noncompetitive=("noncompetitive", "sum"),
        value=("prime_contract_original_amount", "sum"),
    ).reset_index()
    stats["noncomp_rate"] = stats["n_noncompetitive"] / stats["n_contracts"]
    return stats.sort_values("noncomp_rate")


def main() -> int:
    if not LOBBYING.exists():
        print("no lobbying data -- run src/pull_lobbying.py first")
        return 1

    filings = load_filings()
    procurement = filings[filings["is_procurement"]]
    print(f"DOE-targeting filings: {len(filings):,}  "
          f"({len(procurement):,} carry a Procurement activity)")
    print(f"  report years {int(filings.report_year.min())}-"
          f"{int(filings.report_year.max())}, "
          f"{filings.client_name.nunique():,} distinct clients")

    # ---- match clients to resolved vendors -------------------------------
    panel = pd.read_parquet(PANEL_DIR / "vendor_year.parquet")
    vendor_keys = sorted(panel["vendor_key"].dropna().unique())
    clients = sorted(filings["client_name"].dropna().unique())
    print(f"\nmatching {len(clients):,} lobbying clients against "
          f"{len(vendor_keys):,} resolved vendors...")

    matches, review = match_clients(clients, vendor_keys)
    n_matched = len(matches) if not matches.empty else 0
    n_review = len(review) if not review.empty else 0
    print(f"  accepted {n_matched:,} at confidence >= {ACCEPT_FLOOR}")
    print(f"  queued   {n_review:,} for human review "
          f"(ambiguous or below the floor)")
    if not matches.empty:
        print("\n  by method:")
        print(matches["method"].value_counts().to_string().replace("\n", "\n    "))

    # ---- vendor-level lobbying profile, procurement only -----------------
    proc = procurement.merge(matches, on="client_name", how="inner") \
        if not matches.empty else pd.DataFrame()
    if proc.empty:
        print("\nno matched procurement lobbying -- nothing to join")
        return 1

    vendor_lobby = proc.groupby("vendor_key").agg(
        lob_filings=("client_name", "size"),
        lob_clients=("client_name", lambda s: "; ".join(sorted(set(s))[:3])),
        lob_first_year=("report_year", "min"),
        lob_last_year=("report_year", "max"),
        lob_years=("report_year", "nunique"),
        lob_compensation=("compensation_total", "sum"),
        lob_lobbyists=("lobbyist_name", "nunique"),
        lob_confidence=("confidence", "max"),
        lob_method=("method", "first"),
    ).reset_index()

    officials = (proc[proc["doe_officials"] != ""]
                 .groupby("vendor_key")["doe_officials"]
                 .apply(lambda s: "; ".join(sorted({n for row in s for n in row.split("; ")})[:5]))
                 .rename("lob_officials").reset_index())
    vendor_lobby = vendor_lobby.merge(officials, on="vendor_key", how="left")
    vendor_lobby["lob_officials"] = vendor_lobby["lob_officials"].fillna("")
    print(f"\nvendors with matched DOE procurement lobbying: {len(vendor_lobby):,}")

    # ---- client leaderboard, matched and unmatched -----------------------
    # Ranked by reported compensation. Unmatched clients stay in: they lobbied
    # DOE on procurement whether or not their name could be tied to a payee,
    # and dropping them would understate the field and hide the matcher's own
    # coverage gap.
    leaderboard = procurement.groupby("client_name").agg(
        filings=("client_name", "size"),
        compensation=("compensation_total", "sum"),
        first_year=("report_year", "min"),
        last_year=("report_year", "max"),
        years=("report_year", "nunique"),
        lobbyists=("lobbyist_name", "nunique"),
        industry=("client_industry", "first"),
    ).reset_index()

    officials_by_client = (procurement[procurement["doe_officials"] != ""]
                           .groupby("client_name")["doe_officials"]
                           .apply(lambda s: "; ".join(
                               sorted({n for row in s for n in row.split("; ")})[:5]))
                           .rename("officials").reset_index())
    leaderboard = leaderboard.merge(officials_by_client, on="client_name", how="left")

    if not matches.empty:
        leaderboard = leaderboard.merge(
            matches[["client_name", "vendor_key", "confidence", "method"]],
            on="client_name", how="left")
    else:
        leaderboard[["vendor_key", "confidence", "method"]] = None

    spend = panel.groupby("vendor_key").agg(
        doe_spend=("spend", "sum"),
        doe_years=("fiscal_year", "nunique"),
    ).reset_index()
    leaderboard = leaderboard.merge(spend, on="vendor_key", how="left")

    # Non-competitive award history for matched vendors, so the leaderboard can
    # show who is both lobbying and receiving discretionary awards.
    contracts_all = load_contracts()
    vendor_awards = contracts_all.groupby("vendor_key").agg(
        n_awards=("noncompetitive", "size"),
        n_noncomp=("noncompetitive", "sum"),
        award_value=("prime_contract_original_amount", "sum"),
    ).reset_index()
    leaderboard = leaderboard.merge(vendor_awards, on="vendor_key", how="left")
    leaderboard["noncomp_share"] = (
        leaderboard["n_noncomp"] / leaderboard["n_awards"].replace(0, np.nan))
    leaderboard = leaderboard.sort_values("compensation", ascending=False)

    matched_n = int(leaderboard["vendor_key"].notna().sum())
    print(f"\nprocurement-lobbying leaderboard: {len(leaderboard):,} clients "
          f"({matched_n:,} matched to a DOE payee, "
          f"{len(leaderboard) - matched_n:,} unmatched)")
    print(f"  reported compensation, all clients: "
          f"${leaderboard.compensation.sum()/1e6:,.2f}M")

    # ---- contract-level convergence, graded by industry ------------------
    contracts = load_contracts()
    rates = industry_base_rates(contracts)
    print("\nnon-competitive base rate by industry:")
    for row in rates.itertuples():
        print(f"  {row.industry:<24} {row.noncomp_rate:6.1%}  "
              f"({int(row.n_noncompetitive):,} of {int(row.n_contracts):,})")

    contracts = contracts.merge(
        rates[["industry", "noncomp_rate"]], on="industry", how="left")

    # Year-aware join: a filing counts only if it is in the award year or the
    # LOOKBACK_YEARS before it.
    windows = proc[["vendor_key", "report_year", "compensation_total"]].copy()
    expanded = []
    for offset in range(0, LOOKBACK_YEARS + 1):
        shifted = windows.copy()
        shifted["fiscal_year"] = shifted["report_year"] + offset
        expanded.append(shifted)
    windows = pd.concat(expanded, ignore_index=True)
    windows = windows.groupby(["vendor_key", "fiscal_year"]).agg(
        lob_filings_in_window=("report_year", "size"),
        lob_comp_in_window=("compensation_total", "sum"),
    ).reset_index()

    noncomp = contracts[contracts["noncompetitive"] == 1].copy()
    joined = noncomp.merge(windows, on=["vendor_key", "fiscal_year"], how="left")
    joined["lob_filings_in_window"] = joined["lob_filings_in_window"].fillna(0).astype(int)

    # Industry-adjusted salience: how much this award departs from the norm for
    # its own line of work. An industry that is 80% non-competitive makes a
    # non-competitive award unremarkable; one at 10% does not.
    joined["industry_salience"] = 1.0 - joined["noncomp_rate"]

    convergent = joined[joined["lob_filings_in_window"] > 0].copy()
    convergent = convergent.merge(
        vendor_lobby[["vendor_key", "lob_confidence", "lob_method",
                      "lob_officials", "lob_clients"]],
        on="vendor_key", how="left")
    convergent = convergent.sort_values(
        ["industry_salience", "prime_contract_original_amount"], ascending=False)

    print(f"\nnon-competitive contracts with lobbying in the award window: "
          f"{len(convergent):,}")
    print(f"  covering {convergent.vendor_key.nunique():,} vendors, "
          f"FY{int(convergent.fiscal_year.min())}-FY{int(convergent.fiscal_year.max())}, "
          f"${convergent.prime_contract_original_amount.sum()/1e6:,.1f}M")

    if not convergent.empty:
        print("\ntop convergent contracts by industry salience x value:")
        show = convergent.head(12)[
            ["prime_vendor", "fiscal_year", "industry",
             "prime_contract_original_amount", "prime_contract_award_method",
             "noncomp_rate", "lob_filings_in_window"]]
        with pd.option_context("display.width", 220, "display.max_colwidth", 34):
            print(show.to_string(index=False))

    # ---- persist ---------------------------------------------------------
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    filings.drop(columns=["all_targets", "all_activities"]).to_parquet(
        PANEL_DIR / "lobbying_filings.parquet", compression="zstd", index=False)
    vendor_lobby.to_parquet(
        PANEL_DIR / "lobbying_vendors.parquet", compression="zstd", index=False)
    convergent.to_parquet(
        PANEL_DIR / "lobbying_convergence.parquet", compression="zstd", index=False)
    leaderboard.to_parquet(
        PANEL_DIR / "lobbying_leaderboard.parquet", compression="zstd", index=False)
    if not matches.empty:
        matches.to_parquet(
            PANEL_DIR / "lobbying_matches.parquet", compression="zstd", index=False)
    rates.to_parquet(
        PANEL_DIR / "industry_award_rates.parquet", compression="zstd", index=False)

    if not review.empty:
        out = ROOT / "data" / "lobbying_match_review.csv"
        review.to_csv(out, index=False)
        print(f"\nreview queue -> {out.relative_to(ROOT)}  ({len(review):,} pairs)")
        print("  these are near-misses and ambiguous names; confirm by hand before use")

    print("\nwrote lobbying_filings, lobbying_vendors, lobbying_leaderboard, "
          "lobbying_convergence, industry_award_rates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
