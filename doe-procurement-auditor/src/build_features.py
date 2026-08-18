"""Turn the vendor-year panel into a leakage-free supervised learning table.

Framing: standing at the end of fiscal year T and seeing only years <= T,
predict whether a vendor is *entrenched* over the window T+1 .. T+5.

Entrenched means both persistent and material:
    paid in at least PERSIST_YEARS of the next HORIZON years, and
    mean share of DOE vendor spending over that window in the top
    SHARE_PERCENTILE of vendors active in the window.

Two rules keep this honest:

* Every feature is computed from years <= T. Nothing from the label window
  reaches the feature set.
* Splits are temporal, never random. A random split lets the same vendor's
  later years train a model scored on its earlier ones, which leaks the future
  backwards and produces a meaningless AUC in the high 0.9s.

    python3 src/build_features.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PANEL_DIR = ROOT / "data" / "panel"

HORIZON = 5           # label window length, in years
PERSIST_YEARS = 4     # must be paid in >= this many of the window's years
SHARE_PERCENTILE = 90 # and rank at/above this percentile of mean share


def load_panel() -> pd.DataFrame:
    panel = pd.read_parquet(PANEL_DIR / "vendor_year.parquet")
    return panel.sort_values(["vendor_key", "fiscal_year"]).reset_index(drop=True)


def load_contract_features() -> pd.DataFrame:
    """Award-method and contract-shape features per vendor-year, if pulled.

    Award method is the sharpest entrenchment signal the spending feed lacks:
    a vendor whose work arrives mostly through renewals and negotiated
    acquisitions is locked in differently from one winning sealed bids.
    """
    path = ROOT / "data" / "raw_contracts" / "contracts_registered.parquet"
    if not path.exists():
        return pd.DataFrame()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from normalize import normalize_vendor
    from resolve import build_canonical_map

    contracts = pd.read_parquet(path)
    contracts = contracts[contracts["vendor_record_type"] == "Prime Vendor"].copy()
    contracts["fiscal_year"] = pd.to_numeric(contracts["fiscal_year"], errors="coerce")
    contracts["vendor_key"] = contracts["prime_vendor"].map(normalize_vendor)

    weights = contracts.groupby("vendor_key")["prime_contract_current_amount"].sum().to_dict()
    canonical = build_canonical_map({k: float(v or 0) for k, v in weights.items()})
    contracts["vendor_key"] = contracts["vendor_key"].map(lambda k: canonical.get(k, k))

    method = contracts["prime_contract_award_method"].fillna("").str.upper()
    # "Non-competitive" here means the award did not arise from an open
    # solicitation. Renewals are included: repeated renewal is the mechanism by
    # which incumbency compounds.
    noncompetitive = method.str.contains(
        "SOLE SOURCE|NEGOTIATED|RENEWAL|ASSIGNMENT|EMERGENCY|INTERGOVERNMENTAL|MANDATE",
        regex=True,
    )
    contracts["is_noncompetitive"] = noncompetitive.astype(int)
    contracts["is_renewal"] = method.str.contains("RENEWAL").astype(int)

    start = pd.to_datetime(contracts["prime_contract_start_date"], errors="coerce")
    end = pd.to_datetime(contracts["prime_contract_end_date"], errors="coerce")
    contracts["duration_days"] = (end - start).dt.days

    grouped = contracts.groupby(["vendor_key", "fiscal_year"]).agg(
        ct_n_contracts=("prime_contract_id", "nunique"),
        ct_value=("prime_contract_current_amount", "sum"),
        ct_noncompetitive_share=("is_noncompetitive", "mean"),
        ct_renewal_share=("is_renewal", "mean"),
        ct_mean_duration_days=("duration_days", "mean"),
        ct_amendment_ratio=("prime_contract_current_amount", "sum"),
    ).reset_index()

    original = contracts.groupby(["vendor_key", "fiscal_year"])[
        "prime_contract_original_amount"].sum().reset_index(name="ct_original")
    grouped = grouped.merge(original, on=["vendor_key", "fiscal_year"], how="left")
    # >1 means the contract grew after award, a classic lock-in signature.
    grouped["ct_amendment_ratio"] = grouped["ct_value"] / grouped["ct_original"].replace(0, np.nan)
    return grouped.drop(columns=["ct_original"])


def add_history_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Features describing each vendor's trajectory up to and including year T."""
    panel = panel.copy()
    grouped = panel.groupby("vendor_key", sort=False)

    panel["first_year"] = grouped["fiscal_year"].transform("min")
    panel["tenure"] = panel["fiscal_year"] - panel["first_year"] + 1
    panel["years_paid_to_date"] = grouped.cumcount() + 1

    panel["log_spend"] = np.log1p(panel["spend"].clip(lower=0))
    panel["spend_lag1"] = grouped["spend"].shift(1)
    panel["spend_lag3"] = grouped["spend"].shift(3)
    panel["share_lag3"] = grouped["share_of_doe"].shift(3)

    panel["spend_growth_1y"] = panel["spend"] / panel["spend_lag1"].replace(0, np.nan)
    panel["spend_cagr_3y"] = (panel["spend"] / panel["spend_lag3"].replace(0, np.nan)) ** (1 / 3)
    panel["share_delta_3y"] = panel["share_of_doe"] - panel["share_lag3"]

    panel["spend_mean_3y"] = grouped["spend"].transform(
        lambda s: s.rolling(3, min_periods=1).mean())
    panel["spend_std_3y"] = grouped["spend"].transform(
        lambda s: s.rolling(3, min_periods=2).std())
    # Coefficient of variation: steady incumbents look different from spiky ones.
    panel["spend_volatility"] = panel["spend_std_3y"] / panel["spend_mean_3y"].replace(0, np.nan)

    # Consecutive-year streak ending at T, the plainest persistence measure.
    panel["year_gap"] = grouped["fiscal_year"].diff()
    streak, previous_vendor, current = [], None, 0
    for vendor, gap in zip(panel["vendor_key"], panel["year_gap"]):
        current = 1 if vendor != previous_vendor or gap != 1 else current + 1
        streak.append(current)
        previous_vendor = vendor
    panel["consecutive_years"] = streak
    return panel


def add_labels(panel: pd.DataFrame) -> pd.DataFrame:
    """Attach the forward-looking entrenchment label for each (vendor, T)."""
    spend = panel.pivot_table(index="vendor_key", columns="fiscal_year",
                              values="spend", aggfunc="sum")
    share = panel.pivot_table(index="vendor_key", columns="fiscal_year",
                              values="share_of_doe", aggfunc="sum")
    years = sorted(c for c in spend.columns)

    records = []
    for base_year in years:
        window = [y for y in years if base_year < y <= base_year + HORIZON]
        if len(window) < HORIZON:
            continue  # incomplete future; no label can be formed

        active = spend[window].notna().sum(axis=1)
        mean_share = share[window].mean(axis=1)

        # Threshold set among vendors present in the window, so the label means
        # "large relative to peers of that era" rather than a fixed dollar bar.
        present = mean_share[active > 0]
        cutoff = np.nanpercentile(present, SHARE_PERCENTILE) if len(present) else np.inf

        label = ((active >= PERSIST_YEARS) & (mean_share >= cutoff)).astype(int)
        records.append(pd.DataFrame({
            "vendor_key": label.index,
            "fiscal_year": base_year,
            "entrenched_next5": label.values,
            "label_cutoff_share": cutoff,
        }))

    labels = pd.concat(records, ignore_index=True)
    return panel.merge(labels, on=["vendor_key", "fiscal_year"], how="inner")


def main() -> int:
    panel = load_panel()
    print(f"panel: {len(panel):,} vendor-years, "
          f"FY{panel.fiscal_year.min()}-FY{panel.fiscal_year.max()}")

    features = add_history_features(panel)

    contracts = load_contract_features()
    if not contracts.empty:
        features = features.merge(contracts, on=["vendor_key", "fiscal_year"], how="left")
        matched = features["ct_n_contracts"].notna().sum()
        print(f"contracts joined: {matched:,} vendor-years carry award-method features")
        for column in ["ct_n_contracts", "ct_value", "ct_noncompetitive_share",
                       "ct_renewal_share"]:
            features[column] = features[column].fillna(0)

    labelled = add_labels(features)
    print(f"labelled rows: {len(labelled):,}  "
          f"positive rate: {labelled.entrenched_next5.mean():.3%}")
    print("\nlabel base years:")
    summary = labelled.groupby("fiscal_year")["entrenched_next5"].agg(["size", "sum", "mean"])
    print(summary.to_string())

    out = PANEL_DIR / "model_table.parquet"
    labelled.to_parquet(out, compression="zstd", index=False)
    print(f"\nwrote {out.name}  {len(labelled):,} rows x {labelled.shape[1]} cols")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
