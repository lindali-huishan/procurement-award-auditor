"""Which vendors behave unlike their peers?

Every supervised attempt in this project failed for one reason: the question was
"is this vendor big?", and spend size answers that trivially. This module asks a
question size cannot shortcut -- whether a vendor's *behaviour* departs from
firms doing similar work at a similar scale.

Size is removed by construction, in two steps:

1. **Peer groups.** Vendors are bucketed by industry x spend decile *within the
   year*, so comparisons only ever happen between similar-sized firms in the
   same line of work.
2. **Robust residuals.** Each behavioural feature is converted to a robust
   z-score against its peer group's median and MAD. A vendor is unusual only
   relative to its peers, never in absolute dollars.

Because there is no label, three detectors vote and only their consensus is
reported. Any single unsupervised method finds "anomalies" in pure noise; three
independent methods agreeing is far harder to fake.

    Mahalanobis  -- correlated deviation across the whole profile
    IsolationForest -- unusual in a way that is easy to partition off
    LocalOutlierFactor -- unusual relative to nearest neighbours specifically

Interpretability matters more than the score. For every flagged vendor the top
contributing features are reported, so a human can see *why* it surfaced and
dismiss it if the reason is benign.

    python3 src/peer_anomaly.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import MinCovDet
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

ROOT = Path(__file__).resolve().parent.parent
PANEL_DIR = ROOT / "data" / "panel"

# Behavioural features only. No raw dollar amount appears here -- that is the
# whole point. Each describes *how* a vendor transacts, not how much.
BEHAVIOUR = [
    "txn_per_contract",      # payment fragmentation
    "mean_payment",          # typical cheque size for its scale
    "payment_cv",            # burstiness of payment sizes
    "dept_concentration",    # served by one department or many
    "category_spread",       # narrow specialist vs broad supplier
    "capital_ratio",         # capital vs expense mix
    "growth_vs_peers",       # growth relative to peer-group median
    "tenure_ratio",          # tenure relative to peers
    "q4_share",              # share of payments in the year-end run-up
    "name_variants",         # many spellings can indicate messy entities
]

MIN_PEER_GROUP = 30      # below this, "unusual" is not meaningful
CONSENSUS_QUANTILE = 0.98


def build_behaviour(panel: pd.DataFrame, raw_seasonality: pd.DataFrame | None) -> pd.DataFrame:
    frame = panel.copy()
    frame["txn_per_contract"] = frame["txn_count"] / frame["n_contracts"].clip(lower=1)
    frame["mean_payment"] = frame["spend"] / frame["txn_count"].clip(lower=1)
    frame["dept_concentration"] = 1.0 / frame["n_departments"].clip(lower=1)
    frame["category_spread"] = frame["n_expense_categories"]
    frame["name_variants"] = frame["n_name_variants"]
    frame["capital_ratio"] = frame["capital_ratio"].fillna(0)

    frame = frame.sort_values(["vendor_key", "fiscal_year"])
    grouped = frame.groupby("vendor_key", sort=False)
    frame["prev_spend"] = grouped["spend"].shift(1)
    frame["growth"] = frame["spend"] / frame["prev_spend"].replace(0, np.nan)
    frame["tenure_ratio"] = grouped.cumcount() + 1

    # Payment-size dispersion needs the transaction level; approximate with the
    # spread of a vendor's own yearly mean payment over time.
    frame["payment_cv"] = grouped["mean_payment"].transform(
        lambda s: s.rolling(3, min_periods=2).std() / s.rolling(3, min_periods=2).mean())

    if raw_seasonality is not None and not raw_seasonality.empty:
        frame = frame.merge(raw_seasonality, on=["vendor_key", "fiscal_year"], how="left")
    else:
        frame["q4_share"] = np.nan
    return frame


def peer_groups(frame: pd.DataFrame) -> pd.Series:
    """industry x within-year spend decile."""
    industry = frame["industry"].fillna("UNKNOWN")
    decile = frame.groupby("fiscal_year")["spend"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 10, labels=False, duplicates="drop"))
    return (industry.astype(str) + "|" + frame["fiscal_year"].astype(str)
            + "|d" + decile.astype("Int64").astype(str))


def robust_z(frame: pd.DataFrame, columns: list[str], group: pd.Series) -> pd.DataFrame:
    """Median/MAD z-scores computed within peer group."""
    out = pd.DataFrame(index=frame.index)
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        median = values.groupby(group).transform("median")
        mad = (values - median).abs().groupby(group).transform("median")
        # 1.4826 scales MAD to a standard deviation for normal data.
        scale = (mad * 1.4826).replace(0, np.nan)
        out[column] = ((values - median) / scale).clip(-10, 10)
    return out


def detect(residuals: pd.DataFrame) -> pd.DataFrame:
    """Three detectors, each rank-normalised, then averaged into a consensus."""
    matrix = residuals.fillna(0).to_numpy()
    scores = pd.DataFrame(index=residuals.index)

    try:
        robust_cov = MinCovDet(random_state=0, support_fraction=0.9).fit(matrix)
        scores["mahalanobis"] = robust_cov.mahalanobis(matrix)
    except Exception:
        centred = matrix - np.median(matrix, axis=0)
        scores["mahalanobis"] = (centred ** 2).sum(axis=1)

    forest = IsolationForest(n_estimators=300, random_state=0, contamination="auto")
    scores["isolation"] = -forest.fit(matrix).score_samples(matrix)

    neighbours = min(35, max(5, len(matrix) // 20))
    lof = LocalOutlierFactor(n_neighbors=neighbours)
    lof.fit(matrix)
    scores["lof"] = -lof.negative_outlier_factor_

    for column in scores.columns:
        scores[column] = scores[column].rank(pct=True)
    scores["consensus"] = scores[["mahalanobis", "isolation", "lof"]].mean(axis=1)
    # Agreement: low spread across detectors means they tell the same story.
    scores["agreement"] = 1 - scores[["mahalanobis", "isolation", "lof"]].std(axis=1)
    return scores


def explain(residuals: pd.DataFrame, index, top: int = 3) -> str:
    row = residuals.loc[index].dropna()
    if row.empty:
        return ""
    ordered = row.reindex(row.abs().sort_values(ascending=False).index)[:top]
    return ", ".join(
        f"{name} {'+' if value > 0 else ''}{value:.1f}sd" for name, value in ordered.items())


def main() -> int:
    panel = pd.read_parquet(PANEL_DIR / "vendor_year.parquet")
    seasonality_path = PANEL_DIR / "vendor_seasonality.parquet"
    seasonality = (pd.read_parquet(seasonality_path)
                   if seasonality_path.exists() else None)

    frame = build_behaviour(panel, seasonality)
    frame["peer_group"] = peer_groups(frame)

    sizes = frame.groupby("peer_group")["vendor_key"].transform("size")
    usable = frame[sizes >= MIN_PEER_GROUP].copy()
    print(f"{len(frame):,} vendor-years -> {len(usable):,} in peer groups of "
          f">= {MIN_PEER_GROUP} ({usable.peer_group.nunique():,} groups)")

    columns = [c for c in BEHAVIOUR if c in usable.columns]
    # growth_vs_peers is itself a peer-relative quantity.
    usable["growth_vs_peers"] = usable["growth"]
    residuals = robust_z(usable, columns, usable["peer_group"])

    coverage = residuals.notna().mean().sort_values()
    print("\nfeature coverage after peer normalisation:")
    print(coverage.round(3).to_string())

    scores = detect(residuals)
    usable = usable.join(scores)

    cutoff = usable["consensus"].quantile(CONSENSUS_QUANTILE)
    flagged = usable[usable["consensus"] >= cutoff].copy()
    flagged["why"] = [explain(residuals, i) for i in flagged.index]

    print(f"\nflagged {len(flagged):,} vendor-years at the "
          f"{CONSENSUS_QUANTILE:.0%} consensus threshold")
    print("\ntop 15 peer-relative anomalies (recent years):")
    recent = flagged[flagged.fiscal_year >= flagged.fiscal_year.max() - 3]
    show = recent.nlargest(15, "consensus")[
        ["vendor_key", "fiscal_year", "spend", "industry",
         "consensus", "agreement", "why"]]
    with pd.option_context("display.width", 250, "display.max_colwidth", 60):
        print(show.to_string(index=False))

    out = PANEL_DIR / "peer_anomalies.parquet"
    usable[["vendor_key", "fiscal_year", "spend", "industry", "peer_group",
            "consensus", "agreement", "mahalanobis", "isolation", "lof"]].to_parquet(
        out, compression="zstd", index=False)
    print(f"\nwrote {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
