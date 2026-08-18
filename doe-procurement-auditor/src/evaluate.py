"""Can entrenchment actually be predicted? Three tests, in order of strictness.

Entrenchment has no ground truth -- no published list of entrenched DOE vendors
exists. The label is a *constructed* definition, so out-of-time accuracy against
it proves only that the model reproduces that definition. That is necessary but
nowhere near sufficient, so this module runs three tests:

1. WALK-FORWARD ACCURACY
   Refit at each base year, always predicting the following window. Reports the
   model against the size baseline, because a model that cannot beat "rank by
   current share" is not adding anything.

2. LABEL SENSITIVITY
   Re-derive the label under different thresholds and see whether the top-ranked
   vendors survive. If the top 50 churns completely when the percentile moves
   from 90 to 85, the score is an artifact of an arbitrary cutoff, not a finding.

3. EXTERNAL VALIDATION
   Score vendors using only their own history and ask whether the pupil
   transportation sector -- DOE's documented long-running lock-in case, where
   contracts ran for decades without competitive rebidding -- rises to the top
   without ever being told about it. This is the only test whose answer does not
   come from the label's own definition, so it is the one that carries
   credibility.

The EMERGENCE framing (--emergence) drops vendors already in the top decile at
time T. Predicting that a giant stays a giant is autocorrelation; predicting
which unremarkable vendor becomes entrenched is the actual question.

    python3 src/evaluate.py --emergence
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
PANEL_DIR = ROOT / "data" / "panel"
LABEL = "entrenched_next5"
HORIZON = 5

FEATURES = [
    "tenure", "years_paid_to_date", "consecutive_years",
    "log_spend", "spend_growth_1y", "spend_cagr_3y", "spend_volatility",
    "share_of_doe", "share_of_industry", "share_delta_3y",
    "n_contracts", "n_departments", "n_expense_categories", "n_name_variants",
    "capital_ratio", "mocs_registered", "woman_owned", "emerging_business",
    "acts_as_sub", "n_primes_above",
    "ct_n_contracts", "ct_value", "ct_noncompetitive_share",
    "ct_renewal_share", "ct_mean_duration_days", "ct_amendment_ratio",
]

# DOE's pupil transportation contracts are the textbook lock-in case: routes ran
# for decades on contracts that were not competitively rebid. Matched by name
# because the industry field is sparsely populated in the spending feed.
BUS_PATTERNS = ["BUS", "TRANSIT", "TRANSPORT", "COACH", "CHARTER SERVICE"]
# Guard against false positives from these words in unrelated firm names.
BUS_EXCLUDE = ["BUSINESS", "BUSH", "TRANSITION"]


def is_transportation(name: str) -> bool:
    upper = (name or "").upper()
    if any(bad in upper for bad in BUS_EXCLUDE):
        return False
    return any(pattern in upper for pattern in BUS_PATTERNS)


def pick_split(table: pd.DataFrame) -> tuple[int | None, int | None]:
    """Latest cut year whose test year (cut + HORIZON + 1) still exists.

    Using the latest valid cut gives the model the most training history while
    keeping the embargo intact.
    """
    years = sorted(table.fiscal_year.unique())
    for cut in reversed(years):
        if cut + HORIZON + 1 in years:
            return int(cut), int(cut + HORIZON + 1)
    return None, None


def model_matrix(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return frame[columns].replace([np.inf, -np.inf], np.nan).astype(float)


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    k = min(k, len(scores))
    return float(y_true[np.argsort(-scores)[:k]].mean()) if k else float("nan")


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, columns: list[str]):
    model = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_depth=5,
        l2_regularization=1.0, random_state=0,
    )
    model.fit(model_matrix(train, columns), train[LABEL].to_numpy())
    return model, model.predict_proba(model_matrix(test, columns))[:, 1]


def walk_forward(table: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Refit at each base year; test on the first year fully clear of the label window."""
    years = sorted(table.fiscal_year.unique())
    rows = []
    for cut in years:
        test_year = cut + HORIZON + 1
        if test_year not in years:
            continue
        train = table[table.fiscal_year <= cut]
        test = table[table.fiscal_year == test_year]
        if train[LABEL].nunique() < 2 or test[LABEL].nunique() < 2 or len(test) < 50:
            continue

        _, scores = fit_predict(train, test, columns)
        y_true = test[LABEL].to_numpy()
        baseline = np.nan_to_num(test["share_of_doe"].to_numpy())
        rows.append({
            "train<=FY": cut,
            "test FY": test_year,
            "n_test": len(test),
            "pos_rate": y_true.mean(),
            "model_auc": roc_auc_score(y_true, scores),
            "base_auc": roc_auc_score(y_true, baseline),
            "model_pr": average_precision_score(y_true, scores),
            "base_pr": average_precision_score(y_true, baseline),
            "model_p@50": precision_at_k(y_true, scores, 50),
            "base_p@50": precision_at_k(y_true, baseline, 50),
        })
    return pd.DataFrame(rows)


def label_sensitivity(table: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Does the top-50 list survive moving the label threshold?"""
    cut, test_year = pick_split(table)
    if cut is None:
        return pd.DataFrame()

    train = table[table.fiscal_year <= cut]
    test = table[table.fiscal_year == test_year]
    if train[LABEL].nunique() < 2 or test[LABEL].nunique() < 2:
        return pd.DataFrame()

    _, base_scores = fit_predict(train, test, columns)
    reference = set(test.iloc[np.argsort(-base_scores)[:50]]["vendor_key"])

    rows = []
    for percentile in (80, 85, 90, 95):
        # Rebuild the label at this percentile and RETRAIN. Comparing the
        # original ranking against itself would be vacuous; the question is
        # whether a differently-labelled model picks the same vendors.
        alt_train = train.copy()
        alt_test = test.copy()
        for frame in (alt_train, alt_test):
            cutoff = frame.groupby("fiscal_year")["share_of_doe"].transform(
                lambda s: s.quantile(percentile / 100))
            frame[LABEL] = ((frame["consecutive_years"] >= 3)
                            & (frame["share_of_doe"] >= cutoff)).astype(int)
        if alt_train[LABEL].nunique() < 2 or alt_test[LABEL].nunique() < 2:
            continue

        _, alt_scores = fit_predict(alt_train, alt_test, columns)
        alt_top = set(alt_test.iloc[np.argsort(-alt_scores)[:50]]["vendor_key"])
        rows.append({
            "percentile": percentile,
            "pos_rate": alt_train[LABEL].mean(),
            "auc_vs_alt_label": roc_auc_score(alt_test[LABEL].to_numpy(), alt_scores),
            "top50_overlap_with_base": len(reference & alt_top) / 50,
        })
    return pd.DataFrame(rows)


def external_validation(table: pd.DataFrame, columns: list[str], note: str = "") -> None:
    """Do transportation vendors surface without the model being told about them?"""
    if note:
        print(f"  [{note}]")
    cut, test_year = pick_split(table)
    if cut is None:
        print("  not enough years for external validation")
        return

    train = table[table.fiscal_year <= cut]
    test = table[table.fiscal_year == test_year].copy()
    if train[LABEL].nunique() < 2 or len(test) < 50:
        print("  not enough data for external validation")
        return

    _, scores = fit_predict(train, test, columns)
    test["score"] = scores
    test["pct_rank"] = test["score"].rank(pct=True)
    test["is_transport"] = test["vendor_key"].map(is_transportation)

    transport = test[test.is_transport]
    others = test[~test.is_transport]
    if transport.empty:
        print("  no transportation vendors in the test year")
        return

    print(f"  scored FY{test_year} using only data through FY{cut}")
    print(f"  transportation vendors: {len(transport)}  |  others: {len(others)}")
    print(f"  median percentile rank -- transport {transport.pct_rank.median():.3f} "
          f"vs others {others.pct_rank.median():.3f}")

    top100 = test.nlargest(min(100, len(test)), "score")
    share_top = top100.is_transport.mean()
    share_all = test.is_transport.mean()
    lift = share_top / share_all if share_all else float("nan")
    print(f"  transport share of top 100: {share_top:.1%} vs {share_all:.1%} "
          f"overall  (lift {lift:.2f}x)")
    print("\n  highest-scoring transportation vendors:")
    print(transport.nlargest(8, "score")[
        ["vendor_key", "spend", "tenure", "score", "pct_rank"]].to_string(index=False))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emergence", action="store_true",
                        help="drop vendors already top-decile at T")
    args = parser.parse_args()

    table = pd.read_parquet(PANEL_DIR / "model_table.parquet")
    columns = [c for c in FEATURES if c in table.columns]
    full_table = table.copy()  # unfiltered, for external validation

    if args.emergence:
        before = len(table)
        threshold = table.groupby("fiscal_year")["share_of_doe"].transform(
            lambda s: s.quantile(0.90))
        table = table[table["share_of_doe"] < threshold].copy()
        print(f"EMERGENCE framing: dropped vendors already top-decile at T "
              f"({before:,} -> {len(table):,} rows, "
              f"positive rate {table[LABEL].mean():.2%})")

    print(f"\nmodel table: {len(table):,} rows, "
          f"FY{table.fiscal_year.min()}-FY{table.fiscal_year.max()}")

    print("\n" + "=" * 78 + "\n1. WALK-FORWARD ACCURACY (model vs size baseline)\n" + "=" * 78)
    forward = walk_forward(table, columns)
    if forward.empty:
        print("  not enough years yet")
    else:
        print(forward.round(3).to_string(index=False))
        print(f"\n  mean model AUC {forward.model_auc.mean():.3f} "
              f"vs baseline {forward.base_auc.mean():.3f}"
              f"   (lift {forward.model_auc.mean() - forward.base_auc.mean():+.3f})")
        print(f"  mean model P@50 {forward['model_p@50'].mean():.3f} "
              f"vs baseline {forward['base_p@50'].mean():.3f}")

    print("\n" + "=" * 78 + "\n2. LABEL SENSITIVITY (is the ranking an artifact of the cutoff?)\n" + "=" * 78)
    sensitivity = label_sensitivity(table, columns)
    print(sensitivity.round(3).to_string(index=False) if not sensitivity.empty
          else "  not enough years yet")

    print("\n" + "=" * 78 + "\n3. EXTERNAL VALIDATION (does it rediscover pupil transportation?)\n" + "=" * 78)
    # The emergence filter removes top-decile vendors, which is exactly where
    # large incumbent bus contractors sit -- so running this test only on the
    # filtered set would exclude the cases it is meant to detect. Report both.
    external_validation(full_table, columns, note="full population")
    if args.emergence:
        print()
        external_validation(table, columns, note="emergence subset")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
