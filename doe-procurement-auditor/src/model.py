"""Predict vendor entrenchment, validated temporally.

Design choices that matter more than the algorithm:

* **Temporal split with an embargo.** Train on base years <= TRAIN_END, test on
  base years >= TRAIN_END + HORIZON + 1. The gap guarantees no training row's
  5-year label window overlaps a test row's, so nothing about the test period
  reaches the model. ``--demo-leakage`` reruns the same features under a random
  split to show how much apparent skill that would invent.

* **A real baseline.** Ranking vendors by current share of DOE spending already
  predicts entrenchment well. The model has to beat that to be worth anything,
  so it is reported alongside every result.

* **Precision@k, not just AUC.** The practical use is handing an auditor a
  shortlist, so what matters is how many of the top 50 are genuinely entrenched.

    python3 src/model.py
    python3 src/model.py --demo-leakage
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
PANEL_DIR = ROOT / "data" / "panel"
HORIZON = 5
LABEL = "entrenched_next5"

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


def clean(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    matrix = frame[columns].replace([np.inf, -np.inf], np.nan)
    return matrix.astype(float)


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    if len(scores) < k:
        k = len(scores)
    top = np.argsort(-scores)[:k]
    return float(y_true[top].mean())


def evaluate(name: str, y_true: np.ndarray, scores: np.ndarray) -> dict:
    return {
        "model": name,
        "auc": roc_auc_score(y_true, scores),
        "pr_auc": average_precision_score(y_true, scores),
        "p@50": precision_at_k(y_true, scores, 50),
        "p@200": precision_at_k(y_true, scores, 200),
    }


def report(rows: list[dict], title: str) -> None:
    print(f"\n{title}")
    frame = pd.DataFrame(rows)
    frame[["auc", "pr_auc", "p@50", "p@200"]] = frame[
        ["auc", "pr_auc", "p@50", "p@200"]].round(3)
    print(frame.to_string(index=False))


def fit_and_score(train: pd.DataFrame, test: pd.DataFrame, columns: list[str]) -> list[dict]:
    x_train, x_test = clean(train, columns), clean(test, columns)
    y_train, y_test = train[LABEL].to_numpy(), test[LABEL].to_numpy()

    results = [evaluate("baseline: share_of_doe",
                        y_test, np.nan_to_num(test["share_of_doe"].to_numpy()))]

    logistic = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    )
    # A column that is entirely NaN in train has a NaN median, so fillna leaves
    # it NaN; the trailing fillna(0) covers that case. Medians come from train
    # only -- using test statistics would leak.
    medians = x_train.median()
    logistic.fit(x_train.fillna(medians).fillna(0), y_train)
    results.append(evaluate(
        "logistic regression", y_test,
        logistic.predict_proba(x_test.fillna(medians).fillna(0))[:, 1]))

    boosted = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_depth=5,
        l2_regularization=1.0, random_state=0,
    )
    boosted.fit(x_train, y_train)  # native NaN handling; no imputation needed
    results.append(evaluate("gradient boosting", y_test,
                            boosted.predict_proba(x_test)[:, 1]))
    return results, boosted, x_test


def permutation_importance(model, x_test: pd.DataFrame, y_test: np.ndarray,
                           columns: list[str], repeats: int = 3) -> pd.DataFrame:
    base = roc_auc_score(y_test, model.predict_proba(x_test)[:, 1])
    rng = np.random.default_rng(0)
    drops = []
    for column in columns:
        losses = []
        for _ in range(repeats):
            shuffled = x_test.copy()
            shuffled[column] = rng.permutation(shuffled[column].to_numpy())
            losses.append(base - roc_auc_score(y_test, model.predict_proba(shuffled)[:, 1]))
        drops.append({"feature": column, "auc_drop": float(np.mean(losses))})
    return pd.DataFrame(drops).sort_values("auc_drop", ascending=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-leakage", action="store_true",
                        help="also run a random split to show inflated scores")
    args = parser.parse_args()

    table = pd.read_parquet(PANEL_DIR / "model_table.parquet")
    columns = [c for c in FEATURES if c in table.columns]
    base_years = sorted(table.fiscal_year.unique())
    print(f"model table: {len(table):,} rows, base years "
          f"FY{base_years[0]}-FY{base_years[-1]}, {len(columns)} features")

    if len(base_years) < 2:
        print("\nNot enough base years yet -- rerun once the full pull completes.")
        return 1

    # Embargo: leave HORIZON years between the last train base year and the
    # first test base year so no label windows overlap.
    train_end = base_years[0] + (len(base_years) - 1) // 2
    test_start = train_end + HORIZON + 1
    if test_start > base_years[-1]:  # short history: fall back to a 1-year gap
        test_start = train_end + 1
        print("  (history too short for a full embargo; using a 1-year gap)")

    train = table[table.fiscal_year <= train_end]
    test = table[table.fiscal_year >= test_start]
    print(f"  train: base years <= FY{train_end}  ({len(train):,} rows, "
          f"{train[LABEL].mean():.2%} positive)")
    print(f"  test:  base years >= FY{test_start}  ({len(test):,} rows, "
          f"{test[LABEL].mean():.2%} positive)")

    if train.empty or test.empty or train[LABEL].nunique() < 2 or test[LABEL].nunique() < 2:
        print("\nSplit leaves a side without both classes -- need the full pull.")
        return 1

    results, model, x_test = fit_and_score(train, test, columns)
    report(results, "TEMPORAL SPLIT (honest)")

    importance = permutation_importance(model, x_test, test[LABEL].to_numpy(), columns)
    print("\ntop features by permutation AUC drop")
    print(importance.head(12).to_string(index=False))

    if args.demo_leakage:
        shuffled_train, shuffled_test = train_test_split(
            table, test_size=0.3, random_state=0, stratify=table[LABEL])
        leaked, _, _ = fit_and_score(shuffled_train, shuffled_test, columns)
        report(leaked, "RANDOM SPLIT (leaky -- for contrast only)")
        print("\nThe random split scores higher because the same vendor appears on both\n"
              "sides in adjacent years. That gap is the leak, not skill.")

    scored = test.copy()
    scored["entrenchment_score"] = model.predict_proba(x_test)[:, 1]
    out = PANEL_DIR / "scored_vendors.parquet"
    scored.to_parquet(out, compression="zstd", index=False)
    print(f"\nwrote {out.name}")

    latest = scored[scored.fiscal_year == scored.fiscal_year.max()]
    print(f"\ntop 15 by entrenchment score, FY{int(scored.fiscal_year.max())}")
    print(latest.nlargest(15, "entrenchment_score")[
        ["vendor_key", "spend", "share_of_doe", "tenure",
         "ct_noncompetitive_share", "entrenchment_score", LABEL]
    ].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
