"""The five accountability findings, computed straight from the panel.

None of these need a model. They are counting exercises, which is what makes
them defensible.

    1. Vendor concentration   top-N share and HHI over time
    2. M/WBE dollar share     measured against the city's stated target
    3. Payment timing         see the caveat below
    4. Year-end spending      the March-June use-it-or-lose-it spike
    5. Payroll -> contracts   the structural shift, already visible

Caveat on (3): the Spending feed carries only ``issue_date``, the date the check
went out. There is no invoice or receipt date anywhere in this data, so the
elapsed time between a vendor billing DOE and DOE paying cannot be computed.
What *is* measurable is the lag from contract registration to first payment, and
the gaps between successive payments. This module reports those and says plainly
that they are not the prompt-payment metric.

    python3 src/analyze.py
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PANEL_DIR = ROOT / "data" / "panel"
RAW_GLOB = str(ROOT / "data" / "raw" / "fy*" / "page_*.parquet")
CONTRACTS = ROOT / "data" / "raw_contracts" / "contracts_registered.parquet"

# NYC's citywide M/WBE utilization goal, for context on finding 2.
MWBE_TARGET = 0.30


def rule(title: str) -> None:
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


def concentration() -> pd.DataFrame:
    panel = pd.read_parquet(PANEL_DIR / "vendor_year.parquet")
    rows = []
    for year, group in panel.groupby("fiscal_year"):
        total = group["spend"].sum()
        if total <= 0:
            continue
        shares = (group["spend"] / total).sort_values(ascending=False)
        rows.append({
            "fiscal_year": int(year),
            "vendors": len(group),
            "total_$B": total / 1e9,
            "top10_%": shares.head(10).sum() * 100,
            "top50_%": shares.head(50).sum() * 100,
            "top1%_%": shares.head(max(1, len(shares) // 100)).sum() * 100,
            # HHI on shares expressed in percent, the standard 0-10,000 scale.
            "HHI": float(((shares * 100) ** 2).sum()),
        })
    return pd.DataFrame(rows).sort_values("fiscal_year")


def mwbe(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    if not CONTRACTS.exists():
        return pd.DataFrame()
    return con.execute(f"""
        SELECT CAST(fiscal_year AS INT) AS fiscal_year,
               SUM(CASE WHEN prime_vendor_mwbe_category NOT IN
                        ('Non-M/WBE', 'Individuals and Others', '')
                        AND prime_vendor_mwbe_category IS NOT NULL
                   THEN prime_contract_current_amount ELSE 0 END) AS mwbe_value,
               SUM(prime_contract_current_amount) AS total_value
        FROM '{CONTRACTS}'
        WHERE vendor_record_type = 'Prime Vendor'
        GROUP BY 1 ORDER BY 1
    """).df()


def seasonality(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return con.execute(f"""
        SELECT CAST(SUBSTR(issue_date, 6, 2) AS INT) AS month,
               COUNT(*)                              AS txn_count,
               SUM(check_amount) / 1e9               AS spend_B
        FROM read_parquet('{RAW_GLOB}')
        WHERE spending_category <> 'Payroll'
          AND issue_date IS NOT NULL AND LENGTH(issue_date) >= 7
        GROUP BY 1 ORDER BY 1
    """).df()


def payroll_shift(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    frame = con.execute(
        f"SELECT * FROM read_parquet('{PANEL_DIR / 'agency_year.parquet'}')").df()
    wide = frame.pivot_table(index="fiscal_year", columns="spending_category",
                             values="spend", aggfunc="sum").fillna(0)
    wide["total"] = wide.sum(axis=1)
    for column in [c for c in wide.columns if c != "total"]:
        wide[f"{column} %"] = wide[column] / wide["total"] * 100
    return wide


def payment_timing(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Registration-to-first-payment lag. NOT the prompt-payment metric."""
    if not CONTRACTS.exists():
        return pd.DataFrame()
    return con.execute(f"""
        WITH first_pay AS (
            SELECT contract_id, MIN(issue_date) AS first_payment
            FROM read_parquet('{RAW_GLOB}')
            WHERE contract_id IS NOT NULL AND contract_id <> ''
            GROUP BY 1
        )
        SELECT CAST(c.fiscal_year AS INT) AS fiscal_year,
               COUNT(*) AS contracts_matched,
               MEDIAN(DATE_DIFF('day',
                      CAST(c.prime_contract_registration_date AS DATE),
                      CAST(f.first_payment AS DATE))) AS median_days_to_first_payment
        FROM '{CONTRACTS}' c
        JOIN first_pay f ON f.contract_id = c.prime_contract_id
        WHERE c.prime_contract_registration_date <> ''
          AND f.first_payment <> ''
        GROUP BY 1 ORDER BY 1
    """).df()


def main() -> int:
    con = duckdb.connect()
    pd.set_option("display.width", 200)

    rule("1. VENDOR CONCENTRATION")
    conc = concentration()
    print(conc.round(2).to_string(index=False))
    if len(conc) > 1:
        first, last = conc.iloc[0], conc.iloc[-1]
        print(f"\n  top-10 share: {first['top10_%']:.1f}% (FY{first.fiscal_year}) "
              f"-> {last['top10_%']:.1f}% (FY{last.fiscal_year})")
        print(f"  HHI:          {first['HHI']:.0f} -> {last['HHI']:.0f}  "
              "(<1500 unconcentrated, >2500 highly concentrated)")

    rule("2. M/WBE SHARE OF CONTRACT VALUE")
    mw = mwbe(con)
    if mw.empty:
        print("  contracts feed not present")
    else:
        mw["mwbe_%"] = mw.mwbe_value / mw.total_value * 100
        mw["gap_vs_target_pp"] = mw["mwbe_%"] - MWBE_TARGET * 100
        print(mw[["fiscal_year", "mwbe_%", "gap_vs_target_pp"]].round(2).to_string(index=False))
        overall = mw.mwbe_value.sum() / mw.total_value.sum() * 100
        print(f"\n  all years: {overall:.1f}% vs {MWBE_TARGET*100:.0f}% target "
              f"({overall - MWBE_TARGET*100:+.1f} pp)")

    rule("3. PAYMENT TIMING  (see caveat)")
    print("  The data has no invoice date, so the true prompt-payment lag is NOT")
    print("  computable. Reported instead: contract registration -> first payment.")
    timing = payment_timing(con)
    print(timing.to_string(index=False) if not timing.empty else "  no matches")

    rule("4. YEAR-END SPENDING SPIKE (non-payroll, all years pooled)")
    season = seasonality(con)
    if not season.empty:
        season["vs_median_%"] = (
            season.txn_count / season.txn_count.median() - 1) * 100
        names = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                 7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
        season["month"] = season.month.map(names)
        print(season.round(2).to_string(index=False))
        print("\n  NYC's fiscal year ends 30 June, so Mar-Jun is the run-up to year end.")

    rule("5. PAYROLL -> CONTRACTS SHIFT")
    shift = payroll_shift(con)
    percent_columns = [c for c in shift.columns if c.endswith("%")]
    print(shift[percent_columns].round(1).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
