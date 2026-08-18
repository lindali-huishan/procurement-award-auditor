"""Collapse ~21.8M transactions into a vendor x fiscal-year panel.

The giant Parquet set is touched only here. Everything downstream -- features,
labels, models, charts -- works off the panel, which is small enough for pandas.

Normalization is applied to the *distinct* payee names (tens of thousands) and
joined back, rather than run as a UDF over every row. Same result, far faster.

Outputs to data/panel/:
    vendor_map.parquet    raw payee name -> normalized key + classification
    vendor_year.parquet   one row per (vendor, fiscal year)
    agency_year.parquet   DOE totals by spending category, for the payroll shift

    python3 src/build_panel.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from normalize import classify, normalize_vendor  # noqa: E402
from resolve import build_canonical_map  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RAW_GLOB = str(ROOT / "data" / "raw" / "fy*" / "page_*.parquet")
PANEL_DIR = ROOT / "data" / "panel"


def build_vendor_map(con: duckdb.DuckDBPyConnection) -> int:
    # Weight by spend so the canonical spelling is the one carrying the money.
    rows = con.execute(f"""
        SELECT payee_name, COALESCE(SUM(check_amount), 0) AS spend
        FROM read_parquet('{RAW_GLOB}')
        GROUP BY 1
    """).fetchall()
    names = [r[0] for r in rows]
    print(f"  {len(names):,} distinct payee names")

    keys = [normalize_vendor(n) for n in names]
    classes = [classify(n) for n in names]
    print(f"  exact normalization -> {len(set(keys)):,} keys "
          f"({len(names) - len(set(keys)):,} variants merged)")

    # Resolution runs over real vendors only; governmental counterparties and
    # redacted rows must not be fused into anything.
    weights: dict[str, float] = {}
    for key, cls, (_, spend) in zip(keys, classes, rows):
        if cls == "vendor" and key:
            weights[key] = weights.get(key, 0.0) + float(spend or 0)

    canonical = build_canonical_map(weights)
    resolved = [canonical.get(k, k) if c == "vendor" else k
                for k, c in zip(keys, classes)]
    vendor_keys = {k for k, c in zip(keys, classes) if c == "vendor" and k}
    vendor_canon = {canonical.get(k, k) for k in vendor_keys}
    print(f"  entity resolution   -> {len(vendor_canon):,} vendors "
          f"({len(vendor_keys) - len(vendor_canon):,} further merged)")

    table = pa.table({
        "payee_name": pa.array(names, type=pa.string()),
        "vendor_key": pa.array(resolved, type=pa.string()),
        "vendor_key_exact": pa.array(keys, type=pa.string()),
        "payee_class": pa.array(classes, type=pa.string()),
    })
    pq.write_table(table, PANEL_DIR / "vendor_map.parquet", compression="zstd")
    return len(names)


def build_panel(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(f"""
        CREATE OR REPLACE VIEW txn AS
        SELECT r.*, m.vendor_key, m.payee_class
        FROM read_parquet('{RAW_GLOB}') r
        LEFT JOIN read_parquet('{PANEL_DIR / "vendor_map.parquet"}') m
               ON r.payee_name = m.payee_name
    """)

    # Agency-level totals: cheap, and this alone answers the payroll-to-contracts
    # shift without any modeling.
    con.execute(f"""
        COPY (
            SELECT fiscal_year, spending_category,
                   SUM(check_amount) AS spend,
                   COUNT(*)          AS txn_count
            FROM txn GROUP BY 1, 2 ORDER BY 1, 2
        ) TO '{PANEL_DIR / "agency_year.parquet"}' (FORMAT parquet, COMPRESSION zstd)
    """)

    # The vendor panel covers procurement only: payroll rows are department
    # summaries rather than vendors, and redacted payees are not identifiable.
    con.execute(f"""
        COPY (
            WITH v AS (
                SELECT * FROM txn
                WHERE payee_class = 'vendor'
                  AND spending_category <> 'Payroll'
                  AND vendor_key <> ''
                  AND check_amount IS NOT NULL
            ),
            per_vendor_year AS (
                SELECT
                    vendor_key,
                    fiscal_year,
                    SUM(check_amount)                          AS spend,
                    COUNT(*)                                   AS txn_count,
                    COUNT(DISTINCT NULLIF(contract_id, ''))    AS n_contracts,
                    COUNT(DISTINCT NULLIF(department, ''))     AS n_departments,
                    COUNT(DISTINCT NULLIF(expense_category,'')) AS n_expense_categories,
                    COUNT(DISTINCT payee_name)                 AS n_name_variants,
                    SUM(CASE WHEN spending_category = 'Capital Contracts'
                             THEN check_amount ELSE 0 END)     AS capital_spend,
                    MIN(issue_date)                            AS first_issue_date,
                    MAX(issue_date)                            AS last_issue_date,
                    -- Attributes vary across a vendor's rows; take the modal value.
                    MODE(NULLIF(industry, ''))                 AS industry,
                    MODE(NULLIF(mwbe_category, ''))            AS mwbe_category,
                    MAX(CASE WHEN TRIM(woman_owned_business) = 'Yes' THEN 1 ELSE 0 END) AS woman_owned,
                    MAX(CASE WHEN TRIM(emerging_business)   = 'Yes' THEN 1 ELSE 0 END) AS emerging_business,
                    MAX(CASE WHEN TRIM(mocs_registered)     = 'Yes' THEN 1 ELSE 0 END) AS mocs_registered,
                    MAX(CASE WHEN TRIM(sub_vendor)          = 'Yes' THEN 1 ELSE 0 END) AS acts_as_sub,
                    COUNT(DISTINCT NULLIF(associated_prime_vendor, 'N/A')) AS n_primes_above
                FROM v GROUP BY 1, 2
            )
            SELECT
                p.*,
                -- Share of all DOE vendor spending that year, and within industry.
                p.spend / SUM(p.spend) OVER (PARTITION BY p.fiscal_year) AS share_of_doe,
                p.spend / NULLIF(SUM(p.spend) OVER (
                    PARTITION BY p.fiscal_year, p.industry), 0)          AS share_of_industry,
                p.capital_spend / NULLIF(p.spend, 0)                     AS capital_ratio
            FROM per_vendor_year p
            ORDER BY fiscal_year, spend DESC
        ) TO '{PANEL_DIR / "vendor_year.parquet"}' (FORMAT parquet, COMPRESSION zstd)
    """)


def main() -> int:
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()

    total = con.execute(f"SELECT COUNT(*) FROM read_parquet('{RAW_GLOB}')").fetchone()[0]
    print(f"Reading {total:,} transactions")

    print("\nBuilding vendor map...")
    build_vendor_map(con)

    print("\nBuilding panel...")
    build_panel(con)

    rows = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{PANEL_DIR / 'vendor_year.parquet'}')"
    ).fetchone()[0]
    vendors = con.execute(
        f"SELECT COUNT(DISTINCT vendor_key) FROM read_parquet('{PANEL_DIR / 'vendor_year.parquet'}')"
    ).fetchone()[0]
    print(f"\n  vendor_year.parquet  {rows:,} vendor-years, {vendors:,} distinct vendors")
    for path in sorted(PANEL_DIR.glob("*.parquet")):
        print(f"    {path.name:<24} {path.stat().st_size/1e6:>7.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
