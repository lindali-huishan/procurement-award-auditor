"""Pull the DOE Contracts feed -- the award-method and contract-date signals.

The Spending feed says money moved; it never says how the vendor won the work.
This feed carries prime_contract_award_method (competitive bid vs. sole source
vs. renewal), contract start/end/registration dates, original vs. current
amount, and sub-vendor detail. Those are the sharpest entrenchment features.

Tiny next to Spending -- a few thousand rows a year -- so it lands in one
Parquet file per status and joins to spending on the contract id.

    python3 src/pull_contracts.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from checkbook import MAX_RECORDS, contract_criteria, fetch_page, record_count  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw_contracts"
FIRST_YEAR, LAST_YEAR = 2010, 2026

# Amount-like fields become floats; dates stay text and are parsed downstream.
NUMERIC_SUFFIXES = ("_amount", "_spent_to_date")


def pull_status(status: str) -> int:
    rows: list[dict[str, str]] = []
    # Pending contracts are a live snapshot: the domain rejects fiscal_year, so
    # there is one unpartitioned pull rather than a year loop.
    years = range(FIRST_YEAR, LAST_YEAR + 1) if status == "registered" else [None]
    for year in years:
        criteria = contract_criteria(year, status=status)
        total = record_count(criteria, type_of_data="Contracts")
        if not total:
            print(f"  {('FY'+str(year)) if year else 'snapshot':<9}[{status:<10}]          0")
            continue
        got = 0
        for offset in range(1, total + 1, MAX_RECORDS):
            page = fetch_page(
                criteria=criteria,
                records_from=offset,
                max_records=MAX_RECORDS,
                type_of_data="Contracts",
            )
            for record in page.records:
                if year is not None:
                    record["fiscal_year"] = str(year)
                record["contract_status_feed"] = status
            rows.extend(page.records)
            got += len(page.records)
        print(f"  {('FY'+str(year)) if year else 'snapshot':<9}[{status:<10}] {got:>10,}" + ("" if got == total else f"  (expected {total:,})"))

    if not rows:
        return 0

    fields: list[str] = []
    for record in rows:
        for key in record:
            if key not in fields:
                fields.append(key)

    arrays = []
    for field in fields:
        values = [r.get(field, "") for r in rows]
        if field.endswith(NUMERIC_SUFFIXES):
            parsed = []
            for value in values:
                try:
                    parsed.append(float(value))
                except (TypeError, ValueError):
                    parsed.append(None)
            arrays.append(pa.array(parsed, type=pa.float64()))
        else:
            arrays.append(pa.array(values, type=pa.string()))

    table = pa.Table.from_arrays(arrays, names=fields)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"contracts_{status}.parquet"
    pq.write_table(table, path, compression="zstd")
    print(f"  -> {path.name}  {table.num_rows:,} rows x {table.num_columns} cols")
    return table.num_rows


def main() -> int:
    total = 0
    for status in ("registered", "pending"):
        print(f"\n{status.upper()}")
        total += pull_status(status)
    print(f"\nTotal contract rows: {total:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
