"""Pull the full DOE spending history from Checkbook NYC into partitioned Parquet.

~21.8M transactions across FY2010-FY2026. At 20,000 records per call that is
about 1,090 requests. Each page is written to its own Parquet file, so the pull
is resumable: rerunning skips pages already on disk and only fetches the gaps.

Parquet rather than CSV because the same data lands in roughly 1GB instead of
6-8GB, and DuckDB scans it far faster.

    python3 src/pull_spending.py                  # everything
    python3 src/pull_spending.py --years 2024 2025
    python3 src/pull_spending.py --verify         # check completeness, fetch nothing
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from checkbook import (  # noqa: E402
    MAX_RECORDS,
    SPENDING_FIELDS,
    CheckbookError,
    doe_year_criteria,
    fetch_page,
    record_count,
)

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
FIRST_YEAR, LAST_YEAR = 2010, 2026

# check_amount and fiscal_year get real types; everything else stays text so no
# value is silently coerced or lost. row_offset preserves the API's ordering.
SCHEMA = pa.schema(
    [(f, pa.float64() if f == "check_amount" else pa.int32() if f == "fiscal_year" else pa.string())
     for f in SPENDING_FIELDS]
    + [("row_offset", pa.int64())]
)

_print_lock = threading.Lock()


def page_path(fiscal_year: int, offset: int) -> Path:
    return RAW_DIR / f"fy{fiscal_year}" / f"page_{offset:09d}.parquet"


def _to_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def write_page(path: Path, records: list[dict[str, str]], offset: int) -> None:
    """Write one page atomically, so an interrupted run never leaves a torn file."""
    columns: dict[str, list] = {f: [] for f in SPENDING_FIELDS}
    for record in records:
        for field in SPENDING_FIELDS:
            columns[field].append(record.get(field, ""))

    arrays = []
    for field in SPENDING_FIELDS:
        values = columns[field]
        if field == "check_amount":
            arrays.append(pa.array([_to_float(v) for v in values], type=pa.float64()))
        elif field == "fiscal_year":
            arrays.append(pa.array([int(v) if v.isdigit() else None for v in values], type=pa.int32()))
        else:
            arrays.append(pa.array(values, type=pa.string()))
    arrays.append(pa.array(list(range(offset, offset + len(records))), type=pa.int64()))

    table = pa.Table.from_arrays(arrays, schema=SCHEMA)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp, compression="zstd")
    tmp.replace(path)


def fetch_and_write(fiscal_year: int, offset: int, expected: int) -> tuple[int, str]:
    """Fetch one page unless it is already complete on disk. Returns (rows, status)."""
    path = page_path(fiscal_year, offset)
    if path.exists():
        try:
            existing = pq.ParquetFile(path).metadata.num_rows
            if existing == expected:
                return existing, "cached"
        except Exception:
            path.unlink(missing_ok=True)  # corrupt file, refetch

    page = fetch_page(
        criteria=doe_year_criteria(fiscal_year),
        records_from=offset,
        max_records=MAX_RECORDS,
    )
    write_page(path, page.records, offset)
    status = "ok" if len(page.records) == expected else f"short({len(page.records)}/{expected})"
    return len(page.records), status


def plan_year(fiscal_year: int) -> tuple[int, list[tuple[int, int]]]:
    """Return the year's total record count and its list of (offset, expected)."""
    total = record_count(doe_year_criteria(fiscal_year))
    pages = [
        (offset, min(MAX_RECORDS, total - offset + 1))
        for offset in range(1, total + 1, MAX_RECORDS)
    ]
    return total, pages


def rows_on_disk(fiscal_year: int) -> int:
    directory = RAW_DIR / f"fy{fiscal_year}"
    if not directory.exists():
        return 0
    total = 0
    for path in directory.glob("page_*.parquet"):
        try:
            total += pq.ParquetFile(path).metadata.num_rows
        except Exception:
            pass
    return total


def verify(years: list[int]) -> bool:
    print(f"{'year':<8}{'expected':>12}{'on disk':>12}{'':>4}")
    all_ok = True
    grand_expected = grand_actual = 0
    for year in years:
        expected = record_count(doe_year_criteria(year))
        actual = rows_on_disk(year)
        ok = actual == expected
        all_ok &= ok
        grand_expected += expected
        grand_actual += actual
        print(f"FY{year:<6}{expected:>12,}{actual:>12,}{'  ok' if ok else '  MISSING':>4}")
    print(f"{'TOTAL':<8}{grand_expected:>12,}{grand_actual:>12,}")
    return all_ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", type=int, nargs="+", default=list(range(FIRST_YEAR, LAST_YEAR + 1)))
    parser.add_argument("--workers", type=int, default=4, help="concurrent requests (be polite)")
    parser.add_argument("--verify", action="store_true", help="only check completeness")
    args = parser.parse_args()

    if args.verify:
        return 0 if verify(args.years) else 1

    print(f"Planning {len(args.years)} fiscal years...", flush=True)
    plan: list[tuple[int, int, int]] = []
    for year in args.years:
        total, pages = plan_year(year)
        plan.extend((year, offset, expected) for offset, expected in pages)
        print(f"  FY{year}  {total:>10,} rows  {len(pages):>4} pages", flush=True)

    grand_total = sum(expected for _, _, expected in plan)
    print(f"\n{len(plan):,} pages, {grand_total:,} rows, {args.workers} workers\n", flush=True)

    started = time.time()
    done = fetched = cached = 0
    failures: list[str] = []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(fetch_and_write, year, offset, expected): (year, offset)
            for year, offset, expected in plan
        }
        for future in as_completed(futures):
            year, offset = futures[future]
            done += 1
            try:
                rows, status = future.result()
                if status == "cached":
                    cached += 1
                else:
                    fetched += rows
                if status.startswith("short"):
                    failures.append(f"FY{year}@{offset}: {status}")
            except (CheckbookError, Exception) as exc:  # noqa: BLE001
                failures.append(f"FY{year}@{offset}: {type(exc).__name__}: {exc}")

            if done % 10 == 0 or done == len(plan):
                elapsed = time.time() - started
                rate = done / elapsed if elapsed else 0
                eta = (len(plan) - done) / rate if rate else 0
                with _print_lock:
                    print(
                        f"  {done:>5}/{len(plan)} pages | {fetched:>10,} rows fetched "
                        f"| {cached} cached | {elapsed/60:5.1f}m elapsed | ETA {eta/60:5.1f}m",
                        flush=True,
                    )

    print(f"\nDone in {(time.time()-started)/60:.1f} min. {len(failures)} problems.")
    for line in failures[:25]:
        print(f"  ! {line}")
    if len(failures) > 25:
        print(f"  ... and {len(failures)-25} more")

    print()
    return 0 if verify(args.years) else 1


if __name__ == "__main__":
    raise SystemExit(main())
