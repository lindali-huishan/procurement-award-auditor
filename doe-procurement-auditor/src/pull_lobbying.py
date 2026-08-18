"""Pull the NYC City Clerk eLobbyist feed from the Socrata open-data API.

Why this feed matters here: NYC's lobbying law treats attempts to influence
*the determination of a procurement contract* as reportable lobbying, and the
filings carry an explicit ``Procurement`` activity category. That is unusual --
most jurisdictions only cover legislative and rulemaking lobbying, which makes
their lobbying data useless for procurement work. Here it lines up directly
with the award-method question this project already asks.

The feed is small (~80k filings, all years) so this pulls the whole thing and
lets ``lobbying.py`` do the filtering. Two fields carry the payload and both
are denormalised free text that needs real parsing:

* ``lobbyist_targets``  -- ``Education, Department of (DOE)  David Banks``,
  semicolon-joined, frequently with the same target repeated dozens of times
  within one cell.
* ``lobbyist_activities`` -- ``Procurement - <free text>``, semicolon-joined.

Socrata caps a single response at 50,000 rows, so this pages on ``$offset``.

    python3 src/pull_lobbying.py
    python3 src/pull_lobbying.py --verify
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "raw_lobbying"
OUT = OUT_DIR / "lobbying.parquet"

DATASET = "fmf3-knd8"
BASE = f"https://data.cityofnewyork.us/resource/{DATASET}.json"
META = f"https://data.cityofnewyork.us/api/views/{DATASET}.json"
PAGE_SIZE = 50_000

# Socrata allows anonymous access; an app token only raises the rate limit,
# which this does not need at four requests per run.
HEADERS = {"User-Agent": "nyc-doe-spend/1.0", "Accept": "application/json"}


def _get(url: str, retries: int = 4, timeout: int = 120) -> list[dict]:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            last = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"failed after {retries} attempts: {url}\n  {last}")


def remote_count() -> int:
    rows = _get(f"{BASE}?{urllib.parse.urlencode({'$select': 'count(1)'})}")
    return int(next(iter(rows[0].values())))


def fetch_all() -> pd.DataFrame:
    total = remote_count()
    print(f"remote rows: {total:,}")

    frames, offset = [], 0
    while offset < total:
        query = urllib.parse.urlencode({
            "$limit": PAGE_SIZE,
            "$offset": offset,
            # Explicit order makes the paging deterministic. Without it Socrata
            # gives no ordering guarantee and pages can overlap or drop rows.
            "$order": ":id",
        })
        page = _get(f"{BASE}?{query}")
        if not page:
            break
        frames.append(pd.DataFrame(page))
        offset += len(page)
        print(f"  {offset:,} / {total:,}")

    frame = pd.concat(frames, ignore_index=True)
    for column in ("compensation_total", "lobbying_expenses_total", "report_year"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true",
                        help="compare rows on disk against the API's count")
    args = parser.parse_args()

    if args.verify:
        if not OUT.exists():
            print("nothing on disk yet")
            return 1
        local = len(pd.read_parquet(OUT))
        remote = remote_count()
        status = "ok" if local == remote else "STALE -- rerun without --verify"
        print(f"local {local:,} vs remote {remote:,}  [{status}]")
        return 0 if local == remote else 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = fetch_all()
    frame.to_parquet(OUT, compression="zstd", index=False)

    years = pd.to_numeric(frame.get("report_year"), errors="coerce")
    print(f"\nwrote {OUT.relative_to(ROOT)}  {len(frame):,} filings x "
          f"{frame.shape[1]} cols  {OUT.stat().st_size/1e6:.1f} MB")
    print(f"report years {int(years.min())}-{int(years.max())}, "
          f"{frame.client_name.nunique():,} distinct clients")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
