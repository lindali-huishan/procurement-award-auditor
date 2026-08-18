"""Client for the Checkbook NYC XML API (https://www.checkbooknyc.com/api).

Quirks discovered by probing the live endpoint, all of which this module handles:

* The request body must NOT carry an ``<?xml ...?>`` prolog. If it does, the API
  returns HTTP 200 with a zero-byte body and no error message.
* ``max_records`` is capped at 20,000 per call -- 20x the documented 1,000.
* ``records_from`` is 1-indexed and has no depth limit; offsets past 1.5M work.
* Ordering is deterministic, so offset paging does not drop or duplicate rows.
* Transactions have no unique id, and genuinely identical rows exist in the
  source data. Never dedupe on content.
"""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from xml.etree import ElementTree as ET

API_URL = "https://www.checkbooknyc.com/api"
MAX_RECORDS = 20_000
DOE_AGENCY_CODE = "040"

# Every field the Spending feed returns, in the order the API emits them.
SPENDING_FIELDS = [
    "agency",
    "associated_prime_vendor",
    "budget_code",
    "capital_project",
    "contract_id",
    "mocs_registered",
    "contract_purpose",
    "check_amount",
    "department",
    "document_id",
    "expense_category",
    "fiscal_year",
    "industry",
    "issue_date",
    "mwbe_category",
    "woman_owned_business",
    "emerging_business",
    "payee_name",
    "spending_category",
    "sub_contract_reference_id",
    "sub_vendor",
]


class CheckbookError(RuntimeError):
    """The API returned ``<result>failure</result>`` or an unusable body."""


@dataclass(frozen=True)
class Page:
    """One page of transactions plus the total matching the query."""

    records: list[dict[str, str]]
    record_count: int


def contract_criteria(fiscal_year: int | None, status: str = "registered") -> list[tuple[str, str, str]]:
    """Criteria for the Contracts feed, which requires ``status`` and ``category``.

    status: pending | registered.  category: expense | revenue | all.
    """
    criteria = [
        ("agency_code", "value", DOE_AGENCY_CODE),
        ("status", "value", status),
        ("category", "value", "all"),
    ]
    if fiscal_year is not None:
        criteria.append(("fiscal_year", "value", str(fiscal_year)))
    return criteria


def _build_request(
    *,
    type_of_data: str,
    criteria: list[tuple[str, str, str]],
    records_from: int,
    max_records: int,
) -> bytes:
    parts = "".join(
        f"<criteria><name>{name}</name><type>{ctype}</type><value>{value}</value></criteria>"
        for name, ctype, value in criteria
    )
    # No XML prolog -- see module docstring.
    return (
        "<request>"
        f"<type_of_data>{type_of_data}</type_of_data>"
        f"<records_from>{records_from}</records_from>"
        f"<max_records>{max_records}</max_records>"
        f"<search_criteria>{parts}</search_criteria>"
        "</request>"
    ).encode()


def _post(body: bytes, timeout: int) -> str:
    request = urllib.request.Request(
        API_URL,
        data=body,
        method="POST",
        headers={"Content-Type": "application/xml", "User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def fetch_page(
    *,
    criteria: list[tuple[str, str, str]],
    records_from: int = 1,
    max_records: int = MAX_RECORDS,
    type_of_data: str = "Spending",
    retries: int = 5,
    timeout: int = 180,
) -> Page:
    """Fetch one page, retrying transient network and 5xx failures.

    An API-level ``failure`` is not retried: the request itself is malformed, so
    repeating it just hammers a public endpoint to no purpose.
    """
    body = _build_request(
        type_of_data=type_of_data,
        criteria=criteria,
        records_from=records_from,
        max_records=max_records,
    )

    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            raw = _post(body, timeout)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            time.sleep(2**attempt)
            continue

        if not raw.strip():
            # Empty body means a malformed request; retrying cannot help.
            raise CheckbookError(
                f"empty response at records_from={records_from} "
                "(check the request body for an XML prolog)"
            )

        root = ET.fromstring(raw)
        if (root.findtext("status/result") or "").strip() != "success":
            messages = "; ".join(d.text or "" for d in root.iter("description"))
            raise CheckbookError(f"API failure at records_from={records_from}: {messages}")

        count_text = root.findtext("result_records/record_count")
        # Read whatever children each transaction actually has, so the same code
        # serves both the Spending and Contracts feeds.
        records = [
            {child.tag: (child.text or "").strip() for child in txn}
            for txn in root.iter("transaction")
        ]
        return Page(records=records, record_count=int(count_text or 0))

    raise CheckbookError(
        f"network failure at records_from={records_from} after {retries} attempts: {last_error}"
    )


def record_count(criteria: list[tuple[str, str, str]], **kwargs) -> int:
    """Total rows matching ``criteria``, fetched with a single throwaway record."""
    return fetch_page(criteria=criteria, records_from=1, max_records=1, **kwargs).record_count


def doe_year_criteria(fiscal_year: int) -> list[tuple[str, str, str]]:
    return [
        ("agency_code", "value", DOE_AGENCY_CODE),
        ("fiscal_year", "value", str(fiscal_year)),
    ]
