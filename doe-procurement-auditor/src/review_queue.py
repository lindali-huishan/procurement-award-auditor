"""One case list, replacing the Flagged / Lobbying / Overlap pages.

**Why they merged.** Three pages was three answers to the same question. A
reader who found a contract on *Flagged* had to know to re-find the vendor on
*Lobbying*, and *Overlap* existed only to show the intersection of the other
two. Worse, the lobbying page's ``match`` column exposed the *matcher's*
internal state -- ``exact``, ``unmatched`` -- to a reader who wanted to know
about a contract, not about string comparison. That column is gone. Whether a
name matched is now a precondition for showing lobbying evidence at all, not a
column to interpret.

**What replaced the probability.** The model score is one input, not the
ranking. An oversight office does not act on "0.03" -- it acts on a named
reason, a dollar figure, and a document it can request. So each case carries
**flags**, each of which is either a rule with a citation or a peer comparison
with a named comparison set:

``unusual_for_industry`` -- **the primary signal; the queue is ranked by it**
    A discretionary award that the model scores in the most anomalous
    ``ANOMALY_PERCENTILE`` of *all discretionary awards in its own industry*.

    Percentile-within-industry rather than an absolute cut, for two reasons.
    Industries are not comparable -- Professional Services runs 10.3%
    non-competitive and Goods 56.7% -- so one threshold would flag whole
    industries and miss others entirely. And the score distribution among
    discretionary awards is heavily skewed high (median 0.937): most no-bid
    awards look exactly like no-bid awards, which is the expected result and not
    a finding. What matters is the thin tail, and a percentile finds it by
    construction in every industry rather than only where the base rate happens
    to sit low.

    Read plainly, a flag says: *this award looks less like a no-bid award than
    99% of DOE's actual no-bid awards in the same line of work.*
``lobbied``
    The vendor filed procurement lobbying against DOE in the award year or the
    one before. Carries the officials named in the filing -- the single most
    actionable field in the whole tool, because it converts a statistical
    oddity into a specific document request.
``retroactive`` / ``registered_after_expiry``
    Registered unusually long after work began, or after the contract had
    already ended.

    The threshold matters more than it looks. Within this scope **92.6% of
    contracts are registered after their start date** -- late registration is
    how DOE routinely operates, not an exception, so a flag at "more than zero
    days" fires on nearly every row and is worthless for triage. What *is*
    unusual is the tail: the bar here is a full year, which catches 13.8% of the
    scope. The gradient behind it is steep and monotonic -- 17.4% non-formal at
    zero days, 24.9% under 90, 31.0% to 180, 48.3% to a year, and 76.4% beyond
    it -- which is why the lag earns its place as a model *feature* at every
    magnitude while only its extreme earns a flag.
``long_no_bid``
    Discretionary award running four years or more. Long terms are how a
    single non-competitive decision compounds.
Ranking is by flag count, then dollars. A contract that trips three
independent checks is a better use of an auditor's afternoon than one that
scores marginally lower on a single model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from award_model import CONTRACT_TYPE_LABELS, INDUSTRY_LABELS  # noqa: E402
from normalize import normalize_vendor  # noqa: E402

# How deep into its own industry's tail a discretionary award must sit to count
# as an anomaly. 0.01 = the most unusual 1%. Deliberately strict: an earlier
# version used a "half the industry base rate" cut and returned 84 cases, which
# is too many to be describable as outliers when there are only 2,387
# discretionary awards in scope at all.
ANOMALY_PERCENTILE = 0.01

LONG_TERM_DAYS = 1460   # four years

# Registration lag that counts as a flag rather than as business as usual. See
# the module docstring: 92.6% of in-scope contracts are registered late, so the
# bar is the tail (13.8% of the scope), not the mere fact of lateness.
RETRO_FLAG_DAYS = 365

FLAG_LABELS = {
    "unusual_for_industry": "Unusual for its industry",
    "lobbied": "Lobbied DOE",
    "retroactive": "Registered a year+ late",
    "registered_after_expiry": "Registered after it expired",
    "long_no_bid": "Long no-bid term",
}

# Ordered worst-first for display. Lobbying outranks the model score because it
# brings an independent public record rather than another view of the same one.
FLAG_ORDER = ["lobbied", "registered_after_expiry", "unusual_for_industry",
              "retroactive", "long_no_bid"]


def _lobbying_lookup(panel_dir: Path) -> dict:
    """Vendor-key -> lobbying profile, or ``{}`` when the feed is absent."""
    path = panel_dir / "lobbying_vendors.parquet"
    if not path.exists():
        return {}
    vendors = pd.read_parquet(path)
    return {
        str(r.vendor_key): {
            "filings": int(r.lob_filings),
            "years": [int(r.lob_first_year), int(r.lob_last_year)],
            "compensation": float(r.lob_compensation),
            "lobbyists": int(r.lob_lobbyists),
            "confidence": round(float(r.lob_confidence), 3),
            "officials": "" if pd.isna(r.lob_officials) else str(r.lob_officials),
            "clients": "" if pd.isna(r.lob_clients) else str(r.lob_clients),
        }
        for r in vendors.itertuples()
    }


def _convergence_pairs(panel_dir: Path) -> set:
    """(vendor_key, fiscal_year) pairs where lobbying timing points the right way.

    Only filings in the award year or the one before count -- lobbying after an
    award cannot have influenced it. That timing rule lives in
    ``build_lobbying.py``; this just reads its output.
    """
    path = panel_dir / "lobbying_convergence.parquet"
    if not path.exists():
        return set()
    conv = pd.read_parquet(path)
    return set(zip(conv.vendor_key.astype(str), conv.fiscal_year.astype(int)))


def build(scored: pd.DataFrame, industry_rates: list[dict],
          panel_dir: Path, limit: int = 400) -> dict:
    """Assemble the case list from a scored model frame.

    ``scored`` is ``award_model.model_frame`` output with a ``p`` column.
    """
    lobbying = _lobbying_lookup(panel_dir)
    convergence = _convergence_pairs(panel_dir)
    rate_by_industry = {r["key"]: r["rate"] for r in industry_rates}

    # Per-industry anomaly bar, computed over that industry's own discretionary
    # awards. Industries with too few to have a meaningful tail get no bar, so
    # they contribute no anomaly flags rather than a bar fitted to a handful.
    discretionary = scored[scored.tier == "discretionary"]
    bars, ranks = {}, {}
    for key, group in discretionary.groupby(discretionary.industry_key.fillna("")):
        if len(group) < 50:
            continue
        bars[key] = float(group.p.quantile(ANOMALY_PERCENTILE))
        ranks[key] = group.p

    def anomaly_percentile(key, p):
        """Where this award sits in its industry's discretionary distribution."""
        series = ranks.get(key)
        if series is None:
            return None
        return round(float((series < p).mean()) * 100, 2)

    cases = []
    for row in scored.itertuples():
        key = normalize_vendor(str(row.prime_vendor))
        industry_rate = rate_by_industry.get(row.industry_key)
        is_discretionary = row.tier == "discretionary"

        flags, evidence = [], {}

        industry_bar = bars.get(row.industry_key or "")
        anomaly = None
        if is_discretionary and industry_bar is not None and row.p < industry_bar:
            anomaly = anomaly_percentile(row.industry_key or "", row.p)
            flags.append("unusual_for_industry")
            evidence["industry_rate"] = (round(float(industry_rate), 4)
                                        if industry_rate is not None else None)
            evidence["industry_bar"] = round(industry_bar, 4)
            evidence["anomaly_pct"] = anomaly

        profile = lobbying.get(key)
        if profile and (key, int(row.fy)) in convergence:
            flags.append("lobbied")
            evidence["lobbying"] = profile

        if row.registered_after_end:
            flags.append("registered_after_expiry")
        elif row.retro_days >= RETRO_FLAG_DAYS:
            flags.append("retroactive")

        if is_discretionary and pd.notna(row.duration_days) and row.duration_days >= LONG_TERM_DAYS:
            flags.append("long_no_bid")

        if not flags:
            continue

        cases.append({
            "vendor": str(row.prime_vendor)[:64],
            "fy": int(row.fy),
            "amount": float(row.amount),
            "method": str(row.method)[:52],
            "tier": row.tier,
            "industry": INDUSTRY_LABELS.get(row.industry_key, INDUSTRY_LABELS[None]),
            "type": CONTRACT_TYPE_LABELS.get(row.type_key, CONTRACT_TYPE_LABELS[None]),
            "term": int(row.duration_days) if pd.notna(row.duration_days) else None,
            "retro": int(row.retro_days),
            # Display-only context ("this vendor's 47th DOE contract"). Optional:
            # it is counted in ``load_contracts`` over the whole contract file,
            # so a frame assembled some other way may not carry it.
            "prior": int(getattr(row, "vendor_prior_contracts", 0)),
            "p": round(float(row.p), 3),
            "flags": [f for f in FLAG_ORDER if f in flags],
            "anomaly": anomaly,
            "evidence": evidence,
        })

    # Anomalies first, deepest in the tail at the top -- that is the signal the
    # tool exists to surface. Everything else follows, ordered by how many
    # independent checks corroborate it and then by dollars.
    cases.sort(key=lambda c: (
        0 if c["anomaly"] is not None else 1,
        c["anomaly"] if c["anomaly"] is not None else 0,
        -len(c["flags"]),
        -c["amount"],
    ))

    counts = {}
    for case in cases:
        for flag in case["flags"]:
            counts[flag] = counts.get(flag, 0) + 1

    return {
        "cases": cases[:limit],
        "total": len(cases),
        "shown": min(limit, len(cases)),
        "dollars": float(sum(c["amount"] for c in cases)),
        "flag_labels": FLAG_LABELS,
        "flag_order": FLAG_ORDER,
        "flag_counts": counts,
        "multi_flag": int(sum(1 for c in cases if len(c["flags"]) >= 2)),
        "anomaly_percentile": ANOMALY_PERCENTILE,
        "anomalies": int(sum(1 for c in cases if c["anomaly"] is not None)),
        "discretionary_total": int(len(discretionary)),
        "industry_bars": {str(k): round(v, 4) for k, v in bars.items()},
        "long_term_days": LONG_TERM_DAYS,
        "retro_flag_days": RETRO_FLAG_DAYS,
        "lobbying_available": bool(lobbying),
    }
