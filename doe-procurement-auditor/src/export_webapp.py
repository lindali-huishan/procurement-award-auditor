"""Fit the model, build the review queue, write the bundle the site reads.

Deliberately small. Everything here ends up on one of two pages, and anything
that does not is not computed:

* **Competitive-intensity model** -- shipped as logistic-regression coefficients
  over standardised features so the browser scores live input with a dot product
  and a sigmoid. No runtime, no network, and an auditor can recompute it by hand.
  See ``award_model`` for what is in the fit and, more importantly, what was
  measured and deliberately left out.

* **Review queue** -- the case list, ranked by how unusual each no-bid award is
  for its own industry. See ``review_queue``.

Three things this used to export and no longer does, because the pages that read
them were merged away: the 1,200-vendor entrenchment index, the lobbying
leaderboard, and the award shortlist. Between them they were most of the payload
and nothing consumed them. The lobbying *data* is still used -- the queue reads
``lobbying_vendors`` and ``lobbying_convergence`` straight from the panel, which
is cheaper than routing them through JSON. Those panel files are produced by
``build_lobbying.py`` and ``build_panel.py``; this script only ever reads.

    python3 src/export_webapp.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import award_model  # noqa: E402
import review_queue  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PANEL_DIR = ROOT / "data" / "panel"
CONTRACTS = ROOT / "data" / "raw_contracts" / "contracts_registered.parquet"
OUT = ROOT / "webapp_data.json"

# Only the fields the two pages actually read, as an explicit allowlist rather
# than shipping the whole model dict. That dict also carries diagnostics and
# rejected-idea records, which belong in the console and the README -- a payload
# that grows by accident is how the previous version reached 0.91 MB with two
# thirds of it dead.
MODEL_FIELDS = (
    "features", "structural", "means", "stds", "coefficients", "intercept",
    "feature_labels", "industry_labels", "contract_type_labels",
    "industry_reference", "type_reference", "industry_medians",
    "industry_rates", "type_rates", "n_labeled",
)


def main() -> int:
    if not CONTRACTS.exists():
        print(f"missing {CONTRACTS}\nrun: python3 src/pull_contracts.py")
        return 1

    print("Fitting competitive-intensity model...")
    contracts = award_model.load_contracts(CONTRACTS)
    scored: list = []
    model = award_model.train(contracts, scored_out=scored)

    print("Building review queue...")
    queue = review_queue.build(scored[0], model["industry_rates"], PANEL_DIR)
    print(f"  {queue['anomalies']} unusual for their industry, of "
          f"{queue['discretionary_total']:,} discretionary awards; "
          f"{queue['total']:,} cases carry at least one check")
    if not queue["lobbying_available"]:
        print("  no lobbying feed -- run src/pull_lobbying.py then "
              "src/build_lobbying.py to enable that check")

    payload = {
        "meta": {
            # Contract counts, not payment counts. An earlier version put
            # "21.8M payments analysed" in the sidebar; that described the
            # spending feed, which this tool never touches, and sitting beside
            # the model's own figures it read as the tool's coverage.
            "contracts_total": int(len(contracts)),
            "years": [int(contracts.fy.min()), int(contracts.fy.max())],
        },
        "award_model": {key: model[key] for key in MODEL_FIELDS},
        "review_queue": queue,
    }

    OUT.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"\nwrote {OUT.name}  {OUT.stat().st_size/1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
