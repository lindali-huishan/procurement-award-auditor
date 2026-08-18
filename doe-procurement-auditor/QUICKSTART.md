# Quickstart

Two pages, no server required, no API keys, no network at runtime.

## Just look at it

Open **`site/index.html`** by double-clicking. That is the whole tool.

If your browser blocks local fonts (Firefox is strict about this):

```bash
python3 serve.py
```

## Rebuild it from the data in this bundle

```bash
pip install -r requirements.txt
python3 -m pytest tests/ -q        # 123 tests, ~2 s, needs no data
python3 src/export_webapp.py       # fit the model, build the queue  (~3 s)
python3 src/build_site.py          # render site/                    (~1 s)
```

That is the whole loop. Everything those two steps need is in `data/`.

## Where things live

| Path | What it is |
|---|---|
| `site/` | The built tool — open `index.html` |
| `web/style.css`, `web/app.js` | The visual layer. Edit these, re-run `build_site.py` |
| `src/award_taxonomy.py` | The five award tiers. Change a tier here, nowhere else |
| `src/award_model.py` | Features, validation, the fit |
| `src/review_queue.py` | The case list and its flag thresholds |
| `src/export_webapp.py` | Fit → queue → `webapp_data.json` |
| `src/build_site.py` | HTML shells for both pages |
| `tests/` | 123 tests, no data required |
| `briefing/` | The presenter's briefing, PDF and source |

Read **`README.md`** for the analysis reasoning and every result that was tested
and rejected. Read **`HANDOFF.md`** for how to change things safely.

## The two rules worth knowing before you edit

**1. Nothing downstream of the award decision goes into the model.**
Registration lag is the worked example. It predicts well (+0.015 AUC) but it
happens *after* the award, so training on it taught the model that "no-bid and
registered a year late" is normal — and that hid every one of those contracts
from the priority list. It is computed, flagged on the queue, and kept out of
`FEATURES`. `tests/test_award_model.py` guards this.

If you add a feature, ask when its value becomes knowable. At award time it can
go in the fit. Afterwards it is a flag.

**2. The model stops at $25,000 for a reason.**
`SMALL PURCHASE - WRITTEN` is 74% of the contract file and hard-capped: 99.8% of
those orders are priced at exactly $25,000. Let them into the model and
`log_amount` alone reaches 0.982 AUC by recognising the ceiling rather than any
conduct. If you widen the scope, check `log_amount`-alone AUC first — if it
jumps, you have rediscovered the price cap.

## What is not in this bundle

**`data/raw/`** — the 21.8M spending transactions, 321 MB. Only the deeper
analysis scripts (`analyze.py`, `build_panel.py`) read it, and it is fully
regenerable in about 35 minutes:

```bash
python3 src/pull_spending.py       # ~1,090 API calls, resumable
python3 src/pull_spending.py --verify
```

Four large derived panel files were also left out to keep this small
(`model_table`, `peer_anomalies`, `vendor_seasonality`, `vendor_map`,
`scored_vendors`). Nothing the two pages do needs them, and they rebuild from
`data/panel/vendor_year.parquet`, which *is* included:

```bash
python3 src/build_features.py
python3 src/peer_anomaly.py
```

To refresh the lobbying feed (2 API calls, about five seconds):

```bash
python3 src/pull_lobbying.py
python3 src/build_lobbying.py
```

## One thing to be careful about

`data/lobbying_match_review.csv` holds 149 name pairs that were close but not
close enough to publish. **Do not bulk-accept them.** Matching a lobbying client
to a payee is a claim about a named firm, and containment-based matching was
wrong roughly half the time in testing — `Red Hat, Inc.` is contained in
`RED HAT DAY CARE CENTER`. If you confirm pairs by hand, add them to `ALIASES`
in `src/lobbying.py` rather than loosening `ACCEPT_METHODS`.
