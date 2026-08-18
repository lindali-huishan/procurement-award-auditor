# Handoff — NYC DOE Procurement Auditor

Everything needed to run, read, and change this tool. Start here, then read
`README.md` for the analysis reasoning behind it.

## Open it right now

Double-click **`site/index.html`**. That is the whole tool — five pages, all
data embedded, no server, no install, no network.

If your browser blocks local fonts (Firefox is strict about this), run:

```bash
python3 serve.py
```

## The five pages

| Page | File | What it is |
|---|---|---|
| **Score a contract** | `site/index.html` | Set amount, term, registration lag, industry, contract type, M/WBE in the left rail; the likelihood of a no-bid award is the page. Scores in your browser. |
| **Review queue** | `site/review.html` | The case list. Opens on the 25 no-bid awards that are unusual for their own industry, deepest in the tail first, with four corroborating checks and the evidence in each expandable row. |

Methodology lives in `README.md`, not in the tool. The **Method & limits** page was
removed: an auditor opening this wants a reading and a worklist, not a paper.

### Why three pages became one

*Flagged*, *Lobbying*, and *Overlap* were three answers to the same question, and
*Overlap* existed only to show the intersection of the other two. A reader who
found a contract on one page had to know to re-find the vendor on another. The
lobbying page also exposed the matcher's internal state as a `match` column
(`exact`, `unmatched`) to a reader who wanted to know about a contract, not about
string comparison — that column is gone, and whether a name matched is now a
precondition for showing lobbying evidence rather than something to interpret.

The queue ranks by **how many independent checks fired, then by dollars**, not by
model score. A contract tripping three checks is a better use of an afternoon
than one scoring marginally lower on a single model.

## Changing how it looks

The visual layer is two plain files. Edit either, re-run the build, done.

- **`web/style.css`** — design tokens and every component. Ported from the
  Marathon Trainer dashboard, so the two tools share a look: Inter, the blue
  system (`--blue: #2563EB`), 16px cards, the sidebar shell.
- **`web/app.js`** — all page logic, keyed off `<body data-page>`.

```bash
python3 src/build_site.py      # copies web/ into site/, re-renders the 5 pages
```

The HTML shells live in `src/build_site.py`. Nav items are the `NAV` list at
the top; icons are the `ICONS` dict next to it.

## Changing the analysis

```bash
pip install -r requirements.txt
python3 -m pytest tests/ -q     # 81 tests, no data needed
```

Then re-run whichever step you touched, in order:

```bash
python3 src/build_panel.py      # 21.8M rows -> vendor-year panel      [needs data/raw/]
python3 src/build_features.py   # leakage-free features + labels
python3 src/analyze.py          # the five findings                     [needs data/raw/]
python3 src/peer_anomaly.py     # peer-relative outlier detection
python3 src/build_lobbying.py   # lobbying join + convergence flags
python3 src/export_webapp.py    # train award model -> webapp_data.json
python3 src/build_site.py       # render site/
```

The model itself lives in three files, split from what used to be one function
inside `export_webapp.py`:

| File | What it owns |
|---|---|
| `src/award_taxonomy.py` | The five-tier award-method taxonomy. Single source of truth — change a tier here, not in the model. |
| `src/award_model.py` | Features, rolling-origin validation, the fit. |
| `src/review_queue.py` | The unified case list and its flag thresholds. |

Everything except the two marked steps runs off data already in this bundle
and finishes in seconds.

## What is and is not in this bundle

**Included** (~31 MB): all source, all tests, the web layer, the built site, and
the derived data everything downstream reads —

```
data/panel/            24 MB   vendor-year panel, model tables, scored vendors,
                               peer anomalies, lobbying panels
data/raw_contracts/   1.7 MB   58,592 registered contracts
data/raw_lobbying/    5.4 MB   79,678 eLobbyist filings
data/lobbying_match_review.csv  149 name pairs awaiting human confirmation
webapp_data.json      0.7 MB   the bundle the site reads
```

**Not included**: `data/raw/` — the 21.8M spending transactions, **321 MB**.
It is only read by `build_panel.py` and `analyze.py`, and it is fully
regenerable:

```bash
python3 src/pull_spending.py    # ~1,090 API calls, resumable, ~35 min
python3 src/pull_spending.py --verify
```

Everything else in the pipeline works without it. This matches the project's
own convention — the raw pull has never been committed.

To refresh the lobbying feed (2 API calls, about five seconds):

```bash
python3 src/pull_lobbying.py
```

## The one thing to be careful about

`data/lobbying_match_review.csv` holds 149 name pairs that were close but not
close enough to publish. **Do not bulk-accept them.** Matching a lobbying client
to a Checkbook payee is a claim about a named firm, and containment-based
matching was wrong roughly half the time in testing — `Red Hat, Inc.` is
contained in `RED HAT DAY CARE CENTER`. The matcher auto-accepts equivalence
relations only (exact, abbreviation, spelling, possessive, plural, truncation);
everything else is in that file precisely because a human needs to look.

If you confirm pairs by hand, add them to `ALIASES` in `src/lobbying.py` rather
than loosening `ACCEPT_METHODS`. `tests/test_lobbying.py` has the known-bad
pairs as regression tests — keep them passing.

## The rule that keeps the model honest

**Nothing downstream of the award decision goes in the fit.** Registration lag
is the worked example: it predicts well (+0.015 AUC) but it happens after the
award, so training on it taught the model that "no-bid and registered a year
late" is normal — and that hid every one of those contracts from the anomaly
list. It is computed in `build_features`, flagged on the review queue, and
excluded from `FEATURES`. `tests/test_award_model.py` guards both halves of that,
plus a general check that no feature name contains `retro`, `registered_after`,
`spent_to_date`, or `current_amount`.

If you add a feature, ask when its value becomes knowable. At award time, it can
go in the fit. Afterwards, it is a flag.

## Why the model stops at $25,000

`SMALL_PURCHASE_CAP` in `src/award_taxonomy.py` is load-bearing even though the
small-purchase *analysis* was removed. `SMALL PURCHASE - WRITTEN` is 74% of the
contract file and hard-capped: 99.8% of those orders are priced at exactly
$25,000. Let them into the model and `log_amount` alone reaches 0.982 AUC by
recognising the ceiling rather than any conduct. `award_model.model_frame` scopes
above the cap for that reason, and `tests/test_award_model.py` guards it. If you
widen the scope, check `log_amount`-alone AUC first — if it jumps, you have
rediscovered the ceiling.

## Data sources

- **Checkbook NYC** — `https://www.checkbooknyc.com/api`, no key required.
  Quirks documented in `src/checkbook.py` and `README.md`.
- **NYC City Clerk eLobbyist** — NYC Open Data Socrata dataset `fmf3-knd8`,
  no key required. Quirks documented in `src/lobbying.py`.

Both are open records. The tool makes no outbound request at runtime.
