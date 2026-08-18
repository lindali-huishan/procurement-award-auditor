# NYC DOE Spending Accountability

A spend-accountability pipeline for the New York City Department of Education,
built on Checkbook NYC's Spending and Contracts APIs.

The question is not *what will DOE spend next year* — the city's OMB and IBO
already forecast that professionally. It is **who does the money go to, how
fast, and are the rules being followed**, which nobody publishes for DOE.

## Data

| Feed | Rows | Grain |
|---|---|---|
| Spending (FY2010–FY2026) | 21,775,132 | one payment transaction |
| Contracts, registered | 58,592 rows / 58,531 contracts | one contract-vendor-year |
| Contracts, pending | 102 | live snapshot |
| Lobbying filings (RY2021–RY2026) | 79,678 | one lobbyist-client-period filing |

Spending and contracts are pulled from `https://www.checkbooknyc.com/api` and
stored as zstd Parquet — about **330 MB**, versus 6–8 GB as CSV. Lobbying comes
from NYC Open Data (Socrata dataset `fmf3-knd8`) and adds about 6 MB.

### API quirks worth knowing

These cost real time to discover, and are all handled in `src/checkbook.py`:

- **No XML prolog.** If the request body starts with `<?xml ...?>`, the API
  returns HTTP 200 with a **zero-byte body** and no error. This is the single
  most confusing failure mode in the whole feed.
- **`max_records` caps at 20,000**, not the documented 1,000. That turns a
  ~22,000-call pull into ~1,090 calls.
- **`records_from` has no depth limit.** Offsets past 1.5M page fine, so simple
  per-year offset paging works.
- **Ordering is deterministic** — verified: adjacent pages have zero overlap and
  refetching a deep page returns identical rows.
- **The Contracts feed requires `status` and `category`.** Pending contracts
  reject `fiscal_year` entirely; they are a live snapshot.
- **Transactions have no unique id**, and byte-identical rows legitimately
  exist. Never dedupe on content.
- **`vendor_record_type` mixes two grains.** 60 of the 58,592 contract rows are
  `Sub Vendor` — subcontractor *disclosures* hanging off a parent contract that
  is already present as its own row, carrying
  `prime_contract_original_amount = 0`. Filter to `Prime Vendor` or a vendor
  groupby counts one contract several times.
- **`SMALL PURCHASE - WRITTEN` is hard-capped at $25,000** and is 74% of the
  contract file. Its maximum *and* median are both exactly the cap, so the
  amount field is a near-perfect proxy for the award code. See below.

And two in the eLobbyist feed, handled in `src/lobbying.py`:

- **Targets and activities are denormalised free text**, semicolon-joined, with
  heavy repetition — one observed cell held the same council member fourteen
  times across ten kilobytes. Both need exploding and deduplicating before any
  count means anything.
- **The agency/official separator is a double space**, so the usual
  `" ".join(text.split())` tidy-up silently merges the two fields.

## Pipeline

### Getting started

No data ships with this repo -- it is ~320 MB and fully regenerable. Step 1
rebuilds it from the public API in about 35 minutes.

```bash
pip install -r requirements.txt
python3 -m pytest tests/ -q          # 81 tests, no data needed

python3 src/pull_spending.py         # 1. ~1,090 API calls, resumable, ~35 min
python3 src/pull_contracts.py        # 2. contracts feed, ~1 min
python3 src/pull_lobbying.py         # 3. eLobbyist feed, 2 calls, ~5 s
python3 src/build_panel.py           # 4. 21.8M rows -> vendor-year panel, ~9 s
python3 src/build_features.py        # 5. leakage-free features + labels
python3 src/analyze.py               # 6. the five findings
python3 src/peer_anomaly.py          # 7. peer-relative outlier detection
python3 src/build_lobbying.py        # 8. lobbying join + convergence flags
python3 src/export_webapp.py         # 9. train award model -> webapp_data.json
python3 src/build_site.py            # 10. render site/
```

Steps 4-10 take under a minute once the data is local. `pull_spending.py
--verify` and `pull_lobbying.py --verify` reconcile rows on disk against each
API's own record counts, and rerunning the spending puller fetches only missing
pages. Steps 8-10 degrade gracefully: without the lobbying feed the tool builds
exactly as it did before, minus the convergence section.

Everything is standard library plus the five packages in `requirements.txt`;
there is no framework, no service, and no API key -- Checkbook NYC is open.

The giant Parquet set is read only by `build_panel.py`. Everything downstream
works off the panel, which fits in memory comfortably.

`pull_spending.py` writes one Parquet file per page, so it is resumable: rerun
it and it fetches only the gaps. `--verify` reconciles rows on disk against the
API's own record counts.

## Four decisions that determine whether the results mean anything

**1. Vendor name resolution.** The same firm appears as `ACME CORP`,
`Acme Corp.`, and `ACME CORPORATION`. Left alone, every concentration number
comes out too low. But the fuzzy matching that fixes it also wants to merge
`MANHATTAN CHARTER SCHOOL` with `MANHATTAN CHARTER SCHOOL II` — *separate legal
entities* in a charter network. Merging those would manufacture concentration
that does not exist, which is precisely the thing being measured. So
`src/resolve.py` applies an **ordinal guard**: differing trailing numerals or
roman numerals block a merge, unconditionally, before any similarity test runs.

**2. Who counts as a vendor.** Two payee classes have to come out first:

- `N/A (PRIVACY/SECURITY)` is a **redacted** payee spanning hundreds of
  thousands of rows. It is not a firm. Left in, it ranks near the top.
- The School Construction Authority, NYC Transit, retiree health trusts, and
  union welfare funds receive some of the largest transfers in the data — but
  those are **intergovernmental transfers, not procurement**.

Both are flagged in `src/normalize.py` rather than silently dropped, so the
analysis can include or exclude them explicitly and say which it did.

**3. Temporal validation.** Features come only from years ≤ T; labels come from
T+1…T+5. Train and test are separated by an **embargo** of `HORIZON` years so no
training row's label window overlaps a test row's. Run `python3 src/model.py
--demo-leakage` to see what a random split would have claimed instead — the gap
is the leak, not skill.

**4. What may match across systems.** Joining a lobbying client to a Checkbook
payee is a different risk from matching two Checkbook payees. A bad merge inside
Checkbook miscounts a statistic; a bad merge here asserts that a *named firm*
lobbied the agency that then handed it a no-bid contract. So `src/lobbying.py`
auto-accepts **equivalence relations only** — exact, abbreviation
(`INTL`→`INTERNATIONAL`), spelling (`CENTRE`→`CENTER`), possessive
(`CHILDREN'S`→`CHILDRENS`), plural drift, and field truncation
(`HEART SHARE`→`HEARTSHARE`).

Substring containment never auto-accepts, and that rule was forced by the data
rather than chosen in advance. Containment scored well and was wrong about half
the time: `Red Hat, Inc.` is contained in `RED HAT DAY CARE CENTER`, and
`Verizon Corporate Resources Group` in `CORPORATE RESOURCE DEVELOPMENT`. Both
would have published a false accusation. Those pairs, and every ambiguous
name where a runner-up scored within 0.02, go to `data/lobbying_match_review.csv`
for human confirmation instead. Currently 195 names match automatically and 149
sit in the queue.

**5. What counts as a competitive award, and what the model may see.** The
first version of this analysis asked one binary question — competitive or not —
and could only answer it for **8,232 of 58,523** contracts. The other 86% were
dropped for having an award method that fit neither side, and a single dropped
code, `SMALL PURCHASE - WRITTEN`, is **74% of every DOE contract**. A tool
silent on three quarters of the file is not describing DOE procurement.

`src/award_taxonomy.py` replaces the binary with five tiers of competitive
intensity: `formal`, `informal`, `discretionary`, `statutory`, and `renewal`.
The last two are scored out on purpose — DOE has no discretion over a legal
mandate or an intergovernmental purchase, and renewing an option is not a new
award decision — but they are *reported*, not silently dropped.

The small-purchase cap then forces a second decision. Because 99.8% of at-cap
orders are priced at exactly $25,000, any model handed both the amount and those
rows learns "is this $25,000" and reports **0.982 AUC on `log_amount` alone**.
So the model is scoped **above** the cap. The at-cap population is excluded
rather than modelled: it cannot be scored without the amount giving the answer
away, and it cannot be judged by counting either, because the feed names the
agency rather than the individual school — so it cannot distinguish forty schools
each buying once from one office buying forty times. That is a data limit, not a
modelling choice, and it means the tool is silent on 74% of the contract file.

## What the duration feature was doing

Worth stating plainly, because it invalidated the original headline. The first
model reported 0.964 AUC. **`duration_years` alone scored 0.963 of it.**

NYC procurement rules effectively co-determine term length and award method.
The 5th percentile term for a PQVL-based RFP is 1,094 days; the *maximum* for an
emergency award is 729. A single threshold — term under 400 days — classifies at
87.2% accuracy. The feature was not predicting the label so much as restating
it.

Two further problems compounded it. Re-run across rolling origins the old
specification swings from **0.685 to 0.961** (sd 0.076), so the single published
split sat near the top of its own range. And the true relationship is a
staircase, not a ramp: the non-formal rate is 91.2% at or under 400 days, 40.0%
from 400–730, 35.8% from 730–1,460, and 12.1% above. A lone log term smeared
that into a slope and answered ~21% for a 405-day contract whose real rate is
~40%.

The current model enters term length as a log plus three band flags, adds
signals that are *not* downstream of the award method — chiefly **retroactive
registration**, since 24% of DOE contracts are registered after work began and
2.2% after the contract ended — and publishes the rolling spread:

| specification | mean AUC | sd |
|---|---|---|
| old shipped 7 features | 0.885 | 0.076 |
| term length alone | 0.853 | 0.042 |
| this model, no term length | 0.847 | — |
| **this model** | **0.940** | **0.013** |
| gradient booster, same features | 0.930 | 0.052 |

The booster no longer wins, and it doubles the spread — so shipping a
transparent logistic is not a concession here. Term length still does real work
(0.940 → 0.847 without it), which the tool states on screen rather than burying.

## The anomaly rule

The tool's primary signal is **"unusual for its industry"**, and it is deliberately
rare: 25 cases out of 2,387 discretionary awards.

A discretionary award is flagged when the model scores it in the most unusual **1%**
of *its own industry's* no-bid awards. Percentile-within-industry rather than an
absolute cut, because the per-industry thresholds land nowhere near each other:

| Industry | Flagged below |
|---|---|
| Goods | p < 0.277 |
| Standardized services | p < 0.100 |
| Human services | p < 0.077 |
| Professional services | p < 0.006 |

That is a ~50× spread. One absolute cut-off would have flagged whole industries and
missed others entirely.

The rule is also strict on purpose. Among discretionary awards the model's median
reading is **0.94** — most no-bid awards look exactly like no-bid awards, which is
the expected result, not a finding. An earlier version used "half the industry base
rate" and returned 84 cases; that is too many to call outliers when there are only
2,387 discretionary awards in scope at all.

Read plainly, a flag says: *this award looks less like a no-bid award than 99% of
DOE's actual no-bid awards in the same line of work.*

## Four things tested and rejected

Published because someone would otherwise re-derive them.

**Registration lag must not be a predictor — only a flag.** This one cost real
accuracy to get right. Feeding the lag into the model lifts rolling AUC from
0.921 to 0.936, and it was in the fit until the consequence was measured.
Registration happens *after* the award decision, so training on it teaches the
model that "no-bid award, registered very late" is a normal no-bid signature. The
847 discretionary awards registered a year or more late then score p=0.899 on
average, against p=0.772 with the feature removed — and **zero of them reach the
anomaly list**, versus 8 of 26 once it is out. The model was absorbing a red flag
into its definition of normal and hiding precisely the contracts the tool exists
to surface. Lower AUC is the better tool here, and the spread improves too
(0.020 → 0.014). **General rule: anything downstream of the award decision
belongs in the flags, never in the fit.**

**M/WBE certification carries no signal.** Removing it *improved* rolling AUC
from 0.936 to 0.938 and its coefficient was the smallest in the model. Worth
recording as a substantive result rather than only a simplification: whether a
vendor is minority- or women-owned tells you nothing detectable about whether DOE
competed the award. (M/WBE share against the city's 30% utilisation goal is still
a separate finding in `analyze.py`; it is just not a predictor here.)

**Award timing carries nothing.** Calendar month of start, fiscal quarter, days
to the 30 June year end, a July-start flag, and an April–June registration flag
were all tested. Best single feature: 0.621 AUC with sd 0.168 — barely above
chance and wildly unstable. Adding the whole family moved the model by −0.001.
The seasonality is real (5,093 in-scope contracts start on 1 July and run 42.6%
non-formal, against 23.8% for April–June starts) but term length and contract
type already carry it. Note this is *award* timing; the year-end **spending**
spike in `analyze.py` is a separate finding from the payments feed.

**Vendor history carries nothing an auditor can look up.** The version that
worked needed the vendor's own prior rate of non-competitive awards — worth 0.007
AUC, but a statistic the user would have to compute. Every knowable substitute
(prior-contract count, its log, first-time-vendor flag, established-vendor flag)
lands at 0.936 ± 0.020, identical to using no vendor history at all; the gradient
is simply too shallow, 32.8% non-formal for first-time vendors rising to 44.7%
past 35 contracts. So the model uses none, and the scorer asks nothing about it.

**Retroactive registration is not a rare event.** Within the model's scope
**92.6%** of contracts are registered after their start date — late registration
is how DOE operates, not an anomaly, and an earlier version of the review queue
that flagged any lateness returned 8,213 of 9,694 contracts and was useless. What
makes the lag useful is its gradient: 17.4% non-formal at zero days, 24.9% under
90, 31.0% to 180, 48.3% to a year, 76.4% beyond. So it is a model feature at
every magnitude and a *flag* only at its extreme.

## Lobbying convergence

NYC's lobbying law counts attempts to influence **the determination of a
procurement contract**, and filings name the target agency and the individual
official. That is unusual — most jurisdictions cover only legislative and
rulemaking lobbying, which makes their lobbying data useless for procurement
work — and it is what makes this join worth doing at all.

`src/build_lobbying.py` flags non-competitive contracts whose vendor also filed
procurement lobbying against DOE in the award year or the one before. Two things
keep the flag meaningful:

- **Industry-graded, not absolute.** Non-competitive rates differ enormously by
  line of work: Professional Services runs 8.8%, Human Services 30.9%. Flagging
  raw "was sole-sourced" would return the entire Human Services book. Ranking is
  by how unusual the award is *for its own industry*.
- **Timing has to point the right way.** Lobbying after an award cannot have
  influenced it, so only filings in the award year or the year prior count.

**Coverage is the binding limit.** The eLobbyist feed begins at report year
2021; spending runs from FY2010. Convergence is computable for roughly the last
six fiscal years only, and the tool says so rather than comparing a 2021–2026
count against a seventeen-year denominator.

### What a convergence flag is not

Lobbying is lawful, disclosed, and common, and firms with large city contracts
retain lobbyists *because* they are large — the arrow plausibly runs that way.
The sums are also small against the contracts: a few thousand dollars of
reported compensation next to a multi-million-dollar award proves nothing, and
reported compensation covers everything in a filing, not the DOE portion. What
a flag buys is a **named official, a date range, and a stated subject** attached
to a contract that already looked unusual for its industry. That is a file to
request, not a finding.

## Findings

1. **Vendor concentration** — top-N share and HHI over 16 years.
2. **M/WBE share** — measured against the city's 30% utilization goal.
3. **Payment timing** — *with an important caveat, below.*
4. **Year-end spending spike** — the March–June run-up to the 30 June fiscal
   year end.
5. **Payroll → contracts shift** — the structural change, computable with no
   model at all.

### The caveat on payment timing

The Spending feed carries only `issue_date`: the date the check went out. There
is **no invoice or receipt date anywhere in this data**. The elapsed time
between a vendor billing DOE and DOE paying it — the actual prompt-payment
metric, and the thing nonprofits take bridge loans against — is therefore **not
computable from this source**. `analyze.py` reports contract-registration to
first-payment instead, and labels it as the different thing it is. Closing this
properly needs an invoice-level source the Checkbook API does not expose.

## Interpreting the entrenchment score

Entrenchment is **not** wrongdoing. Vendors persist for legitimate reasons:
switching costs are real, some services have only two capable providers in the
region, and incumbents are sometimes simply good. The score measures **lock-in
and its risk factors** — tenure, share trajectory, renewal dependence,
non-competitive award mix — not misconduct. Any writeup should say so.
