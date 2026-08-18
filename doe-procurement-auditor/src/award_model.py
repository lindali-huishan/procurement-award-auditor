"""The competitive-intensity model: features, honest validation, and the fit.

**What changed and why.** The previous award classifier reported 0.964 AUC on a
single temporal split. Three things were wrong with that number:

1. **Duration was the whole model.** ``duration_years`` alone scored 0.963 of
   that 0.964. NYC procurement rules effectively co-determine term length and
   award method -- no such thing as a short PQVL award (5th percentile 1,094
   days) or a long emergency one (max 729) -- so the feature restates the label
   rather than predicting it.
2. **The scope excluded the contract file's centre of mass.** Small purchases,
   74% of all contracts, were dropped for having no clean binary label.
3. **One split hid enormous instability.** Re-run across rolling origins, the
   old specification swings from 0.685 to 0.961 (sd 0.076). The single number
   published happened to land near the top of that range.

This module addresses all three. Scope is the *discretion-eligible* tiers above
the small-purchase cap (see ``award_taxonomy``), which both admits the
above-cap informal contracts the old scope threw away and keeps the model away
from the at-cap population where amount trivially identifies the award code.
Features add three signals that are not downstream of the award method, chiefly
retroactive registration. Validation is rolling-origin, and the spread is
published alongside the mean because the spread is the honest part.

Measured on rolling origins (train fy<=T, embargo T+1, test T+2..T+3):

===========================  ========  ======
specification                mean AUC      sd
===========================  ========  ======
old shipped 7 features          0.885   0.076
duration alone                  0.853   0.042
this model, no duration         0.843   0.043
**this model**                  0.929   0.019
gradient booster, same feats    0.930   0.052
===========================  ========  ======

The booster matches the mean and doubles the spread, so the transparent
logistic is not a concession here -- it is the better-behaved model, and it
still ships as coefficients an auditor can recompute by hand.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from award_taxonomy import (  # noqa: E402
    DISCRETION_ELIGIBLE, SMALL_PURCHASE_CAP, tier_of,
)

# Features are ordered so the published coefficient table reads in groups:
# structural terms first, then discretion signals, then vendor history.
FEATURES = [
    # Term length enters as a log plus three band flags, because the true
    # relationship is a staircase, not a ramp: within the model's scope the
    # non-formal rate runs 91.2% at or under 400 days, 40.0% from 400-730,
    # 35.8% from 730-1460, and 12.1% above. A single log term smears that into a
    # gentle slope and answers ~21% for a 405-day contract whose real rate is
    # ~40%. The middle band is not decoration -- adding it lifted rolling AUC
    # from 0.932 to 0.940 and cut the spread from 0.018 to 0.013.
    "log_duration",
    "dur_under_1y",       # <= 400 days
    "dur_1_to_2y",        # 400-730 days, the band a two-flag version got wrong
    "dur_over_4y",        # >= 1460 days
    "log_amount",
    "amount_vs_industry",  # size relative to the industry's median award
    # The feed's real industry taxonomy, not a collapsed proxy. An earlier
    # version folded these into is_goods/is_services/is_construction and lost
    # most of the signal: the true spread runs from Professional Services at
    # 10.3% non-formal to Goods at 56.7%, and DOE has *no* construction
    # category at all -- that flag was dead weight with a zero coefficient.
    # "Not Classified" is the reference category.
    "ind_goods",
    "ind_human_services",
    "ind_professional_services",
    "ind_standardized_services",
    # Contract type, likewise real rather than two hand-picked flags. Spread
    # runs from Consultant at 7.5% to Requirements-Services at 96.7%.
    # "Other" is the reference category.
    "type_programs",
    "type_requirements_services",
    "type_doe_requirements",
    "type_supplies",
    "type_consultant",
    "type_work_labor",
]

# Plain-English industry taxonomy, exactly as the feed classifies contracts.
INDUSTRIES = {
    "Goods": "ind_goods",
    "Human Services": "ind_human_services",
    "Professional Services": "ind_professional_services",
    "Standardized Services": "ind_standardized_services",
}
INDUSTRY_LABELS = {
    "ind_goods": "Goods",
    "ind_human_services": "Human services",
    "ind_professional_services": "Professional services",
    "ind_standardized_services": "Standardized services",
    None: "Not classified",
}

# Contract types, grouped and relabelled in the buyer's language. The feed's own
# codes are cryptic (``DEPT OF ED-REQUIREMENT CONTRACT``) and several are
# vanishingly rare, so near-duplicates are merged and the long tail falls into
# the reference category.
CONTRACT_TYPES = {
    "PROGRAMS": "type_programs",
    "PROGRAMS (NOT TAX LEVY FUNDED)": "type_programs",
    "REQUIREMENTS-SERVICES": "type_requirements_services",
    "DEPT OF ED-REQUIREMENT CONTRACT": "type_doe_requirements",
    "DOITT-REQUIREMENTSCONTRACT(RC)": "type_doe_requirements",
    "REQUIREMENTS-GOODS": "type_doe_requirements",
    "SUPPLIES/MATERIALS/EQUIPMENT": "type_supplies",
    "CONSULTANT": "type_consultant",
    "WORK/LABOR": "type_work_labor",
}
CONTRACT_TYPE_LABELS = {
    "type_programs": "Student programs",
    "type_requirements_services": "Requirements — services",
    "type_doe_requirements": "DOE requirements contract",
    "type_supplies": "Supplies, materials & equipment",
    "type_consultant": "Consultant",
    "type_work_labor": "Work & labour",
    None: "Other / one-off",
}

# Features whose value is fixed by the term length. Reported separately so the
# non-structural contribution can be quoted honestly.
STRUCTURAL = ("log_duration", "dur_under_1y", "dur_1_to_2y", "dur_over_4y")

# Registration lag is computed, flagged on the review queue, and deliberately
# NOT a model feature. It predicts well -- adding it lifts rolling AUC from 0.921
# to 0.936 -- and it was in the model until measurement showed the cost.
#
# The problem is direction. Registration happens *after* the award decision, so
# feeding it in teaches the model that "no-bid award, registered very late" is
# part of a normal no-bid signature. It then scores those contracts as
# unremarkable: the 847 discretionary awards registered a year or more late
# averaged p=0.899 with the feature in, against p=0.772 with it out. The
# consequence is what matters -- with the feature in, **zero** of them reached the
# anomaly list; with it out, 8 of the 26 anomalies are late-registered contracts.
#
# The model was absorbing a red flag into its definition of normal, and hiding
# exactly the contracts the tool exists to surface. Lower AUC is the better tool
# here, and the spread improves too (0.020 -> 0.014). Anything downstream of the
# award decision belongs in the flags, not the fit.
REGISTRATION_FINDING = {
    "auc_with": 0.936, "sd_with": 0.020,
    "auc_without": 0.921, "sd_without": 0.014,
    "late_p_with": 0.899, "late_p_without": 0.772,
    "late_discretionary": 847,
    "anomalies_late_with": 0, "anomalies_late_without": 8,
}

# M/WBE certification was dropped for carrying no signal at all: removing it
# *improved* rolling AUC from 0.936 to 0.938, and its coefficient was the
# smallest in the model. Worth noting as a substantive result rather than only a
# simplification -- whether a vendor is minority- or women-owned tells you
# nothing detectable about whether DOE competed the award.
MWBE_FINDING = {"auc_with": 0.936, "auc_without": 0.938}

# Award timing was tested as a feature family and rejected. Kept here because a
# negative result someone else would otherwise re-derive is worth publishing --
# the same reason the entrenchment classifier's failure is on the method page.
#
# Tested: calendar month of start, fiscal quarter of start, days from start to
# the 30 June fiscal-year end, a July-start flag, and an April-June registration
# flag. Best single feature was days-to-fiscal-year-end at 0.621 AUC with an sd
# of 0.168 -- barely above chance and wildly unstable. Added to the model the
# whole family moved the mean by -0.001.
#
# The seasonality is real but already spoken for: 5,093 of the scope's contracts
# start on 1 July and run 42.6% non-formal against 23.8% for April-June starts.
# Term length and contract type already carry it, so the date adds nothing on
# top. Note this is *award* timing; the year-end **spending** spike is a separate
# finding computed in ``src/analyze.py`` from the payments feed.
DATE_FINDING = {
    "tested": ["start month", "fiscal quarter", "days to fiscal year end",
               "July start", "Apr-Jun registration"],
    "best_single_auc": 0.621,
    "best_single_sd": 0.168,
    "delta_when_added": -0.001,
    "july_start_n": 5093,
    "july_start_rate": 0.426,
    "q4_start_rate": 0.238,
}

# A vendor-history feature was dropped rather than replaced. The version that
# worked -- the vendor's own prior-year rate of non-formal awards -- was worth
# 0.007 AUC and halved the spread, but it required whoever is using the scorer to
# already know a statistic they would have to compute. Every version an auditor
# could actually look up carries nothing: prior-contract count, log of it, a
# first-time-vendor flag and an established-vendor flag all land at 0.936 +/-
# 0.020, identical to using no vendor history at all. The underlying gradient is
# simply too shallow -- 32.8% non-formal for first-time vendors rising to 44.7%
# for those with 35+ prior contracts.
#
# So the scorer asks nothing about vendor history, and the model is honest about
# not using it, rather than shipping a control that moves the needle by nothing.
# ``vendor_prior_contracts`` is still computed for display on the review queue,
# where "this vendor's 47th DOE contract" is useful context for a human even
# though it is useless to the fit.
VENDOR_HISTORY_FINDING = {
    "with_rate_auc": 0.943, "with_rate_sd": 0.010,
    "without_auc": 0.936, "without_sd": 0.020,
    "knowable_variants_auc": 0.936,
}

TRAIN_END = 2019   # final fiscal year used for the shipped fit
EMBARGO = 2020     # never trained on, never tested on
TEST_START = 2021


def load_contracts(path: Path) -> pd.DataFrame:
    """Prime-vendor contract rows with a tier attached.

    Sub-vendor rows are excluded: they are subcontractor *disclosures* hanging
    off a parent contract that is already present as its own row, and they
    carry ``prime_contract_original_amount == 0``. Only 60 exist, but keeping
    them would double-count a vendor's contracts in any groupby.
    """
    frame = pd.read_parquet(path)
    frame = frame[(frame.vendor_record_type == "Prime Vendor")
                  & (frame.prime_contract_original_amount > 0)].copy()
    frame["method"] = frame.prime_contract_award_method.fillna("").str.strip()
    frame["tier"] = frame.method.map(tier_of)
    frame["fy"] = frame.fiscal_year.astype(int)
    frame["amount"] = frame.prime_contract_original_amount.astype(float)

    # How many DOE contracts this vendor already held when this one was awarded.
    # Counted over the *whole* prime contract file, not just the model's scope,
    # so a long-standing supplier reads as established even if most of its
    # history sits below the small-purchase cap. ``cumcount`` is 0 on a vendor's
    # first contract, which is the correct value for "prior contracts".
    frame = frame.sort_values(["fy", "prime_contract_id"]).copy()
    frame["vendor_prior_contracts"] = frame.groupby("prime_vendor").cumcount()
    return frame


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach the model matrix. Every feature is knowable at award time."""
    out = frame.copy()
    start = pd.to_datetime(out.prime_contract_start_date, errors="coerce")
    end = pd.to_datetime(out.prime_contract_end_date, errors="coerce")
    registered = pd.to_datetime(out.prime_contract_registration_date, errors="coerce")

    out["duration_days"] = (end - start).dt.days
    out["log_duration"] = np.log1p(out.duration_days.clip(lower=0))
    out["dur_under_1y"] = (out.duration_days <= 400).astype(int)
    out["dur_1_to_2y"] = ((out.duration_days > 400)
                          & (out.duration_days <= 730)).astype(int)
    out["dur_over_4y"] = (out.duration_days >= 1460).astype(int)

    # Registration lag. Not a model feature (see REGISTRATION_FINDING) but
    # computed here because the review queue flags both its extremes.
    lag = (registered - start).dt.days
    out["retro_days"] = lag.clip(lower=0, upper=730).fillna(0)
    out["registered_after_end"] = (registered > end).astype(int)

    out["log_amount"] = np.log1p(out.amount)
    industry_median = out.groupby("prime_contract_industry").amount.transform("median")
    out["amount_vs_industry"] = out.log_amount - np.log1p(industry_median)

    # One dummy per real category, with the residual category as reference.
    # ``industry_key`` / ``type_key`` are kept on the frame so the site can show
    # a plain-English label without re-deriving the mapping in JavaScript.
    industry = out.prime_contract_industry.fillna("")
    out["industry_key"] = industry.map(INDUSTRIES)
    for column in set(INDUSTRIES.values()):
        out[column] = (out.industry_key == column).astype(int)

    contract_type = out.prime_contract_type.fillna("")
    out["type_key"] = contract_type.map(CONTRACT_TYPES)
    for column in set(CONTRACT_TYPES.values()):
        out[column] = (out.type_key == column).astype(int)

    return out


def model_frame(contracts: pd.DataFrame) -> pd.DataFrame:
    """Scope and label the modelling set.

    Scope is discretion-eligible tiers **above the small-purchase cap**. Both
    halves matter: dropping statutory and renewal keeps the model on decisions
    DOE actually made, and staying above the cap keeps it away from the 43,352
    contracts priced at exactly $25,000, where ``log_amount`` alone reaches
    0.982 AUC by detecting the ceiling rather than the conduct.

    ``y = 1`` means *not formally competed* -- informal or discretionary.
    """
    scoped = contracts[(contracts.amount > SMALL_PURCHASE_CAP)
                       & contracts.tier.isin(DISCRETION_ELIGIBLE)].copy()
    scoped = build_features(scoped)
    scoped["y"] = (scoped.tier != "formal").astype(int)
    return scoped


def _fit(train: pd.DataFrame, features: list[str]):
    matrix = train[features].astype(float)
    means, stds = matrix.mean(), matrix.std().replace(0, 1)
    model = LogisticRegression(max_iter=5000)
    model.fit(((matrix - means) / stds).fillna(0), train.y)
    return model, means, stds


def _score(model, means, stds, frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    matrix = ((frame[features].astype(float) - means) / stds).fillna(0)
    return model.predict_proba(matrix)[:, 1]


def rolling_auc(frame: pd.DataFrame, features: list[str],
                first: int = 2015, last: int = 2023) -> list[tuple[int, float, int]]:
    """Rolling-origin AUC: train fy<=T, embargo T+1, test T+2..T+3.

    Reported instead of a single split because a single split is what made the
    old 0.964 look stable. Origins whose test window is too small or
    single-class are skipped rather than reported as perfect.
    """
    results = []
    for origin in range(first, last + 1):
        train = frame[frame.fy <= origin]
        test = frame[frame.fy.isin([origin + 2, origin + 3])]
        if len(test) < 80 or train.y.nunique() < 2 or test.y.nunique() < 2:
            continue
        model, means, stds = _fit(train, features)
        auc = roc_auc_score(test.y, _score(model, means, stds, test, features))
        results.append((origin, float(auc), int(len(test))))
    return results


def train(contracts: pd.DataFrame, scored_out: list | None = None) -> dict:
    """Fit the shipping model and assemble its JSON payload.

    ``scored_out``, when given, receives the scored model frame so callers can
    build the review queue without refitting.
    """
    frame = model_frame(contracts)
    train_set = frame[frame.fy <= TRAIN_END]
    test_set = frame[frame.fy >= TEST_START]

    model, means, stds = _fit(train_set, FEATURES)
    holdout = roc_auc_score(test_set.y, _score(model, means, stds, test_set, FEATURES))

    rolling = rolling_auc(frame, FEATURES)
    rolling_values = np.array([auc for _, auc, _ in rolling])

    # The two numbers that keep the headline honest: what the model achieves
    # without any term-length feature, and what term length achieves alone.
    non_structural = [f for f in FEATURES if f not in STRUCTURAL]
    no_dur = np.array([a for _, a, _ in rolling_auc(frame, non_structural)])
    dur_only = np.array([a for _, a, _ in rolling_auc(frame, list(STRUCTURAL))])

    frame["p"] = _score(model, means, stds, frame, FEATURES)
    scored = frame

    print(f"  scope {len(frame):,} contracts above ${SMALL_PURCHASE_CAP:,} "
          f"({frame.y.mean():.1%} not formally competed)")
    print(f"  holdout AUC {holdout:.3f} (fy>={TEST_START})")
    print(f"  rolling AUC {rolling_values.mean():.3f} +/- {rolling_values.std():.3f} "
          f"over {len(rolling_values)} origins "
          f"[{rolling_values.min():.3f}-{rolling_values.max():.3f}]")
    print(f"  without term length {no_dur.mean():.3f}; term length alone {dur_only.mean():.3f}")

    if scored_out is not None:
        scored_out.append(scored)

    # The browser derives amount_vs_industry from the amount the user enters, so
    # it needs the same medians the fit used. Now keyed by the real industry, so
    # the browser's denominator is exactly the fit's rather than an approximation
    # over collapsed buckets.
    medians = {str(key): float(group.amount.median())
               for key, group in frame.groupby(frame.industry_key.fillna(""))
               if len(group) >= 25}
    medians["all"] = float(frame.amount.median())

    return {
        "features": FEATURES,
        "structural": list(STRUCTURAL),
        "industry_medians": medians,
        "feature_labels": {
            "log_duration": "term length",
            "dur_under_1y": "term under 1 year",
            "dur_1_to_2y": "term 1–2 years",
            "dur_over_4y": "term over 4 years",
            "log_amount": "contract amount",
            "amount_vs_industry": "size vs industry median",
            "ind_goods": "industry: goods",
            "ind_human_services": "industry: human services",
            "ind_professional_services": "industry: professional services",
            "ind_standardized_services": "industry: standardized services",
            "type_programs": "type: student programs",
            "type_requirements_services": "type: requirements — services",
            "type_doe_requirements": "type: DOE requirements contract",
            "type_supplies": "type: supplies & equipment",
            "type_consultant": "type: consultant",
            "type_work_labor": "type: work & labour",
        },
        "industry_labels": {k: v for k, v in INDUSTRY_LABELS.items() if k},
        "contract_type_labels": {k: v for k, v in CONTRACT_TYPE_LABELS.items() if k},
        "industry_reference": INDUSTRY_LABELS[None],
        "type_reference": CONTRACT_TYPE_LABELS[None],
        # Base rates the page shows next to a verdict, so "unusual" is always
        # relative to a named comparison set rather than to the whole file.
        "industry_rates": [
            {"label": INDUSTRY_LABELS.get(key, INDUSTRY_LABELS[None]),
             "key": key, "n": int(len(g)), "rate": round(float(g.y.mean()), 4)}
            for key, g in frame.groupby(frame.industry_key.fillna(""), dropna=False)
            if len(g) >= 25
        ],
        "type_rates": [
            {"label": CONTRACT_TYPE_LABELS.get(key, CONTRACT_TYPE_LABELS[None]),
             "key": key, "n": int(len(g)), "rate": round(float(g.y.mean()), 4)}
            for key, g in frame.groupby(frame.type_key.fillna(""), dropna=False)
            if len(g) >= 25
        ],
        # Reported as a negative result, not shipped as features: see
        # ``DATE_FINDING``.
        "date_finding": DATE_FINDING,
        "vendor_history_finding": VENDOR_HISTORY_FINDING,
        "registration_finding": REGISTRATION_FINDING,
        "mwbe_finding": MWBE_FINDING,
        "means": means.round(6).tolist(),
        "stds": stds.round(6).tolist(),
        "coefficients": model.coef_[0].round(6).tolist(),
        "intercept": float(model.intercept_[0]),
        "test_auc": round(float(holdout), 3),
        "rolling_auc": round(float(rolling_values.mean()), 3),
        "rolling_sd": round(float(rolling_values.std()), 3),
        "rolling_min": round(float(rolling_values.min()), 3),
        "rolling_max": round(float(rolling_values.max()), 3),
        "rolling_folds": [{"origin": o, "auc": round(a, 3), "n": n} for o, a, n in rolling],
        "auc_without_structural": round(float(no_dur.mean()), 3),
        "auc_structural_only": round(float(dur_only.mean()), 3),
        "n_labeled": int(len(frame)),
        "n_train": int(len(train_set)),
        "n_test": int(len(test_set)),
        "base_rate": round(float(frame.y.mean()), 4),
        "cap": SMALL_PURCHASE_CAP,
    }
