#!/usr/bin/env python3

from pathlib import Path
from collections import Counter
import hashlib
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from scipy.stats import binomtest
except Exception:
    binomtest = None


ROOT = Path("/workspace/AromaExperiments")

CANDIDATES = (
    ROOT
    / "outputs/natural_ugr_v1/development/stage2/"
      "candidate_outcomes_v1/candidate_outcomes_v1.csv"
)

FEATURES = (
    ROOT
    / "outputs/natural_ugr_v1/development/stage2/"
      "feature_archive_v1/stage2_sample_features_v1.csv"
)

FEATURE_NAMES = (
    ROOT
    / "outputs/natural_ugr_v1/development/stage2/"
      "feature_archive_v1/stage2_sample_feature_names_v1.json"
)

FOLDS = (
    ROOT
    / "artifacts/natural_ugr_v1/development/"
      "router_analysis_v1/fold_manifest_v1.csv"
)

OUT = (
    ROOT
    / "artifacts/natural_ugr_v2/development/"
      "hierarchical_router_v1"
)

OOF_PATH = OUT / "oof_selections_v1.csv"
SUMMARY_PATH = OUT / "development_summary_v1.json"
SPEC_PATH = OUT / "hierarchical_router_spec_v1.json"

WHOLE_BUNDLE = OUT / "router_whole_hierarchical_v1.joblib"
UNIFIED_BUNDLE = OUT / "router_unified_hierarchical_v1.joblib"


C_VALUE = 1.0
GATE_THRESHOLD = 0.5
MAX_PAIRS_PER_SAMPLE = 128

FAMILIES = {
    "whole_hierarchical": {
        "WHOLE",
    },
    "unified_hierarchical": {
        "MN",
        "SA",
        "WHOLE",
    },
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)
    return h.hexdigest()


def as01(x):
    if isinstance(x, (bool, np.bool_)):
        return int(x)

    if isinstance(x, str):
        return int(
            x.strip().lower()
            in {"1", "true", "t", "yes"}
        )

    return int(x)


# ============================================================
# LOAD FROZEN NATURAL DEVELOPMENT DATA
# ============================================================

cand = pd.read_csv(CANDIDATES)
feat = pd.read_csv(FEATURES)
folds = pd.read_csv(FOLDS)

for df in [cand, feat, folds]:
    df["raw_sample_id"] = df["raw_sample_id"].astype(str)

with FEATURE_NAMES.open("r") as f:
    obj = json.load(f)

if isinstance(obj, list):
    feature_names = obj
elif "feature_names" in obj:
    feature_names = obj["feature_names"]
elif "sample_feature_names" in obj:
    feature_names = obj["sample_feature_names"]
else:
    raise RuntimeError("Cannot recover feature names")

if len(feature_names) != 47:
    raise RuntimeError(
        f"Expected 47 features; got {len(feature_names)}"
    )

cand["candidate_correct"] = (
    cand["candidate_correct"].map(as01)
)

cand["baseline_correct"] = (
    cand["baseline_correct"].map(as01)
)

cand["candidate_method"] = (
    cand["candidate_method"]
    .astype(str)
    .str.upper()
)

cand["candidate_gain"] = (
    pd.to_numeric(
        cand["candidate_gain"],
        errors="coerce",
    )
    .fillna(0.0)
)

sample_order = (
    folds
    .sort_values("sample_index")
    ["raw_sample_id"]
    .tolist()
)

if len(sample_order) != 292:
    raise RuntimeError(
        f"Expected N=292, got {len(sample_order)}"
    )

fold_map = dict(
    zip(
        folds["raw_sample_id"],
        folds["fold"].astype(int),
    )
)

feature_map = {
    row["raw_sample_id"]:
        row[feature_names].to_numpy(
            dtype=np.float64
        )
    for _, row in feat.iterrows()
}


# ============================================================
# CANDIDATE REPRESENTATION
# ============================================================

def candidate_channels(method, gain):
    method = str(method).upper()
    gain = float(gain)

    c = np.zeros(9, dtype=np.float64)

    if method == "MN":
        c[0] = 1.0
        c[3] = gain
        c[6] = gain * gain

    elif method == "SA":
        c[1] = 1.0
        c[4] = gain
        c[7] = gain * gain

    elif method == "WHOLE":
        c[2] = 1.0
        c[5] = gain
        c[8] = gain * gain

    elif method == "IDENTITY":
        pass

    else:
        raise RuntimeError(
            f"Unknown method: {method}"
        )

    return c


def ranking_features(sample_x, method, gain):
    c = candidate_channels(method, gain)

    interactions = np.outer(
        c,
        sample_x,
    ).reshape(-1)

    return np.concatenate([
        c,
        interactions,
    ])


groups = {}

for sid in sample_order:
    g = (
        cand[
            cand["raw_sample_id"] == sid
        ]
        .sort_values("candidate_index")
        .reset_index(drop=True)
    )

    if len(g) != 43:
        raise RuntimeError(
            f"{sid}: candidate count {len(g)} != 43"
        )

    x = feature_map[sid]

    Z = np.vstack([
        ranking_features(
            x,
            row["candidate_method"],
            row["candidate_gain"],
        )
        for _, row in g.iterrows()
    ])

    groups[sid] = (g, Z)


# ============================================================
# HIERARCHICAL STAGE A — INTERVENTION GATE
# ============================================================

def gate_label(sid, methods):
    g, _ = groups[sid]

    identity = g[
        g["candidate_method"] == "IDENTITY"
    ]

    identity_correct = int(
        identity["candidate_correct"].iloc[0]
    )

    possible_repair = int(
        g[
            g["candidate_method"].isin(methods)
        ]["candidate_correct"].max()
    )

    # Intervention is useful only when baseline is wrong
    # and at least one available action can repair it.
    return int(
        identity_correct == 0
        and
        possible_repair == 1
    )


def fit_gate(ids, methods):
    X = np.vstack([
        feature_map[sid]
        for sid in ids
    ])

    y = np.asarray([
        gate_label(sid, methods)
        for sid in ids
    ])

    model = Pipeline([
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "logistic",
            LogisticRegression(
                penalty="l2",
                C=C_VALUE,
                solver="lbfgs",
                max_iter=3000,
            ),
        ),
    ])

    model.fit(X, y)

    return model


# ============================================================
# HIERARCHICAL STAGE B — ACTION RANKER
# ============================================================

def build_pairs(ids, methods):
    X_parts = []
    y_parts = []
    w_parts = []

    for sid in ids:
        g, Z = groups[sid]

        mask = (
            g["candidate_method"]
            .isin(methods)
            .to_numpy()
        )

        gg = g.loc[mask].reset_index(drop=True)
        ZZ = Z[mask]

        y = (
            gg["candidate_correct"]
            .to_numpy(dtype=int)
        )

        correct = np.where(y == 1)[0]
        wrong = np.where(y == 0)[0]

        if len(correct) == 0 or len(wrong) == 0:
            continue

        pairs = [
            (i, j)
            for i in correct
            for j in wrong
        ]

        if len(pairs) > MAX_PAIRS_PER_SAMPLE:
            idx = np.linspace(
                0,
                len(pairs) - 1,
                MAX_PAIRS_PER_SAMPLE,
                dtype=int,
            )

            pairs = [
                pairs[k]
                for k in idx
            ]

        n = len(pairs)

        positive = np.vstack([
            ZZ[i] - ZZ[j]
            for i, j in pairs
        ])

        X_parts.extend([
            positive,
            -positive,
        ])

        y_parts.extend([
            np.ones(n, dtype=int),
            np.zeros(n, dtype=int),
        ])

        w_parts.append(
            np.full(
                2 * n,
                1.0 / (2 * n),
                dtype=np.float64,
            )
        )

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    w = np.concatenate(w_parts)

    w *= len(w) / w.sum()

    return X, y, w


def fit_ranker(ids, methods):
    X, y, w = build_pairs(
        ids,
        methods,
    )

    model = Pipeline([
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "logistic",
            LogisticRegression(
                penalty="l2",
                C=C_VALUE,
                solver="lbfgs",
                max_iter=3000,
            ),
        ),
    ])

    model.fit(
        X,
        y,
        logistic__sample_weight=w,
    )

    return model


METHOD_PRIORITY = {
    "MN": 0,
    "SA": 1,
    "WHOLE": 2,
}


def identity_row(sid):
    g, _ = groups[sid]

    x = g[
        g["candidate_method"] == "IDENTITY"
    ]

    return x.iloc[0]


def choose_nonidentity(
    ranker,
    sid,
    methods,
):
    g, Z = groups[sid]

    mask = (
        g["candidate_method"]
        .isin(methods)
        .to_numpy()
    )

    gg = (
        g.loc[mask]
        .copy()
        .reset_index(drop=True)
    )

    ZZ = Z[mask]

    gg["score"] = (
        ranker.decision_function(ZZ)
    )

    gg["priority"] = (
        gg["candidate_method"]
        .map(METHOD_PRIORITY)
    )

    gg = gg.sort_values(
        [
            "score",
            "candidate_gain",
            "priority",
        ],
        ascending=[
            False,
            True,
            True,
        ],
        kind="mergesort",
    )

    return gg.iloc[0]


# ============================================================
# GROUPED 5-FOLD OOF REPRODUCTION
# ============================================================

records = []

for family, methods in FAMILIES.items():

    for fold in sorted(
        folds["fold"].unique()
    ):
        train_ids = [
            sid
            for sid in sample_order
            if fold_map[sid] != fold
        ]

        test_ids = [
            sid
            for sid in sample_order
            if fold_map[sid] == fold
        ]

        gate = fit_gate(
            train_ids,
            methods,
        )

        ranker = fit_ranker(
            train_ids,
            methods,
        )

        for sid in test_ids:
            probability = float(
                gate.predict_proba(
                    feature_map[sid].reshape(1, -1)
                )[0, 1]
            )

            intervene = (
                probability
                >=
                GATE_THRESHOLD
            )

            if intervene:
                selected = choose_nonidentity(
                    ranker,
                    sid,
                    methods,
                )
            else:
                selected = identity_row(sid)

            g, _ = groups[sid]

            baseline = int(
                g["baseline_correct"].iloc[0]
            )

            correct = int(
                selected["candidate_correct"]
            )

            oracle = int(
                g[
                    g["candidate_method"].isin(
                        {"IDENTITY", *methods}
                    )
                ]["candidate_correct"].max()
            )

            records.append({
                "family":
                    family,

                "fold":
                    int(fold),

                "raw_sample_id":
                    sid,

                "gate_probability":
                    probability,

                "intervene":
                    int(intervene),

                "baseline_correct":
                    baseline,

                "selected_method":
                    str(
                        selected["candidate_method"]
                    ),

                "selected_gain":
                    float(
                        selected["candidate_gain"]
                    ),

                "selected_correct":
                    correct,

                "repair":
                    int(
                        baseline == 0
                        and
                        correct == 1
                    ),

                "break":
                    int(
                        baseline == 1
                        and
                        correct == 0
                    ),

                "oracle_correct":
                    oracle,
            })


oof = pd.DataFrame(records)

summaries = {}

for family in FAMILIES:
    x = oof[
        oof["family"] == family
    ]

    summaries[family] = {
        "n":
            int(len(x)),

        "correct":
            int(
                x["selected_correct"].sum()
            ),

        "accuracy":
            float(
                x["selected_correct"].mean()
            ),

        "baseline_correct":
            int(
                x["baseline_correct"].sum()
            ),

        "repairs":
            int(
                x["repair"].sum()
            ),

        "breaks":
            int(
                x["break"].sum()
            ),

        "interventions":
            int(
                x["intervene"].sum()
            ),

        "oracle_correct":
            int(
                x["oracle_correct"].sum()
            ),

        "oracle_accuracy":
            float(
                x["oracle_correct"].mean()
            ),

        "selection_methods":
            {
                str(k): int(v)
                for k, v
                in Counter(
                    x["selected_method"]
                ).items()
            },
    }


whole = summaries[
    "whole_hierarchical"
]

unified = summaries[
    "unified_hierarchical"
]


paired = (
    oof[
        oof["family"]
        ==
        "whole_hierarchical"
    ][
        [
            "raw_sample_id",
            "selected_correct",
        ]
    ]
    .rename(
        columns={
            "selected_correct":
                "whole_correct"
        }
    )
    .merge(
        oof[
            oof["family"]
            ==
            "unified_hierarchical"
        ][
            [
                "raw_sample_id",
                "selected_correct",
            ]
        ]
        .rename(
            columns={
                "selected_correct":
                    "unified_correct"
            }
        ),
        on="raw_sample_id",
    )
)


unified_only = int(
    (
        (paired["unified_correct"] == 1)
        &
        (paired["whole_correct"] == 0)
    ).sum()
)

whole_only = int(
    (
        (paired["whole_correct"] == 1)
        &
        (paired["unified_correct"] == 0)
    ).sum()
)


p = None

if (
    binomtest is not None
    and
    unified_only + whole_only > 0
):
    p = float(
        binomtest(
            min(
                unified_only,
                whole_only,
            ),
            n=(
                unified_only
                +
                whole_only
            ),
            p=0.5,
            alternative="two-sided",
        ).pvalue
    )


primary = {
    "unified_minus_whole_correct":
        int(
            unified["correct"]
            -
            whole["correct"]
        ),

    "unified_minus_whole_pp":
        float(
            100
            *
            (
                unified["correct"]
                -
                whole["correct"]
            )
            /
            292
        ),

    "unified_only_correct":
        unified_only,

    "whole_only_correct":
        whole_only,

    "exploratory_exact_p":
        p,
}


# Core reproduction check only.
expected = {
    "whole_correct":
        100,

    "unified_correct":
        104,

    "whole_breaks":
        0,

    "unified_breaks":
        0,

    "unified_only":
        5,

    "whole_only":
        1,
}

actual = {
    "whole_correct":
        whole["correct"],

    "unified_correct":
        unified["correct"],

    "whole_breaks":
        whole["breaks"],

    "unified_breaks":
        unified["breaks"],

    "unified_only":
        unified_only,

    "whole_only":
        whole_only,
}

if actual != expected:
    raise RuntimeError(
        "Hierarchical formulation did not reproduce "
        f"the quick diagnostic.\nExpected={expected}\n"
        f"Actual={actual}"
    )


# ============================================================
# FINAL DEVELOPMENT FIT
# ============================================================

bundle_paths = {}

for family, methods in FAMILIES.items():

    gate = fit_gate(
        sample_order,
        methods,
    )

    ranker = fit_ranker(
        sample_order,
        methods,
    )

    bundle = {
        "artifact":
            "Natural UGR v2 hierarchical router",

        "status":
            (
                "FITTED ON ALL NATURAL DEVELOPMENT "
                "SAMPLES BEFORE UNTOUCHED FINAL COHORT"
            ),

        "family":
            family,

        "nonidentity_methods":
            sorted(methods),

        "sample_feature_names":
            feature_names,

        "sample_feature_count":
            47,

        "ranking_feature_count":
            432,

        "gate_threshold":
            GATE_THRESHOLD,

        "C":
            C_VALUE,

        "max_pairs_per_sample":
            MAX_PAIRS_PER_SAMPLE,

        "gate_target":
            (
                "IDENTITY incorrect AND at least one "
                "allowed nonidentity candidate correct"
            ),

        "ranker_target":
            (
                "within-sample correct-vs-incorrect "
                "nonidentity candidate preference"
            ),

        "gate_model":
            gate,

        "ranker_model":
            ranker,
    }

    path = (
        WHOLE_BUNDLE
        if family == "whole_hierarchical"
        else
        UNIFIED_BUNDLE
    )

    joblib.dump(
        bundle,
        path,
    )

    bundle_paths[family] = path


# ============================================================
# WRITE ARCHIVE
# ============================================================

oof.to_csv(
    OOF_PATH,
    index=False,
)

summary = {
    "artifact":
        "Natural UGR v2 hierarchical development summary",

    "evidence_class":
        (
            "EXPLORATORY DEVELOPMENT; formulation chosen "
            "after observing Natural UGR v1 development"
        ),

    "population_n":
        292,

    "families":
        summaries,

    "primary_comparison":
        primary,

    "interpretation":
        (
            "Hierarchical intervention gating followed by "
            "conditional action ranking converts the Natural "
            "development comparison from v1's negative point "
            "estimate to a positive Unified-vs-Whole point "
            "estimate while producing zero breaks in both "
            "hierarchical policies. Untouched final evidence "
            "is required for confirmation."
        ),
}

SUMMARY_PATH.write_text(
    json.dumps(
        summary,
        indent=2,
        sort_keys=True,
    )
    +
    "\n"
)

spec = {
    "artifact":
        "Natural UGR v2 hierarchical router specification",

    "development_status":
        "FROZEN AFTER EXPLORATORY DEVELOPMENT, BEFORE FINAL",

    "population_n":
        292,

    "folds":
        5,

    "fold_source":
        str(
            FOLDS.relative_to(ROOT)
        ),

    "gate": {
        "features":
            47,

        "model":
            "StandardScaler + L2 LogisticRegression",

        "C":
            C_VALUE,

        "threshold":
            GATE_THRESHOLD,

        "target":
            (
                "identity incorrect and at least one "
                "available nonidentity action correct"
            ),
    },

    "ranker": {
        "model":
            "pairwise StandardScaler + L2 LogisticRegression",

        "C":
            C_VALUE,

        "dimensions":
            432,

        "maximum_pairs_per_sample":
            MAX_PAIRS_PER_SAMPLE,

        "sample_weighting":
            "equal total pair weight per source sample",
    },

    "whole_family":
        [
            "IDENTITY",
            "WHOLE",
        ],

    "unified_family":
        [
            "IDENTITY",
            "MN",
            "SA",
            "WHOLE",
        ],

    "final_primary_comparison":
        "unified_hierarchical vs whole_hierarchical",

    "development_tuning_after_this_freeze":
        False,

    "untouched_final_observed":
        False,

    "input_sha256": {
        "candidate_outcomes":
            sha256(CANDIDATES),

        "sample_features":
            sha256(FEATURES),

        "feature_names":
            sha256(FEATURE_NAMES),

        "fold_manifest":
            sha256(FOLDS),
    },

    "router_bundles": {
        family: {
            "path":
                str(
                    path.relative_to(ROOT)
                ),

            "sha256":
                sha256(path),
        }
        for family, path
        in bundle_paths.items()
    },

    "oof_sha256":
        sha256(OOF_PATH),

    "summary_sha256":
        sha256(SUMMARY_PATH),
}

SPEC_PATH.write_text(
    json.dumps(
        spec,
        indent=2,
        sort_keys=True,
    )
    +
    "\n"
)


# ============================================================
# PRINT RESULT
# ============================================================

print()
print("=" * 78)
print("NATURAL UGR V2 HIERARCHICAL DEVELOPMENT FREEZE")
print("=" * 78)

for family in [
    "whole_hierarchical",
    "unified_hierarchical",
]:
    x = summaries[family]

    print()
    print(family)

    print(
        f"  accuracy: "
        f"{x['correct']}/{x['n']} "
        f"= {100*x['accuracy']:.3f}%"
    )

    print(
        f"  repairs / breaks: "
        f"{x['repairs']} / {x['breaks']}"
    )

    print(
        f"  interventions: "
        f"{x['interventions']}/{x['n']}"
    )

    print(
        f"  oracle: "
        f"{x['oracle_correct']}/{x['n']} "
        f"= {100*x['oracle_accuracy']:.3f}%"
    )

    print(
        f"  selections: "
        f"{x['selection_methods']}"
    )


print()
print("PRIMARY")
print(
    "Unified - Whole:",
    f"{primary['unified_minus_whole_pp']:+.3f} pp"
)

print(
    "Unified-only / Whole-only:",
    unified_only,
    "/",
    whole_only,
)

print(
    "Exploratory paired p:",
    p,
)

print()
print("Bundles:")

for family, path in bundle_paths.items():
    print(
        family,
        sha256(path),
        path.relative_to(ROOT),
    )

print()
print(
    "HIERARCHICAL DEVELOPMENT REPRODUCTION: PASS"
)

print(
    "FULL DEVELOPMENT ROUTERS FIT: PASS"
)

print(
    "UNTOUCHED FINAL OBSERVED: False"
)
