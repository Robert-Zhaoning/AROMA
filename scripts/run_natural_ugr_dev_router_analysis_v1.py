import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# FROZEN INPUTS
# ============================================================

PROTOCOL = Path(
    'manifests/natural_ugr_v1/protocol_v1.json'
)

CANDIDATES = Path(
    'outputs/natural_ugr_v1/development/stage2/candidate_outcomes_v1/candidate_outcomes_v1.csv'
)

CANDIDATE_FREEZE = Path(
    'manifests/natural_ugr_v1/f2f_raw_candidate_outcomes_freeze_v1.json'
)

FEATURE_CSV = Path(
    'outputs/natural_ugr_v1/development/stage2/feature_archive_v1/stage2_sample_features_v1.csv'
)

FEATURE_MATRIX = Path(
    'outputs/natural_ugr_v1/development/stage2/feature_archive_v1/stage2_sample_feature_matrix_v1.npy'
)

FEATURE_NAMES = Path(
    'outputs/natural_ugr_v1/development/stage2/feature_archive_v1/stage2_sample_feature_names_v1.json'
)

FEATURE_FREEZE = Path(
    'manifests/natural_ugr_v1/f2e_feature_archive_freeze_v1.json'
)

ANALYSIS_ROOT = Path(
    "artifacts/natural_ugr_v1/"
    "development/router_analysis_v1"
)

ANALYSIS_SPEC = (
    ANALYSIS_ROOT
    /
    "router_analysis_spec_v1.json"
)

FOLD_MANIFEST = (
    ANALYSIS_ROOT
    /
    "fold_manifest_v1.csv"
)

EXECUTION_FREEZE = (
    ANALYSIS_ROOT
    /
    "router_analysis_execution_freeze_v1.json"
)

OUT = (
    ANALYSIS_ROOT
    /
    "results_v1"
)

OOF_SELECTIONS = (
    OUT
    /
    "oof_selections_v1.csv"
)

SUMMARY = (
    OUT
    /
    "summary_v1.json"
)

COMPARATOR = (
    OUT
    /
    "selected_restricted_comparator_v1.json"
)

BUNDLE_MANIFEST = (
    OUT
    /
    "router_bundle_manifest_v1.json"
)

METADATA = (
    OUT
    /
    "router_analysis_metadata_v1.json"
)

ROUTER_BUNDLES = {
    "lowrank_restricted":
        OUT
        /
        "router_lowrank_restricted_v1.joblib",

    "whole_restricted":
        OUT
        /
        "router_whole_restricted_v1.joblib",

    "unified":
        OUT
        /
        "router_unified_v1.joblib",
}


EXPECTED_N = 292
EXPECTED_CANDIDATE_ROWS = 12556

C_VALUE = 1.0
MAX_ITER = 10000

CANDIDATE_CHANNELS = [
    "is_MN",
    "is_SA",
    "is_WHOLE",
    "MN_gain",
    "SA_gain",
    "WHOLE_gain",
    "MN_gain_sq",
    "SA_gain_sq",
    "WHOLE_gain_sq",
]

FAMILIES = {
    "lowrank_restricted":
        {
            "IDENTITY",
            "MN",
            "SA",
        },

    "whole_restricted":
        {
            "IDENTITY",
            "WHOLE",
        },

    "unified":
        {
            "IDENTITY",
            "MN",
            "SA",
            "WHOLE",
        },
}

METHOD_RANK = {
    "IDENTITY":
        0,

    "MN":
        1,

    "SA":
        2,

    "WHOLE":
        3,
}


# ============================================================
# HELPERS
# ============================================================

def sha256(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        while True:
            b = f.read(
                1024 * 1024
            )

            if not b:
                break

            h.update(b)

    return h.hexdigest()


def as01(
    series,
    name,
):
    values = (
        series
        .astype(str)
        .str.strip()
        .str.lower()
    )

    mapping = {
        "true":
            1,

        "false":
            0,

        "1":
            1,

        "0":
            0,

        "1.0":
            1,

        "0.0":
            0,
    }

    out = values.map(
        mapping
    )

    if out.isna().any():
        bad = (
            series[
                out.isna()
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        raise RuntimeError(
            f"{name}: cannot parse "
            f"boolean values {bad}"
        )

    return out.astype(
        int
    )


def new_model():

    return Pipeline(
        [
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
                    max_iter=MAX_ITER,
                ),
            ),
        ]
    )


def positive_probability(
    model,
    X,
):

    classes = (
        model[
            "logistic"
        ]
        .classes_
        .tolist()
    )

    if 1 not in classes:
        raise RuntimeError(
            f"Positive class absent: {classes}"
        )

    pos_idx = classes.index(
        1
    )

    return (
        model.predict_proba(
            X
        )[
            :,
            pos_idx
        ]
    )


def choose_candidate(
    frame,
):

    if len(frame) == 0:
        raise RuntimeError(
            "Cannot choose from empty candidate set."
        )

    best = float(
        frame[
            "_predicted_correctness"
        ].max()
    )

    # Protocol tie means equal predicted correctness.
    tied = frame[
        frame[
            "_predicted_correctness"
        ]
        ==
        best
    ].copy()


    tied[
        "_identity"
    ] = (
        tied[
            "candidate_method"
        ]
        ==
        "IDENTITY"
    ).astype(
        int
    )

    tied[
        "_method_rank"
    ] = (
        tied[
            "candidate_method"
        ]
        .map(
            METHOD_RANK
        )
    )


    if tied[
        "_method_rank"
    ].isna().any():

        raise RuntimeError(
            "Unknown method in tie-break."
        )


    tied = tied.sort_values(
        [
            "_identity",
            "candidate_gain",
            "_method_rank",
        ],
        ascending=[
            False,
            True,
            True,
        ],
        kind="mergesort",
    )


    return tied.iloc[
        0
    ]


def family_mask(
    frame,
    family,
):

    allowed = FAMILIES[
        family
    ]

    return (
        frame[
            "candidate_method"
        ]
        .isin(
            allowed
        )
        .to_numpy()
    )


# ============================================================
# VERIFY NATURAL EXECUTION FREEZE + LOAD FROZEN INPUTS
# ============================================================

if not EXECUTION_FREEZE.exists():
    raise FileNotFoundError(
        EXECUTION_FREEZE
    )


execution = json.loads(
    EXECUTION_FREEZE.read_text(
        encoding="utf-8"
    )
)


if (
    execution[
        "status"
    ]
    !=
    "FROZEN BEFORE NATURAL UGR DEVELOPMENT ROUTER ANALYSIS"
):
    raise RuntimeError(
        "Unexpected Natural router-analysis "
        "execution freeze status."
    )


for name, spec in (
    execution[
        "files"
    ].items()
):

    path = Path(
        spec[
            "path"
        ]
    )

    if not path.is_file():
        raise RuntimeError(
            f"Missing frozen F3 input: {name}: {path}"
        )

    actual = sha256(
        path
    )

    if actual != spec[
        "sha256"
    ]:
        raise RuntimeError(
            "Frozen F3 input drift:\n"
            f"{name}\n"
            f"path={path}\n"
            f"expected={spec['sha256']}\n"
            f"actual={actual}"
        )


if (
    execution[
        "population_n"
    ]
    !=
    EXPECTED_N
):
    raise RuntimeError(
        "Execution-freeze population mismatch."
    )


if (
    execution[
        "candidate_rows"
    ]
    !=
    EXPECTED_CANDIDATE_ROWS
):
    raise RuntimeError(
        "Execution-freeze candidate-row mismatch."
    )


if execution[
    "final_cohort_observed"
] is not False:

    raise RuntimeError(
        "Final cohort was observed before development routing."
    )


print(
    "PASS: frozen Natural router-analysis inputs verified"
)

print(
    "Execution freeze SHA256:",
    sha256(
        EXECUTION_FREEZE
    ),
)


# ============================================================
# REFUSE RESULT OVERWRITE
# ============================================================

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


for p in [
    OOF_SELECTIONS,
    SUMMARY,
    COMPARATOR,
    BUNDLE_MANIFEST,
    METADATA,
    *ROUTER_BUNDLES.values(),
]:

    if p.exists():

        raise RuntimeError(
            "Natural router-analysis result already exists: "
            f"{p}"
        )


# ============================================================
# LOAD FROZEN NATURAL INPUTS
# ============================================================

cand = pd.read_csv(
    CANDIDATES,
    dtype={
        "raw_sample_id":
            str,

        "cohort_uid":
            str,
    },
)


features = pd.read_csv(
    FEATURE_CSV,
    dtype={
        "raw_sample_id":
            str,

        "cohort_uid":
            str,
    },
)


X_SAMPLE = np.load(
    FEATURE_MATRIX
)


feature_name_obj = json.loads(
    FEATURE_NAMES.read_text(
        encoding="utf-8"
    )
)


sample_feature_names = list(
    feature_name_obj[
        "feature_names"
    ]
)


folds = pd.read_csv(
    FOLD_MANIFEST,
    dtype={
        "raw_sample_id":
            str,

        "cohort_uid":
            str,
    },
)


if len(cand) != EXPECTED_CANDIDATE_ROWS:

    raise RuntimeError(
        "Candidate row count mismatch."
    )


if len(features) != EXPECTED_N:

    raise RuntimeError(
        "Feature sample count mismatch."
    )


if X_SAMPLE.shape != (
    EXPECTED_N,
    47,
):

    raise RuntimeError(
        f"Unexpected sample feature shape: "
        f"{X_SAMPLE.shape}"
    )


if len(
    sample_feature_names
) != 47:

    raise RuntimeError(
        "Feature-name count mismatch."
    )


if len(folds) != EXPECTED_N:

    raise RuntimeError(
        "Fold-manifest sample count mismatch."
    )


if not np.isfinite(
    X_SAMPLE
).all():

    raise RuntimeError(
        "Non-finite sample feature."
    )


feature_ids = (
    features[
        "raw_sample_id"
    ]
    .astype(str)
    .tolist()
)


fold_ids = (
    folds[
        "raw_sample_id"
    ]
    .astype(str)
    .tolist()
)


if feature_ids != fold_ids:

    raise RuntimeError(
        "Feature/fold ordering mismatch."
    )


feature_uids = (
    features[
        "cohort_uid"
    ]
    .astype(str)
    .tolist()
)


fold_uids = (
    folds[
        "cohort_uid"
    ]
    .astype(str)
    .tolist()
)


if feature_uids != fold_uids:

    raise RuntimeError(
        "Feature/fold cohort UID mismatch."
    )


sample_ids = feature_ids


for column in [
    "baseline_correct",
    "candidate_correct",
    "repair",
    "break",
]:

    cand[
        column
    ] = as01(
        cand[
            column
        ],
        column,
    )


cand[
    "raw_sample_id"
] = (
    cand[
        "raw_sample_id"
    ]
    .astype(str)
)


cand[
    "cohort_uid"
] = (
    cand[
        "cohort_uid"
    ]
    .astype(str)
)


cand[
    "candidate_method"
] = (
    cand[
        "candidate_method"
    ]
    .astype(str)
)


cand[
    "candidate_gain"
] = pd.to_numeric(
    cand[
        "candidate_gain"
    ],
    errors="raise",
).astype(
    np.float64
)


sample_index_map = {
    sid:
        i
    for i, sid
    in enumerate(
        sample_ids
    )
}


fold_map = dict(
    zip(
        folds[
            "raw_sample_id"
        ].astype(str),
        folds[
            "fold"
        ].astype(int),
    )
)


if set(
    cand[
        "raw_sample_id"
    ].unique()
) != set(
    sample_ids
):

    raise RuntimeError(
        "Candidate/sample population mismatch."
    )


candidate_counts = (
    cand[
        "raw_sample_id"
    ]
    .value_counts()
)


if not (
    candidate_counts
    ==
    43
).all():

    raise RuntimeError(
        "Candidate count per sample != 43."
    )


cand[
    "_sample_index"
] = (
    cand[
        "raw_sample_id"
    ]
    .map(
        sample_index_map
    )
)


cand[
    "_fold"
] = (
    cand[
        "raw_sample_id"
    ]
    .map(
        fold_map
    )
)


if cand[
    [
        "_sample_index",
        "_fold",
    ]
].isna().any().any():

    raise RuntimeError(
        "Failed sample-index/fold mapping."
    )


cand[
    "_sample_index"
] = (
    cand[
        "_sample_index"
    ]
    .astype(int)
)


cand[
    "_fold"
] = (
    cand[
        "_fold"
    ]
    .astype(int)
)


per_sample_fold_count = (
    cand.groupby(
        "raw_sample_id"
    )[
        "_fold"
    ]
    .nunique()
)


if not (
    per_sample_fold_count
    ==
    1
).all():

    raise RuntimeError(
        "Candidate rows from a sample "
        "cross fold boundaries."
    )


if set(
    cand[
        "candidate_method"
    ].unique()
) != {
    "IDENTITY",
    "MN",
    "SA",
    "WHOLE",
}:

    raise RuntimeError(
        "Unexpected candidate-method set."
    )


print(
    "Candidate rows:",
    len(
        cand
    ),
)

print(
    "Sample features:",
    X_SAMPLE.shape,
)

print(
    "Development samples:",
    len(
        sample_ids
    ),
)

print(
    "Fold counts:",
    folds[
        "fold"
    ]
    .value_counts()
    .sort_index()
    .to_dict(),
)


# ============================================================
# BUILD FROZEN 479-D DESIGN
# ============================================================

methods = (
    cand[
        "candidate_method"
    ]
    .to_numpy(
        dtype=str
    )
)

gain = (
    cand[
        "candidate_gain"
    ]
    .to_numpy(
        dtype=np.float64
    )
)


is_mn = (
    methods
    ==
    "MN"
).astype(
    np.float64
)

is_sa = (
    methods
    ==
    "SA"
).astype(
    np.float64
)

is_whole = (
    methods
    ==
    "WHOLE"
).astype(
    np.float64
)


C = np.column_stack(
    [
        is_mn,
        is_sa,
        is_whole,
        is_mn * gain,
        is_sa * gain,
        is_whole * gain,
        is_mn * gain ** 2,
        is_sa * gain ** 2,
        is_whole * gain ** 2,
    ]
)


if C.shape != (
    EXPECTED_CANDIDATE_ROWS,
    9,
):

    raise RuntimeError(
        f"Unexpected candidate-channel "
        f"shape: {C.shape}"
    )


sample_idx = (
    cand[
        "_sample_index"
    ]
    .to_numpy(
        dtype=int
    )
)

Xs = X_SAMPLE[
    sample_idx
]


interactions = np.concatenate(
    [
        Xs
        *
        C[
            :,
            j:j+1
        ]
        for j in range(
            C.shape[
                1
            ]
        )
    ],
    axis=1,
)


DESIGN = np.concatenate(
    [
        Xs,
        C,
        interactions,
    ],
    axis=1,
)


design_feature_names = (
    sample_feature_names
    +
    CANDIDATE_CHANNELS
    +
    [
        f"{s}__X__{c}"
        for c in CANDIDATE_CHANNELS
        for s in sample_feature_names
    ]
)


if DESIGN.shape != (
    EXPECTED_CANDIDATE_ROWS,
    479,
):

    raise RuntimeError(
        f"Unexpected design shape: "
        f"{DESIGN.shape}"
    )


if len(
    design_feature_names
) != 479:

    raise RuntimeError(
        "Design feature-name count mismatch."
    )


if not np.isfinite(
    DESIGN
).all():

    raise RuntimeError(
        "Non-finite value in design matrix."
    )


identity_mask = (
    methods
    ==
    "IDENTITY"
)


if not np.all(
    C[
        identity_mask
    ]
    ==
    0.0
):

    raise RuntimeError(
        "IDENTITY candidate channels "
        "are not all zero."
    )


print(
    "Design matrix:",
    DESIGN.shape
)


# ============================================================
# OOF ROUTER
# ============================================================

y = (
    cand[
        "candidate_correct"
    ]
    .to_numpy(
        dtype=int
    )
)


def run_oof(
    family,
):

    fam = family_mask(
        cand,
        family,
    )

    rows = []


    for fold_idx in range(
        5
    ):

        train = (
            fam
            &
            (
                cand[
                    "_fold"
                ].to_numpy()
                !=
                fold_idx
            )
        )

        valid = (
            fam
            &
            (
                cand[
                    "_fold"
                ].to_numpy()
                ==
                fold_idx
            )
        )


        train_y = y[
            train
        ]


        if set(
            np.unique(
                train_y
            ).tolist()
        ) != {
            0,
            1,
        }:

            raise RuntimeError(
                f"{family} fold {fold_idx}: "
                "training target lacks both classes."
            )


        model = new_model()

        model.fit(
            DESIGN[
                train
            ],
            train_y,
        )


        probs = positive_probability(
            model,
            DESIGN[
                valid
            ],
        )


        valid_rows = (
            cand.loc[
                valid,
                [
                    "raw_sample_id",
                    "cohort_uid",
                    "ground_truth",
                    "baseline_prediction",
                    "baseline_correct",
                    "candidate_index",
                    "candidate_method",
                    "candidate_gain",
                    "candidate_prediction",
                    "candidate_correct",
                    "repair",
                    "break",
                ],
            ]
            .copy()
        )


        valid_rows[
            "_predicted_correctness"
        ] = probs


        heldout_ids = (
            folds[
                folds[
                    "fold"
                ].astype(int)
                ==
                fold_idx
            ][
                "raw_sample_id"
            ]
            .tolist()
        )


        for sid in heldout_ids:

            group = valid_rows[
                valid_rows[
                    "raw_sample_id"
                ]
                ==
                sid
            ]


            if len(group) == 0:
                raise RuntimeError(
                    f"{family} fold {fold_idx}: "
                    f"missing held-out sample {sid}"
                )


            chosen = choose_candidate(
                group
            )


            rows.append(
                {
                    "router":
                        family,

                    "fold":
                        int(
                            fold_idx
                        ),

                    "raw_sample_id":
                        str(
                            sid
                        ),

                    "cohort_uid":
                        str(
                            chosen[
                                "cohort_uid"
                            ]
                        ),

                    "ground_truth":
                        int(
                            chosen[
                                "ground_truth"
                            ]
                        ),

                    "baseline_prediction":
                        int(
                            chosen[
                                "baseline_prediction"
                            ]
                        ),

                    "baseline_correct":
                        int(
                            chosen[
                                "baseline_correct"
                            ]
                        ),

                    "candidate_index":
                        int(
                            chosen[
                                "candidate_index"
                            ]
                        ),

                    "candidate_method":
                        str(
                            chosen[
                                "candidate_method"
                            ]
                        ),

                    "candidate_gain":
                        float(
                            chosen[
                                "candidate_gain"
                            ]
                        ),

                    "predicted_correctness":
                        float(
                            chosen[
                                "_predicted_correctness"
                            ]
                        ),

                    "candidate_prediction":
                        int(
                            chosen[
                                "candidate_prediction"
                            ]
                        ),

                    "correct":
                        int(
                            chosen[
                                "candidate_correct"
                            ]
                        ),

                    "repair":
                        int(
                            chosen[
                                "repair"
                            ]
                        ),

                    "break":
                        int(
                            chosen[
                                "break"
                            ]
                        ),
                }
            )


    result = pd.DataFrame(
        rows
    )


    if len(
        result
    ) != EXPECTED_N:

        raise RuntimeError(
            f"{family}: expected "
            f"{EXPECTED_N} OOF selections; "
            f"got {len(result)}"
        )


    if result[
        "raw_sample_id"
    ].nunique() != EXPECTED_N:

        raise RuntimeError(
            f"{family}: duplicate/missing "
            "OOF sample selections."
        )


    expected_order = set(
        sample_ids
    )

    if set(
        result[
            "raw_sample_id"
        ]
    ) != expected_order:

        raise RuntimeError(
            f"{family}: OOF population mismatch."
        )


    return result


# ============================================================
# OOF FOR ALL THREE ROUTERS
# ============================================================

oof_parts = []

for family in [
    "lowrank_restricted",
    "whole_restricted",
    "unified",
]:

    print(
        f"Running OOF: {family}"
    )

    part = run_oof(
        family
    )

    oof_parts.append(
        part
    )


oof = pd.concat(
    oof_parts,
    ignore_index=True,
)


if len(oof) != (
    EXPECTED_N
    *
    3
):
    raise RuntimeError(
        "Combined OOF row count mismatch."
    )


oof.to_csv(
    OOF_SELECTIONS,
    index=False,
)


# ============================================================
# SUMMARY METRICS
# ============================================================

identity = cand[
    cand[
        "candidate_method"
    ]
    ==
    "IDENTITY"
].copy()


if len(
    identity
) != EXPECTED_N:

    raise RuntimeError(
        "Identity row count mismatch."
    )


baseline_correct = int(
    identity[
        "candidate_correct"
    ].sum()
)

baseline_acc = float(
    identity[
        "candidate_correct"
    ].mean()
)


def summarize_family(
    family,
):

    selected = (
        oof[
            oof[
                "router"
            ]
            ==
            family
        ]
        .copy()
    )


    selected = (
        selected
        .set_index(
            "raw_sample_id"
        )
        .loc[
            sample_ids
        ]
        .reset_index()
    )


    router_correct = int(
        selected[
            "correct"
        ].sum()
    )

    router_acc = float(
        selected[
            "correct"
        ].mean()
    )

    repairs = int(
        selected[
            "repair"
        ].sum()
    )

    breaks = int(
        selected[
            "break"
        ].sum()
    )


    nonzero = selected[
        selected[
            "candidate_method"
        ]
        !=
        "IDENTITY"
    ]


    if len(
        nonzero
    ):

        mean_nonzero_gain = float(
            nonzero[
                "candidate_gain"
            ].mean()
        )

    else:
        mean_nonzero_gain = 0.0


    fam_mask = family_mask(
        cand,
        family,
    )


    oracle_by_sample = (
        cand.loc[
            fam_mask
        ]
        .groupby(
            "raw_sample_id"
        )[
            "candidate_correct"
        ]
        .max()
        .reindex(
            sample_ids
        )
    )


    if oracle_by_sample.isna().any():
        raise RuntimeError(
            f"{family}: oracle population mismatch."
        )


    oracle_correct = int(
        oracle_by_sample.sum()
    )

    oracle_acc = float(
        oracle_by_sample.mean()
    )


    oracle_headroom = (
        oracle_acc
        -
        baseline_acc
    )

    router_gain = (
        router_acc
        -
        baseline_acc
    )


    if oracle_headroom > 0:

        fraction_captured = float(
            router_gain
            /
            oracle_headroom
        )

    else:
        fraction_captured = None


    method_counts = {
        str(k):
            int(v)
        for k, v
        in selected[
            "candidate_method"
        ].value_counts().items()
    }


    gain_counts = {
        f"{float(k):.2f}":
            int(v)
        for k, v
        in selected[
            "candidate_gain"
        ].value_counts().sort_index().items()
    }


    fold_accuracy = {
        str(int(k)):
            float(v)
        for k, v
        in selected.groupby(
            "fold"
        )[
            "correct"
        ].mean().items()
    }


    return {
        "router":
            family,

        "n":
            EXPECTED_N,

        "baseline_correct":
            baseline_correct,

        "baseline_accuracy":
            baseline_acc,

        "router_correct":
            router_correct,

        "router_accuracy":
            router_acc,

        "router_minus_baseline_pp":
            100.0
            *
            (
                router_acc
                -
                baseline_acc
            ),

        "repairs":
            repairs,

        "breaks":
            breaks,

        "mean_selected_nonzero_gain":
            mean_nonzero_gain,

        "nonidentity_selections":
            int(
                len(
                    nonzero
                )
            ),

        "selection_method_counts":
            method_counts,

        "selection_gain_counts":
            gain_counts,

        "fold_accuracy":
            fold_accuracy,

        "candidate_oracle_correct":
            oracle_correct,

        "candidate_oracle_accuracy":
            oracle_acc,

        "candidate_oracle_headroom_pp":
            100.0
            *
            oracle_headroom,

        "fraction_candidate_oracle_headroom_captured":
            fraction_captured,
    }


family_summary = {
    family:
        summarize_family(
            family
        )
    for family in [
        "lowrank_restricted",
        "whole_restricted",
        "unified",
    ]
}


# ============================================================
# PREDECLARED RESTRICTED COMPARATOR SELECTION
# ============================================================

low = family_summary[
    "lowrank_restricted"
]

whole = family_summary[
    "whole_restricted"
]


if (
    low[
        "router_correct"
    ]
    >
    whole[
        "router_correct"
    ]
):

    selected_restricted = (
        "lowrank_restricted"
    )

    selection_reason = (
        "higher exact OOF accuracy"
    )


elif (
    whole[
        "router_correct"
    ]
    >
    low[
        "router_correct"
    ]
):

    selected_restricted = (
        "whole_restricted"
    )

    selection_reason = (
        "higher exact OOF accuracy"
    )


elif (
    low[
        "breaks"
    ]
    <
    whole[
        "breaks"
    ]
):

    selected_restricted = (
        "lowrank_restricted"
    )

    selection_reason = (
        "accuracy tied; fewer OOF breaks"
    )


elif (
    whole[
        "breaks"
    ]
    <
    low[
        "breaks"
    ]
):

    selected_restricted = (
        "whole_restricted"
    )

    selection_reason = (
        "accuracy tied; fewer OOF breaks"
    )


elif (
    low[
        "mean_selected_nonzero_gain"
    ]
    <
    whole[
        "mean_selected_nonzero_gain"
    ]
):

    selected_restricted = (
        "lowrank_restricted"
    )

    selection_reason = (
        "accuracy and breaks tied; "
        "smaller mean selected nonzero gain"
    )


elif (
    whole[
        "mean_selected_nonzero_gain"
    ]
    <
    low[
        "mean_selected_nonzero_gain"
    ]
):

    selected_restricted = (
        "whole_restricted"
    )

    selection_reason = (
        "accuracy and breaks tied; "
        "smaller mean selected nonzero gain"
    )


else:

    selected_restricted = (
        "lowrank_restricted"
    )

    selection_reason = (
        "all previous criteria tied; "
        "predeclared lowrank tie-break"
    )


comparator_obj = {
    "selected_restricted_comparator":
        selected_restricted,

    "selection_reason":
        selection_reason,

    "selection_rule": [
        "higher exact OOF accuracy",
        "fewer OOF breaks if tied",
        (
            "smaller mean selected nonzero gain "
            "if still tied"
        ),
        "lowrank_restricted if still tied",
    ],

    "lowrank_restricted": {
        "router_correct":
            low[
                "router_correct"
            ],

        "router_accuracy":
            low[
                "router_accuracy"
            ],

        "breaks":
            low[
                "breaks"
            ],

        "mean_selected_nonzero_gain":
            low[
                "mean_selected_nonzero_gain"
            ],
    },

    "whole_restricted": {
        "router_correct":
            whole[
                "router_correct"
            ],

        "router_accuracy":
            whole[
                "router_accuracy"
            ],

        "breaks":
            whole[
                "breaks"
            ],

        "mean_selected_nonzero_gain":
            whole[
                "mean_selected_nonzero_gain"
            ],
    },
}


COMPARATOR.write_text(
    json.dumps(
        comparator_obj,
        indent=2,
        sort_keys=True,
    )
    +
    "\n",
    encoding="utf-8",
)


# ============================================================
# FIT FINAL DEVELOPMENT ROUTERS
# ============================================================

bundle_info = {}


for family in [
    "lowrank_restricted",
    "whole_restricted",
    "unified",
]:

    fam = family_mask(
        cand,
        family,
    )


    model = new_model()

    model.fit(
        DESIGN[
            fam
        ],
        y[
            fam
        ],
    )


    bundle = {
        "version":
            "NATURAL_UGR_ROUTER_V1",

        "router_family":
            family,

        "allowed_methods":
            sorted(
                FAMILIES[
                    family
                ],
                key=lambda x:
                    METHOD_RANK[
                        x
                    ],
            ),

        "model":
            model,

        "sample_feature_names":
            sample_feature_names,

        "candidate_channels":
            CANDIDATE_CHANNELS,

        "design_feature_names":
            design_feature_names,

        "design_feature_count":
            479,

        "candidate_tie_break": [
            "IDENTITY",
            "smaller gain",
            "MN",
            "SA",
            "WHOLE",
        ],

        "C":
            C_VALUE,

        "solver":
            "lbfgs",

        "max_iter":
            MAX_ITER,

        "training_samples":
            EXPECTED_N,

        "training_candidate_rows":
            int(
                fam.sum()
            ),

        "protocol_sha256":
            sha256(
                PROTOCOL
            ),

        "candidate_outcomes_sha256":
            sha256(
                CANDIDATES
            ),

        "feature_matrix_sha256":
            sha256(
                FEATURE_MATRIX
            ),

        "fold_manifest_sha256":
            sha256(
                FOLD_MANIFEST
            ),

        "analysis_spec_sha256":
            sha256(
                ANALYSIS_SPEC
            ),

        "execution_freeze_sha256":
            sha256(
                EXECUTION_FREEZE
            ),
    }


    path = ROUTER_BUNDLES[
        family
    ]

    joblib.dump(
        bundle,
        path,
    )


    bundle_info[
        family
    ] = {
        "path":
            str(
                path
            ),

        "sha256":
            sha256(
                path
            ),

        "training_candidate_rows":
            int(
                fam.sum()
            ),
    }


BUNDLE_MANIFEST.write_text(
    json.dumps(
        {
            "version":
                "NATURAL_UGR_ROUTER_BUNDLE_MANIFEST_V1",

            "selected_restricted_comparator":
                selected_restricted,

            "routers":
                bundle_info,
        },
        indent=2,
        sort_keys=True,
    )
    +
    "\n",
    encoding="utf-8",
)


# ============================================================
# SUMMARY + METADATA
# ============================================================

summary_obj = {
    "experiment":
        "natural_ugr_v1",

    "development_population_n":
        EXPECTED_N,

    "baseline": {
        "correct":
            baseline_correct,

        "accuracy":
            baseline_acc,
    },

    "routers":
        family_summary,

    "selected_restricted_comparator":
        selected_restricted,

    "restricted_selection_reason":
        selection_reason,
}


SUMMARY.write_text(
    json.dumps(
        summary_obj,
        indent=2,
        sort_keys=True,
    )
    +
    "\n",
    encoding="utf-8",
)


metadata = {
    "experiment":
        "natural_ugr_dev_router_analysis_v1",

    "status":
        (
            "development OOF complete; "
            "three final development routers fit"
        ),

    "router_training_performed":
        True,

    "restricted_comparator_selected":
        True,

    "final_cohort_observed":
        False,

    "selected_restricted_comparator":
        selected_restricted,

    "sha256": {
        "protocol":
            sha256(
                PROTOCOL
            ),

        "candidate_outcomes":
            sha256(
                CANDIDATES
            ),

        "candidate_outcome_freeze":
            sha256(
                CANDIDATE_FREEZE
            ),

        "feature_matrix":
            sha256(
                FEATURE_MATRIX
            ),

        "feature_names":
            sha256(
                FEATURE_NAMES
            ),

        "feature_freeze":
            sha256(
                FEATURE_FREEZE
            ),

        "analysis_spec":
            sha256(
                ANALYSIS_SPEC
            ),

        "fold_manifest":
            sha256(
                FOLD_MANIFEST
            ),

        "execution_freeze":
            sha256(
                EXECUTION_FREEZE
            ),

        "oof_selections":
            sha256(
                OOF_SELECTIONS
            ),

        "summary":
            sha256(
                SUMMARY
            ),

        "selected_comparator":
            sha256(
                COMPARATOR
            ),

        "bundle_manifest":
            sha256(
                BUNDLE_MANIFEST
            ),

        "router_lowrank":
            bundle_info[
                "lowrank_restricted"
            ][
                "sha256"
            ],

        "router_whole":
            bundle_info[
                "whole_restricted"
            ][
                "sha256"
            ],

        "router_unified":
            bundle_info[
                "unified"
            ][
                "sha256"
            ],
    },
}


METADATA.write_text(
    json.dumps(
        metadata,
        indent=2,
        sort_keys=True,
    )
    +
    "\n",
    encoding="utf-8",
)


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 96)
print(
    "NATURAL UGR DEVELOPMENT ROUTER ANALYSIS COMPLETE"
)
print("=" * 96)

print(
    f"Baseline: "
    f"{baseline_correct}/{EXPECTED_N} "
    f"= {baseline_acc:.6f}"
)

print()


for family in [
    "lowrank_restricted",
    "whole_restricted",
    "unified",
]:

    s = family_summary[
        family
    ]

    print(
        f"{family}: "
        f"{s['router_correct']}/{EXPECTED_N} "
        f"= {s['router_accuracy']:.6f} "
        f"({s['router_minus_baseline_pp']:+.3f} pp vs baseline)"
    )

    print(
        f"  repairs={s['repairs']} "
        f"breaks={s['breaks']} "
        f"oracle={s['candidate_oracle_accuracy']:.6f}"
    )

    print(
        f"  selected methods="
        f"{s['selection_method_counts']}"
    )

    print(
        f"  mean selected nonzero gain="
        f"{s['mean_selected_nonzero_gain']:.6f}"
    )


print()

print(
    "SELECTED RESTRICTED COMPARATOR:",
    selected_restricted,
)

print(
    "Reason:",
    selection_reason,
)

print()

print(
    "OOF selections SHA256:",
    sha256(
        OOF_SELECTIONS
    )
)

print(
    "Summary SHA256:",
    sha256(
        SUMMARY
    )
)

print(
    "Comparator SHA256:",
    sha256(
        COMPARATOR
    )
)

print(
    "Bundle manifest SHA256:",
    sha256(
        BUNDLE_MANIFEST
    )
)

print(
    "Metadata SHA256:",
    sha256(
        METADATA
    )
)

print("=" * 96)
