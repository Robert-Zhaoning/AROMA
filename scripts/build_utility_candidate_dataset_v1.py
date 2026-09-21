import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# FROZEN INPUTS
# ============================================================

PROTOCOL = Path(
    "outputs/proc_count_sa_dev_v1/"
    "utility_router_protocol_v1/"
    "utility_router_protocol_v1.json"
)

EXPECTED_PROTOCOL_SHA256 = (
    "2bf3ba6504c8915c846c810adf2a6652e31caba9658"
    "ba4993fd1de4d4959c5c8"
)

FEATURES = Path(
    "outputs/proc_count_sa_dev_v1/"
    "utility_features_v1/"
    "sa_dev_utility_features_v1.csv"
)

SWEEP = Path(
    "outputs/proc_count_sa_dev_v1/"
    "sa_vs_mn_sweep_v1/"
    "sa_vs_mn_results.csv"
)

OUT = Path(
    "outputs/proc_count_sa_dev_v1/"
    "utility_candidate_dataset_v1"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

FINAL = (
    OUT
    / "utility_candidate_dataset_v1.csv"
)

METADATA = (
    OUT
    / "metadata.json"
)


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


# ============================================================
# PROTOCOL AUDIT
# ============================================================

if sha256(PROTOCOL) != EXPECTED_PROTOCOL_SHA256:

    raise RuntimeError(
        "Utility-router protocol SHA mismatch."
    )


protocol = json.loads(
    PROTOCOL.read_text(
        encoding="utf-8"
    )
)


expected_feature_sha = (
    protocol[
        "source_artifacts"
    ][
        "utility_features_sha256"
    ]
)

expected_sweep_sha = (
    protocol[
        "source_artifacts"
    ][
        "candidate_sweep_sha256"
    ]
)


if sha256(FEATURES) != expected_feature_sha:

    raise RuntimeError(
        "Utility-feature archive SHA mismatch."
    )


if sha256(SWEEP) != expected_sweep_sha:

    raise RuntimeError(
        "Candidate sweep SHA mismatch."
    )


print(
    "PASS: frozen protocol and source artifacts verified"
)


# ============================================================
# LOAD
# ============================================================

features = pd.read_csv(
    FEATURES
)

sweep = pd.read_csv(
    SWEEP
)


features[
    "sample_id"
] = (
    features[
        "sample_id"
    ].astype(str)
)

features[
    "cohort_uid"
] = (
    features[
        "cohort_uid"
    ].astype(str)
)

sweep[
    "sample_id"
] = (
    sweep[
        "sample_id"
    ].astype(str)
)


# ============================================================
# POPULATION = alpha > 1
# ============================================================

up = features[
    features[
        "selected_alpha"
    ].astype(float)
    > 1.0
].copy()


up = (
    up.sort_values(
        "sample_id"
    )
    .reset_index(
        drop=True
    )
)


if len(up) != 160:

    raise RuntimeError(
        f"Expected upward N=160, got {len(up)}"
    )


if up[
    "sample_id"
].nunique() != 160:

    raise RuntimeError(
        "Upward sample IDs not unique."
    )


print(
    "PASS: exact upward development population N=160"
)


# ============================================================
# FROZEN CANDIDATE GRID
# ============================================================

BETAS = [
    float(x)
    for x in protocol[
        "candidate_set"
    ][
        "beta_grid"
    ]
]


if BETAS != [
    0.50,
    0.75,
    1.00,
    1.25,
    1.50,
    1.75,
    2.00,
]:

    raise RuntimeError(
        f"Unexpected beta grid: {BETAS}"
    )


# ============================================================
# ALLOWED SAMPLE FEATURES
# ============================================================

ctrl_cols = sorted(
    [
        c
        for c in features.columns
        if c.startswith(
            "ctrl__"
        )
    ]
)


if len(ctrl_cols) != 39:

    raise RuntimeError(
        f"Expected 39 controller features; "
        f"got {len(ctrl_cols)}"
    )


mechanistic_cols = [
    "pg_norm",
    "h_norm",
    "sa_sensitivity",
    "cos_pug_puh",
    "gradient_capture",
    "activation_capture",
]


for c in mechanistic_cols:

    if c not in features.columns:

        raise RuntimeError(
            f"Missing mechanistic feature: {c}"
        )


# Ground-truth-dependent columns are explicitly forbidden
# from the model feature set.
forbidden_model_features = {
    "ground_truth",
    "baseline_correct",
    "candidate_prediction",
    "candidate_correct",
}


# ============================================================
# BUILD 15 CANDIDATES PER SAMPLE
# ============================================================

rows = []


for _, sample in up.iterrows():

    sid = str(
        sample[
            "sample_id"
        ]
    )

    uid = str(
        sample[
            "cohort_uid"
        ]
    )

    gt = int(
        sample[
            "ground_truth"
        ]
    )

    baseline_pred = int(
        sample[
            "baseline_prediction"
        ]
    )


    # --------------------------------------------------------
    # Shared GT-free sample feature payload
    # --------------------------------------------------------

    base = {
        "sample_id":
            sid,

        "cohort_uid":
            uid,

        "selected_alpha":
            float(
                sample[
                    "selected_alpha"
                ]
            ),

        "selected_score":
            float(
                sample[
                    "selected_score"
                ]
            ),
    }


    for c in ctrl_cols:

        base[c] = float(
            sample[c]
        )


    # Log transforms are defined prospectively here
    # only from GT-free mechanistic quantities.

    pg = float(
        sample[
            "pg_norm"
        ]
    )

    hn = float(
        sample[
            "h_norm"
        ]
    )

    sens = float(
        sample[
            "sa_sensitivity"
        ]
    )


    if (
        pg <= 0
        or hn <= 0
        or sens <= 0
    ):

        raise RuntimeError(
            f"{sid}: non-positive mechanistic norm."
        )


    base[
        "mech__log_pg_norm"
    ] = float(
        np.log10(pg)
    )

    base[
        "mech__log_h_norm"
    ] = float(
        np.log10(hn)
    )

    base[
        "mech__log_sa_sensitivity"
    ] = float(
        np.log10(sens)
    )

    base[
        "mech__cos_pug_puh"
    ] = float(
        sample[
            "cos_pug_puh"
        ]
    )

    base[
        "mech__gradient_capture"
    ] = float(
        sample[
            "gradient_capture"
        ]
    )

    base[
        "mech__activation_capture"
    ] = float(
        sample[
            "activation_capture"
        ]
    )


    # --------------------------------------------------------
    # Candidate 0: identity / abstain
    # --------------------------------------------------------

    identity = dict(
        base
    )

    identity.update(
        {
            "candidate_id":
                "IDENTITY",

            "candidate_method":
                "IDENTITY",

            "candidate_beta":
                0.0,

            "candidate_beta_sq":
                0.0,

            "candidate_is_identity":
                1,

            "candidate_is_mn":
                0,

            "candidate_is_sa":
                0,

            # supervision only
            "ground_truth":
                gt,

            "candidate_prediction":
                baseline_pred,

            "candidate_correct":
                int(
                    baseline_pred
                    == gt
                ),
        }
    )

    rows.append(
        identity
    )


    # --------------------------------------------------------
    # MN / SA intervention candidates
    # --------------------------------------------------------

    for method in [
        "MN",
        "SA",
    ]:

        for beta in BETAS:

            candidate = sweep[
                (
                    sweep[
                        "sample_id"
                    ]
                    == sid
                )
                &
                (
                    sweep[
                        "method"
                    ]
                    == method
                )
                &
                np.isclose(
                    sweep[
                        "beta"
                    ].astype(float),
                    beta,
                )
            ]


            if len(candidate) != 1:

                raise RuntimeError(
                    f"{sid}: expected exactly one "
                    f"{method} beta={beta} row; "
                    f"got {len(candidate)}"
                )


            r = candidate.iloc[0]


            # Cross-artifact audits.
            if int(
                r[
                    "ground_truth"
                ]
            ) != gt:

                raise RuntimeError(
                    f"{sid}: GT mismatch across artifacts."
                )


            if not np.isclose(
                float(
                    r[
                        "selected_alpha"
                    ]
                ),
                float(
                    sample[
                        "selected_alpha"
                    ]
                ),
                atol=1e-12,
                rtol=0.0,
            ):

                raise RuntimeError(
                    f"{sid}: alpha mismatch across artifacts."
                )


            pred = int(
                r[
                    "prediction"
                ]
            )


            item = dict(
                base
            )


            item.update(
                {
                    "candidate_id":
                        f"{method}_b{beta:.2f}",

                    "candidate_method":
                        method,

                    "candidate_beta":
                        float(
                            beta
                        ),

                    "candidate_beta_sq":
                        float(
                            beta ** 2
                        ),

                    "candidate_is_identity":
                        0,

                    "candidate_is_mn":
                        int(
                            method
                            == "MN"
                        ),

                    "candidate_is_sa":
                        int(
                            method
                            == "SA"
                        ),

                    # supervision only
                    "ground_truth":
                        gt,

                    "candidate_prediction":
                        pred,

                    "candidate_correct":
                        int(
                            pred
                            == gt
                        ),
                }
            )


            rows.append(
                item
            )


# ============================================================
# FINAL TABLE
# ============================================================

df = pd.DataFrame(
    rows
)


EXPECTED_ROWS = (
    160
    *
    15
)


if len(df) != EXPECTED_ROWS:

    raise RuntimeError(
        f"Expected {EXPECTED_ROWS} rows; "
        f"got {len(df)}"
    )


counts = (
    df.groupby(
        "sample_id"
    )
    .size()
)


if not np.all(
    counts.to_numpy()
    == 15
):

    raise RuntimeError(
        "Not every sample has exactly "
        "15 candidate rows."
    )


if df[
    [
        "sample_id",
        "candidate_id",
    ]
].duplicated().any():

    raise RuntimeError(
        "Duplicate sample/candidate pair."
    )


# ============================================================
# NUMERIC FINITE AUDIT
# ============================================================

model_sample_cols = (
    ctrl_cols
    +
    [
        "selected_alpha",
        "selected_score",
        "mech__log_pg_norm",
        "mech__log_h_norm",
        "mech__log_sa_sensitivity",
        "mech__cos_pug_puh",
        "mech__gradient_capture",
        "mech__activation_capture",
    ]
)


candidate_cols = [
    "candidate_beta",
    "candidate_beta_sq",
    "candidate_is_identity",
    "candidate_is_mn",
    "candidate_is_sa",
]


model_feature_cols = (
    model_sample_cols
    +
    candidate_cols
)


if (
    forbidden_model_features
    &
    set(
        model_feature_cols
    )
):

    raise RuntimeError(
        "GT-dependent feature leakage detected."
    )


if not np.isfinite(
    df[
        model_feature_cols
    ].to_numpy(
        dtype=np.float64
    )
).all():

    raise RuntimeError(
        "Non-finite model feature."
    )


# ============================================================
# LABEL / ORACLE AUDIT
# ============================================================

identity = df[
    df[
        "candidate_method"
    ]
    == "IDENTITY"
]


if len(identity) != 160:

    raise RuntimeError(
        "Identity candidate count mismatch."
    )


oracle_by_sample = (
    df.groupby(
        "sample_id"
    )[
        "candidate_correct"
    ]
    .max()
)


oracle_correct = int(
    oracle_by_sample.sum()
)


# intervention-only oracle
intervention_oracle = (
    df[
        df[
            "candidate_method"
        ]
        != "IDENTITY"
    ]
    .groupby(
        "sample_id"
    )[
        "candidate_correct"
    ]
    .max()
)


intervention_oracle_correct = int(
    intervention_oracle.sum()
)


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    FINAL,
    index=False,
)


metadata = {
    "experiment":
        "utility_candidate_dataset_v1",

    "population_n_samples":
        160,

    "candidates_per_sample":
        15,

    "rows":
        len(df),

    "candidate_definition":
        "1 identity + MN/SA x 7 beta values",

    "controller_feature_count":
        len(
            ctrl_cols
        ),

    "mechanistic_feature_count":
        6,

    "model_feature_columns":
        model_feature_cols,

    "supervision_columns": [
        "ground_truth",
        "candidate_prediction",
        "candidate_correct",
    ],

    "leakage_policy":
        "supervision columns excluded from model inputs",

    "identity_correct":
        int(
            identity[
                "candidate_correct"
            ].sum()
        ),

    "intervention_oracle_correct":
        intervention_oracle_correct,

    "full_oracle_with_identity_correct":
        oracle_correct,

    "protocol_sha256":
        sha256(
            PROTOCOL
        ),

    "utility_features_sha256":
        sha256(
            FEATURES
        ),

    "candidate_sweep_sha256":
        sha256(
            SWEEP
        ),

    "candidate_dataset_sha256":
        sha256(
            FINAL
        ),
}


METADATA.write_text(
    json.dumps(
        metadata,
        indent=2,
    ),
    encoding="utf-8",
)


# ============================================================
# PRINT
# ============================================================

print()
print("=" * 92)
print(
    "UTILITY CANDIDATE DATASET COMPLETE"
)
print("=" * 92)

print(
    "Independent samples:",
    160,
)

print(
    "Candidates/sample:",
    15,
)

print(
    "Total rows:",
    len(df),
)

print(
    "Controller features:",
    len(ctrl_cols),
)

print(
    "GT-free model features:",
    len(
        model_feature_cols
    ),
)

print()
print(
    "Identity correct:",
    f"{int(identity['candidate_correct'].sum())}/160",
)

print(
    "Intervention-only oracle:",
    f"{intervention_oracle_correct}/160",
)

print(
    "Full oracle incl. identity:",
    f"{oracle_correct}/160",
)

print()
print(
    "Candidate label rate:",
    f"{df['candidate_correct'].mean():.6f}",
)

print()
print(
    "Dataset:",
    FINAL,
)

print(
    "Dataset SHA256:",
    sha256(
        FINAL
    ),
)

print(
    "Metadata SHA256:",
    sha256(
        METADATA
    ),
)

print("=" * 92)
