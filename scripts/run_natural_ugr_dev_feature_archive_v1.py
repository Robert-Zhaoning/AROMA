#!/usr/bin/env python3

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")

import natural_ugr_stage2_common_v1 as C


EXPECTED_N = 292

DEV = Path(
    "manifests/natural_ugr_v1/"
    "development_population_v1.csv"
)

PROTOCOL = Path(
    "manifests/natural_ugr_v1/"
    "protocol_v1.json"
)

MAPPING = Path(
    "manifests/natural_ugr_v1/"
    "controller_feature_mapping_v1.json"
)

MECH = Path(
    "outputs/natural_ugr_v1/"
    "development/stage2/"
    "mechanistic_collection_v1/"
    "mechanistic_summary_v1.csv"
)

OUT = Path(
    "outputs/natural_ugr_v1/"
    "development/stage2/"
    "feature_archive_v1"
)

MECH_NAMES = [
    "mech__log_pg_norm",
    "mech__log_h_norm",
    "mech__log_sa_sensitivity",
    "mech__cos_pug_puh",
    "mech__gradient_capture",
    "mech__activation_capture",
]


def main():

    if not MECH.is_file():
        raise FileNotFoundError(
            MECH
        )

    finals = [
        OUT
        /
        "stage2_sample_features_v1.csv",

        OUT
        /
        "stage2_sample_feature_matrix_v1.npy",

        OUT
        /
        "stage2_sample_feature_names_v1.json",

        OUT
        /
        "metadata.json",
    ]

    if any(
        p.exists()
        for p in finals
    ):
        raise RuntimeError(
            "Final feature artifact already exists."
        )

    dev = pd.read_csv(
        DEV
    )

    mech = pd.read_csv(
        MECH
    )

    protocol = json.loads(
        PROTOCOL.read_text()
    )

    mapping = json.loads(
        MAPPING.read_text()
    )

    if len(dev) != EXPECTED_N:
        raise RuntimeError(
            "Development N mismatch."
        )

    if len(mech) != EXPECTED_N:
        raise RuntimeError(
            "Mechanistic N mismatch."
        )

    raw_ids = np.array(
        [
            str(
                int(x)
            )
            for x in dev[
                "question_id"
            ]
        ],
        dtype=str,
    )

    uids = (
        dev[
            "natural_ugr_uid"
        ]
        .astype(str)
        .to_numpy()
    )

    if not np.array_equal(
        mech[
            "raw_sample_id"
        ]
        .astype(str)
        .to_numpy(),
        raw_ids,
    ):
        raise RuntimeError(
            "Dev/mechanistic sample ordering mismatch."
        )

    if not np.array_equal(
        mech[
            "cohort_uid"
        ]
        .astype(str)
        .to_numpy(),
        uids,
    ):
        raise RuntimeError(
            "Dev/mechanistic UID ordering mismatch."
        )

    hist_order = list(
        mapping[
            "historical_order"
        ]
    )

    hist_to_nat = dict(
        mapping[
            "mapping_historical_to_natural"
        ]
    )

    if len(
        hist_order
    ) != 39:
        raise RuntimeError(
            "Controller feature count != 39."
        )

    sample_feature_names = list(
        protocol[
            "sample_features"
        ][
            "feature_order"
        ]
    )

    expected_names = (
        hist_order
        +
        [
            "selected_alpha",
            "selected_score",
        ]
        +
        MECH_NAMES
    )

    if (
        sample_feature_names
        !=
        expected_names
    ):
        raise RuntimeError(
            "Frozen 47-feature order mismatch."
        )

    rows = []

    for idx in range(
        EXPECTED_N
    ):

        d = dev.iloc[
            idx
        ]

        m = mech.iloc[
            idx
        ]

        record = {
            "raw_sample_id":
                raw_ids[
                    idx
                ],

            "cohort_uid":
                uids[
                    idx
                ],

            "question_id":
                int(
                    d[
                        "question_id"
                    ]
                ),

            "image_id":
                str(
                    d[
                        "image_id"
                    ]
                ),

            "baseline_prediction":
                int(
                    d[
                        "baseline_prediction"
                    ]
                ),
        }

        # Exact F1-frozen natural Stage-1 feature source.
        for hist_name in hist_order:

            nat_name = hist_to_nat[
                hist_name
            ]

            record[
                hist_name
            ] = float(
                d[
                    nat_name
                ]
            )

        record[
            "selected_alpha"
        ] = float(
            d[
                "selected_alpha"
            ]
        )

        record[
            "selected_score"
        ] = float(
            d[
                "selected_score"
            ]
        )

        for name in MECH_NAMES:

            record[
                name
            ] = float(
                m[
                    name
                ]
            )

        rows.append(
            record
        )

    df = pd.DataFrame(
        rows
    )

    X47 = (
        df[
            sample_feature_names
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    if X47.shape != (
        EXPECTED_N,
        47,
    ):
        raise RuntimeError(
            f"Unexpected X47 shape: {X47.shape}"
        )

    if not np.isfinite(
        X47
    ).all():
        raise RuntimeError(
            "Non-finite X47."
        )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUT
        /
        "stage2_sample_features_v1.csv",
        index=False,
    )

    np.save(
        OUT
        /
        "stage2_sample_feature_matrix_v1.npy",
        X47,
    )

    (
        OUT
        /
        "stage2_sample_feature_names_v1.json"
    ).write_text(
        json.dumps(
            {
                "feature_count":
                    47,

                "controller_feature_count":
                    39,

                "mechanistic_feature_count":
                    6,

                "controller_feature_source":
                    "frozen_natural_stage1_archive",

                "feature_names":
                    sample_feature_names,
            },
            indent=2,
        )
        +
        "\n"
    )

    metadata = {
        "experiment":
            "natural_ugr_dev_stage2_feature_archive_v1",

        "population_n":
            EXPECTED_N,

        "feature_count":
            47,

        "controller_feature_count":
            39,

        "controller_feature_source":
            "frozen_natural_stage1_archive",

        "ground_truth_used_in_feature_construction":
            False,

        "candidate_outcomes_used_in_feature_construction":
            False,

        "feature_csv_sha256":
            C.sha256_file(
                OUT
                /
                "stage2_sample_features_v1.csv"
            ),

        "feature_matrix_sha256":
            C.sha256_file(
                OUT
                /
                "stage2_sample_feature_matrix_v1.npy"
            ),

        "feature_names_sha256":
            C.sha256_file(
                OUT
                /
                "stage2_sample_feature_names_v1.json"
            ),
    }

    (
        OUT
        /
        "metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
        )
        +
        "\n"
    )

    print(
        "NATURAL UGR 47-FEATURE ARCHIVE: PASS"
    )

    print(
        "X47:",
        X47.shape,
    )

    print(
        "Controller source: FROZEN NATURAL ARCHIVE"
    )

    print(
        "GT used: False"
    )

    print(
        "Candidate outcomes used: False"
    )


if __name__ == "__main__":
    main()
