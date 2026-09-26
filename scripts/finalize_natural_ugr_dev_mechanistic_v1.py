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

U4_PATH = Path(
    "outputs/aroma2/csa_gate_a_v3/"
    "U4_primary.npy"
)

GRAD = Path(
    "outputs/natural_ugr_v1/"
    "development/stage2/"
    "gradient_collection_v1"
)

OUT = Path(
    "outputs/natural_ugr_v1/"
    "development/stage2/"
    "mechanistic_collection_v1"
)

EXPECTED_U4_SHA256 = (
    "af50da2cf49268cc55dc33d44dad50c0cd5a4fa29cda6cad3c01cc0b4ae63f14"
)


def main():

    required = [
        GRAD / "gradient_matrix.npy",
        GRAD / "pooled_h_matrix.npy",
        GRAD / "sample_ids.npy",
        GRAD / "cohort_uids.npy",
        GRAD / "token_lengths.npy",
        GRAD / "mu.npy",
    ]

    for p in required:
        if not p.is_file():
            raise FileNotFoundError(
                p
            )

    if C.sha256_file(
        U4_PATH
    ) != EXPECTED_U4_SHA256:
        raise RuntimeError(
            "U4 hash mismatch."
        )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    finals = [
        OUT / "mn_directions.npy",
        OUT / "sa_directions.npy",
        OUT / "mechanistic_summary_v1.csv",
        OUT / "metadata.json",
    ]

    if any(
        p.exists()
        for p in finals
    ):
        raise RuntimeError(
            "Mechanistic final artifact already exists."
        )

    dev = pd.read_csv(
        DEV
    )

    G = np.load(
        GRAD
        /
        "gradient_matrix.npy"
    ).astype(
        np.float64
    )

    H = np.load(
        GRAD
        /
        "pooled_h_matrix.npy"
    ).astype(
        np.float64
    )

    sample_ids = np.load(
        GRAD
        /
        "sample_ids.npy",
        allow_pickle=True,
    ).astype(str)

    cohort_uids = np.load(
        GRAD
        /
        "cohort_uids.npy",
        allow_pickle=True,
    ).astype(str)

    token_lengths = np.load(
        GRAD
        /
        "token_lengths.npy"
    ).astype(
        np.int64
    )

    mu = np.load(
        GRAD
        /
        "mu.npy"
    ).astype(
        np.float64
    )

    U4 = np.load(
        U4_PATH
    ).astype(
        np.float64
    )

    if G.shape != (
        EXPECTED_N,
        128,
    ):
        raise RuntimeError(
            f"Bad G shape: {G.shape}"
        )

    if H.shape != (
        EXPECTED_N,
        128,
    ):
        raise RuntimeError(
            f"Bad H shape: {H.shape}"
        )

    if U4.shape != (
        128,
        4,
    ):
        raise RuntimeError(
            f"Bad U4 shape: {U4.shape}"
        )

    expected_ids = np.array(
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

    expected_uids = (
        dev[
            "natural_ugr_uid"
        ]
        .astype(str)
        .to_numpy()
    )

    if not np.array_equal(
        sample_ids,
        expected_ids,
    ):
        raise RuntimeError(
            "Gradient/dev sample ordering mismatch."
        )

    if not np.array_equal(
        cohort_uids,
        expected_uids,
    ):
        raise RuntimeError(
            "Gradient/dev UID ordering mismatch."
        )

    PG = (
        G
        @
        U4
        @
        U4.T
    )

    PH = (
        H
        @
        U4
        @
        U4.T
    )

    g_norm = np.linalg.norm(
        G,
        axis=1,
    )

    h_norm = np.linalg.norm(
        H,
        axis=1,
    )

    pg_norm = np.linalg.norm(
        PG,
        axis=1,
    )

    ph_norm = np.linalg.norm(
        PH,
        axis=1,
    )

    for name, values in [
        ("g_norm", g_norm),
        ("h_norm", h_norm),
        ("pg_norm", pg_norm),
        ("ph_norm", ph_norm),
    ]:

        if not np.isfinite(
            values
        ).all():
            raise RuntimeError(
                f"Non-finite {name}."
            )

        if float(
            values.min()
        ) <= 1e-12:
            raise RuntimeError(
                f"{name} too small."
            )

    MN = (
        PH
        /
        ph_norm[
            :,
            None,
        ]
    )

    SA = (
        PG
        /
        pg_norm[
            :,
            None,
        ]
    )

    if not np.allclose(
        np.linalg.norm(
            MN,
            axis=1,
        ),
        1.0,
        atol=1e-10,
        rtol=0.0,
    ):
        raise RuntimeError(
            "MN normalization failure."
        )

    if not np.allclose(
        np.linalg.norm(
            SA,
            axis=1,
        ),
        1.0,
        atol=1e-10,
        rtol=0.0,
    ):
        raise RuntimeError(
            "SA normalization failure."
        )

    den = (
        pg_norm
        *
        ph_norm
    )

    cos_pug_puh = np.divide(
        np.sum(
            PG
            *
            PH,
            axis=1,
        ),
        den,
        out=np.zeros(
            EXPECTED_N,
            dtype=np.float64,
        ),
        where=
            den > 1e-30,
    )

    gradient_capture = np.divide(
        pg_norm,
        g_norm,
        out=np.zeros_like(
            pg_norm
        ),
        where=
            g_norm > 1e-30,
    )

    activation_capture = np.divide(
        ph_norm,
        h_norm,
        out=np.zeros_like(
            ph_norm
        ),
        where=
            h_norm > 1e-30,
    )

    # Exact historically recovered definition.
    sa_sensitivity = (
        h_norm
        *
        pg_norm
    )

    if float(
        sa_sensitivity.min()
    ) <= 0:
        raise RuntimeError(
            "Non-positive SA sensitivity."
        )

    summary = pd.DataFrame(
        {
            "raw_sample_id":
                sample_ids,

            "cohort_uid":
                cohort_uids,

            "selected_alpha":
                dev[
                    "selected_alpha"
                ]
                .astype(float)
                .to_numpy(),

            "selected_score":
                dev[
                    "selected_score"
                ]
                .astype(float)
                .to_numpy(),

            "mu":
                mu,

            "token_length":
                token_lengths,

            "g_norm":
                g_norm,

            "h_norm":
                h_norm,

            "pg_norm":
                pg_norm,

            "ph_norm":
                ph_norm,

            "sa_sensitivity":
                sa_sensitivity,

            "cos_pug_puh":
                cos_pug_puh,

            "gradient_capture":
                gradient_capture,

            "activation_capture":
                activation_capture,

            "mech__log_pg_norm":
                np.log10(
                    pg_norm
                ),

            "mech__log_h_norm":
                np.log10(
                    h_norm
                ),

            "mech__log_sa_sensitivity":
                np.log10(
                    sa_sensitivity
                ),

            "mech__cos_pug_puh":
                cos_pug_puh,

            "mech__gradient_capture":
                gradient_capture,

            "mech__activation_capture":
                activation_capture,
        }
    )

    if summary.shape != (
        EXPECTED_N,
        20,
    ):
        raise RuntimeError(
            f"Unexpected summary shape: "
            f"{summary.shape}"
        )

    numeric = (
        summary
        .select_dtypes(
            include=[
                np.number
            ]
        )
        .to_numpy(
            dtype=np.float64
        )
    )

    if not np.isfinite(
        numeric
    ).all():
        raise RuntimeError(
            "Non-finite mechanistic summary."
        )

    np.save(
        OUT
        /
        "mn_directions.npy",
        MN.astype(
            np.float64
        ),
    )

    np.save(
        OUT
        /
        "sa_directions.npy",
        SA.astype(
            np.float64
        ),
    )

    summary.to_csv(
        OUT
        /
        "mechanistic_summary_v1.csv",
        index=False,
    )

    metadata = {
        "experiment":
            "natural_ugr_dev_mechanistic_v1",

        "population_n":
            EXPECTED_N,

        "ground_truth_used":
            False,

        "U4_sha256":
            C.sha256_file(
                U4_PATH
            ),

        "definitions": {
            "MN":
                "normalize(P_U h)",

            "SA":
                "normalize(P_U g)",

            "sa_sensitivity":
                "||h||_2 * ||P_U g||_2",

            "gradient_capture":
                "||P_U g||_2 / ||g||_2",

            "activation_capture":
                "||P_U h||_2 / ||h||_2",

            "logs":
                "base-10",
        },
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
        "NATURAL UGR MECHANISTIC FINALIZATION: PASS"
    )

    print(
        "MN:",
        MN.shape,
    )

    print(
        "SA:",
        SA.shape,
    )

    print(
        "summary:",
        summary.shape,
    )


if __name__ == "__main__":
    main()
