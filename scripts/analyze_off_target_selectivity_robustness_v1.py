#!/usr/bin/env python3

"""
AROMA C2 post-freeze inferential robustness analysis.

IMPORTANT:
- Does NOT replace the frozen primary analysis.
- Primary remains:
    20,000 paired bootstrap replicates
    seed = 20260922
- This script evaluates Monte-Carlo / inferential stability only.

Analyses:
1. 200,000 paired sample-level bootstrap replicates.
2. Bootstrap stability across several fixed seeds.
3. Exact / Monte-Carlo paired sign-flip randomization test
   on the N=1000 sample-level burden differences.
"""

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


SAMPLE_FILE = Path(
    "artifacts/off_target_selectivity_v1/"
    "endpoint_analysis_v1/"
    "sample_break_burdens_v1.csv"
)

PRIMARY_FILE = Path(
    "artifacts/off_target_selectivity_v1/"
    "endpoint_analysis_v1/"
    "primary_summary_v1.json"
)

EXPECTED_SAMPLE_SHA = (
    "5518f9dd35eb632ddbe7d0486d8ceb1d79a40a43aa69441f1a7459b34c5bc5e8"
)

EXPECTED_PRIMARY_SHA = (
    "3a6fbb3ed626148099ebd63d26ae7b6038a562a7540436a084b26f12bbb39d62"
)

OUT_DIR = Path(
    "artifacts/off_target_selectivity_v1/"
    "robustness_inference_v1"
)

N = 1000

LARGE_BOOT_REPS = 200_000

STABILITY_REPS = 100_000

STABILITY_SEEDS = [
    20260922,
    11,
    101,
    1009,
    10007,
]

RANDOMIZATION_REPS = 1_000_000
RANDOMIZATION_SEED = 20260926


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def bootstrap_ci(
    x,
    reps,
    seed,
    batch_size=1000,
):
    x = np.asarray(
        x,
        dtype=np.float64,
    )

    rng = np.random.default_rng(
        seed
    )

    vals = np.empty(
        reps,
        dtype=np.float64,
    )

    pos = 0

    while pos < reps:

        m = min(
            batch_size,
            reps - pos,
        )

        idx = rng.integers(
            0,
            len(x),
            size=(
                m,
                len(x),
            ),
        )

        vals[
            pos:
            pos + m
        ] = (
            x[idx]
            .mean(axis=1)
        )

        pos += m

    lo, hi = np.quantile(
        vals,
        [
            0.025,
            0.975,
        ],
    )

    return {
        "mean":
            float(
                x.mean()
            ),

        "ci_lo":
            float(lo),

        "ci_hi":
            float(hi),

        "bootstrap_prob_effect_lt_zero":
            float(
                (
                    vals < 0
                ).mean()
            ),

        "bootstrap_prob_effect_gt_zero":
            float(
                (
                    vals > 0
                ).mean()
            ),
    }


def sign_flip_test(
    x,
    reps,
    seed,
    batch_size=5000,
):
    """
    Paired randomization/sign-flip test.

    Null:
        method labels are exchangeable within sample,
        so each sample-level paired difference can
        independently reverse sign.

    Two-sided test statistic:
        abs(mean paired difference)
    """

    x = np.asarray(
        x,
        dtype=np.float64,
    )

    observed = float(
        x.mean()
    )

    rng = np.random.default_rng(
        seed
    )

    extreme = 0
    total = 0

    pos = 0

    while pos < reps:

        m = min(
            batch_size,
            reps - pos,
        )

        signs = rng.integers(
            0,
            2,
            size=(
                m,
                len(x),
            ),
            dtype=np.int8,
        )

        signs = (
            signs * 2 - 1
        )

        null_means = (
            signs
            *
            x[None, :]
        ).mean(
            axis=1
        )

        extreme += int(
            (
                np.abs(
                    null_means
                )
                >=
                abs(
                    observed
                )
            ).sum()
        )

        total += m
        pos += m

    # +1 correction for Monte-Carlo randomization p.
    p_two = (
        extreme + 1
    ) / (
        total + 1
    )

    return {
        "observed_effect":
            observed,

        "randomization_replicates":
            total,

        "seed":
            seed,

        "extreme_replicates":
            extreme,

        "two_sided_p":
            float(
                p_two
            ),
    }


def main():

    print(
        "=" * 72
    )
    print(
        "C2 — POST-FREEZE INFERENTIAL ROBUSTNESS"
    )
    print(
        "PRIMARY RESULT IS NOT REPLACED"
    )
    print(
        "=" * 72
    )

    if sha256(
        SAMPLE_FILE
    ) != EXPECTED_SAMPLE_SHA:
        raise RuntimeError(
            "Sample burden artifact SHA mismatch."
        )

    if sha256(
        PRIMARY_FILE
    ) != EXPECTED_PRIMARY_SHA:
        raise RuntimeError(
            "Primary summary SHA mismatch."
        )

    df = pd.read_csv(
        SAMPLE_FILE
    )

    if len(df) != N:
        raise RuntimeError(
            f"Expected N={N}, got {len(df)}"
        )

    x = (
        df[
            "mn_minus_whole"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    observed = float(
        x.mean()
    )

    print()
    print(
        "Observed MN-WHOLE:",
        f"{100*observed:+.6f} pp",
    )

    print()
    print(
        "===== 200,000-REPLICATE BOOTSTRAP ====="
    )

    large = bootstrap_ci(
        x,
        LARGE_BOOT_REPS,
        20260922,
    )

    print(
        "effect:",
        f"{100*large['mean']:+.6f} pp",
    )

    print(
        "95% CI:",
        f"[{100*large['ci_lo']:+.6f}, "
        f"{100*large['ci_hi']:+.6f}] pp",
    )

    print(
        "bootstrap P(effect < 0):",
        f"{large['bootstrap_prob_effect_lt_zero']:.6f}",
    )

    print()
    print(
        "===== SEED STABILITY ====="
    )

    stability = []

    for seed in STABILITY_SEEDS:

        r = bootstrap_ci(
            x,
            STABILITY_REPS,
            seed,
        )

        stability.append(
            {
                "seed":
                    seed,

                **r,
            }
        )

        print(
            f"seed={seed:8d}  "
            f"CI=["
            f"{100*r['ci_lo']:+.6f}, "
            f"{100*r['ci_hi']:+.6f}] pp  "
            f"P(Δ<0)="
            f"{r['bootstrap_prob_effect_lt_zero']:.5f}"
        )

    print()
    print(
        "===== PAIRED SIGN-FLIP RANDOMIZATION TEST ====="
    )

    randomization = sign_flip_test(
        x,
        RANDOMIZATION_REPS,
        RANDOMIZATION_SEED,
    )

    print(
        "observed:",
        f"{100*randomization['observed_effect']:+.6f} pp",
    )

    print(
        "replicates:",
        randomization[
            "randomization_replicates"
        ],
    )

    print(
        "two-sided p:",
        f"{randomization['two_sided_p']:.8f}",
    )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(
        stability
    ).to_csv(
        OUT_DIR /
        "bootstrap_seed_stability_v1.csv",
        index=False,
        lineterminator="\n",
    )

    summary = {
        "role":
            (
                "post-freeze inferential robustness; "
                "does not replace frozen primary analysis"
            ),

        "observed_mn_minus_whole":
            observed,

        "observed_mn_minus_whole_pp":
            100.0
            *
            observed,

        "large_bootstrap": {
            "replicates":
                LARGE_BOOT_REPS,

            "seed":
                20260922,

            **large,
        },

        "seed_stability": {
            "replicates_per_seed":
                STABILITY_REPS,

            "seeds":
                STABILITY_SEEDS,

            "results":
                stability,
        },

        "paired_sign_flip_randomization":
            randomization,
    }

    (
        OUT_DIR /
        "robustness_summary_v1.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        )
        +
        "\n",
        encoding="utf-8",
    )

    print()
    print(
        "===== OUTPUT HASHES ====="
    )

    for p in sorted(
        OUT_DIR.iterdir()
    ):
        print(
            p,
            sha256(p),
        )

    print()
    print(
        "=" * 72
    )
    print(
        "C2 ROBUSTNESS INFERENCE COMPLETE"
    )
    print(
        "=" * 72
    )


if __name__ == "__main__":
    main()
