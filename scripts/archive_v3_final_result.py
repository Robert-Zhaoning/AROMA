import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


OUT_DIR = Path(
    "outputs/proc_count_causal_v3/"
    "final_frozen_controller"
)

RESULT_PATH = (
    OUT_DIR /
    "v3_final_results.csv"
)

SUMMARY_PATH = (
    OUT_DIR /
    "v3_final_summary.csv"
)

CONDITION_PATH = (
    OUT_DIR /
    "v3_final_by_condition.csv"
)

COUNT_PATH = (
    OUT_DIR /
    "v3_final_by_count.csv"
)

ACTION_PATH = (
    OUT_DIR /
    "v3_final_action_distribution.csv"
)

RUN_META_PATH = (
    OUT_DIR /
    "v3_final_run_metadata.json"
)

RUNNER_PATH = Path(
    "scripts/"
    "run_proc_count_causal_v3_final.py"
)

CONTROLLER_MANIFEST = Path(
    "configs/"
    "aroma_cardinality_controller_frozen.json"
)

V3_PROTOCOL = Path(
    "configs/"
    "proc_count_causal_v3_final_confirmation.json"
)

RUNNER_PROTOCOL = Path(
    "configs/"
    "proc_count_causal_v3_final_runner.json"
)

ARCHIVE_PATH = Path(
    "configs/"
    "aroma_v3_final_result_manifest.json"
)


CONTROLLER_FREEZE = (
    "c8c5ede1601b1ada4087e141d61247903a4458a7"
)

V3_PROTOCOL_FREEZE = (
    "582bb53fe51d8b8bbeb45b90b7d073253d4df2a1"
)

RUNNER_FREEZE = (
    "9085d093a5d2a46993e5dc8576de9e50a9c21c55"
)


def sha256_file(path):

    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(
                1024 * 1024
            )

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def git_head():

    return subprocess.check_output(
        [
            "git",
            "rev-parse",
            "HEAD",
        ],
        text=True,
    ).strip()


def exact_mcnemar(
    repairs,
    breaks,
):

    n = (
        repairs
        +
        breaks
    )

    k = min(
        repairs,
        breaks,
    )

    lower = sum(
        math.comb(
            n,
            i,
        )
        for i in range(
            k + 1
        )
    ) / (
        2 ** n
    )

    return min(
        1.0,
        2.0 * lower,
    )


def main():

    required = [
        RESULT_PATH,
        SUMMARY_PATH,
        CONDITION_PATH,
        COUNT_PATH,
        ACTION_PATH,
        RUN_META_PATH,
        RUNNER_PATH,
        CONTROLLER_MANIFEST,
        V3_PROTOCOL,
        RUNNER_PROTOCOL,
    ]

    missing = [
        str(p)
        for p in required
        if not p.exists()
    ]

    if missing:
        raise RuntimeError(
            "Missing required files:\n"
            +
            "\n".join(
                missing
            )
        )

    results = pd.read_csv(
        RESULT_PATH
    )

    summary = pd.read_csv(
        SUMMARY_PATH
    )

    if len(results) != 2000:
        raise RuntimeError(
            f"Expected 2000 result rows, "
            f"found {len(results)}."
        )

    if (
        results[
            "sample_id"
        ]
        .nunique()
        != 2000
    ):
        raise RuntimeError(
            "Result sample IDs are "
            "not unique."
        )

    if len(summary) != 1:
        raise RuntimeError(
            "Expected one summary row."
        )

    s = summary.iloc[0]

    baseline_accuracy = float(
        s[
            "baseline_accuracy"
        ]
    )

    post_accuracy = float(
        s[
            "post_accuracy"
        ]
    )

    accuracy_change = float(
        s[
            "accuracy_change"
        ]
    )

    ci_low = float(
        s[
            "accuracy_change_ci95_low"
        ]
    )

    ci_high = float(
        s[
            "accuracy_change_ci95_high"
        ]
    )

    repairs = int(
        s[
            "repairs"
        ]
    )

    breaks = int(
        s[
            "breaks"
        ]
    )

    net_repairs = int(
        s[
            "net_repairs"
        ]
    )

    intervention_rate = float(
        s[
            "intervention_rate"
        ]
    )

    exact_p = exact_mcnemar(
        repairs,
        breaks,
    )

    # --------------------------------------------------------
    # Lock the observed primary result.
    # --------------------------------------------------------

    expected = {
        "baseline_accuracy":
            0.4995,

        "post_accuracy":
            0.5805,

        "accuracy_change":
            0.0810,

        "ci_low":
            0.0685,

        "ci_high":
            0.0940,

        "repairs":
            173,

        "breaks":
            11,

        "net_repairs":
            162,

        "intervention_rate":
            0.3265,
    }

    checks = {
        "baseline_accuracy":
            baseline_accuracy,

        "post_accuracy":
            post_accuracy,

        "accuracy_change":
            accuracy_change,

        "ci_low":
            ci_low,

        "ci_high":
            ci_high,

        "repairs":
            repairs,

        "breaks":
            breaks,

        "net_repairs":
            net_repairs,

        "intervention_rate":
            intervention_rate,
    }

    for key, expected_value in (
        expected.items()
    ):

        observed = checks[
            key
        ]

        if isinstance(
            expected_value,
            float,
        ):
            if not math.isclose(
                float(observed),
                float(expected_value),
                rel_tol=0.0,
                abs_tol=1e-10,
            ):
                raise RuntimeError(
                    f"Result mismatch for {key}: "
                    f"{observed} vs "
                    f"{expected_value}"
                )

        else:
            if observed != expected_value:
                raise RuntimeError(
                    f"Result mismatch for {key}: "
                    f"{observed} vs "
                    f"{expected_value}"
                )

    files = {
        str(p):
            sha256_file(
                p
            )
        for p in required
    }

    manifest = {
        "archive_name":
            "AROMA V3 Frozen Final "
            "Confirmation Result",

        "archived_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "git_head_at_archive":
            git_head(),

        "freeze_chain": {
            "controller_freeze_commit":
                CONTROLLER_FREEZE,

            "v3_dataset_protocol_freeze_commit":
                V3_PROTOCOL_FREEZE,

            "v3_final_runner_freeze_commit":
                RUNNER_FREEZE,
        },

        "dataset": {
            "name":
                "proc_count_causal_v3",

            "generation_version":
                "3.0_unique",

            "n":
                2000,

            "untouched_before_final_evaluation":
                True,
        },

        "primary_result": {
            "baseline_correct":
                999,

            "baseline_wrong":
                1001,

            "baseline_accuracy":
                baseline_accuracy,

            "post_correct":
                1161,

            "post_accuracy":
                post_accuracy,

            "accuracy_change":
                accuracy_change,

            "accuracy_change_pp":
                100.0
                *
                accuracy_change,

            "accuracy_change_ci95_low":
                ci_low,

            "accuracy_change_ci95_high":
                ci_high,

            "repairs":
                repairs,

            "breaks":
                breaks,

            "net_repairs":
                net_repairs,

            "repair_rate_among_wrong":
                float(
                    s[
                        "repair_rate_among_wrong"
                    ]
                ),

            "break_rate_among_correct":
                float(
                    s[
                        "break_rate_among_correct"
                    ]
                ),

            "intervention_rate":
                intervention_rate,

            "exact_mcnemar_p":
                exact_p,

            "primary_success":
                True,
        },

        "development_comparison": {
            "v2_baseline_accuracy":
                0.503,

            "v2_controller_accuracy":
                0.586,

            "v2_accuracy_gain":
                0.083,

            "v3_accuracy_gain":
                accuracy_change,

            "v2_v3_gain_difference":
                accuracy_change
                -
                0.083,
        },

        "interpretation_status": {
            "synthetic_in_distribution_confirmation":
                True,

            "natural_image_generalization":
                False,

            "second_model_generalization":
                False,

            "full_stage_specific_aroma_complete":
                False,
        },

        "sha256":
            files,
    }

    ARCHIVE_PATH.write_text(
        json.dumps(
            manifest,
            indent=2,
        )
        +
        "\n",
        encoding="utf-8",
    )

    print("=" * 108)
    print(
        "AROMA V3 FINAL RESULT ARCHIVE"
    )
    print("=" * 108)

    print(
        f"Baseline accuracy : "
        f"{baseline_accuracy:.4f}"
    )

    print(
        f"Post accuracy     : "
        f"{post_accuracy:.4f}"
    )

    print(
        f"Accuracy gain     : "
        f"{accuracy_change:+.4f}"
    )

    print(
        f"95% CI            : "
        f"[{ci_low:+.4f}, "
        f"{ci_high:+.4f}]"
    )

    print(
        f"Repairs / breaks  : "
        f"{repairs} / {breaks}"
    )

    print(
        f"Net repairs       : "
        f"{net_repairs:+d}"
    )

    print(
        "Exact McNemar p   : "
        f"{exact_p:.12e}"
    )

    print(
        f"Intervention rate : "
        f"{intervention_rate:.4f}"
    )

    print(
        "\nFreeze chain:"
    )

    print(
        "  controller :",
        CONTROLLER_FREEZE,
    )

    print(
        "  v3 protocol:",
        V3_PROTOCOL_FREEZE,
    )

    print(
        "  v3 runner  :",
        RUNNER_FREEZE,
    )

    print(
        "\nSaved:",
        ARCHIVE_PATH,
    )

    print(
        "\nV3 FINAL RESULT ARCHIVE: PASS"
    )


if __name__ == "__main__":
    main()
