import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


FILES = [
    Path(
        "configs/"
        "aroma_cardinality_controller_frozen.json"
    ),
    Path(
        "outputs/proc_count_causal_v2/"
        "controller/final_frozen_controller/"
        "aroma_cardinality_controller.joblib"
    ),
    Path(
        "outputs/proc_count_causal_v2/"
        "controller/final_frozen_controller/"
        "final_hyperparameter_grid.csv"
    ),
    Path(
        "outputs/proc_count_causal_v2/"
        "controller/final_frozen_controller/"
        "selected_hyperparameter_oof.csv"
    ),
    Path(
        "scripts/"
        "fit_v2_final_frozen_controller.py"
    ),
    Path(
        "scripts/"
        "run_v2_baseline_utility_controller.py"
    ),
    Path(
        "scripts/"
        "build_v2_compressed_action_matrix.py"
    ),
]


OUT = Path(
    "configs/"
    "aroma_controller_freeze_record.json"
)


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            block = f.read(
                1024 * 1024
            )

            if not block:
                break

            h.update(block)

    return h.hexdigest()


def git_head():
    try:
        return subprocess.check_output(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            text=True,
        ).strip()
    except Exception:
        return None


def main():

    missing = [
        str(p)
        for p in FILES
        if not p.exists()
    ]

    if missing:
        raise RuntimeError(
            "Missing freeze files:\n"
            + "\n".join(missing)
        )

    record = {
        "freeze_name":
            "AROMA Adaptive Cardinality "
            "Controller pre-v3 freeze",

        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "development_dataset":
            "Proc-Count-Causal v2.1_unique",

        "final_confirmation_dataset":
            "Proc-Count-Causal v3 "
            "(not yet evaluated)",

        "model":
            "meta-llama/"
            "Llama-3.2-11B-Vision-Instruct",

        "head":
            "L18H13",

        "actions":
            [
                0.0,
                1.0,
                1.5,
                2.0,
                4.0,
            ],

        "feature_family":
            "full_geometry",

        "feature_count":
            39,

        "ridge_alpha":
            0.01,

        "utility_threshold":
            0.1,

        "development_oof": {
            "baseline_accuracy":
                0.503,

            "post_accuracy":
                0.586,

            "repairs":
                91,

            "breaks":
                8,

            "net_repairs":
                83,

            "intervention_rate":
                0.333,

            "mcnemar_exact_p":
                5.90963701961033e-19,
        },

        "retuning_after_freeze": {
            "head":
                False,

            "actions":
                False,

            "features":
                False,

            "ridge_alpha":
                False,

            "utility_threshold":
                False,

            "decision_rule":
                False,

            "primary_endpoint":
                False,
        },

        "git_head_before_freeze_commit":
            git_head(),

        "sha256": {
            str(p):
                sha256(p)
            for p in FILES
        },
    }

    OUT.write_text(
        json.dumps(
            record,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("=" * 100)
    print("AROMA CONTROLLER FREEZE RECORD")
    print("=" * 100)

    print(
        json.dumps(
            record,
            indent=2,
        )
    )

    print("\nSaved:", OUT)
    print("\nFREEZE RECORD COMPLETE")


if __name__ == "__main__":
    main()
