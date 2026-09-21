import math
from pathlib import Path

import numpy as np
import pandas as pd


RESULTS = Path(
    "outputs/proc_count_sa_dev_v1/"
    "sa_vs_mn_sweep_v1/"
    "sa_vs_mn_results.csv"
)

DIRECTION = Path(
    "outputs/proc_count_sa_dev_v1/"
    "sa_direction_diagnostic_v1/"
    "sa_dev_direction_per_sample.csv"
)

OUT = Path(
    "outputs/proc_count_sa_dev_v1/"
    "sa_vs_mn_upward_forensics_v1"
)

OUT.mkdir(parents=True, exist_ok=True)

BETAS = [
    0.75,
    1.00,
    1.50,
    1.75,
    2.00,
]


def exact_mcnemar(b, c):
    n = int(b + c)

    if n == 0:
        return 1.0

    k = int(min(b, c))

    tail = sum(
        math.comb(n, i)
        for i in range(k + 1)
    ) / (2 ** n)

    return float(min(1.0, 2.0 * tail))


def pair_at_beta(flat, beta):
    sub = flat[
        np.isclose(
            flat["beta"].astype(float),
            beta,
        )
        &
        (
            flat["selected_alpha"].astype(float)
            > 1.0
        )
    ].copy()

    mn = (
        sub[sub["method"] == "MN"]
        .sort_values("sample_id")
        .reset_index(drop=True)
    )

    sa = (
        sub[sub["method"] == "SA"]
        .sort_values("sample_id")
        .reset_index(drop=True)
    )

    if len(mn) != 160 or len(sa) != 160:
        raise RuntimeError(
            f"beta={beta}: expected 160 upward pairs; "
            f"got MN={len(mn)}, SA={len(sa)}"
        )

    if not np.array_equal(
        mn["sample_id"].astype(str).to_numpy(),
        sa["sample_id"].astype(str).to_numpy(),
    ):
        raise RuntimeError(
            f"beta={beta}: pairing mismatch"
        )

    out = mn[
        [
            "sample_id",
            "ground_truth",
            "selected_alpha",
            "projected_gradient_norm",
        ]
    ].copy()

    out["mn_prediction"] = (
        mn["prediction"].astype(int).to_numpy()
    )

    out["sa_prediction"] = (
        sa["prediction"].astype(int).to_numpy()
    )

    gt = out["ground_truth"].astype(int).to_numpy()

    out["mn_correct"] = (
        out["mn_prediction"].to_numpy() == gt
    ).astype(int)

    out["sa_correct"] = (
        out["sa_prediction"].to_numpy() == gt
    ).astype(int)

    out["sa_minus_mn"] = (
        out["sa_correct"]
        - out["mn_correct"]
    )

    return out


flat = pd.read_csv(RESULTS)

direction = pd.read_csv(DIRECTION)
direction["sample_id"] = direction["sample_id"].astype(str)

all_rows = []
action_rows = []
tertile_rows = []


print("=" * 100)
print("UPWARD-ONLY SA vs MN FORENSICS")
print("=" * 100)
print("Population: alpha > 1 only, N=160")


for beta in BETAS:

    pair = pair_at_beta(
        flat,
        beta,
    )

    pair = pair.merge(
        direction[
            [
                "sample_id",
                "cos_pug_puh",
                "pg_norm",
            ]
        ],
        on="sample_id",
        validate="one_to_one",
    )

    pair["log10_pg_norm"] = np.log10(
        np.maximum(
            pair["pg_norm"].astype(float),
            1e-30,
        )
    )

    sa_only = int(
        (
            (pair["sa_correct"] == 1)
            &
            (pair["mn_correct"] == 0)
        ).sum()
    )

    mn_only = int(
        (
            (pair["mn_correct"] == 1)
            &
            (pair["sa_correct"] == 0)
        ).sum()
    )

    row = {
        "beta": beta,
        "mn_correct": int(pair["mn_correct"].sum()),
        "sa_correct": int(pair["sa_correct"].sum()),
        "sa_minus_mn_pp":
            float(pair["sa_minus_mn"].mean() * 100),
        "sa_only_correct": sa_only,
        "mn_only_correct": mn_only,
        "mcnemar_p":
            exact_mcnemar(sa_only, mn_only),
    }

    all_rows.append(row)

    # ========================================================
    # BY ROUTER ACTION
    # ========================================================

    for alpha in [1.5, 2.0, 4.0]:

        g = pair[
            np.isclose(
                pair["selected_alpha"].astype(float),
                alpha,
            )
        ]

        if len(g) == 0:
            continue

        b = int(
            (
                (g["sa_correct"] == 1)
                &
                (g["mn_correct"] == 0)
            ).sum()
        )

        c = int(
            (
                (g["mn_correct"] == 1)
                &
                (g["sa_correct"] == 0)
            ).sum()
        )

        action_rows.append(
            {
                "beta": beta,
                "alpha": alpha,
                "n": len(g),
                "mn_accuracy":
                    float(g["mn_correct"].mean()),
                "sa_accuracy":
                    float(g["sa_correct"].mean()),
                "sa_minus_mn_pp":
                    float(g["sa_minus_mn"].mean() * 100),
                "sa_only_correct": b,
                "mn_only_correct": c,
                "net": b - c,
                "mcnemar_p":
                    exact_mcnemar(b, c),
            }
        )

    # ========================================================
    # TERTILES WITHIN UPWARD N=160 ONLY
    # ========================================================

    for variable in [
        "cos_pug_puh",
        "log10_pg_norm",
    ]:

        q1, q2 = np.quantile(
            pair[variable],
            [1/3, 2/3],
        )

        labels = np.where(
            pair[variable] <= q1,
            "low",
            np.where(
                pair[variable] <= q2,
                "mid",
                "high",
            ),
        )

        tmp = pair.copy()
        tmp["group"] = labels

        for group in [
            "low",
            "mid",
            "high",
        ]:

            g = tmp[
                tmp["group"] == group
            ]

            b = int(
                (
                    (g["sa_correct"] == 1)
                    &
                    (g["mn_correct"] == 0)
                ).sum()
            )

            c = int(
                (
                    (g["mn_correct"] == 1)
                    &
                    (g["sa_correct"] == 0)
                ).sum()
            )

            tertile_rows.append(
                {
                    "beta": beta,
                    "variable": variable,
                    "group": group,
                    "n": len(g),
                    "mn_accuracy":
                        float(g["mn_correct"].mean()),
                    "sa_accuracy":
                        float(g["sa_correct"].mean()),
                    "sa_minus_mn_pp":
                        float(
                            g["sa_minus_mn"].mean()
                            * 100
                        ),
                    "sa_only_correct": b,
                    "mn_only_correct": c,
                    "net": b - c,
                }
            )


overall = pd.DataFrame(all_rows)
by_action = pd.DataFrame(action_rows)
by_tertile = pd.DataFrame(tertile_rows)

overall.to_csv(
    OUT / "upward_overall_by_beta.csv",
    index=False,
)

by_action.to_csv(
    OUT / "upward_by_action.csv",
    index=False,
)

by_tertile.to_csv(
    OUT / "upward_by_tertile.csv",
    index=False,
)


print()
print("=" * 100)
print("UPWARD OVERALL")
print("=" * 100)

print(
    overall.to_string(index=False)
)


print()
print("=" * 100)
print("BY ACTION")
print("=" * 100)

print(
    by_action.to_string(index=False)
)


print()
print("=" * 100)
print("BETA=0.75 — WHERE DOES THE SA ADVANTAGE COME FROM?")
print("=" * 100)

print(
    by_action[
        np.isclose(
            by_action["beta"],
            0.75,
        )
    ].to_string(
        index=False
    )
)


print()
print("=" * 100)
print("UPWARD-ONLY COSINE TERTILES")
print("=" * 100)

print(
    by_tertile[
        by_tertile["variable"]
        == "cos_pug_puh"
    ].to_string(
        index=False
    )
)


print()
print("=" * 100)
print("UPWARD-ONLY PROJECTED-GRADIENT TERTILES")
print("=" * 100)

print(
    by_tertile[
        by_tertile["variable"]
        == "log10_pg_norm"
    ].to_string(
        index=False
    )
)


print()
print("=" * 100)
print("UPWARD FORENSIC ANALYSIS COMPLETE")
print("=" * 100)
print("Output:", OUT)
