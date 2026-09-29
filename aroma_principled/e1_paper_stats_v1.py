import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

IN = Path(
    "aroma_principled/results/e1_continuous_v1/full160.jsonl"
)

OUTDIR = Path(
    "aroma_principled/results/e1_continuous_v1/paper_stats_v1"
)
OUTDIR.mkdir(parents=True, exist_ok=True)

N_BOOT = 2000
SEED = 20260929


def load_jsonl(path):
    return pd.DataFrame([
        json.loads(x)
        for x in path.read_text().splitlines()
        if x.strip()
    ])


def metrics(df):
    x = df["linear_pred_delta_mu"].to_numpy(dtype=float)
    y = df["delta_mu"].to_numpy(dtype=float)

    X = np.column_stack([
        np.ones(len(x)),
        x,
    ])

    coef, *_ = np.linalg.lstsq(
        X,
        y,
        rcond=None,
    )

    intercept = float(coef[0])
    slope = float(coef[1])

    err = y - x

    if np.std(x) > 0 and np.std(y) > 0:
        pearson = float(pearsonr(x, y).statistic)
        spearman = float(spearmanr(x, y).statistic)
    else:
        pearson = np.nan
        spearman = np.nan

    nonzero = (
        (np.abs(x) > 1e-15)
        |
        (np.abs(y) > 1e-15)
    )

    if nonzero.any():
        sign_agreement = float(
            np.mean(
                np.sign(x[nonzero])
                ==
                np.sign(y[nonzero])
            )
        )
    else:
        sign_agreement = np.nan

    return {
        "n_rows": int(len(df)),
        "n_samples": int(df["sample_id"].nunique()),
        "slope": slope,
        "intercept": intercept,
        "pearson": pearson,
        "spearman": spearman,
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "sign_agreement": sign_agreement,
    }


def cluster_bootstrap(df, rng):
    ids = np.array(
        sorted(df["sample_id"].astype(str).unique())
    )

    blocks = {
        sid: df[
            df["sample_id"].astype(str) == sid
        ]
        for sid in ids
    }

    out = {
        k: []
        for k in [
            "slope",
            "intercept",
            "pearson",
            "spearman",
            "mae",
            "rmse",
            "sign_agreement",
        ]
    }

    for _ in range(N_BOOT):
        draw = rng.choice(
            ids,
            size=len(ids),
            replace=True,
        )

        boot = pd.concat(
            [blocks[sid] for sid in draw],
            ignore_index=True,
        )

        m = metrics(boot)

        for k in out:
            out[k].append(m[k])

    ci = {}

    for k, vals in out.items():
        a = np.asarray(vals, dtype=float)

        ci[f"{k}_ci_lo"] = float(
            np.nanpercentile(a, 2.5)
        )

        ci[f"{k}_ci_hi"] = float(
            np.nanpercentile(a, 97.5)
        )

    return ci


df = load_jsonl(IN)

expected_rows = 160 * 13 * 2

if len(df) != expected_rows:
    raise RuntimeError(
        f"Expected {expected_rows} rows; got {len(df)}"
    )

if df["sample_id"].nunique() != 160:
    raise RuntimeError(
        f"Expected 160 samples; got "
        f"{df['sample_id'].nunique()}"
    )

print("=" * 110)
print("E1 PAPER STATISTICS v1")
print("=" * 110)
print("rows:", len(df))
print("samples:", df["sample_id"].nunique())
print("betas:", sorted(df["beta"].unique()))
print()

rng = np.random.default_rng(SEED)

rows = []

for beta in sorted(df["beta"].unique()):

    beta_df = df[
        np.isclose(df["beta"], beta)
    ]

    for direction in [
        "all",
        "activation",
        "gradient",
    ]:

        if direction == "all":
            z = beta_df.copy()
        else:
            z = beta_df[
                beta_df["direction"] == direction
            ].copy()

        m = metrics(z)
        ci = cluster_bootstrap(z, rng)

        row = {
            "beta": float(beta),
            "direction": direction,
            **m,
            **ci,
        }

        rows.append(row)

        if direction == "all":
            print(
                f"beta={beta:>4.2f} "
                f"slope={m['slope']:.4f} "
                f"intercept={m['intercept']:+.6f} "
                f"Pearson={m['pearson']:.4f} "
                f"Spearman={m['spearman']:.4f} "
                f"MAE={m['mae']:.6f} "
                f"RMSE={m['rmse']:.6f} "
                f"sign={m['sign_agreement']:.2%}"
            )

stats = pd.DataFrame(rows)

stats.to_csv(
    OUTDIR / "calibration_by_beta.csv",
    index=False,
)


# ------------------------------------------------------------
# Local-regime pooled analysis
# beta <= .10 and beta <= .20
# ------------------------------------------------------------

local_rows = []

for max_beta in [
    0.05,
    0.10,
    0.20,
]:

    for direction in [
        "all",
        "activation",
        "gradient",
    ]:

        z = df[
            df["beta"] <= max_beta + 1e-12
        ].copy()

        if direction != "all":
            z = z[
                z["direction"] == direction
            ]

        m = metrics(z)
        ci = cluster_bootstrap(z, rng)

        local_rows.append({
            "max_beta": max_beta,
            "direction": direction,
            **m,
            **ci,
        })

local = pd.DataFrame(local_rows)

local.to_csv(
    OUTDIR / "local_regime_summary.csv",
    index=False,
)


# ------------------------------------------------------------
# Descriptive geometry / intervention integrity
# one geometry row per sample
# ------------------------------------------------------------

geom = (
    df[
        [
            "sample_id",
            "rho",
            "cos_pg_ph",
            "h_norm",
            "ph_norm",
            "g_shift_norm",
            "pg_shift_norm",
        ]
    ]
    .drop_duplicates("sample_id")
)

geometry_summary = {
    "n_samples": int(len(geom)),
    "rho_median": float(geom["rho"].median()),
    "rho_q25": float(geom["rho"].quantile(.25)),
    "rho_q75": float(geom["rho"].quantile(.75)),
    "cos_median": float(geom["cos_pg_ph"].median()),
    "cos_q25": float(geom["cos_pg_ph"].quantile(.25)),
    "cos_q75": float(geom["cos_pg_ph"].quantile(.75)),
    "cos_nonpositive_fraction": float(
        np.mean(geom["cos_pg_ph"] <= 0)
    ),
    "max_shift_norm_abs_error": float(
        df["shift_norm_abs_error"].max()
    ),
}

with (
    OUTDIR / "geometry_summary.json"
).open("w") as f:
    json.dump(
        geometry_summary,
        f,
        indent=2,
    )


print()
print("=" * 110)
print("LOCAL REGIME")
print("=" * 110)

show = local[
    local["direction"] == "all"
]

for _, r in show.iterrows():
    print(
        f"beta<={r['max_beta']:.2f}: "
        f"slope={r['slope']:.4f} "
        f"[{r['slope_ci_lo']:.4f}, "
        f"{r['slope_ci_hi']:.4f}]  "
        f"Pearson={r['pearson']:.4f}  "
        f"Spearman={r['spearman']:.4f}  "
        f"MAE={r['mae']:.6f}  "
        f"sign={r['sign_agreement']:.2%}"
    )

print()
print("=" * 110)
print("GEOMETRY")
print("=" * 110)

for k, v in geometry_summary.items():
    print(k, "=", v)

print()
print("Saved:")
print(" ", OUTDIR / "calibration_by_beta.csv")
print(" ", OUTDIR / "local_regime_summary.csv")
print(" ", OUTDIR / "geometry_summary.json")
print("=" * 110)
