#!/usr/bin/env python3

"""
Reconstruct the exact 1000 x 39 feature matrix used to fit the
frozen AROMA v2 full-geometry controller.

This script:
- performs NO model inference,
- performs NO controller fitting for prediction,
- performs NO hyperparameter tuning,
- modifies NO frozen artifacts.

It reconstructs the original feature matrix by importing the
original frozen feature-construction functions, then verifies the
result against the StandardScaler statistics stored in the frozen
controller bundle.
"""

from pathlib import Path
import hashlib
import importlib.util
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ============================================================
# Paths
# ============================================================

ROOT = Path("/workspace/AromaExperiments")

SOURCE_SCRIPT = (
    ROOT
    / "scripts"
    / "run_v2_baseline_utility_controller.py"
)

CONF_PATH = (
    ROOT
    / "outputs"
    / "proc_count_causal_v2"
    / "confirmation_l18h13_a1p5_expanded_0_15"
    / "confirmation_results.csv"
)

BUNDLE_PATH = (
    ROOT
    / "outputs"
    / "proc_count_causal_v2"
    / "controller"
    / "final_frozen_controller"
    / "aroma_cardinality_controller.joblib"
)

OUT_DIR = (
    ROOT
    / "outputs"
    / "proc_count_causal_v2"
    / "controller"
    / "training_support_reconstruction"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MATRIX_OUT = (
    OUT_DIR
    / "v2_controller_training_features_39.csv"
)

SUPPORT_OUT = (
    OUT_DIR
    / "v2_controller_training_support.csv"
)

AUDIT_OUT = (
    OUT_DIR
    / "reconstruction_audit.json"
)


# ============================================================
# Helpers
# ============================================================

def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


# ============================================================
# Load original feature-construction code
# ============================================================

spec = importlib.util.spec_from_file_location(
    "v2_controller_original",
    SOURCE_SCRIPT,
)

module = importlib.util.module_from_spec(
    spec
)

spec.loader.exec_module(
    module
)

add_geometry_features = (
    module.add_geometry_features
)

build_feature_families = (
    module.build_feature_families
)


# ============================================================
# Load frozen controller
# ============================================================

bundle = joblib.load(
    BUNDLE_PATH
)

feature_names = list(
    bundle["feature_names"]
)

if len(feature_names) != 39:
    raise RuntimeError(
        f"Expected 39 frozen feature names; "
        f"found {len(feature_names)}."
    )

if bundle["feature_family"] != "full_geometry":
    raise RuntimeError(
        "Frozen controller is not full_geometry."
    )


# ============================================================
# Reconstruct exact baseline table
# ============================================================

conf = pd.read_csv(
    CONF_PATH
)

print("=" * 118)
print("V2 CONTROLLER TRAINING-SUPPORT RECONSTRUCTION")
print("=" * 118)

print("Source rows:", len(conf))
print(
    "Unique sample_id:",
    conf["sample_id"].nunique(),
)

base = (
    conf[
        np.isclose(
            conf["alpha"].astype(float),
            1.0,
        )
    ]
    .copy()
)

base = (
    base
    .sort_values(
        "sample_id"
    )
    .reset_index(
        drop=True
    )
)

if len(base) != 1000:
    raise RuntimeError(
        f"Expected 1000 alpha=1 rows; "
        f"found {len(base)}."
    )

if base["sample_id"].nunique() != 1000:
    raise RuntimeError(
        "Expected 1000 unique sample IDs."
    )

print("Recovered baseline rows:", len(base))
print(
    "Unique baseline samples:",
    base["sample_id"].nunique(),
)


# ============================================================
# Re-run original geometry construction
# ============================================================

features_df = (
    add_geometry_features(
        base
    )
)

families = (
    build_feature_families(
        features_df
    )
)

full_names = list(
    families["full_geometry"]
)

print("\nFrozen bundle feature count:",
      len(feature_names))

print("Original script full_geometry count:",
      len(full_names))

if full_names != feature_names:
    print("\nFEATURE ORDER MISMATCH")

    for i, (a, b) in enumerate(
        zip(
            full_names,
            feature_names,
        )
    ):
        if a != b:
            print(
                i,
                "script=",
                a,
                "bundle=",
                b,
            )

    raise RuntimeError(
        "Feature order does not match "
        "frozen bundle."
    )

print("Feature names/order: PASS")


# ============================================================
# Construct exact X
# ============================================================

X = (
    features_df[
        feature_names
    ]
    .to_numpy(
        dtype=float
    )
)

if X.shape != (1000, 39):
    raise RuntimeError(
        f"Unexpected feature matrix shape: "
        f"{X.shape}"
    )

if not np.isfinite(X).all():
    raise RuntimeError(
        "Non-finite values in reconstructed X."
    )

print("Feature matrix shape:", X.shape)
print("Finite-value audit: PASS")


# ============================================================
# Refit ONLY a StandardScaler for checksum.
#
# This is not controller retraining; it reconstructs the
# deterministic preprocessing statistics from the recovered
# feature matrix.
# ============================================================

reconstructed_scaler = (
    StandardScaler()
    .fit(X)
)

recon_mean = np.asarray(
    reconstructed_scaler.mean_,
    dtype=float,
)

recon_scale = np.asarray(
    reconstructed_scaler.scale_,
    dtype=float,
)


# ============================================================
# Compare against every frozen action scaler
# ============================================================

comparison_rows = []

all_pass = True

for alpha in bundle["nonnoop_actions"]:

    frozen_scaler = (
        bundle[
            "models"
        ][alpha]
        .named_steps[
            "scale"
        ]
    )

    frozen_mean = np.asarray(
        frozen_scaler.mean_,
        dtype=float,
    )

    frozen_scale = np.asarray(
        frozen_scaler.scale_,
        dtype=float,
    )

    mean_abs_diff = np.abs(
        recon_mean
        -
        frozen_mean
    )

    scale_abs_diff = np.abs(
        recon_scale
        -
        frozen_scale
    )

    mean_match = np.allclose(
        recon_mean,
        frozen_mean,
        rtol=1e-10,
        atol=1e-12,
    )

    scale_match = np.allclose(
        recon_scale,
        frozen_scale,
        rtol=1e-10,
        atol=1e-12,
    )

    this_pass = (
        mean_match
        and
        scale_match
    )

    all_pass = (
        all_pass
        and
        this_pass
    )

    comparison_rows.append(
        {
            "alpha":
                float(alpha),

            "max_mean_abs_diff":
                float(
                    mean_abs_diff.max()
                ),

            "max_scale_abs_diff":
                float(
                    scale_abs_diff.max()
                ),

            "mean_match":
                bool(
                    mean_match
                ),

            "scale_match":
                bool(
                    scale_match
                ),

            "pass":
                bool(
                    this_pass
                ),
        }
    )


comparison = pd.DataFrame(
    comparison_rows
)

print("\n" + "=" * 118)
print("FROZEN SCALER CHECKSUM")
print("=" * 118)

print(
    comparison.to_string(
        index=False
    )
)

if not all_pass:
    raise RuntimeError(
        "Reconstructed training matrix does "
        "not reproduce frozen scaler statistics."
    )

print(
    "\nSCALER RECONSTRUCTION: PASS"
)


# ============================================================
# Per-feature support
# ============================================================

Z = (
    X
    -
    recon_mean
) / recon_scale


support_rows = []

for j, feature in enumerate(
    feature_names
):

    raw = X[:, j]
    z = Z[:, j]

    support_rows.append(
        {
            "feature_index":
                j,

            "feature":
                feature,

            "train_mean":
                float(
                    recon_mean[j]
                ),

            "train_scale":
                float(
                    recon_scale[j]
                ),

            "raw_min":
                float(
                    raw.min()
                ),

            "raw_q001":
                float(
                    np.quantile(
                        raw,
                        0.001,
                    )
                ),

            "raw_q01":
                float(
                    np.quantile(
                        raw,
                        0.01,
                    )
                ),

            "raw_q05":
                float(
                    np.quantile(
                        raw,
                        0.05,
                    )
                ),

            "raw_median":
                float(
                    np.quantile(
                        raw,
                        0.50,
                    )
                ),

            "raw_q95":
                float(
                    np.quantile(
                        raw,
                        0.95,
                    )
                ),

            "raw_q99":
                float(
                    np.quantile(
                        raw,
                        0.99,
                    )
                ),

            "raw_q999":
                float(
                    np.quantile(
                        raw,
                        0.999,
                    )
                ),

            "raw_max":
                float(
                    raw.max()
                ),

            "z_min":
                float(
                    z.min()
                ),

            "z_q001":
                float(
                    np.quantile(
                        z,
                        0.001,
                    )
                ),

            "z_q01":
                float(
                    np.quantile(
                        z,
                        0.01,
                    )
                ),

            "z_q05":
                float(
                    np.quantile(
                        z,
                        0.05,
                    )
                ),

            "z_median":
                float(
                    np.quantile(
                        z,
                        0.50,
                    )
                ),

            "z_q95":
                float(
                    np.quantile(
                        z,
                        0.95,
                    )
                ),

            "z_q99":
                float(
                    np.quantile(
                        z,
                        0.99,
                    )
                ),

            "z_q999":
                float(
                    np.quantile(
                        z,
                        0.999,
                    )
                ),

            "z_max":
                float(
                    z.max()
                ),

            "abs_z_max":
                float(
                    np.abs(
                        z
                    ).max()
                ),
        }
    )


support = pd.DataFrame(
    support_rows
)

support.to_csv(
    SUPPORT_OUT,
    index=False,
)


# ============================================================
# Save reconstructed matrix
# ============================================================

matrix_out = pd.DataFrame(
    X,
    columns=feature_names,
)

matrix_out.insert(
    0,
    "sample_id",
    base[
        "sample_id"
    ].astype(str).values,
)

matrix_out.to_csv(
    MATRIX_OUT,
    index=False,
)


# ============================================================
# Audit metadata
# ============================================================

audit = {
    "reconstruction_pass":
        bool(all_pass),

    "n_samples":
        int(X.shape[0]),

    "n_features":
        int(X.shape[1]),

    "source_confirmation_path":
        str(CONF_PATH.relative_to(ROOT)),

    "source_confirmation_sha256":
        sha256_file(
            CONF_PATH
        ),

    "source_feature_code_path":
        str(SOURCE_SCRIPT.relative_to(ROOT)),

    "source_feature_code_sha256":
        sha256_file(
            SOURCE_SCRIPT
        ),

    "frozen_bundle_path":
        str(BUNDLE_PATH.relative_to(ROOT)),

    "frozen_bundle_sha256":
        sha256_file(
            BUNDLE_PATH
        ),

    "reconstructed_matrix_path":
        str(MATRIX_OUT.relative_to(ROOT)),

    "support_table_path":
        str(SUPPORT_OUT.relative_to(ROOT)),

    "max_mean_abs_diff":
        float(
            comparison[
                "max_mean_abs_diff"
            ].max()
        ),

    "max_scale_abs_diff":
        float(
            comparison[
                "max_scale_abs_diff"
            ].max()
        ),

    "feature_names_exact_match":
        True,
}


with open(
    AUDIT_OUT,
    "w",
) as f:

    json.dump(
        audit,
        f,
        indent=2,
        sort_keys=True,
    )


# ============================================================
# Print most relevant support features
# ============================================================

print("\n" + "=" * 118)
print("HIGH-NUMERAL TRAINING SUPPORT")
print("=" * 118)

focus = support[
    support[
        "feature"
    ].isin(
        [
            "baseline_numprob_12",
            "baseline_numprob_13",
            "baseline_numprob_14",
            "baseline_numprob_15",
        ]
    )
]

print(
    focus[
        [
            "feature",
            "train_mean",
            "train_scale",
            "raw_min",
            "raw_q99",
            "raw_q999",
            "raw_max",
            "z_min",
            "z_q99",
            "z_q999",
            "z_max",
        ]
    ].to_string(
        index=False
    )
)


print("\n" + "=" * 118)
print("15 NARROWEST TRAINING FEATURES")
print("=" * 118)

print(
    support.sort_values(
        "train_scale",
        ascending=True,
    )[
        [
            "feature",
            "train_mean",
            "train_scale",
            "raw_min",
            "raw_max",
            "z_min",
            "z_max",
        ]
    ]
    .head(15)
    .to_string(
        index=False
    )
)


print("\n" + "=" * 118)
print("TRAINING-SUPPORT RECONSTRUCTION COMPLETE")
print("=" * 118)

print(
    "Exact feature-order match: PASS"
)

print(
    "Frozen scaler checksum: PASS"
)

print(
    "No model inference performed."
)

print(
    "No controller prediction model retrained."
)

print(
    "No hyperparameter tuning performed."
)

print(
    "No frozen artifact modified."
)

print("\nSaved:")
print(MATRIX_OUT)
print(SUPPORT_OUT)
print(AUDIT_OUT)

