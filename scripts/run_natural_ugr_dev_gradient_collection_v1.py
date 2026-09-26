#!/usr/bin/env python3

import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    MllamaForConditionalGeneration,
)

sys.path.insert(0, "scripts")

import natural_ugr_stage2_common_v1 as C
import run_tallyqa_natural_confirmation_v2_final as nat


EXPECTED_N = 292

DEV = Path(
    "manifests/natural_ugr_v1/"
    "development_population_v1.csv"
)

PROTOCOL = Path(
    "manifests/natural_ugr_v1/"
    "protocol_v1.json"
)

IMAGE_ROOT = Path(
    "data/tallyqa_natural_confirmation_v2/images"
)

OUT = Path(
    "outputs/natural_ugr_v1/"
    "development/stage2/"
    "gradient_collection_v1"
)

PARTIAL = (
    OUT
    /
    "partial_gradient_collection_v1.npz"
)

FINAL_FILES = [
    OUT / "gradient_matrix.npy",
    OUT / "pooled_h_matrix.npy",
    OUT / "sample_ids.npy",
    OUT / "cohort_uids.npy",
    OUT / "token_lengths.npy",
    OUT / "mu.npy",
    OUT / "metadata.json",
]

EXPECTED_PROTOCOL_SHA256 = (
    "1daedfee079bc81f5c6171be6af9f4d84175ed50eaec8043b9e3217a16e73d47"
)

EXPECTED_DEV_SHA256 = (
    "de509f52bbd7cac0974282f49c2a2e33e5d80450dbbcf2e13cee3eda1f01c1ca"
)


def main():

    if C.sha256_file(
        PROTOCOL
    ) != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError(
            "F1 protocol hash mismatch."
        )

    if C.sha256_file(
        DEV
    ) != EXPECTED_DEV_SHA256:
        raise RuntimeError(
            "F1 development population hash mismatch."
        )

    if any(
        p.exists()
        for p in FINAL_FILES
    ):
        raise RuntimeError(
            "Final gradient artifacts already exist."
        )

    dev = pd.read_csv(
        DEV
    )

    if len(dev) != EXPECTED_N:
        raise RuntimeError(
            f"Expected N={EXPECTED_N}; got {len(dev)}"
        )

    if not (
        dev["selected_alpha"].astype(float)
        >
        1.0
    ).all():
        raise RuntimeError(
            "Non-upward sample in frozen dev population."
        )

    sample_ids = np.array(
        [
            str(
                int(x)
            )
            for x in dev[
                "question_id"
            ]
        ],
        dtype=object,
    )

    cohort_uids = (
        dev[
            "natural_ugr_uid"
        ]
        .astype(str)
        .to_numpy(
            dtype=object
        )
    )

    if len(
        set(
            sample_ids.tolist()
        )
    ) != EXPECTED_N:
        raise RuntimeError(
            "question_id is not unique."
        )

    if len(
        set(
            cohort_uids.tolist()
        )
    ) != EXPECTED_N:
        raise RuntimeError(
            "cohort UID is not unique."
        )

    for _, row in dev.iterrows():

        p = (
            IMAGE_ROOT
            /
            str(
                row[
                    "image"
                ]
            )
        )

        if not p.is_file():
            raise FileNotFoundError(
                p
            )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    done_ids = []
    done_uids = []
    gradients = []
    pooled_h = []
    mus = []
    token_lengths = []

    if PARTIAL.exists():

        p = np.load(
            PARTIAL,
            allow_pickle=True,
        )

        done_ids = (
            p["sample_ids"]
            .astype(str)
            .tolist()
        )

        done_uids = (
            p["cohort_uids"]
            .astype(str)
            .tolist()
        )

        gradients = [
            x.astype(
                np.float64
            )
            for x in p[
                "gradients"
            ]
        ]

        pooled_h = [
            x.astype(
                np.float64
            )
            for x in p[
                "pooled_h"
            ]
        ]

        mus = (
            p["mu"]
            .astype(
                np.float64
            )
            .tolist()
        )

        token_lengths = (
            p[
                "token_lengths"
            ]
            .astype(
                np.int64
            )
            .tolist()
        )

        n = len(
            done_ids
        )

        if not (
            n
            ==
            len(done_uids)
            ==
            len(gradients)
            ==
            len(pooled_h)
            ==
            len(mus)
            ==
            len(token_lengths)
        ):
            raise RuntimeError(
                "Partial gradient length mismatch."
            )

        if done_ids != (
            sample_ids[:n]
            .astype(str)
            .tolist()
        ):
            raise RuntimeError(
                "Partial sample IDs are not "
                "an exact frozen prefix."
            )

        if done_uids != (
            cohort_uids[:n]
            .astype(str)
            .tolist()
        ):
            raise RuntimeError(
                "Partial cohort UIDs are not "
                "an exact frozen prefix."
            )

        print(
            f"RESUME: {n}/{EXPECTED_N}"
        )

    print(
        "Loading processor..."
    )

    processor = (
        AutoProcessor
        .from_pretrained(
            C.MODEL_ID,
            revision=
                C.MODEL_REVISION,
        )
    )

    numeral_ids = (
        C.numeral_token_ids(
            processor
        )
    )

    print(
        "Loading model..."
    )

    model = (
        MllamaForConditionalGeneration
        .from_pretrained(
            C.MODEL_ID,
            revision=
                C.MODEL_REVISION,
            torch_dtype=
                torch.bfloat16,
            device_map="auto",
            attn_implementation=
                "eager",
        )
    )

    model.eval()

    for param in model.parameters():
        param.requires_grad_(
            False
        )

    layer = (
        model
        .model
        .language_model
        .layers[
            C.LAYER
        ]
    )

    o_proj = (
        layer
        .cross_attn
        .o_proj
    )

    if (
        o_proj.in_features
        !=
        C.HIDDEN_SIZE
    ):
        raise RuntimeError(
            "Unexpected o_proj input width."
        )

    start_idx = len(
        done_ids
    )

    for idx in range(
        start_idx,
        EXPECTED_N,
    ):

        row = dev.iloc[
            idx
        ]

        sid = str(
            sample_ids[
                idx
            ]
        )

        uid = str(
            cohort_uids[
                idx
            ]
        )

        image_path = (
            IMAGE_ROOT
            /
            str(
                row[
                    "image"
                ]
            )
        )

        image = (
            Image.open(
                image_path
            )
            .convert(
                "RGB"
            )
        )

        inputs = (
            nat.prepare_tallyqa_inputs(
                processor,
                image,
                str(
                    row[
                        "question"
                    ]
                ),
            )
        )

        inputs = C.move_inputs(
            inputs,
            model,
        )

        saved = {}

        handle = (
            o_proj
            .register_forward_pre_hook(
                C.make_hook(
                    saved
                )
            )
        )

        try:

            out = model(
                **inputs,
                use_cache=False,
                return_dict=True,
            )

            if saved.get(
                "calls",
                0,
            ) <= 0:
                raise RuntimeError(
                    f"{sid}: gradient hook did not fire."
                )

            mu = C.differentiable_mu(
                out.logits,
                numeral_ids,
            )

            g = torch.autograd.grad(
                outputs=mu,
                inputs=saved[
                    "h_leaf"
                ],
                retain_graph=False,
                create_graph=False,
            )[
                0
            ]

        finally:
            handle.remove()

        g_np = (
            g[
                0
            ]
            .detach()
            .float()
            .cpu()
            .numpy()
            .astype(
                np.float64
            )
        )

        h_np = (
            saved[
                "h_base"
            ][
                0
            ]
            .detach()
            .float()
            .cpu()
            .numpy()
            .astype(
                np.float64
            )
        )

        if g_np.shape != (
            C.HEAD_DIM,
        ):
            raise RuntimeError(
                f"{sid}: bad gradient shape "
                f"{g_np.shape}"
            )

        if h_np.shape != (
            C.HEAD_DIM,
        ):
            raise RuntimeError(
                f"{sid}: bad h shape "
                f"{h_np.shape}"
            )

        if not np.isfinite(
            g_np
        ).all():
            raise RuntimeError(
                f"{sid}: non-finite gradient."
            )

        if not np.isfinite(
            h_np
        ).all():
            raise RuntimeError(
                f"{sid}: non-finite activation."
            )

        done_ids.append(
            sid
        )

        done_uids.append(
            uid
        )

        gradients.append(
            g_np
        )

        pooled_h.append(
            h_np
        )

        mus.append(
            float(
                mu
                .detach()
                .cpu()
                .item()
            )
        )

        token_lengths.append(
            int(
                saved[
                    "token_length"
                ]
            )
        )

        np.savez_compressed(
            PARTIAL,
            sample_ids=
                np.array(
                    done_ids,
                    dtype=object,
                ),
            cohort_uids=
                np.array(
                    done_uids,
                    dtype=object,
                ),
            gradients=
                np.stack(
                    gradients
                ),
            pooled_h=
                np.stack(
                    pooled_h
                ),
            mu=
                np.array(
                    mus,
                    dtype=np.float64,
                ),
            token_lengths=
                np.array(
                    token_lengths,
                    dtype=np.int64,
                ),
        )

        print(
            f"[{idx + 1:03d}/{EXPECTED_N}] "
            f"{sid} "
            f"mu={mus[-1]:.6f} "
            f"|g|={np.linalg.norm(g_np):.6g} "
            f"|h|={np.linalg.norm(h_np):.6g}",
            flush=True,
        )

        del out
        del mu
        del g
        del inputs
        del image
        del saved

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    G = np.stack(
        gradients
    ).astype(
        np.float64
    )

    H = np.stack(
        pooled_h
    ).astype(
        np.float64
    )

    IDS = np.array(
        done_ids,
        dtype=object,
    )

    UIDS = np.array(
        done_uids,
        dtype=object,
    )

    TOK = np.array(
        token_lengths,
        dtype=np.int64,
    )

    MU = np.array(
        mus,
        dtype=np.float64,
    )

    if G.shape != (
        EXPECTED_N,
        C.HEAD_DIM,
    ):
        raise RuntimeError(
            f"Bad final G shape: {G.shape}"
        )

    if H.shape != (
        EXPECTED_N,
        C.HEAD_DIM,
    ):
        raise RuntimeError(
            f"Bad final H shape: {H.shape}"
        )

    if not np.array_equal(
        IDS.astype(str),
        sample_ids.astype(str),
    ):
        raise RuntimeError(
            "Final sample ordering mismatch."
        )

    if not np.array_equal(
        UIDS.astype(str),
        cohort_uids.astype(str),
    ):
        raise RuntimeError(
            "Final cohort ordering mismatch."
        )

    np.save(
        OUT
        /
        "gradient_matrix.npy",
        G,
    )

    np.save(
        OUT
        /
        "pooled_h_matrix.npy",
        H,
    )

    np.save(
        OUT
        /
        "sample_ids.npy",
        IDS,
    )

    np.save(
        OUT
        /
        "cohort_uids.npy",
        UIDS,
    )

    np.save(
        OUT
        /
        "token_lengths.npy",
        TOK,
    )

    np.save(
        OUT
        /
        "mu.npy",
        MU,
    )

    metadata = {
        "experiment":
            "natural_ugr_dev_gradient_collection_v1",

        "population_n":
            EXPECTED_N,

        "ground_truth_used":
            False,

        "model_id":
            C.MODEL_ID,

        "model_revision":
            C.MODEL_REVISION,

        "layer":
            C.LAYER,

        "head":
            C.HEAD,

        "head_dim":
            C.HEAD_DIM,

        "gradient_target":
            (
                "exact historical UGR differentiable_mu"
            ),

        "gradient_semantics":
            (
                "shared additive perturbation across "
                "all token positions at L18H13 o_proj input"
            ),

        "gradient_norm": {
            "min":
                float(
                    np.linalg.norm(
                        G,
                        axis=1,
                    ).min()
                ),

            "median":
                float(
                    np.median(
                        np.linalg.norm(
                            G,
                            axis=1,
                        )
                    )
                ),

            "max":
                float(
                    np.linalg.norm(
                        G,
                        axis=1,
                    ).max()
                ),
        },

        "h_norm": {
            "min":
                float(
                    np.linalg.norm(
                        H,
                        axis=1,
                    ).min()
                ),

            "median":
                float(
                    np.median(
                        np.linalg.norm(
                            H,
                            axis=1,
                        )
                    )
                ),

            "max":
                float(
                    np.linalg.norm(
                        H,
                        axis=1,
                    ).max()
                ),
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

    print()
    print(
        "=" * 78
    )

    print(
        "NATURAL UGR DEV GRADIENT COLLECTION COMPLETE"
    )

    print(
        "G:",
        G.shape,
    )

    print(
        "H:",
        H.shape,
    )

    print(
        "=" * 78
    )


if __name__ == "__main__":
    main()
