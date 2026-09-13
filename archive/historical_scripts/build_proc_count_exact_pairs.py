import json
from pathlib import Path

import pandas as pd


INPUT = Path(
    "outputs/proc_count_causal_v1/"
    "baseline_results.jsonl"
)

OUTPUT = Path(
    "outputs/proc_count_causal_v1/"
    "exact_matched_pairs.csv"
)


def main():
    records = [
        json.loads(line)
        for line in INPUT.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    df = pd.DataFrame(
        records
    )

    pairs = []

    pair_id = 0

    for (
        gt,
        condition,
    ), group in df.groupby(
        [
            "ground_truth",
            "condition",
        ]
    ):

        correct = (
            group[
                group[
                    "correct"
                ]
                == True
            ]
            .sort_values(
                "sample_id"
            )
            .reset_index(
                drop=True
            )
        )

        wrong = (
            group[
                group[
                    "correct"
                ]
                == False
            ]
            .sort_values(
                "sample_id"
            )
            .reset_index(
                drop=True
            )
        )

        n_pairs = min(
            len(correct),
            len(wrong),
        )

        for i in range(
            n_pairs
        ):
            c = correct.iloc[i]
            w = wrong.iloc[i]

            pairs.append(
                {
                    "pair_id":
                        pair_id,

                    "ground_truth":
                        gt,

                    "condition":
                        condition,

                    "correct_sample_id":
                        c[
                            "sample_id"
                        ],

                    "wrong_sample_id":
                        w[
                            "sample_id"
                        ],

                    "correct_replicate":
                        c[
                            "replicate"
                        ],

                    "wrong_replicate":
                        w[
                            "replicate"
                        ],

                    "correct_prediction":
                        c[
                            "prediction"
                        ],

                    "wrong_prediction":
                        w[
                            "prediction"
                        ],
                }
            )

            pair_id += 1

    out = pd.DataFrame(
        pairs
    )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out.to_csv(
        OUTPUT,
        index=False,
    )

    print("=" * 80)
    print(
        "Proc-Count-Causal Exact Matching"
    )
    print("=" * 80)

    print(
        "Total baseline samples:",
        len(df),
    )

    print(
        "Correct:",
        int(
            df["correct"].sum()
        ),
    )

    print(
        "Wrong:",
        int(
            (
                ~df["correct"]
            ).sum()
        ),
    )

    print(
        "Exact matched pairs:",
        len(out),
    )

    if len(out) > 0:

        assert all(
            out[
                "ground_truth"
            ].notna()
        )

        print(
            "\nPairs by count:"
        )

        print(
            out[
                "ground_truth"
            ]
            .value_counts()
            .sort_index()
            .to_string()
        )

        print(
            "\nPairs by condition:"
        )

        print(
            out[
                "condition"
            ]
            .value_counts()
            .to_string()
        )

    print(
        "\nSaved:",
        OUTPUT,
    )


if __name__ == "__main__":
    main()
