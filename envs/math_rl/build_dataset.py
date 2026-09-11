"""Bake ``train.jsonl``/``validation.jsonl`` for the ``math_rl`` RLE world.

Run **once, at ``docker build`` time** (see ``Dockerfile``) -- never inside
the running container. This is the "embed at build time, don't let each
instance download" requirement from ``envs/README.md``: it calls the same
Hugging Face dataset loader vendored from the recipe
(``envs.math_rl.dataset_source``, pinned against
``loom_cookbook.recipes.math_rl.math_env`` by
``tests/test_env_grading_parity.py``), then writes plain JSONL that the
deployed environment reads from disk with no further downloads.

Usage::

    python build_dataset.py --out-dir . --max-train 4000 --max-validation 500
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from envs.math_rl.dataset_source import (
    _get_hendrycks_math_test,
    _get_hendrycks_math_train,
    question_suffix,
)

# Kept in lockstep (by value, not source -- there is no shared import
# without depending on loom_cookbook) with
# ``loom_cookbook.recipes.math_rl.math_env.MathEnv.standard_fewshot_prefix``,
# which is what the local (non-RLE) path prepends by default
# (``MathDatasetBuilder.convo_prefix == "standard"``). Baking a *different*
# prompt shape here (previously: a system-role instruction, no few-shot
# example) meant use_rle=True and use_rle=False trained/evaluated on
# different prompt distributions for an otherwise-identical config --
# see ``tests/test_env_grading_parity.py``'s
# ``test_math_rl_question_suffix_matches_the_recipe_value`` for the sibling
# check on ``question_suffix``, which this mirrors for the fewshot prefix.
STANDARD_FEWSHOT_PREFIX = [
    {
        "role": "user",
        "content": "How many r's are in strawberry?" + question_suffix(),
    },
    {
        "role": "assistant",
        "content": (
            "Let's spell the word out and number all the letters: 1) s 2) t 3) r 4) a"
            " 5) w 6) b 7) e 8) r 9) r 10) y. We have r's at positions 3, 8, and 9."
            " \\boxed{3}"
        ),
    },
]


def _row_to_record(row: dict) -> dict:
    question = row["problem"] + question_suffix()
    return {
        "messages": [
            *STANDARD_FEWSHOT_PREFIX,
            {"role": "user", "content": question},
        ],
        "problem": row["problem"],
        "solution": row["solution"],
    }


def build(out_dir: Path, max_train: int | None, max_validation: int | None, seed: int) -> None:
    train_ds = _get_hendrycks_math_train().shuffle(seed=seed)
    test_ds = _get_hendrycks_math_test()
    if max_train is not None:
        train_ds = train_ds.select(range(min(max_train, len(train_ds))))
    if max_validation is not None:
        test_ds = test_ds.select(range(min(max_validation, len(test_ds))))

    out_dir.mkdir(parents=True, exist_ok=True)
    for split_name, ds in (("train", train_ds), ("validation", test_ds)):
        path = out_dir / f"{split_name}.jsonl"
        with open(path, "w") as f:
            for row in ds:
                f.write(json.dumps(_row_to_record(row)) + "\n")
        print(f"wrote {len(ds)} rows to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("."))
    parser.add_argument("--max-train", type=int, default=4000)
    parser.add_argument("--max-validation", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    build(args.out_dir, args.max_train, args.max_validation, args.seed)
