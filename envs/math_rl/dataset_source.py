"""Dataset loading for the ``math_rl`` environment.

Vendored from the recipe so this environment builds and runs without
installing ``loom_cookbook`` -- see ``envs/README.md``. ``_get_hendrycks_math_test``
and ``_get_hendrycks_math_train`` are kept byte-identical to
``loom_cookbook.recipes.math_rl.math_env``; ``question_suffix`` is
re-shaped from ``MathEnv.question_suffix`` (a classmethod there, a plain
function here) so it is checked by return value instead of source in
``tests/test_env_grading_parity.py``.
"""

from __future__ import annotations

from typing import cast

from datasets import Dataset, concatenate_datasets, get_dataset_config_names, load_dataset


def question_suffix() -> str:
    return " Write your answer in \\boxed{} format."


def _get_hendrycks_math_test() -> Dataset:
    test_dataset = load_dataset("HuggingFaceH4/MATH-500", name="default", split="test")
    return cast(Dataset, test_dataset)


def _get_hendrycks_math_train() -> Dataset:
    # For Hendrycks MATH, the standard is to use both the "train" and "test" splits for
    # training. The "test" split here is NOT the same as the MATH-500 test split above,
    # which is a commonly-held-out subset of 500 of the below 12.5k problems. To construct
    # a clean training set, we filter out problems that exist in the MATH-500 test set,
    # resulting in 12000 train and 500 test problems.

    test_problems: set[str] = {
        problem["problem"]  # pyright: ignore[reportArgumentType, reportCallIssue]
        for problem in _get_hendrycks_math_test()
    }

    dataset_name = "EleutherAI/hendrycks_math"
    configs = get_dataset_config_names(dataset_name)
    pieces = []
    for cfg in configs:
        for split in ("train", "test"):
            ds = load_dataset(dataset_name, name=cfg, split=split)
            ds = ds.filter(lambda example: example["problem"] not in test_problems)
            pieces.append(ds)
    full_dataset = concatenate_datasets(pieces)

    return full_dataset
