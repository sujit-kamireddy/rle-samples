"""Generate ``train.jsonl``/``validation.jsonl`` from the upstream corpus.

This is a maintainer utility, not part of the Docker build. Normal sample
images use the compact checked-in snapshot under ``data/``. It uses
``load_deepcoder_tasks`` from ``envs/code_rl/dataset_source.py`` -- a copy of
the recipe's task loader (DeepCoder-Preview dataset, normalized test cases).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from envs.code_rl.dataset_source import load_deepcoder_tasks

SYSTEM_PROMPT = (
    "You are an expert competitive programmer. Read the problem, then submit"
    " a complete Python solution via the terminal answer tool."
)


def _row_to_record(task) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task.problem},
        ],
        "tests": task.tests,
        "starter_code": task.starter_code,
    }


def build(out_dir: Path, max_train: int | None, max_validation: int | None, seed: int) -> None:
    # Pass the caps down rather than slicing afterwards: the loader streams
    # when it knows the limit, so rows that would be discarded here are never
    # downloaded at all.
    train_tasks = load_deepcoder_tasks(split="train", seed=seed, max_tasks=max_train)
    test_tasks = load_deepcoder_tasks(split="test", seed=seed, max_tasks=max_validation)

    out_dir.mkdir(parents=True, exist_ok=True)
    for split_name, tasks in (("train", train_tasks), ("validation", test_tasks)):
        path = out_dir / f"{split_name}.jsonl"
        with open(path, "w") as f:
            for task in tasks:
                f.write(json.dumps(_row_to_record(task)) + "\n")
        print(f"wrote {len(tasks)} rows to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("."))
    parser.add_argument("--max-train", type=int, default=2000)
    parser.add_argument("--max-validation", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    build(args.out_dir, args.max_train, args.max_validation, args.seed)
