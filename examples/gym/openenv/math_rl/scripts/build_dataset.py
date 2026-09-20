"""Regenerate the ``env_data/train.jsonl.gz``/``validation.jsonl.gz`` snapshot for
the ``math_rl`` RLE world.

This is a **maintainer tool**, not part of the Docker build: it needs a real
Hugging Face/network connection, so it is run once, ahead of time, from a
full checkout of this repository (where the
``examples.gym.openenv.math_rl.scripts.dataset_source`` import below
resolves), and its output is gzip-compressed and checked into ``env_data/``
(see ``env_data/NOTICE.md``). The image itself only ever reads that
checked-in snapshot from local disk -- no network access or per-instance
download, and no dependency on this script or ``dataset_source.py`` at
build or run time.

Usage (from the repository root)::

    python -m examples.gym.openenv.math_rl.scripts.build_dataset \\
        --out-dir examples/gym/openenv/math_rl/env_data --max-train 4000 --max-validation 500
    gzip examples/gym/openenv/math_rl/env_data/train.jsonl examples/gym/openenv/math_rl/env_data/validation.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from examples.gym.openenv.math_rl.scripts.dataset_source import (
    _get_hendrycks_math_test,
    _get_hendrycks_math_train,
    question_suffix,
)

# One-shot few-shot prefix prepended to every prompt, showing the model the
# expected \boxed{} answer format on an unrelated warm-up question before
# the real problem.
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


def _write_job_data_manifest(job_data_dir: Path, split_name: str, num_rows: int) -> None:
    """Writes `job_data/<split>.jsonl`: one `reset()` task payload per row of
    the baked `env_data/<split>.jsonl.gz` this snapshot produces -- never the
    problem/solution content itself, just enough (`seed`, `split`) for a
    training job to reproducibly pick that row via `EpisodePicker.pick()`.
    Must stay in lockstep with the row counts below; see `job_data/README.md`."""
    job_data_dir.mkdir(parents=True, exist_ok=True)
    path = job_data_dir / f"{split_name}.jsonl"
    with open(path, "w") as f:
        for seed in range(num_rows):
            f.write(json.dumps({"seed": seed, "split": split_name}) + "\n")
    print(f"wrote {num_rows} rows to {path}")


def build(
    out_dir: Path,
    job_data_dir: Path,
    max_train: int | None,
    max_validation: int | None,
    seed: int,
) -> None:
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
        _write_job_data_manifest(job_data_dir, split_name, len(ds))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("."))
    parser.add_argument(
        "--job-data-out-dir",
        type=Path,
        default=None,
        help="Where to write job_data/train.jsonl + job_data/validation.jsonl "
        "(defaults to --out-dir's sibling 'job_data' directory).",
    )
    parser.add_argument("--max-train", type=int, default=4000)
    parser.add_argument("--max-validation", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    job_data_out_dir = args.job_data_out_dir or args.out_dir.parent / "job_data"
    build(args.out_dir, job_data_out_dir, args.max_train, args.max_validation, args.seed)
