"""Regenerate the ``env_data/train.jsonl.gz``/``validation.jsonl.gz`` snapshot
for the ``code_repair`` RLE world.

This is a **maintainer tool**, not part of the Docker build: it needs a real
Hugging Face/network connection, so it is run once, ahead of time, from a
full checkout of this repository (where the
``examples.gym.openenv.code_repair.scripts.dataset_source`` import below
resolves), and its output is gzip-compressed and checked into ``env_data/``
(see ``env_data/NOTICE.md``). The image itself only ever reads that
checked-in snapshot from local disk, and clones ``django/django`` itself
(see ``Dockerfile``) -- no per-instance network access at build or run time,
and no dependency on this script or ``dataset_source.py``.

Usage (from the repository root)::

    python -m examples.gym.openenv.code_repair.scripts.build_dataset \\
        --out-dir examples/gym/openenv/code_repair/env_data
    gzip examples/gym/openenv/code_repair/env_data/train.jsonl \\
        examples/gym/openenv/code_repair/env_data/validation.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from examples.gym.openenv.code_repair.scripts.dataset_source import get_django_instances


def _write_job_data_manifest(job_data_dir: Path, split_name: str, num_rows: int) -> None:
    """Writes `job_data/<split>.jsonl`: one `reset()` task payload per row of
    the baked `env_data/<split>.jsonl.gz` this snapshot produces -- never the
    instance content itself, just enough (`seed`, `split`) for a training job
    to reproducibly pick that row via `EpisodePicker.pick()`. Must stay in
    lockstep with the row counts below; see `job_data/README.md`."""
    job_data_dir.mkdir(parents=True, exist_ok=True)
    path = job_data_dir / f"{split_name}.jsonl"
    with open(path, "w") as f:
        for seed in range(num_rows):
            f.write(json.dumps({"seed": seed, "split": split_name}) + "\n")
    print(f"wrote {num_rows} rows to {path}")


def build(out_dir: Path, job_data_dir: Path, num_validation: int, seed: int) -> None:
    instances = get_django_instances()
    # Sorted first so the shuffle (and therefore the train/validation split)
    # is reproducible regardless of the order the dataset happens to be
    # downloaded in.
    instances.sort(key=lambda row: row["instance_id"])
    random.Random(seed).shuffle(instances)

    num_validation = min(num_validation, len(instances) - 1)
    splits = {
        "validation": instances[:num_validation],
        "train": instances[num_validation:],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    for split_name, rows in splits.items():
        path = out_dir / f"{split_name}.jsonl"
        with open(path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        print(f"wrote {len(rows)} rows to {path}")
        _write_job_data_manifest(job_data_dir, split_name, len(rows))


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
    parser.add_argument(
        "--num-validation",
        type=int,
        default=8,
        help="How many of the 40 usable django/3.2 instances to hold out for validation.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    job_data_out_dir = args.job_data_out_dir or args.out_dir.parent / "job_data"
    build(args.out_dir, job_data_out_dir, args.num_validation, args.seed)
