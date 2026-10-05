"""Builds a stratified, disjoint train/validation split from the task pool.

Why not a positional slice
---------------------------
The committed `job_data/train.jsonl` / `validation.jsonl` used to be the first N and last M
`task_index` values in file order -- an artifact of upstream tarball ordering, not a deliberate
sample. That gives no control over which `reward_mode`s or how much combo-PII (2+
`pii_categories`, the hardest compliance case) land in either split, and no guarantee the two
splits' difficulty mix even resembles each other.

This script instead stratifies on `(reward_mode, pii_bucket)` -- `pii_bucket` is `len(pii_categories)`
capped at 2 ("2+") -- and draws both splits from the *same* per-cell pool, proportionally to each
cell's natural share of the full 5,000-task metadata file (no artificial oversampling of rare
cells; the user's call was to keep natural frequency and lean on absolute size instead). Train and
validation are carved from disjoint slices of each cell's shuffled pool, so the two files can never
overlap even though both are sampled from the same population.

Apportionment is largest-remainder (Hamilton): each cell gets `floor(population * target /
5000)`, then leftover slots go to the cells with the largest fractional remainder, so split sizes
land on the requested totals exactly rather than approximately.

`--write` also regenerates `job_data/full.jsonl`, every `task_index` in the pool (unfiltered,
unshuffled). It isn't read by `rle.toml` (`training_file`/`validation_file` only point at the two
stratified splits) -- it's kept purely so the population those splits were drawn from is
inspectable as a plain `.jsonl` instead of the gzipped task-meta blob.

Usage
-----
    python tools/build_training_set.py                      # dry run, prints the plan only
    python tools/build_training_set.py --write               # regenerates job_data/*.jsonl
    python tools/build_training_set.py --train-size 2000 --eval-size 150 --write
"""

from __future__ import annotations

import argparse
import gzip
import json
import pathlib
import random
import sys
from collections import Counter, defaultdict

HERE = pathlib.Path(__file__).resolve().parent
# This tool lives under `_internal/`, a sibling of the real sample; the content it reads and
# writes stays in the real sample tree, two levels further up and back down into `data_code_agent`.
SAMPLE_ROOT = HERE.parents[2] / "data_code_agent"
TASK_META = (
    SAMPLE_ROOT / "rle" / "server" / "tasks" / "task-meta" / "FineEnvs__data-agent-harbor-train.json.gz"
)
TRAIN_OUT = SAMPLE_ROOT / "job_data" / "train.jsonl"
VALIDATION_OUT = SAMPLE_ROOT / "job_data" / "validation.jsonl"
# Not load-bearing -- rle.toml only points `training_file`/`validation_file` at TRAIN_OUT and
# VALIDATION_OUT. This is the full population those two are stratified-sampled from, kept
# alongside them purely so the pool is inspectable without re-reading the gzipped task-meta.
FULL_OUT = SAMPLE_ROOT / "job_data" / "full.jsonl"
SPLIT = "FineEnvs/data-agent-harbor-train"

DEFAULT_TRAIN_SIZE = 1024
DEFAULT_EVAL_SIZE = 150
# Matches `[train.options].seed` in `rle/rle.toml` -- not load-bearing for training itself, just
# reused here so "the project's seed" stays a single number instead of two unrelated ones.
DEFAULT_SEED = 42


def _pii_bucket(count: int) -> str:
    return "2+" if count >= 2 else str(count)


def _load_cells(task_meta: list[dict[str, object]]) -> dict[tuple[str, str], list[int]]:
    """Returns {(reward_mode, pii_bucket): [task_index, ...]} over the whole pool."""
    cells: dict[tuple[str, str], list[int]] = defaultdict(list)
    for task_index, entry in enumerate(task_meta):
        key = (str(entry["reward_mode"]), _pii_bucket(len(entry["pii_categories"])))
        cells[key].append(task_index)
    return cells


def _apportion(cell_sizes: dict[tuple[str, str], int], target: int, total: int) -> dict[tuple[str, str], int]:
    """Largest-remainder apportionment of `target` slots across cells, proportional to each
    cell's share of `total`. Sums to exactly `target`."""
    quotas = {key: size * target / total for key, size in cell_sizes.items()}
    floors = {key: int(quota) for key, quota in quotas.items()}
    remainder = target - sum(floors.values())
    # Largest fractional remainder first; tie-break on key for determinism.
    order = sorted(quotas, key=lambda k: (-(quotas[k] - floors[k]), k))
    for key in order[:remainder]:
        floors[key] += 1
    return floors


def plan(
    task_meta: list[dict[str, object]], train_size: int, eval_size: int, seed: int
) -> tuple[list[int], list[int], dict[tuple[str, str], tuple[int, int, int]]]:
    """Returns (train_indices, eval_indices, {cell: (population, eval_n, train_n)})."""
    rng = random.Random(seed)
    cells = _load_cells(task_meta)
    population = len(task_meta)
    cell_sizes = {key: len(indices) for key, indices in cells.items()}

    # Eval first, then train from what's left in each cell -- the two can never overlap because
    # train is drawn from the remainder after eval's slice is removed, not from the full cell.
    eval_alloc = _apportion(cell_sizes, eval_size, population)
    remaining_sizes = {key: cell_sizes[key] - eval_alloc[key] for key in cell_sizes}
    train_alloc = _apportion(remaining_sizes, train_size, population - eval_size)

    train_indices: list[int] = []
    eval_indices: list[int] = []
    summary: dict[tuple[str, str], tuple[int, int, int]] = {}
    for key, indices in cells.items():
        shuffled = indices[:]
        rng.shuffle(shuffled)
        eval_n = eval_alloc[key]
        train_n = train_alloc[key]
        assert eval_n + train_n <= len(shuffled), f"cell {key} over-allocated: {eval_n}+{train_n} > {len(shuffled)}"
        eval_indices.extend(shuffled[:eval_n])
        train_indices.extend(shuffled[eval_n : eval_n + train_n])
        summary[key] = (len(indices), eval_n, train_n)

    rng.shuffle(train_indices)
    rng.shuffle(eval_indices)
    return train_indices, eval_indices, summary


def _print_summary(summary: dict[tuple[str, str], tuple[int, int, int]], train_size: int, eval_size: int) -> None:
    print(f"{'reward_mode':14s} {'pii':4s} {'pool':>6s} {'eval':>6s} {'train':>6s}")
    for key in sorted(summary):
        pool, eval_n, train_n = summary[key]
        print(f"{key[0]:14s} {key[1]:4s} {pool:6d} {eval_n:6d} {train_n:6d}")
    total_eval = sum(v[1] for v in summary.values())
    total_train = sum(v[2] for v in summary.values())
    print(f"{'total':14s} {'':4s} {sum(v[0] for v in summary.values()):6d} {total_eval:6d} {total_train:6d}")
    assert total_eval == eval_size and total_train == train_size, "apportionment did not sum to target"


def _write_jsonl(path: pathlib.Path, task_indices: list[int]) -> None:
    lines = [json.dumps({"split": SPLIT, "task_index": i}) for i in task_indices]
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-size", type=int, default=DEFAULT_TRAIN_SIZE)
    parser.add_argument("--eval-size", type=int, default=DEFAULT_EVAL_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--write", action="store_true", help="write job_data/train.jsonl + validation.jsonl")
    args = parser.parse_args()

    if not TASK_META.exists():
        sys.exit(f"missing task metadata: {TASK_META}")
    with gzip.open(TASK_META) as handle:
        task_meta = json.load(handle)

    if args.train_size + args.eval_size > len(task_meta):
        sys.exit(
            f"train-size ({args.train_size}) + eval-size ({args.eval_size}) exceeds the pool "
            f"({len(task_meta)})"
        )

    train_indices, eval_indices, summary = plan(task_meta, args.train_size, args.eval_size, args.seed)
    _print_summary(summary, args.train_size, args.eval_size)
    print(f"\ntrain {len(train_indices)} / eval {len(eval_indices)} / disjoint: "
          f"{len(set(train_indices) & set(eval_indices)) == 0}")

    if not args.write:
        print("\ndry run only -- pass --write to regenerate job_data/train.jsonl + validation.jsonl")
        return 0

    _write_jsonl(TRAIN_OUT, train_indices)
    _write_jsonl(VALIDATION_OUT, eval_indices)
    _write_jsonl(FULL_OUT, list(range(len(task_meta))))
    print(f"\nwrote {TRAIN_OUT} ({len(train_indices)} lines)")
    print(f"wrote {VALIDATION_OUT} ({len(eval_indices)} lines)")
    print(f"wrote {FULL_OUT} ({len(task_meta)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
