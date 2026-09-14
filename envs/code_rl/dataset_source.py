"""Dataset loading for the ``code_rl`` environment.

Vendored from the recipe so this environment builds and runs without
installing ``loom_cookbook`` -- see ``envs/README.md``. Kept byte-identical
in behaviour to ``loom_cookbook.recipes.code_rl.code_env``; the paired
recipe test asserts the two agree.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    from datasets import Dataset

from envs.code_rl.grading.code_grading import taco_to_lcb_format
from envs.code_rl.grading.lcb_utils import fetch_live_code_bench_system_prompt

logger = logging.getLogger(__name__)

_DATASET_ID = "agentica-org/DeepCoder-Preview-Dataset"
_SPLIT_CONFIGS: dict[str, tuple[str, ...]] = {
    "train": ("primeintellect", "taco", "lcbv5"),
    "test": ("codeforces", "lcbv5"),
}
_MIN_SHUFFLE_BUFFER = 100


@dataclass(frozen=True)
class DeepcoderTask:
    """A single code task with problem statement and test cases."""

    problem: str
    tests: list[dict[str, Any]]
    starter_code: str | None = None


def _load_deepcoder_split(split: Literal["train", "test"]) -> "Dataset":
    from datasets import Dataset, concatenate_datasets, load_dataset

    logger.info("Loading DeepCoder dataset split: %s", split)
    if split == "train":
        names = ("primeintellect", "taco", "lcbv5")
    else:
        names = ("codeforces", "lcbv5")

    datasets = []
    for name in names:
        logger.info(f"  Loading {name}...")
        ds = load_dataset("agentica-org/DeepCoder-Preview-Dataset", name=name, split=split)
        datasets.append(cast(Dataset, ds))

    return cast(Dataset, concatenate_datasets(datasets))


def _ensure_dict(metadata: Any) -> dict[str, Any]:
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            logger.warning("Failed to deserialize metadata: %s", metadata)
            return {}
    if isinstance(metadata, dict):
        return metadata
    return {}


def _normalize_tests(raw_tests: Any, metadata: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize test cases to a unified format."""
    tests = raw_tests
    if isinstance(tests, str):
        try:
            tests = json.loads(tests)
        except json.JSONDecodeError:
            logger.warning("Failed to deserialize tests. Dropping sample.")
            return []
    if isinstance(tests, dict) and "inputs" in tests and "outputs" in tests:
        tests = taco_to_lcb_format(tests)
    if isinstance(tests, dict):
        tests = [tests]

    normalized: list[dict[str, Any]] = []
    for test in tests or []:
        if not isinstance(test, dict):
            continue
        testtype = test.get("testtype") or "stdin_stdout"
        test_metadata = _ensure_dict(test.get("metadata", {}))
        if testtype == "functional":
            func_name = test_metadata.get("func_name") or metadata.get("func_name")
            if func_name is not None:
                test_metadata["func_name"] = str(func_name)
        normalized.append(
            {
                "input": str(test.get("input", "")),
                "output": str(test.get("output", "")),
                "testtype": testtype,
                "metadata": test_metadata or {"func_name": None},
            }
        )
    return normalized


def _build_question(example: dict[str, Any]) -> str | None:
    """Build the question text with LCB system prompt."""
    question = example.get("question") or example.get("prompt") or example.get("problem")
    if not isinstance(question, str) or not question.strip():
        return None
    starter_code = example.get("starter_code")
    if isinstance(starter_code, str) and starter_code.strip():
        return fetch_live_code_bench_system_prompt(question, starter_code)
    return fetch_live_code_bench_system_prompt(question)


def _row_to_task(row: dict[str, Any]) -> DeepcoderTask | None:
    """Convert one raw dataset row into a task, or ``None`` if unusable.

    The body is lifted verbatim from the recipe's loop so that a task built
    here is byte-identical to one the in-process path would build; only the
    *selection* of rows differs (see ``load_deepcoder_tasks``).
    """
    # Extract and normalize metadata
    metadata = _ensure_dict(row.get("metadata", {}))

    # Normalize test cases
    raw_tests = row.get("tests") or row.get("ground_truth")
    tests = _normalize_tests(raw_tests, metadata)
    if not tests:
        return None

    # Build problem prompt
    problem = _build_question(row)
    if problem is None:
        return None

    # Extract starter code if present
    starter_code = row.get("starter_code")
    if isinstance(starter_code, str) and not starter_code.strip():
        starter_code = None

    return DeepcoderTask(
        problem=problem,
        tests=tests,
        starter_code=starter_code if isinstance(starter_code, str) else None,
    )


def _config_sizes(split: Literal["train", "test"]) -> dict[str, int]:
    """Row count per config, from dataset metadata -- no rows are downloaded."""
    from datasets import load_dataset_builder

    sizes: dict[str, int] = {}
    for name in _SPLIT_CONFIGS[split]:
        builder = load_dataset_builder(_DATASET_ID, name=name)
        info = builder.info.splits
        sizes[name] = info[split].num_examples if info and split in info else 0
    return sizes


def _allocate(sizes: dict[str, int], wanted: int) -> dict[str, int]:
    """Split ``wanted`` across configs in proportion to their row counts.

    Sampling each config proportionally is what keeps the subset's config mix
    the same as a shuffle of the whole split would have produced. Largest
    remainder, so the parts sum to exactly ``wanted``.
    """
    total = sum(sizes.values())
    if total <= 0:
        return {name: 0 for name in sizes}
    if wanted >= total:
        return dict(sizes)

    exact = {name: wanted * rows / total for name, rows in sizes.items()}
    alloc = {name: min(int(value), sizes[name]) for name, value in exact.items()}
    for name in sorted(exact, key=lambda n: exact[n] - int(exact[n]), reverse=True):
        if sum(alloc.values()) >= wanted:
            break
        if alloc[name] < sizes[name]:
            alloc[name] += 1
    return alloc


def _stream_tasks(name: str, split: str, seed: int, take: int) -> list[DeepcoderTask]:
    """Pull rows from one config until ``take`` usable tasks are collected."""
    from datasets import load_dataset

    if take <= 0:
        return []
    stream = load_dataset(_DATASET_ID, name=name, split=split, streaming=True)
    # Shuffling an IterableDataset permutes shard order and fills a reservoir
    # before yielding anything, so the buffer is a download floor: sizing it to
    # the sample keeps `lcbv5` (599 rows spread over 3.4 GB) from being read in
    # full just to pick ~50 rows.
    buffer = max(_MIN_SHUFFLE_BUFFER, take * 2)
    stream = stream.shuffle(seed=seed, buffer_size=buffer)

    tasks: list[DeepcoderTask] = []
    for item in stream:
        task = _row_to_task(cast(dict[str, Any], item))
        if task is not None:
            tasks.append(task)
            if len(tasks) >= take:
                break
    return tasks


def load_deepcoder_tasks(
    split: Literal["train", "test"] = "train",
    seed: int = 0,
    max_tasks: int | None = None,
) -> list[DeepcoderTask]:
    """Load tasks from the DeepCoder dataset.

    Args:
        split: Which split to load ("train" or "test")
        seed: Random seed for shuffling (train split only)
        max_tasks: Stop once this many tasks are collected. Streams the
            dataset instead of downloading it, which is the difference
            between fetching ~7.8 GB and ~1 GB for the 2200 examples an
            image actually bakes in. ``None`` downloads everything, matching
            the recipe.

    Returns:
        List of DeepcoderTask instances with normalized test cases
    """
    if max_tasks is not None:
        sizes = _config_sizes(split)
        alloc = _allocate(sizes, max_tasks)
        logger.info("Streaming %s split, per-config sample: %s", split, alloc)
        tasks: list[DeepcoderTask] = []
        for name, take in alloc.items():
            tasks.extend(_stream_tasks(name, split, seed, take))
        # Rows arrive grouped by config; interleave them so a truncated or
        # sequentially-consumed dataset still sees the full mix.
        random.Random(seed).shuffle(tasks)
        return tasks[:max_tasks]

    ds = _load_deepcoder_split(split)
    if split == "train":
        ds = ds.shuffle(seed=seed)

    logger.info(f"Processing {len(ds)} examples into tasks...")
    tasks = []
    for item in ds:
        task = _row_to_task(cast(dict[str, Any], item))
        if task is not None:
            tasks.append(task)

    return tasks
