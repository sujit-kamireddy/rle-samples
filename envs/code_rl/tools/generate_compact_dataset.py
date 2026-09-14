"""Regenerate the compact, checked-in code_rl dataset snapshot.

This maintainer-only utility fetches a fixed DeepCoder-Preview revision via
the Hugging Face datasets server, keeps a deterministic 900/100 task split,
and preserves at most three complete test cases and 12 KiB of test
input/output per task. Docker builds never run it; they copy the generated
gzip JSONL files under ``data/`` instead.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import gzip
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from envs.code_rl.build_dataset import _row_to_record
from envs.code_rl.dataset_source import _row_to_task


DATASET_ID = "agentica-org/DeepCoder-Preview-Dataset"
DATASET_REVISION = "177913a7bd43791646ef6a43645caa3c871ab3db"
DATASET_LICENSE = "MIT"
ROWS_ENDPOINT = "https://datasets-server.huggingface.co/rows"
PAGE_SIZE = 25
MAX_FETCH_ATTEMPTS = 4
DEFAULT_MAX_TEST_BYTES_PER_TASK = 12 * 1024


def _fetch_rows(config: str, split: str, offset: int, length: int, revision: str) -> list[dict[str, Any]]:
    query = urlencode(
        {
            "dataset": DATASET_ID,
            "config": config,
            "split": split,
            "offset": offset,
            "length": length,
            "revision": revision,
        }
    )
    request = Request(f"{ROWS_ENDPOINT}?{query}", headers={"User-Agent": "rle-samples-snapshot-generator"})
    for attempt in range(1, MAX_FETCH_ATTEMPTS + 1):
        try:
            with urlopen(request, timeout=120) as response:  # noqa: S310 - fixed HTTPS endpoint above.
                payload = json.load(response)
            return [entry["row"] for entry in payload["rows"]]
        except HTTPError as error:
            if error.code not in {429, 500, 502, 503, 504} or attempt == MAX_FETCH_ATTEMPTS:
                raise
            print(
                f"Fetching {config}/{split} at offset {offset} failed with HTTP {error.code}; "
                f"retrying ({attempt}/{MAX_FETCH_ATTEMPTS})...",
                file=sys.stderr,
            )
        except (TimeoutError, URLError) as error:
            if attempt == MAX_FETCH_ATTEMPTS:
                raise
            print(
                f"Fetching {config}/{split} at offset {offset} failed ({error}); "
                f"retrying ({attempt}/{MAX_FETCH_ATTEMPTS})...",
                file=sys.stderr,
            )
        time.sleep(attempt)
    raise RuntimeError("unreachable")


def _page_offsets(total_rows: int, target_rows: int, seed: int) -> list[int]:
    full_page_count = total_rows // PAGE_SIZE
    if full_page_count == 0:
        raise ValueError(f"{total_rows} rows cannot fill a {PAGE_SIZE}-row page")
    offsets = [page * PAGE_SIZE for page in range(full_page_count)]
    random.Random(seed).shuffle(offsets)
    return offsets


def _compact_test_cases(
    tests: list[dict[str, Any]],
    max_tests_per_task: int,
    max_test_bytes_per_task: int,
) -> list[dict[str, Any]]:
    compact_tests: list[dict[str, Any]] = []
    total_bytes = 0
    for test in tests:
        test_bytes = len(str(test.get("input", "")).encode("utf-8")) + len(
            str(test.get("output", "")).encode("utf-8")
        )
        if (
            len(compact_tests) == max_tests_per_task
            or test_bytes > max_test_bytes_per_task
            or total_bytes + test_bytes > max_test_bytes_per_task
        ):
            continue
        compact_tests.append(test)
        total_bytes += test_bytes
    return compact_tests


def _collect_records(
    *,
    config: str,
    split: str,
    total_rows: int,
    target_rows: int,
    max_tests_per_task: int,
    max_test_bytes_per_task: int,
    seed: int,
    revision: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for offset in _page_offsets(total_rows, target_rows, seed):
        for row in _fetch_rows(config, split, offset, PAGE_SIZE, revision):
            task = _row_to_task(row)
            if task is None:
                continue
            task = replace(
                task,
                tests=_compact_test_cases(
                    task.tests,
                    max_tests_per_task,
                    max_test_bytes_per_task,
                ),
            )
            if not task.tests:
                continue
            records.append(_row_to_record(task))
            if len(records) == target_rows:
                random.Random(seed).shuffle(records)
                return records
    raise RuntimeError(
        f"Collected {len(records)} valid {config}/{split} rows, but need {target_rows}. "
        "Increase the candidate-page set or select a different source config."
    )


def _write_jsonl_gzip(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("wb") as raw_file:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_file, mtime=0) as compressed_file:
            for record in records:
                compressed_file.write(json.dumps(record).encode("utf-8"))
                compressed_file.write(b"\n")


def _file_metadata(path: Path, rows: int) -> dict[str, int | str]:
    return {
        "bytes": path.stat().st_size,
        "rows": rows,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
    )
    parser.add_argument("--train-count", type=int, default=900)
    parser.add_argument("--validation-count", type=int, default=100)
    parser.add_argument("--max-tests-per-task", type=int, default=3)
    parser.add_argument(
        "--max-test-bytes-per-task",
        type=int,
        default=DEFAULT_MAX_TEST_BYTES_PER_TASK,
    )
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--revision", default=DATASET_REVISION)
    args = parser.parse_args()

    if min(
        args.train_count,
        args.validation_count,
        args.max_tests_per_task,
        args.max_test_bytes_per_task,
    ) <= 0:
        raise ValueError("train count, validation count, test count, and test byte cap must be positive")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_records = _collect_records(
        config="taco",
        split="train",
        total_rows=7436,
        target_rows=args.train_count,
        max_tests_per_task=args.max_tests_per_task,
        max_test_bytes_per_task=args.max_test_bytes_per_task,
        seed=args.seed,
        revision=args.revision,
    )
    validation_records = _collect_records(
        config="codeforces",
        split="test",
        total_rows=408,
        target_rows=args.validation_count,
        max_tests_per_task=args.max_tests_per_task,
        max_test_bytes_per_task=args.max_test_bytes_per_task,
        seed=args.seed + 1,
        revision=args.revision,
    )

    train_path = args.out_dir / "train.jsonl.gz"
    validation_path = args.out_dir / "validation.jsonl.gz"
    _write_jsonl_gzip(train_path, train_records)
    _write_jsonl_gzip(validation_path, validation_records)
    metadata = {
        "format_version": 1,
        "source": {
            "dataset": DATASET_ID,
            "license": DATASET_LICENSE,
            "revision": args.revision,
            "url": f"https://huggingface.co/datasets/{DATASET_ID}",
        },
        "selection": {
            "max_tests_per_task": args.max_tests_per_task,
            "max_test_bytes_per_task": args.max_test_bytes_per_task,
            "seed": args.seed,
            "train": {"config": "taco", "rows": args.train_count, "split": "train"},
            "validation": {"config": "codeforces", "rows": args.validation_count, "split": "test"},
        },
        "files": {
            train_path.name: _file_metadata(train_path, len(train_records)),
            validation_path.name: _file_metadata(validation_path, len(validation_records)),
        },
    }
    (args.out_dir / "source.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
