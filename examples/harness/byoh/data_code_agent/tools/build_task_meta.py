"""Rebuilds private task metadata and public questions from the upstream tarball.

`rle_deprecated/server/env.py` grades a rollout itself rather than trusting a score
relayed back through the harness, so it needs each task's expected answer and
reward mode locally. That is what this script bakes.

It also bakes the sensitive-data label the compliance check grades against
(`has_pii`). Labelling happens here, offline and once, because the CSVs are not
available at grading time -- `pull_bucket.py` pulls them into the harness
container when the rollout starts, and they are gone by the time `/grade` runs.

Ordering is load-bearing. A task is addressed by integer `task_index`, and
that index is the position in the directory listing sorted by task directory
name, so entries are emitted in exactly that order and `--verify` guards it.
The default output remains the legacy answer key without questions; verification
checks its original grading fields, permitting intentional label updates.
With --openenv, only the verifier's explicit QUESTION is added, never private
metadata. Missing or blank questions are errors. This separate output must match
every legacy field and row in order, and verification also checks questions.
OpenEnv writes refuse grading/label drift rather than silently changing it.

Usage:

    python tools/build_task_meta.py --verify            # legacy grading fields
    python tools/build_task_meta.py --write             # legacy key, no questions
    python tools/build_task_meta.py --openenv --verify  # all standalone fields
    python tools/build_task_meta.py --openenv --write   # rle question key
"""

from __future__ import annotations

import argparse
import gzip
import json
import pathlib
import re
import sys
import tarfile
import tomllib
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from pii_taxonomy import classify  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
SAMPLE_ROOT = HERE.parent
TARBALL = SAMPLE_ROOT / "agent" / "vendor" / "harbor-datasets.tar.gz"
OUT_DIR = SAMPLE_ROOT / "rle_deprecated" / "server" / "vendor" / "task-meta"
OPENENV_OUT_DIR = SAMPLE_ROOT / "rle" / "server" / "vendor" / "task-meta"
OVERRIDES = HERE / "pii_overrides.json"
DATASET = "FineEnvs__data-agent-harbor-train"

# `instruction.md` lists the task's input files as a markdown bullet list.
_FILE_LINE = re.compile(r"^- (\S+\.\w+)$", re.M)

# Legacy verification deliberately excludes labels to permit relabelling.
LEGACY_FIELDS = ("task_name", "expected_answer", "reward_mode", "atol", "rtol")


def _read_tasks(tarball: pathlib.Path) -> dict[str, dict[str, str]]:
    """Returns {task_dir: {"toml": ..., "instruction": ...}} from the tarball."""
    raw: dict[str, dict[str, str]] = {}
    with tarfile.open(tarball, "r:gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            parts = pathlib.PurePosixPath(member.name).parts
            # <dataset>/tasks/<task-dir>/<file>
            if len(parts) < 4 or parts[1] != "tasks":
                continue
            if parts[3] == "task.toml":
                key = "toml"
            elif parts[3] == "instruction.md":
                key = "instruction"
            else:
                continue
            handle = tar.extractfile(member)
            if handle is None:
                continue
            raw.setdefault(parts[2], {})[key] = handle.read().decode("utf-8")
    return raw


def build(*, openenv: bool = False) -> list[dict[str, object]]:
    if not TARBALL.exists():
        sys.exit(f"missing dataset tarball: {TARBALL}")

    overrides = json.loads(OVERRIDES.read_text()) if OVERRIDES.exists() else {}
    raw = _read_tasks(TARBALL)

    entries: list[dict[str, object]] = []
    for task_dir in sorted(raw):
        files = raw[task_dir]
        if "toml" not in files:
            continue
        meta = tomllib.loads(files["toml"])
        verifier_env = meta.get("verifier", {}).get("env", {})
        task_meta = meta.get("metadata", {})
        task_name = meta["task"]["name"]
        question = verifier_env.get("QUESTION")
        if openenv and (not isinstance(question, str) or not question.strip()):
            sys.exit(f"missing or empty verifier QUESTION for task: {task_name}")

        instruction = files.get("instruction", "")
        categories = classify(
            task_meta.get("kaggle_dataset_name", ""),
            verifier_env.get("QUESTION", "") or meta["task"].get("description", ""),
            " ".join(_FILE_LINE.findall(instruction)),
        )
        has_pii = bool(categories)
        label_source = "heuristic"

        override = overrides.get(task_name)
        if override is not None:
            has_pii = bool(override["has_pii"])
            label_source = "verified"

        entries.append(
            {
                "task_name": task_name,
                **({"question": question} if openenv else {}),
                "expected_answer": verifier_env.get("EXPECTED_ANSWER", ""),
                "reward_mode": verifier_env.get("REWARD_MODE", ""),
                # Absent means "use the grader's default tolerance". Baking the
                # default in keeps this file self-describing; `env.py` applies
                # the same 1e-3 when the field is empty.
                "atol": verifier_env.get("ATOL", "1e-3"),
                "rtol": verifier_env.get("RTOL", "1e-3"),
                "has_pii": has_pii,
                "pii_categories": categories,
                "pii_label_source": label_source,
            }
        )
    return entries


def _verify_rows(
    committed: list[dict[str, object]],
    entries: list[dict[str, object]],
    *,
    allow_question_update: bool = False,
    fields: tuple[str, ...] | None = None,
) -> None:
    """Compare selected fields (all by default), preserving row order."""
    if len(committed) != len(entries):
        sys.exit(f"length drift: committed {len(committed)} vs built {len(entries)}")
    missing = object()
    drift = [
        (index, field)
        for index, (old, new) in enumerate(zip(committed, entries))
        for field in (fields if fields is not None else sorted(old.keys() | new.keys()))
        if not (allow_question_update and field == "question")
        and old.get(field, missing) != new.get(field, missing)
    ]
    if drift:
        for index, field in drift[:10]:
            print(f"  index {index} {field}: committed value differs from built value")
        sys.exit(f"task metadata drift in {len(drift)} field(s); no file written")


def _summarise(entries: list[dict[str, object]]) -> None:
    total = len(entries)
    positive = sum(1 for e in entries if e["has_pii"])
    verified = sum(1 for e in entries if e["pii_label_source"] == "verified")
    print(f"tasks            : {total}")
    print(f"sensitive        : {positive} ({positive / total * 100:.1f}%)")
    print(f"hand-verified    : {verified}")
    by_split: Counter[str] = Counter()
    pii_split: Counter[str] = Counter()
    for e in entries:
        split = str(e["task_name"]).split("/")[0]
        by_split[split] += 1
        if e["has_pii"]:
            pii_split[split] += 1
    print("per split        :")
    for split, count in sorted(by_split.items()):
        print(f"  {split:22s} {count:5d} tasks  {pii_split[split]:5d} sensitive")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="regenerate the key")
    parser.add_argument("--verify", action="store_true", help="diff against the key")
    parser.add_argument(
        "--openenv", action="store_true",
        help="target rle with required questions and full legacy-field parity",
    )
    args = parser.parse_args()
    if not (args.write or args.verify):
        parser.error("pass --write or --verify")

    entries = build(openenv=args.openenv)
    _summarise(entries)
    out_dir = OPENENV_OUT_DIR if args.openenv else OUT_DIR
    out_path = out_dir / f"{DATASET}.json.gz"

    if args.openenv:
        legacy_path = OUT_DIR / out_path.name
        if not legacy_path.exists():
            sys.exit(f"missing legacy metadata for parity verification: {legacy_path}")
        with gzip.open(legacy_path) as handle:
            legacy = json.load(handle)
        without_questions = [
            {key: value for key, value in entry.items() if key != "question"}
            for entry in entries
        ]
        _verify_rows(legacy, without_questions)

    if args.verify:
        if not out_path.exists():
            sys.exit(f"nothing to verify against: {out_path}")
        with gzip.open(out_path) as handle:
            committed = json.load(handle)
        _verify_rows(committed, entries, fields=None if args.openenv else LEGACY_FIELDS)
        fields_label = "every field" if args.openenv else "grading fields"
        print(f"\ntask metadata matches the committed file (order and {fields_label})")
        return 0

    if args.openenv and out_path.exists():
        with gzip.open(out_path) as handle:
            committed = json.load(handle)
        _verify_rows(committed, entries, allow_question_update=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    # mtime=0 so rebuilding identical content produces an identical file and
    # does not show up as a spurious diff.
    with gzip.GzipFile(out_path, "wb", mtime=0) as handle:
        handle.write(json.dumps(entries).encode("utf-8"))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
