"""Bakes the slim task index the harness runs from.

Why a separate index instead of the whole suite
-----------------------------------------------
`server/opencode_direct.py` runs `opencode` in the harness container and `../rle` does the
grading, so the only things it needs per task are the instruction text, the bucket coordinates
for the input files, and the agent timeout. The upstream suite is 216MB extracted.

Shipping the rest is not merely wasteful, it is unsafe. Every `task.toml` carries
`metadata.gold_answer` and `verifier.env.EXPECTED_ANSWER`, and every `instruction.md` embeds its
question verbatim -- which `task.toml`'s `task.description` repeats. That pair is a join key: an
agent with a shell can `grep -rlF "<its own question>"` the suite, land on its own task directory,
and read its own gold answer without doing any analysis. Measured on this suite, that resolves the
agent's own answer for 25 of 25 sampled tasks.

Filesystem permissions could hide the suite, but not shipping the answers at all is the stronger
and simpler guarantee: the grading key lives only in `../rle`'s container, which hands the agent
no shell.

Output
------
`harness/vendor/task-index.json.gz`: a gzipped JSON array, one entry per task, ordered by task
directory name with dotted directories skipped -- a task's index is its identity, and `../rle`'s
answer key is keyed by it, so the order is load-bearing rather than cosmetic.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
import tarfile
import tomllib
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
TARBALL = REPO_ROOT / "harness" / "vendor" / "harbor-datasets.tar.gz"
INDEX = REPO_ROOT / "harness" / "vendor" / "task-index.json.gz"

# Docker cannot COPY above its build context (`harness/`), so the fetcher is mirrored into
# `vendor/` for the image to pick up. `tools/pull_bucket.py` stays the source of truth; this copy is
# generated, and `--verify` fails when the two drift.
FETCHER_SRC = REPO_ROOT / "tools" / "pull_bucket.py"
FETCHER_DST = REPO_ROOT / "harness" / "vendor" / "pull_bucket.py"

# Keys that must never reach the harness container. Checked against the emitted index rather than
# assumed from the code that builds it, so that adding a field to the row below cannot quietly
# start shipping the answer key.
FORBIDDEN_KEYS = ("gold_answer", "EXPECTED_ANSWER")

DEFAULT_AGENT_TIMEOUT_SEC = 600.0


def _task_rows(tf: tarfile.TarFile) -> list[dict[str, Any]]:
    """Builds one row per task, ordered by task directory name.

    Task directories sort by name and dotted ones are skipped; this
    mirrors that rather than trusting tar member order, which follows however the archive was
    written.
    """
    tasks: dict[str, dict[str, tarfile.TarInfo]] = {}
    for member in tf.getmembers():
        parts = Path(member.name).parts
        if len(parts) < 3 or parts[1] != "tasks":
            continue
        name = parts[2]
        if name.startswith("."):
            continue
        leaf = "/".join(parts[3:])
        if leaf in ("task.toml", "instruction.md"):
            tasks.setdefault(name, {})[leaf] = member

    rows: list[dict[str, Any]] = []
    for name in sorted(tasks):
        members = tasks[name]
        missing = {"task.toml", "instruction.md"} - members.keys()
        if missing:
            raise SystemExit(f"task {name} is missing {sorted(missing)}")
        config = tomllib.loads(tf.extractfile(members["task.toml"]).read().decode())
        env = config.get("environment", {}).get("env", {})
        rows.append(
            {
                "name": name,
                "instruction": tf.extractfile(members["instruction.md"]).read().decode(),
                "bucket_base_url": str(env.get("BUCKET_BASE_URL", "")),
                "bucket_prefix": str(env.get("BUCKET_PREFIX", "")),
                "agent_timeout_sec": float(
                    config.get("agent", {}).get("timeout_sec", DEFAULT_AGENT_TIMEOUT_SEC)
                ),
            }
        )
    return rows


def _check(rows: list[dict[str, Any]], tf: tarfile.TarFile) -> list[str]:
    """Returns the problems that should stop a write, printing advisory findings as it goes."""
    problems: list[str] = []
    blob = json.dumps(rows)
    for key in FORBIDDEN_KEYS:
        if key in blob:
            problems.append(f"index contains {key!r}")

    for i, row in enumerate(rows):
        if not row["bucket_base_url"] or not row["bucket_prefix"]:
            problems.append(f"task {row['name']} (index {i}) has no bucket coordinates")

    # Advisory, not fatal: a handful of upstream instructions embed the prompt-generator's own
    # scratchpad, which can quote the answer. Nothing here can fix that -- the text is the task --
    # but a silent regression in how many of them there are is worth surfacing.
    leaky = _leaky_instructions(rows, tf)
    if leaky:
        print(
            f"note: {len(leaky)} instruction(s) appear to quote their own gold answer "
            f"(upstream data issue): {', '.join(leaky[:5])}",
            file=sys.stderr,
        )
    return problems


def _leaky_instructions(rows: list[dict[str, Any]], tf: tarfile.TarFile) -> list[str]:
    golds: dict[str, str] = {}
    for member in tf.getmembers():
        parts = Path(member.name).parts
        if len(parts) >= 4 and parts[1] == "tasks" and parts[-1] == "task.toml":
            text = tf.extractfile(member).read().decode()
            if match := re.search(r'gold_answer\s*=\s*"(.*)"', text):
                golds[parts[2]] = match.group(1)

    leaky = []
    for row in rows:
        gold = golds.get(row["name"], "")
        # Short golds ("3", "yes") collide with ordinary prose constantly; only a distinctive
        # answer appearing verbatim is evidence of a leak rather than a coincidence.
        if len(gold) >= 6 and gold in row["instruction"]:
            leaky.append(row["name"])
    return leaky


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true", help="build and write the index")
    group.add_argument(
        "--verify", action="store_true", help="rebuild and compare against the checked-in index"
    )
    args = parser.parse_args()

    with tarfile.open(TARBALL) as tf:
        rows = _task_rows(tf)
        problems = _check(rows, tf)

    if problems:
        for problem in problems:
            print(f"error: {problem}", file=sys.stderr)
        return 2

    payload = gzip.compress(json.dumps(rows).encode(), 9)

    if args.verify:
        if not INDEX.exists():
            print(f"error: {INDEX} does not exist", file=sys.stderr)
            return 2
        # Compare decompressed, not byte-for-byte: gzip output is not reproducible across zlib
        # builds, and a false mismatch here would be indistinguishable from a real one.
        if json.loads(gzip.decompress(INDEX.read_bytes())) != rows:
            print(f"error: {INDEX} is stale; re-run with --write", file=sys.stderr)
            return 2
        if not FETCHER_DST.exists() or FETCHER_DST.read_bytes() != FETCHER_SRC.read_bytes():
            print(f"error: {FETCHER_DST} differs from {FETCHER_SRC}; re-run with --write", file=sys.stderr)
            return 2
        print(f"ok: {INDEX.name} matches the suite ({len(rows)} tasks); {FETCHER_DST.name} in sync")
        return 0

    INDEX.write_bytes(payload)
    FETCHER_DST.write_bytes(FETCHER_SRC.read_bytes())
    print(f"wrote {INDEX} ({len(rows)} tasks, {len(payload) / 1e6:.2f} MB)")
    print(f"wrote {FETCHER_DST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
