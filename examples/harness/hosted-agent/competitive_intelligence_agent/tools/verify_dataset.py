"""Audits ``job_data/`` for train/eval contamination and reports its mix.

Run from the sample root::

    python tools/verify_dataset.py

Exits non-zero if any evaluation scenario also appears in training, so this can
be wired into CI. Add ``--write`` to repair a contaminated training file in
place by dropping the offending training rows.

Why hash the scenario instead of comparing ``task_id``
------------------------------------------------------

Every row carries a unique ``task_id``, and the generator draws training and
evaluation rows from disjoint id ranges, so an id comparison finds nothing and
reports a clean split. It is measuring the wrong thing. The generator composes
each scenario from a pool of companies, rumours and document sets, and two
draws that land on the same combination produce byte-identical scenarios under
different ids. Those are the same question, and a policy that has been trained
on one has seen the other.

So the identity of a task here is its *content*: every field except
``task_id``, canonically serialised and hashed. That is what this script
compares.

What repair does, and what it deliberately does not do
------------------------------------------------------

``--write`` drops training rows, never evaluation rows. The evaluation set is
the published measuring stick for this sample -- the reward curve in
``README.md`` was measured on exactly these 120 rows -- so shrinking it would
silently change what the numbers mean and make a replication incomparable with
the original. Dropping the training twin closes the leak and leaves the
measurement intact.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JOB_DATA = ROOT / "job_data"


def load(path: Path) -> list[dict]:
    """Parses a JSONL file, keeping each row's original text under ``_line``.

    Repair drops whole lines rather than re-serialising the ones it keeps, so a
    cleaned file differs from its input only by the lines removed.
    """
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        row["_line"] = line
        rows.append(row)
    return rows


def scenario_hash(row: dict) -> str:
    """Identity of a row's scenario, ignoring the id the generator assigned it."""
    task = {k: v for k, v in row["task"].items() if k != "task_id"}
    return hashlib.sha256(json.dumps(task, sort_keys=True).encode()).hexdigest()


def base_id(row: dict) -> str:
    """The scenario id with any ``#rN`` replicate suffix removed."""
    return str(row["task"].get("task_id", "")).split("#", 1)[0]


def expected_decision(row: dict) -> str:
    """The decision the rubric expects: material, immaterial, or abstain.

    The task rows carry this as two fields rather than one verdict string:
    ``expects_abstain`` outranks ``material``, because a task built on an
    unverified rumour expects the agent to decline to rule either way no matter
    what the underlying ground truth happens to be.
    """
    task = row["task"]
    if task.get("expects_abstain"):
        return "abstain"
    return "material" if task.get("material") else "immaterial"


def describe(name: str, rows: list[dict]) -> None:
    variants = collections.Counter(r["task"].get("variant", "?") for r in rows)
    decisions = collections.Counter(expected_decision(r) for r in rows)

    print(f"\n{name}: {len(rows)} rows, {len(set(map(base_id, rows)))} scenarios, "
          f"{len({scenario_hash(r) for r in rows})} distinct")
    print("  variants:  " + ", ".join(
        f"{k} {v} ({v / len(rows):.1%})" for k, v in variants.most_common()))
    print("  decisions: " + ", ".join(
        f"{k} {v} ({v / len(rows):.1%})" for k, v in decisions.most_common()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=JOB_DATA / "train.jsonl")
    parser.add_argument("--eval", dest="eval_", type=Path,
                        default=JOB_DATA / "validation.jsonl")
    parser.add_argument("--write", action="store_true",
                        help="drop contaminated rows from the training file")
    args = parser.parse_args()

    train, evaluation = load(args.train), load(args.eval_)
    describe(f"train ({args.train.name})", train)
    describe(f"eval  ({args.eval_.name})", evaluation)

    eval_hashes = {scenario_hash(r) for r in evaluation}
    contaminated = [r for r in train if scenario_hash(r) in eval_hashes]

    print(f"\ncontamination: {len(contaminated)} training rows share a scenario "
          f"with the evaluation set")
    if not contaminated:
        print("clean: no evaluation scenario appears in training")
        return 0

    for row in sorted(contaminated, key=base_id):
        print(f"  train {base_id(row)} duplicates an evaluation scenario")

    if not args.write:
        print("\nrerun with --write to drop these training rows")
        return 1

    keep = [r for r in train if scenario_hash(r) not in eval_hashes]
    args.train.write_text("".join(r["_line"] + "\n" for r in keep))
    print(f"\nwrote {len(keep)} rows to {args.train} "
          f"({len(train) - len(keep)} dropped)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
