"""Loads the ``django/django`` SWE-bench-Verified instances this sample uses.

Downloads ``princeton-nlp/SWE-bench_Verified`` from Hugging Face and filters
it down to one repo (``django/django``) and one version bucket (``3.2``):
the largest single-repo, single-environment slice available -- 43 real,
human-verified issues that all check out and run against the exact same
Python/dependency setup (see ``../Dockerfile``), so the whole pool fits in
one Docker image without per-instance environment variance. 3 of the 43
have a malformed ``FAIL_TO_PASS`` entry upstream (the test's docstring
instead of its "test_name (module.Class)" id) and are dropped, leaving 40
-- see ``get_django_instances``. See ``build_dataset.py`` for how these rows
become the checked-in snapshot.
"""

from __future__ import annotations

import json
import re
from typing import Any, cast

from datasets import Dataset, load_dataset

REPO = "django/django"
VERSION = "3.2"


def _to_test_label(test_id: str) -> str:
    """Converts a unittest-style id (``"test_name (module.path.ClassName)"``)
    into the dotted label ``tests/runtests.py`` accepts
    (``"module.path.ClassName.test_name"``)."""
    match = re.match(r"^(\S+)\s+\(([\w.]+)\)$", test_id.strip())
    if not match:
        raise ValueError(f"unrecognized test id format: {test_id!r}")
    name, class_path = match.groups()
    return f"{class_path}.{name}"


def _row_to_record(row: dict[str, Any]) -> dict[str, Any]:
    fail_to_pass = [_to_test_label(t) for t in json.loads(row["FAIL_TO_PASS"])]
    return {
        "instance_id": row["instance_id"],
        "base_commit": row["base_commit"],
        "problem_statement": row["problem_statement"],
        "test_patch": row["test_patch"],
        "fail_to_pass": fail_to_pass,
    }


def get_django_instances() -> list[dict[str, Any]]:
    dataset = cast(
        Dataset, load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    )
    dataset = dataset.filter(lambda r: r["repo"] == REPO and r["version"] == VERSION)
    records = []
    for row in dataset:
        try:
            records.append(_row_to_record(row))
        except ValueError as e:
            # A handful of SWE-bench-Verified instances have a malformed
            # FAIL_TO_PASS entry (the test's docstring instead of its
            # "test_name (module.Class)" id -- a known upstream data quirk,
            # not something this sample can recover a real test id from).
            # Skip them rather than baking an instance nothing can grade.
            print(f"skipping {row['instance_id']}: {e}")
    return records
