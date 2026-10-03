"""Metadata builder regression tests: content guarantees and drift detection."""

from __future__ import annotations

import contextlib
import copy
import gzip
import hashlib
import importlib.util
import io
import json
import pathlib
import sys
import tomllib
import unittest
from unittest import mock


SAMPLE_ROOT = pathlib.Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "build_task_meta", SAMPLE_ROOT / "tools" / "build_task_meta.py"
)
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)

# Captured before this builder added `question`; still the hash of every other
# field plus row order, so a refactor that silently drops or reorders the
# original dataset's content is caught even though verification now also
# compares `question`.
ROWS_WITHOUT_QUESTION_SHA256 = "4408d5c4c790bc505a781c94c6a3ae31d008d43405d200fdab285794564a0dbe"


class MetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with gzip.open(builder.OUT_DIR / f"{builder.DATASET}.json.gz") as handle:
            cls.rows = json.load(handle)
        cls.raw = builder._read_tasks(builder.TARBALL)
        cls.tasks = [
            tomllib.loads(files["toml"])
            for _, files in sorted(cls.raw.items())
            if "toml" in files
        ]

    def test_original_fields_and_order_match_pre_question_baseline(self):
        original = [
            {key: value for key, value in row.items() if key != "question"}
            for row in self.rows
        ]
        canonical = json.dumps(
            original, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        self.assertEqual(len(original), 5000)
        self.assertEqual(hashlib.sha256(canonical).hexdigest(), ROWS_WITHOUT_QUESTION_SHA256)

    def test_identities_and_questions_match_explicit_upstream_source(self):
        self.assertEqual(len(self.rows), len(self.tasks))
        names = [row["task_name"] for row in self.rows]
        self.assertEqual(len(set(names)), len(names))
        self.assertEqual(names, [task["task"]["name"] for task in self.tasks])
        for index, (row, task) in enumerate(zip(self.rows, self.tasks)):
            with self.subTest(index=index):
                self.assertIsInstance(row["question"], str)
                self.assertTrue(row["question"].strip())
                self.assertEqual(row["question"], task["verifier"]["env"]["QUESTION"])

    def test_builder_reproduces_the_committed_file(self):
        with mock.patch.object(builder, "_read_tasks", return_value=self.raw):
            self.assertEqual(builder.build(), self.rows)

    def test_verification_rejects_drift_in_every_field(self):
        for field in self.rows[0]:
            with self.subTest(field=field):
                changed = copy.deepcopy(self.rows[:1])
                changed[0][field] = None
                with contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(SystemExit, "metadata drift"):
                        builder._verify_rows(self.rows[:1], changed)

    def test_verification_rejects_missing_extra_fields_and_reordering(self):
        missing = dict(self.rows[0])
        missing.pop("question")
        extra = dict(self.rows[0], unexpected=None)
        for changed in ([missing], [extra], self.rows[:2][::-1]):
            with self.subTest(changed_length=len(changed)):
                with contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(SystemExit, "metadata drift"):
                        builder._verify_rows(self.rows[: len(changed)], changed)
        with self.assertRaisesRegex(SystemExit, "length drift"):
            builder._verify_rows(self.rows, self.rows[:-1])

    def test_write_guard_allows_only_question_changes(self):
        original = {key: value for key, value in self.rows[0].items() if key != "question"}
        builder._verify_rows([original], self.rows[:1], allow_question_update=True)
        for field in original:
            with self.subTest(field=field):
                changed = dict(self.rows[0])
                changed[field] = None
                with contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(SystemExit, "metadata drift"):
                        builder._verify_rows([original], [changed], allow_question_update=True)

    def test_write_refuses_field_drift_before_opening_output(self):
        changed = copy.deepcopy(self.rows)
        changed[0]["has_pii"] = not changed[0]["has_pii"]
        with (
            mock.patch.object(builder, "build", return_value=changed),
            mock.patch.object(builder, "_summarise"),
            mock.patch.object(sys, "argv", ["build_task_meta.py", "--write"]),
            mock.patch.object(gzip, "GzipFile", wraps=gzip.GzipFile) as gzip_file,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            with self.assertRaisesRegex(SystemExit, "metadata drift"):
                builder.main()
            self.assertEqual(gzip_file.call_count, 1)
            self.assertNotIn("w", gzip_file.call_args.args[1])

    def test_missing_blank_or_nonstring_question_fails_explicitly(self):
        for assignment in ("", 'QUESTION = ""', 'QUESTION = "   "', "QUESTION = 42"):
            with self.subTest(assignment=assignment):
                raw = {
                    "task": {
                        "toml": (
                            '[task]\nname = "test/task"\n'
                            'description = "Not a question source"\n'
                            f"[verifier.env]\n{assignment}\n"
                        )
                    }
                }
                with mock.patch.object(builder, "_read_tasks", return_value=raw):
                    with self.assertRaisesRegex(SystemExit, "missing or empty verifier QUESTION"):
                        builder.build()

    def test_cli_verification_checks_questions_and_every_field(self):
        for field in self.rows[0]:
            with self.subTest(field=field):
                changed = copy.deepcopy(self.rows)
                changed[0][field] = None
                with (
                    mock.patch.object(builder, "build", return_value=changed),
                    mock.patch.object(builder, "_summarise"),
                    mock.patch.object(sys, "argv", ["build_task_meta.py", "--verify"]),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    with self.assertRaisesRegex(SystemExit, "metadata drift"):
                        builder.main()


if __name__ == "__main__":
    unittest.main()
