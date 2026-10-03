"""Corpus gate: python -m unittest rle.tests.test_full_dataset."""

import unittest

from .parity import (
    BASELINE_SPLIT, LEGACY_ROWS_SHA256, Pair, Report, answers,
    datasets, quiet_transport, require, skip_without_legacy,
)


class FullDatasetParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        skip_without_legacy()

    def test_every_vendored_row_all_eight_cases(self):
        splits = datasets()
        expected = sum(len(rows) * 8 for rows in splits.values())
        report = Report("full_dataset", expected)
        try:
            require(BASELINE_SPLIT in splits, "Baseline split is missing")
            for split, rows in splits.items():
                hashes = report.add_dataset(split, rows)
                if split == BASELINE_SPLIT:
                    require(
                        hashes["original_fields_sha256"] == LEGACY_ROWS_SHA256,
                        "Original metadata baseline changed",
                    )
                require(bool(rows), "Empty vendored split")
            with quiet_transport(), Pair() as pair:
                for split, rows in splits.items():
                    for index, row in enumerate(rows):
                        for variant, answer in answers(row):
                            for disclose in (False, True):
                                case_id = f"{split}:{index}:{variant}:{int(disclose)}"
                                payloads = ({"columns_reported": ["synthetic_column"]},) if disclose else ()
                                left, right = pair.case(split, index, answer, payloads)
                                report.record(case_id, left, right)
                                if left["task_name"] != row["task_name"]:
                                    report.data["mismatches"].append({
                                        "case_id": case_id,
                                        "field_diffs": [{"field": "task_name", "reason": "dataset identity mismatch"}],
                                    })
                                if variant in ("gold", "normalized") and not left["is_success"]:
                                    report.data["degenerate_case_ids"].append(case_id)
                                if variant == "nonmatching" and left["is_success"]:
                                    report.data["degenerate_case_ids"].append(case_id)
            self.assertEqual(report.data["cases_executed"], expected)
            self.assertEqual(len(report.data["mismatches"]), 0, "Parity mismatches; see redacted report")
            self.assertEqual(report.data["skips"], [])
        finally:
            report.write()


if __name__ == "__main__":
    unittest.main()
