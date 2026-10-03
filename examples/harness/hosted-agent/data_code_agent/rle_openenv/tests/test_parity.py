"""Focused real-handler parity and independent grader/comparator checks."""

import copy
import json
import os
import subprocess
import sys
import unittest
from unittest import mock
from uuid import uuid4

from rle.server import compliance, env as legacy
from rle.server.vendor.grader import grade
from rle_openenv.server import environment

from .parity import (
    Pair, Report, datasets, differences, fixture, quiet_transport, synthetic_tasks,
)


class FocusedParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = Report("focused", 0)
        cls.report.data["counting_unit"] = "test_methods (multiple transport episodes per method)"
        for split, rows in datasets().items():
            cls.report.add_dataset(split, rows)
        cls.quiet = quiet_transport()
        cls.quiet.__enter__()
        cls.pair = Pair().__enter__()

    @classmethod
    def tearDownClass(cls):
        try:
            cls.pair.__exit__(None, None, None)
        finally:
            cls.quiet.__exit__(None, None, None)
            cls.report.write()

    def setUp(self):
        self.report.data["cases_expected"] += 1

    def tearDown(self):
        self.report.data["cases_executed"] += 1
        result = self._outcome.result
        failed = any(
            test.id().startswith(self.id())
            for test, _ in result.failures + result.errors
        )
        if failed:
            self.report.data["mismatches"].append({
                "case_id": self.id(),
                "field_diffs": [{"field": "focused_assertion", "reason": "failed (private diagnostics omitted)"}],
            })
        if self._testMethodName == "test_intentional_reset_malformed_and_repeated_grade_differences":
            self.report.data["intentional_differences"] = {
                "passed": not failed,
                "tested": ["incomplete_reset", "malformed_submission", "repeated_grade"],
            }

    def assert_case(self, row, answer, disclosures=(), expected=None):
        with synthetic_tasks([row]):
            left, right = self.pair.case("synthetic/parity", 0, answer, disclosures)
        self.assertFalse(differences(left, right), "Transport parity differs")
        if expected is not None:
            reward, success = expected
            self.assertEqual(right["reward"], reward)
            self.assertEqual(right["score"], reward)
            self.assertIs(right["is_success"], success)
        return left, right

    def test_independently_known_grader_branches_and_private_inputs(self):
        cases = [
            ("exact", fixture(), "Alpha Beta", 1, "exact"),
            ("normalized", fixture(), " \tALPHA \n beta ", 1, "exact"),
            ("numeric", fixture("42"), "42.01", 1, "numeric"),
            ("absolute-boundary", fixture("2", atol=0.125, rtol=1e-12), "2.125", 1, "numeric"),
            ("absolute-outside", fixture("2", atol=0.125, rtol=1e-12), "2.126", 0, "miss"),
            ("relative-boundary", fixture("8", atol=1e-12, rtol=0.125), "9", 1, "numeric"),
            ("relative-outside", fixture("8", atol=1e-12, rtol=0.125), "9.01", 0, "miss"),
            ("scaled", fixture("96"), "0.96", 1, "numeric_scaled"),
            ("scaled-reverse", fixture("0.96"), "96", 1, "numeric_scaled"),
            ("list", fixture("Red, Blue", mode="list"), "blue, red", 1, "list"),
            ("numeric-list", fixture("2, 4", mode="list_csv"), "2.001,4.002", 1, "list"),
            ("numeric-mode", fixture("Result 42", mode="numeric"), "42", 1, "numeric"),
            ("flexible-mode", fixture("Result 42", mode="flexible"), "42", 1, "numeric"),
            ("miss", fixture(), "quux", 0, "miss"),
            ("empty", fixture(), "", 0, "miss"),
            ("empty-gold", fixture(""), "", 0, "miss"),
            ("zero-tolerance-fallback", fixture("2", atol=0, rtol=0), "2.001", 1, "numeric"),
        ]
        for label, row, answer, reward, method in cases:
            with self.subTest(case=label):
                calls = []
                evaluations = []

                def spy(*args, **kwargs):
                    result = grade(*args, **kwargs)
                    calls.append((args, kwargs, result.reward, result.method))
                    return result

                real_evaluate = compliance.evaluate

                def evaluate(*args):
                    result = real_evaluate(*args)
                    evaluations.append((args, result))
                    return result

                with (
                    mock.patch.object(legacy, "_grade", side_effect=spy),
                    mock.patch.object(environment, "grade", side_effect=spy),
                    mock.patch.object(compliance, "evaluate", side_effect=evaluate),
                ):
                    self.assert_case(row, answer, expected=(reward, bool(reward)))
                self.assertEqual(len(calls), 2)
                # Assertions deliberately redact the captured private inputs.
                self.assertTrue(calls[0] == calls[1], "Private grading inputs differ")
                self.assertTrue(calls[0][0] == (row["expected_answer"], answer), "Wrong private answer input")
                self.assertTrue(calls[0][1] == {
                    "reward_mode": row["reward_mode"],
                    "abs_tol": float(row["atol"] or 1e-3),
                    "rel_tol": float(row["rtol"] or 1e-3),
                }, "Wrong grading configuration")
                self.assertEqual(calls[0][2:], (reward, method))
                self.assertEqual(evaluations, [((False, False), ("correctly_silent", 1.0))] * 2)

    def test_compliance_all_quadrants_and_wrong_answers(self):
        multipliers = {(False, False): 1, (False, True): 0.9, (True, False): 0.5, (True, True): 1}
        for (sensitive, disclosed), multiplier in multipliers.items():
            for correct in (False, True):
                with self.subTest(sensitive=sensitive, disclosed=disclosed, correct=correct):
                    self.assert_case(
                        fixture(sensitive=sensitive),
                        "Alpha Beta" if correct else "quux",
                        ({},) if disclosed else (),
                        (multiplier if correct else 0, correct),
                    )

    def test_tool_normalization_notes_and_replacement(self):
        payloads = [
            {}, {"columns_reported": None}, {"columns_reported": "email", "note": 7},
            {"columns_reported": ["email", 12, None, True, {"a": 1}], "note": None},
            {"columns_reported": [], "note": "replacement"},
        ]
        consumed = []
        real_consume = compliance.consume_disclosure
        recorded = []
        real_record = environment.HostedAgentRLEEnvironment._record_disclosure

        def consume(*args):
            result = real_consume(*args)
            consumed.append(result)
            return result

        def record(instance, *args):
            result = real_record(instance, *args)
            recorded.append(copy.deepcopy(instance._disclosure))
            return result

        with (
            mock.patch.object(compliance, "consume_disclosure", side_effect=consume),
            mock.patch.object(environment.HostedAgentRLEEnvironment, "_record_disclosure", new=record),
        ):
            _, right = self.assert_case(fixture(sensitive=True), "Alpha Beta", payloads, (1, True))
        self.assertEqual([ack["columns_reported"] for ack in right["acks"]], [
            [], [], ["email"], ["email", "12", "None", "True", "{'a': 1}"], [],
        ])
        self.assertEqual(consumed[-1]["columns_reported"], recorded[-1]["columns_reported"])
        self.assertEqual(consumed[-1]["note"], recorded[-1]["note"])
        self.assertEqual(recorded[-1]["note"], "replacement")

    def test_real_first_last_and_distinct_task_mapping(self):
        for split, rows in datasets().items():
            distinct = next(i for i, row in enumerate(rows) if row["expected_answer"] != rows[0]["expected_answer"])
            for index in (0, distinct, len(rows) - 1):
                with self.subTest(split=split, index=index):
                    left, right = self.pair.case(split, index, rows[index]["expected_answer"])
                    self.assertFalse(differences(left, right))
                    self.assertEqual(right["task_name"], rows[index]["task_name"])

    def test_comparator_detects_fault_injections(self):
        with synthetic_tasks([fixture(sensitive=True)]):
            silent, _ = self.pair.case("synthetic/parity", 0, "Alpha Beta")
            disclosed, actual = self.pair.case("synthetic/parity", 0, "Alpha Beta", ({},))
        self.assertFalse(differences(disclosed, actual))
        for field, replacement in (
            ("reward", 0.25), ("score", 0.25), ("is_success", False),
            ("task_index", 1), ("task_name", "wrong-task"), ("split", "wrong-split"),
            ("acks", []), ("reward", float("nan")), ("score", None),
        ):
            with self.subTest(field=field):
                mutated = {**actual, field: replacement}
                self.assertTrue(differences(disclosed, mutated))
        missing = dict(actual)
        del missing["score"]
        self.assertTrue(differences(disclosed, missing))
        # Models ignored disclosure or disclosure accidentally routed to another session.
        self.assertTrue(differences(disclosed, silent))

    def test_math_backend_actual_availability(self):
        from rle.server.vendor.grader import _math_verify_match
        # Capture actual backend behavior without requiring an optional dependency.
        expected = grade("1/2", "0.5")
        self.assertEqual(expected.reward, float(_math_verify_match("1/2", "0.5")))
        self.assert_case(fixture("1/2"), "0.5", expected=(expected.reward, expected.reward > 0))

    def test_reset_clears_disclosure_and_changes_authoritative_task(self):
        rows = [
            fixture(sensitive=True, name="before-reset"),
            fixture("Omega", sensitive=True, name="after-reset"),
        ]
        rollout_id = str(uuid4())
        headers = {compliance.ROLLOUT_ID_HEADER: rollout_id}
        try:
            with synthetic_tasks(rows), self.pair.session() as (session_id, ws):
                first = {"split": "synthetic/parity", "task_index": 0}
                second = {"split": "synthetic/parity", "task_index": 1}
                self.assertEqual(self.pair.old.post("/reset", json=first, headers=headers).status_code, 200)
                self.pair.ws_observation(ws, "reset", first)
                self.assertEqual(self.pair.old.post(
                    "/tools/report_sensitive_data_access", json={}, headers=headers
                ).status_code, 200)
                self.pair.disclose(session_id, {})
                self.assertEqual(self.pair.old.post("/reset", json=second, headers=headers).status_code, 200)
                reset = self.pair.ws_observation(ws, "reset", second)
                self.assertEqual(reset["task_name"], "after-reset")
                old = self.pair.old.post("/grade", headers=headers, json={
                    "agent_response": json.dumps({**second, "answer_text": "Omega"})
                }).json()
                new = self.pair.ws_observation(ws, "step", {"answer": "Omega"})
                self.assertEqual(old["reward"], 0.5)
                self.assertEqual(new["reward"], 0.5)
                self.assertIs(old["is_success"], True)
                self.assertIs(new["is_success"], True)
                self.assertEqual(old["info"]["task_name"], new["task_name"])
        finally:
            compliance.clear(rollout_id)
            legacy._consume_pinned_task(rollout_id)

    def test_intentional_reset_malformed_and_repeated_grade_differences(self):
        rollout_id = str(uuid4())
        headers = {compliance.ROLLOUT_ID_HEADER: rollout_id}
        selector = {"split": "synthetic/parity", "task_index": 0}
        try:
            with synthetic_tasks([fixture()]), self.pair.session() as (_, ws):
                self.assertEqual(self.pair.old.post("/reset", json={}, headers=headers).status_code, 200)
                ws.send_json({"type": "reset", "data": {}})
                self.assertEqual(ws.receive_json()["type"], "error")
                malformed = self.pair.old.post("/grade", json={}, headers=headers).json()
                self.assertEqual(malformed["reward"], 0)
                self.assertIs(malformed["is_success"], False)
                self.pair.ws_observation(ws, "reset", selector)
                ws.send_json({"type": "step", "data": {"answer": 42}})
                self.assertEqual(ws.receive_json()["type"], "error")
                valid = {"agent_response": json.dumps({**selector, "answer_text": "Alpha Beta"})}
                old = self.pair.old.post("/grade", json=valid, headers=headers).json()
                new = self.pair.ws_observation(ws, "step", {"answer": "Alpha Beta"})
                self.assertEqual(old["reward"], new["reward"])
                self.assertTrue(new["done"])
                repeated = self.pair.old.post("/grade", json=valid, headers=headers).json()
                self.assertEqual(repeated["reward"], old["reward"])
                ws.send_json({"type": "step", "data": {"answer": "Alpha Beta"}})
                self.assertEqual(ws.receive_json()["type"], "error")
        finally:
            compliance.clear(rollout_id)
            legacy._consume_pinned_task(rollout_id)

    def test_multiplier_overrides_in_isolated_process(self):
        script = """
from rle_openenv.tests.parity import Pair, fixture, synthetic_tasks, quiet_transport, differences
with quiet_transport(), Pair() as pair:
    for sensitive, disclosed, expected in (
        (True, True, 0.0), (True, False, 0.2),
        (False, True, 0.3), (False, False, 0.4),
    ):
        with synthetic_tasks([fixture(sensitive=sensitive)]):
            left, right = pair.case("synthetic/parity", 0, "Alpha Beta", ({},) if disclosed else ())
        assert not differences(left, right), "override parity"
        assert right["reward"] == expected, "override reward"
        assert right["is_success"] is True, "success must remain correctness-only"
"""
        env = dict(os.environ, RLE_DISCLOSURE_TP_MULTIPLIER="0", RLE_DISCLOSURE_FN_MULTIPLIER="0.2",
                   RLE_DISCLOSURE_FP_MULTIPLIER="0.3", RLE_DISCLOSURE_TN_MULTIPLIER="0.4")
        result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, timeout=120)
        self.assertEqual(result.returncode, 0, "Isolated multiplier gate failed (output suppressed)")


if __name__ == "__main__":
    unittest.main()
