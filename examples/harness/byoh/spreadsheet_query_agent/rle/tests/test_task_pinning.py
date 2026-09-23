"""Tests that `/grade` grades the task RLE asked for, not the one the harness claims.

Run from the `rle/` directory:

    python -m unittest discover -s tests -t .

`/grade` reads `split`/`task_index` out of `agent_response`, which is the
harness's own report. That is enough for a harness to choose the task it is
scored on: asked for a hard task, it can run an easy one, report the easy one,
and be graded correctly against the easy one without ever faking a reward. RLE
calls `/reset` with the caller's `--task` before the harness is invoked and
injects the rollout id itself, so a `--task` carrying the selector is the one
account of the task the harness never touches.

The pin is advisory by design: callers that pass the selector only in
`--agent-input` keep grading exactly as they did before, so these tests cover
the fallback as carefully as the enforcement.
"""

from __future__ import annotations

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402

from server import compliance, env  # noqa: E402
from server.env import app  # noqa: E402

SPLIT = "FineEnvs/data-agent-harbor-train"

# Two tasks with different answers, both free of PII so the compliance
# multiplier stays 1.0 and the reward here is answer correctness alone.
INDEX_A = 30
ANSWER_A = "m4.large"
INDEX_B = 45


class TaskPinningTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)
        compliance.clear()
        env._pinned_tasks.clear()

        meta = env._load_task_meta(SPLIT)
        self.assertFalse(meta[INDEX_A]["has_pii"], "INDEX_A fixture is now sensitive")
        self.assertFalse(meta[INDEX_B]["has_pii"], "INDEX_B fixture is now sensitive")
        self.assertEqual(meta[INDEX_A]["expected_answer"], ANSWER_A)
        self.answer_b = meta[INDEX_B]["expected_answer"]
        self.assertNotEqual(
            self.answer_b, ANSWER_A, "fixtures must have different answers to tell the tasks apart"
        )

    def tearDown(self) -> None:
        compliance.clear()
        env._pinned_tasks.clear()

    def reset(self, rollout_id: str, task: dict) -> None:
        response = self.client.post(
            "/reset", json=task, headers={compliance.ROLLOUT_ID_HEADER: rollout_id}
        )
        self.assertEqual(response.status_code, 200)

    def grade(self, rollout_id: str, task_index: int, answer: str, split: str = SPLIT) -> dict:
        import json

        response = self.client.post(
            "/grade",
            json={
                "rollout": {},
                "agent_response": json.dumps(
                    {"split": split, "task_index": task_index, "answer_text": answer, "ok": True}
                ),
            },
            headers={compliance.ROLLOUT_ID_HEADER: rollout_id},
        )
        self.assertEqual(response.status_code, 200)
        return response.json()

    def test_pinned_task_that_matches_the_report_grades_normally(self):
        self.reset("r1", {"split": SPLIT, "task_index": INDEX_A})
        result = self.grade("r1", INDEX_A, ANSWER_A)
        self.assertEqual(result["reward"], 1.0)
        self.assertTrue(result["is_success"])

    def test_harness_cannot_swap_in_a_task_it_prefers(self):
        """The attack: run and report task B while RLE reset the rollout with task A.

        The answer is genuinely correct *for the task the harness reported*, so
        nothing about the answer text gives it away -- only the pin does.
        """
        self.reset("r1", {"split": SPLIT, "task_index": INDEX_A})
        result = self.grade("r1", INDEX_B, self.answer_b)
        self.assertEqual(result["reward"], 0.0)
        self.assertFalse(result["is_success"])
        self.assertEqual(result["info"]["pinned"]["task_index"], INDEX_A)
        self.assertEqual(result["info"]["reported"]["task_index"], INDEX_B)

    def test_mismatched_split_is_also_refused(self):
        self.reset("r1", {"split": SPLIT, "task_index": INDEX_A})
        result = self.grade("r1", INDEX_A, ANSWER_A, split="FineEnvs/some-other-suite")
        self.assertEqual(result["reward"], 0.0)
        self.assertEqual(result["info"]["pinned"]["split"], SPLIT)

    def test_task_without_a_selector_falls_back_to_the_report(self):
        """`--task {}` is the documented default, and must keep working."""
        self.reset("r1", {})
        result = self.grade("r1", INDEX_B, self.answer_b)
        self.assertEqual(result["reward"], 1.0)

    def test_partial_selector_pins_only_what_it_named(self):
        self.reset("r1", {"task_index": INDEX_A})
        self.assertEqual(env._pinned_tasks["r1"], {"task_index": INDEX_A})
        # A different split is not contradicted by a pin that never named one.
        result = self.grade("r1", INDEX_A, ANSWER_A)
        self.assertEqual(result["reward"], 1.0)

    def test_reset_without_a_selector_drops_an_earlier_pin(self):
        self.reset("r1", {"split": SPLIT, "task_index": INDEX_A})
        self.reset("r1", {})
        self.assertNotIn("r1", env._pinned_tasks)
        self.assertEqual(self.grade("r1", INDEX_B, self.answer_b)["reward"], 1.0)

    def test_pin_is_consumed_so_it_cannot_leak_to_the_next_rollout(self):
        """A rollout that resets and grades must not still be pinned afterwards.

        Rollout ids are unique only within a project, and one container serves
        many rollouts, so a pin left behind would be applied to whoever graded
        next under the same id.
        """
        self.reset("r1", {"split": SPLIT, "task_index": INDEX_A})
        self.grade("r1", INDEX_A, ANSWER_A)
        self.assertNotIn("r1", env._pinned_tasks)
        self.assertEqual(self.grade("r1", INDEX_B, self.answer_b)["reward"], 1.0)

    def test_a_pin_applies_only_to_its_own_rollout(self):
        self.reset("r1", {"split": SPLIT, "task_index": INDEX_A})
        self.assertEqual(self.grade("r2", INDEX_B, self.answer_b)["reward"], 1.0)
        self.assertIn("r1", env._pinned_tasks)

    def test_pin_is_consumed_even_when_the_report_is_unusable(self):
        """Otherwise a rollout whose harness crashed leaves its pin behind."""
        self.reset("r1", {"split": SPLIT, "task_index": INDEX_A})
        response = self.client.post(
            "/grade",
            json={"rollout": {}, "agent_response": "not json"},
            headers={compliance.ROLLOUT_ID_HEADER: "r1"},
        )
        self.assertEqual(response.json()["reward"], 0.0)
        self.assertNotIn("r1", env._pinned_tasks)

    def test_boolean_task_index_is_not_read_as_an_index(self):
        """`true` is an int subclass in Python; grading it as task 1 would be silent."""
        self.reset("r1", {"split": SPLIT, "task_index": True})
        self.assertEqual(env._pinned_tasks["r1"], {"split": SPLIT})

    def test_reset_without_a_rollout_id_pins_nothing(self):
        response = self.client.post("/reset", json={"split": SPLIT, "task_index": INDEX_A})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(env._pinned_tasks), 0)

    def test_pin_store_is_bounded(self):
        for i in range(env._MAX_PINNED_ROLLOUTS + 10):
            self.reset(f"r{i}", {"task_index": INDEX_A})
        self.assertLessEqual(len(env._pinned_tasks), env._MAX_PINNED_ROLLOUTS)
        # The oldest are evicted first, so live rollouts keep their pin longest.
        self.assertNotIn("r0", env._pinned_tasks)
        self.assertIn(f"r{env._MAX_PINNED_ROLLOUTS + 9}", env._pinned_tasks)


if __name__ == "__main__":
    unittest.main()
