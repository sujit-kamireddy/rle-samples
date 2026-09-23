"""Rollout-scoped isolation: one container, several rollouts, no bleed.

RLE injects `x-rle-rollout-id` on both kinds of traffic it routes here --
control (`/reset`, `/grade`) and harness tool calls (`/tools/<name>`) --
resolving it from its own rollout mapping and stripping any caller-supplied
copy first. It is therefore the only key this container can correlate a
disclosure by, and the only thing keeping two concurrent rollouts'
compliance decisions apart. These tests pin that the three handlers which
touch rollout state actually key off it.

Run from the `rle/` directory:

    python -m unittest discover -s tests -t .
"""

from __future__ import annotations

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402

from server import compliance  # noqa: E402
from server.env import app  # noqa: E402

SPLIT = "FineEnvs/data-agent-harbor-train"
TOOL_PATH = "/tools/report_sensitive_data_access"

# A sensitive task, so a dropped disclosure is visible as a changed verdict.
PII_INDEX = 35
PII_ANSWER = "EstimatedSalary"


class RolloutIsolationTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)
        compliance.clear()

        from server.env import _load_task_meta

        meta = _load_task_meta(SPLIT)
        self.assertTrue(meta[PII_INDEX]["has_pii"], "PII_INDEX fixture is no longer sensitive")
        self.assertEqual(meta[PII_INDEX]["expected_answer"], PII_ANSWER)

    def tearDown(self) -> None:
        compliance.clear()

    def disclose(self, rollout_id: str) -> None:
        response = self.client.post(
            TOOL_PATH,
            json={"columns_reported": ["salary"]},
            headers={compliance.ROLLOUT_ID_HEADER: rollout_id},
        )
        self.assertEqual(response.status_code, 200)

    def grade_body(self, task_index: int, answer: str) -> dict:
        return {
            "rollout": {},
            "agent_response": (
                '{"split": "%s", "task_index": %d, "answer_text": "%s", "ok": true}'
                % (SPLIT, task_index, answer)
            ),
        }

    def test_reset_drops_only_the_named_rollouts_stale_disclosure(self) -> None:
        """A withdrawn rollout must not hand its disclosure to the next one."""
        self.disclose("stale")
        self.disclose("in-flight")

        response = self.client.post(
            "/reset",
            json={"split": SPLIT, "task_index": PII_INDEX},
            headers={compliance.ROLLOUT_ID_HEADER: "stale"},
        )
        self.assertEqual(response.status_code, 200)

        self.assertIsNone(compliance.consume_disclosure("stale"))
        self.assertIsNotNone(
            compliance.consume_disclosure("in-flight"),
            "reset cleared a rollout it was not called for",
        )

    def test_reset_without_a_rollout_id_clears_nothing(self) -> None:
        """The no-argument clear would wipe every in-flight rollout."""
        self.disclose("untouched")

        response = self.client.post("/reset", json={})
        self.assertEqual(response.status_code, 200)

        self.assertIsNotNone(compliance.consume_disclosure("untouched"))

    def test_an_unattributable_disclosure_is_refused(self) -> None:
        """Without the header there is no rollout to credit, and a shared
        fallback key would let concurrent rollouts overwrite each other."""
        response = self.client.post(TOOL_PATH, json={"columns_reported": ["salary"]})
        self.assertEqual(response.status_code, 400)
        self.assertIn(compliance.ROLLOUT_ID_HEADER, response.json()["detail"])

    def test_a_disclosure_is_not_visible_to_another_rollout(self) -> None:
        self.disclose("owner")
        self.assertIsNone(compliance.consume_disclosure("bystander"))
        self.assertIsNotNone(compliance.consume_disclosure("owner"))

    def test_grade_without_a_rollout_id_reports_unmeasured_not_missed(self) -> None:
        """Mis-plumbing must not be scored as the agent staying silent."""
        response = self.client.post("/grade", json=self.grade_body(PII_INDEX, PII_ANSWER))
        self.assertEqual(response.status_code, 200)

        body = response.json()
        disclosure = body["info"]["disclosure"]
        self.assertEqual(disclosure["verdict"], "unmeasured_no_rollout_id")
        self.assertEqual(disclosure["multiplier"], 1.0)
        self.assertEqual(body["reward"], body["info"]["answer_reward"])

    def test_grade_consumes_only_its_own_rollouts_disclosure(self) -> None:
        self.disclose("graded")
        self.disclose("other")

        response = self.client.post(
            "/grade",
            json=self.grade_body(PII_INDEX, PII_ANSWER),
            headers={compliance.ROLLOUT_ID_HEADER: "graded"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["info"]["disclosure"]["verdict"], "correctly_disclosed")

        self.assertIsNone(compliance.consume_disclosure("graded"))
        self.assertIsNotNone(
            compliance.consume_disclosure("other"),
            "grading one rollout consumed another's disclosure",
        )


if __name__ == "__main__":
    unittest.main()
