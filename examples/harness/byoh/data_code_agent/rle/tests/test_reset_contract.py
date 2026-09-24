"""Pins what RLE's BYOH path actually requires of `/reset` and `/health`.

Read off vienna `origin/master`
(`EntryPoints/Services/ExecuteRollout/HttpRolloutSandboxClient.cs`,
`ExecuteRolloutService.cs`), because every assumption here has drifted at
least once:

* `ResetAsync` returns a bare `Task`. The response body is never
  deserialized -- observation, messages and reward are all discarded -- so
  only the status code decides whether the rollout survives. Any non-2xx
  raises `RolloutDependencyException` and the rollout is thrown away.
* The request body is `task.GetRawText()`: the caller's `--task` JSON
  forwarded *verbatim*, deliberately not wrapped in an OpenEnv
  `ResetRequest`. This handler therefore has to tolerate an arbitrary JSON
  object. Tightening the schema would turn a caller's own task payload into
  a 422 and kill the rollout.
* Readiness is probed at the plain `health` path before reset is attempted.
* The `_env/` control prefix is retired; vienna carries its own guard test
  against reintroducing it (`HttpRolloutSandboxClientTests.cs`).

Run from the `rle/` directory:

    python -m unittest discover -s tests -t .
"""

from __future__ import annotations

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402

from server.env import app  # noqa: E402


class ResetContractTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_reset_accepts_the_raw_task_json_rle_forwards(self) -> None:
        """RLE posts the caller's task object as-is, with no envelope."""
        payloads = [
            {},
            {"split": "FineEnvs/data-agent-harbor-train", "task_index": 7},
            {"seed": 1},
            {"unexpected": {"deeply": {"nested": True}}, "n": 3},
        ]
        for payload in payloads:
            with self.subTest(payload=payload):
                response = self.client.post("/reset", json=payload)
                self.assertEqual(
                    response.status_code,
                    200,
                    f"Non-2xx reset discards the rollout; got {response.text}",
                )

    def test_reset_accepts_a_request_with_no_body(self) -> None:
        response = self.client.post("/reset")
        self.assertEqual(response.status_code, 200)

    def test_health_is_served_on_the_plain_readiness_path(self) -> None:
        """`ReadinessProbePath` is `health`; a miss here stalls provisioning."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)

    def test_the_retired_env_control_prefix_is_not_reintroduced(self) -> None:
        stale = sorted(
            route.path
            for route in app.routes
            if getattr(route, "path", "").startswith("/_env/")
        )
        self.assertEqual(stale, [])


if __name__ == "__main__":
    unittest.main()
