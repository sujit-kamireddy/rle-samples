"""Focused lifecycle, privacy, and thread-isolation tests for local OpenEnv."""

from __future__ import annotations

import json
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from fastapi.testclient import TestClient
from pydantic import ValidationError

from rle.server import compliance
from rle.server.app import app, build_app
from rle.server.environment import (
    GradeAction,
    DataCodeAgentRLEnvironment,
    TaskState,
    grade,
    load_tasks,
)

SPLIT = "FineEnvs/data-agent-harbor-train"


class EnvironmentTests(unittest.TestCase):
    def make_env(self, index: int | None = 35) -> DataCodeAgentRLEnvironment:
        env = DataCodeAgentRLEnvironment()
        self.addCleanup(env.close)
        if index is not None:
            env.reset(split=SPLIT, task_index=index)
        return env

    def test_environment_module_is_the_service_package(self):
        self.assertEqual(
            type(self.make_env(index=None)).__module__,
            "rle.server.environment",
        )

    def test_grade_action_is_strict_and_cannot_override_task(self):
        for payload in (
            {},
            {"answer": None},
            {"answer": 123},
            {"answer": "x", "task_index": 30},
            {"answer": "x", "reward": 1},
            {"type": "call_tool", "name": "grade"},
        ):
            with self.subTest(payload=payload), self.assertRaises(ValidationError):
                GradeAction.model_validate(payload)

    def test_uninitialized_terminal_and_closed_guards(self):
        env = self.make_env(None)
        self.assertEqual(env.state.status, "uninitialized")
        for operation in (
            lambda: env.step(GradeAction(answer="x")),
            lambda: env.report_sensitive_data_access([], None),
        ):
            with self.assertRaisesRegex(RuntimeError, "active episode"):
                operation()
        env.reset(split=SPLIT, task_index=35)
        observation = env.step(GradeAction(answer="incorrect"))
        self.assertTrue(observation.done)
        self.assertEqual(observation.reward, 0)
        self.assertEqual(env.state.attempt, 1)
        for operation in (
            lambda: env.step(GradeAction(answer="EstimatedSalary")),
            lambda: env.report_sensitive_data_access([], None),
        ):
            with self.assertRaisesRegex(RuntimeError, "active episode"):
                operation()
        env.close()
        env.close()
        self.assertEqual(env.state.status, "closed")
        with self.assertRaisesRegex(RuntimeError, "closed"):
            env.reset(split=SPLIT, task_index=35)

    def test_reset_rejects_selectors_atomically(self):
        env = self.make_env()
        env.report_sensitive_data_access(["income"], None)
        before = env.state
        invalid = [
            {},
            {"split": SPLIT},
            {"split": SPLIT, "task_index": -1},
            {"split": SPLIT, "task_index": True},
            {"split": SPLIT, "task_index": "35"},
            {"split": SPLIT, "task_index": len(load_tasks(SPLIT))},
            {"split": "../invalid", "task_index": 0},
            {"split": SPLIT, "task_index": 35, "index": 35},
            {"split": SPLIT, "task_index": 35, "seed": 0},
        ]
        for selectors in invalid:
            with self.subTest(selectors=selectors), self.assertRaises(ValueError):
                env.reset(**selectors)
            self.assertEqual(env.state, before)
        self.assertEqual(env.step(GradeAction(answer="EstimatedSalary")).reward, 1)

    def test_reset_clears_only_its_instance_and_returns_detached_data(self):
        first, second = self.make_env(), self.make_env()
        for env in (first, second):
            env.report_sensitive_data_access(["income"], None)
        previous_id = first.state.episode_id
        reset = first.reset(split=SPLIT, task_index=35)
        reset.metadata["caller"] = "modified"
        reset.question = "modified"
        self.assertNotEqual(first.state.episode_id, previous_id)
        self.assertEqual(first.state.attempt, 0)
        grade_one = first.step(GradeAction(answer="EstimatedSalary"))
        self.assertEqual(grade_one.reward, 0.5)
        self.assertNotIn("caller", grade_one.metadata)
        self.assertNotEqual(grade_one.question, "modified")
        self.assertEqual(second.step(GradeAction(answer="EstimatedSalary")).reward, 1)

    def test_correctness_success_is_independent_of_zero_multiplier(self):
        env = self.make_env()
        with patch.object(compliance, "FALSE_NEGATIVE", 0):
            result = env.step(GradeAction(answer="EstimatedSalary"))
        self.assertEqual(result.reward, 0)
        self.assertEqual(result.score, 0)
        self.assertTrue(result.is_success)
        self.assertTrue(env.state.is_success)

    def test_grader_failure_is_explicit_safe_and_does_not_end_episode(self):
        env = self.make_env()
        with (
            patch("rle.server.environment.grade", side_effect=ValueError("private")),
            self.assertLogs("rle.server.environment", level="ERROR") as logs,
            self.assertRaisesRegex(RuntimeError, "Grading failed") as error,
        ):
            env.step(GradeAction(answer="x"))
        self.assertNotIn("private", str(error.exception))
        self.assertNotIn("private", str(logs.output))
        self.assertEqual(env.state.status, "active")
        self.assertEqual(env.state.attempt, 0)

    def test_public_snapshots_do_not_expose_private_records(self):
        env = self.make_env()
        env.report_sensitive_data_access(["income"], "private note sentinel")
        result = env.step(GradeAction(answer="incorrect"))
        forbidden = {
            "expected_answer", "has_pii", "pii_categories", "atol", "rtol",
            "reward_mode", "disclosure", "multiplier", "verdict",
        }
        for model in (env.state, result):
            data = model.model_dump()
            self.assertFalse(forbidden.intersection(data))
            self.assertNotIn("private note sentinel", json.dumps(data))
        with self.assertRaises(ValidationError):
            env.state.status = "active"
        self.assertEqual(env.state.status, "graded")
        with self.assertRaises(ValidationError):
            TaskState(expected_answer="forbidden")

    def test_wrong_task_answer_cannot_change_selected_task(self):
        env = self.make_env(30)
        result = env.step(GradeAction(answer="EstimatedSalary"))
        self.assertEqual(result.task_index, 30)
        self.assertEqual(result.reward, 0)
        self.assertFalse(result.is_success)

    def test_metadata_is_shared_read_only_not_mutable_episode_state(self):
        self.assertIs(load_tasks(SPLIT), load_tasks(SPLIT))
        with self.assertRaises(ValidationError):
            load_tasks(SPLIT)[0].question = "changed"
        first, second = self.make_env(), self.make_env()
        self.assertIsNot(first.mcp_server, second.mcp_server)
        self.assertIsNot(first._lock, second._lock)

    def test_same_instance_serialization_does_not_block_other_instances(self):
        for mutation in ("disclose", "reset", "close"):
            with self.subTest(mutation=mutation):
                first, second = self.make_env(), self.make_env(30)
                grading, release, mutation_started = (
                    threading.Event(), threading.Event(), threading.Event()
                )

                def blocked_grade(*args, **kwargs):
                    if args[1] == "EstimatedSalary":
                        grading.set()
                        if not release.wait(10):
                            raise TimeoutError("Test did not release grading")
                    return grade(*args, **kwargs)

                def mutate():
                    mutation_started.set()
                    if mutation == "disclose":
                        return first.report_sensitive_data_access(["income"], None)
                    if mutation == "reset":
                        return first.reset(split=SPLIT, task_index=30)
                    return first.close()

                with (
                    patch("rle.server.environment.grade", side_effect=blocked_grade),
                    ThreadPoolExecutor(max_workers=3) as pool,
                ):
                    try:
                        pending_grade = pool.submit(
                            first.step, GradeAction(answer="EstimatedSalary")
                        )
                        self.assertTrue(grading.wait(5))
                        pending_mutation = pool.submit(mutate)
                        self.assertTrue(mutation_started.wait(5))
                        independent = pool.submit(
                            second.step, GradeAction(answer="m4.large")
                        )
                        self.assertEqual(independent.result(timeout=5).reward, 1)
                        self.assertEqual(first.state.status, "active")
                        self.assertFalse(pending_mutation.done())
                    finally:
                        release.set()
                    self.assertEqual(pending_grade.result(timeout=5).reward, 0.5)
                    if mutation == "disclose":
                        with self.assertRaises(RuntimeError):
                            pending_mutation.result(timeout=5)
                    else:
                        pending_mutation.result(timeout=5)
                        expected = "active" if mutation == "reset" else "closed"
                        self.assertEqual(first.state.status, expected)

    def test_concurrent_grades_have_exactly_one_terminal_submission(self):
        env = self.make_env()
        barrier = threading.Barrier(2)

        def submit():
            barrier.wait(timeout=5)
            try:
                return env.step(GradeAction(answer="EstimatedSalary")).done
            except RuntimeError:
                return False

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: submit(), range(2)))
        self.assertEqual(sorted(results), [False, True])
        self.assertEqual(env.state.attempt, 1)


class AppTests(unittest.TestCase):
    def test_build_app_rejects_non_finite_positive_limits(self):
        for capacity, timeout in ((0, 1), (True, 1), (2, 0), (2, float("nan"))):
            with self.subTest(capacity=capacity, timeout=timeout):
                with self.assertRaises(ValueError):
                    build_app(max_concurrent_envs=capacity, session_timeout=timeout)

    def test_app_falls_back_to_defaults_when_env_unset(self):
        with patch.dict("os.environ", {}, clear=True):
            with TestClient(app()) as client:
                rpc = lambda request_id: client.post(
                    "/mcp",
                    json={
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "method": "openenv/session/create",
                        "params": {},
                    },
                ).json()
                first, second = rpc(1), rpc(2)
        self.assertNotIn("error", first)
        self.assertEqual(second["error"]["data"]["max_sessions"], 1)

    def test_app_still_rejects_invalid_env_override(self):
        with patch.dict(
            "os.environ", {"OPENENV_MAX_CONCURRENT_ENVS": "0"}, clear=True
        ), self.assertRaises(ValueError):
            app()

    def test_schema_is_grade_action_and_safe_state(self):
        with TestClient(build_app(max_concurrent_envs=2, session_timeout=60)) as client:
            response = client.get("/schema")
            self.assertEqual(response.status_code, 200)
            schema = response.json()
        self.assertIn("answer", schema["action"]["properties"])
        self.assertFalse(schema["action"]["additionalProperties"])
        self.assertNotIn("expected_answer", json.dumps(schema))
        self.assertNotIn("has_pii", json.dumps(schema))


if __name__ == "__main__":
    unittest.main()
