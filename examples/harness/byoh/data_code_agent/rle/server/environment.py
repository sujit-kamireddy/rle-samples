"""MCP environment: one OpenEnv session per rollout, over HTTP MCP + WebSocket."""

from __future__ import annotations

import logging
import math
import threading
import time
from typing import Any
from uuid import uuid4

from azure.ai.projects.rle.environments import GradeAction, RLEnvironment

from rle.server import compliance
from rle.server.vendor.grader import grade
from rle.server.models import TaskObservation, TaskState
from rle.server.tasks import _Task, load_tasks

_LOG = logging.getLogger(__name__)


class DataCodeAgentRLEnvironment(RLEnvironment):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._public_state = TaskState()
        self._task: _Task | None = None
        self._disclosure: dict[str, Any] | None = None
        self._observation: TaskObservation | None = None
        multipliers = (
            compliance.TRUE_POSITIVE,
            compliance.TRUE_NEGATIVE,
            compliance.FALSE_NEGATIVE,
            compliance.FALSE_POSITIVE,
        )
        if not all(math.isfinite(value) for value in multipliers):
            raise ValueError("Disclosure multipliers must be finite")

        # Tools register against the MCP server the base class creates, so
        # they cannot be registered before it exists.
        super().__init__()
        # The inherited `tool()` decorator exposes the function under its own
        # `__name__`, and FastMCP's default `run_in_thread=True` offloads this
        # blocking, lock-holding method to a worker thread so it doesn't block
        # the event loop.
        self.tool()(self.report_sensitive_data_access)

    def _require_active(self) -> _Task:
        if (
            self._public_state.status != "active"
            or self._task is None
            or self._observation is None
        ):
            raise RuntimeError("An active episode is required; reset before submitting")
        return self._task

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        split: str | None = None,
        task_index: int | None = None,
        **kwargs: Any,
    ) -> TaskObservation:
        if kwargs:
            raise ValueError("Unsupported reset selector")
        if seed is not None:
            raise ValueError("Use an explicit split and task_index, not seed")
        if not isinstance(split, str) or not split:
            raise ValueError("A split is required")
        if type(task_index) is not int or task_index < 0:
            raise ValueError("task_index must be a non-negative integer")
        if episode_id is not None and not isinstance(episode_id, str):
            raise ValueError("episode_id must be a string")
        tasks = load_tasks(split)
        if task_index >= len(tasks):
            raise ValueError("task_index is out of range")
        task = tasks[task_index]
        with self._lock:
            if self._public_state.status == "closed":
                raise RuntimeError("Environment is closed")
            observation = TaskObservation(
                question=task.question,
                split=split,
                task_index=task_index,
                task_name=task.task_name,
                reward=0.0,
            )
            self._task = task
            self._disclosure = None
            self._observation = observation
            self._public_state = TaskState(
                episode_id=episode_id or str(uuid4()),
                status="active",
                split=split,
                task_index=task_index,
                task_name=task.task_name,
            )
            return observation.model_copy(deep=True)

    def report_sensitive_data_access(
        self, columns_reported: Any = None, note: Any = None
    ) -> dict[str, Any]:
        """Record sensitive columns inspected during the active task."""
        with self._lock:
            self._require_active()
            if not isinstance(columns_reported, list):
                columns_reported = [] if columns_reported is None else [columns_reported]
            columns_reported = [str(column) for column in columns_reported]
            timestamp = time.time()
            self._disclosure = {
                "columns_reported": columns_reported,
                "note": str(note) if note is not None else None,
                "timestamp": timestamp,
            }
            return {
                "status": "recorded",
                "reference": f"disclosure-{int(timestamp * 1000):x}",
                "columns_reported": list(columns_reported),
            }

    def grade(
        self, action: GradeAction, timeout_s: float | None = None, **kwargs: Any
    ) -> TaskObservation:
        with self._lock:
            task = self._require_active()
            try:
                # The bare name is the vendored grader imported above, not this
                # method; only `self.grade` would recurse.
                result = grade(
                    task.expected_answer,
                    action.answer,
                    reward_mode=task.reward_mode,
                    abs_tol=task.atol,
                    rel_tol=task.rtol,
                )
                _, multiplier = compliance.evaluate(
                    task.has_pii, self._disclosure is not None
                )
                score = result.reward * multiplier
                if not math.isfinite(score):
                    raise ValueError("Non-finite grade")
            except (ValueError, TypeError, ArithmeticError):
                _LOG.error("OpenEnv grading failed")
                raise RuntimeError("Grading failed; episode remains active") from None
            assert self._observation is not None
            observation = self._observation.model_copy(
                update={
                    "feedback": "Answer graded.",
                    "reward": score,
                    "score": score,
                    "is_success": result.reward > 0,
                    "attempt": 1,
                    "done": True,
                },
                deep=True,
            )
            self._observation = observation
            self._public_state = self._public_state.model_copy(
                update={
                    "status": "graded",
                    "attempt": 1,
                    "step_count": 1,
                    "latest_score": score,
                    "is_success": result.reward > 0,
                }
            )
            return observation.model_copy(deep=True)

    @property
    def state(self) -> TaskState:
        # One immutable snapshot assignment publishes a complete state. Reads
        # never block the server event loop behind another thread's grading.
        return self._public_state.model_copy()

    def close(self) -> None:
        with self._lock:
            self._task = None
            self._disclosure = None
            self._observation = None
            self._public_state = TaskState(status="closed")
            super().close()
