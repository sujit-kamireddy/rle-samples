"""SDK MCP environment for one competitive-programming problem per episode."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, Optional

from azure.ai.projects.rle.environments import GradeAction, RLEnvironment

from .code_grading import check_correctness, extract_code_from_model
from .dataset import EpisodePicker, load_jsonl, reject_unknown_selectors
from .models import CodeObservation, CodeState

# Format-penalty coefficient -- see the module docstring.
FORMAT_COEF = 0.1

DEFAULT_MAX_TURNS = 2


class CodeRLEnvironment(RLEnvironment):
    """DeepCoder-style task with a typed ``check_solution`` MCP tool."""

    def __init__(
        self,
        dataset_path: Optional[str] = None,
        grading_timeout: int = 6,
        grading_overall_timeout: float = 30.0,
        max_turns: int = DEFAULT_MAX_TURNS,
        validation_dataset_path: Optional[str] = None,
    ):
        default_dataset_path = Path(__file__).resolve().parents[1] / "env_data" / "train.jsonl.gz"
        dataset_path = dataset_path or os.environ.get("CODE_RL_DATASET_PATH", str(default_dataset_path))
        self._rows = EpisodePicker(load_jsonl(dataset_path))
        # Resolved lazily (see _validation_rows()), not loaded here: most
        # callers (including every existing test) construct this with only
        # a train file on disk and never request split="validation", and
        # eagerly loading here would make that an unconditional
        # FileNotFoundError. Defaults to the matching sibling validation file
        # for either plain JSONL or the checked-in gzip snapshot.
        default_validation_name = (
            "validation.jsonl.gz" if str(dataset_path).endswith(".gz") else "validation.jsonl"
        )
        self._validation_dataset_path = validation_dataset_path or os.environ.get(
            "CODE_RL_VALIDATION_DATASET_PATH",
            str(Path(dataset_path).with_name(default_validation_name)),
        )
        self._validation_rows: Optional[EpisodePicker] = None
        self._grading_timeout = int(os.environ.get("CODE_RL_GRADING_TIMEOUT", grading_timeout))
        # Bounds total grading time across all of a sample's test cases (see
        # check_correctness's overall_timeout docstring) -- keeps a single
        # step() call well under the RLE gateway's own timeout even for
        # submissions/problems that never trip the per-test grading_timeout
        # but have many test cases. Comfortably below the ~45-120s Bad
        # Gateway/timeout window observed from the RLE data plane.
        self._grading_overall_timeout = float(
            os.environ.get("CODE_RL_GRADING_OVERALL_TIMEOUT", grading_overall_timeout)
        )
        self._max_turns = int(os.environ.get("CODE_RL_MAX_TURNS", max_turns))
        self._current_row: Optional[dict] = None
        super().__init__()
        self._set_state(CodeState())
        self.tool()(self.check_solution)

    def _rows_for_split(self, split: str) -> EpisodePicker:
        """Resolve which baked dataset file a ``reset()`` picks from.

        ``split="validation"`` picks from the matching validation JSONL
        snapshot (loaded lazily and cached on first use) instead of the
        training file. Without this, validation metrics would be computed
        against different rows of the same training file.
        """
        if split == "train":
            return self._rows
        if split == "validation":
            if self._validation_rows is None:
                self._validation_rows = EpisodePicker(load_jsonl(self._validation_dataset_path))
            return self._validation_rows
        raise ValueError(f"Unknown split {split!r}; expected 'train' or 'validation'.")

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        split: str = "train",
        **kwargs: Any,
    ) -> CodeObservation:
        reject_unknown_selectors(kwargs)
        index, row = self._rows_for_split(split).pick(seed)
        self._current_row = row
        self._set_state(
            CodeState(
                episode_id=episode_id,
                row_index=index,
                split=split,
                active=True,
            )
        )
        return CodeObservation(
            done=False,
            reward=None,
            messages=row["messages"],
            starter_code=row.get("starter_code"),
            problem_id=str(index),
            split=split,
        )

    def grade(
        self,
        action: GradeAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> CodeObservation:
        """Grade ``GradeAction.response``; prior tool use is not required."""
        return asyncio.run(self._grade_async(action, timeout_s=timeout_s, **kwargs))

    async def step_async(
        self,
        action: GradeAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> CodeObservation:
        """Grade without nesting an event loop on OpenEnv's WebSocket path."""
        if not isinstance(action, GradeAction):
            raise TypeError("Expected GradeAction for a non-MCP action")
        return await self._grade_async(action, timeout_s=timeout_s, **kwargs)

    async def _grade_async(
        self,
        action: GradeAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> CodeObservation:
        if kwargs:
            raise ValueError(f"Unsupported grade option(s): {sorted(kwargs)}")
        state = self.state
        if self._current_row is None or not state.active or state.graded:
            raise RuntimeError("An active ungraded episode is required")
        row = self._current_row
        tests = row.get("tests")
        if not tests:
            state.active = False
            state.graded = True
            self._set_state(state)
            return CodeObservation(
                done=True,
                reward=0.0,
                metadata={"error": "row has no 'tests' field to grade against"},
                check_solution_calls=state.check_solution_calls,
                last_check_passed=state.last_check_passed,
            )

        code = extract_code_from_model(action.response or "")
        has_code_block = code is not None
        passed = False
        details = None
        if has_code_block:
            try:
                passed, details = await check_correctness(
                    tests,
                    code,
                    timeout=self._grading_timeout,
                    overall_timeout=self._grading_overall_timeout,
                )
            except Exception as exc:  # pragma: no cover
                state.active = False
                state.graded = True
                self._set_state(state)
                return CodeObservation(
                    done=True,
                    reward=0.0,
                    metadata={"error": f"grading failed: {exc}"},
                    check_solution_calls=state.check_solution_calls,
                    last_check_passed=state.last_check_passed,
                )

        reward = FORMAT_COEF * ((1.0 if has_code_block else 0.0) - 1.0) + (
            1.0 if passed else 0.0
        )
        state.active = False
        state.graded = True
        state.step_count += 1
        self._set_state(state)
        return CodeObservation(
            done=True,
            reward=reward,
            metadata={"passed": passed, "format": has_code_block, "details": details},
            check_solution_calls=state.check_solution_calls,
            last_check_passed=state.last_check_passed,
        )

    async def check_solution(self, code: str) -> dict[str, Any]:
        """Execute Python code against the active problem's hidden tests."""
        state = self.state
        if self._current_row is None or not state.active or state.graded:
            raise RuntimeError("An active ungraded episode is required")
        state.check_solution_calls += 1
        if state.check_solution_calls > self._max_turns:
            self._set_state(state)
            return {
                "error": (
                    f"check_solution budget exhausted after {self._max_turns} call(s). "
                    "Submit your final answer now."
                ),
                "passed": False,
                "check_solution_calls": state.check_solution_calls,
            }
        tests = self._current_row.get("tests")
        if not tests:
            result = {"error": "row has no 'tests' field to grade against", "passed": False}
            state.last_check_passed = False
            state.last_check_details = result
            self._set_state(state)
            return {**result, "check_solution_calls": state.check_solution_calls}
        try:
            passed, details = await check_correctness(
                tests,
                code,
                timeout=self._grading_timeout,
                overall_timeout=self._grading_overall_timeout,
            )
            result = {"passed": passed, "details": details}
        except Exception as exc:  # pragma: no cover
            passed = False
            result = {"error": str(exc), "passed": False}
        state.last_check_passed = passed
        state.last_check_details = result
        state.step_count += 1
        self._set_state(state)
        return {**result, "check_solution_calls": state.check_solution_calls}
