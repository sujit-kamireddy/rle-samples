"""A deterministic MCP Gym environment built on the Azure SDK."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from azure.ai.projects.rle.environments import GradeAction, RLEnvironment

from .models import ArithmeticObservation, ArithmeticState

LEFT = 2
RIGHT = 3
EXPECTED_ANSWER = LEFT + RIGHT


class ArithmeticRLEnvironment(RLEnvironment):
    """One addition task with a typed MCP tool and gated final reward."""

    def __init__(self) -> None:
        self._instance_id = str(uuid4())
        super().__init__()
        self._set_state(ArithmeticState(instance_id=self._instance_id))
        self.tool()(self.add)

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        left: int | None = None,
        right: int | None = None,
        **kwargs: Any,
    ) -> ArithmeticObservation:
        """Start the sample's single deterministic task."""
        if kwargs:
            raise ValueError(f"Unsupported reset selector(s): {sorted(kwargs)}")
        if seed is not None:
            raise ValueError("mcp_rl uses explicit left and right selectors, not seed")
        if (left, right) != (LEFT, RIGHT):
            raise ValueError("reset requires the deterministic selector left=2, right=3")
        if episode_id is not None and not isinstance(episode_id, str):
            raise ValueError("episode_id must be a string")

        self._set_state(
            ArithmeticState(
                instance_id=self._instance_id,
                episode_id=episode_id,
                left=left,
                right=right,
                expected_answer=EXPECTED_ANSWER,
                active=True,
            )
        )
        return ArithmeticObservation(
            prompt=(
                f"Use the add tool to calculate {LEFT} + {RIGHT}, then submit only "
                "the integer result for grading."
            ),
            left=left,
            right=right,
        )

    def add(self, left: int, right: int) -> dict[str, int | str]:
        """Add two integers. Call this before submitting the final answer."""
        state = self.state
        if not state.active or state.graded:
            raise RuntimeError("An active ungraded episode is required")
        if (left, right) != (state.left, state.right):
            raise ValueError(
                f"add requires the active task operands left={state.left}, right={state.right}"
            )
        result = left + right
        state.tool_called = True
        state.tool_call_count += 1
        state.last_tool_result = result
        self._set_state(state)
        return {
            "result": result,
            "instance_id": self._instance_id,
            "tool_call_count": state.tool_call_count,
        }

    def grade(
        self,
        action: GradeAction,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> ArithmeticObservation:
        """Award one only for the right answer after an MCP tool call."""
        if kwargs:
            raise ValueError(f"Unsupported grade option(s): {sorted(kwargs)}")
        state = self.state
        if not state.active or state.graded:
            raise RuntimeError("An active ungraded episode is required")

        answer = (action.answer or "").strip()
        correct = (
            state.tool_called
            and state.last_tool_result == state.expected_answer
            and answer == str(state.expected_answer)
        )
        state.graded = True
        self._set_state(state)
        return ArithmeticObservation(
            done=True,
            reward=1.0 if correct else 0.0,
            left=state.left,
            right=state.right,
            tool_called=state.tool_called,
            is_success=correct,
            feedback=(
                "Correct answer after tool execution."
                if correct
                else "Reward requires both an MCP tool call and the exact integer answer."
            ),
        )
