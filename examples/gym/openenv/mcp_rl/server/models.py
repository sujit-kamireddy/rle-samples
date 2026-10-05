"""Observation and state models for the deterministic arithmetic episode."""

from __future__ import annotations

from openenv.core.env_server.types import Observation, State


class ArithmeticObservation(Observation):
    """The task shown at reset and the terminal grading result."""

    prompt: str | None = None
    left: int | None = None
    right: int | None = None
    tool_called: bool = False
    is_success: bool = False
    feedback: str | None = None


class ArithmeticState(State):
    """Public episode state, including proof of an MCP call on this instance."""

    instance_id: str | None = None
    left: int | None = None
    right: int | None = None
    expected_answer: int | None = None
    active: bool = False
    tool_called: bool = False
    tool_call_count: int = 0
    last_tool_result: int | None = None
    graded: bool = False
