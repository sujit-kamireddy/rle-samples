"""Observation and explicit per-episode state for ``math_rl``."""

from __future__ import annotations

from typing import Any, Optional

from openenv.core.env_server.types import Observation, State
from pydantic import Field


class MathObservation(Observation):
    """Problem messages at reset and grading details at termination."""

    messages: list[dict[str, Any]] = Field(default_factory=list)
    problem_id: Optional[str] = None
    helper_called: bool = False
    helper_call_count: int = 0


class MathState(State):
    """Same-instance episode state, including optional MCP helper usage."""

    instance_id: Optional[str] = None
    row_index: Optional[int] = None
    split: Optional[str] = None
    active: bool = False
    graded: bool = False
    helper_called: bool = False
    helper_call_count: int = 0
    last_candidate_expression: Optional[str] = None
    last_comparison_expression: Optional[str] = None
    last_equivalent: Optional[bool] = None
