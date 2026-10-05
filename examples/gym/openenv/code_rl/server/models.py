"""Observation and per-episode state models for ``code_rl``."""

from __future__ import annotations

from typing import Any

from openenv.core.env_server.types import Observation, State
from pydantic import Field


class CodeObservation(Observation):
    """The selected problem or terminal grading result."""

    messages: list[dict[str, Any]] = Field(default_factory=list)
    starter_code: str | None = None
    problem_id: str | None = None
    split: str | None = None
    check_solution_calls: int = 0
    last_check_passed: bool | None = None


class CodeState(State):
    """Explicit state shared by reset, MCP tools, and final grading."""

    row_index: int | None = None
    split: str | None = None
    active: bool = False
    graded: bool = False
    check_solution_calls: int = 0
    last_check_passed: bool | None = None
    last_check_details: dict[str, Any] | None = None
