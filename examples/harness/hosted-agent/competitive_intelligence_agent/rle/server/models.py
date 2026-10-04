"""Observation and state models for the competitive-intelligence OpenEnv episodes."""

from __future__ import annotations

from typing import Any, Optional

from openenv.core.env_server.types import Observation, State


class CompetitiveIntelObservation(Observation):
    """What ``reset`` and ``grade`` hand back.

    ``Observation`` forbids extra fields, so everything the caller may read is
    declared. ``info`` carries the grading payload ``grade`` builds: ``metrics``
    (the rubric's per-dimension scores), the parsed and expected verdicts,
    tool-call counts, and the agent's raw response text.

    ``prompt`` is the one field RLE renders into the agent's initial Responses
    input (see ``reset``'s docstring); every other field here is for this
    container's own callers (the test suite, ``smoke_grade.py``) and is never
    seen by the agent.
    """

    task_id: Optional[str] = None
    variant: Optional[str] = None
    query: Optional[str] = None
    prompt: Optional[str] = None
    tool_surface: Optional[str] = None
    tools: list[str] = []
    is_success: bool = False
    info: dict[str, Any] = {}


class CompetitiveIntelState(State):
    """Episode state, as ``state`` reports it between calls."""

    task_id: Optional[str] = None
    variant: Optional[str] = None
    graded: bool = False
