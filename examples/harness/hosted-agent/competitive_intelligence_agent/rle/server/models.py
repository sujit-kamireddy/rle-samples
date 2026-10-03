"""Observation and state models for the competitive-intelligence OpenEnv episodes."""

from __future__ import annotations

from typing import Any, Optional

from openenv.core.env_server.types import Observation, State


class CompetitiveIntelObservation(Observation):
    """What ``reset`` and ``grade`` hand back.

    ``Observation`` forbids extra fields, so everything the caller may read is
    declared. ``info`` carries the grading payload the legacy ``/grade`` route
    returned under the same name, unchanged, so a reward read here can be
    compared with one read there field by field.
    """

    task_id: Optional[str] = None
    variant: Optional[str] = None
    query: Optional[str] = None
    tool_surface: Optional[str] = None
    tools: list[str] = []
    is_success: bool = False
    info: dict[str, Any] = {}


class CompetitiveIntelState(State):
    """Episode state, as ``state`` reports it between calls."""

    task_id: Optional[str] = None
    variant: Optional[str] = None
    graded: bool = False
