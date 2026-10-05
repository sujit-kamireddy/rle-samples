"""Opt-in standalone app. Serve one worker; each session owns an environment."""

from __future__ import annotations

import math
import os

from fastapi import FastAPI
from azure.ai.projects.rle.environments import create_app
from openenv.core.env_server.types import ConcurrencyConfig

from .environment import (
    GradeAction,
    DataCodeAgentRLEnvironment,
    TaskObservation,
    TaskState,
)

#: One rollout per container, matching what an RLE sandbox actually schedules:
#: this environment's sessions are isolated enough to raise it, but a real
#: deployment never benefits from raising it, only from more containers.
DEFAULT_MAX_CONCURRENT_ENVS = 1

#: How long a detached session may idle before it is reclaimed. A rollout that
#: stops calling tools and never grades would otherwise hold its slot for the
#: life of the container.
DEFAULT_SESSION_TIMEOUT_SECONDS = 600.0


def build_app(
    *,
    max_concurrent_envs: int = DEFAULT_MAX_CONCURRENT_ENVS,
    session_timeout: float = DEFAULT_SESSION_TIMEOUT_SECONDS,
) -> FastAPI:
    if type(max_concurrent_envs) is not int or max_concurrent_envs < 1:
        raise ValueError("max_concurrent_envs must be a positive integer")
    if (
        isinstance(session_timeout, bool)
        or not math.isfinite(session_timeout)
        or session_timeout <= 0
    ):
        raise ValueError("session_timeout must be finite and positive")
    return create_app(
        DataCodeAgentRLEnvironment,
        GradeAction,
        TaskObservation,
        state_cls=TaskState,
        env_name="byoh_rle",
        concurrency_config=ConcurrencyConfig(
            max_concurrent_envs=max_concurrent_envs,
            session_timeout=session_timeout,
        ),
    )


def app() -> FastAPI:
    """Uvicorn factory. Defaults above apply unless OPENENV_MAX_CONCURRENT_ENVS or
    OPENENV_SESSION_TIMEOUT_SECONDS overrides them; both still validate as finite
    and positive.
    """
    try:
        capacity = int(
            os.environ.get("OPENENV_MAX_CONCURRENT_ENVS", DEFAULT_MAX_CONCURRENT_ENVS)
        )
        timeout = float(
            os.environ.get(
                "OPENENV_SESSION_TIMEOUT_SECONDS", DEFAULT_SESSION_TIMEOUT_SECONDS
            )
        )
    except ValueError:
        raise ValueError(
            "OPENENV_MAX_CONCURRENT_ENVS must be a positive integer and "
            "OPENENV_SESSION_TIMEOUT_SECONDS must be a finite positive number"
        ) from None
    return build_app(max_concurrent_envs=capacity, session_timeout=timeout)
