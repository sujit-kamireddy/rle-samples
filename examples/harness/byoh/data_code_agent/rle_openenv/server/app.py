"""Opt-in standalone app. Serve one worker; each session owns an environment."""

from __future__ import annotations

import math
import os

from fastapi import FastAPI
from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.types import ConcurrencyConfig

from .environment import (
    GradeAction,
    ByohRLEEnvironment,
    TaskObservation,
    TaskState,
)


def build_app(*, max_concurrent_envs: int, session_timeout: float) -> FastAPI:
    if type(max_concurrent_envs) is not int or max_concurrent_envs < 1:
        raise ValueError("max_concurrent_envs must be a positive integer")
    if (
        isinstance(session_timeout, bool)
        or not math.isfinite(session_timeout)
        or session_timeout <= 0
    ):
        raise ValueError("session_timeout must be finite and positive")
    return create_app(
        ByohRLEEnvironment,
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
    """Uvicorn factory requiring explicit capacity and detached-idle timeout."""
    try:
        capacity = int(os.environ["OPENENV_MAX_CONCURRENT_ENVS"])
        timeout = float(os.environ["OPENENV_SESSION_TIMEOUT_SECONDS"])
    except (KeyError, ValueError):
        raise ValueError(
            "Set OPENENV_MAX_CONCURRENT_ENVS to a positive integer and "
            "OPENENV_SESSION_TIMEOUT_SECONDS to a finite positive number"
        ) from None
    return build_app(max_concurrent_envs=capacity, session_timeout=timeout)
