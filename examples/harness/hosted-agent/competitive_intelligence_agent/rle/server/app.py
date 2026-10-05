"""The ASGI app. ``create_app`` supplies the protocol; this file supplies capacity.

``create_app`` builds one environment per OpenEnv session, so capacity here is
how many rollouts a single container may host at once. Run **one** uvicorn
worker: session ids belong to the process that minted them.

``GradeAction`` is passed as the action class so a step payload is deserialised
into it before ``grade`` is called. Without it the environment would be handed a
bare ``Action`` and ``RLEnvironment._step_impl`` would reject it.
"""

from __future__ import annotations

import os

from fastapi import FastAPI
from azure.ai.projects.rle.environments import create_app
from openenv.core.env_server.types import ConcurrencyConfig

from .environment import (
    CompetitiveIntelEnvironment,
    CompetitiveIntelObservation,
    CompetitiveIntelState,
    GradeAction,
)

#: One rollout at a time, matching what a Harness sandbox actually schedules.
#: This is also the only value OpenEnv will accept here: it refuses
#: ``max_concurrent_envs > 1`` unless the environment class sets
#: ``SUPPORTS_CONCURRENT_SESSIONS = True``, and this one deliberately does not.
#: Per-episode state is isolated on the ``ToolSession``, but the simulated
#: ``WORLD`` is built once per process and shared, so concurrent sessions are
#: not proven safe. Raising this raises ``ConcurrencyConfigurationError`` at
#: startup rather than failing quietly mid-rollout.
DEFAULT_MAX_CONCURRENT_ENVS = 1

#: How long a detached session may idle before it is reclaimed. A rollout that
#: has stopped calling tools and will never call ``grade`` would otherwise hold
#: its slot for the life of the container.
DEFAULT_SESSION_TIMEOUT_SECONDS = 600.0


def build_app(
    *,
    max_concurrent_envs: int = DEFAULT_MAX_CONCURRENT_ENVS,
    session_timeout: float = DEFAULT_SESSION_TIMEOUT_SECONDS,
) -> FastAPI:
    """The environment's ASGI app, with explicit capacity."""
    if max_concurrent_envs < 1:
        raise ValueError("max_concurrent_envs must be a positive integer")
    if session_timeout <= 0:
        raise ValueError("session_timeout must be positive")
    return create_app(
        CompetitiveIntelEnvironment,
        GradeAction,
        CompetitiveIntelObservation,
        state_cls=CompetitiveIntelState,
        env_name="competitive_intelligence_agent",
        concurrency_config=ConcurrencyConfig(
            max_concurrent_envs=max_concurrent_envs,
            session_timeout=session_timeout,
        ),
    )


app = build_app(
    max_concurrent_envs=int(
        os.environ.get("OPENENV_MAX_CONCURRENT_ENVS", DEFAULT_MAX_CONCURRENT_ENVS)
    ),
    session_timeout=float(
        os.environ.get("OPENENV_SESSION_TIMEOUT_SECONDS", DEFAULT_SESSION_TIMEOUT_SECONDS)
    ),
)
