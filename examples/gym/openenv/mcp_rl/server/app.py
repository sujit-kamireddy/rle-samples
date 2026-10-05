"""ASGI entry point for the SDK RLEnvironment sample."""

from __future__ import annotations

from fastapi import FastAPI
from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.types import ConcurrencyConfig

from azure.ai.projects.rle.environments import GradeAction

from .environment import ArithmeticRLEnvironment
from .models import ArithmeticObservation, ArithmeticState


def build_app() -> FastAPI:
    """Build a single-session app; managed RLE scales with containers."""
    return create_app(
        ArithmeticRLEnvironment,
        GradeAction,
        ArithmeticObservation,
        state_cls=ArithmeticState,
        env_name="mcp_rl",
        concurrency_config=ConcurrencyConfig(
            max_concurrent_envs=1,
            session_timeout=300.0,
        ),
    )


app = build_app()
