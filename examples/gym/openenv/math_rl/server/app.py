"""ASGI app exposing the SDK ``MathRLEnvironment`` MCP contract.

Usage::

    uvicorn examples.gym.openenv.math_rl.server.app:app --host 0.0.0.0 --port 8000

or, standalone inside the built container::

    python -m examples.gym.openenv.math_rl.server.app

Serve it with one worker so MCP calls and final grading use the same instance.
"""

from __future__ import annotations

import os

from fastapi import FastAPI
from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.types import ConcurrencyConfig

from azure.ai.projects.rle.environments import GradeAction

from .math_rl_environment import MathRLEnvironment
from .models import MathObservation, MathState


def build_app() -> FastAPI:
    return create_app(
        MathRLEnvironment,
        GradeAction,
        MathObservation,
        state_cls=MathState,
        env_name="math_rl",
        concurrency_config=ConcurrencyConfig(
            max_concurrent_envs=1,
            session_timeout=300.0,
        ),
    )


app = build_app()


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))


if __name__ == "__main__":
    main()
