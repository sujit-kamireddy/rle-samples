"""ASGI entry point for the SDK ``code_rl`` MCP environment."""

from __future__ import annotations

import os

from fastapi import FastAPI
from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.types import ConcurrencyConfig
from azure.ai.projects.rle.environments import GradeAction

from .code_rl_environment import CodeRLEnvironment
from .models import CodeObservation, CodeState


def build_app() -> FastAPI:
    return create_app(
        CodeRLEnvironment,
        GradeAction,
        CodeObservation,
        state_cls=CodeState,
        env_name="code_rl",
        concurrency_config=ConcurrencyConfig(max_concurrent_envs=1, session_timeout=300.0),
    )


app = build_app()


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))


if __name__ == "__main__":
    main()
