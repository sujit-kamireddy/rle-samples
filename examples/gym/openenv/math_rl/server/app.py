"""FastAPI app exposing ``MathRLEnvironment`` over the OpenEnv HTTP/WS contract.

Built with ``openenv``'s own ``create_fastapi_app`` -- see
``examples/gym/openenv/code_rl/server/app.py`` for why this no longer needs a hand-rolled
runtime.

Usage::

    uvicorn examples.gym.openenv.math_rl.server.app:app --host 0.0.0.0 --port 8000

or, standalone inside the built container::

    python -m examples.gym.openenv.math_rl.server.app

Serve it with a single worker. ``max_concurrent_envs=1`` below reflects
that each container is leased to one Foundry RLE instance/episode series
at a time; a second worker would let two connections each get "the"
container's own session, breaking that assumption.
"""

from __future__ import annotations

import os

from openenv.core.env_server.http_server import create_fastapi_app

try:
    from .math_rl_environment import MathRLEnvironment
except ImportError:  # pragma: no cover - standalone container import path
    from math_rl_environment import MathRLEnvironment
from .schema import MathAction, MathObservation

app = create_fastapi_app(
    MathRLEnvironment,
    MathAction,
    MathObservation,
    max_concurrent_envs=1,
    env_name="math_rl",
)


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))


if __name__ == "__main__":
    main()
