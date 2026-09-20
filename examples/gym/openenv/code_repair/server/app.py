"""FastAPI app exposing ``CodeRepairEnvironment`` over the OpenEnv HTTP/WS
contract. Built with ``openenv``'s own ``create_fastapi_app`` -- see
``examples/gym/openenv/code_rl/server/app.py`` for why (statefulness lives
on the ``/ws`` route's per-connection session).

Usage::

    uvicorn examples.gym.openenv.code_repair.server.app:app --host 0.0.0.0 --port 8000

or, standalone inside the built container::

    python -m examples.gym.openenv.code_repair.server.app

Serve it with a single worker: ``max_concurrent_envs=1`` reflects that each
container is leased to one Foundry RLE instance/episode series at a time.
"""

from __future__ import annotations

import os

from openenv.core.env_server.http_server import create_fastapi_app

try:
    from .code_repair_environment import CodeRepairEnvironment
except ImportError:  # pragma: no cover - standalone container import path
    from code_repair_environment import CodeRepairEnvironment
from .schema import CodeRepairAction, CodeRepairObservation

app = create_fastapi_app(
    CodeRepairEnvironment,
    CodeRepairAction,
    CodeRepairObservation,
    max_concurrent_envs=1,
    env_name="code_repair",
)


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))


if __name__ == "__main__":
    main()
