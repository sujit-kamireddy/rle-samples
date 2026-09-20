"""FastAPI app exposing ``CodeRLEnvironment`` over the OpenEnv HTTP/WS contract.

Built with ``openenv``'s own ``create_fastapi_app`` -- unlike the earlier,
hand-rolled server this replaced, there is no reason not to use it: the
statefulness this environment needs (``step()`` seeing what ``reset()``
stored) now lives on the ``/ws`` route's per-connection session, which
``openenv``'s own server already provides one instance per connection for.
See ``examples/gym/openenv/README.md`` for the tradeoff (this pulls in ``openenv``'s
transitive dependency stack, unlike the runtime it replaced).

Usage::

    uvicorn examples.gym.openenv.code_rl.server.app:app --host 0.0.0.0 --port 8000

or, standalone inside the built container::

    python -m examples.gym.openenv.code_rl.server.app

Serve it with a single worker. ``max_concurrent_envs=1`` below reflects
that each container is leased to one Foundry RLE instance/episode series
at a time; a second worker would let two connections each get "the"
container's own session, breaking that assumption.
"""

from __future__ import annotations

import os

from openenv.core.env_server.http_server import create_fastapi_app

try:
    from .code_rl_environment import CodeRLEnvironment
except ImportError:  # pragma: no cover - standalone container import path
    from code_rl_environment import CodeRLEnvironment
from .schema import CodeAction, CodeObservation

app = create_fastapi_app(
    CodeRLEnvironment,
    CodeAction,
    CodeObservation,
    max_concurrent_envs=1,
    env_name="code_rl",
)


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))


if __name__ == "__main__":
    main()
