"""RLE harness container for the FineEnvs data-agent tasks.

Unlike ``../../code_repair``'s `rle/`, this environment owns no sandbox and
mocks no tools: the sandbox, the agent loop, and Harbor's own grader all run
inside the separately deployed ``../harbor-server``, driven by ``../agent``.
This environment exists only so RLE has something to `/reset` and `/grade` --
the two calls the `Harness`/`BYOH` subtype always makes regardless of what the
harness itself does.

`/reset` is close to a no-op: this sample's harness ignores whatever it
returns (`--agent-input` on the `azd ai rle rollout` command line already
carries the Harbor task selector -- `task_index`, `split`, `harness`,
`sandbox` -- so the issue text `code_repair`'s `/reset` hands over has no
equivalent here).

`/grade` cannot re-run Harbor's own verifier -- it never touches the sandbox
that ran it, and by the time `/grade` is called that sandbox may already be
gone. But it does not have to trust ``../agent`` either. RLE attaches the
same rollout id to every call it makes for one rollout, as the
`x-rle-rollout-id` header -- on this container's `/reset` and `/grade` calls,
and as the `rollout_id` field in ``../agent``'s `/invoke` body. ``../agent``
uses that id to trigger the run on harbor-server
(`POST /correlated-rollouts/{rollout_id}`); `/grade` here uses the *same* id
to fetch harbor-server's own record of what happened
(`GET /correlated-rollouts/{rollout_id}`) directly, over the network, without
going through ``../agent`` at all. So a harness that lies about the reward
gains nothing: nothing downstream of Harbor's own verifier ever reads what
``../agent`` says.
"""

from __future__ import annotations

import os
from typing import Any, Optional
from uuid import uuid4

import httpx
from fastapi import Body, Request
from pydantic import Field

from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import (
    Action,
    EnvironmentMetadata,
    Observation,
    ResetRequest,
    ResetResponse,
    State,
)

# The same harbor-server ../agent drives. `/grade` reaches it directly,
# independent of ../agent, to fetch the rollout id it is grading.
HARBOR_SERVER_URL = os.environ["HARBOR_SERVER_URL"]
# RLE's own header name for the rollout id it correlates every call by
# (`RleSandboxRequestHeaders.RolloutId` in RLE's sandbox client) -- server-set,
# never caller-supplied, so a harness cannot forge it.
ROLLOUT_ID_HEADER = "x-rle-rollout-id"
# harbor-server's `run_rollout` can run for minutes; a grade call has to wait
# at least that long for a legitimately still-running rollout to land.
GRADE_FETCH_TIMEOUT_S = float(os.environ.get("DATA_AGENT_GRADE_TIMEOUT_S", "1800"))


class DataAgentAction(Action):
    """Unused: this sample's harness never calls `/step`."""

    message: str = Field(default="", description="Unused placeholder.")


class DataAgentObservation(Observation):
    """Rollout-visible messages returned by `/reset`."""

    messages: list[dict[str, Any]] = Field(default_factory=list)


class DataAgentEnvironment(Environment[DataAgentAction, DataAgentObservation, State]):
    """Holds no rollout state: `/grade` fetches everything it needs from harbor-server."""

    def __init__(self) -> None:
        super().__init__()
        self._state = State(episode_id=None, step_count=0)

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **task_data: Any,
    ) -> DataAgentObservation:
        del seed, task_data
        episode_id = episode_id or str(uuid4())
        self._state = State(episode_id=episode_id, step_count=0)
        return DataAgentObservation(done=False, reward=None, messages=[])

    def step(
        self,
        action: DataAgentAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> DataAgentObservation:
        """Unused by real invocations: RLE calls `/grade` directly. Kept only
        so OpenEnv's schema can still exercise this environment."""
        del action, timeout_s, kwargs
        self._state.step_count += 1
        return DataAgentObservation(done=True, reward=None, messages=[])

    @property
    def state(self) -> State:
        return self._state

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="data_agent",
            description="RLE harness fronting a FineEnvs Harbor data-agent task.",
            version="0.1.0",
        )


app = create_app(
    DataAgentEnvironment,
    DataAgentAction,
    DataAgentObservation,
    env_name="data_agent",
    max_concurrent_envs=1,
)


# RLE's Harness rollout proxy is being migrated off the `_env/`-prefixed
# paths onto these same plain ones (`/health`, `/reset`, `/grade`). Until
# every deployed RLE region has that change, keep both: the aliases below
# forward to the exact same handlers so neither generation of RLE breaks.
# Same rationale as `code_repair/rle/server/env.py`'s identical aliases.
@app.get("/_env/health")
async def env_health() -> dict[str, str]:
    return {"status": "healthy"}


def _registered_endpoint(path: str, method: str):
    method = method.upper()
    for route in app.routes:
        if getattr(route, "path", None) == path and method in getattr(route, "methods", set()):
            return route.endpoint
    raise RuntimeError(f"No {method} route registered for {path!r}")


_reset_endpoint = _registered_endpoint("/reset", "POST")


@app.post("/_env/reset")
async def env_reset(request: ResetRequest = Body(default_factory=ResetRequest)) -> ResetResponse:
    return await _reset_endpoint(request)


async def _fetch_harbor_result(rollout_id: str) -> Optional[dict[str, Any]]:
    async with httpx.AsyncClient(timeout=GRADE_FETCH_TIMEOUT_S) as client:
        response = await client.get(f"{HARBOR_SERVER_URL}/correlated-rollouts/{rollout_id}")
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()


@app.post("/grade")
async def grade_rollout(request: Request, rollout: dict[str, Any] = Body(default_factory=dict)) -> dict[str, Any]:
    """Fetches Harbor's own reward for this rollout id, straight from harbor-server."""
    del rollout
    rollout_id = request.headers.get(ROLLOUT_ID_HEADER)
    reported = await _fetch_harbor_result(rollout_id) if rollout_id else None
    if reported is None:
        # Either RLE sent no rollout id (older region -- see the `_env/`
        # aliases' rationale above), or harbor-server never recorded a
        # result for it: `../agent` was withdrawn, crashed, or never
        # started before grading. Score zero rather than hang -- nothing
        # else in this container bounds how long `/grade` waits.
        return {
            "reward": 0.0,
            "is_success": False,
            "info": {"reason": "No result recorded on harbor-server for this rollout id."},
        }
    reward = reported.get("reward")
    ok = bool(reported.get("ok", True))
    return {
        "reward": float(reward) if reward is not None else 0.0,
        # `ok` reports whether Harbor's own agent loop and sandbox completed
        # without error; `reward` can still be a real (possibly zero) score
        # even when `ok` is false, so surface both rather than collapsing
        # them into one flag.
        "is_success": ok and reward is not None and reward > 0,
        "info": {
            "task_name": reported.get("task_name"),
            "n_turns": reported.get("n_turns"),
            "ok": ok,
            "error": reported.get("error"),
        },
    }


@app.post("/_env/grade")
async def env_grade_rollout(request: Request, rollout: dict[str, Any] = Body(default_factory=dict)) -> dict[str, Any]:
    return await grade_rollout(request, rollout)
