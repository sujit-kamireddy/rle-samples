"""RLE harness container for the FineEnvs data-agent tasks.

This environment owns no sandbox and mocks no tools: the sandbox, the agent
loop, and Harbor's own grader all run inside the separately deployed
``../harbor-server``, driven by ``../agent``. This environment exists only so
RLE has something to `/reset` and `/grade` -- the two calls the
`Harness`/`BYOH` subtype always makes regardless of what the harness itself
does.

`/reset` is close to a no-op: this sample's harness ignores whatever it
returns (`--agent-input` on the `azd ai rle rollout` command line already
carries the Harbor task selector -- `task_index`, `split`, `harness`,
`sandbox` -- so `/reset` has no per-task payload to hand over here).

`/grade` reads `reward` straight out of ``agent_response`` -- the JSON string
``../agent``'s `/invoke` returned as `output_text`, which RLE forwards to
`/grade` verbatim in the request body. That string is Harbor's own verifier
result (see ``../agent/app.py``'s `run_harbor_rollout`): `../agent` only
relays it, it never computes `reward` itself. Nothing here calls out to
`harbor-server` or anywhere else -- `/grade` is a pure parse of data RLE
already handed it.
"""

from __future__ import annotations

import json
from typing import Any, Optional
from uuid import uuid4

from fastapi import Body
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


class DataAgentAction(Action):
    """Unused: this sample's harness never calls `/step`."""

    message: str = Field(default="", description="Unused placeholder.")


class DataAgentObservation(Observation):
    """Rollout-visible messages returned by `/reset`."""

    messages: list[dict[str, Any]] = Field(default_factory=list)


class DataAgentEnvironment(Environment[DataAgentAction, DataAgentObservation, State]):
    """Holds no rollout state: `/grade` reads `agent_response`, not anything stored here."""

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


@app.post("/grade")
async def grade_rollout(
    rollout: dict[str, Any] = Body(default_factory=dict),
    agent_response: str = Body(default=""),
) -> dict[str, Any]:
    """Parses Harbor's verifier result out of `agent_response`.

    `agent_response` is `../agent`'s `/invoke` `output_text`, forwarded by RLE
    verbatim in this request's body -- see `run_harbor_rollout` in
    `../agent/app.py` for what it contains. `../agent` never computes
    `reward` itself; it only relays what harbor-server's verifier already
    produced, so parsing it here costs nothing a live fetch would have
    bought.
    """
    del rollout
    try:
        reported = json.loads(agent_response) if agent_response else None
    except ValueError:
        reported = None
    if not isinstance(reported, dict):
        # `../agent` was withdrawn or crashed before it ever posted an
        # outcome, or it returned something this shim did not produce. Score
        # zero rather than raise: a malformed report is not grounds to fail
        # the whole rollout.
        return {
            "reward": 0.0,
            "is_success": False,
            "info": {"reason": "agent_response was missing or not the expected JSON."},
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
async def env_grade_rollout(
    rollout: dict[str, Any] = Body(default_factory=dict),
    agent_response: str = Body(default=""),
) -> dict[str, Any]:
    return await grade_rollout(rollout, agent_response)
