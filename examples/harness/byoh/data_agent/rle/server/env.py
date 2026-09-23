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
equivalent here). `/grade` cannot run Harbor's own verifier either, because it
never touches the sandbox that verifier ran in -- Harbor already ran it and
computed a reward before this container ever hears about the rollout. So
``../agent`` posts that reward to `/tools/harbor.report_result` as its last
step (the same channel a harness normally uses for side-effecting tool
calls), and `/grade` reads back whatever was posted for this episode.
"""

from __future__ import annotations

import os
from typing import Any, Optional
from uuid import uuid4

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
from fastapi import Body

# `/grade` runs after `agent/app.py`'s Harbor call returns, and that call can
# run for minutes. Without a bound, a harness that crashes before reporting a
# result leaves `/grade` waiting forever instead of scoring the rollout zero.
REPORT_TIMEOUT_S = float(os.environ.get("DATA_AGENT_REPORT_TIMEOUT_S", "1800"))

_current_environment: Optional["DataAgentEnvironment"] = None


class DataAgentAction(Action):
    """Unused: this sample's harness never calls `/step`, only `/tools/*`."""

    message: str = Field(default="", description="Unused placeholder.")


class DataAgentObservation(Observation):
    """Rollout-visible messages returned by `/reset`."""

    messages: list[dict[str, Any]] = Field(default_factory=list)


class DataAgentEnvironment(Environment[DataAgentAction, DataAgentObservation, State]):
    """One rollout's worth of state: nothing but the reward Harbor reported."""

    def __init__(self) -> None:
        super().__init__()
        self._state = State(episode_id=None, step_count=0)
        self._reported_result: Optional[dict[str, Any]] = None
        # max_concurrent_envs=1 below means this container ever hosts one
        # live instance at a time, so the module-level routes below can
        # reach this rollout's state through this singleton reference --
        # the same pattern `code_repair/rle/server/env.py` uses.
        global _current_environment
        _current_environment = self

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **task_data: Any,
    ) -> DataAgentObservation:
        del seed, task_data
        episode_id = episode_id or str(uuid4())
        self._state = State(episode_id=episode_id, step_count=0)
        self._reported_result = None
        return DataAgentObservation(done=False, reward=None, messages=[])

    def step(
        self,
        action: DataAgentAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> DataAgentObservation:
        """Unused by real invocations: the harness calls
        `/tools/harbor.report_result` and RLE calls `/grade` directly. Kept
        only so OpenEnv's schema can still exercise this environment."""
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


@app.post("/tools/harbor.report_result")
async def harbor_report_result(arguments: dict[str, Any]) -> dict[str, Any]:
    """Records the reward `agent/app.py` got back from Harbor's `run_rollout`.

    This is the only "tool" this sample's harness calls -- there is no
    workspace or sandbox to mock, because Harbor already ran the agent loop
    and its own verifier before this call arrives. `/grade` below just reads
    back what was recorded here.
    """
    if _current_environment is None:
        raise RuntimeError("No active rollout; call /reset first.")
    _current_environment._reported_result = arguments
    return {"recorded": True}


@app.post("/grade")
async def grade_rollout(rollout: dict[str, Any]) -> dict[str, Any]:
    """Reports the reward Harbor computed for this rollout's task."""
    del rollout
    if _current_environment is None:
        raise RuntimeError("No active rollout; call /reset first.")
    reported = _current_environment._reported_result
    if reported is None:
        # The harness never reported a result -- withdrawn, crashed, or timed
        # out before its Harbor call returned. Score zero rather than hang:
        # nothing else in this container bounds how long `/grade` waits.
        return {
            "reward": 0.0,
            "is_success": False,
            "info": {"reason": "No result reported by the harness before grading."},
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
async def env_grade_rollout(rollout: dict[str, Any]) -> dict[str, Any]:
    return await grade_rollout(rollout)
