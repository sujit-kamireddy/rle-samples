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

`/grade` computes `reward` itself, rather than trusting a number `../agent`
reports: `reward` alone is unforgeable-*looking* but cheap to fabricate,
since Harbor's grader (vendored below, byte-identical across every task in
this dataset) is a pure, public function of `(gold, candidate)` -- a
misbehaving harness could compute the winning `candidate` string and just
claim the score it produces. What `/grade` trusts instead is the *raw
answer text* the sandbox produced (`answer_text`, harvested by
`../harbor-server` from `/workdir/answer.txt` via Harbor's own `artifacts`
mechanism -- see `../harbor-server/server/app.py`), which it grades here
with its own copy of the same deterministic function Harbor itself runs.
This still trusts that `answer_text` is what the sandbox actually wrote
(nothing outside the sandbox can prove that independently), but it removes
the ability to claim an arbitrary `reward` for an arbitrary answer.
"""

from __future__ import annotations

import gzip
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from fastapi import Body
from pydantic import Field

from .vendor.grader import grade as _grade

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

_VENDOR_DIR = Path(__file__).parent / "vendor"
_TASK_META_DIR = _VENDOR_DIR / "task-meta"


def _task_meta_path(split: str) -> Path:
    # Matches the on-disk convention the vendored dataset tarball itself
    # uses for HF repo ids (`FineEnvs/data-agent-harbor-train` ->
    # `FineEnvs__data-agent-harbor-train`) -- see `../harbor-server`'s
    # dataset vendoring script for the same substitution.
    return _TASK_META_DIR / f"{split.replace('/', '__')}.json.gz"


@lru_cache(maxsize=8)
def _load_task_meta(split: str) -> list[dict[str, Any]]:
    path = _task_meta_path(split)
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


class DataAgentAction(Action):
    """Unused: this sample's harness never calls `/step`."""

    message: str = Field(default="", description="Unused placeholder.")


class DataAgentObservation(Observation):
    """Rollout-visible messages returned by `/reset`."""

    messages: list[dict[str, Any]] = Field(default_factory=list)


class DataAgentEnvironment(Environment[DataAgentAction, DataAgentObservation, State]):
    """Holds no rollout state: `/grade` grades `agent_response`, not anything stored here."""

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
    """Grades `answer_text` out of `agent_response` with Harbor's own grader.

    `agent_response` is `../agent`'s `/invoke` `output_text`, forwarded by RLE
    verbatim in this request's body -- see `run_harbor_rollout` in
    `../agent/app.py` for what it contains: the task selector (`split`,
    `task_index`) and the sandbox's raw `answer_text`, nothing more. `reward`
    is computed here, not read out of anything `../agent` sent -- see this
    module's docstring for why.
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

    ok = bool(reported.get("ok", True))
    answer_text = reported.get("answer_text")
    split = reported.get("split")
    task_index = reported.get("task_index")

    if not ok or answer_text is None or split is None or task_index is None:
        return {
            "reward": 0.0,
            "is_success": False,
            "info": {
                "reason": "Harbor rollout did not complete or did not report an answer.",
                "ok": ok,
                "error": reported.get("error"),
            },
        }

    try:
        task_meta_list = _load_task_meta(split)
        task_meta = task_meta_list[task_index]
    except (FileNotFoundError, IndexError, TypeError) as exc:
        return {
            "reward": 0.0,
            "is_success": False,
            "info": {"reason": f"No vendored task metadata for split={split!r} task_index={task_index!r}: {exc}"},
        }

    result = _grade(
        task_meta["expected_answer"],
        answer_text,
        reward_mode=task_meta.get("reward_mode") or "",
        abs_tol=float(task_meta.get("atol") or 1e-3),
        rel_tol=float(task_meta.get("rtol") or 1e-3),
    )
    return {
        "reward": result.reward,
        "is_success": result.reward > 0,
        "info": {
            "task_name": task_meta.get("task_name"),
            "method": result.method,
        },
    }


@app.post("/_env/grade")
async def env_grade_rollout(
    rollout: dict[str, Any] = Body(default_factory=dict),
    agent_response: str = Body(default=""),
) -> dict[str, Any]:
    return await grade_rollout(rollout, agent_response)
