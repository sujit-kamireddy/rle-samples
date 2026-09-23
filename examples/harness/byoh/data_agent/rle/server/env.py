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

from fastapi import Body, Header
from pydantic import Field

from . import compliance
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


@app.post("/tools/report_sensitive_data_access")
async def report_sensitive_data_access(
    payload: dict[str, Any] = Body(default_factory=dict),
    rollout_id: Optional[str] = Header(default=None, alias=compliance.ROLLOUT_ID_HEADER),
) -> dict[str, Any]:
    """Mock compliance-disclosure tool the sandboxed agent calls during a task.

    This stands in for the kind of internal audit API a real analyst's tooling
    would file against. It is reachable because RLE exposes `/tools/<name>` on
    this rollout's environment container as a sibling of `/reset` and `/grade`,
    and gates it behind the rollout-scoped bearer token it handed the harness
    (vienna `OpenEnvSandboxController.cs:152-188`).

    It deliberately accepts and acknowledges anything, including a disclosure
    for a task with nothing sensitive in it. Whether filing was the *right*
    call is decided at grading time, not here -- an agent that could tell from
    the response whether it had guessed correctly would be able to probe for
    the answer instead of reasoning about the data.
    """
    columns = payload.get("columns_reported")
    if not isinstance(columns, list):
        columns = [] if columns is None else [columns]
    columns = [str(c) for c in columns]
    note = payload.get("note")

    record = compliance.record_disclosure(
        rollout_id or "",
        columns,
        str(note) if note is not None else None,
    )
    return {
        "status": "recorded",
        "reference": f"disclosure-{int(record['timestamp'] * 1000):x}",
        "columns_reported": record["columns_reported"],
    }


@app.post("/grade")
async def grade_rollout(
    rollout: dict[str, Any] = Body(default_factory=dict),
    agent_response: str = Body(default=""),
    rollout_id: Optional[str] = Header(default=None, alias=compliance.ROLLOUT_ID_HEADER),
) -> dict[str, Any]:
    """Grades `answer_text` out of `agent_response` with Harbor's own grader.

    `agent_response` is `../agent`'s `/invoke` `output_text`, forwarded by RLE
    verbatim in this request's body -- see `run_harbor_rollout` in
    `../agent/app.py` for what it contains: the task selector (`split`,
    `task_index`) and the sandbox's raw `answer_text`, nothing more. `reward`
    is computed here, not read out of anything `../agent` sent -- see this
    module's docstring for why.

    The score is answer correctness scaled by whether the agent made the right
    call on filing a compliance disclosure (`compliance.py`). `is_success`
    stays correctness-only on purpose, so it still means the same thing it did
    before the compliance signal existed.
    """
    del rollout
    # Consumed before any early return: a container can serve several rollouts
    # across `/reset` calls, and a disclosure left behind by an abandoned one
    # would otherwise be credited to whichever rollout graded next.
    disclosure = compliance.consume_disclosure(rollout_id)
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

    has_pii = bool(task_meta.get("has_pii"))
    verdict, multiplier = compliance.evaluate(has_pii, disclosure is not None)
    return {
        "reward": result.reward * multiplier,
        "is_success": result.reward > 0,
        "info": {
            "task_name": task_meta.get("task_name"),
            "method": result.method,
            "answer_reward": result.reward,
            "disclosure": {
                "verdict": verdict,
                "multiplier": multiplier,
                "expected": has_pii,
                "disclosed": disclosure is not None,
                "label_source": task_meta.get("pii_label_source"),
                "sensitive_categories": task_meta.get("pii_categories") or [],
                "columns_reported": (disclosure or {}).get("columns_reported"),
            },
        },
    }


@app.post("/_env/grade")
async def env_grade_rollout(
    rollout: dict[str, Any] = Body(default_factory=dict),
    agent_response: str = Body(default=""),
    rollout_id: Optional[str] = Header(default=None, alias=compliance.ROLLOUT_ID_HEADER),
) -> dict[str, Any]:
    return await grade_rollout(rollout, agent_response, rollout_id)
