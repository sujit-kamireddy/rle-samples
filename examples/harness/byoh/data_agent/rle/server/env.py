"""RLE rollout container for the FineEnvs data-agent tasks.

A plain FastAPI app speaking exactly the four things RLE's `Harness`/`BYOH`
subtype asks of a rollout container -- `/health`, `/reset`, `/tools/<name>`
and `/grade` -- and nothing else. Deliberately *not* an OpenEnv
environment: the agent never acts through this container, so there is no
`step` to implement, no episode state to hold and no observation anyone
reads. Modelling it as a stepped MDP meant carrying an `Action`, an
`Observation` and a `step()` that every real invocation ignored, plus ten
catalogue and schema routes RLE never calls.

The agent loop runs inside the separately deployed ``../harbor-server``,
driven by ``../agent``. This container holds the answer key, serves one
mocked compliance tool, and scores the result.

`/health` is the readiness probe RLE polls before it will attempt `/reset`:
a sandbox reports Running once the container is scheduled, which is well
before uvicorn has bound its port.

`/reset` is a liveness formality on the BYOH path, not a data channel: RLE's
`ResetAsync` returns a bare `Task`, so the response body -- observation,
messages, reward -- is never deserialized, and only the status code decides
the rollout's fate (any non-2xx raises `RolloutDependencyException` and
discards it). RLE POSTs the caller's `--task` JSON *verbatim* rather than an
OpenEnv `ResetRequest` envelope, so this handler must keep tolerating an
arbitrary object; the selector the harness actually runs on arrives
separately as `--agent-input` (`task_index`, `split`). Only a Gym/OpenEnv
target reads an observation back, and that target skips this call entirely.
See `../rle/tests/test_reset_contract.py`.

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

from fastapi import Body, FastAPI, Header, HTTPException

from . import compliance
from .vendor.grader import grade as _grade

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


app = FastAPI(title="data-agent-rle")


@app.get("/health")
async def health() -> dict[str, str]:
    """Readiness probe RLE polls before it will attempt `/reset`."""
    return {"status": "healthy"}


@app.post("/reset")
async def reset(
    task: dict[str, Any] = Body(default_factory=dict),
    rollout_id: Optional[str] = Header(default=None, alias=compliance.ROLLOUT_ID_HEADER),
) -> dict[str, Any]:
    """Opens a rollout. No per-task setup happens here.

    The body is the caller's task JSON forwarded verbatim, and RLE reads
    nothing of this response but its status code -- see this module's
    docstring. What the call is good for is isolation: one container serves
    several rollouts across successive `/reset` calls, so this drops any
    disclosure still sitting under this rollout id from an attempt that was
    withdrawn or died before `/grade` could consume it. Left in place, that
    stale record would be credited to whoever graded next under the same id.
    """
    del task
    if rollout_id:
        # Never the no-argument form: that clears every rollout's disclosure,
        # including ones still in flight in this same container.
        compliance.clear(rollout_id)
    return {"status": "ready", "rollout_id": rollout_id}


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

    if not rollout_id:
        # RLE resolves this header from its own rollout mapping and strips any
        # caller-supplied copy, so a missing one is broken plumbing rather than
        # anything the agent did. Refusing beats recording under a shared
        # fallback key: that key is unattributable, and two concurrent rollouts
        # filing under it would silently overwrite each other. The refusal says
        # nothing about the data, so it leaks no grading signal either.
        raise HTTPException(
            status_code=400,
            detail=(
                f"Missing {compliance.ROLLOUT_ID_HEADER}; "
                "this disclosure cannot be attributed to a rollout."
            ),
        )

    record = compliance.record_disclosure(
        rollout_id,
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
    if rollout_id:
        verdict, multiplier = compliance.evaluate(has_pii, disclosure is not None)
    else:
        # RLE always injects the rollout id, so its absence means the
        # disclosure channel is mis-plumbed rather than that the agent chose
        # to stay silent. Scoring a missed disclosure here would invent a
        # behavioural failure out of a deployment fault, so the decision is
        # left unscored and the reason is recorded instead.
        verdict, multiplier = "unmeasured_no_rollout_id", 1.0
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

