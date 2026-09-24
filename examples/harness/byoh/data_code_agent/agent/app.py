"""Example BYOH agent harness for the data-code agent RLE.

Deploy this anywhere reachable over HTTPS, then point ``rle/rle.toml``'s
``baseUrl`` at ``<this-url>/invoke`` and run ``azd ai rle publish``.

This is the whole harness: it speaks RLE's ``Harness``/``BYOH`` invocation
contract *and* runs the agent. A rollout fetches its task's input files and
drives `opencode` in this container against the task instruction, pointed at the
model endpoint RLE supplies -- which is RLE's capture proxy, so the rollout graph
sees every call the agent makes. See ``opencode_direct.py`` for the rollout
itself, and ``../README.md`` for how this and ``../rle`` deploy together.

RLE invokes a harness asynchronously. The start request only starts work; the
answer is collected from a per-invocation resource RLE derives from the
registered URL by appending ``/rollouts/{operation_id}``. A harness never
supplies that address, so RLE only ever calls back under the path you
registered.

Two identifiers arrive and they answer different questions. ``rollout_id`` is
the correlation handle both sides log. ``operation_id`` is what RLE polls and
cancels: RLE mints it per invocation, and because the poll and cancel legs carry
no credential, it is also what authorizes those calls. Key your own state on
``operation_id``; ``rollout_id`` is unique per project, not globally, so two
projects can legitimately send the same one to a shared harness.

1. Start. RLE POSTs the rollout to the registered URL and the harness
   acknowledges with ``202`` as soon as it has taken ownership -- before the
   rollout has run. ``retry_after_ms`` is advisory; RLE clamps it.

       POST <base-url>
       {
         "rollout_id": "...",
         "operation_id": "...",
         "agent_input": {
           "task_index": 0,
           "split": "FineEnvs/data-agent-harbor-train"
         },
         "rollout_context": {
           "model_endpoint": "https://.../rle/v1.0/capture-proxy/v1",
           "model_api_key": "...",
           "sandbox_tools_endpoint": "https://.../rollouts/<rollout-id>/tools",
           "sandbox_tools_token": "..."
         }
       }

       202 Accepted
       {"retry_after_ms": 500}

2. Poll. RLE GETs the rollout resource until it reports an outcome. A rollout is
   one blocking call that fetches the task's input files and runs the agent to
   completion, so this can poll for minutes before an outcome appears.

       GET <base-url>/rollouts/{operation_id}

       200 {"status": "running"}
       200 {"status": "succeeded", "output_text": "{\"answer_text\": \"...\", \"split\": \"...\", ...}"}
       200 {"status": "failed", "error": {"message": "..."}}

   ``output_text`` is a JSON string, not free text: the agent's own raw
   answer (`answer_text`) plus the task selector, produced by
   `run_harness_rollout` below. RLE forwards this same string to `../rle`'s
   `/grade` verbatim, as `agent_response` -- that is how the answer gets
   there. This shim never computes or even sees a `reward`; `/grade` grades
   `answer_text` itself, against its own vendored copy of the grader.

3. Withdraw. If RLE stops waiting -- the caller disconnected, or the rollout
   deadline passed -- it DELETEs the rollout resource so the harness can stop
   spending tokens. RLE checks only the status, and may not send this at all, so
   treat it as advisory and keep your own timeout -- but do answer 2xx, because
   RLE records anything else as a cleanup failure.

       DELETE <base-url>/rollouts/{operation_id}

RLE never attaches caller, workspace, or identity headers to these requests, so
protect the endpoint yourself (for example, require a shared secret header and
validate it before starting a rollout).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

import httpx
from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

import opencode_direct
import telemetry

telemetry.configure()

app = FastAPI(title="byoh-data-code-agent")

logger = logging.getLogger("byoh-data-code-agent")

RETRY_AFTER_MS = 500

DEFAULT_SPLIT = os.environ.get("HARNESS_SPLIT", "FineEnvs/data-agent-harbor-train")

# The default model endpoint, and the only reason this service needs one at all.
# Under RLE every rollout carries its own `model_endpoint` -- RLE's capture proxy,
# not a real endpoint -- so these matter for a local run and for resolving which
# model id to hand `opencode` when it is not named explicitly.
_LLM_URL = os.environ.get("MODEL_URL", "")
_MODEL = os.environ.get("MODEL_ID", "")
_API_KEY = os.environ.get("MODEL_API_KEY", "") or None
_AUTH_HEADER = os.environ.get("MODEL_AUTH_HEADER", "") or "Authorization"


def _served_models(llm_url: str) -> list[str]:
    """Lists the model ids an OpenAI-spec endpoint serves."""
    if _API_KEY:
        value = f"Bearer {_API_KEY}" if _AUTH_HEADER.lower() == "authorization" else _API_KEY
        headers = {_AUTH_HEADER: value}
    else:
        headers = {}
    response = httpx.get(f"{llm_url.rstrip('/')}/models", headers=headers, timeout=30.0)
    response.raise_for_status()
    return [m["id"] for m in response.json().get("data", []) if m.get("id")]


# Resolved at import so a misconfigured deployment is visible in the startup log
# rather than on its first rollout. Never fatal: the endpoint named here is only a
# default, and a rollout that names its own works regardless of what this found.
if _LLM_URL and not _MODEL:
    try:
        _served = _served_models(_LLM_URL)
        # Only an unambiguous answer is usable. With several models and none named,
        # `opencode` would be pointed at a model id the endpoint does not recognise,
        # so leaving it empty and failing loudly per rollout beats guessing.
        _MODEL = _served[0] if len(_served) == 1 else ""
        if not _MODEL:
            logger.warning(
                "%s serves %d models and MODEL_ID is unset; set it explicitly.", _LLM_URL, len(_served)
            )
    except Exception as exc:  # noqa: BLE001 - startup must survive an unreachable default endpoint
        logger.warning("could not list models at %s: %s: %s", _LLM_URL, type(exc).__name__, exc)


class RolloutContext(BaseModel):
    # These names are RLE's wire contract, not ours. `model_endpoint`/`model_api_key` are the
    # route to the model, pointed at RLE's capture proxy so the trajectory is recorded;
    # `sandbox_tools_endpoint` addresses the per-rollout container that also serves `/reset` and
    # `/grade` (tools are a sibling of those, not a path under them), and `sandbox_tools_token` is
    # a rollout-scoped capability that expires with the rollout.
    #
    # RLE renamed three of these (`capture_proxy_endpoint` -> `model_endpoint`,
    # `capture_proxy_session_key` -> `model_api_key`, `sandbox_tools_bearer_token` ->
    # `sandbox_tools_token`). Both spellings are accepted because a harness is customer-deployed
    # and outlives any single RLE release: a sample that only understood the new names would break
    # against a region still serving the old ones, and the failure would surface as a 422 on
    # dispatch, which looks like a harness bug rather than a version skew.
    model_config = ConfigDict(protected_namespaces=())

    model_endpoint: str = Field(validation_alias=AliasChoices("model_endpoint", "capture_proxy_endpoint"))
    model_api_key: str = Field(validation_alias=AliasChoices("model_api_key", "capture_proxy_session_key"))
    sandbox_tools_endpoint: str
    sandbox_tools_token: str = Field(
        validation_alias=AliasChoices("sandbox_tools_token", "sandbox_tools_bearer_token"))


class InvocationRequest(BaseModel):
    rollout_id: str
    operation_id: str
    agent_input: dict[str, Any]
    rollout_context: RolloutContext


class RolloutStore:
    """Tracks what this process knows about each rollout it has accepted.

    Records are keyed by ``operation_id``, because that is the segment RLE puts in
    its poll and cancel URLs.

    A single process dictionary is enough for a sample, and it is the smallest
    thing that demonstrates the contract. Real harnesses that run more than one
    replica need shared state instead: RLE's poll can land on any replica, and a
    replica that has never heard of the rollout cannot answer for it.

    The record is created by the start request, before the acknowledgement is
    written, because RLE's first poll is immediate and may arrive before the
    harness has done anything at all.
    """

    def __init__(self) -> None:
        self._states: dict[str, dict[str, Any]] = {}
        # Tasks are held because asyncio only keeps a weak reference to a running
        # task; dropping this would let the garbage collector cancel the rollout
        # partway through.
        self._tasks: dict[str, asyncio.Task[None]] = {}

    def accept(self, operation_id: str, task: asyncio.Task[None]) -> None:
        self._states[operation_id] = {"status": "running"}
        self._tasks[operation_id] = task

    def succeed(self, operation_id: str, output_text: str) -> None:
        self._finish(operation_id, {"status": "succeeded", "output_text": output_text})

    def fail(self, operation_id: str, message: str) -> None:
        self._finish(operation_id, {"status": "failed", "error": {"message": message}})

    def read(self, operation_id: str) -> dict[str, Any] | None:
        state = self._states.get(operation_id)
        return dict(state) if state is not None else None

    def withdraw(self, operation_id: str) -> None:
        """Forgets the rollout and stops its agent loop."""
        self._states.pop(operation_id, None)
        task = self._tasks.pop(operation_id, None)
        if task is not None:
            task.cancel()

    def _finish(self, operation_id: str, state: dict[str, Any]) -> None:
        # A withdrawn rollout is gone. Recording an outcome for it would
        # resurrect a resource RLE has already stopped waiting on.
        if operation_id in self._states:
            self._states[operation_id] = state
        self._tasks.pop(operation_id, None)


ROLLOUTS = RolloutStore()


async def _resolve_rollout_model(endpoint: str, api_key: str) -> str:
    """Returns the model id this rollout's capture proxy serves.

    RLE binds the model to the rollout, not to this deployment: the proxy named in
    `rollout_context.model_endpoint` serves whatever checkpoint the training session samples,
    and nothing in the invoke payload names it. `MODEL_ID` names a model on the *default*
    endpoint and is resolved once at import, so reusing it here hands `opencode` an id the
    rollout is not running. The calls still reach the proxy and still get captured, which is
    what makes this worth resolving rather than assuming: `opencode` applies the named model's
    tool-calling and reasoning conventions to a different model, so the agent burns its turns
    and never writes an answer, and the rollout grades zero with no error anywhere.
    """
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(f"{endpoint.rstrip('/')}/models", headers=headers)
    response.raise_for_status()
    served = [m["id"] for m in response.json().get("data", []) if m.get("id")]
    # Only an unambiguous answer is usable, for the same reason as the startup probe.
    return served[0] if len(served) == 1 else ""


async def _rollout_model(rollout_context: RolloutContext, rlog: telemetry.RolloutLog) -> str:
    """Picks the model id to hand `opencode` for one rollout.

    Prefers what the rollout's own proxy reports; falls back to `MODEL_ID` so a local run
    against a plain endpoint still works. Failing to resolve is worth a warning rather than
    silence, because the fallback is the wrong model under RLE.
    """
    if rollout_context.model_endpoint:
        try:
            resolved = await _resolve_rollout_model(
                rollout_context.model_endpoint, rollout_context.model_api_key
            )
        except Exception as exc:  # noqa: BLE001 - an unreachable /models must not end the rollout
            rlog.event(
                "model_resolution_failed",
                level=logging.WARNING,
                endpoint=rollout_context.model_endpoint,
                error=f"{type(exc).__name__}: {exc}",
                fallback_model=_MODEL or None,
            )
        else:
            if resolved:
                rlog.event("model_resolved", model=resolved, endpoint=rollout_context.model_endpoint)
                return resolved
            rlog.event(
                "model_resolution_ambiguous",
                level=logging.WARNING,
                endpoint=rollout_context.model_endpoint,
                fallback_model=_MODEL or None,
            )
    if not _MODEL:
        raise RuntimeError(
            "no model to run: this rollout's endpoint named none and MODEL_ID is unset"
        )
    return _MODEL


async def run_harness_rollout(
    rollout_id: str, rollout_context: RolloutContext, agent_input: dict[str, Any], rlog: telemetry.RolloutLog
) -> str:
    """Runs one task and returns the agent's own answer text, as a JSON string.

    `opencode` runs here, in this container, as a child process of this service --
    see `opencode_direct.py`. The endpoint it is pointed at is RLE's capture proxy
    (`model_endpoint`), not a real model endpoint, so every call the agent makes
    lands in the rollout graph.

    The returned JSON becomes this invocation's `output_text`. RLE hands that same
    string back verbatim as `agent_response` on the `/grade` call (see
    `../rle/server/env.py`), which is what lets `/grade` compute the reward there.
    This service never computes or even sees a `reward`, and that is deliberate:
    the grader is public and deterministic, so a harness allowed to report its own
    score could simply report a perfect one.
    """
    task_index = int(agent_input.get("task_index", 0))
    split = agent_input.get("split", DEFAULT_SPLIT)
    model = await _rollout_model(rollout_context, rlog)
    rlog.event(
        "task_selected",
        task_index=task_index,
        split=split,
        model=model,
        # Correlation for the side-by-side demo: the model endpoint and sandbox-tools endpoint
        # RLE handed this rollout, so a Cloud Run log line can be matched against the
        # `azd ai rle rollout` CLI's own "model-call capture" / "sandbox" stages by URL, not just
        # by rollout_id.
        model_url=rollout_context.model_endpoint or _LLM_URL or None,
        sandbox_tools_url=rollout_context.sandbox_tools_endpoint or None,
        compliance_capability=bool(rollout_context.sandbox_tools_endpoint and rollout_context.sandbox_tools_token),
    )

    try:
        result = await opencode_direct.run_rollout(
            task_index=task_index,
            model=model,
            llm_url=rollout_context.model_endpoint or _LLM_URL,
            api_key=rollout_context.model_api_key or "",
            # Not rollout parameters: this is the capability the agent needs in
            # order to file a compliance disclosure against `../rle`'s
            # `/tools/report_sensitive_data_access`. It reaches the agent as
            # environment variables on that rollout's own `opencode` process.
            compliance_endpoint=rollout_context.sandbox_tools_endpoint,
            compliance_token=rollout_context.sandbox_tools_token,
            rollout_id=rollout_id,
            rlog=rlog,
        )
    except opencode_direct.RolloutError as exc:
        rlog.event("rollout_error", level=logging.ERROR, error=str(exc))
        raise RuntimeError(str(exc)) from exc

    if not result.get("ok", True):
        # A rollout the agent lost is not a harness failure: `../rle/server/env.py` grades
        # `ok: False` as a zero. Only the setup faults above -- which raise -- are failures.
        rlog.event("rollout_no_answer", level=logging.WARNING, error=result.get("error"))

    rlog.event(
        "answer_ready",
        answer_chars=len(result.get("answer_text") or ""),
        answer_present=result.get("answer_text") is not None,
    )

    # `../rle/server/env.py`'s `/grade` parses this same JSON back out of
    # `agent_response` -- keep the key names in sync with that.
    return json.dumps(
        {
            "split": split,
            "task_index": task_index,
            "answer_text": result.get("answer_text"),
            "ok": result.get("ok", True),
            "error": result.get("error"),
        }
    )


async def run_rollout(request: InvocationRequest) -> None:
    """Runs the rollout and records where the poll can find the answer.

    Nothing is raised out of here. The start request has already been answered by
    the time this runs, so a failure has nowhere to go except the poll resource.
    """
    rlog = telemetry.RolloutLog(logger, request.rollout_id, request.operation_id)
    try:
        output_text = await run_harness_rollout(
            request.rollout_id, request.rollout_context, request.agent_input, rlog
        )
    except asyncio.CancelledError:
        # RLE withdrew the rollout, which already removed it from the store.
        rlog.event("rollout_withdrawn")
        raise
    except Exception as error:  # noqa: BLE001 - every failure has to reach the poll
        # RLE deliberately does not echo this wording back to the caller, so an
        # async harness that does not log it has no record of why it failed.
        rlog.event("rollout_failed", level=logging.ERROR, error=str(error))
        ROLLOUTS.fail(request.operation_id, str(error))
        return
    rlog.event("rollout_succeeded")
    ROLLOUTS.succeed(request.operation_id, output_text)


@app.post("/invoke", status_code=202)
async def invoke(request: InvocationRequest) -> dict[str, int]:
    """Takes ownership of the rollout and returns immediately."""
    logger.info(
        "invoke received",
        extra={
            "rollout_id": request.rollout_id,
            "operation_id": request.operation_id,
            "event": "invoke_received",
            "fields": {
                "agent_input": request.agent_input,
                "model_url": request.rollout_context.model_endpoint or None,
                "sandbox_tools_url": request.rollout_context.sandbox_tools_endpoint or None,
            },
        },
    )
    task = asyncio.create_task(run_rollout(request))
    ROLLOUTS.accept(request.operation_id, task)
    return {"retry_after_ms": RETRY_AFTER_MS}


@app.get("/invoke/rollouts/{operation_id}")
async def poll(operation_id: str) -> JSONResponse:
    """Reports whether the rollout is still running, and its answer once it is not."""
    state = ROLLOUTS.read(operation_id)
    if state is None:
        return JSONResponse({"error": "not_found"}, status_code=404)
    return JSONResponse(state)


@app.delete("/invoke/rollouts/{operation_id}", status_code=204)
async def withdraw(operation_id: str) -> Response:
    """Stops work RLE is no longer waiting for.

    Idempotent on purpose: a repeat or a late arrival must not be an error, or
    RLE would record a cleanup failure for work that is already gone.
    """
    logger.info(
        "withdraw received",
        extra={"operation_id": operation_id, "rollout_id": "", "event": "withdraw_received", "fields": {}},
    )
    ROLLOUTS.withdraw(operation_id)
    return Response(status_code=204)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"ok": True, "model": _MODEL or None}
