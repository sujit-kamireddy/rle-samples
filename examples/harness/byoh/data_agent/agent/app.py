"""Example BYOH agent harness for the FineEnvs data-agent RLE.

Deploy this anywhere reachable over HTTPS, then point ``rle/rle.toml``'s
``baseUrl`` at ``<this-url>/invoke`` and run ``azd ai rle publish``.

Unlike a harness that runs its own agent loop, this shim does not. The agent
loop, sandbox, and tool calls all live inside a separately deployed Harbor
environment server (``../harbor-server``) -- Hugging Face's `OpenEnv`
packaging of the `Harbor` project, the harness behind the
`FineEnvs/data-agent-harbor-*` datasets on Hugging Face. This shim's only job
is translating between RLE's `Harness`/`BYOH` invocation contract and Harbor's
own `run_rollout` call, so that RLE's rollout graph and capture proxy see the
one call Harbor's agent loop makes to the model. See ``../README.md`` for the
three-piece layout (harbor-server / agent / rle) and how they deploy together.

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
   acknowledges with ``202`` as soon as it has taken ownership -- before Harbor's
   `run_rollout` has run. ``retry_after_ms`` is advisory; RLE clamps it.

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

2. Poll. RLE GETs the rollout resource until it reports an outcome. Harbor's
   `run_rollout` is one blocking call that boots the sandbox, runs the agent to
   completion, and grades the workspace, so this can poll for minutes before an
   outcome appears.

       GET <base-url>/rollouts/{operation_id}

       200 {"status": "running"}
       200 {"status": "succeeded", "output_text": "{\"answer_text\": \"...\", \"split\": \"...\", ...}"}
       200 {"status": "failed", "error": {"message": "..."}}

   ``output_text`` is a JSON string, not free text: the sandbox's own raw
   answer (`answer_text`) plus the task selector, produced by
   `run_harbor_rollout` below. RLE forwards this same string to `../rle`'s
   `/grade` verbatim, as `agent_response` -- that is how the answer gets
   there. This shim never computes or even sees a `reward`; `/grade` grades
   `answer_text` itself, against its own vendored copy of Harbor's grader.

3. Withdraw. If RLE stops waiting -- the caller disconnected, or the rollout
   deadline passed -- it DELETEs the rollout resource so the harness can stop
   spending tokens. RLE checks only the status, and may not send this at all, so
   treat it as advisory and keep your own timeout -- but do answer 2xx, because
   RLE records anything else as a cleanup failure.

       DELETE <base-url>/rollouts/{operation_id}

RLE never attaches caller, workspace, or identity headers to these requests, so
protect the endpoint yourself (for example, require a shared secret header and
validate it before dispatching to Harbor).
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

app = FastAPI(title="byoh-data-agent")

logger = logging.getLogger("byoh-data-agent")

RETRY_AFTER_MS = 500

# The separately deployed harbor-server (../harbor-server) that owns the
# sandbox, the agent loop, and Harbor's own grader. This shim never runs any
# of that itself -- it only tells Harbor which task to run and which model
# endpoint to call, and relays back the sandbox's raw answer text. See
# `run_harbor_rollout` for what it relays and why grading it is `../rle`'s
# job, not this shim's.
HARBOR_SERVER_URL = os.environ["HARBOR_SERVER_URL"]
DEFAULT_SPLIT = os.environ.get("HARBOR_SPLIT", "FineEnvs/data-agent-harbor-train")
DEFAULT_HARNESS = os.environ.get("HARBOR_HARNESS", "opencode")
DEFAULT_SANDBOX = os.environ.get("HARBOR_SANDBOX", "e2b")
# `run_rollout` boots a sandbox, runs an agent to completion, and grades it in
# one blocking call -- minutes, not seconds. httpx's own default (5s) would
# abort a rollout that was actually still working.
_HARBOR_ROLLOUT_TIMEOUT_S = float(os.environ.get("HARBOR_ROLLOUT_TIMEOUT_S", "1800"))


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
    written, because RLE's first poll is immediate and may arrive before Harbor
    has done anything at all.
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


async def run_harbor_rollout(rollout_id: str, rollout_context: RolloutContext, agent_input: dict[str, Any]) -> str:
    """Asks the deployed harbor-server to run one task, pointed at RLE's model.

    Posts to harbor-server's `/correlated-rollouts/{rollout_id}` rather than calling
    `HarborEnv.run_rollout` directly only to get harbor-server's own request logging
    for free; the two otherwise trigger the identical rollout.

    Returns the sandbox's raw answer text (`answer_text`, harvested by harbor-server
    from `/workdir/answer.txt` before its sandbox was torn down) plus the task
    selector, as a JSON string that becomes this invocation's `output_text`. RLE
    hands that same string back verbatim as `agent_response` on the `/grade` call
    (see `../rle/server/env.py`), which is what lets `/grade` grade it there --
    this shim never computes or even sees a `reward`, so it has nothing to lie
    about beyond whether it ran the rollout at all.
    """
    task_index = agent_input.get("task_index", 0)
    split = agent_input.get("split", DEFAULT_SPLIT)
    harness = agent_input.get("harness", DEFAULT_HARNESS)
    sandbox = agent_input.get("sandbox", DEFAULT_SANDBOX)

    async with httpx.AsyncClient(timeout=_HARBOR_ROLLOUT_TIMEOUT_S) as client:
        response = await client.post(
            f"{HARBOR_SERVER_URL}/correlated-rollouts/{rollout_id}",
            json={
                "split": split,
                "task_index": task_index,
                "harness": harness,
                "sandbox": sandbox,
                # `llm_url`/`api_key` point Harbor's agent loop at RLE's capture
                # proxy instead of a real model endpoint -- the same wiring
                # `code_repair/agent/app.py`'s `create_model_client` does for a
                # harness that calls the model directly. Here Harbor makes the
                # call, but the endpoint it is told to call is still RLE's.
                "llm_url": rollout_context.model_endpoint,
                "api_key": rollout_context.model_api_key,
                # Not rollout parameters: this is the capability the agent needs
                # inside the sandbox to file a compliance disclosure against
                # `../rle`'s `/tools/report_sensitive_data_access`.
                # `../harbor-server` lifts them back out and turns them into
                # sandbox environment variables -- see its `rollout_tools.py`.
                "sandbox_tools_endpoint": rollout_context.sandbox_tools_endpoint,
                "sandbox_tools_bearer_token": rollout_context.sandbox_tools_token,
            },
        )
        response.raise_for_status()
        result = response.json()

    if not result.get("ok", True):
        raise RuntimeError(result.get("error") or "Harbor rollout failed with no error message")

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
    """Runs the Harbor rollout and records where the poll can find the answer.

    Nothing is raised out of here. The start request has already been answered by
    the time this runs, so a failure has nowhere to go except the poll resource.
    """
    try:
        output_text = await run_harbor_rollout(
            request.rollout_id, request.rollout_context, request.agent_input
        )
    except asyncio.CancelledError:
        # RLE withdrew the rollout, which already removed it from the store.
        logger.info("Rollout %s withdrawn", request.rollout_id)
        raise
    except Exception as error:  # noqa: BLE001 - every failure has to reach the poll
        # RLE deliberately does not echo this wording back to the caller, so an
        # async harness that does not log it has no record of why it failed.
        logger.exception("Rollout %s failed", request.rollout_id)
        ROLLOUTS.fail(request.operation_id, str(error))
        return
    ROLLOUTS.succeed(request.operation_id, output_text)


@app.post("/invoke", status_code=202)
async def invoke(request: InvocationRequest) -> dict[str, int]:
    """Takes ownership of the rollout and returns immediately."""
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
    ROLLOUTS.withdraw(operation_id)
    return Response(status_code=204)


@app.get("/health")
async def health() -> dict[str, bool]:
    return {"ok": True}
