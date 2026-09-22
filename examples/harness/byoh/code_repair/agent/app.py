"""Example BYOH agent harness for the code-repair RLE.

Deploy this anywhere reachable over HTTPS and register its URL with
``azd ai rle init --type Harness --subtype BYOH --base-url <this-url>/invoke``.

RLE invokes a harness asynchronously. The start request only starts work; the
answer is collected from a per-rollout resource RLE derives from the registered
URL by appending ``/rollouts/{rollout_id}``. A harness never supplies that
address, so RLE only ever calls back under the path you registered.

1. Start. RLE POSTs the rollout to the registered URL and the harness
   acknowledges with ``202`` as soon as it has taken ownership -- before the
   agent loop has run. ``retry_after_ms`` is advisory; RLE clamps it.

       POST <base-url>
       {
         "rollout_id": "...",
         "agent_input": {"...": "agent-specific input"},
         "rollout_context": {
           "model_endpoint": "https://.../v1",
           "model_api_key": "...",
           "sandbox_tools_endpoint": "https://.../tools",
           "sandbox_tools_bearer_token": "..."
         }
       }

       202 Accepted
       {"retry_after_ms": 500}

2. Poll. RLE GETs the rollout resource until it reports an outcome. The first
   poll is immediate, so a harness that finishes at once costs one extra round
   trip rather than a poll interval.

       GET <base-url>/rollouts/{rollout_id}

       200 {"status": "running"}
       200 {"status": "succeeded", "output_text": "..."}
       200 {"status": "failed", "error": {"message": "..."}}

3. Withdraw. If RLE stops waiting -- the caller disconnected, or the rollout
   deadline passed -- it DELETEs the rollout resource so the harness can stop
   spending tokens. RLE never reads the response, and may not send it at all, so
   treat it as advisory and keep your own timeout.

       DELETE <base-url>/rollouts/{rollout_id}

RLE never attaches caller, workspace, or identity headers to these requests, so
protect the endpoint yourself (for example, require a shared secret header and
validate it before dispatching to the agent loop).

This file and ``../../../hosted-agent/code_repair/agent/main.py`` share the same
``run_agent_loop``: the harness's native agent loop is unchanged between the
two RLE subtypes. Only how ``model`` and ``call_tool`` are constructed
differs -- see ``create_model_client``/``call_tool`` here versus the
header-driven equivalents there. The context field names match too: each one is
its hosted-agent header minus the ``x-client-rle-`` prefix, with hyphens written
as underscores.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx
from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse
from openai import AsyncOpenAI
from pydantic import BaseModel

app = FastAPI(title="byoh-code-repair-agent")

logger = logging.getLogger("byoh-code-repair-agent")

RETRY_AFTER_MS = 500


class RolloutContext(BaseModel):
    model_endpoint: str
    model_api_key: str
    sandbox_tools_endpoint: str
    sandbox_tools_bearer_token: str


class InvocationRequest(BaseModel):
    rollout_id: str
    agent_input: dict[str, Any]
    rollout_context: RolloutContext


class RolloutStore:
    """Tracks what this process knows about each rollout it has accepted.

    A single process dictionary is enough for a sample, and it is the smallest
    thing that demonstrates the contract. Real harnesses that run more than one
    replica need shared state instead: RLE's poll can land on any replica, and a
    replica that has never heard of the rollout cannot answer for it.

    The record is created by the start request, before the acknowledgement is
    written, because RLE's first poll is immediate and may arrive before the
    agent loop has done anything at all.
    """

    def __init__(self) -> None:
        self._states: dict[str, dict[str, Any]] = {}
        # Tasks are held because asyncio only keeps a weak reference to a running
        # task; dropping this would let the garbage collector cancel the rollout
        # partway through.
        self._tasks: dict[str, asyncio.Task[None]] = {}

    def accept(self, rollout_id: str, task: asyncio.Task[None]) -> None:
        self._states[rollout_id] = {"status": "running"}
        self._tasks[rollout_id] = task

    def succeed(self, rollout_id: str, output_text: str) -> None:
        self._finish(rollout_id, {"status": "succeeded", "output_text": output_text})

    def fail(self, rollout_id: str, message: str) -> None:
        self._finish(rollout_id, {"status": "failed", "error": {"message": message}})

    def read(self, rollout_id: str) -> dict[str, Any] | None:
        state = self._states.get(rollout_id)
        return dict(state) if state is not None else None

    def withdraw(self, rollout_id: str) -> None:
        """Forgets the rollout and stops its agent loop."""
        self._states.pop(rollout_id, None)
        task = self._tasks.pop(rollout_id, None)
        if task is not None:
            task.cancel()

    def _finish(self, rollout_id: str, state: dict[str, Any]) -> None:
        # A withdrawn rollout is gone. Recording an outcome for it would
        # resurrect a resource RLE has already stopped waiting on.
        if rollout_id in self._states:
            self._states[rollout_id] = state
        self._tasks.pop(rollout_id, None)


ROLLOUTS = RolloutStore()


def create_model_client(rollout_context: RolloutContext) -> AsyncOpenAI:
    """Points the harness's normal model client at the endpoint RLE supplied.

    That endpoint is RLE's capture proxy. It speaks the same model dialect as
    the harness's real model endpoint and returns the ordinary response the
    harness expects, while recording a detailed trajectory for the rollout graph.
    """
    return AsyncOpenAI(
        base_url=rollout_context.model_endpoint,
        api_key=rollout_context.model_api_key,
    )


async def call_tool(rollout_context: RolloutContext, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Routes a tool call to RLE's sandbox instead of the real production tool."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            # sandbox_tools_endpoint already ends in /tools -- see the envelope
            # above. Appending another /tools produced /tools/tools/<name>,
            # which RLE forwards to the sandbox as a path it does not serve.
            f"{rollout_context.sandbox_tools_endpoint}/{tool_name}",
            json=arguments,
            headers={"Authorization": f"Bearer {rollout_context.sandbox_tools_bearer_token}"},
        )
        response.raise_for_status()
        return response.json()


# A harness's output contract belongs in its own system prompt. Unlike the
# Gym/OpenEnv subtype -- where RLE drives the model and the environment's
# observation carries the instruction -- here the harness owns the model call
# and the environment never sees the prompt. `/reset` hands over the raw
# GitHub issue, and a real issue does not tell you to answer in unified-diff
# form. Without this, a model answers the way a person would -- prose, or a
# fenced code block -- and the `workspace.apply_patch` mock's real `git apply`
# rejects it with "No valid patches in input", so the loop below returns early
# and the rollout scores 0 for a formatting reason rather than for a wrong fix.
PATCH_FORMAT_INSTRUCTION = """\
Reply with a unified diff and nothing else.

- Output only the diff: no explanation, no commentary, no Markdown code fences.
- Use the form `git diff` produces, with `a/` and `b/` path prefixes and `@@` hunk headers.
- Paths must be relative to the repository root.
- The diff must apply cleanly with `git apply`.

Expected shape:

--- a/pkg/module.py
+++ b/pkg/module.py
@@ -10,7 +10,7 @@ def example(value):
-    return value * 2
+    return value * 3
"""


async def run_agent_loop(model: AsyncOpenAI, rollout_context: RolloutContext, agent_input: dict[str, Any]) -> str:
    """The harness's native agent loop: read the issue, propose a patch, open a PR.

    Unchanged between RLE subtypes apart from how `model` and tool calls are
    routed -- see this module's ``create_model_client``/``call_tool`` versus
    ``../../../hosted-agent/code_repair/agent/main.py``'s header-driven equivalents.
    """
    issue = agent_input["issue"]

    completion = await model.chat.completions.create(
        model="code-repair-agent",
        messages=[
            {
                "role": "system",
                "content": f"You fix reported issues in this repository.\n\n{PATCH_FORMAT_INSTRUCTION}",
            },
            {"role": "user", "content": issue},
        ],
    )
    patch = completion.choices[0].message.content or ""

    apply_result = await call_tool(rollout_context, "workspace.apply_patch", {"patch": patch})
    if not apply_result.get("applied"):
        return f"Failed to apply patch: {apply_result.get('error')}"

    await call_tool(
        rollout_context,
        "github.create_pull_request",
        {
            "branch": "fix/reported-issue",
            "title": "Fix reported issue",
            "body": f"Applies the following patch in response to the report:\n\n{patch}",
        },
    )
    return "Opened a pull request with the fix."


async def run_rollout(request: InvocationRequest) -> None:
    """Runs the agent loop and records where the poll can find the answer.

    Nothing is raised out of here. The start request has already been answered by
    the time this runs, so a failure has nowhere to go except the poll resource.
    """
    try:
        model = create_model_client(request.rollout_context)
        output_text = await run_agent_loop(model, request.rollout_context, request.agent_input)
    except asyncio.CancelledError:
        # RLE withdrew the rollout, which already removed it from the store.
        logger.info("Rollout %s withdrawn", request.rollout_id)
        raise
    except Exception as error:  # noqa: BLE001 - every failure has to reach the poll
        # RLE deliberately does not echo this wording back to the caller, so an
        # async harness that does not log it has no record of why it failed.
        logger.exception("Rollout %s failed", request.rollout_id)
        ROLLOUTS.fail(request.rollout_id, str(error))
        return
    ROLLOUTS.succeed(request.rollout_id, output_text)


@app.post("/invoke", status_code=202)
async def invoke(request: InvocationRequest) -> dict[str, int]:
    """Takes ownership of the rollout and returns immediately."""
    task = asyncio.create_task(run_rollout(request))
    ROLLOUTS.accept(request.rollout_id, task)
    return {"retry_after_ms": RETRY_AFTER_MS}


@app.get("/invoke/rollouts/{rollout_id}")
async def poll(rollout_id: str) -> JSONResponse:
    """Reports whether the rollout is still running, and its answer once it is not."""
    state = ROLLOUTS.read(rollout_id)
    if state is None:
        return JSONResponse({"error": "not_found"}, status_code=404)
    return JSONResponse(state)


@app.delete("/invoke/rollouts/{rollout_id}", status_code=204)
async def withdraw(rollout_id: str) -> Response:
    """Stops work RLE is no longer waiting for.

    Idempotent on purpose: RLE sends this best-effort and never reads the
    response, so a repeat or a late arrival must not be an error.
    """
    ROLLOUTS.withdraw(rollout_id)
    return Response(status_code=204)


@app.get("/health")
async def health() -> dict[str, bool]:
    return {"ok": True}
