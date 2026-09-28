"""Foundry Hosted Agent for the data-code agent RLE.

Build and register this agent, then point ``rle/rle.toml``'s ``agentName`` /
``agentVersion`` at it and run ``azd ai rle publish``. Unlike the ``BYOH``
sample there is no URL to register and nothing to host yourself: RLE resolves
the agent from the project and calls its Responses API directly.

This is the whole harness. It speaks the Responses protocol *and* runs the
agent: a rollout fetches its task's input files and drives ``opencode`` in this
container against the task instruction, pointed at the model endpoint RLE
supplies -- RLE's capture proxy, so the rollout graph sees every call the agent
makes. See ``opencode_direct.py`` for the rollout itself and ``../README.md``
for how this and ``../rle`` deploy together.

How RLE calls a Hosted Agent
----------------------------
One synchronous request per rollout. There is no acknowledgement, no poll
resource and no cancel leg -- all of which the ``BYOH`` sample has to
implement, because BYOH is asynchronous and RLE polls a URL you registered.
Here RLE calls::

    POST {project}/agents/{agentName}/endpoint/protocols/openai/responses?api-version=v1
    Authorization: Bearer <caller token>
    x-ms-client-request-id: <rollout id>
    Foundry-Features: HostedAgents=V1Preview
    x-client-rle-rollout-id: ...
    x-client-rle-model-endpoint: https://.../rle/v1.0/capture-proxy/v1
    x-client-rle-model-api-key: ...
    x-client-rle-sandbox-tools-endpoint: https://.../rollouts/<rollout-id>/tools
    x-client-rle-sandbox-tools-token: ...

    {
      "agent_session_id": "<rollout id>",
      "input": [{"type": "message", "role": "user",
                 "content": [{"type": "input_text", "text": "<agent_input>"}]}],
      "background": false, "stream": false, "store": false
    }

``agent_input`` is not a JSON object here. RLE serialises whatever was
configured -- for this sample ``{"task_index": 0, "split": "..."}`` -- into that
single ``input_text``, so it arrives as a JSON *string* that has to be parsed
back out. ``parse_agent_input`` below does that, and also accepts a bare task
index so ``azd ai rle rollout --agent-input 3`` works.

``agentVersion`` from ``rle.toml`` is target metadata only: RLE records it but
does not send it on the wire, so this container cannot read it.

RLE takes the final assistant message text as this rollout's ``output_text``
and forwards that same string verbatim to ``../rle``'s ``/grade`` as
``agent_response``. That is how the answer gets graded. This service never
computes or even sees a ``reward``: the grader is public and deterministic, so
a harness allowed to report its own score could simply report a perfect one.

If a response comes back with no final assistant text, RLE sends one follow-up
turn on the same ``agent_session_id`` asking for a final answer. This handler
always emits a message, so that path is never taken.

Timeouts
--------
The service-side budget for Execute Rollout is 1800s
(``RleExecuteRolloutOptions.RequestTimeoutSeconds``) and the Hosted Agent
invoker sets nothing shorter, so a long ``opencode`` run is viable here. The
one exception is the *HTTP* Execute Rollout path, which a Foundry data-plane
gateway cuts at roughly 120s; ``azd ai rle rollout`` and training both drive
Execute Rollout over a WebSocket, which is not cut. ``ROLLOUT_TIMEOUT_S`` below
keeps this agent failing itself first -- an agent that times out returns a
transcript and a zero, while overrunning the service returns a bare error and
loses the episode. Lower it via the environment if you are calling the HTTP
path.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import traceback
from collections.abc import AsyncIterator
from typing import Any

import httpx
from azure.ai.agentserver.responses import (
    CreateResponse,
    ResponseContext,
    ResponsesAgentServerHost,
    ResponsesServerOptions,
)
from azure.ai.agentserver.responses.aio.streaming import ResponseEventStream
from azure.ai.agentserver.responses.models import ResponseStreamEvent

import opencode_direct
from rollout_context import RolloutContext

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger("hosted-agent-data-code-agent")

DEFAULT_SPLIT = os.environ.get("HARNESS_SPLIT", "FineEnvs/data-agent-harbor-train")

#: Whole-rollout budget: fetching the task's input files plus one `opencode`
#: run, which the task index itself times out at `agent_timeout_sec` (600s for
#: most tasks). Sized to sit above that and well under the service's 1800s so
#: the agent is always the one that gives up first.
ROLLOUT_TIMEOUT_S = float(os.environ.get("RLE_ROLLOUT_TIMEOUT_S", "900"))

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


def parse_agent_input(input_text: str) -> dict[str, Any]:
    """Recovers the task selector from the Responses input text.

    The Hosted Agent protocol carries one text field, so RLE serialises
    ``agent_input`` into it. A configured object arrives as its JSON text; a
    bare value arrives as itself. Both forms select a task, so both are
    accepted -- rejecting the bare form would break
    ``azd ai rle rollout --agent-input 3``, which is the quickest way to
    exercise a published agent.
    """
    text = (input_text or "").strip()
    if not text:
        return {}
    try:
        decoded: Any = json.loads(text)
    except json.JSONDecodeError:
        decoded = text
    if isinstance(decoded, dict):
        return decoded
    if isinstance(decoded, bool):
        # `bool` is an `int` in Python, and `True` would silently select task 1.
        raise ValueError(f"agent_input must name a task index or a task object, got {text!r}")
    if isinstance(decoded, int):
        return {"task_index": decoded}
    if isinstance(decoded, str) and decoded.strip().isdigit():
        return {"task_index": int(decoded.strip())}
    raise ValueError(f"agent_input must name a task index or a task object, got {text!r}")


async def _resolve_rollout_model(endpoint: str, api_key: str) -> str:
    """Returns the model id this rollout's capture proxy serves.

    RLE binds the model to the rollout, not to this deployment: the proxy named in
    the `x-client-rle-model-endpoint` header serves whatever checkpoint the training
    session samples, and nothing in the request names it. `MODEL_ID` names a model on
    the *default* endpoint and is resolved once at import, so reusing it here hands
    `opencode` an id the rollout is not running. The calls still reach the proxy and
    still get captured, which is what makes this worth resolving rather than assuming:
    `opencode` applies the named model's tool-calling and reasoning conventions to a
    different model, so the agent burns its turns and never writes an answer, and the
    rollout grades zero with no error anywhere.
    """
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(f"{endpoint.rstrip('/')}/models", headers=headers)
    response.raise_for_status()
    served = [m["id"] for m in response.json().get("data", []) if m.get("id")]
    # Only an unambiguous answer is usable, for the same reason as the startup probe.
    return served[0] if len(served) == 1 else ""


async def _rollout_model(rollout: RolloutContext) -> str:
    """Picks the model id to hand `opencode` for one rollout.

    Prefers what the rollout's own proxy reports; falls back to `MODEL_ID` so a local run
    against a plain endpoint still works. Failing to resolve is worth a warning rather than
    silence, because the fallback is the wrong model under RLE.
    """
    if rollout.model_endpoint:
        try:
            resolved = await _resolve_rollout_model(
                rollout.model_endpoint, rollout.model_api_key or ""
            )
        except Exception as exc:  # noqa: BLE001 - an unreachable /models must not end the rollout
            logger.warning(
                "could not list models at %s: %s: %s; falling back to MODEL_ID=%r",
                rollout.model_endpoint, type(exc).__name__, exc, _MODEL,
            )
        else:
            if resolved:
                return resolved
            logger.warning(
                "%s named no single model; falling back to MODEL_ID=%r",
                rollout.model_endpoint, _MODEL,
            )
    if not _MODEL:
        raise RuntimeError(
            "no model to run: this rollout's endpoint named none and MODEL_ID is unset"
        )
    return _MODEL


async def run_harness_rollout(rollout: RolloutContext, agent_input: dict[str, Any]) -> str:
    """Runs one task and returns the agent's own answer text, as a JSON string.

    `opencode` runs here, in this container, as a child process of this service --
    see `opencode_direct.py`. The endpoint it is pointed at is RLE's capture proxy
    (`x-client-rle-model-endpoint`), not a real model endpoint, so every call the
    agent makes lands in the rollout graph.

    The returned JSON becomes this rollout's `output_text`. RLE hands that same
    string back verbatim as `agent_response` on the `/grade` call (see
    `../rle/server/env.py`), which is what lets `/grade` compute the reward there.
    """
    task_index = int(agent_input.get("task_index", 0))
    split = agent_input.get("split", DEFAULT_SPLIT)
    model = await _rollout_model(rollout)

    try:
        result = await opencode_direct.run_rollout(
            task_index=task_index,
            model=model,
            llm_url=rollout.model_endpoint or _LLM_URL,
            api_key=rollout.model_api_key or "",
            # Not rollout parameters: this is the capability the agent needs in
            # order to file a compliance disclosure against `../rle`'s
            # `/tools/report_sensitive_data_access`. It reaches the agent as
            # environment variables on that rollout's own `opencode` process.
            compliance_endpoint=rollout.sandbox_tools_endpoint or "",
            compliance_token=rollout.sandbox_tools_token or "",
            rollout_id=rollout.rollout_id or "",
        )
    except opencode_direct.RolloutError as exc:
        raise RuntimeError(str(exc)) from exc

    if not result.get("ok", True):
        # A rollout the agent lost is not a harness failure: `../rle/server/env.py` grades
        # `ok: False` as a zero. Only the setup faults above -- which raise -- are failures.
        logger.warning("Rollout %s produced no answer: %s", rollout.rollout_id, result.get("error"))

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


app = ResponsesAgentServerHost(
    options=ResponsesServerOptions(
        # A rollout is a single bounded attempt that RLE discards and reissues
        # on failure, so durable checkpointing would add recovery paths to a
        # trajectory that is thrown away anyway.
        resilient_background=False,
        steerable_conversations=False,
        sse_keep_alive_interval_seconds=15,
    )
)


@app.response_handler
async def handle_response(
    request: CreateResponse,
    context: ResponseContext,
    cancellation_signal: asyncio.Event,
) -> AsyncIterator[ResponseStreamEvent]:
    """Runs one rollout and returns its answer as the final assistant message."""
    rollout = RolloutContext.from_headers(context.client_headers)
    agent_input = parse_agent_input(await context.get_input_text())

    stream = ResponseEventStream(response_id=context.response_id, request=request)
    yield stream.emit_created()
    yield stream.emit_in_progress()

    logger.info(
        "rollout %s accepted  timeout=%ss  %s",
        rollout.rollout_id or "(none)", ROLLOUT_TIMEOUT_S, rollout.describe(),
    )
    started = time.monotonic()
    try:
        if not rollout.in_rollout:
            raise RuntimeError(
                "no x-client-rle-model-endpoint header: this build answers RLE rollouts "
                "only, and without the capture proxy nothing would be recorded."
            )
        output_text = await asyncio.wait_for(
            run_harness_rollout(rollout, agent_input), timeout=ROLLOUT_TIMEOUT_S
        )
    except Exception:  # noqa: BLE001 - every failure has to reach the grader as text
        # An exception escaping this generator reaches RLE as a bare
        # `{"output": [], "status": "failed"}`: no traceback, nothing in the
        # reward to distinguish two rollouts that failed for unrelated reasons.
        # Returning the traceback as the answer costs nothing and makes the
        # failure legible -- `/grade` puts it in `agent_response`, so it shows
        # up in metrics and logs, and it still scores zero, which is the honest
        # reward for a rollout that produced no answer.
        logger.exception("rollout %s failed after %.1fs", rollout.rollout_id, time.monotonic() - started)
        output_text = json.dumps(
            {
                "split": agent_input.get("split", DEFAULT_SPLIT),
                "task_index": agent_input.get("task_index"),
                "answer_text": None,
                "ok": False,
                "error": "ROLLOUT ERROR\n" + traceback.format_exc(),
            }
        )
    else:
        logger.info("rollout %s complete in %.1fs", rollout.rollout_id, time.monotonic() - started)

    async for event in stream.output_item_message(output_text):
        yield event
    yield stream.emit_completed()


if __name__ == "__main__":
    app.run()
