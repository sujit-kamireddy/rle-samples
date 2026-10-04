"""Responses host for the competitive-intelligence agent under RLE.

Structurally the same host as the deployed agent's ``main.py`` -- the same
``ResponsesAgentServerHost``, the same input parsing, the same event stream
shape (each tool call and its output as function-call items, then the report as
a message). What changes is where the model and the tools live, and that is
decided per request from the headers RLE attaches.

Production traffic and rollout traffic both land here. When the rollout headers
are absent, :meth:`RolloutContext.in_rollout` is false and the request is
rejected rather than quietly answered against a half-configured client: this
build exists to be trained, and the deployed ``main.py`` is what serves users.
Keeping one entry point that recognises both cases is what makes the Hosted
Agent subtype worth using -- the artifact being trained is the artifact being
run, not a training-only copy of it that can drift away from it.

Nothing here is cached across requests. RLE runs rollouts concurrently against
one warm container, and the model endpoint, its key, and the tool routes are
all rollout-scoped; a client built once at import would serve every rollout
through whichever request happened to build it, mixing two rollouts' model
calls into one capture and corrupting both trajectories.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import traceback
from collections.abc import AsyncIterator

from azure.ai.agentserver.responses import (
    CreateResponse,
    ResponseContext,
    ResponsesAgentServerHost,
    ResponsesServerOptions,
)
from azure.ai.agentserver.responses.aio.streaming import ResponseEventStream
from azure.ai.agentserver.responses.models import ResponseStreamEvent
from openai import AsyncOpenAI

from pipeline import CompetitiveIntelligenceAgent
from request_parsing import parse_response_input
from rollout_context import RolloutContext, RolloutTools
from telemetry import configure as _configure_telemetry, logger, short as _short

# Configured at import so startup lines are visible, and re-asserted per request
# because the host can silence this logger when it starts. See `telemetry`.

#: Sent as the ``model`` field on every completion. RLE's capture proxy routes
#: by session, not by this name, but it records it -- so it names the policy
#: under training rather than the production deployment.
MODEL_NAME = os.environ.get("RLE_MODEL_NAME", "competitive-intelligence-policy")

#: A rollout's whole phase chain, not one model call. Five sequential calls at
#: minimum -- three research phases, analysis, report -- and up to thirteen when
#: every phase spends its full tool budget, plus the tool round-trips between
#: them.
#:
#: This was 108s, sized to clear "the gateway's ~120s cut on Execute Rollout".
#: That cut is real but it belongs to the *HTTP* transport; `azd ai rle rollout`
#: drives Execute Rollout over a WebSocket, which is not cut -- the service-side
#: budget there is `RleExecuteRolloutOptions.RequestTimeoutSeconds`, 1800s, and
#: `HostedAgentRolloutInvoker` sets nothing shorter. Measured: a rollout ran the
#: agent to the full 108s and still returned a graded result over the socket.
#:
#: 108 was therefore not protecting against anything on this path, and it was
#: costing whole rollouts: a Qwen3-32B run spent 102.5s across seven model calls
#: -- thinking tokens dominate, 355-1045 completion tokens per call -- and was
#: killed mid-chain with four good tool calls already made and no report.
#:
#: Keep it well under 1800s so the agent still fails itself first: its own
#: timeout returns timings and a transcript, whereas overrunning the service
#: returns a bare error with neither. Lower it via the environment for the HTTP
#: transport, where the ~120s cut does apply.
#:
#: Raised from 420s alongside RLE_MODEL_TIMEOUT_S. 420 left no room for a cold
#: Loom engine: one 240s model load plus the six or seven normal calls that
#: follow it overruns 420 even though every individual call succeeded, which
#: would turn a slow start into a dropped rollout rather than a slow one. A
#: healthy run's rollouts measured 250s at the median and 513s at the maximum,
#: so 420 was already cutting the tail of a *warm* engine.
ROLLOUT_TIMEOUT_S = float(os.environ.get("RLE_ROLLOUT_TIMEOUT_S", "900"))

app = ResponsesAgentServerHost(
    options=ResponsesServerOptions(
        # Production runs the chain as a durable multi-turn task so a long
        # research run survives a restart. A rollout is a single bounded
        # attempt that RLE discards and reissues on failure, so resilience
        # here would add checkpoint writes and recovery paths to a trajectory
        # that is thrown away anyway.
        resilient_background=False,
        steerable_conversations=False,
        sse_keep_alive_interval_seconds=15,
    )
)


def _build_agent(context: ResponseContext) -> CompetitiveIntelligenceAgent:
    """Assembles this request's agent from its rollout headers."""
    rollout = RolloutContext.from_headers(context.client_headers)
    if not rollout.in_rollout:
        raise RuntimeError(
            "This build serves RLE rollouts only and requires the "
            "x-client-rle-* headers. Production traffic is served by the "
            "deployed agent's main.py."
        )
    client = AsyncOpenAI(
        base_url=rollout.model_endpoint,
        api_key=rollout.model_api_key or "unused",
        # One model call gets 240s. The SDK default is a 600s timeout with
        # retries, so an endpoint that stalls consumes the whole rollout and is
        # reported as a bare 408 with nothing to debug from.
        #
        # This was 100s, which was under the floor rather than above it. A Loom
        # session whose base model is not resident has to load it before it can
        # serve the first token, and a run that starts against a cold engine
        # sends its whole first eval wave into that load at once. At 100s every
        # one of them timed out, so no call was captured, so every capture came
        # back `trainable: false` and the failure-rate breaker ended the run at
        # 100% -- diagnosed as a dead agent three times before the timings in
        # the rollout trace showed a single clean 100.09s cut.
        #
        # 240s is chosen to sit above a cold load and still well below
        # ROLLOUT_TIMEOUT_S, so a genuinely stalled endpoint is still the
        # agent's own timeout rather than the service's.
        timeout=float(os.environ.get("RLE_MODEL_TIMEOUT_S", "240")),
        max_retries=0,
    )
    return CompetitiveIntelligenceAgent(
        context=rollout,
        model=MODEL_NAME,
        client=client,
        tools=RolloutTools(rollout),
        # The agent retries a slow model call only while the rollout can still
        # pay for it, so it has to know what it was given.
        rollout_budget_s=ROLLOUT_TIMEOUT_S,
    )


@app.response_handler
async def handle_response(
    request: CreateResponse,
    context: ResponseContext,
    cancellation_signal: asyncio.Event,
) -> AsyncIterator[ResponseStreamEvent]:
    """Runs one research turn and returns its tools and report as a response."""
    # Re-asserted here, not only at import: the host configures logging when it
    # starts, which is after this module is imported, and that can silence this
    # logger. See `_configure_telemetry`.
    _configure_telemetry()
    input_text = await context.get_input_text()
    topic, execution_mode = parse_response_input(input_text)
    agent = _build_agent(context)
    rollout_id = agent._context.rollout_id

    stream = ResponseEventStream(response_id=context.response_id, request=request)
    yield stream.emit_created()
    yield stream.emit_in_progress()

    # The rollout id is the join key for the demo: it is what `azd ai rle
    # rollout` prints, what selects this log stream, and what tags every line
    # below. Logging it first means a viewer can confirm the two panes are
    # showing the same rollout before anything interesting happens.
    logger.info(
        "[%s] REQUEST accepted  mode=%s  timeout=%ss",
        _short(rollout_id),
        execution_mode,
        ROLLOUT_TIMEOUT_S,
    )
    started = time.monotonic()
    try:
        run = await asyncio.wait_for(agent.run(topic), timeout=ROLLOUT_TIMEOUT_S)

        # Emitting each tool call and its output keeps the response shape identical
        # to production's, so a rollout transcript and a production transcript can
        # be read side by side.
        for tool in run.tool_executions:
            call_id = str(tool["call_id"])
            async for event in stream.output_item_function_call(
                name=str(tool["name"]),
                call_id=call_id,
                arguments=json.dumps(tool["arguments"], ensure_ascii=True),
            ):
                yield event
            async for event in stream.output_item_function_call_output(
                call_id=call_id,
                output=str(tool["output"]),
            ):
                yield event

        report = run.report
    except Exception:
        # An exception escaping this generator reaches RLE as a bare
        # `{"output": [], "status": "failed", "error": "server_error"}`: no
        # traceback and no partial transcript. The runtime logs *are* retrievable
        # -- `azd ai agent monitor --session-id <rollout-id> --follow` replays
        # this container's stdout -- but that is a second pane someone has to be
        # watching, and the reward lands in the first one. Two rollouts that
        # failed for completely different reasons scored an identical 0.278,
        # which read like a stable baseline rather than the total failure it was.
        #
        # Returning the traceback as the report costs nothing and makes the
        # failure legible: `/grade` puts it in `agent_response`, so it shows up
        # in metrics and logs. It still scores near zero, which is the honest
        # reward for a rollout that produced no decision.
        report = "ROLLOUT ERROR\n" + traceback.format_exc()
        # A rollout that overruns is reported to the caller with no clue where
        # the time went, so the per-call timings are folded into the response
        # body itself rather than left only in the logs.
        report += "\nCALL TIMINGS\n" + json.dumps(agent.call_timings, indent=2)
        report += f"\nMODEL ENDPOINT {getattr(agent, '_context', None) and agent._context.model_endpoint!r}"
        logger.exception(
            "[%s] REQUEST FAILED after %.1fs",
            _short(rollout_id),
            time.monotonic() - started,
        )
    else:
        # Counts what reached the sandbox, matching `ROLLOUT done` and the
        # grader's `n_tool_calls`. `run.tool_executions` also carries the calls
        # the agent answered itself, which would report a higher number here
        # than either of the other two views of the same rollout.
        logger.info(
            "[%s] REQUEST complete  %.1fs  %d tool call(s)",
            _short(rollout_id),
            time.monotonic() - started,
            len(agent.tool_timings),
        )

    # The report is the last message, and it is what RLE hands to `/grade` as
    # `agent_response`.
    async for event in stream.output_item_message(report):
        yield event
    yield stream.emit_completed()


if __name__ == "__main__":
    app.run()
