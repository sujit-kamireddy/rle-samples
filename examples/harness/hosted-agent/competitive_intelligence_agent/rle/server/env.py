"""RLE harness container for the competitive-intelligence agent.

This is the *environment* half of a Harness/HostedAgent RLE. The agent half is
the real competitive-intelligence agent, published as a Foundry Hosted Agent
and bound to this environment by ``agentName``/``agentVersion`` in
``rle.toml``. RLE calls exactly four things here and nothing else:

    GET  /health   readiness, polled before /reset
    POST /reset    the caller's task, verbatim
    POST /tools/*  the agent's tool calls, proxied per rollout
    POST /grade    {"rollout": ..., "agent_response": "..."} -> a reward

There is no ``/step`` and no OpenEnv ``Environment`` here. Those belong to the
Gym/OpenEnv subtype, where RLE drives the model turn by turn. In a Harness RLE
the agent owns its own loop, so this container only sets the task up, serves
the tools, and scores the result.

## What is shared with Loom RL training

``rl/tasks.py``, ``rl/world.py``, ``rl/simulated_tools.py`` and
``rl/grading.py`` are imported here unmodified -- the same task generator, the
same simulated world, the same tools and the same rubric that
``rl/env.py`` uses for the Gym-style training runs. Sharing the modules rather
than reimplementing them is the point: a reward produced here means what a
reward produced there means, because it is the same code path. Only the
*driver* differs, and that difference is the reason for this environment. In
``rl/env.py`` Loom drives a re-implementation of the agent's decision step; here
the deployed agent drives itself, so what gets scored is the agent's real
phase chain, its real prompts and its real tool use.

## Tool surface

``routine``. Every evidence source the task holds is served as a callable tool:
``web_search`` for public sources, ``fabric_iq_query`` for the governed
portfolio row, ``onelake_knowledge_search`` for internal documents and their
``onelake://`` URLs, and ``org_context_lookup`` for organizational routing --
plus the two mutating decoys the rubric scores restraint against.

This deliberately diverges from the deployed scheduled path, which short-circuits
its Fabric and Work IQ phases and injects that context from disk. Injection made
the rubric partly unwinnable: a brief cannot cite an internal document the
session never served, and stakeholders only count as grounded once a lookup has
served them. An agent handed its evidence cannot be scored on gathering it, nor
improve at it. ``work_iq_search`` stays off the surface because production runs
this path with ``WORK_IQ_MODE=mock``, so the live lookup is unreachable there
too. See ``rl/README.md`` section 8.

## Where the taskset lives

Not in this image. A taskset is the caller's: every Execute Rollout call
carries its own task and RLE posts it to ``/reset`` exactly as the calling job
supplied it. The task rows are the ones ``rl/build_dataset.py`` generates, so
adding, reweighting or holding out tasks is a dataset change and never an
image rebuild.
"""

from __future__ import annotations

import logging
import os
import json
import re
import traceback
from typing import Any, Optional
from uuid import uuid4

from fastapi import Body, FastAPI, Header, HTTPException
from loom_cookbook.tool_use import ToolInput

from rl.grading import grade_episode
from rl.simulated_tools import ToolSession, session_tools
from rl.tasks import Task
from rl.world import build_world

# The simulated world is deterministic and read-only once built, and building
# it walks the whole company/team/product graph. Rollouts share one instance
# rather than each paying for a rebuild; `ToolSession` keeps all per-rollout
# state.
WORLD = build_world()

#: The surface both halves of a rollout must agree on. The harness container
#: and the agent package read the *same* variable name with the *same*
#: default, which is the only reason they cannot silently disagree -- they
#: are deployed separately and nothing plumbs the value from one to the
#: other. `tests/test_rle_vendor_parity.py` asserts the two defaults match.
#:
#: Default `production`: `routine` serves four invented semantic tools that
#: the deployed agent does not have. A policy trained on it scored 0.97 tool
#: discipline in the simulator and 0.43 on the real benchmark, because only
#: `web_search` was common to both. `routine` is kept to reproduce those runs.
TOOL_SURFACE = os.environ.get("COMPETITIVE_INTEL_TOOL_SURFACE", "production")

logger = logging.getLogger("competitive-intel-rle")

app = FastAPI(title="competitive-intel-rle")


class Rollout:
    """The one attempt this container is currently hosting.

    A sandbox serves a single rollout at a time, so one module-level record is
    enough. A container serving several at once would key this by the
    ``x-rle-rollout-id`` header instead.
    """

    def __init__(self) -> None:
        self.rollout_id: Optional[str] = None
        self.task: Optional[Task] = None
        self.session: Optional[ToolSession] = None
        self.tools: dict[str, Any] = {}

    def start(self, rollout_id: str, task: Task) -> ToolSession:
        session = ToolSession(
            task=task,
            world=WORLD,
            work_iq_mode="mock",
            tool_surface=TOOL_SURFACE,
        )
        self.rollout_id = rollout_id
        self.task = task
        self.session = session
        self.tools = {t.name: t for t in session_tools(session, TOOL_SURFACE)}
        return session

    def require(self) -> tuple[Task, ToolSession]:
        if self.task is None or self.session is None:
            raise HTTPException(status_code=409, detail="No active rollout; call /reset first.")
        return self.task, self.session


ROLLOUT = Rollout()


@app.get("/health")
async def health() -> dict[str, str]:
    """Readiness probe.

    RLE polls this before ``/reset``: a sandbox reports Running once the
    container is scheduled, which is well before uvicorn has bound its port.
    """
    return {"status": "healthy"}


@app.post("/reset")
async def reset(
    task: dict[str, Any] = Body(default_factory=dict),
    rollout_id: Optional[str] = Header(default=None, alias="x-rle-rollout-id"),
) -> dict[str, Any]:
    """Sets one rollout up from the caller's task.

    The body is the task exactly as the calling job supplied it -- one row of
    ``data/rl/*.jsonl``. RLE does not wrap it, and reads nothing from this
    response beyond its status code, so what is returned here is for a human
    driving the container by hand with ``azd ai rle run``. The agent gets the
    task's context by calling the retrieval tools this container serves.
    """
    try:
        parsed = Task.from_json(task)
    except (KeyError, TypeError) as error:
        # Grading a task this container failed to understand would report a
        # reward for work the caller never requested, so fail loudly instead.
        raise HTTPException(
            status_code=400,
            detail=f"Task row could not be parsed: {error}",
        ) from error

    rollout_id = rollout_id or str(uuid4())
    ROLLOUT.start(rollout_id, parsed)
    logger.info("reset rollout=%s task=%s variant=%s", rollout_id, parsed.task_id, parsed.variant)
    return {
        "rollout_id": rollout_id,
        "task_id": parsed.task_id,
        "variant": parsed.variant,
        "query": parsed.query,
        "tool_surface": TOOL_SURFACE,
        "tools": sorted(ROLLOUT.tools),
    }


@app.post("/tools/{tool_name}")
async def call_tool(
    tool_name: str,
    arguments: dict[str, Any] = Body(default_factory=dict),
    call_id: Optional[str] = Header(default=None, alias="x-rle-tool-call-id"),
) -> dict[str, Any]:
    """Serves one simulated tool call for the active rollout.

    Dispatches through ``FunctionTool.run``, the same entry point Loom's episode
    loop uses, so arguments are validated against the same pydantic model and a
    bad call produces the same error payload the policy would see in training.
    A call recorded here is therefore the call the rubric scores.

    ``run`` never raises: invalid arguments and tool faults both come back as an
    error ``ToolResult``. That is deliberate -- a malformed call is the agent's
    mistake, and the rubric needs to see the wasted call rather than have the
    whole attempt fault.
    """
    _, _session = ROLLOUT.require()
    tool = ROLLOUT.tools.get(tool_name)
    if tool is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Tool {tool_name!r} is not on the {TOOL_SURFACE!r} surface. "
                f"Available: {sorted(ROLLOUT.tools)}."
            ),
        )
    result = await tool.run(ToolInput(arguments=arguments, call_id=call_id or ""))
    messages = result.messages or []
    # The tool's single message carries a JSON string, and that string is what
    # the model sees as the tool's output during training.
    content = messages[0].get("content", "") if messages else ""
    return {"content": content}


_OUTPUT_TEXT_START = re.compile(r'"type"\s*:\s*"output_text"\s*,\s*"text"\s*:\s*"')


def _salvage_output_text(envelope_text: str) -> list[str]:
    """The ``output_text`` parts of an envelope ``json.loads`` cannot read.

    RLE sanitises the rollout before handing it to ``/grade``, rewriting
    anything that looks like a credential to ``[REDACTED]``. Response ids
    (``caresp_0958111b...``), session ids and the model endpoint all match that
    shape, and the substitution lands *inside* the serialised Responses
    envelope, so the envelope stops being valid JSON: on
    ``ftjob-c78645c32cf846dab90c4aca`` 116 of 245 rollouts failed to parse.

    Returning the raw envelope in that case -- which is what falling through to
    ``agent_response`` does -- hands ``parse_decision`` a report whose quotes are
    all backslash-escaped, so it never matches. ``format`` scores 0, and every
    dimension that needs a parsed decision collapses to 0 with it. The damage is
    not subtle: rollouts whose verdict was still *exactly right* scored 0.03
    instead of 0.84, and the parse rate tracked the reward exactly -- 100% and
    0.665 at step 0 against 32.5% and 0.340 at step 2. Training was mostly
    ranking whether redaction happened to corrupt the envelope, which is
    independent of anything the policy controls, so the run could not hillclimb.

    Scanning for the text parts directly sidesteps the damage, because the
    corruption is almost always in the ids rather than in the report. That
    recovers 102 of the 116; the remaining 14 are genuine ``Loom sampling
    failed`` 502s whose text is an error message and *should* score near zero.
    """
    chunks: list[str] = []
    for match in _OUTPUT_TEXT_START.finditer(envelope_text):
        index = match.end()
        buffer: list[str] = []
        while index < len(envelope_text):
            char = envelope_text[index]
            if char == "\\":
                buffer.append(envelope_text[index : index + 2])
                index += 2
                continue
            if char == '"':
                break
            buffer.append(char)
            index += 1
        raw = "".join(buffer)
        try:
            chunks.append(json.loads(f'"{raw}"'))
        except json.JSONDecodeError:
            # A redaction inside the report itself can eat an escape sequence.
            # The text is still worth more than the envelope around it.
            chunks.append(raw)
    return chunks


def _final_text(agent_response: str) -> str:
    """The agent's report, out of whatever shape RLE handed back.

    For a HostedAgent environment, `agent_response` is the serialised Responses
    object, not the report: ``{"id": "caresp_...", "output": [...]}``. Grading it
    as-is looks for the fenced decision block inside a JSON envelope, where the
    report's own quotes are backslash-escaped, so `parse_decision` never matches.
    Every rollout then scored an identical 0.278 -- the floor from the three
    dimensions that score without a decision -- which is a constant reward, and a
    constant reward is zero GRPO advantage and zero gradient. The run looked
    healthy right up to `grad_norm 0.0`.

    A plain string is passed through, which is what the local end-to-end test and
    a Gym-style caller both send.
    """
    text = agent_response.strip()
    if not text.startswith("{"):
        return agent_response
    try:
        envelope = json.loads(text)
    except json.JSONDecodeError:
        # Sanitisation broke the envelope. Read the text parts out of it anyway
        # rather than grading the envelope; see `_salvage_output_text`.
        salvaged = _salvage_output_text(text)
        return "\n\n".join(salvaged) if salvaged else agent_response
    if not isinstance(envelope, dict) or "output" not in envelope:
        return agent_response

    chunks: list[str] = []
    for item in envelope.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for part in item.get("content") or []:
            if isinstance(part, dict) and part.get("type") == "output_text":
                chunks.append(str(part.get("text") or ""))
    if not chunks:
        # A failed response carries `output: []` and an error instead. Grade the
        # error text: it scores near zero, and it says why in the metrics.
        error = envelope.get("error")
        if isinstance(error, dict):
            return f"ROLLOUT FAILED: {error.get('code')}: {error.get('message')}"
    return "\n\n".join(chunks)


#: RLE reads a known error field out of a failed response, then caps it at 200
#: characters (`RolloutUpstreamError.MaxDetailLength`) and cuts the tail. So the
#: cause is written shortest-first -- type, then message, then the frame it came
#: from -- and the message is bounded so the frame survives the cut. A whole
#: traceback would arrive as noise; it goes to the log, which this container
#: keeps, rather than into a field sized for a sentence.
_MAX_GRADING_DETAIL = 200
_MAX_GRADING_MESSAGE = 96


def _grading_failure_detail(error: BaseException) -> str:
    """Names a grading crash compactly enough to survive RLE's 200-char cap."""
    parts = [f"grading failed: {type(error).__name__}"]
    message = " ".join(str(error).split())
    if message:
        if len(message) > _MAX_GRADING_MESSAGE:
            message = message[:_MAX_GRADING_MESSAGE].rstrip() + "..."
        parts.append(f": {message}")
    frames = traceback.extract_tb(error.__traceback__)
    if frames:
        parts.append(f" at {os.path.basename(frames[-1].filename)}:{frames[-1].lineno}")
    return "".join(parts)[:_MAX_GRADING_DETAIL]


@app.post("/grade")
async def grade(payload: dict[str, Any] = Body(default_factory=dict)) -> dict[str, Any]:
    """Scores the finished rollout with the training rubric, unchanged.

    RLE sends ``{"rollout": <sanitized rollout>, "agent_response": "<final
    answer>"}``. The reward comes from ``rl.grading.grade_episode``, which reads
    both the agent's final text and the tool calls recorded on the session --
    so tool discipline, evidence fidelity and action restraint are scored from
    what the agent *did*, not only from what it said.

    An unhandled exception here used to leave Starlette to answer, which returns
    the plain-text body ``Internal Server Error``. RLE only extracts a message
    from a JSON ``error``/``message``/``detail`` field, so nothing was extracted
    and the caller was told a bare ``HTTP 500`` -- the rollout was lost with no
    way to say why from outside this container. Answering with a JSON ``detail``
    is what makes the cause travel.

    The status stays 5xx, which RLE classifies as retryable, and that is right
    here even though the rubric is deterministic and takes no I/O: its input is
    the agent's own sampled text, so a retry re-samples the input that broke it
    and may well grade cleanly. Validation groups are size one, so a failure not
    retried is a permanently lost eval point.
    """
    task, session = ROLLOUT.require()
    try:
        agent_response = _final_text(payload.get("agent_response") or "")
        result = await grade_episode(task, session, agent_response)
    except HTTPException:
        raise
    except Exception as error:
        logger.exception(
            "grade failed rollout=%s task=%s variant=%s",
            ROLLOUT.rollout_id,
            task.task_id,
            task.variant,
        )
        raise HTTPException(
            status_code=500, detail=_grading_failure_detail(error)
        ) from error
    raw_reward = result.reward
    # Managed RLE requires [0, 1]. This is a backstop, not the mechanism:
    # `grade_episode` already maps the format penalty's [-FORMAT_COEF, 1] onto
    # [0, 1] by rescaling, which keeps the ordering between a badly formatted
    # report and an empty one. Clamping alone would flatten every unparsed
    # rollout scoring under FORMAT_COEF to exactly 0.0, and a group whose
    # members all score 0.0 yields no advantage and no gradient -- precisely
    # the rollouts a cold policy produces most. Do not drop the rescale on the
    # grounds that this line catches it.
    reward = min(1.0, max(0.0, raw_reward))
    # RLE surfaces `info` as the caller's `result`; anything reported outside
    # it is dropped.
    return {
        "reward": reward,
        "is_success": bool(result.metrics.get("verdict_exact", 0.0) >= 1.0),
        "info": {
            "task_id": task.task_id,
            "variant": task.variant,
            "raw_reward": raw_reward,
            "metrics": result.metrics,
            "parsed": result.decision.parsed,
            "verdict": result.decision.material,
            "expected_verdict": task.material,
            "n_tool_calls": len(session.calls),
            "tools_called": [call.name for call in session.calls],
            "agent_response": agent_response,
        },
    }
