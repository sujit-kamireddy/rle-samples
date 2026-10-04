"""The competitive-intelligence agent's routine phase chain, RLE-routed.

This is the agent half of the Harness/HostedAgent RLE. It runs the same phases
the deployed agent runs on its scheduled ``routine`` path, in the same order,
with the same prompt text, against the same tool surface -- and it is the thing
RLE trains. A Hosted Agent RLE exists to hillclimb the agent already in
production, so the reward has to be earned by the agent's own loop rather than
by a re-implementation of its decision step.

## The routine path is five model calls, three of them with tools

``phases.PHASES`` has five entries. On the deployed routine path two of them
make no model call at all -- ``agent.py._execute_phase`` short-circuits Fabric
and Work IQ, substituting a fixed snapshot and a mock payload read off disk.
Here all three research phases are real, tool-calling phases:

    1. public_evidence    ``web_search`` (capped at 2 by ``phases.py``)
    2. fabric_context     ``fabric_iq_query``, ``onelake_knowledge_search``
    3. work_context       ``org_context_lookup``
    4. decision_analysis  model call, tools attached but unused
    5. final_report       model call, tools attached but unused, emits the brief

The decoys (``send_email``, ``update_tracker_record``) are offered at every
research phase, because restraint is only measurable if the wrong action is
available at every point the agent could take it.

Retrieval is the agent's job here for a reason. Injecting the context made two
rubric dimensions unwinnable: a brief cannot cite an internal document the
policy was never served, and stakeholders are only scored as grounded if the
session actually served them. An agent handed its evidence cannot be scored on
gathering evidence, and cannot improve at it either -- which is the whole point
of training. Retrieval also costs two extra model calls, and the rollout has to
fit inside RLE's Execute Rollout window.

## Prompts are read, never copied

Phase objectives come from ``phases.py`` -- the deployed agent's own module. The
crew that owns this agent owns its prompt optimization, so a reworded objective
has to reach training by itself; a copy here would silently train against a
prompt production had already moved off.

System instructions are ``rl.prompts.ROUTINE_SYSTEM_PROMPT``: production's
``.agent_configs/baseline/instructions.md`` plus the machine-readable decision
block. Those are the instructions the Loom RL runs already train against, so a
reward measured here means the same thing as a reward measured there.

The final turn pairs that system prompt with ``rl.prompts.routine_user_prompt``,
which is the exact pair ``rl/env.py`` uses for the Gym environment. That is what
makes a reward from this environment comparable to the Gym baseline rather than
merely similar to it.

It is also what makes the rollout gradeable at all. An earlier version asked for
the report with ``reporting.synthesis_prompt``, production's Markdown brief
format. The model wrote the brief faithfully and omitted the fenced JSON
decision block, because that prompt never asks for one -- so
``rl.grading.parse_decision`` found nothing on every rollout and the reward sat
at the no-decision floor. The brief is the human deliverable; the block is what
makes its central claim checkable. Grading prose alone would mean inferring the
verdict the agent reached, and a rubric that has to guess the answer cannot
score it.

## What differs from the deployed agent, and why

Deliberate, and confined to infrastructure the reward does not read:

* **No durable checkpointing.** Production wraps the chain in a
  ``multi_turn_task`` so a long research run survives a restart. A rollout is a
  single bounded attempt RLE discards on failure anyway, so the chain runs
  inline. Checkpointing changes latency and recovery, not the report.
* **No OneLake publish and no report ledger.** Both are side effects on tenant
  storage. A training run doing them a few thousand times would write junk into
  a real lakehouse.
* **Governed context is retrieved, not injected.** Production's snapshot is
  fixed and ships in its image; here the context varies per task, so it is
  served by the environment as tools the policy has to choose to call.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from openai import APIConnectionError, APIStatusError, AsyncOpenAI

from phases import PHASES, ResearchPhase
from rl_prompts import PRODUCTION_SYSTEM_PROMPT, ROUTINE_SYSTEM_PROMPT, routine_user_prompt
from rollout_context import RolloutContext, RolloutTools
from telemetry import (
    clip as _clip,
    compact as _compact,
    decision as _decision,
    digest as _digest,
    endpoint as _endpoint,
    logger,
    short as _short,
)
from tool_specs import TOOL_SPECS_BY_SURFACE

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

#: Research phases, in order, and the retrieval tools each one offers, per
#: surface.
#:
#: Production filters its toolbox with ``ResearchPhase.tool_keywords``, but those
#: keywords describe the names production's toolbox publishes, not the simulated
#: surface, so the mapping is spelled out here instead of inferred.
#:
#: On the production surface there is no Work IQ tool: the deployed toolbox has
#: no org-context endpoint, and stakeholder routing comes out of the semantic
#: model's own ``Stakeholders`` table. The phase therefore keeps its objective
#: but spends it on the same Fabric tools, which is what the real agent does.
RESEARCH_PHASE_TOOLS_BY_SURFACE: dict[str, dict[str, tuple[str, ...]]] = {
    "routine": {
        "public_evidence": ("web_search",),
        "fabric_context": ("fabric_iq_query", "onelake_knowledge_search"),
        "work_context": ("org_context_lookup",),
    },
    "production": {
        "public_evidence": ("web_search",),
        "fabric_context": (
            "competitive_fabric___DiscoverArtifacts",
            "competitive_fabric___GetSemanticModelSchema",
            "competitive_fabric___ExecuteQuery",
            "competitive_fabric___ValueSearch",
        ),
        "work_context": (
            "competitive_fabric___ExecuteQuery",
            "competitive_fabric___GetReportMetadata",
            "competitive_fabric___ResolveReportIdFromUrl",
        ),
    },
}

RESEARCH_PHASE_TOOLS: dict[str, tuple[str, ...]] = RESEARCH_PHASE_TOOLS_BY_SURFACE[
    TOOL_SURFACE
]

#: The specs for the surface in force.
SURFACE_TOOL_SPECS = TOOL_SPECS_BY_SURFACE[TOOL_SURFACE]

#: System instructions matched to the surface. The rules are the same on
#: both; what differs is the tool names and the discovery protocol they
#: require, and a prompt that names tools the environment does not serve is
#: worse than no prompt at all.
SURFACE_SYSTEM_PROMPT = (
    PRODUCTION_SYSTEM_PROMPT if TOOL_SURFACE == "production" else ROUTINE_SYSTEM_PROMPT
)

#: Offered at every research phase. Restraint is only measurable if the wrong
#: action is available at every point the agent could take it.
DECOY_TOOL_NAMES = ("send_email", "update_tracker_record")

#: Tool-call budget for a phase ``phases.py`` leaves uncapped. A rollout has to
#: finish inside RLE's Execute Rollout window.
DEFAULT_MAX_TOOL_CALLS = 3

#: The budget has to scale with the *protocol*, not just the question.
#:
#: On the routine surface one `fabric_iq_query` returns the governed row, so
#: three calls is generous. Production publishes a discovery protocol instead:
#: DiscoverArtifacts -> GetSemanticModelSchema -> ExecuteQuery for the exposure
#: row -> ExecuteQuery for the documents, and ValueSearch when a filter literal
#: has to be found first. That is four calls before any evidence arrives, so a
#: budget of three would cut the phase off mid-protocol and train the policy to
#: skip evidence gathering -- penalising it for the protocol rather than for
#: waste, which is exactly the failure this realignment exists to remove.
#: Per-phase tool-call caps that the *surface* changes, keyed by surface.
#:
#: `phases.py` is the source of truth for a phase's budget and the fallback
#: below inherits it, so a phase only appears here when the production tool
#: surface makes the phase cost something different from what production pays.
#: `fabric_context` deliberately does not appear: the Optimizer-tuned config
#: caps it at 5 and training must spend exactly what the deployed agent spends,
#: or the policy learns a budget it will not have at inference time.
#:
#: `work_context` does: production reaches it with Work IQ tools, and there is
#: no Work IQ tool on the real toolbox, so the phase has to re-enter Fabric --
#: which costs a discovery call before it can query anything.
SURFACE_MAX_TOOL_CALLS: dict[str, dict[str, int]] = {
    "routine": {},
    "production": {"work_context": 4},
}

#: Output budget per model call. Sized for a *reasoning* model, not for the
#: answer -- which is why it looks generous for a one-line search query.
#:
#: A reasoning model spends this budget on its analysis channel before it emits
#: the tool call, and a tool call that runs out of budget mid-arguments is not
#: a soft failure on this stack: it is fatal to the whole rollout. The capture
#: proxy reads the sampled tokens up to an end-of-message token, and when the
#: cap cut them short it takes whatever partial text it has as the arguments
#: verbatim (``capture_proxy/loom/mai.py:357`` -- the body scan simply stops at
#: ``len(response)``). The truncated call still counts as parsed, so the
#: response is not labelled ``finish_reason="length"``; it reaches
#: ``capture_proxy/loom/client.py:982``, where ``json.loads`` of the half-
#: written argument object fails and the call comes back as
#: ``502 tool-call arguments must be valid JSON``.
#:
#: That 502 is raised outside the renderer's own try/except, so it is not
#: retried or downgraded -- it propagates to this agent and ends the rollout
#: with no usable trace. At 1200 this reproduced on the *first* model call of
#: every rollout. Keep the headroom.
MAX_OUTPUT_TOKENS = 4096

#: How many times a model call may be attempted, initial try included.
#:
#: The client is built with ``max_retries=0`` (``main.py``), which was the right
#: call for the failure it was aimed at -- an endpoint that stalls, where the
#: SDK's own retries stack 600s timeouts until the rollout window is gone. It is
#: the wrong call for the failure that actually costs rollouts. On
#: ftjob-6e2120bcf67d4d91beaa317b six of the step-6 validation rollouts scored
#: exactly 0.0718 and were dropped with "Capture must be marked trainable"; the
#: bodies show a single 502 ``Loom sampling failed`` arriving 2.88s into the
#: first call, after which the agent had nothing to capture and the harness
#: dropped the whole group. They cluster in the seconds after
#: ``save_sampler_weights``, so the eval wave is meeting an engine mid
#: weight-swap. One sub-three-second blip should not cost a rollout.
#:
#: Raised 4 -> 6 once the eval waves of ftjob-2587a4d3e3074e6faaa24fa8 were
#: classified by capture document rather than by reward, which exposed a second
#: failure mode the floor-reward marker had been missing entirely: a rollout
#: that runs six to eight turns and *then* loses a model call, returning a
#: ``ROLLOUT ERROR`` traceback with ``parsed: false``. Bucketing each wave's
#: deaths by arrival time shows three different outage lengths rather than one.
#: Step 0 lost its first 8 rollouts inside 4s, which is a cold engine. Step 12
#: lost 23 in a single 170s burst. Step 8 lost 32 of 39 across a sustained 536s.
#: Four attempts over 85s of backoff covers only the first of those.
MODEL_CALL_ATTEMPTS = 6
#: Backoff before each retry, so a weight-swap has time to finish.
#:
#: This was ``(2.0, 6.0)``, which spends every attempt inside the first ten
#: seconds of an outage. Timestamping the 91 zero-turn rollouts of
#: ftjob-2587a4d3e3074e6faaa24fa8 shows they arrive in 15 bursts, not as
#: scattered singles -- 8, 13, 16 and 30 rollouts dying together -- and the
#: multi-rollout bursts span a median of 29s and a maximum of 143s. A ladder
#: that gives up at 10s is therefore guaranteed to miss the median outage it
#: was written for. ``(5, 20, 60)`` spans 85s of backoff across four attempts,
#: which covers the median comfortably and most of the tail, while staying a
#: small fraction of ``ROLLOUT_TIMEOUT_S``.
#:
#: Extended to ``(5, 20, 60, 120, 120)`` -- 325s across six attempts -- because
#: 85s covers none of the two outages that did the real damage: the 170s burst
#: that destroyed step 12's eval and the 536s one that destroyed step 8's. The
#: budget for this is already sitting idle. A clean rollout on that run took a
#: median 258s of its 900s window, so riding out a multi-minute outage costs
#: slack rather than anything the rollout needed, and the deadline check below
#: is what stops it from costing more than the slack.
MODEL_CALL_BACKOFF_S = (5.0, 20.0, 60.0, 120.0, 120.0)
#: The boundary between a *fast* failure and a *slow* one.
#:
#: This used to veto the retry outright, which is what made
#: ``APITimeoutError`` unretryable in practice. It now only classifies, because
#: the real constraint -- "do not let retries overrun the rollout window" -- is
#: enforced directly against the deadline by ``_affords_another_attempt``.
MODEL_CALL_RETRY_MAX_ELAPSED_S = 30.0
#: How many *slow* failures may be retried in a single model call.
#:
#: Slow failures are the expensive half: on ftjob-2587a4d3e3074e6faaa24fa8 half
#: of the zero-turn rollouts were a lone ``APITimeoutError`` at exactly 240.0s,
#: never retried, so the rollout spent a quarter of its budget and captured
#: nothing. Retrying one of those is worth it precisely because the first call
#: is what absorbs a cold engine load -- the same load that the 240s timeout
#: was raised from 100s to survive -- so the second attempt tends to meet a
#: warm engine. Allowing only one keeps the worst case at two timeouts rather
#: than letting a genuinely stalled endpoint eat the window a slice at a time.
MODEL_CALL_SLOW_RETRIES = 1


def _is_transient(exc: BaseException) -> bool:
    """Whether a failed model call is worth attempting again.

    5xx and connection drops are the server saying it could not serve this
    request, which is not a statement about the request. 4xx is, so it is not
    retried -- notably the 400 behind a 502 ``Loom sampling failed with HTTP
    400``, which means prompt plus output budget overran the context window and
    will overrun it just as surely on the second attempt.

    ``APITimeoutError`` *is* transient -- it subclasses ``APIConnectionError``
    and means the same thing, that the server did not serve this request. It
    used to be excluded here to keep retries from overrunning the rollout
    window, but that traded a real cost for a silent one: it made the single
    most common failure on ftjob-2587a4d3e3074e6faaa24fa8 (half of all
    zero-turn rollouts, a lone 240.0s timeout) permanently unretryable, so
    those rollouts captured nothing and their whole group of 8 was discarded.
    The window is now protected where it is actually spent, by
    ``MODEL_CALL_SLOW_RETRIES`` and the deadline check in
    ``_affords_another_attempt``, so this predicate can answer the question it
    is named for.
    """
    if isinstance(exc, APIConnectionError):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code >= 500
    return False


def _specs_for(phase_key: str) -> list[dict[str, Any]]:
    """The tool specs one research phase offers, in ``SURFACE_TOOL_SPECS`` order."""
    names = set(RESEARCH_PHASE_TOOLS.get(phase_key, ())) | set(DECOY_TOOL_NAMES)
    specs = [s for s in SURFACE_TOOL_SPECS if s["function"]["name"] in names]
    missing = names - {s["function"]["name"] for s in specs}
    if missing:
        raise KeyError(
            f"Phase {phase_key!r} asks for tools the agent does not offer: "
            f"{sorted(missing)}. Regenerate tool_specs.py."
        )
    return specs


@dataclass
class PhaseTrace:
    """One phase's output and the tool calls it made."""

    key: str
    text: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class AgentRun:
    """Everything one invocation produced."""

    report: str
    phases: list[PhaseTrace]

    @property
    def tool_executions(self) -> list[dict[str, Any]]:
        return [call for phase in self.phases for call in phase.tool_calls]


def _phase(key: str) -> ResearchPhase:
    for phase in PHASES:
        if phase.key == key:
            return phase
    raise KeyError(f"No phase named {key!r} in the deployed agent's PHASES.")


def _grounded_prompt(phase: ResearchPhase, topic: str, prior: Sequence[str]) -> str:
    """The prompt shape from ``agent.py._run_grounded_phase``."""
    return (
        f"Competitive intelligence topic: {topic}\n\n"
        f"Current phase: {phase.title}\n"
        f"Phase objective: {phase.instructions}\n\n"
        "Prior grounded outputs:\n" + ("\n\n".join(prior) if prior else "(none)")
    )


def _analysis_prompt(phase: ResearchPhase, topic: str, prior: Sequence[str]) -> str:
    """The prompt shape ``agent.py`` uses for a phase with no tools."""
    return (
        f"Competitive intelligence topic: {topic}\n\n"
        f"Objective: {phase.instructions}\n\n"
        "Grounded evidence:\n" + "\n\n".join(prior)
    )


#: Fields the materiality rule is defined on. A retrieved payload carrying any
#: of them is a governed record, whichever tool returned it: the production
#: surface serves them out of the semantic model through
#: ``competitive_fabric___ExecuteQuery`` and the routine surface through
#: ``fabric_iq_query``, so matching the payload rather than the tool name keeps
#: one rule true on both.
GOVERNED_RECORD_FIELDS = (
    "exposure_score",
    "strategic_alignment",
    "materiality_threshold",
)

#: Characters of verbatim record text carried into the tool-free turns. Whole
#: records are kept or dropped against this budget, never truncated: half a JSON
#: object is not evidence, and a number cut at the decimal point is worse than
#: an absent one.
GOVERNED_RECORD_BUDGET = 4000


def _governed_records(phases: Sequence[PhaseTrace]) -> list[str]:
    """The governed rows this run retrieved, as the tools returned them.

    Phases hand off in prose. A research phase reports its findings in its own
    words and only that retelling reaches the tool-free analysis and report
    turns, so whatever the phase saw but did not write down is gone when the
    phase ends -- including, sometimes, the row the verdict is defined on.

    Measured over 280 replicate validation rollouts of a policy held fixed, the
    exposure score reached the graded turn in 85% of rollouts, the strategic
    alignment in 67%, and both in 65%. So the handoff does drop the rule's
    operands, but far less often than a first pass suggested, and the cost is
    correspondingly modest: pairing within task to hold difficulty constant,
    having both operands is worth +0.027 reward (t=0.98, not significant) and
    +0.10 on evidence fidelity (t=1.73), which is where the effect actually
    lives. Citation grounding checks the reported exposure against the served
    value within 0.005, and a number that survived as a number passes that
    check where a number recalled through a summary does not.

    What this is *not* is the reason the eval curve is flat. The same
    measurement puts the handoff lottery at 12% of within-task reward variance
    (ICC 0.271 -> 0.329 when restricted to rollouts that kept both operands),
    so the other 88% is elsewhere and no amount of this fixes it.

    Only what this rollout actually retrieved is carried forward. A run that
    never queried the semantic model still reaches the verdict empty-handed, so
    retrieval keeps deciding the score -- the difference is that it now decides
    it through the evidence rather than through whether a summary happened to
    repeat a number.
    """
    records: list[str] = []
    budget = GOVERNED_RECORD_BUDGET
    for phase in phases:
        for call in phase.tool_calls:
            output = call.get("output") or ""
            if not any(field in output for field in GOVERNED_RECORD_FIELDS):
                continue
            # The protocol makes the model re-query as it narrows down, so the
            # same row comes back more than once. Repeating it wastes the budget
            # and reads as corroboration it is not.
            if output in records or len(output) > budget:
                continue
            records.append(output)
            budget -= len(output)
    return records


def _records_block(records: Sequence[str]) -> str:
    """The governed rows, framed as the retrieved evidence they are."""
    if not records:
        return ""
    body = "\n\n".join(records)
    return (
        "Governed records retrieved during this run, verbatim as the tools "
        f"returned them:\n\n{body}\n\n"
        "These are the authoritative values for the fields they contain. Where "
        "a summary above disagrees with them, they win."
    )


def _assistant_turn(message: Any, calls: Sequence[Any]) -> dict[str, Any]:
    """Re-encodes one assistant turn for the next request's history.

    A reasoning model emits ``<think>...</think>`` ahead of its answer and the
    server hands that back split out of ``content``. Replaying the turn
    without it re-renders the assistant message shorter than the one that was
    sampled, so the capture cannot see the next turn's prompt as an extension
    of this turn's sequence. Every turn is then recorded as its own root --
    ``per_turn_capture_only`` -- and an episode reward earned by the final
    report has no path back to the tool calls that earned it. Twelve turns
    train as twelve disconnected single-turn samples, which is why five
    successive learning rates all produced a flat eval curve.

    The reasoning has to go back inside ``content`` rather than in a separate
    ``reasoning_content`` field, because linkage is decided on exact tokens
    and only this form reproduces them. Qwen3's template takes two different
    branches:

    * Given ``reasoning_content``, it leaves ``content`` alone. Ours is the
      ``"\\n\\n"`` that trailed ``</think>``, which is truthy, so the template
      prepends a newline to the tool call: ``</think>\\n\\n\\n<tool_call>``.
      The sampled text had ``</think>\\n\\n<tool_call>``. One token of drift,
      at the first sampled token, and the prefix match fails.
    * Given the block inside ``content``, it splits the reasoning back out
      itself and ``lstrip``s what remains. That leaves ``content`` empty,
      the newline is not prepended, and the render is byte-identical.

    Measured against the real template on a captured turn: the first form
    diverges at character 2397, the second matches exactly. The ``</think>``
    guard keeps a server that already inlines the block from being wrapped
    twice.

    Qwen3's template only keeps reasoning on assistant turns that follow the
    last user message. Each phase opens its history with exactly one user
    message, so every assistant turn in a phase qualifies and the whole phase
    links into a single sequence.
    """
    content = message.content or ""
    # The OpenAI SDK's models allow extra fields, so this survives on the
    # parsed message when the server sends it and is simply absent when it
    # does not -- a non-reasoning model keeps the previous behaviour.
    reasoning = getattr(message, "reasoning_content", None)
    if reasoning and "</think>" not in content:
        content = f"<think>{reasoning}</think>{content}"
    turn: dict[str, Any] = {"role": "assistant", "content": content}
    if calls:
        turn["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            }
            for call in calls
        ]
    return turn


class CompetitiveIntelligenceAgent:
    """Runs one competitive-intelligence research turn.

    A new instance is built per request. The model client and the tool
    transport come from that request's :class:`RolloutContext`, because RLE's
    model proxy and tool routes are rollout-scoped: a client cached on the
    module would serve every concurrent rollout through whichever request built
    it last, attributing one rollout's model calls to another's capture and
    corrupting both trajectories.
    """

    def __init__(
        self,
        *,
        context: RolloutContext,
        model: str,
        client: AsyncOpenAI,
        tools: Optional[RolloutTools] = None,
        instructions: str = SURFACE_SYSTEM_PROMPT,
        max_output_tokens: int = MAX_OUTPUT_TOKENS,
        rollout_budget_s: Optional[float] = None,
    ) -> None:
        self._context = context
        self._model = model
        self._client = client
        self._tools = tools
        self._instructions = instructions
        self._max_output_tokens = max_output_tokens
        self._rollout_budget_s = rollout_budget_s
        # Set in `run()`, because the budget is measured from the moment the
        # work starts, not from the moment this object is built.
        self._deadline: Optional[float] = None
        self.call_timings: list[dict[str, Any]] = []
        self.tool_timings: list[dict[str, Any]] = []
        self.blocked_calls: list[str] = []
        self._tag = _short(context.rollout_id)
        self._phase_no = 0

    def _log(self, line: str) -> None:
        """One telemetry line, tagged with the rollout this process is serving.

        The tag matters more than it looks. One warm container serves several
        rollouts at once, so untagged lines from concurrent rollouts interleave
        into something unreadable. The tag is the short form of the rollout id,
        which is also the agent session id RLE opens (it sets
        ``agent_session_id`` to the rollout id), and the id the
        ``azd ai rle rollout`` CLI prints -- so a line here can be matched to a
        line there by eye.
        """
        logger.info("[%s] %s", self._tag, line)

    async def _create_with_retry(self, kwargs: dict[str, Any]) -> Any:
        """Issues one model call, retrying a fast transient failure.

        Every attempt is recorded in ``call_timings``, including the ones that
        succeed on retry. A rollout that only completed because the first
        sampling call 502'd and the second did not is a materially different
        rollout from one that went straight through, and the trace is where that
        gets noticed -- silently swallowing the retry would trade a visible
        failure for an invisible one.
        """
        last: BaseException
        slow_retries = 0
        for attempt in range(MODEL_CALL_ATTEMPTS):
            started = time.monotonic()
            try:
                return await self._client.chat.completions.create(**kwargs)
            except Exception as exc:
                elapsed = time.monotonic() - started
                self.call_timings.append(
                    {
                        "seconds": round(elapsed, 2),
                        "error": f"{type(exc).__name__}: {exc}"[:300],
                    }
                )
                last = exc
                slow = elapsed >= MODEL_CALL_RETRY_MAX_ELAPSED_S
                backoff = MODEL_CALL_BACKOFF_S[
                    min(attempt, len(MODEL_CALL_BACKOFF_S) - 1)
                ]
                retryable = (
                    _is_transient(exc)
                    and attempt < MODEL_CALL_ATTEMPTS - 1
                    # Checked for every retry now, not only slow ones. At 85s
                    # the ladder was too short to overrun anything by itself;
                    # at 325s a fast failure repeated six times sleeps long
                    # enough to eat the rollout window on its own, so the guard
                    # has to cover the case it was not originally written for.
                    # A caller that declared no budget keeps the old behaviour,
                    # where fast retries are unguarded.
                    and (
                        self._deadline is None
                        or self._affords_another_attempt(backoff, elapsed)
                    )
                )
                if retryable and slow:
                    # A slow failure has already spent real budget, so it is
                    # rationed once more by count, and -- unlike a fast one --
                    # is never retried at all without a declared budget.
                    retryable = (
                        slow_retries < MODEL_CALL_SLOW_RETRIES
                        and self._affords_another_attempt(backoff, elapsed)
                    )
                if not retryable:
                    self._log(
                        f"  <- model  FAILED after {elapsed:.1f}s  "
                        f"{type(exc).__name__}: {str(exc)[:200]}"
                    )
                    raise
                if slow:
                    slow_retries += 1
                self._log(
                    f"  <- model  TRANSIENT after {elapsed:.1f}s  "
                    f"{type(exc).__name__}: {str(exc)[:160]}  "
                    f"retry {attempt + 2}/{MODEL_CALL_ATTEMPTS} in {backoff:.0f}s"
                )
                await asyncio.sleep(backoff)
        raise last

    def _affords_another_attempt(self, backoff: float, elapsed: float) -> bool:
        """Whether the rollout can still pay for an attempt costing ``elapsed``.

        This is the guard that replaces the blanket ``APITimeoutError``
        exclusion. It answers the question that exclusion was a proxy for --
        will retrying overrun the rollout window? -- against the actual
        deadline, so a 240s timeout is retried when there is room for it and
        refused when there is not.

        Returning ``False`` without a deadline is deliberate. A caller that
        did not declare a budget gets exactly the old behaviour, where no slow
        failure is ever retried, rather than an unbounded retry loop.
        """
        if self._deadline is None:
            return False
        return (time.monotonic() + backoff + elapsed) <= self._deadline

    async def _complete(
        self,
        history: list[dict[str, Any]],
        *,
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> Any:
        """One model call.

        ``tools=None`` means "the model must not call a tool" -- the synthesis
        phases, where ``agent.py`` withdraws the manifest entirely. Here the
        manifest stays attached and ``tool_choice="none"`` forbids the call
        instead. Same guarantee, but the capture proxy can see it.

        The proxy labels a captured path as agent work only if its turns carry
        a tool manifest; everything else is auxiliary and untrainable
        (``capture_proxy/export.py:_assign_roles`` -- "title generators,
        summarisers, classifiers"). That heuristic reads a coding agent, which
        never withdraws its tools. This agent withdraws them for exactly the
        two phases the reward grades, so a literal port captured 55 trainable
        tokens out of ~2000 and trained on nothing but the tool calls.

        ``n_tools`` counts the manifest, not the choice
        (``capture_proxy/server.py:1010``), so attaching the manifest is enough
        to restore those phases to trainable.

        ``tool_choice="none"`` would state the prohibition directly, but the
        `openai_chat` path forwards it untranslated to the serving stack -- only
        the anthropic/google/responses dialects convert it -- and SGLang 500s on
        it. So the manifest goes on its own and the prompt carries the
        instruction, exactly as it already did. `run` drops any tool call that
        comes back on a synthesis turn.
        """
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "system", "content": self._instructions}, *history],
            "max_completion_tokens": self._max_output_tokens,
            "tools": tools or SURFACE_TOOL_SPECS,
        }
        offered = [t["function"]["name"] for t in kwargs["tools"]]
        # The endpoint is reported once, in the rollout's wiring line. Repeating
        # it on every call added 70 characters that never changed and pushed the
        # part that does change off the edge of a split pane.
        self._log(
            f"  -> model  msgs={len(kwargs['messages'])}"
            f"  tools={len(offered)}  max_out={self._max_output_tokens}"
        )
        started = time.monotonic()
        completion = await self._create_with_retry(kwargs)
        elapsed = time.monotonic() - started
        usage = getattr(completion, "usage", None)
        choice = completion.choices[0]
        requested = list(getattr(choice.message, "tool_calls", None) or [])
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        # `web_search, web_search` and `web_search x2` say the same thing; the
        # second leaves room for the token counts on the same line.
        wanted = Counter(c.function.name for c in requested)
        # Truncating here is safe in a way it is not on the `offers` line: each
        # tool that actually gets dispatched prints its own `>>` line straight
        # after, so a clipped name is recoverable from the next two lines.
        detail = (
            "wants " + _clip(
                ", ".join(
                    name if n == 1 else f"{name} x{n}" for name, n in wanted.items()
                ),
                30,
            )
            if requested
            else f"answered {len((choice.message.content or '').strip())}ch"
        )
        self._log(
            f"  <- model  {elapsed:.1f}s  {prompt_tokens}->{completion_tokens}tok"
            f"  {detail}"
        )
        self.call_timings.append(
            {
                "seconds": round(time.monotonic() - started, 2),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }
        )
        return completion.choices[0].message

    async def _complete_text(
        self, history: list[dict[str, Any]], *, what: str,
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> str:
        """A synthesis call, which must come back as prose rather than a tool call.

        The manifest stays attached on these turns so the capture proxy scores
        them as agent work, which leaves the model free to call a tool when it
        was asked to write. A tool call here arrives with ``content = None``, and
        taking that as the empty string yields an empty report and a near-zero
        reward -- indistinguishable from a model that answered badly. So ask
        once more, telling it plainly that the tools are spent.

        ``tools`` is threaded through because *which* manifest is attached
        decides whether the turn links. A synthesis that closes a retrieval
        phase runs on that phase's own history, so it extends a prefix the
        capture has already seen -- but only if the manifest is the one that
        phase used. Falling through to the full surface rewrites the tool block
        at the front of the prompt, and since tools render ahead of the
        conversation, every token after them shifts and the prefix match fails
        on a turn that was otherwise a perfect continuation. Measured on
        rollout b3bc01d1: breaks at turns 2 and 9, both synthesis calls, tool
        count 3 -> 9 and 6 -> 9, divergence at token 1444 in the manifest
        rather than anywhere in the conversation.
        """
        message = await self._complete(history, tools=tools)
        if (message.content or "").strip():
            return message.content
        nudge = list(history) + [
            {
                "role": "user",
                "content": (
                    f"The research budget is spent and the tools are no longer "
                    f"available. Write the {what} directly, as prose."
                ),
            }
        ]
        retry = await self._complete(nudge, tools=tools)
        return retry.content or ""

    async def _call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        if self._tools is None:
            raise NotImplementedError(
                f"Wire {name!r} to the production toolbox to serve non-rollout traffic."
            )
        # `send_email` and `update_tracker_record` are offered but must never be
        # used: they are the acting-on-the-world tools, and reaching for one is
        # what `score_tool_discipline` penalises. It is the most consequential
        # thing a rollout can do, so it is marked rather than left to read like
        # any other call, and it is a warning so it survives a coarser filter.
        egress = name in DECOY_TOOL_NAMES
        # Model traffic uses `->`/`<-` and sandbox traffic `>>`/`<<`, so the two
        # kinds of call a rollout makes are one glance apart in the pane rather
        # than one word apart. It also keeps each marker greppable on its own.
        kind = "EGRESS!" if egress else "  >>"
        line = f"{kind} {name}  {_compact(arguments, limit=44)}"
        if egress:
            logger.warning("[%s] %s", self._tag, line)
        else:
            self._log(line)
        started = time.monotonic()
        try:
            output = await self._tools.call(name, arguments)
        except Exception as exc:
            elapsed = time.monotonic() - started
            self._log(
                f"  <- {kind} {name} FAILED after {elapsed:.1f}s  "
                f"{type(exc).__name__}: {str(exc)[:200]}"
            )
            raise
        elapsed = time.monotonic() - started
        # An error the environment *answers* with is a 200 carrying an error
        # body, so it does not raise. Surfacing it is the point: a rollout that
        # quietly retried bad arguments three times looks identical to a
        # well-behaved one in a latency trace.
        verdict = "ERROR" if output.lstrip().startswith('{"error"') else "ok"
        self.tool_timings.append(
            {
                "name": name,
                "seconds": round(elapsed, 2),
                "verdict": verdict,
                "egress": egress,
            }
        )
        # The digest replaced the byte count: a reader who can see "2 results:
        # Google introduces Agent2Agent" does not also need "834ch", and the
        # 8 columns it cost pushed the digest off the edge of a split pane.
        # A verdict is only printed when there is something wrong with it.
        flag = "ERROR " if verdict != "ok" else ""
        self._log(f"  << {name}  {elapsed:.1f}s  {flag}{_digest(output, limit=42)}")
        return output

    async def _research_phase(
        self, phase_key: str, topic: str, prior: Sequence[str]
    ) -> PhaseTrace:
        """One research phase: the model retrieves, then reports its findings."""
        phase = _phase(phase_key)
        specs = _specs_for(phase_key)
        offered = {spec["function"]["name"] for spec in specs}
        cap = SURFACE_MAX_TOOL_CALLS[TOOL_SURFACE].get(
            phase_key, phase.max_tool_calls or DEFAULT_MAX_TOOL_CALLS
        )
        self._phase_no += 1
        retrieval = [n for n in offered if n not in DECOY_TOOL_NAMES]
        # Two lines, not one. The offered tools are what makes a later
        # `!! blocked` line legible -- it is only meaningful against what this
        # phase allowed -- so clipping the list to fit would hide the half that
        # matters. At the widest real surface the single line ran 90 columns
        # against a 70-column budget, and no abbreviation of the banner closed
        # a 20-column gap without truncating a tool name.
        self._log(f"-- PHASE {self._phase_no}/5 {phase_key}  budget {cap}")
        self._log(f"   offers {_clip(', '.join(sorted(retrieval)), 56)}")
        history: list[dict[str, Any]] = [
            {"role": "user", "content": _grounded_prompt(phase, topic, prior)}
        ]
        executions: list[dict[str, Any]] = []
        # Production caps phase 1 (`ResearchPhase.max_tool_calls`) and leaves the
        # rest uncapped. Honour its cap where it sets one and apply a default
        # where it does not: an uncapped loop would spend the rollout's latency
        # budget on behaviour production does not allow, and train for it.
        for _ in range(cap):
            message = await self._complete(history, tools=specs)
            calls = list(getattr(message, "tool_calls", None) or [])
            history.append(_assistant_turn(message, calls))
            if not calls:
                self._log(
                    f"-- PHASE {self._phase_no}/5 {phase_key} done  "
                    f"{len(executions)} call(s), model stopped early"
                )
                return PhaseTrace(phase.key, message.content or "", executions)
            for call in calls:
                try:
                    arguments = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError:
                    # The environment validates arguments and answers with an
                    # error the model can read, so an unparsable call is still
                    # a call: it goes through, and it counts against the cap.
                    arguments = {}
                # Only dispatch what this phase actually offered. The model can
                # emit any name, and the environment's tool routes are reachable
                # with this rollout's token, so an unchecked forward would let a
                # guessed name reach a route the rubric does not watch.
                if call.function.name not in offered:
                    self.blocked_calls.append(call.function.name)
                    self._log(
                        f"  !! blocked {call.function.name}"
                        f" -- not offered in {phase_key}"
                    )
                    output = (
                        f"No tool named {call.function.name!r} is available in "
                        f"this phase. Available: {sorted(offered)}."
                    )
                else:
                    output = await self._call_tool(call.function.name, arguments)
                executions.append(
                    {
                        "call_id": call.id,
                        "name": call.function.name,
                        "arguments": arguments,
                        "output": output,
                    }
                )
                history.append(
                    {"role": "tool", "tool_call_id": call.id, "content": output}
                )

        # Budget spent: ask for the phase's findings with the tools withdrawn.
        self._log(
            f"-- PHASE {self._phase_no}/5 {phase_key} budget spent "
            f"({len(executions)}); asking for findings"
        )
        text = await self._complete_text(
            history, what=f"{phase.key} findings", tools=specs
        )
        return PhaseTrace(phase.key, text, executions)

    async def run(self, topic: str) -> AgentRun:
        """Runs the routine phase chain and returns its final report."""
        run_started = time.monotonic()
        if self._rollout_budget_s is not None:
            self._deadline = run_started + self._rollout_budget_s
        # The wiring goes on its own line, once. What the reader needs on the
        # first line is the question this rollout was asked -- that is the thing
        # the CLI pane next to it is also about.
        self._log(f"TASK  {topic[:68]}")
        for line in self._context.describe():
            self._log(f"WIRED {line}")
        phases: list[PhaseTrace] = []
        for phase_key in RESEARCH_PHASE_TOOLS:
            prior_texts = [phase.text for phase in phases if phase.text]
            phases.append(await self._research_phase(phase_key, topic, prior_texts))
        prior = [phase.text for phase in phases if phase.text]
        # The prose above is what the phases chose to write down. This is what
        # they actually retrieved, and the verdict is defined on it.
        records = _records_block(_governed_records(phases))
        if records:
            prior.append(records)
            self._log(f"   carried {len(records)} chars of governed records")

        self._phase_no += 1
        self._log(f"-- PHASE {self._phase_no}/5 decision_analysis  no retrieval")
        analysis_phase = _phase("decision_analysis")
        analysis = await self._complete_text(
            [{"role": "user", "content": _analysis_prompt(analysis_phase, topic, prior)}],
            what="decision analysis",
        )
        phases.append(PhaseTrace(analysis_phase.key, analysis))
        prior.append(analysis)

        report = await self._complete_text(
            [{"role": "user", "content": routine_user_prompt(topic, prior)}],
            what="final report",
        )
        phases.append(PhaseTrace("final_report", report))
        attempted_egress = [t["name"] for t in self.tool_timings if t["egress"]]
        model_calls = len(self.call_timings)
        elapsed = time.monotonic() - run_started
        # Count what reached the sandbox, not what the model asked for. A call
        # this phase did not offer is answered locally and never dispatched, so
        # counting every call the model made reported one more tool call than
        # the grader saw -- the environment's `n_tool_calls` counts its own tool
        # routes. Two panes of the same rollout disagreeing on the headline
        # number is worse than not printing it.
        dispatched = [t["name"] for t in self.tool_timings]
        blocked = (
            f"  {len(self.blocked_calls)} blocked"
            f"[{', '.join(sorted(set(self.blocked_calls)))}]"
            if self.blocked_calls
            else ""
        )
        self._log(
            f"ROLLOUT done  {elapsed:.1f}s  {model_calls} model  "
            f"{len(dispatched)} tool{blocked}"
        )
        # The lines in the stream that are about the business question rather
        # than the plumbing, so they go last, where a reader looks first.
        for line in _decision(report):
            self._log(f"DECISION  {line}")
        if attempted_egress:
            logger.warning(
                "[%s] ROLLOUT egress  this rollout reached for %s -- "
                "tool_discipline will penalise it",
                self._tag,
                ", ".join(sorted(set(attempted_egress))),
            )
        # Where the time went is the only thing worth knowing when a rollout
        # runs long, so it is reported on every rollout rather than only on the
        # ones that overran.
        model_s = sum(t["seconds"] for t in self.call_timings)
        tool_s = sum(t["seconds"] for t in self.tool_timings)
        slowest = max(self.call_timings, key=lambda t: t["seconds"], default=None)
        self._log(
            f"ROLLOUT budget  model={model_s:.1f}s  tools={tool_s:.1f}s"
            f"  other={max(elapsed - model_s - tool_s, 0.0):.1f}s"
            + (f"  max call={slowest['seconds']:.1f}s" if slowest else "")
        )
        return AgentRun(report=report, phases=phases)
