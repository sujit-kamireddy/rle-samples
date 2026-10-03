"""The competitive-intelligence RLE as an OpenEnv ``RLEnvironment``.

This is the same environment as [`../rle`](../../rle), expressed against the
authoring interface that ships in ``azure-ai-projects[rle]``. The legacy
container hand-rolls four HTTP routes; this one subclasses
``RLEnvironment`` and lets the SDK supply the protocol:

    ``reset``          the caller's task, verbatim          (was ``POST /reset``)
    MCP ``tools/*``    the agent's tool calls               (was ``POST /tools/*``)
    ``grade``          the rubric, over the final answer    (was ``POST /grade``)

``/health`` and the request plumbing come from ``create_app``, so they are no
longer written here.

## What is reused rather than reimplemented

Everything that decides a reward. ``rl/tasks.py``, ``rl/world.py``,
``rl/simulated_tools.py`` and ``rl/grading.py`` are imported unmodified -- the
same task generator, simulated world, tools and rubric that both the Loom RL
runs and the legacy harness use. The three response-shaping helpers are
imported from the legacy module itself rather than copied, because each one
encodes a production failure that was expensive to find (see their docstrings)
and a second copy would be a second thing to keep correct.

That import also gives this module the legacy ``WORLD`` and ``TOOL_SURFACE``:
one shared world per process, and a surface the two environments cannot
silently disagree on.

## One episode per session, not one per container

The legacy container keeps a single module-level ``Rollout``, because a Harness
sandbox serves one rollout at a time. ``create_app`` instead builds one
environment per OpenEnv session, so the per-rollout state that used to be
global is just instance state here and several sessions can run concurrently in
one container.

## Where the taskset lives

Not in this image, for the same reason as the legacy one: every rollout carries
its own task and it arrives whole in the ``reset`` call. Baking ``data/rl/`` in
would also ship the expected decision and the verified snapshot, which is answer
key, into an image the agent's sandbox can reach.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
from typing import Any, Callable, Optional

from azure.ai.projects.rle.environments import GradeAction, RLEnvironment
from loom_cookbook.tool_use import ToolInput
from openenv.core.env_server.types import Observation, State

from rle_openenv.rl.grading import grade_episode
from rle_openenv.rl.simulated_tools import ToolSession, session_tools
from rle_openenv.rl.tasks import Task
from rle_openenv.rl.world import build_world
from rle_openenv.server.rollout_text import (
    final_text as _final_text,
    grading_failure_detail as _grading_failure_detail,
)

# The simulated world is deterministic and read-only once built, and building
# it walks the whole company/team/product graph. Rollouts share one instance
# rather than each paying for a rebuild; `ToolSession` keeps all per-rollout
# state.
WORLD = build_world()

#: The surface both halves of a rollout must agree on. The harness container
#: and the agent package read the *same* variable name with the *same*
#: default, which is the only reason they cannot silently disagree -- they are
#: deployed separately and nothing plumbs the value from one to the other.
#:
#: Default `production`: `routine` serves four invented semantic tools that the
#: deployed agent does not have. A policy trained on it scored 0.97 tool
#: discipline in the simulator and 0.43 on the real benchmark, because only
#: `web_search` was common to both. `routine` is kept to reproduce those runs.
TOOL_SURFACE = os.environ.get("COMPETITIVE_INTEL_TOOL_SURFACE", "production")

logger = logging.getLogger("competitive-intel-rle-openenv")



class CompetitiveIntelObservation(Observation):
    """What ``reset`` and ``grade`` hand back.

    ``Observation`` forbids extra fields, so everything the caller may read is
    declared. ``info`` carries the grading payload the legacy ``/grade`` route
    returned under the same name, unchanged, so a reward read here can be
    compared with one read there field by field.
    """

    task_id: Optional[str] = None
    variant: Optional[str] = None
    query: Optional[str] = None
    tool_surface: Optional[str] = None
    tools: list[str] = []
    is_success: bool = False
    info: dict[str, Any] = {}


class CompetitiveIntelState(State):
    """Episode state, as ``state`` reports it between calls."""

    task_id: Optional[str] = None
    variant: Optional[str] = None
    graded: bool = False


def _tool_templates() -> list[Any]:
    """The surface's tools, read off the class instead of an instance.

    MCP tools register once, when the environment is constructed; the tools
    themselves bind to a ``ToolSession`` that does not exist until ``reset``.
    ``session_tools`` only reads attributes, and ``FunctionTool`` is a
    descriptor that returns itself when read from the class, so reading the
    surface from ``ToolSession`` yields unbound tools -- names, descriptions and
    argument models, with no session attached. That is exactly what registration
    needs; the session is looked up per call, in the wrapper below.
    """
    return session_tools(ToolSession, TOOL_SURFACE)


def _make_tool_wrapper(template: Any, resolve: Callable[[str], Any]) -> Callable[..., Any]:
    """An MCP-callable wrapper around one simulated tool.

    FastMCP builds a tool's JSON schema from the wrapped function's signature,
    so the signature is rebuilt here from the tool's own pydantic argument
    model. The schema an agent sees over MCP is therefore generated from the
    same model ``FunctionTool.run`` validates against, and cannot drift from it.

    Dispatch goes through ``FunctionTool.run``, the entry point Loom's episode
    loop uses, so a call recorded here is the call the rubric scores. ``run``
    never raises: invalid arguments and tool faults both come back as an error
    result. That is deliberate -- a malformed call is the agent's mistake, and
    the rubric needs to see the wasted call rather than have the attempt fault.
    """
    parameters = []
    annotations: dict[str, Any] = {}
    for name, field in template._params_model.model_fields.items():
        default = inspect.Parameter.empty if field.is_required() else field.default
        parameters.append(
            inspect.Parameter(
                name,
                inspect.Parameter.KEYWORD_ONLY,
                default=default,
                annotation=field.annotation,
            )
        )
        annotations[name] = field.annotation
    annotations["return"] = str

    async def call(**arguments: Any) -> str:
        tool = resolve(template.name)
        result = await tool.run(ToolInput(arguments=arguments, call_id=""))
        messages = result.messages or []
        # The tool's single message carries a JSON string, and that string is
        # what the model saw as the tool's output during training.
        return messages[0].get("content", "") if messages else ""

    call.__name__ = template.name
    call.__doc__ = template.description
    call.__signature__ = inspect.Signature(parameters, return_annotation=str)
    call.__annotations__ = annotations
    return call


class CompetitiveIntelEnvironment(RLEnvironment):
    """One competitive-intelligence rollout, driven by the deployed agent."""

    def __init__(self) -> None:
        self._task: Optional[Task] = None
        self._session: Optional[ToolSession] = None
        self._tools: dict[str, Any] = {}
        # Tools register against the MCP server the base class creates, so they
        # cannot be registered before it exists.
        super().__init__()
        self._set_state(CompetitiveIntelState())
        self._tool_names = [template.name for template in _tool_templates()]
        for template in _tool_templates():
            self.tool()(_make_tool_wrapper(template, self._require_tool))

    def _require_tool(self, name: str) -> Any:
        """The live tool for ``name``, or a loud failure if there is no episode.

        A tool call before ``reset`` is a protocol error, not an agent mistake,
        so it raises rather than coming back as a tool result the rubric would
        score as a wasted call.
        """
        if self._session is None:
            raise RuntimeError(f"No active episode; call reset before {name!r}.")
        tool = self._tools.get(name)
        if tool is None:
            raise RuntimeError(
                f"Tool {name!r} is not on the {TOOL_SURFACE!r} surface. "
                f"Available: {sorted(self._tools)}."
            )
        return tool

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **task: Any,
    ) -> CompetitiveIntelObservation:
        """Sets one rollout up from the caller's task.

        The keyword arguments are the task exactly as the calling job supplied
        it -- one row of ``data/rl/*.jsonl`` -- because ``ResetRequest`` allows
        extra fields and passes them through to any signature that accepts
        ``**kwargs``. The agent gets the task's context by calling the retrieval
        tools this environment serves, not from this response.

        ``seed`` is accepted and unused: the world is built deterministically
        once per process and every other input to a rollout arrives in the task.
        """
        try:
            parsed = Task.from_json(task)
        except (KeyError, TypeError) as error:
            # Grading a task this environment failed to understand would report
            # a reward for work the caller never requested, so fail loudly.
            raise ValueError(f"Task row could not be parsed: {error}") from error

        self._task = parsed
        self._session = ToolSession(
            task=parsed,
            world=WORLD,
            work_iq_mode="mock",
            tool_surface=TOOL_SURFACE,
        )
        self._tools = {tool.name: tool for tool in session_tools(self._session, TOOL_SURFACE)}
        self._set_state(
            CompetitiveIntelState(
                episode_id=episode_id,
                task_id=parsed.task_id,
                variant=parsed.variant,
            )
        )
        logger.info(
            "reset episode=%s task=%s variant=%s", episode_id, parsed.task_id, parsed.variant
        )
        return CompetitiveIntelObservation(
            done=False,
            reward=0.0,
            task_id=parsed.task_id,
            variant=parsed.variant,
            query=parsed.query,
            tool_surface=TOOL_SURFACE,
            tools=sorted(self._tools),
        )

    def grade(
        self,
        action: GradeAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> CompetitiveIntelObservation:
        """Scores the finished rollout with the training rubric, unchanged.

        The reward comes from ``rl.grading.grade_episode``, which reads both the
        agent's final text and the tool calls recorded on the session -- so tool
        discipline, evidence fidelity and action restraint are scored from what
        the agent *did*, not only from what it said.

        ``grade_episode`` is a coroutine and this method is not, which is what
        ``create_app`` expects: it runs a synchronous ``step`` in a thread pool,
        so there is no running loop here to conflict with.
        """
        if self._task is None or self._session is None:
            raise RuntimeError("No active episode; call reset before grade.")

        try:
            agent_response = _final_text(action.answer or "")
            result = asyncio.run(grade_episode(self._task, self._session, agent_response))
        except Exception as error:
            logger.exception(
                "grade failed task=%s variant=%s", self._task.task_id, self._task.variant
            )
            # Named compactly on purpose: RLE caps a rollout's error detail at
            # 200 characters and cuts the tail, so the cause is written
            # shortest-first. The full traceback went to the log above.
            raise RuntimeError(_grading_failure_detail(error)) from error

        raw_reward = result.reward
        # Managed RLE requires [0, 1]. This is a backstop, not the mechanism:
        # `grade_episode` already maps the format penalty's [-FORMAT_COEF, 1]
        # onto [0, 1] by rescaling, which keeps the ordering between a badly
        # formatted report and an empty one. Clamping alone would flatten every
        # unparsed rollout scoring under FORMAT_COEF to exactly 0.0, and a group
        # whose members all score 0.0 yields no advantage and no gradient --
        # precisely the rollouts a cold policy produces most. Do not drop the
        # rescale on the grounds that this line catches it.
        reward = min(1.0, max(0.0, raw_reward))

        state = self.state
        state.graded = True
        self._set_state(state)

        return CompetitiveIntelObservation(
            done=True,
            reward=reward,
            task_id=self._task.task_id,
            variant=self._task.variant,
            is_success=bool(result.metrics.get("verdict_exact", 0.0) >= 1.0),
            info={
                "task_id": self._task.task_id,
                "variant": self._task.variant,
                "raw_reward": raw_reward,
                "metrics": result.metrics,
                "parsed": result.decision.parsed,
                "verdict": result.decision.material,
                "expected_verdict": self._task.material,
                "n_tool_calls": len(self._session.calls),
                "tools_called": [call.name for call in self._session.calls],
                "agent_response": agent_response,
            },
        )

    def close(self) -> None:
        """Drops the episode. The world is process-wide and outlives it."""
        self._task = None
        self._session = None
        self._tools = {}
