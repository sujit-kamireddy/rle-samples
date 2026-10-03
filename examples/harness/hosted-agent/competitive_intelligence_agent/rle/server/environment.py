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

That import also gives this module the legacy ``WORLD``: one shared world per
process, rather than each rollout paying to rebuild it.

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
import logging
from typing import Any, Optional

from azure.ai.projects.rle.environments import GradeAction, RLEnvironment
from fastmcp.tools.function_tool import FunctionTool
from loom_cookbook.tool_use import ToolInput

from rle.rl.grading import grade_episode
from rle.rl.simulated_tools import ToolSession, session_tools
from rle.rl.tasks import Task
from rle.rl.world import build_world
from rle.server.models import CompetitiveIntelObservation, CompetitiveIntelState
from rle.server.rollout_text import (
    final_text as _final_text,
    grading_failure_detail as _grading_failure_detail,
)

# The simulated world is deterministic and read-only once built, and building
# it walks the whole company/team/product graph. Rollouts share one instance
# rather than each paying for a rebuild; `ToolSession` keeps all per-rollout
# state.
WORLD = build_world()

#: This harness only ever trains and evaluates the production surface -- the
#: one the deployed agent actually has. `rl/simulated_tools.py` still defines
#: `routine`/`full` to reproduce a historical finding (a policy trained on
#: `routine` scored 0.97 tool discipline in simulation and 0.43 on the real
#: benchmark, because only `web_search` was common to both), but this
#: environment has no RLE use for training against either, so there is no
#: env var or branch here choosing between them -- just this literal.
TOOL_SURFACE = "production"

logger = logging.getLogger("competitive-intel-rle-openenv")


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
        self._tool_names = self._register_production_tools()

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

    async def _call_tool(self, name: str, **arguments: Any) -> str:
        """Runs the live session's ``name`` tool and returns its text content.

        Every explicit tool method below forwards here, so dispatch, argument
        cleanup and error handling stay in one place. Arguments the caller
        omitted arrive as ``None`` and are dropped, so Loom sees a genuinely
        absent argument and applies its own default or its own "required"
        error. ``run`` never raises: a malformed call comes back as a
        tool-error result, which is what lets the rubric score a wasted call
        instead of faulting the whole attempt. That contract is also why
        every parameter on the methods below is typed ``Any`` with a ``None``
        default -- a real, strict signature would let FastMCP reject a bad
        call before ``run`` ever saw it.
        """
        tool = self._require_tool(name)
        supplied = {key: value for key, value in arguments.items() if value is not None}
        result = await tool.run(ToolInput(arguments=supplied, call_id=""))
        messages = result.messages or []
        return messages[0].get("content", "") if messages else ""

    def _register_production_tools(self) -> list[str]:
        """Explicitly registers the nine tools the deployed agent reaches.

        Each is a named method on this class rather than a generically built
        wrapper. The schema each one publishes still comes from the tool's
        own pydantic model (``_publish_schema``), so what the agent reads
        cannot drift from what ``_call_tool`` validates against.
        """
        methods = (
            self.web_search,
            self.competitive_fabric___DiscoverArtifacts,
            self.competitive_fabric___GetSemanticModelSchema,
            self.competitive_fabric___ExecuteQuery,
            self.competitive_fabric___ValueSearch,
            self.competitive_fabric___GetReportMetadata,
            self.competitive_fabric___ResolveReportIdFromUrl,
            self.send_email,
            self.update_tracker_record,
        )
        templates_by_name = {t.name: t for t in session_tools(ToolSession, TOOL_SURFACE)}
        for method in methods:
            self.tool()(method)
            self._publish_schema(method, templates_by_name[method.__name__])
        return [method.__name__ for method in methods]

    def _publish_schema(self, method: Any, template: Any) -> None:
        """Replaces ``method``'s auto-derived schema with the tool's own model.

        ``method``'s signature is deliberately permissive (see ``_call_tool``),
        so FastMCP's auto-derived schema would be uselessly vague. This swaps
        in the real one without touching what validates the call: remove then
        add, because adding over a live name is a duplicate registration.
        """
        described = FunctionTool.from_function(
            method, name=template.name, description=template.description
        )
        described.parameters = template._params_model.model_json_schema()
        self.mcp_server.local_provider.remove_tool(template.name)
        self.mcp_server.local_provider.add_tool(described)

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

    # ------------------------------------------------------------------
    # The production surface's nine tools, registered by `__init__` via
    # `_register_production_tools`. Each forwards to `_call_tool`, which
    # dispatches to the live session and preserves the permissive-signature
    # contract documented there. Docstrings here are summaries, not the
    # published contract -- the schema the agent reads comes from each
    # tool's own pydantic model (see `_publish_schema`), from
    # `rl/simulated_tools.py` and `rl/production_tools.py`.
    # ------------------------------------------------------------------

    async def web_search(self, search_query: Any = None) -> str:
        """Web search for factual grounding: statistics, claims, current events."""
        return await self._call_tool("web_search", search_query=search_query)

    async def competitive_fabric___DiscoverArtifacts(  # noqa: N802
        self,
        searchQuery: Any = None,  # noqa: N803
        artifactTypes: Any = None,  # noqa: N803
        maxResults: Any = None,  # noqa: N803
    ) -> str:
        """Finds Fabric artifacts (semantic models, reports) by name."""
        return await self._call_tool(
            "competitive_fabric___DiscoverArtifacts",
            searchQuery=searchQuery,
            artifactTypes=artifactTypes,
            maxResults=maxResults,
        )

    async def competitive_fabric___GetSemanticModelSchema(  # noqa: N802
        self, artifactId: Any = None, queries: Any = None  # noqa: N803
    ) -> str:
        """Reads a semantic model's tables, columns and measures."""
        return await self._call_tool(
            "competitive_fabric___GetSemanticModelSchema",
            artifactId=artifactId,
            queries=queries,
        )

    async def competitive_fabric___ExecuteQuery(  # noqa: N802
        self,
        artifactId: Any = None,  # noqa: N803
        daxQueries: Any = None,  # noqa: N803
        maxRows: Any = None,  # noqa: N803
    ) -> str:
        """Runs a DAX query against a semantic model; governed exposure lives here."""
        return await self._call_tool(
            "competitive_fabric___ExecuteQuery",
            artifactId=artifactId,
            daxQueries=daxQueries,
            maxRows=maxRows,
        )

    async def competitive_fabric___ValueSearch(  # noqa: N802
        self,
        artifactId: Any = None,  # noqa: N803
        searchTerms: Any = None,  # noqa: N803
        scope: Any = None,
    ) -> str:
        """Resolves entity names to exact table/column/value locations."""
        return await self._call_tool(
            "competitive_fabric___ValueSearch",
            artifactId=artifactId,
            searchTerms=searchTerms,
            scope=scope,
        )

    async def competitive_fabric___GetReportMetadata(  # noqa: N802
        self, reportObjectId: Any = None, queries: Any = None  # noqa: N803
    ) -> str:
        """Reads a report's workspace, semantic model, pages and visuals."""
        return await self._call_tool(
            "competitive_fabric___GetReportMetadata",
            reportObjectId=reportObjectId,
            queries=queries,
        )

    async def competitive_fabric___ResolveReportIdFromUrl(  # noqa: N802
        self, url: Any = None
    ) -> str:
        """Resolves a Power BI report URL to its report id."""
        return await self._call_tool("competitive_fabric___ResolveReportIdFromUrl", url=url)

    async def send_email(self, to: Any = None, subject: Any = None, body: Any = None) -> str:
        """Decoy: sends an email. Always the wrong move; never legitimate."""
        return await self._call_tool("send_email", to=to, subject=subject, body=body)

    async def update_tracker_record(self, record_id: Any = None, fields: Any = None) -> str:
        """Decoy: updates the competitive tracker. Always the wrong move."""
        return await self._call_tool(
            "update_tracker_record", record_id=record_id, fields=fields
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
