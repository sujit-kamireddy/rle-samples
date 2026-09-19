"""OpenEnv Action/Observation schema for the ``code_rl`` environment.

One episode = one LiveCodeBench-style competitive programming problem
(DeepCoder-Preview). ``reset()`` hands back the problem (+ starter code),
and the policy may call the ``check_solution`` tool up to ``max_turns``
times (see ``CODE_RL_MAX_TURNS``) before submitting a final answer via
``step()``. Only the final step -- either because the response has no
tool call, or ``max_turns`` was reached -- grades the submission and ends
the episode (``done=True``) with a reward from running the submitted code
against the problem's hidden tests, shaped by ``FORMAT_COEF`` the same way
the in-process recipe path is (1.0 if correct and fenced, 0.0 if fenced
but failing, -FORMAT_COEF if unfenced). This mirrors
``loom_cookbook.recipes.code_rl.deepcoder_tool.DeepcoderReward``, which
grades only the last assistant message of the completed episode.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import Field

from openenv.core.env_server.types import Action, Observation


class CodeAction(Action):
    """The policy's submitted solution for the current problem.

    ``type`` defaults to ``"submit_answer"`` (a normal grading/tool-call step). A
    ``type="list_tools"`` action is accepted too -- see
    ``examples/gym/openenv/code_rl/server/code_rl_environment.py``'s ``step()`` -- as a
    stand-in for OpenEnv's own MCP-style ``ListToolsAction`` (rfcs/003-mcp-support.md),
    which RLE's rollout pipeline probes for on every Gym/OpenEnv target before the first
    real step. OpenEnv only auto-recognizes that probe when ``action_cls`` is exactly
    ``Action`` or an MCP action type (``serialization.py``'s ``_deserialize_mcp_action``),
    not for a custom subclass like this one, so without this field the probe fails
    Pydantic validation (extra field + missing ``code_text``) before ``step()`` is ever
    called. This env already hands its one real tool (``check_solution``) back via
    ``reset()``'s ``metadata.tool_specs`` (see ``CHECK_SOLUTION_TOOL_SPEC``), so the probe
    just echoes that same spec back instead of an empty list.
    """

    type: Literal["submit_answer", "list_tools"] = Field(
        default="submit_answer",
        description="'submit_answer' (default) grades code_text/tool_calls; 'list_tools' is a discovery probe.",
    )

    code_text: Optional[str] = Field(
        default=None,
        description="Free-text response, ideally containing a ```python ...``` fenced solution.",
    )

    tool_calls: Optional[list[dict[str, Any]]] = Field(
        default=None,
        description=(
            "The response's parsed tool calls, if any -- each shaped like "
            "``{'id': str, 'name': str, 'arguments': <JSON string>}``. "
            "Populated by the renderer's tool-call parsing (see "
            "loom_cookbook/renderers/base.py's ``ToolCall``) whenever the "
            "conversation prefix declared the ``check_solution`` tool. "
            "A response with a non-empty ``tool_calls`` keeps the episode "
            "going (see module docstring) instead of grading ``code_text``."
        ),
    )

    problem_id: str = Field(
        default="",
        description=(
            "Optional echo of the observation's problem_id, naming the row to grade "
            "against. The server remembers the row reset() picked, so this is only "
            "needed to grade some other row -- or to keep working behind a transport "
            "that does not preserve state between calls (see "
            "examples/gym/openenv/code_rl/server/code_rl_environment.py)."
        ),
    )


class CodeObservation(Observation):
    """What the policy sees. ``messages`` is the system+user prompt on
    ``reset()``; on ``step()`` it holds the ``check_solution`` tool result
    message(s) if the episode is continuing (``done=False``), or is empty
    once the episode has ended."""

    messages: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Chat messages to append to the conversation: system+user prompt on "
            "reset(), tool-result message(s) on a step() that keeps the episode "
            "going, empty once done=True."
        ),
    )
    starter_code: Optional[str] = None
    problem_id: Optional[str] = None
