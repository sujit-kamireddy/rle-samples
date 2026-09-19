"""OpenEnv Action/Observation schema for the ``math_rl`` environment.

One episode = one Hendrycks MATH problem. It's single-turn: ``reset()``
hands back the problem, the policy submits one free-text answer via
``step()``, and the episode ends immediately (``done=True``) with a
reward from grading that answer against the reference solution, shaped by
``FORMAT_COEF`` the same way the in-process recipe path is (1.0 if correct
and formatted, 0.0 if formatted but wrong, -FORMAT_COEF if unformatted).
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import Field

from openenv.core.env_server.types import Action, Observation


class MathAction(Action):
    """The policy's submitted answer for the current problem.

    ``type`` defaults to ``"submit_answer"`` (a normal grading step). A
    ``type="list_tools"`` action is accepted too -- see
    ``examples/gym/openenv/math_rl/server/math_rl_environment.py``'s ``step()`` -- as a
    stand-in for OpenEnv's own MCP-style ``ListToolsAction`` (rfcs/003-mcp-support.md),
    which RLE's rollout pipeline probes for on every Gym/OpenEnv target before the first
    real step, tools or not. OpenEnv only auto-recognizes that probe when ``action_cls``
    is exactly ``Action`` or an MCP action type (``serialization.py``'s
    ``_deserialize_mcp_action``), not for a custom subclass like this one, so without this
    field the probe fails Pydantic validation (extra field + missing ``answer_text``)
    before ``step()`` is ever called. This env has no tools, so it just answers with an
    empty tool list instead of erroring.
    """

    type: Literal["submit_answer", "list_tools"] = Field(
        default="submit_answer",
        description="'submit_answer' (default) grades answer_text; 'list_tools' is a no-tools discovery probe.",
    )

    answer_text: Optional[str] = Field(
        default=None,
        description="Free-text answer, ideally containing \\boxed{...} (see MathObservation.system_prompt).",
    )

    problem_id: str = Field(
        default="",
        description=(
            "Optional echo of the observation's problem_id, naming the row to grade "
            "against. The server remembers the row reset() picked, so this is only "
            "needed to grade some other row -- or to keep working behind a transport "
            "that does not preserve state between calls (see "
            "examples/gym/openenv/math_rl/server/math_rl_environment.py)."
        ),
    )


class MathObservation(Observation):
    """What the policy sees. ``messages`` is empty after ``step()`` since
    the episode always ends there (single-turn)."""

    messages: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Chat messages to append to the conversation (system+user prompt on reset, empty on step).",
    )
    problem_id: Optional[str] = None
