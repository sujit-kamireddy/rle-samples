"""OpenEnv Action/Observation schema for the ``math_rl`` environment.

One episode = one Hendrycks MATH problem. It's single-turn: ``reset()``
hands back the problem, the policy submits one free-text answer via
``step()``, and the episode ends immediately (``done=True``) with a
reward from grading that answer against the reference solution, shaped by
``FORMAT_COEF`` the same way the in-process recipe path is (1.0 if correct
and formatted, 0.0 if formatted but wrong, -FORMAT_COEF if unformatted).
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import Field

from openenv.core.env_server.types import Action, Observation


class MathAction(Action):
    """The policy's submitted answer for the current problem."""

    answer_text: str = Field(
        description="Free-text answer, ideally containing \\boxed{...} (see MathObservation.system_prompt)."
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
