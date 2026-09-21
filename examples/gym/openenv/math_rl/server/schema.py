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
    """The policy's submitted answer for the current problem.

    This environment has exactly one action, so this class is a plain
    ``Action`` subclass rather than a discriminated union (compare
    ``code_rl``, which adds a ``check_solution`` tool alongside its final
    answer). Foundry RLE reads the action vocabulary from this class's JSON
    Schema, served at ``GET /schema``, and ``rle.toml``'s
    ``model_response_field = "answer_text"`` names the property below that
    receives the model's completion text.

    No ``problem_id`` field: Foundry RLE's rollout pipeline only drives Gym/OpenEnv
    targets over ``/ws`` (one environment instance for the life of the connection), so
    ``step()`` can always resolve the episode's row from what ``reset()`` stored on
    ``self`` -- there is no stateless-HTTP case here to work around.
    """

    answer_text: str = Field(
        default="",
        description="Free-text answer, ideally containing \\boxed{...} (see MathObservation.system_prompt).",
    )


class MathObservation(Observation):
    """What the policy sees. ``messages`` is empty after ``step()`` since
    the episode always ends there (single-turn)."""

    messages: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Chat messages to append to the conversation (system+user prompt on reset, empty on step).",
    )
    problem_id: Optional[str] = None
