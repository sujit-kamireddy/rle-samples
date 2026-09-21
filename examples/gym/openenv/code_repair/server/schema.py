"""OpenEnv Action/Observation schema for the ``code_repair`` environment.

One episode = one real SWE-bench-Verified instance, drawn from the 40
``django/django`` (version ``3.2``) instances baked into the image (see
``env_data/``). ``reset(seed, split)`` picks a row the same way
``math_rl``/``code_rl`` do -- indexing ``seed % len(rows)`` into a fixed
shuffled permutation -- and hands back the real GitHub issue text as the
problem statement; the policy submits its whole proposed fix as a unified
diff in a single ``step()`` call, which applies the patch to a fresh
``git worktree`` checkout and grades it immediately by running the
instance's real regression test via Django's own test runner.

This is the Gym/OpenEnv sibling of ``examples/harness/{byoh,hosted-agent}``'s
"code repair" example -- but unlike those Harness examples (where an
external agent is invoked over HTTP and edits the checkout turn-by-turn via
mocked ``workspace.apply_patch``/``github.create_pull_request`` tool
calls), there is no agent here and no tool mocks: an external RL
trainer/policy calls ``reset()``/``step()`` on this environment directly
over OpenEnv's own HTTP/WS contract, and the whole candidate fix is the
``step()`` action itself.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from openenv.core.env_server.types import Action, Observation


class CodeRepairAction(Action):
    """The policy's whole proposed fix for the current instance.

    This environment has exactly one action, so this class is a plain
    ``Action`` subclass rather than a discriminated union (compare ``code_rl``,
    which adds a ``check_solution`` tool alongside its final answer). Foundry
    RLE reads the action vocabulary from this class's JSON Schema, served at
    ``GET /schema``, and ``rle.toml``'s ``model_response_field = "patch"``
    names the property below that receives the model's completion text.

    No ``problem_id``/``episode_id`` fields: Foundry RLE's rollout pipeline only
    drives Gym/OpenEnv targets over ``/ws`` (one environment instance for the
    life of the connection), so ``step()`` can always resolve the episode's
    row and on-disk workspace from what ``reset()`` stored on ``self`` -- there
    is no stateless-HTTP case here to work around.
    """

    patch: str = Field(
        default="",
        description=(
            "Unified diff (as produced by e.g. `git diff`) to apply to the "
            "instance's checkout. Graded in full on this single step() call -- "
            "there is no follow-up turn to revise it."
        ),
    )


class CodeRepairObservation(Observation):
    """What the policy sees. ``messages`` is the issue's problem statement on
    ``reset()``; empty on ``step()``, since the episode always ends there."""

    messages: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Chat messages to append to the conversation: the real GitHub "
            "issue text as a single user message on reset(), empty on step() "
            "since this environment is single-turn (done=True on every step())."
        ),
    )
    instance_id: str | None = None
    problem_id: str | None = Field(
        default=None,
        description="The row index reset() picked, for logging/debugging.",
    )
    episode_id: str | None = Field(
        default=None,
        description="The on-disk rollout workspace's id, for logging/debugging.",
    )
