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
    """The policy's whole proposed fix for the current instance."""

    patch: str = Field(
        default="",
        description=(
            "Unified diff (as produced by e.g. `git diff`) to apply to the "
            "instance's checkout. Graded in full on this single step() call -- "
            "there is no follow-up turn to revise it."
        ),
    )

    problem_id: str = Field(
        default="",
        description=(
            "Optional echo of the observation's problem_id, naming which row to "
            "grade against. The server remembers the row reset() picked, so this "
            "is only needed to grade some other row -- or to keep working behind "
            "a transport that does not preserve state between calls (see "
            "examples/gym/openenv/code_repair/server/code_repair_environment.py)."
        ),
    )

    episode_id: str | None = Field(
        default=None,
        description=(
            "Optional echo of the observation's episode_id, naming which "
            "rollout's on-disk workspace to grade against. Only needed over "
            "OpenEnv's plain HTTP /reset+/step routes, which build a fresh "
            "environment instance per request -- so step() cannot rely on the "
            "instance that served reset() still being around. Not needed over "
            "/ws, where one instance serves the whole connection."
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
        description="Echoed back by reset() -- see CodeRepairAction.problem_id.",
    )
    episode_id: str | None = Field(
        default=None,
        description="Echoed back by reset() -- see CodeRepairAction.episode_id.",
    )
