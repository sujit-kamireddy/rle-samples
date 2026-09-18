"""OpenEnv Action/Observation schema for the ``code_repair`` environment.

One episode = one real SWE-bench-Lite instance (see ``fixtures/instance.json``:
``psf/requests`` at its pinned base commit, sourced from
``princeton-nlp/SWE-bench_Lite``). ``reset()`` hands back the real GitHub
issue text as the problem statement; the policy submits its whole proposed
fix as a unified diff in a single ``step()`` call, which applies the patch
to a fresh checkout and grades it immediately by running the instance's
real regression test.

This is the Gym/OpenEnv sibling of ``examples/harness/{byoh,hosted-agent}``'s
"code repair" example, reusing the same real instance/fixtures for narrative
parity -- but unlike those Harness examples (where an external agent is
invoked over HTTP and edits the checkout turn-by-turn via mocked
``workspace.apply_patch``/``github.create_pull_request`` tool calls), there
is no agent here and no tool mocks: an external RL trainer/policy calls
``reset()``/``step()`` on this environment directly over OpenEnv's own
HTTP/WS contract, and the whole candidate fix is the ``step()`` action
itself.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from openenv.core.env_server.types import Action, Observation


class CodeRepairAction(Action):
    """The policy's whole proposed fix for the current instance."""

    patch: str = Field(
        description=(
            "Unified diff (as produced by e.g. `git diff`) to apply to the "
            "instance's checkout. Graded in full on this single step() call -- "
            "there is no follow-up turn to revise it."
        )
    )

    episode_id: str | None = Field(
        default=None,
        description=(
            "Optional echo of the observation's episode_id, naming which "
            "rollout's on-disk workspace to grade against. Only needed over "
            "OpenEnv's plain HTTP /reset+/step routes, which build a fresh "
            "environment instance per request (see "
            "examples/gym/openenv/code_rl/server/app.py's docstring) -- so "
            "step() cannot rely on the instance that served reset() still "
            "being around. Not needed over /ws, where one instance serves "
            "the whole connection."
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
    episode_id: str | None = Field(
        default=None,
        description="Echoed back by reset() -- see CodeRepairAction.episode_id.",
    )
