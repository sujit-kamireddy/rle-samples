"""OpenEnv Action/Observation schema for the ``code_rl`` environment.

One episode = one LiveCodeBench-style competitive programming problem
(DeepCoder-Preview). ``reset()`` hands back the problem (+ starter code),
and the policy may call the ``check_solution`` tool up to ``max_turns``
times (see ``CODE_RL_MAX_TURNS``) before submitting a final answer. Only a
``submit_answer`` action grades the submission and ends the episode
(``done=True``) with a reward from running the submitted code against the
problem's hidden tests, shaped by ``FORMAT_COEF`` (1.0 if correct and
fenced, 0.0 if fenced but failing, -FORMAT_COEF if unfenced).

This is the sample to copy when an environment needs tools. Foundry RLE
reads the action vocabulary from ``CodeAction``'s JSON Schema, served at
``GET /schema``: the variant declaring ``rle.toml``'s
``model_response_field`` (``code_text``) is the one that receives the
model's completion text, and every other variant is offered to the model
as a tool named after its discriminator. So ``check_solution`` below is a
tool purely by virtue of being a second union member -- there is no
separate tool spec to write, hand back, or keep in sync.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Optional, Union

from pydantic import Field, RootModel

from openenv.core.env_server.types import Action, Observation


class CheckSolutionAction(Action):
    """Execute the proposed solution against the task's test cases.

    Use this to test your code before providing your final answer.
    """

    type: Literal["check_solution"] = "check_solution"

    code: str = Field(description="Python code implementing the solution.")


class SubmitAnswerAction(Action):
    """Submit the final solution. This ends the episode and grades the answer."""

    type: Literal["submit_answer"] = "submit_answer"

    code_text: str = Field(
        default="",
        description="Free-text response, ideally containing a ```python ...``` fenced solution.",
    )


class CodeAction(
    RootModel[
        Annotated[
            Union[CheckSolutionAction, SubmitAnswerAction],
            Field(discriminator="type"),
        ]
    ]
):
    """Everything the policy may do, as a Pydantic discriminated union.

    A ``RootModel`` union is what turns ``GET /schema``'s ``action`` into a
    ``oneOf`` with a ``discriminator``, which is how Foundry RLE tells one
    action apart from another. Reach the concrete action through ``.root``
    (see ``CodeRLEnvironment._step_impl``).

    No ``problem_id`` field on either member: Foundry RLE's rollout pipeline only
    drives Gym/OpenEnv targets over ``/ws`` (one environment instance for the life
    of the connection), so ``step()`` can always resolve the episode's row from
    what ``reset()`` stored on ``self`` -- there is no stateless-HTTP case here to
    work around.
    """


class CodeObservation(Observation):
    """What the policy sees. ``messages`` is the system+user prompt on
    ``reset()``; on a ``check_solution`` step it holds the tool result, and
    it is empty once the episode has ended."""

    messages: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Chat messages to append to the conversation: system+user prompt on "
            "reset(), the tool result on a check_solution step, empty once "
            "done=True. Foundry RLE reads only the text of these messages when "
            "answering a tool call, since it -- not this environment -- owns the "
            "tool call's id."
        ),
    )
    starter_code: Optional[str] = None
    problem_id: Optional[str] = None
