"""Focused runtime tests for the SDK MCP math environment."""

from __future__ import annotations

import pytest

pytest.importorskip("openenv")
pytest.importorskip("azure.ai.projects.rle.environments")

from azure.ai.projects.rle.environments import GradeAction

from examples.gym.openenv.math_rl.server.grading import extract_boxed
from examples.gym.openenv.math_rl.server.math_rl_environment import (
    FORMAT_COEF,
    MathRLEnvironment,
)


@pytest.fixture()
def env():
    environment = MathRLEnvironment()
    yield environment
    environment.close()


def _reference_submission(env: MathRLEnvironment) -> str:
    return f"Work omitted. Final answer: \\boxed{{{extract_boxed(env._current_row['solution'])}}}"


def test_seed_and_split_selection_remain_deterministic(env):
    first = env.reset(seed=7, split="train")
    first_index = env.state.row_index
    second = env.reset(seed=7, split="train")

    assert first.problem_id == second.problem_id == str(first_index)
    assert env.state.split == "train"
    assert env.state.instance_id

    validation = env.reset(seed=7, split="validation")
    assert validation.problem_id == str(env.state.row_index)
    assert env.state.split == "validation"


def test_boxed_correct_answer_does_not_require_helper(env):
    env.reset(seed=0, split="train")

    result = env.grade(GradeAction(answer=_reference_submission(env)))

    assert result.done is True
    assert result.reward == 1.0
    assert result.metadata["correct"] is True
    assert result.helper_called is False
    assert env.state.helper_call_count == 0


def test_unboxed_answer_preserves_format_penalty(env):
    env.reset(seed=0, split="train")
    reference = extract_boxed(env._current_row["solution"])

    result = env.grade(GradeAction(answer=reference))

    assert result.reward == -FORMAT_COEF
    assert result.metadata == {
        "correct": False,
        "format": False,
        "helper_called": False,
        "helper_call_count": 0,
    }


def test_typed_equivalence_helper_records_same_instance_state(env):
    env.reset(seed=0, split="train")
    instance_id = env.state.instance_id

    check = env.check_equivalence(
        candidate_expression="1/2",
        comparison_expression="\\frac{1}{2}",
    )

    assert check["equivalent"] is True
    assert check["instance_id"] == instance_id
    assert check["helper_call_count"] == 1
    assert env.state.helper_called is True
    assert env.state.last_candidate_expression == "1/2"
    assert env.state.last_comparison_expression == "\\frac{1}{2}"
    assert env.state.last_equivalent is True

    result = env.grade(GradeAction(answer=_reference_submission(env)))
    assert result.reward == 1.0
    assert result.helper_called is True
    assert result.helper_call_count == 1


def test_helper_state_is_isolated_between_environment_instances(env):
    other = MathRLEnvironment()
    try:
        env.reset(seed=0, split="train")
        other.reset(seed=0, split="train")
        env.check_equivalence("1 + 1", "2")

        assert env.state.helper_call_count == 1
        assert other.state.helper_call_count == 0
        assert env.state.instance_id != other.state.instance_id
    finally:
        other.close()


def test_validation_and_lifecycle_errors_are_explicit(env):
    with pytest.raises(ValueError, match="Unknown split"):
        env.reset(seed=0, split="test")
    with pytest.raises(RuntimeError, match="active ungraded episode"):
        env.check_equivalence("1", "1")

    env.reset(seed=0, split="train")
    with pytest.raises(ValueError, match="non-empty"):
        env.check_equivalence("", "1")
    env.grade(GradeAction(answer="not boxed"))
    with pytest.raises(RuntimeError, match="active ungraded episode"):
        env.grade(GradeAction(answer="\\boxed{0}"))
