"""Runtime contract tests; skipped when the private RLE SDK is unavailable."""

from __future__ import annotations

import pytest

pytest.importorskip("openenv")
pytest.importorskip("azure.ai.projects.rle.environments")

from azure.ai.projects.rle.environments import GradeAction

from examples.gym.openenv.mcp_rl.server.environment import (
    ArithmeticRLEnvironment,
)


@pytest.fixture()
def env():
    environment = ArithmeticRLEnvironment()
    yield environment
    environment.close()


def test_reward_requires_tool_execution(env):
    env.reset(left=2, right=3)

    result = env.grade(GradeAction(answer="5"))

    assert result.done is True
    assert result.reward == 0
    assert result.tool_called is False


def test_correct_answer_after_typed_tool_call_earns_one(env):
    env.reset(left=2, right=3)

    tool_result = env.add(left=2, right=3)
    result = env.grade(GradeAction(answer="5"))

    assert tool_result["result"] == 5
    assert tool_result["instance_id"] == env.state.instance_id
    assert env.state.tool_call_count == 1
    assert result.reward == 1
    assert result.done is True


def test_tool_rejects_operands_other_than_the_active_task(env):
    env.reset(left=2, right=3)

    with pytest.raises(ValueError, match="active task operands"):
        env.add(1, 4)

    result = env.grade(GradeAction(answer="5"))
    assert result.reward == 0
    assert env.state.tool_called is False


def test_tool_state_is_isolated_to_the_instance(env):
    other = ArithmeticRLEnvironment()
    try:
        env.reset(left=2, right=3)
        other.reset(left=2, right=3)
        env.add(2, 3)

        assert env.state.tool_called is True
        assert other.state.tool_called is False
        assert env.state.instance_id != other.state.instance_id
    finally:
        other.close()


def test_reset_rejects_unknown_or_nondeterministic_selectors(env):
    with pytest.raises(ValueError, match="Unsupported reset selector"):
        env.reset(unexpected=True)
    with pytest.raises(ValueError, match="deterministic selector"):
        env.reset()
    with pytest.raises(ValueError, match="deterministic selector"):
        env.reset(left=3, right=2)
