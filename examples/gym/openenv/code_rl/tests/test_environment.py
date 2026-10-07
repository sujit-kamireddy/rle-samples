"""Focused runtime tests for the SDK MCP code environment."""

from __future__ import annotations

import asyncio
import signal

import pytest

pytest.importorskip("openenv")
pytest.importorskip("azure.ai.projects.rle.environments")

from azure.ai.projects.rle.environments import GradeAction

from examples.gym.openenv.code_rl.server import code_rl_environment as environment_module
from examples.gym.openenv.code_rl.server.code_grading import check_correctness
from examples.gym.openenv.code_rl.server.code_rl_environment import CodeRLEnvironment


@pytest.fixture()
def env():
    environment = CodeRLEnvironment()
    yield environment
    environment.close()


@pytest.fixture()
def passing_grader(monkeypatch):
    calls = []

    async def fake_check(tests, generation, timeout, overall_timeout):
        calls.append((tests, generation, timeout, overall_timeout))
        return True, {"result": [True]}

    monkeypatch.setattr(environment_module, "check_correctness", fake_check)
    return calls


def test_train_and_validation_selectors_use_the_requested_split(env):
    train = env.reset(seed=0, split="train")
    validation = env.reset(seed=0, split="validation")

    assert train.split == "train"
    assert validation.split == "validation"
    assert env.state.split == "validation"
    assert env.state.row_index is not None


def test_reset_rejects_unknown_selectors_and_splits(env):
    with pytest.raises(ValueError, match="Unknown reset selector"):
        env.reset(seed=0, unexpected=True)
    with pytest.raises(ValueError, match="Unknown split"):
        env.reset(seed=0, split="test")


def test_check_solution_is_typed_and_updates_same_instance_state(env, passing_grader):
    env.reset(seed=0, split="train")

    tools = asyncio.run(env._async_list_tools())
    result = asyncio.run(env.check_solution("print('candidate')"))

    assert [tool.name for tool in tools] == ["check_solution"]
    assert tools[0].inputSchema["properties"]["code"] == {"type": "string"}
    assert tools[0].inputSchema["required"] == ["code"]
    assert result["passed"] is True
    assert result["check_solution_calls"] == 1
    assert env.state.check_solution_calls == 1
    assert env.state.last_check_passed is True
    assert passing_grader[0][1] == "print('candidate')"


def test_tool_state_is_isolated_between_environment_instances(env, passing_grader):
    other = CodeRLEnvironment()
    try:
        env.reset(seed=0)
        other.reset(seed=0)
        asyncio.run(env.check_solution("print(1)"))

        assert env.state.check_solution_calls == 1
        assert other.state.check_solution_calls == 0
    finally:
        other.close()


def test_tool_budget_returns_recoverable_result(env, passing_grader):
    env.reset(seed=0)
    asyncio.run(env.check_solution("print(1)"))
    asyncio.run(env.check_solution("print(1)"))

    exhausted = asyncio.run(env.check_solution("print(1)"))

    assert exhausted["passed"] is False
    assert "budget exhausted" in exhausted["error"]
    assert env.state.check_solution_calls == 3
    assert len(passing_grader) == 2


def test_correct_final_answer_does_not_require_prior_tool_call(env, passing_grader):
    env.reset(seed=0)

    result = asyncio.run(
        env.step_async(GradeAction(response="Explanation\n```python\nprint(42)\n```"))
    )

    assert result.done is True
    assert result.reward == 1.0
    assert result.check_solution_calls == 0
    assert env.state.graded is True
    assert env.state.active is False
    assert passing_grader[0][1] == "print(42)"


def test_unfenced_final_answer_keeps_format_penalty_without_execution(env, passing_grader):
    env.reset(seed=0)

    result = env.grade(GradeAction(response="print(42)"))

    assert result.done is True
    assert result.reward == -0.1
    assert result.metadata["format"] is False
    assert env.state.graded is True
    assert env.state.active is False
    assert passing_grader == []


def test_missing_tests_terminal_error_deactivates_episode(env):
    env.reset(seed=0)
    env._current_row = {"messages": []}

    result = env.grade(GradeAction(response="```python\nprint(42)\n```"))

    assert result.done is True
    assert result.reward == 0.0
    assert result.metadata["error"] == "row has no 'tests' field to grade against"
    assert env.state.graded is True
    assert env.state.active is False


def test_grader_terminal_error_deactivates_episode(env, monkeypatch):
    async def failing_check(tests, generation, timeout, overall_timeout):
        raise RuntimeError("grader unavailable")

    monkeypatch.setattr(environment_module, "check_correctness", failing_check)
    env.reset(seed=0)

    result = env.grade(GradeAction(response="```python\nprint(42)\n```"))

    assert result.done is True
    assert result.reward == 0.0
    assert result.metadata["error"] == "grading failed: grader unavailable"
    assert env.state.graded is True
    assert env.state.active is False


def test_last_fenced_block_is_submitted(env, passing_grader):
    env.reset(seed=0)

    env.grade(
        GradeAction(
            response="```python\nprint('draft')\n```\n```python\nprint('final')\n```"
        )
    )

    assert passing_grader[0][1] == "print('final')"


@pytest.mark.skipif(not hasattr(signal, "SIGALRM"), reason="grader targets the Linux container")
def test_hidden_test_grader_executes_in_disposable_child_process():
    passed, details = asyncio.run(
        check_correctness(
            [
                {
                    "input": "2\n",
                    "output": "4\n",
                    "testtype": "stdin_stdout",
                    "metadata": {},
                }
            ],
            "value = int(input())\nprint(value * 2)",
            timeout=2,
            overall_timeout=5,
        )
    )

    assert passed is True
    assert details["result"] == [True]
