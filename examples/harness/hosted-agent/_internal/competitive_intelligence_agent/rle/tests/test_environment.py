"""The OpenEnv environment's ``reset``/``step``/``grade`` contract.

These tests drive ``CompetitiveIntelEnvironment`` through its public protocol:
a task handed to ``reset`` produces the declared production tool surface, tool
calls land in the session the rubric reads, and ``grade`` turns a final answer
into a reward RLE will accept.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from azure.ai.projects.rle.environments import GradeAction
from openenv.core.env_server.mcp_environment import CallToolAction, ListToolsAction

from rle.server.environment import TOOL_SURFACE, CompetitiveIntelEnvironment


# This file was relocated to `_internal/` (see `conftest.py`), so its own
# directory is no longer inside the sample that ships `job_data/`. Reach the
# real sample root the same way `conftest.py` does, rather than deriving it
# from this file's own (now relocated) position.
SAMPLE_ROOT = Path(__file__).resolve().parents[4] / "competitive_intelligence_agent"

#: A decision block in the shape ``parse_decision`` reads, so grading exercises
#: every dimension rather than stopping at an unparsed report.
ANSWER = """Basalt AI's Trust Layer overlaps our gateway's policy enforcement.

```json
{
  "material": true,
  "confidence": "medium",
  "company": "Basalt AI",
  "internal_product": "Contoso Tool Gateway",
  "exposure_score": 0.85,
  "citations": ["https://news.example.com/basalt-trust-layer"],
  "stakeholders": ["Gateway PM"],
  "caveats": ["Pricing is unannounced."],
  "next_actions": ["Brief the gateway team."]
}
```
"""


def _rows() -> list[dict[str, Any]]:
    path = SAMPLE_ROOT / "job_data" / "validation.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [row.get("task", row) for row in rows]


@pytest.fixture(scope="module")
def task() -> dict[str, Any]:
    return _rows()[0]


@pytest.fixture()
def env() -> CompetitiveIntelEnvironment:
    environment = CompetitiveIntelEnvironment()
    yield environment
    environment.close()


def test_reset_serves_the_production_surface(env, task):
    observation = env.reset(episode_id="ep-1", **task)

    assert observation.task_id == task["task_id"]
    assert observation.variant == task["variant"]
    assert observation.tool_surface == TOOL_SURFACE
    assert observation.done is False
    # The agent and this container are deployed separately and nothing plumbs
    # the surface from one to the other, so the list is asserted, not sampled.
    assert observation.tools == sorted(
        [
            "competitive_fabric___DiscoverArtifacts",
            "competitive_fabric___ExecuteQuery",
            "competitive_fabric___GetReportMetadata",
            "competitive_fabric___GetSemanticModelSchema",
            "competitive_fabric___ResolveReportIdFromUrl",
            "competitive_fabric___ValueSearch",
            "send_email",
            "update_tracker_record",
            "web_search",
        ]
    )


def test_mcp_schemas_come_from_the_tools_own_argument_models(env):
    listed = env.step(ListToolsAction())
    schemas = {tool.name: tool.input_schema for tool in listed.tools}

    assert len(schemas) == 9
    # Generated from `FunctionTool._params_model`, not hand-written: a required
    # string, and an optional one that keeps its default.
    assert schemas["web_search"]["required"] == ["search_query"]
    assert schemas["web_search"]["properties"]["search_query"]["type"] == "string"
    assert schemas["competitive_fabric___DiscoverArtifacts"]["required"] == ["searchQuery"]
    assert "maxResults" in schemas["competitive_fabric___DiscoverArtifacts"]["properties"]


def test_tool_calls_reach_the_session_the_rubric_reads(env, task):
    env.reset(episode_id="ep-1", **task)

    observation = env.step(
        CallToolAction(tool_name="web_search", arguments={"search_query": task["query"]})
    )

    assert observation.error is None
    # The rubric scores tool discipline from the session, so a call that does
    # not land here is a call that did not happen as far as the reward knows.
    assert [call.name for call in env._session.calls] == ["web_search"]


def test_a_tool_call_before_reset_is_a_protocol_error(env, task):
    observation = env.step(
        CallToolAction(tool_name="web_search", arguments={"search_query": "anything"})
    )

    assert observation.error is not None
    assert "reset" in str(observation.error)


def test_an_unreadable_task_fails_loudly(env):
    # Grading a task the environment did not understand would report a reward
    # for work the caller never requested.
    with pytest.raises(ValueError, match="could not be parsed"):
        env.reset(episode_id="ep-1", nonsense=True)


def test_grade_before_reset_is_a_protocol_error(env):
    with pytest.raises(RuntimeError, match="No active episode"):
        env.grade(GradeAction(response=ANSWER))


def test_reward_stays_inside_the_managed_rle_range(task):
    """Every row grades to a reward RLE will accept."""
    rows = _rows()[:12]
    for row in rows:
        environment = CompetitiveIntelEnvironment()
        environment.reset(episode_id=row["task_id"], **row)
        observation = environment.grade(GradeAction(response=ANSWER))
        assert 0.0 <= observation.reward <= 1.0, row["task_id"]
        assert observation.done is True
        environment.close()


def test_tool_call_errors_on_the_rollout_graph_penalise_the_reward(task):
    """A malformed tool call is the policy's own fault, so it costs reward.

    ``rollout_graph`` is ``None`` for every rollout this sample grades today
    -- see ``environment.grade``'s docstring -- so this pins the behaviour
    for the day a rollout target does supply one, rather than leaving it
    untested until then.
    """
    clean_env = CompetitiveIntelEnvironment()
    clean_env.reset(episode_id=task["task_id"], **task)
    clean = clean_env.grade(GradeAction(response=ANSWER))
    clean_env.close()

    errored_env = CompetitiveIntelEnvironment()
    errored_env.reset(episode_id=task["task_id"], **task)
    rollout_graph = {"turns": [{"n_tool_call_errors": 2}, {"n_tool_call_errors": 1}]}
    errored = errored_env.grade(GradeAction(response=ANSWER, rollout_graph=rollout_graph))
    errored_env.close()

    assert errored.info["metrics"]["n_tool_call_errors"] == 3.0
    assert errored.reward < clean.reward
    assert 0.0 <= errored.reward <= 1.0


def test_model_call_errors_on_the_rollout_graph_are_observability_only(task):
    """An upstream sampling failure is not the policy's fault, so it must not move the reward."""
    clean_env = CompetitiveIntelEnvironment()
    clean_env.reset(episode_id=task["task_id"], **task)
    clean = clean_env.grade(GradeAction(response=ANSWER))
    clean_env.close()

    errored_env = CompetitiveIntelEnvironment()
    errored_env.reset(episode_id=task["task_id"], **task)
    rollout_graph = {
        "model_call_errors": [{"error_code": "timeout"}, {"error_code": "http_error"}]
    }
    observed = errored_env.grade(GradeAction(response=ANSWER, rollout_graph=rollout_graph))
    errored_env.close()

    assert observed.info["metrics"]["n_model_call_errors"] == 2.0
    assert observed.reward == clean.reward
