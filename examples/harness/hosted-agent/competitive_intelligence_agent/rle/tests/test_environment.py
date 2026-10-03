"""The OpenEnv environment, checked against the harness it was ported from.

The point of these tests is parity. This environment exists to express the same
rollout through a different protocol, so the thing worth asserting is that the
protocol is all that changed: the same task, the same tool calls and the same
final answer must produce the same reward here as they do through ``../rle``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from azure.ai.projects.rle.environments import GradeAction
from fastapi.testclient import TestClient
from openenv.core.env_server.mcp_environment import CallToolAction, ListToolsAction

from rle.server.environment import TOOL_SURFACE, CompetitiveIntelEnvironment

# `rle_deprecated/` is scheduled for deletion, and this environment no longer
# imports anything from it: `rle/` carries its own copy of the world, the
# tasks, the tools and the rubric. The legacy app is loaded here only so
# `test_grade_matches_the_legacy_harness` can hold the duplicate to account
# while both copies exist. It skips, rather than errors, the moment
# `rle_deprecated/` goes, and this block goes with it.
try:
    from rle_deprecated.server.env import app as legacy_app
except ImportError:  # pragma: no cover - the state after `rle_deprecated/` is deleted
    legacy_app = None


SAMPLE_ROOT = Path(__file__).resolve().parent.parent.parent

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
        env.grade(GradeAction(answer=ANSWER))


@pytest.mark.skipif(legacy_app is None, reason="the legacy `rle_deprecated/` harness has been removed")
def test_grade_matches_the_legacy_harness(task):
    """Same task, same tool call, same answer -- same reward.

    This is the test that justifies the port, and now also the test that keeps
    the duplicated ``rl/`` honest. Both environments are driven through their
    own public protocol over their own copy of the rubric, so what is compared
    is two complete paths, not two calls into one shared function.
    """
    openenv_env = CompetitiveIntelEnvironment()
    openenv_env.reset(episode_id="ep-1", **task)
    openenv_env.step(
        CallToolAction(tool_name="web_search", arguments={"search_query": task["query"]})
    )
    ported = openenv_env.step(GradeAction(answer=ANSWER))
    openenv_env.close()

    with TestClient(legacy_app) as client:
        assert client.post("/reset", json=task).status_code == 200
        assert (
            client.post(
                "/tools/web_search", json={"search_query": task["query"]}
            ).status_code
            == 200
        )
        legacy = client.post("/grade", json={"agent_response": ANSWER}).json()

    assert ported.reward == legacy["reward"]
    assert ported.is_success == legacy["is_success"]
    assert ported.info == legacy["info"]
    # A reward that is identically zero on both sides would pass the three
    # assertions above while proving nothing about the rubric.
    assert 0.0 < ported.reward <= 1.0
    assert ported.info["parsed"] is True


def test_reward_stays_inside_the_managed_rle_range(task):
    """Every row grades to a reward RLE will accept."""
    rows = _rows()[:12]
    for row in rows:
        environment = CompetitiveIntelEnvironment()
        environment.reset(episode_id=row["task_id"], **row)
        observation = environment.grade(GradeAction(answer=ANSWER))
        assert 0.0 <= observation.reward <= 1.0, row["task_id"]
        assert observation.done is True
        environment.close()
