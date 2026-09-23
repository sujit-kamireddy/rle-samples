"""Tests for the opencode rollout path.

The ordering test is the important one. A task's index is its identity: `../../rle`'s answer key is
keyed by position, so an index built in a different order than the suite's would not fail anything
-- it would grade every rollout against the wrong task's answer and still look like a working
system.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from server import opencode_direct as od

REPO = Path(__file__).resolve().parent.parent.parent
ANSWER_KEY = (
    REPO / "rle" / "server" / "vendor" / "task-meta" / "FineEnvs__data-agent-harbor-train.json.gz"
)


@pytest.fixture(scope="module")
def rows() -> list[dict]:
    return od._index()


def test_index_is_present_and_complete(rows):
    assert len(rows) == 5000
    assert all(r["instruction"] and r["bucket_prefix"] for r in rows)


def test_index_carries_no_grading_key(rows):
    """The whole point of the slim index: the answer must not be reachable from this container."""
    blob = json.dumps(rows)
    assert "gold_answer" not in blob
    assert "EXPECTED_ANSWER" not in blob


def test_index_order_matches_the_answer_key(rows):
    """Position must mean the same task here and in `../../rle`'s key, or every grade is wrong."""
    key = json.loads(gzip.decompress(ANSWER_KEY.read_bytes()))
    assert len(key) == len(rows)
    for i, (entry, row) in enumerate(zip(key, rows)):
        # The suite's `[task] name` carries a prefix (`train-verify/...`) that the directory
        # name does not; the part after it is the task identity.
        _, _, name = entry["task_name"].partition("/")
        assert name == row["name"], f"index {i} disagrees with the answer key"


def test_task_row_rejects_out_of_range(rows):
    with pytest.raises(od.RolloutError, match="out of range"):
        od.task_row(len(rows))
    with pytest.raises(od.RolloutError, match="out of range"):
        od.task_row(-1)


def test_opencode_config_points_the_model_at_the_capture_proxy():
    config = od._opencode_config("openai/my-checkpoint", "https://capture.example/v1")
    provider = config["provider"]["openai"]
    assert "my-checkpoint" in provider["models"]
    # opencode reads baseURL from the provider's `options`, not its root; putting it anywhere else
    # silently calls the real provider instead of the capture proxy.
    assert provider["options"]["baseURL"] == "https://capture.example/v1"


def test_opencode_config_defaults_the_provider():
    config = od._opencode_config("my-checkpoint", "https://capture.example/v1")
    assert "openai" in config["provider"]
    assert "my-checkpoint" in config["provider"]["openai"]["models"]


def test_instruction_paths_are_rewritten_into_the_rollout_dir(rows):
    """Two concurrent rollouts must not share the hardcoded task paths."""
    instruction = rows[0]["instruction"]
    assert od.TASK_INPUT_DIR in instruction and od.TASK_ANSWER_FILE in instruction
    rewritten = instruction.replace(od.TASK_INPUT_DIR, "/w/input").replace(
        od.TASK_ANSWER_FILE, "/w/answer.txt"
    )
    assert od.TASK_INPUT_DIR not in rewritten
    assert od.TASK_ANSWER_FILE not in rewritten


def test_compliance_clause_survived_into_the_index(rows):
    """The disclosure demo depends on this text reaching the agent."""
    assert "report_sensitive_data_access" in rows[0]["instruction"]
