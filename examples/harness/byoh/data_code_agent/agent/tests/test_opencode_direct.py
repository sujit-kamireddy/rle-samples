"""Tests for the opencode rollout path.

The ordering test is the important one. A task's index is its identity: `../../rle`'s answer key is
keyed by position, so an index built in a different order than the suite's would not fail anything
-- it would grade every rollout against the wrong task's answer and still look like a working
system.
"""

from __future__ import annotations

import asyncio
import gzip
import inspect
import json
import os
import sys
from pathlib import Path

import pytest

import opencode_direct as od

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


def test_opencode_config_declares_an_output_budget():
    """An absent budget is what makes Loom reject a prompt that is well inside the window.

    opencode only sends `max_output_tokens` when the model declares a limit, and the capture
    proxy can lower a budget that is present but never invent one -- so without this the
    request carries no budget and Loom's own default decides whether the rollout survives.
    """
    config = od._opencode_config("my-checkpoint", "https://capture.example/v1")
    limit = config["provider"]["openai"]["models"]["my-checkpoint"]["limit"]
    assert limit["output"] == od._DEFAULT_OUTPUT_TOKENS
    assert limit["context"] == od._DEFAULT_CONTEXT_TOKENS


def test_output_budget_is_overridable(monkeypatch):
    monkeypatch.setenv("MODEL_OUTPUT_TOKENS", "512")
    monkeypatch.setenv("MODEL_CONTEXT_TOKENS", "16384")
    limit = od._opencode_config("m", "https://capture.example/v1")["provider"]["openai"]["models"]["m"]["limit"]
    assert limit == {"context": 16384, "output": 512}


@pytest.mark.parametrize("value", ["", "   ", "nonsense", "0", "-1"])
def test_unusable_budget_overrides_fall_back(monkeypatch, value):
    """A malformed override must not remove the budget; that reinstates the failure."""
    monkeypatch.setenv("MODEL_OUTPUT_TOKENS", value)
    limit = od._opencode_config("m", "https://capture.example/v1")["provider"]["openai"]["models"]["m"]["limit"]
    assert limit["output"] == od._DEFAULT_OUTPUT_TOKENS


def test_namespaced_checkpoint_ids_keep_the_capture_proxy():
    """`Qwen/Qwen3-32B` names a checkpoint, not a provider.

    Reading that namespace as a provider drops `baseURL`, because only real providers take
    one -- and the agent then calls the real provider instead of the capture proxy, so the
    rollout captures nothing.
    """
    for model in ("Qwen/Qwen3-32B", "MAI/MAI-Code-1.1-Flash"):
        assert od._split_model(model) == ("openai", model)
        provider = od._opencode_config(model, "https://capture.example/v1")["provider"]["openai"]
        assert model in provider["models"]
        assert provider["options"]["baseURL"] == "https://capture.example/v1"


def test_a_real_provider_prefix_is_still_honoured():
    assert od._split_model("anthropic/claude-sonnet-4") == ("anthropic", "claude-sonnet-4")


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


def test_the_promised_file_list_is_filled_in(tmp_path):
    """The filled list must match what the other 4801 tasks already ship: sorted `- <name>`."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    for name in ("us-east-1.csv", "ap-south-1.csv"):
        (input_dir / name).write_text("1,2\n")
    instruction = f"Files (in {input_dir}, no subfolders):\n- (see {input_dir})\n\nQuestion:"
    filled = od._with_file_list(instruction, input_dir)
    assert "- ap-south-1.csv\n- us-east-1.csv" in filled
    assert "(see " not in filled


def test_the_placeholder_is_a_minority_of_the_suite(rows):
    """Pins the premise of `_with_file_list`.

    Filling the placeholder is only a repair while most tasks render a real list; if upstream
    ever ships the placeholder everywhere, it is deliberate and we would be the ones diverging.
    """
    placeholder = sum(1 for r in rows if "- (see /home/user/input)" in r["instruction"])
    assert 0 < placeholder < len(rows) / 10, placeholder


def test_our_rendering_matches_a_real_multi_file_task(rows, tmp_path):
    """Builds the list from the filenames of a task that ships its own, and compares."""
    marker = "Files (in /home/user/input, no subfolders):\n"
    for row in rows:
        body = row["instruction"].split(marker, 1)[-1].split("\n\n", 1)[0]
        names = [line[2:] for line in body.splitlines() if line.startswith("- ")]
        if len(names) < 2 or body.strip() == "- (see /home/user/input)":
            continue
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        for name in names:
            (input_dir / name).write_text("1\n")
        rendered = od._with_file_list(f"- (see {input_dir})", input_dir)
        assert rendered == body.strip(), row["name"]
        return
    pytest.fail("no multi-file task found to compare against")


def test_an_instruction_without_the_placeholder_is_untouched(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "a.csv").write_text("1\n")
    instruction = "Files:\n- a.csv\n"
    assert od._with_file_list(instruction, input_dir) == instruction


def test_an_empty_input_dir_leaves_the_placeholder_alone(tmp_path):
    """Better the original text than a heading with nothing under it."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    instruction = f"- (see {input_dir})"
    assert od._with_file_list(instruction, input_dir) == instruction


def test_a_lost_episode_is_graded_not_failed(monkeypatch, tmp_path):
    """opencode failing is the agent losing the episode, not the harness breaking.

    The distinction decides whether one runaway transcript scores zero or aborts the
    whole training run, so it is asserted rather than left to the caller's judgement.
    """
    row = od._index()[0]

    async def explode(*args, **kwargs):
        raise od.RolloutError("opencode exited 1: context window exceeded")

    async def no_fetch(*args, **kwargs):
        return None

    monkeypatch.setattr(od, "_run", explode)
    monkeypatch.setattr(od, "_fetch_inputs", no_fetch)

    result = asyncio.run(
        od.run_rollout(
            task_index=0,
            model="openai/stub",
            llm_url="http://127.0.0.1:1",
            api_key="k",
            rollout_id="r1",
            work_root=tmp_path,
        )
        if "work_root" in inspect.signature(od.run_rollout).parameters
        else od.run_rollout(
            task_index=0,
            model="openai/stub",
            llm_url="http://127.0.0.1:1",
            api_key="k",
            rollout_id="r1",
        )
    )

    assert result["ok"] is False
    assert result["answer_text"] is None
    assert "context window exceeded" in result["error"]
    assert row["name"]


def test_opencode_session_id_requires_one_session():
    event = json.dumps({"type": "step-finish", "sessionID": "ses_first"})
    assert od._opencode_session_id(f"diagnostic line\n{event}\n{event}") == "ses_first"
    assert od._opencode_session_id(event + '\n{"sessionID":"ses_second"}') is None
    assert od._opencode_session_id('{"type":"step-finish"}\nnot JSON') is None


def test_a_timed_out_opencode_preserves_safe_progress(tmp_path):
    async def run():
        return await od._run(
            [sys.executable, "-u", "-c",
             "import json,time; print(json.dumps({'type':'step-start','secret':'SECRET-OUTPUT'}), flush=True); time.sleep(10)"],
            cwd=tmp_path,
            env=os.environ.copy(),
            timeout_sec=0.2,
            what="opencode",
        )

    with pytest.raises(od.RolloutError) as exc:
        asyncio.run(run())
    assert "1 steps started" in str(exc.value)
    assert "last=step-start" in str(exc.value)
    assert "SECRET-OUTPUT" not in str(exc.value)


def test_missing_answer_resumes_the_same_session(monkeypatch, tmp_path):
    calls = []

    async def no_fetch(*args, **kwargs):
        pass

    async def run(command, *, cwd, env, timeout_sec, what):
        calls.append((command, cwd, env, timeout_sec, what))
        if len(calls) == 2:
            (cwd / "answer.txt").write_text(" 42 \n")
        return '{"type":"step-finish","sessionID":"ses_this_rollout"}\n'

    monkeypatch.setattr(od, "_fetch_inputs", no_fetch)
    monkeypatch.setattr(od, "_run", run)
    result = asyncio.run(od._run_in(
        work=tmp_path,
        row={"name": "test", "instruction": "Write /workdir/answer.txt", "agent_timeout_sec": 600},
        model="Qwen/Qwen3-32B",
        llm_url="https://capture.invalid/v1",
        api_key="rollout-key",
        compliance_endpoint="https://tools.invalid",
        compliance_token="rollout-token",
    ))

    assert result == {"ok": True, "error": None, "answer_text": "42"}
    assert len(calls) == 2
    assert "--session" not in calls[0][0]
    assert calls[1][0][calls[1][0].index("--session") + 1] == "ses_this_rollout"
    assert str(tmp_path / "answer.txt") in calls[1][0][-1]
    assert calls[1][1] == calls[0][1] == tmp_path
    assert calls[1][2]["XDG_DATA_HOME"] == calls[0][2]["XDG_DATA_HOME"]
    assert calls[1][2]["OPENAI_API_KEY"] == calls[0][2]["OPENAI_API_KEY"]
    assert 0 < calls[1][3] <= od._ANSWER_RETRY_TIMEOUT_SEC


@pytest.mark.parametrize("first_output", [
    '{"type":"step-finish","sessionID":"ses_this_rollout"}\n',
    "no session events\n",
])
def test_missing_answer_stays_a_zero_reward_outcome(monkeypatch, tmp_path, caplog, first_output):
    calls = []

    async def no_fetch(*args, **kwargs):
        pass

    async def run(command, *, cwd, env, timeout_sec, what):
        calls.append(command)
        (cwd / "answer.txt").write_text(" \n")
        return first_output + "SECRET-OUTPUT"

    monkeypatch.setattr(od, "_fetch_inputs", no_fetch)
    monkeypatch.setattr(od, "_run", run)
    result = asyncio.run(od._run_in(
        work=tmp_path,
        row={"name": "test", "instruction": "Write /workdir/answer.txt", "agent_timeout_sec": 600},
        model="Qwen/Qwen3-32B",
        llm_url="https://capture.invalid/v1",
        api_key="rollout-key",
        compliance_endpoint="",
        compliance_token="",
    ))

    assert result == {"ok": False, "error": "agent did not write answer.txt", "answer_text": None}
    assert len(calls) == (2 if "ses_this_rollout" in first_output else 1)
    assert "SECRET-OUTPUT" not in caplog.text


def test_existing_answer_does_not_trigger_retry(monkeypatch, tmp_path):
    calls = []

    async def no_fetch(*args, **kwargs):
        pass

    async def run(command, *, cwd, env, timeout_sec, what):
        calls.append(command)
        (cwd / "answer.txt").write_text("answer")
        return "no session events"

    monkeypatch.setattr(od, "_fetch_inputs", no_fetch)
    monkeypatch.setattr(od, "_run", run)
    result = asyncio.run(od._run_in(
        work=tmp_path,
        row={"name": "test", "instruction": "Write /workdir/answer.txt", "agent_timeout_sec": 600},
        model="Qwen/Qwen3-32B",
        llm_url="https://capture.invalid/v1",
        api_key="rollout-key",
        compliance_endpoint="",
        compliance_token="",
    ))

    assert result == {"ok": True, "error": None, "answer_text": "answer"}
    assert len(calls) == 1


def test_failed_answer_retry_is_an_explicit_agent_failure(monkeypatch, tmp_path):
    calls = []

    async def no_fetch(*args, **kwargs):
        pass

    async def run(command, *, cwd, env, timeout_sec, what):
        calls.append(command)
        if len(calls) == 2:
            raise od.RolloutError("opencode answer retry exceeded 120s")
        return '{"type":"step-finish","sessionID":"ses_this_rollout"}\n'

    monkeypatch.setattr(od, "_fetch_inputs", no_fetch)
    monkeypatch.setattr(od, "_run", run)
    result = asyncio.run(od._run_in(
        work=tmp_path,
        row={"name": "test", "instruction": "Write /workdir/answer.txt", "agent_timeout_sec": 600},
        model="Qwen/Qwen3-32B",
        llm_url="https://capture.invalid/v1",
        api_key="rollout-key",
        compliance_endpoint="",
        compliance_token="",
    ))

    assert len(calls) == 2
    assert result == {"ok": False, "error": "opencode answer retry exceeded 120s", "answer_text": None}


def test_the_compaction_budget_fits_the_training_window():
    """opencode compacts at `context - min(20000, output)`, then the server adds `output`.

    The sum of those is the largest request this harness can produce, so it has to stay under
    the training session's window. When it does not, opencode compacts too late by exactly the
    difference and every long rollout dies with "Requested prompt tokens + max_tokens exceeds
    the maximum context length" -- as a 502 with no answer, not as anything that names a budget.
    """
    window = 32768
    reserved = min(20000, od._DEFAULT_OUTPUT_TOKENS)
    compaction_threshold = od._DEFAULT_CONTEXT_TOKENS - reserved

    assert compaction_threshold > 0
    assert compaction_threshold + od._DEFAULT_OUTPUT_TOKENS < window


def test_the_workspace_rules_steer_the_agent_off_whole_file_reads():
    """One oversized tool result overflows the window before compaction can run.

    opencode truncates tool output at 2000 lines or 50KiB, and 50KiB of numeric CSV is more
    tokens than this window holds, so its own guard does not bound the prompt. These rules are
    the harness's only per-rollout lever over that, and they have to keep naming both ways an
    agent dumps a file into the transcript.
    """
    rules = od._WORKSPACE_RULES

    assert "cat" in rules
    assert "read" in rules
    assert "pandas" in rules


def test_the_workspace_rules_give_away_nothing_about_the_answer():
    """These rules ship into the agent's working directory, which is the reward-hacking surface.

    `../rle` holds the grading key precisely so the agent cannot reach it; a file this harness
    writes next to the agent must not reintroduce what that separation exists to prevent.
    """
    rules = od._WORKSPACE_RULES.lower()

    for leak in ("gold", "expected_answer", "verifier", "reward", "grade"):
        assert leak not in rules


def test_the_rules_are_written_where_opencode_actually_reads_them(tmp_path):
    """The config-dir copy is the delivered one; the project copy is only a fallback.

    opencode always reads `AGENTS.md` beside its config, but finds a project-root copy through
    VCS detection -- which does not fire here, because this harness runs opencode on
    `OPENCODE_FAKE_VCS` rather than in a real repository. Writing only the project copy is
    silently a no-op, which is exactly what happened before this was pinned down, so the
    config-dir copy is the assertion that matters.
    """
    config_dir = tmp_path / "home" / ".config" / "opencode"
    work = tmp_path / "work"
    work.mkdir()

    od._write_workspace_rules(config_dir, work)

    assert (config_dir / "AGENTS.md").read_text() == od._WORKSPACE_RULES
    assert (work / "AGENTS.md").read_text() == od._WORKSPACE_RULES
