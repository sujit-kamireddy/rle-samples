"""Runs one rollout by driving `opencode` as a child process of this server.

Isolation
---------
Rollouts share this container and `opencode` runs with `--dangerously-skip-permissions`, so the
boundary between two concurrent rollouts is a directory, not a VM. Two things make that safe. RLE
scopes the capture-proxy session key and the sandbox-tools URL and token per rollout, so neither is
usable across them. And this container holds no grading key to find -- see "What the harness is
given" below.

What the harness is given
-------------------------
A slim index (`vendor/task-index.json.gz`, built by `tools/build_task_index.py`), carrying only
each task's instruction, bucket coordinates and agent timeout -- not the full task suite.

That is a security boundary, not a size optimisation. Every `task.toml` in the suite carries
`metadata.gold_answer` and `verifier.env.EXPECTED_ANSWER`, and every `instruction.md` quotes its
question verbatim while `task.description` repeats it. An agent with a shell can therefore
`grep -rlF "<its own question>"` the suite, land on its own task directory and read its own answer
without analysing anything -- measured at 25 of 25 sampled tasks. So the answers do not ship here:
the grading key lives in `../rle`'s container, which gives the agent no shell.

Path rewriting
--------------
A task's instruction names absolute paths -- the CSVs at `/home/user/input`, the answer at
`/workdir/answer.txt`. Those are fine when the rollout owns the whole container, and collide the
moment two run side by side. Each rollout gets a private directory and the two absolute paths are
rewritten to point inside it. Nothing downstream depends on the original strings: `/grade` is
handed the answer *text*, never a path.
"""

from __future__ import annotations

import asyncio
import functools
import gzip
import json
import logging
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Shipped into the image by the Dockerfile. The copies inside the baked task suite are identical,
# but they are per-task copies of a file this repo also owns; reading the image's copy keeps the
# fetcher's version tied to the server that calls it.
PULL_BUCKET = Path(os.environ.get("PULL_BUCKET_PATH", "/opt/pull_bucket.py"))

# Built by `tools/build_task_index.py`; see "What the harness is given" above for why this is not
# the full task suite.
TASK_INDEX = Path(
    os.environ.get("TASK_INDEX_PATH", str(Path(__file__).resolve().parent / "vendor" / "task-index.json.gz"))
)

# The paths a task instruction hardcodes. See the module docstring.
TASK_INPUT_DIR = "/home/user/input"
TASK_ANSWER_FILE = "/workdir/answer.txt"

# `opencode` needs `provider/model`; it only honours a custom `baseURL` for these providers, so a
# model named without a provider is assumed to be OpenAI-compatible -- which is what RLE's capture
# proxy serves.
DEFAULT_PROVIDER = "openai"
_BASE_URL_PROVIDERS = frozenset({"anthropic", "google", "openai"})

_DEFAULT_AGENT_TIMEOUT_SEC = 600.0

# These two numbers are not preferences -- they are this harness's half of a contract with
# whatever serves the model, and getting them wrong is what makes a long rollout die without
# an answer.
#
# opencode compacts the transcript when its running token total reaches
# `limit.context - min(20000, limit.output)`, so `limit.output` is the headroom it holds back
# for the reply. The server, meanwhile, adds its own default output budget to the prompt and
# rejects the pair once it passes the model's window. If the two budgets disagree, opencode
# compacts too late by exactly the difference and the request is refused with
# "Requested prompt tokens + max_tokens exceeds the maximum context length" -- which reaches
# the agent as an opaque 502 and a rollout that never writes an answer.
#
# So `_DEFAULT_OUTPUT_TOKENS` tracks the capture proxy's own default budget rather than being
# chosen independently, and the context sits below the model's window to absorb the overshoot
# of a single turn that lands a large tool result after the last compaction check.
_DEFAULT_OUTPUT_TOKENS = 8192
_DEFAULT_CONTEXT_TOKENS = 30720
_ANSWER_RETRY_TIMEOUT_SEC = 120.0

# Compaction cannot save a transcript from a single oversized tool result. opencode checks its
# running total *between* steps, so one tool call that returns more than the whole window leaves
# the next request already over the limit, with nothing to compact away. opencode's own guard is
# far too loose to prevent that here: it truncates tool output at 2000 lines or 50KiB, and 50KiB
# of numeric CSV is roughly 34k tokens -- more than this window holds. Observed exactly that: a
# single `read` of an 801-line CSV took the prompt from 7,456 to 41,387 tokens in one step and
# killed the episode.
#
# `opencode` reads `AGENTS.md` from its working directory, which is the only per-rollout lever
# this harness has over how the agent spends its context. These are operating rules for a small
# window, not hints about the answer -- the task, the data and the grading are untouched.
_WORKSPACE_RULES = """# Working rules

You have a small context window. Reading a data file in full will overflow it and end the
session with no answer, so treat the transcript as the scarce resource it is.

- Never print a data file in full -- no `cat`, and no whole-file `read` of anything in the input
  directory. That applies however large the file looks; CSV rows are dense in tokens.
- Inspect data by computing over it. Write a Python script and run it; `pandas` and `numpy` are
  installed. `df.shape`, `df.columns`, `df.dtypes` and `df.head()` tell you the structure without
  spending the window on the contents.
- Print only what you need to see. Aggregate, filter or slice first, and keep each command's
  output to a few lines.
- Write the answer to the file the task names, exactly as instructed.
"""


class RolloutError(RuntimeError):
    """A rollout that could not be completed. The message reaches RLE as the failure reason."""


@functools.cache
def _index() -> list[dict[str, Any]]:
    """Loads the task index once per process.

    Cached because this is ~10MB of JSON and every rollout needs one row of it; a per-rollout parse
    would put that on the latency path for no benefit. The file is read-only and baked into the
    image, so there is nothing to invalidate.
    """
    try:
        raw = gzip.decompress(TASK_INDEX.read_bytes())
    except OSError as exc:
        raise RolloutError(
            f"task index {TASK_INDEX} is unreadable ({exc}). Build it with "
            "`python tools/build_task_index.py --write`."
        ) from exc
    return json.loads(raw)


def task_row(task_index: int) -> dict[str, Any]:
    """Returns the index entry `task_index` selects.

    A task's index *is* its identity -- `../rle`'s vendored answer key is keyed by it -- so the
    index is built in the suite's own task order (sorted by directory name, dot-directories
    skipped) and must not be re-sorted here. `tools/build_task_index.py` owns that ordering.
    """
    rows = _index()
    if not 0 <= task_index < len(rows):
        raise RolloutError(f"task_index {task_index} out of range ({len(rows)} tasks)")
    return rows[task_index]


def _split_model(model: str) -> tuple[str, str]:
    """Splits `provider/model`, but only where the prefix really names a provider.

    Model ids are not reliably provider-qualified: RLE serves checkpoints under their upstream
    id, and those carry a namespace of their own (`Qwen/Qwen3-32B`, `MAI/MAI-Code-1.1-Flash`).
    Partitioning on the first `/` unconditionally reads that namespace as a provider, which
    leaves `_opencode_config` unable to attach `baseURL` -- so the agent silently calls the real
    provider instead of the capture proxy, and nothing is captured.
    """
    provider, sep, model_id = model.partition("/")
    if sep and provider in _BASE_URL_PROVIDERS:
        return provider, model_id
    return DEFAULT_PROVIDER, model


def _env_int(name: str, default: int) -> int:
    """Read a positive integer override, ignoring anything unusable.

    A malformed budget must not take the rollout down: falling back to the default keeps
    the request inside the window, which is the whole point of sending one.
    """
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _write_workspace_rules(config_dir: Path, work: Path) -> None:
    """Puts the working rules where `opencode` will actually read them.

    `opencode` gathers instructions from two sources: `AGENTS.md` next to its config, which it
    always reads, and `AGENTS.md`/`CONTEXT.md` found from the project root, which it locates
    through VCS detection. This harness runs opencode on `OPENCODE_FAKE_VCS` rather than in a
    real repository, and the project-root lookup does not fire -- writing only that copy is
    silently a no-op. Both are written so the rules do not depend on which path works.
    """
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "AGENTS.md").write_text(_WORKSPACE_RULES)
    (work / "AGENTS.md").write_text(_WORKSPACE_RULES)


def _opencode_config(model: str, base_url: str) -> dict[str, Any]:
    """Builds the `opencode.json` that registers the model and points it at the capture proxy.

    `baseURL` sits under the provider's `options`, not at the provider root -- opencode ignores it
    anywhere else, and the failure mode is a silent call to the real provider rather than an error.
    Registering the model at all is what lets opencode accept a name that is not in its built-in
    registry, which every RLE-trained checkpoint is.

    `limit` does not reach the request: opencode's `/v1/responses` path sends no output budget
    at all, whatever the model declares. What `limit` does is set the point at which opencode
    compacts the transcript, which is the only lever this harness has over how large a prompt
    it can present. See `_DEFAULT_OUTPUT_TOKENS` for why those two numbers are chosen together.
    """
    provider, model_id = _split_model(model)
    model_config: dict[str, Any] = {
        "limit": {
            "context": _env_int("MODEL_CONTEXT_TOKENS", _DEFAULT_CONTEXT_TOKENS),
            "output": _env_int("MODEL_OUTPUT_TOKENS", _DEFAULT_OUTPUT_TOKENS),
        }
    }
    provider_config: dict[str, Any] = {"models": {model_id: model_config}}
    if base_url and provider in _BASE_URL_PROVIDERS:
        provider_config["options"] = {"baseURL": base_url}
    return {"provider": {provider: provider_config}}


async def _run(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout_sec: float,
    what: str,
) -> str:
    """Runs one subprocess to completion, returning its combined output.

    The process is killed rather than merely waited on when it runs long: `opencode` spawns its own
    children, so letting the coroutine time out while the process kept running would leak an agent
    into the container for the life of the server.
    """
    proc = await asyncio.create_subprocess_exec(
        *command,
        cwd=str(cwd),
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    communication = asyncio.create_task(proc.communicate())
    try:
        stdout, _ = await asyncio.wait_for(asyncio.shield(communication), timeout=timeout_sec)
    except asyncio.TimeoutError:
        if proc.returncode is None:
            proc.kill()
        stdout, _ = await communication
        raise RolloutError(
            f"{what} exceeded {timeout_sec:.0f}s; {_opencode_progress(stdout or b'')}"
        ) from None
    except asyncio.CancelledError:
        if proc.returncode is None:
            proc.kill()
        await communication
        raise

    output = stdout.decode("utf-8", "replace") if stdout else ""
    if proc.returncode != 0:
        raise RolloutError(f"{what} exited {proc.returncode}: {output[-2000:]}")
    return output


def _opencode_progress(output: bytes) -> str:
    """Summarize timed-out JSON events without logging tool results or task data."""
    steps = len(re.findall(rb'"type"\s*:\s*"step-start"', output))
    completed = len(re.findall(rb'"type"\s*:\s*"step-finish"', output))
    tools = len(re.findall(rb'"type"\s*:\s*"tool"', output))
    last = "none"
    for line in reversed(output[-32768:].splitlines()):
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict) and isinstance(event.get("type"), str):
            last = event["type"]
            break
    return f"OpenCode progress: {steps} steps started, {completed} finished, {tools} tool events; last={last}"


def _opencode_session_id(output: str) -> str | None:
    """Find the single session in OpenCode's JSON event stream."""
    sessions: set[str] = set()
    for line in output.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict):
            session_id = event.get("sessionID")
            if isinstance(session_id, str) and session_id.startswith("ses_"):
                sessions.add(session_id)
    return next(iter(sessions)) if len(sessions) == 1 else None


async def run_rollout(
    *,
    task_index: int,
    model: str,
    llm_url: str,
    api_key: str,
    compliance_endpoint: str = "",
    compliance_token: str = "",
    rollout_id: str = "",
) -> dict[str, Any]:
    """Runs one task end to end and returns the agent's own answer text."""
    row = task_row(task_index)

    tmp = tempfile.mkdtemp(prefix=f"rollout-{rollout_id or 'anon'}-")
    work = Path(tmp)
    try:
        return await _run_in(
            work=work,
            row=row,
            model=model,
            llm_url=llm_url,
            api_key=api_key,
            compliance_endpoint=compliance_endpoint,
            compliance_token=compliance_token,
        )
    finally:
        # `ignore_errors` because a finished rollout should not fail on cleanup; the worst case is
        # a leftover directory, which the next container restart clears.
        #
        # KEEP_ROLLOUT_DIR keeps it instead. A rollout that grades zero leaves nothing behind to
        # explain why -- the agent's transcript, the files it saw and the answer it did or did not
        # write all live here -- and the window to catch it mid-run is a few seconds.
        if not os.environ.get("KEEP_ROLLOUT_DIR"):
            shutil.rmtree(work, ignore_errors=True)
        else:
            logger.warning("KEEP_ROLLOUT_DIR set; leaving %s in place", work)


def _with_file_list(instruction: str, input_dir: Path) -> str:
    """Fills in the file list the instruction promises but leaves unrendered.

    This is a repair, not an embellishment. 4801 of the 5000 tasks name their files outright,
    one `- <name>` per line in sorted order; the remaining 199 carry the literal placeholder
    `- (see <dir>)` instead, which is upstream's renderer failing to substitute. Filling it in
    puts those 199 back in the format the other 96% already ship, which is why the output here
    is that same sorted `- <name>` -- a task the agent sees should not depend on which side of
    a rendering bug it landed on.

    Nothing else about the task changes: the schema is still the agent's to discover, which is
    what the instruction's "inspect the data first" step asks for.
    """
    placeholder = f"- (see {input_dir})"
    if placeholder not in instruction:
        return instruction
    names = sorted(entry.name for entry in input_dir.iterdir() if entry.is_file())
    if not names:
        return instruction
    return instruction.replace(placeholder, "\n".join(f"- {name}" for name in names))


async def _run_in(
    *,
    work: Path,
    row: dict[str, Any],
    model: str,
    llm_url: str,
    api_key: str,
    compliance_endpoint: str,
    compliance_token: str,
) -> dict[str, Any]:
    input_dir = work / "input"
    answer_file = work / "answer.txt"
    home = work / "home"
    home.mkdir(parents=True)

    await _fetch_inputs(row, input_dir, work)

    instruction = (
        row["instruction"]
        .replace(TASK_INPUT_DIR, str(input_dir))
        .replace(TASK_ANSWER_FILE, str(answer_file))
    )
    instruction = _with_file_list(instruction, input_dir)

    config_dir = home / ".config" / "opencode"
    config_dir.mkdir(parents=True)
    (config_dir / "opencode.json").write_text(
        json.dumps(_opencode_config(model, llm_url), indent=2)
    )

    # `opencode` collects instructions from two places: `AGENTS.md` beside its config, which is
    # always read, and `AGENTS.md`/`CONTEXT.md` discovered from the project root, which depends
    # on VCS detection and did not fire here (this harness runs on `OPENCODE_FAKE_VCS`, not a
    # real repository). Write both, so the rules do not hinge on that discovery working.
    _write_workspace_rules(config_dir, work)

    env = {
        **os.environ,
        "HOME": str(home),
        # opencode writes state and logs under these; per-rollout values keep two concurrent
        # rollouts from sharing a session store.
        "XDG_DATA_HOME": str(home / "share"),
        "XDG_STATE_HOME": str(home / "state"),
        "XDG_CONFIG_HOME": str(home / ".config"),
        # opencode refuses to run outside a repo; it accepts a stub rather than a real one.
        "OPENCODE_FAKE_VCS": "git",
        "OPENAI_API_KEY": api_key,
        "OPENAI_BASE_URL": llm_url,
    }
    # The capability the agent needs to file a compliance disclosure. Set per process, never
    # process-global: concurrent rollouts would otherwise overwrite each other's credentials and
    # misattribute a disclosure with no error anywhere.
    if compliance_endpoint and compliance_token:
        env["COMPLIANCE_ENDPOINT"] = compliance_endpoint
        env["COMPLIANCE_TOKEN"] = compliance_token

    provider, model_id = _split_model(model)
    timeout_sec = float(row["agent_timeout_sec"])
    started_at = asyncio.get_running_loop().time()
    try:
        output = await _run(
            [
                "opencode",
                f"--model={provider}/{model_id}",
                "run",
                "--format=json",
                "--thinking",
                "--dangerously-skip-permissions",
                "--",
                instruction,
            ],
            cwd=work,
            env=env,
            timeout_sec=timeout_sec,
            what="opencode",
        )
    except RolloutError as exc:
        # The agent lost this episode on its own terms: it talked itself past the model's
        # context window, ran out of time, or crashed. `../rle/server/env.py` already grades
        # `ok: False` as a zero, which is the honest score for an episode that produced no
        # answer. Reporting it as a harness failure instead would abort the entire training
        # run over one bad episode, which is how a single runaway transcript takes down a
        # job that the other rollouts in its group were completing normally.
        logger.warning("task %s ended without an answer: %s", row["name"], exc)
        return {"ok": False, "error": str(exc), "answer_text": None}

    answer_text = _read_answer(answer_file)
    if answer_text is None:
        session_id = _opencode_session_id(output)
        remaining = timeout_sec - (asyncio.get_running_loop().time() - started_at)
        if session_id and remaining > 1:
            try:
                await _run(
                    [
                        "opencode",
                        f"--model={provider}/{model_id}",
                        "run",
                        "--session",
                        session_id,
                        "--format=json",
                        "--thinking",
                        "--dangerously-skip-permissions",
                        "--",
                        f"The required answer file is still missing. Continue this task in the same session "
                        f"and use a tool to write your final answer to {answer_file}. "
                        "Do not finish until the file contains the answer.",
                    ],
                    cwd=work,
                    env=env,
                    timeout_sec=min(remaining, _ANSWER_RETRY_TIMEOUT_SEC),
                    what="opencode answer retry",
                )
            except RolloutError as exc:
                reason = str(exc).split(": ", 1)[0]
                logger.warning("task %s %s", row["name"], reason)
                return {"ok": False, "error": reason, "answer_text": None}
            answer_text = _read_answer(answer_file)
        if answer_text is None:
            logger.warning("task %s produced no answer file after opencode finished", row["name"])
    return {
        "ok": answer_text is not None,
        "error": None if answer_text is not None else "agent did not write answer.txt",
        "answer_text": answer_text,
    }


async def _fetch_inputs(row: dict[str, Any], input_dir: Path, work: Path) -> None:
    """Pulls this task's files into `input_dir`.

    The fetcher is run as a subprocess rather than imported because it reads its configuration from
    the environment at module scope; a subprocess gives each rollout its own read of that, which an
    import into this long-lived process would not.
    """
    if not row.get("bucket_base_url") or not row.get("bucket_prefix"):
        raise RolloutError(f"task {row['name']} has no bucket coordinates to fetch inputs from")

    input_dir.mkdir(parents=True, exist_ok=True)
    await _run(
        [sys.executable, str(PULL_BUCKET)],
        cwd=work,
        env={
            **os.environ,
            "BUCKET_BASE_URL": row["bucket_base_url"],
            "BUCKET_PREFIX": row["bucket_prefix"],
            "BUCKET_DEST": str(input_dir),
            "BUCKET_STAGE": str(work / ".input-stage"),
        },
        # Generous: one prefix in this collection is 844MB. The fetcher is resumable, but nothing
        # retries it here the way a task container's healthcheck did, so it gets one long attempt.
        timeout_sec=900.0,
        what="pull_bucket",
    )


def _read_answer(answer_file: Path) -> str | None:
    """Returns the agent's answer, or `None` when it wrote no non-empty answer.

    A missing file is not an error here: `/grade` already treats a missing answer as an ungraded,
    zero-reward rollout, and that is a truer record of what happened than failing the rollout.
    """
    try:
        return answer_file.read_text().strip() or None
    except OSError:
        return None
