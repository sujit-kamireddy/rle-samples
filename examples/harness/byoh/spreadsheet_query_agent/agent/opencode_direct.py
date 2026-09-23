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
    provider, sep, model_id = model.partition("/")
    return (provider, model_id) if sep else (DEFAULT_PROVIDER, model)


def _opencode_config(model: str, base_url: str) -> dict[str, Any]:
    """Builds the `opencode.json` that registers the model and points it at the capture proxy.

    `baseURL` sits under the provider's `options`, not at the provider root -- opencode ignores it
    anywhere else, and the failure mode is a silent call to the real provider rather than an error.
    Registering the model at all is what lets opencode accept a name that is not in its built-in
    registry, which every RLE-trained checkpoint is.
    """
    provider, model_id = _split_model(model)
    provider_config: dict[str, Any] = {"models": {model_id: {}}}
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
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout_sec)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise RolloutError(f"{what} exceeded {timeout_sec:.0f}s") from None

    output = stdout.decode("utf-8", "replace") if stdout else ""
    if proc.returncode != 0:
        raise RolloutError(f"{what} exited {proc.returncode}: {output[-2000:]}")
    return output


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
        shutil.rmtree(work, ignore_errors=True)


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

    config_dir = home / ".config" / "opencode"
    config_dir.mkdir(parents=True)
    (config_dir / "opencode.json").write_text(
        json.dumps(_opencode_config(model, llm_url), indent=2)
    )

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
        timeout_sec=float(row["agent_timeout_sec"]),
        what="opencode",
    )

    answer_text = _read_answer(answer_file)
    if answer_text is None:
        logger.warning(
            "task %s produced no answer file; agent output tail: %s",
            row["name"],
            output[-500:],
        )
    return {"ok": True, "error": None, "answer_text": answer_text}


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
    """Returns the agent's answer, or `None` when it never wrote one.

    A missing file is not an error here: `/grade` already treats a missing answer as an ungraded,
    zero-reward rollout, and that is a truer record of what happened than failing the rollout.
    """
    try:
        return answer_file.read_text().strip()
    except OSError:
        return None
