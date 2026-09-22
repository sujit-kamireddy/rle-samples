"""RLE harness container for a real SWE-bench-Lite instance.

Wraps `psf/requests` at its pinned base commit (see `fixtures/instance.json`,
sourced from `princeton-nlp/SWE-bench_Lite`). The harness (`agent/`) is
given the real GitHub issue text and must edit a disposable checkout to fix
it, then open a pull request, exactly like the "code repair agent" example
in the RLE design docs, but grounded in a real repo, real issue, and real
regression test rather than synthetic data.

This is a Harness container, so RLE calls exactly four things on it and
nothing else:

    GET  /health   readiness, polled before /reset
    POST /reset    the caller's task, verbatim
    POST /tools/*  the harness's tool calls, proxied per rollout
    POST /grade    {"rollout": ..., "agent_response": "..."} -> a reward

There is deliberately no `/step` and no OpenEnv `Environment`, `Action`, or
`Observation` here. Those belong to the Gym: OpenEnv subtype, where RLE drives
the model through the environment step by step. In a Harness RLE the harness
owns its own loop, so the only thing the environment does is set the task up,
serve the tools, and score the result. RLE checks `/reset` for a success status
and reads nothing from its body, so returning an `Observation` here would be
returning a value nobody looks at.

Each rollout gets its own working copy of the repo under `/tmp/rollouts/`.
`/reset` seeds it from the read-only `/opt/base-repo` reference clone baked
into the image, then applies the hidden test patch (`fixtures/test_patch.diff`)
that adds the regression test the real fix must satisfy. That patch is never
shown to the harness, only used for grading. `workspace.apply_patch` mocks the
production tool the harness uses to edit the checkout;
`github.create_pull_request` mocks submitting the fix for review.

## Where the taskset lives

Not in this image. A taskset is the caller's: every Execute Rollout call
carries its own task, and RLE posts that task to `/reset` exactly as the
calling job supplied it. `fixtures/instance.json` is a single instance baked
in so the sample is runnable on its own, and `/reset` below checks the task it
receives against it rather than ignoring the task, so a caller that sends a
different `instance_id` gets told why the container cannot serve it instead of
silently grading the wrong repo. Scaling this to a real taskset means making
the checkout follow `instance_id`, not changing the protocol.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from fastapi import Body, FastAPI, Header, HTTPException

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
# Overridable so this sample can be exercised outside its image, where the
# checkout and scratch space live wherever the caller put them, without
# editing the file. Inside the image the defaults are the paths the Dockerfile
# creates, so a deployed rollout is unaffected.
BASE_REPO_DIR = Path(os.environ.get("CODE_REPAIR_BASE_REPO_DIR", "/opt/base-repo"))
ROLLOUTS_DIR = Path(os.environ.get("CODE_REPAIR_ROLLOUTS_DIR", "/tmp/rollouts"))
# `/grade` runs a test suite the agent has just edited, and a bad patch can
# leave it hanging rather than failing. Nothing else bounds that, so without a
# timeout one rollout waits forever instead of scoring zero.
GRADING_TIMEOUT_S = float(os.environ.get("CODE_REPAIR_GRADING_TIMEOUT_S", "120"))

INSTANCE: dict[str, Any] = json.loads((FIXTURES_DIR / "instance.json").read_text())
TEST_PATCH = (FIXTURES_DIR / "test_patch.diff").read_text()

logger = logging.getLogger("code-repair-rle")

app = FastAPI(title="code-repair-rle")


class Rollout:
    """The one attempt this container is currently hosting.

    A sandbox serves a single rollout at a time, so one module-level record is
    enough and every route below reads this. A container that served several at
    once would key all of this by the `x-rle-rollout-id` header instead.
    """

    def __init__(self) -> None:
        self.rollout_id: Optional[str] = None
        self.workspace: Optional[Path] = None
        self.pull_requests: dict[str, dict[str, Any]] = {}

    def start(self, rollout_id: str, workspace: Path) -> None:
        self.rollout_id = rollout_id
        self.workspace = workspace
        self.pull_requests = {}

    def require_workspace(self) -> Path:
        if self.workspace is None:
            raise HTTPException(status_code=409, detail="No active rollout; call /reset first.")
        return self.workspace


ROLLOUT = Rollout()


@app.get("/health")
async def health() -> dict[str, str]:
    """Readiness probe. RLE polls this before `/reset`, because a sandbox
    reports Running once the container is scheduled, which is well before
    uvicorn has bound its port."""
    return {"status": "healthy"}


@app.post("/reset")
async def reset(
    task: dict[str, Any] = Body(default_factory=dict),
    rollout_id: Optional[str] = Header(default=None, alias="x-rle-rollout-id"),
) -> dict[str, Any]:
    """Prepares a fresh checkout for one rollout.

    The body is the caller's task exactly as the calling job supplied it. RLE
    does not wrap it and does not read this response beyond its status code.
    """
    requested = task.get("instance_id")
    if requested is not None and requested != INSTANCE["instance_id"]:
        # Failing here is the point. A container that quietly graded its own
        # baked-in instance against a task asking for another one would report
        # a reward for work the caller never requested.
        raise HTTPException(
            status_code=400,
            detail=(
                f"This environment version serves {INSTANCE['instance_id']!r}, "
                f"but the task requested {requested!r}. Publish a version whose "
                f"image carries that instance, or send a task for this one."
            ),
        )
    if requested is None:
        logger.info(
            "Task carried no instance_id; serving the bundled instance %s.",
            INSTANCE["instance_id"],
        )

    rollout_id = rollout_id or str(uuid4())
    workspace = ROLLOUTS_DIR / rollout_id
    if workspace.exists():
        shutil.rmtree(workspace)
    shutil.copytree(BASE_REPO_DIR, workspace)
    # The hidden regression test is applied here, not shipped to the harness:
    # /reset receives task-specific grading data that the harness never sees,
    # only the agent-visible task input does.
    subprocess.run(
        ["git", "apply", "-"],
        input=TEST_PATCH,
        cwd=workspace,
        text=True,
        check=True,
    )
    ROLLOUT.start(rollout_id, workspace)
    return {"instance_id": INSTANCE["instance_id"], "rollout_id": rollout_id}


@app.post("/tools/workspace.apply_patch")
async def workspace_apply_patch(arguments: dict[str, Any]) -> dict[str, Any]:
    """Mocks the production `workspace.apply_patch` tool the harness calls."""
    try:
        subprocess.run(
            ["git", "apply", "-"],
            input=arguments["patch"],
            cwd=ROLLOUT.require_workspace(),
            text=True,
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as error:
        return {"applied": False, "error": error.stderr}
    return {"applied": True}


@app.post("/tools/github.create_pull_request")
async def github_create_pull_request(arguments: dict[str, Any]) -> dict[str, Any]:
    """Mocks the production `github.create_pull_request` tool the harness calls."""
    ROLLOUT.require_workspace()
    ROLLOUT.pull_requests[arguments["branch"]] = {
        "title": arguments.get("title", ""),
        "body": arguments.get("body", ""),
    }
    return {"number": len(ROLLOUT.pull_requests)}


@app.post("/grade")
async def grade(payload: dict[str, Any] = Body(default_factory=dict)) -> dict[str, Any]:
    """Runs the real regression test against the harness's edited checkout.

    RLE sends `{"rollout": <sanitized rollout>, "agent_response": "<final
    answer>"}`. The reward comes from the workspace rather than from
    `agent_response`, because in this task the answer is the patch the harness
    already applied and the claim it makes about that patch proves nothing. The
    answer is still recorded below, so a reward can be read back against what
    the harness said it did.
    """
    agent_response = payload.get("agent_response")
    workspace = ROLLOUT.require_workspace()
    try:
        result = subprocess.run(
            ["python", "-m", "pytest", *INSTANCE["fail_to_pass"], "-q"],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=GRADING_TIMEOUT_S,
            env={
                **os.environ,
                # The 2016 checkout under test has to win the import over any
                # other `requests` on the path. `python -m` already puts cwd
                # first, but that is incidental; naming the workspace makes the
                # precedence explicit.
                "PYTHONPATH": str(workspace),
                # Third-party setuptools-entry-point pytest plugins (for example
                # anyio's, pulled in transitively by starlette) target newer
                # pytest internals than the fail_to_pass test's pinned
                # pytest==6.2.5 and crash it on collection. This project's own
                # test suite needs no such plugin, so disable autoloading them;
                # pytest's own built-in plugins are unaffected.
                "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            },
        )
        tests_passed = result.returncode == 0
        test_output = (result.stdout + result.stderr)[-2000:]
    except subprocess.TimeoutExpired:
        # A patch that hangs the suite is a failed fix, not a broken grader, so
        # it scores zero here instead of faulting the rollout.
        tests_passed = False
        test_output = f"fail_to_pass tests timed out after {GRADING_TIMEOUT_S}s."
    opened_pull_request = len(ROLLOUT.pull_requests) > 0
    reward = 0.8 * float(tests_passed) + 0.2 * float(opened_pull_request)
    # RLE's grader contract is `{reward, is_success, info}`. Only `reward` is
    # required; `is_success` is optional and reported here because it carries
    # signal the shaped reward does not. It tracks the tests alone, which are
    # what decide whether the reported issue is actually fixed, while the pull
    # request is the harness's second tool call and earns its own slice of the
    # reward above. RLE surfaces `info` as the caller's `result`, so anything
    # reported outside `info` is dropped.
    return {
        "reward": reward,
        "is_success": tests_passed,
        "info": {
            "reason": (
                f"fail_to_pass tests {'passed' if tests_passed else 'failed'}; "
                f"pull request {'opened' if opened_pull_request else 'missing'}."
            ),
            "instance_id": INSTANCE["instance_id"],
            "agent_response": agent_response,
            "test_output": test_output,
        },
    }
