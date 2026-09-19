"""RLE harness container for a real SWE-bench-Lite instance.

Wraps `psf/requests` at its pinned base commit (see `fixtures/instance.json`,
sourced from `princeton-nlp/SWE-bench_Lite`). The harness (`../agent`) is
given the real GitHub issue text and must edit a disposable checkout to fix
it, then open a pull request, exactly like the "code repair agent" example
in the RLE design docs -- but grounded in a real repo, real issue, and real
regression test rather than synthetic data.

Each rollout gets its own working copy of the repo under `/tmp/rollouts/`.
`reset()` seeds it from the read-only `/opt/base-repo` reference clone baked
into the image, then applies the hidden test patch (`fixtures/test_patch.diff`)
that adds the regression test the real fix must satisfy -- this patch is
never shown to the harness, only used for grading. `workspace.apply_patch`
mocks the production tool the harness uses to edit the checkout;
`github.create_pull_request` mocks submitting the fix for review.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from pydantic import Field

from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import (
    Action,
    EnvironmentMetadata,
    Observation,
    State,
)

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
BASE_REPO_DIR = Path("/opt/base-repo")
ROLLOUTS_DIR = Path("/tmp/rollouts")

INSTANCE: dict[str, Any] = json.loads((FIXTURES_DIR / "instance.json").read_text())
TEST_PATCH = (FIXTURES_DIR / "test_patch.diff").read_text()

_current_environment: Optional["CodeRepairEnvironment"] = None


class CodeRepairAction(Action):
    """Final agent response supplied to `/grade` at the end of a rollout."""

    message: str = Field(default="", description="The harness's final output_text.")


class CodeRepairObservation(Observation):
    """Rollout-visible messages returned by `/reset`."""

    messages: list[dict[str, Any]] = Field(default_factory=list)


class CodeRepairEnvironment(Environment[CodeRepairAction, CodeRepairObservation, State]):
    """One isolated attempt to fix `INSTANCE["instance_id"]`."""

    def __init__(self) -> None:
        super().__init__()
        self._state = State(episode_id=None, step_count=0)
        self._workspace: Optional[Path] = None
        self._pull_requests: dict[str, dict[str, Any]] = {}
        # max_concurrent_envs=1 below means this container ever hosts one
        # live instance at a time, so the module-level routes below can
        # reach this rollout's state through this singleton reference.
        global _current_environment
        _current_environment = self

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **task_data: Any,
    ) -> CodeRepairObservation:
        del seed, task_data
        episode_id = episode_id or str(uuid4())
        self._state = State(episode_id=episode_id, step_count=0)
        self._workspace = ROLLOUTS_DIR / episode_id
        if self._workspace.exists():
            shutil.rmtree(self._workspace)
        shutil.copytree(BASE_REPO_DIR, self._workspace)
        # The hidden regression test is applied here, not shipped to the
        # harness: /reset receives task-specific grading data that the
        # harness never sees, only the agent-visible task input below does.
        subprocess.run(
            ["git", "apply", "-"],
            input=TEST_PATCH,
            cwd=self._workspace,
            text=True,
            check=True,
        )
        self._pull_requests = {}
        return CodeRepairObservation(
            done=False,
            reward=None,
            messages=[{"role": "user", "content": INSTANCE["problem_statement"]}],
        )

    def step(
        self,
        action: CodeRepairAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> CodeRepairObservation:
        """Unused by real invocations: the harness calls `/tools/*` and RLE
        calls `/grade` directly. Kept only so the local `azd ai rle run`
        shell and OpenEnv's schema can still exercise this environment."""
        del action, timeout_s, kwargs
        self._state.step_count += 1
        return CodeRepairObservation(done=True, reward=None, messages=[])

    @property
    def state(self) -> State:
        return self._state

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="code_repair_agent",
            description=f"RLE harness for {INSTANCE['instance_id']}.",
            version="0.1.0",
        )


app = create_app(
    CodeRepairEnvironment,
    CodeRepairAction,
    CodeRepairObservation,
    env_name="code_repair_agent",
    max_concurrent_envs=1,
)


def _current_workspace() -> Path:
    if _current_environment is None or _current_environment._workspace is None:
        raise RuntimeError("No active rollout; call /reset first.")
    return _current_environment._workspace


@app.post("/tools/workspace.apply_patch")
async def workspace_apply_patch(arguments: dict[str, Any]) -> dict[str, Any]:
    """Mocks the production `workspace.apply_patch` tool the harness calls."""
    try:
        subprocess.run(
            ["git", "apply", "-"],
            input=arguments["patch"],
            cwd=_current_workspace(),
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
    _current_environment._pull_requests[arguments["branch"]] = {
        "title": arguments.get("title", ""),
        "body": arguments.get("body", ""),
    }
    return {"number": len(_current_environment._pull_requests)}


@app.post("/grade")
async def grade_rollout(rollout: dict[str, Any]) -> dict[str, Any]:
    """Runs the real regression test against the harness's edited checkout."""
    del rollout
    workspace = _current_workspace()
    result = subprocess.run(
        ["python", "-m", "pytest", *INSTANCE["fail_to_pass"], "-q"],
        cwd=workspace,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            # Third-party setuptools-entry-point pytest plugins (for example
            # anyio's, pulled in transitively by openenv/starlette) target
            # newer pytest internals than the fail_to_pass test's pinned
            # pytest==6.2.5 and crash it on collection. This project's own
            # test suite needs no such plugin, so disable autoloading them;
            # pytest's own built-in plugins are unaffected.
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        },
    )
    tests_passed = result.returncode == 0
    opened_pull_request = len(_current_environment._pull_requests) > 0
    reward = 0.8 * float(tests_passed) + 0.2 * float(opened_pull_request)
    return {
        "reward": reward,
        "reason": (
            f"fail_to_pass tests {'passed' if tests_passed else 'failed'}; "
            f"pull request {'opened' if opened_pull_request else 'missing'}."
        ),
        "test_output": (result.stdout + result.stderr)[-2000:],
    }

