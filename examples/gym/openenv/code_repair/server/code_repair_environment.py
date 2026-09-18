"""``code_repair`` OpenEnv environment: one real SWE-bench-Lite instance per
episode, graded in a single ``step()`` call.

``reset()`` copies the pinned, read-only reference checkout baked into the
image (``/opt/base-repo``, see ``Dockerfile``) into a fresh working
directory, applies the instance's hidden regression-test patch
(``fixtures/test_patch.diff`` -- never shown to the policy, only used for
grading), and returns the real GitHub issue text
(``fixtures/instance.json``'s ``problem_statement``) as the episode's only
observation message.

``step(action)`` applies ``action.patch`` (the policy's whole proposed fix,
as a unified diff) to that same working directory via ``git apply``, then
runs the instance's real ``fail_to_pass`` regression test with ``pytest``.
Reward is ``1.0`` if the patch applies cleanly and the test then passes,
``0.0`` otherwise (whether from a patch that fails to apply, or one that
applies but doesn't fix the bug). The episode always ends here
(``done=True``) -- there is no follow-up turn, unlike the multi-turn
``examples/harness/{byoh,hosted-agent}`` "code repair" example this mirrors
for narrative parity (see ``models.py``'s module docstring for the
Gym-vs-Harness distinction).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import CodeRepairAction, CodeRepairObservation
except ImportError:  # pragma: no cover - standalone container import path
    from models import CodeRepairAction, CodeRepairObservation

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
BASE_REPO_DIR = Path(os.environ.get("CODE_REPAIR_BASE_REPO_DIR", "/opt/base-repo"))
ROLLOUTS_DIR = Path(os.environ.get("CODE_REPAIR_ROLLOUTS_DIR", "/tmp/rollouts"))

INSTANCE: dict[str, Any] = json.loads((FIXTURES_DIR / "instance.json").read_text())
TEST_PATCH = (FIXTURES_DIR / "test_patch.diff").read_text()

# Bounds how long a single grading step() may run -- comfortably below the
# RLE gateway's own request timeout, same rationale as the harness sample's
# /grade endpoint (see examples/harness/byoh/rle/server/env.py).
GRADING_TIMEOUT_S = float(os.environ.get("CODE_REPAIR_GRADING_TIMEOUT_S", "60"))


class CodeRepairEnvironment(Environment[CodeRepairAction, CodeRepairObservation, State]):
    """One isolated attempt to fix ``INSTANCE["instance_id"]``, graded on the
    single ``step()`` call that ends the episode."""

    def __init__(self) -> None:
        super().__init__()
        self._state = State(episode_id=None, step_count=0)
        self._workspace: Optional[Path] = None

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> CodeRepairObservation:
        del seed, kwargs
        episode_id = episode_id or str(uuid4())
        self._state = State(episode_id=episode_id, step_count=0)
        self._workspace = ROLLOUTS_DIR / episode_id
        if self._workspace.exists():
            shutil.rmtree(self._workspace)
        self._workspace.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(BASE_REPO_DIR, self._workspace)
        # The hidden regression test is applied here, never sent to the
        # policy -- only fixtures/instance.json's problem_statement is
        # policy-visible, in the returned observation below.
        subprocess.run(
            ["git", "apply", "-"],
            input=TEST_PATCH,
            cwd=self._workspace,
            text=True,
            check=True,
        )
        return CodeRepairObservation(
            done=False,
            reward=None,
            messages=[{"role": "user", "content": INSTANCE["problem_statement"]}],
            instance_id=INSTANCE["instance_id"],
            episode_id=episode_id,
        )

    def step(
        self,
        action: CodeRepairAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> CodeRepairObservation:
        del timeout_s, kwargs
        self._state.step_count += 1
        workspace = self._workspace
        if workspace is None:
            # Plain HTTP /reset+/step build a fresh environment instance per
            # request (see models.py's CodeRepairAction.episode_id docstring),
            # so this instance never saw its own reset() -- fall back to the
            # workspace the client's echoed episode_id names. Both land on the
            # same on-disk directory either way, since ROLLOUTS_DIR is shared
            # by every instance in this container.
            if not action.episode_id:
                raise RuntimeError(
                    "No active rollout for this instance and no episode_id "
                    "was given; call reset() first and echo its episode_id "
                    "in step()'s action."
                )
            workspace = ROLLOUTS_DIR / action.episode_id
            if not workspace.is_dir():
                raise RuntimeError(f"No rollout workspace for episode_id={action.episode_id!r}.")

        apply_result = subprocess.run(
            ["git", "apply", "-"],
            input=action.patch,
            cwd=workspace,
            text=True,
            capture_output=True,
        )
        if apply_result.returncode != 0:
            return CodeRepairObservation(
                done=True,
                reward=0.0,
                messages=[],
                instance_id=INSTANCE["instance_id"],
                episode_id=action.episode_id or self._state.episode_id,
                metadata={"applied": False, "error": apply_result.stderr},
            )

        test_result = subprocess.run(
            ["python", "-m", "pytest", *INSTANCE["fail_to_pass"], "-q"],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=GRADING_TIMEOUT_S,
            env={
                **os.environ,
                # See Dockerfile / sitecustomize.py: third-party
                # setuptools-entry-point pytest plugins (e.g. anyio's,
                # pulled in transitively by openenv/starlette) target newer
                # pytest internals than this instance's pinned
                # pytest==6.2.5 and crash it on collection.
                "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            },
        )
        passed = test_result.returncode == 0
        return CodeRepairObservation(
            done=True,
            reward=1.0 if passed else 0.0,
            messages=[],
            instance_id=INSTANCE["instance_id"],
            episode_id=action.episode_id or self._state.episode_id,
            metadata={
                "applied": True,
                "passed": passed,
                "test_output": (test_result.stdout + test_result.stderr)[-2000:],
            },
        )

    @property
    def state(self) -> State:
        return self._state
