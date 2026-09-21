"""``code_repair`` OpenEnv environment: one real SWE-bench-Verified instance
per episode, graded in a single ``step()`` call.

``reset(seed, split)`` picks a row from the baked ``env_data/`` snapshot (40
``django/django`` version-``3.2`` instances -- see ``env_data/NOTICE.md``)
the same way ``math_rl``/``code_rl`` do, then materializes that row's
``base_commit`` as a fresh working directory via
``git worktree add --detach <path> <base_commit>`` against the read-only
``django/django`` clone baked into the image (``/opt/base-repo``, see
``Dockerfile``) -- not a full copy, and no network access, since the clone
already has every commit's blobs local. It then applies the instance's
hidden regression-test patch (``test_patch`` -- never shown to the policy,
only used for grading) and returns the real GitHub issue text
(``problem_statement``) as the episode's only observation message.

``step(action)`` applies ``action.patch`` (the policy's whole proposed fix,
as a unified diff) to that same working directory via ``git apply``, then
runs the instance's real ``fail_to_pass`` regression test(s) with Django's
own ``tests/runtests.py`` (no ``pip install -e .`` needed -- that script
puts the checkout on ``sys.path`` itself). Reward is ``1.0`` if the patch
applies cleanly and the test(s) then pass, ``0.0`` otherwise (whether from
a patch that fails to apply, or one that applies but doesn't fix the bug).
The episode always ends here (``done=True``) -- there is no follow-up
turn. The worktree is removed after grading either way, since each episode
gets its own and nothing outside this step() needs it again.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from .dataset import EpisodePicker, load_jsonl, reject_unknown_selectors
from .schema import CodeRepairAction, CodeRepairObservation

BASE_REPO_DIR = Path(os.environ.get("CODE_REPAIR_BASE_REPO_DIR", "/opt/base-repo"))
ROLLOUTS_DIR = Path(os.environ.get("CODE_REPAIR_ROLLOUTS_DIR", "/tmp/rollouts"))

# Bounds how long a single grading step() may run -- comfortably below the
# RLE gateway's own request timeout, same rationale as the harness sample's
# /grade endpoint (see examples/harness/byoh/code_repair/rle/server/env.py). Django's
# test runner (migrations + app loading on every invocation) is slower to
# start than the single pytest-collected test this replaces, hence the
# larger default than other samples in this repo.
GRADING_TIMEOUT_S = float(os.environ.get("CODE_REPAIR_GRADING_TIMEOUT_S", "120"))


class CodeRepairEnvironment(Environment[CodeRepairAction, CodeRepairObservation, State]):
    """One isolated attempt to fix a randomly-picked SWE-bench instance,
    graded on the single ``step()`` call that ends the episode.

    A real ``openenv`` ``Environment`` subclass -- see
    ``examples/gym/openenv/code_rl/server/code_rl_environment.py``'s class
    docstring for why holding the episode's row on ``self`` is safe under
    ``openenv``'s own ``/ws`` per-connection session model.
    """

    def __init__(
        self,
        dataset_path: Optional[str] = None,
        validation_dataset_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        default_dataset_path = Path(__file__).resolve().parents[1] / "env_data" / "train.jsonl.gz"
        dataset_path = dataset_path or os.environ.get("CODE_REPAIR_DATASET_PATH", str(default_dataset_path))
        self._rows = EpisodePicker(load_jsonl(dataset_path))
        default_validation_name = (
            "validation.jsonl.gz" if str(dataset_path).endswith(".gz") else "validation.jsonl"
        )
        self._validation_dataset_path = validation_dataset_path or os.environ.get(
            "CODE_REPAIR_VALIDATION_DATASET_PATH",
            str(Path(dataset_path).with_name(default_validation_name)),
        )
        self._validation_rows: Optional[EpisodePicker] = None
        self._state = State(episode_id=None, step_count=0)
        self._current_row: Optional[dict] = None
        self._workspace: Optional[Path] = None

    def _rows_for_split(self, split: str) -> EpisodePicker:
        if split == "train":
            return self._rows
        if split == "validation":
            if self._validation_rows is None:
                self._validation_rows = EpisodePicker(load_jsonl(self._validation_dataset_path))
            return self._validation_rows
        raise ValueError(f"Unknown split {split!r}; expected 'train' or 'validation'.")

    def _make_workspace(self, episode_id: str, row: dict) -> Path:
        workspace = ROLLOUTS_DIR / episode_id
        _remove_worktree(workspace)
        workspace.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "worktree", "add", "--detach", "--quiet", str(workspace), row["base_commit"]],
            cwd=BASE_REPO_DIR,
            check=True,
            capture_output=True,
            text=True,
        )
        # The hidden regression test is applied here, never sent to the
        # policy -- only the row's problem_statement is policy-visible, in
        # the returned observation.
        subprocess.run(
            ["git", "apply", "-"],
            input=row["test_patch"],
            cwd=workspace,
            text=True,
            check=True,
        )
        return workspace

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        split: str = "train",
        **kwargs: Any,
    ) -> CodeRepairObservation:
        reject_unknown_selectors(kwargs)
        index, row = self._rows_for_split(split).pick(seed)
        self._current_row = row
        episode_id = episode_id or str(uuid4())
        self._workspace = self._make_workspace(episode_id, row)
        self._state = State(episode_id=episode_id, step_count=0, extra={"row_index": index})
        return CodeRepairObservation(
            done=False,
            reward=None,
            messages=[{"role": "user", "content": row["problem_statement"]}],
            instance_id=row["instance_id"],
            problem_id=str(index),
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
        row = self._current_row
        workspace = self._workspace
        if row is None or workspace is None or not workspace.is_dir():
            # Only reachable if step() is called before reset() -- this env is
            # only served over /ws, so the row/workspace reset() set up are
            # always still on self.
            raise RuntimeError("step() called before reset(): no active rollout for this instance.")

        episode_id = self._state.episode_id
        try:
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
                    instance_id=row["instance_id"],
                    problem_id=str(self._state.extra.get("row_index", "")),
                    episode_id=episode_id,
                    metadata={"applied": False, "error": apply_result.stderr},
                )

            test_result = subprocess.run(
                [
                    "python",
                    "tests/runtests.py",
                    "--settings=test_sqlite",
                    "--verbosity",
                    "2",
                    *row["fail_to_pass"],
                ],
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=GRADING_TIMEOUT_S,
                env={**os.environ, "PYTHONPATH": str(workspace)},
            )
            passed = test_result.returncode == 0
            return CodeRepairObservation(
                done=True,
                reward=1.0 if passed else 0.0,
                messages=[],
                instance_id=row["instance_id"],
                problem_id=str(self._state.extra.get("row_index", "")),
                episode_id=episode_id,
                metadata={
                    "applied": True,
                    "passed": passed,
                    "test_output": (test_result.stdout + test_result.stderr)[-2000:],
                },
            )
        finally:
            # Single-turn: nothing else will use this episode's workspace
            # again, so free it (and the worktree metadata pointing at it)
            # right away rather than waiting for the next reset() to notice
            # it's stale.
            _remove_worktree(workspace)
            if self._workspace == workspace:
                self._workspace = None

    @property
    def state(self) -> State:
        return self._state


def _remove_worktree(path: Path) -> None:
    """Best-effort cleanup of a per-episode ``git worktree`` checkout.

    ``git worktree remove`` (not a plain ``rm -rf``) is required so
    ``BASE_REPO_DIR``'s ``.git/worktrees/`` metadata doesn't accumulate
    stale entries across episodes -- see the module docstring.
    """
    if not path.exists():
        return
    result = subprocess.run(
        ["git", "worktree", "remove", "--force", str(path)],
        cwd=BASE_REPO_DIR,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        # Fall back to a plain removal so a stray episode directory can't
        # wedge future reset()s onto the same episode_id; prune the now-
        # dangling worktree metadata entry separately.
        shutil.rmtree(path, ignore_errors=True)
        subprocess.run(
            ["git", "worktree", "prune"],
            cwd=BASE_REPO_DIR,
            capture_output=True,
            text=True,
        )
