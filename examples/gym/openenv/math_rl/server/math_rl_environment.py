"""``math_rl`` OpenEnv environment: one Hendrycks MATH problem per episode.

``reset()`` selects a problem, ``step()`` grades the submitted answer and
ends the episode. Grading runs in this container using ``safe_grade``
(sympy / math-verify, see ``grading.py``) -- a self-contained port of
``loom_cookbook.recipes.math_rl.math_env``'s grading code, kept in sync by
hand rather than by installing ``loom_cookbook`` itself (see
``examples/gym/openenv/README.md``).

The reward *composition* mirrors the same recipe by hand: ``FORMAT_COEF``
below must equal ``loom_cookbook.rl.problem_env.ProblemEnv``'s
``format_coef`` default, and the ``format``/``correct`` -> reward formula
must match ``MathEnv.step()`` (with ``forced_commit`` disabled, which is
this environment's only mode) -- a submission without ``\\boxed{}`` scores
``-FORMAT_COEF`` without attempting to grade the raw text, exactly like the
in-process path.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from .dataset import EpisodePicker, load_jsonl
from .grading import extract_boxed, safe_grade

try:
    from ..models import MathAction, MathObservation
except ImportError:  # pragma: no cover - standalone container import path
    from models import MathAction, MathObservation


# Must equal loom_cookbook.rl.problem_env.ProblemEnv's format_coef default --
# see the module docstring.
FORMAT_COEF = 0.1


class MathRLEnvironment(Environment[MathAction, MathObservation, State]):
    """One-shot math-answer environment: reset -> problem, step -> reward.

    A real ``openenv`` ``Environment`` subclass -- see
    ``examples/gym/openenv/code_rl/server/code_rl_environment.py``'s class docstring for
    why holding the episode's row on ``self`` is safe under ``openenv``'s
    own ``/ws`` per-connection session model.
    """

    def __init__(
        self,
        dataset_path: Optional[str] = None,
        grader: str = "sympy",
        grading_timeout: float = 1.0,
        validation_dataset_path: Optional[str] = None,
    ):
        super().__init__()
        default_dataset_path = Path(__file__).resolve().parents[1] / "env_data" / "train.jsonl.gz"
        dataset_path = dataset_path or os.environ.get("MATH_RL_DATASET_PATH", str(default_dataset_path))
        self._rows = EpisodePicker(load_jsonl(dataset_path))
        # Resolved lazily (see _rows_for_split()), not loaded here: most
        # callers (including every existing test) construct this with only
        # a train file on disk and never request split="validation", and
        # eagerly loading here would make that an unconditional
        # FileNotFoundError. Defaults to the matching sibling validation file
        # for either plain JSONL or the checked-in gzip snapshot.
        default_validation_name = (
            "validation.jsonl.gz" if str(dataset_path).endswith(".gz") else "validation.jsonl"
        )
        self._validation_dataset_path = validation_dataset_path or os.environ.get(
            "MATH_RL_VALIDATION_DATASET_PATH",
            str(Path(dataset_path).with_name(default_validation_name)),
        )
        self._validation_rows: Optional[EpisodePicker] = None
        self._grader = grader
        self._grading_timeout = grading_timeout
        self._state = State(episode_id=None, step_count=0)
        self._current_row: Optional[dict] = None

    def _rows_for_split(self, split: str) -> EpisodePicker:
        """Resolve which baked dataset file a ``reset()`` picks from.

        ``split="validation"`` picks from ``validation.jsonl`` (loaded
        lazily and cached on first use) instead of the training file --
        without this, "validation" metrics were actually computed against
        different rows of the *same* training file.
        """
        if split == "train":
            return self._rows
        if split == "validation":
            if self._validation_rows is None:
                self._validation_rows = EpisodePicker(load_jsonl(self._validation_dataset_path))
            return self._validation_rows
        raise ValueError(f"Unknown split {split!r}; expected 'train' or 'validation'.")

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        split: str = "train",
        **kwargs: Any,
    ) -> MathObservation:
        index, row = self._rows_for_split(split).pick(seed)
        self._current_row = row
        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            extra={"row_index": index},
        )
        return MathObservation(
            done=False,
            reward=None,
            messages=row["messages"],
            problem_id=str(index),
        )

    def step(
        self,
        action: MathAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> MathObservation:
        if action.type == "list_tools":
            # Stand-in for OpenEnv's own ListToolsAction handling -- see
            # MathAction's docstring in ../models.py for why this env has to answer
            # this itself. This env exposes no tools, so the answer is just empty.
            return MathObservation(done=False, reward=None, messages=[], tools=[])

        self._state.step_count += 1
        row = self._rows.row_for_episode(action.problem_id, self._current_row)

        try:
            given = extract_boxed(action.answer_text or "")
            has_format = True
        except ValueError:
            # No \boxed{} in the submission. The in-process path (MathEnv.step
            # with forced_commit disabled) never grades the raw text in this
            # case -- it forces correct=0 -- so neither do we.
            given = None
            has_format = False

        correct = False
        if has_format:
            reference = extract_boxed(row["solution"])
            correct = safe_grade(
                given, reference, grader=self._grader, timeout=self._grading_timeout
            )

        format_score = 1.0 if has_format else 0.0
        correct_score = 1.0 if correct else 0.0
        reward = FORMAT_COEF * (format_score - 1.0) + correct_score

        return MathObservation(
            done=True,
            reward=reward,
            messages=[],
            metadata={"correct": correct, "format": has_format},
        )

    @property
    def state(self) -> State:
        return self._state
