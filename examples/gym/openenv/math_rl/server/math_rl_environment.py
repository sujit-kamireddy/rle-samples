"""SDK MCP math environment: one Hendrycks MATH problem per episode.

``reset()`` selects a problem, the optional ``check_equivalence`` MCP tool
checks two model-supplied expressions, and ``grade()`` grades the final
``GradeAction.response``. Grading runs in this container using ``safe_grade``.

A submission without ``\\boxed{}`` scores ``-FORMAT_COEF`` (below) without
attempting to grade the raw text: this rewards "submitted a correctly
formatted final answer" strictly more than "got lucky with unformatted
text", while still keeping the format penalty small next to the correctness
signal.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from azure.ai.projects.rle.environments import GradeAction, RLEnvironment

from .dataset import EpisodePicker, load_jsonl, reject_unknown_selectors
from .grading import extract_boxed, normalize_answer, safe_grade
from .models import MathObservation, MathState


# The penalty applied when a submission has no `\boxed{...}` answer to grade
# -- see the module docstring.
FORMAT_COEF = 0.1


class MathRLEnvironment(RLEnvironment):
    """Dataset-backed math environment with an optional equivalence tool."""

    def __init__(
        self,
        dataset_path: Optional[str] = None,
        grader: str = "sympy",
        grading_timeout: float = 1.0,
        validation_dataset_path: Optional[str] = None,
    ):
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
        self._instance_id = str(uuid4())
        self._current_row: Optional[dict] = None
        super().__init__()
        self._set_state(MathState(instance_id=self._instance_id))
        self.tool()(self.check_equivalence)

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
        reject_unknown_selectors(kwargs)
        index, row = self._rows_for_split(split).pick(seed)
        self._current_row = row
        self._set_state(
            MathState(
                instance_id=self._instance_id,
                episode_id=episode_id or str(uuid4()),
                step_count=0,
                row_index=index,
                split=split,
                active=True,
            )
        )
        return MathObservation(
            done=False,
            reward=None,
            messages=row["messages"],
            problem_id=str(index),
        )

    def check_equivalence(
        self,
        candidate_expression: str,
        comparison_expression: str,
    ) -> dict[str, bool | int | str]:
        """Safely check whether two model-supplied expressions are equivalent.

        This helper never receives or reveals the dataset's reference answer.
        It is useful for checking a simplification before final submission.
        """
        state = self.state
        if not state.active or state.graded or self._current_row is None:
            raise RuntimeError("An active ungraded episode is required")
        if not candidate_expression.strip() or not comparison_expression.strip():
            raise ValueError("Both expressions must be non-empty")

        equivalent = bool(
            safe_grade(
                candidate_expression,
                comparison_expression,
                grader=self._grader,
                timeout=self._grading_timeout,
            )
        )
        state.helper_called = True
        state.helper_call_count += 1
        state.last_candidate_expression = candidate_expression
        state.last_comparison_expression = comparison_expression
        state.last_equivalent = equivalent
        self._set_state(state)
        return {
            "equivalent": equivalent,
            "candidate_normalized": normalize_answer(candidate_expression) or "",
            "comparison_normalized": normalize_answer(comparison_expression) or "",
            "helper_call_count": state.helper_call_count,
            "instance_id": self._instance_id,
        }

    def grade(
        self,
        action: GradeAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> MathObservation:
        if kwargs:
            raise ValueError(f"Unsupported grade option(s): {sorted(kwargs)}")
        state = self.state
        if not state.active or state.graded:
            raise RuntimeError("An active ungraded episode is required")
        if self._current_row is None:
            raise ValueError("grade() called before reset(): no row to grade against.")
        row = self._current_row

        try:
            given = extract_boxed(action.response or "")
            has_format = True
        except ValueError:
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

        state.graded = True
        state.active = False
        self._set_state(state)
        return MathObservation(
            done=True,
            reward=reward,
            messages=[],
            helper_called=state.helper_called,
            helper_call_count=state.helper_call_count,
            metadata={
                "correct": correct,
                "format": has_format,
                "helper_called": state.helper_called,
                "helper_call_count": state.helper_call_count,
            },
        )
