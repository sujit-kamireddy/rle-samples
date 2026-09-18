"""``safe_grade``, extracted from the recipe's ``math_env.py``.

``math_env.py`` pulls in ``chz``, the Azure sessions SDK, ``datasets``,
``renderers`` and ``rl.problem_env`` for the parts of ``MathEnv`` this
environment doesn't need (episode/renderer plumbing) -- only ``safe_grade``
itself is required to grade an answer. Copied verbatim (this environment
installs no ``loom_cookbook``, see ``examples/gym/openenv/README.md``); the paired
recipe test in ``tests/test_env_grading_parity.py`` asserts the two agree.
"""

from __future__ import annotations

import logging
import math

from .math_grading import (
    grade_answer,
    grade_answer_math_verify,
    run_with_timeout_signal,
)

logger = logging.getLogger(__name__)


def safe_grade(given_answer: str, ground_truth: str, grader: str = "sympy", timeout: float = 1.0):
    if grader == "sympy":
        grader_func = grade_answer
    elif grader == "math_verify":
        grader_func = grade_answer_math_verify
    else:
        raise ValueError(f"Invalid grader: {grader}")
    out = run_with_timeout_signal(
        grader_func, args=(given_answer, ground_truth), timeout_seconds=int(math.ceil(timeout))
    )
    if out is None:
        logger.warning(f"Timeout grading {given_answer} against {ground_truth}")
        return False
    return out
