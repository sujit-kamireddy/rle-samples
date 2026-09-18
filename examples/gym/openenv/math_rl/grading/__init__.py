"""Grading for the ``math_rl`` environment, vendored from the recipe.

See ``examples/gym/openenv/README.md``: environments install no ``loom_cookbook``, so the
grading logic lives here instead.
"""

from examples.gym.openenv.math_rl.grading.math_env import safe_grade
from examples.gym.openenv.math_rl.grading.math_grading import extract_boxed

__all__ = [
    "extract_boxed",
    "safe_grade",
]
