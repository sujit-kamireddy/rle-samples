"""JSONL dataset loading + episode-index picking shared by every recipe's
OpenEnv server.

Each recipe's ``build_dataset.py`` bakes ``train.jsonl``/``validation.jsonl``
into the Docker image at build time (see ``envs/README.md``, "Datasets are
baked in at build time, not downloaded per instance"). At server start, the
environment reads that same baked file straight off local disk -- no
network access, no per-instance download.
"""

from __future__ import annotations

import json
import random
from pathlib import Path


def load_jsonl(path: str | Path) -> list[dict]:
    """Read a JSONL file (one JSON object per line) into a list of dicts."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class EpisodePicker:
    """Picks the next dataset row for a ``reset()`` call.

    ``seed`` (if given) makes the pick reproducible *and* indexes into a
    fixed shuffled permutation of every row, rather than each seed
    independently drawing a uniformly random one. ``loom_cookbook/rl/rle_env.py``
    always calls ``pick()`` with a run of consecutive integer seeds for one
    epoch (``base_seed + i`` for ``i in range(num_examples)`` --
    see ``_RLEDataset``/``_RLEGroupBuilder``), so consecutive seeds now land
    on distinct permutation slots and therefore distinct rows: a real
    traversal of the dataset for any ``num_examples <= len(rows)``, wrapping
    around (and repeating the same permutation order) past that -- instead
    of the "balls into bins" behavior independent random draws would give,
    where roughly 1/e of rows go unpicked and others are duplicated within
    what's meant to be one epoch. ``seed=None`` (no reproducibility
    requested) still draws uniformly at random, matching the local
    (``use_rle=False``) recipe path's non-seeded episodes.
    """

    def __init__(self, rows: list[dict]):
        if not rows:
            raise ValueError("dataset is empty -- did build_dataset.py run at image build time?")
        self._rows = rows
        # Shuffled once, with a fixed internal seed, at load time -- not
        # reseeded per pick() call, and not derived from any seed a caller
        # passes in, so that seed % len(rows) below is a stable index into
        # one consistent permutation for the life of this picker.
        self._permutation = list(range(len(rows)))
        random.Random(0).shuffle(self._permutation)

    def __len__(self) -> int:
        return len(self._rows)

    def pick(self, seed: int | None = None) -> tuple[int, dict]:
        if seed is None:
            index = random.randrange(len(self._rows))
        else:
            index = self._permutation[seed % len(self._rows)]
        return index, self._rows[index]

    def by_index(self, index: int) -> dict:
        return self._rows[index % len(self._rows)]

    def row_for_episode(self, problem_id: str | None, fallback: dict | None = None) -> dict:
        """Resolve the episode's row, preferring the one the action names.

        ``fallback`` is what ``reset()`` stored on the environment, and is
        the normal path: ``openenv``'s own ``/ws`` route keeps one
        environment instance for the life of a connection (see
        ``envs/code_rl/server/code_rl_environment.py``), so ``step()``
        already knows which row it is grading.

        ``problem_id`` overrides it. Passing it lets a caller grade against
        a specific row, and keeps these environments correct behind any
        transport that does *not* carry state between calls -- the
        ``openenv`` package's HTTP surface builds a fresh ``Environment``
        per request, so an environment served that way reaches ``step()``
        with ``fallback=None``.
        """
        if problem_id is not None and str(problem_id) != "":
            try:
                return self.by_index(int(problem_id))
            except (TypeError, ValueError):
                raise ValueError(f"problem_id must be a row index, got {problem_id!r}") from None
        if fallback is not None:
            return fallback
        raise ValueError(
            "step() could not resolve the episode's row: the action carried no problem_id "
            "and this environment instance never served the matching reset(). Echo the "
            "observation's problem_id back in the action -- see envs/README.md."
        )
