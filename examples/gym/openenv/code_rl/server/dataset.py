"""JSONL dataset loading + episode-index picking for the ``code_rl`` server.

The sample ships a compact, gzip-compressed ``train.jsonl``/``validation.jsonl``
snapshot. At server start, the environment reads it from local disk -- no
network access or per-instance download.
"""

from __future__ import annotations

import gzip
import json
import random
from pathlib import Path


def load_jsonl(path: str | Path) -> list[dict]:
    """Read a JSONL file (one JSON object per line) into a list of dicts."""
    rows = []
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def reject_unknown_selectors(selectors: dict) -> None:
    """Raise if ``reset`` was handed a selector this environment does not understand.

    Call this from ``reset`` with its trailing ``**kwargs``. The signature has to end in
    ``**kwargs`` for this to see anything: OpenEnv's ``ResetRequest`` is ``extra="allow"``
    and its server then filters the payload down to the parameters ``reset`` actually
    declares (``_get_valid_kwargs`` in ``core/env_server/http_server.py``), so a selector
    this environment never declared is dropped twice over before ``reset`` is entered.

    Dropping it is the dangerous outcome, which is why this is loud. RLE passes the job's
    task through to ``reset`` verbatim and cannot validate it -- ``GET /schema`` describes
    actions, observations and state, never the reset signature -- so a job that names its
    episode some other way (``{"instance_id": ...}`` rather than ``{"seed": ...}``) leaves
    ``seed`` at ``None``, and ``EpisodePicker.pick`` then draws a RANDOM episode. The
    rollout completes, the grade is real, and it belongs to a different episode than the
    one the job asked for. Nothing in the response marks it wrong.
    """
    if selectors:
        raise ValueError(
            f"Unknown reset selector(s): {sorted(selectors)}. This environment selects an "
            f"episode with 'seed' and 'split'; pass the episode's identity as a seed, not "
            f"as its content."
        )


class EpisodePicker:
    """Picks the next dataset row for a ``reset()`` call.

    ``seed`` (if given) makes the pick reproducible *and* indexes into a
    fixed shuffled permutation of every row, rather than each seed
    independently drawing a uniformly random one. A training client is
    expected to call ``pick()`` with a run of consecutive integer seeds for
    one epoch (``base_seed + i`` for ``i in range(num_examples)``), so
    consecutive seeds now land on distinct permutation slots and therefore
    distinct rows: a real traversal of the dataset for any
    ``num_examples <= len(rows)``, wrapping around (and repeating the same
    permutation order) past that -- instead of the "balls into bins"
    behavior independent random draws would give, where roughly 1/e of rows
    go unpicked and others are duplicated within what's meant to be one
    epoch. ``seed=None`` (no reproducibility requested) still draws
    uniformly at random.
    """

    def __init__(self, rows: list[dict]):
        if not rows:
            raise ValueError("dataset is empty -- the compact snapshot may be missing or invalid")
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
