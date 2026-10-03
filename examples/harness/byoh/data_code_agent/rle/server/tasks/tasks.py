"""Task dataset loading for the data-code-agent OpenEnv episodes."""

from __future__ import annotations

import gzip
import json
import logging
import math
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict, StrictStr, ValidationError

_LOG = logging.getLogger(__name__)
_TASK_META_DIR = Path(__file__).parent / "task-meta"


class _Task(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    task_name: StrictStr
    question: StrictStr
    expected_answer: StrictStr
    reward_mode: str = ""
    atol: float
    rtol: float
    has_pii: bool


@lru_cache(maxsize=8)
def load_tasks(split: str) -> tuple[_Task, ...]:
    paths = {
        path.name.removesuffix(".json.gz").replace("__", "/"): path
        for path in _TASK_META_DIR.glob("*.json.gz")
    }
    if split not in paths:
        raise ValueError("Unsupported split")
    try:
        with gzip.open(paths[split], "rt", encoding="utf-8") as handle:
            rows = json.load(handle)
        tasks = tuple(
            _Task.model_validate(
                {
                    **row,
                    "reward_mode": row.get("reward_mode") or "",
                    "atol": float(row.get("atol") or 1e-3),
                    "rtol": float(row.get("rtol") or 1e-3),
                }
            )
            for row in rows
        )
        if not tasks or any(
            not task.question.strip()
            or not math.isfinite(task.atol)
            or not math.isfinite(task.rtol)
            for task in tasks
        ):
            raise ValueError("Invalid task records")
    except (OSError, ValueError, TypeError, AttributeError, ValidationError):
        # Validation errors can contain private answer-key input values.
        _LOG.error("Unable to load validated OpenEnv task metadata")
        raise RuntimeError("Task metadata is unavailable or invalid") from None
    return tasks
