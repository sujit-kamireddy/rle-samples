"""Structured, rollout-correlated logging.

Cloud Run (and Cloud Logging generally) treats a JSON object written to stdout as a *structured*
log entry: reserved keys like ``severity`` and ``message`` populate the entry's own fields, and
everything else lands in ``jsonPayload``, filterable in Logs Explorer. That is the whole point of
this module -- a demo that runs ``azd ai rle rollout`` in one pane and tails Cloud Run logs in
another needs every line either side prints to carry the same ``rollout_id``, so
``jsonPayload.rollout_id="<id>"`` reconstructs one rollout's timeline across both.

``operation_id`` rides along too: it is what ``app.py`` keys its own state on (see its docstring),
so a log line missing it is one that could not be tied back to a poll or withdraw call either.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any

_CORRELATION_FIELDS = ("rollout_id", "operation_id", "event", "elapsed_ms")


class JsonFormatter(logging.Formatter):
    """Renders one log record as one JSON line.

    Only ``_CORRELATION_FIELDS`` and an explicit ``fields`` dict (passed via ``extra``) are
    promoted into the payload -- an allowlist, not everything on the record, so a stray ``extra=``
    elsewhere in the codebase cannot silently widen what this emits.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "severity": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }
        for name in _CORRELATION_FIELDS:
            value = getattr(record, name, None)
            if value is not None:
                payload[name] = value
        extra_fields = getattr(record, "fields", None)
        if extra_fields:
            payload.update(extra_fields)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure(level: int = logging.INFO) -> None:
    """Points the root logger at stdout with `JsonFormatter`.

    Idempotent: called once from `app.py` at import time, but tests import `app` repeatedly in the
    same process, and a second `addHandler` would duplicate every line.
    """
    root = logging.getLogger()
    if any(isinstance(h.formatter, JsonFormatter) for h in root.handlers):
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.handlers = [handler]
    root.setLevel(level)


class RolloutLog:
    """Emits one correlated, elapsed-time-stamped event per lifecycle stage.

    Mirrors the shape `azd ai rle rollout` itself prints (`[  2m7.7s] Working: ...` /
    `(✓) Done: ...`) so the two timelines read the same way side by side, one in the CLI pane, one
    in Cloud Run logs -- that pairing is the point, not a coincidence.
    """

    def __init__(self, logger: logging.Logger, rollout_id: str, operation_id: str = "") -> None:
        self._logger = logger
        self._rollout_id = rollout_id
        self._operation_id = operation_id
        self._start = time.monotonic()

    def event(self, name: str, level: int = logging.INFO, **fields: Any) -> None:
        elapsed_ms = int((time.monotonic() - self._start) * 1000)
        self._logger.log(
            level,
            name,
            extra={
                "rollout_id": self._rollout_id,
                "operation_id": self._operation_id,
                "event": name,
                "elapsed_ms": elapsed_ms,
                "fields": fields,
            },
        )
