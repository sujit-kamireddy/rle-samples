"""Normalize interactive and Foundry routine Responses input."""

from __future__ import annotations

import json
from typing import Any


def parse_response_input(value: str) -> tuple[str, str]:
    """Return topic and execution mode from Responses protocol input text."""
    text = value.strip()
    if not text:
        raise ValueError("Provide a non-empty topic.")

    try:
        decoded: Any = json.loads(text)
    except json.JSONDecodeError:
        decoded = text

    # RLE renders the environment's `reset` observation into this agent's
    # first turn as a one-element messages array, e.g.
    # `[{"role": "user", "content": "<json text>"}]`, because the renderer
    # always emits a list even for a single message. Unwrap it here so the
    # dict logic below -- written for the plain `{"topic": ..., ...}` shape --
    # still applies to the one real message inside.
    if (
        isinstance(decoded, list)
        and len(decoded) == 1
        and isinstance(decoded[0], dict)
        and isinstance(decoded[0].get("content"), str)
    ):
        content = decoded[0]["content"]
        try:
            decoded = json.loads(content)
        except json.JSONDecodeError:
            decoded = content

    if isinstance(decoded, str):
        return decoded.strip(), "interactive"
    if not isinstance(decoded, dict):
        raise ValueError("Responses input must be text or a JSON object encoded as text.")

    payload = decoded
    # `str(None)` is `"None"` -- a perfectly non-empty string. Stringifying
    # before the emptiness check let a payload with no recognised topic key
    # through as the literal topic "None", and the rollout then researched it:
    # five phases, a real reward, and nothing in the result to say the task had
    # never arrived. The value is checked before it is stringified.
    raw_topic = (
        payload.get("topic")
        or payload.get("message")
        or payload.get("query")
        # A bare `{"input": "..."}` payload, as a manual rollout invocation
        # might send, is accepted too.
        or payload.get("input")
    )
    topic = str(raw_topic).strip() if raw_topic is not None else ""
    if not topic:
        raise ValueError(
            "Provide a non-empty topic, as 'topic', 'message', 'query' or 'input'."
        )

    execution_mode = str(
        payload.get("execution_mode") or "interactive"
    ).strip().lower()
    if execution_mode not in {"interactive", "routine"}:
        raise ValueError("execution_mode must be 'interactive' or 'routine'.")
    return topic, execution_mode
