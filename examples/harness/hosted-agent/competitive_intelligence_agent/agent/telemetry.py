"""Rollout telemetry for the hosted agent.

A rollout spends most of its wall clock inside this container, and from the
outside that is a single opaque pause: ``azd ai rle rollout`` prints
``Working: Running the rollout.`` and nothing else until a reward appears.
Everything that decides the reward -- which tools the policy reached for, what
came back, where the latency went -- happens here and is otherwise invisible.

The stream this writes to is stdout, which the hosted-agent runtime collects and
``azd ai agent monitor --session-id <id> --follow`` replays. The id that selects
it is the rollout id: RLE sets ``agent_session_id`` to the rollout id when it
invokes the agent, so the id the CLI prints is the id that opens this stream.
That shared id is what lets the two be read side by side.

This module is deliberately free of the agent-server and model dependencies so
the logging behaviour can be tested on its own.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from typing import Any, Optional

#: One logger for the whole agent, so a single re-assertion re-enables every
#: line the rollout writes, wherever it was written from.
LOGGER_NAME = "competitive-intel-rle-agent"

logger = logging.getLogger(LOGGER_NAME)


def configure(stream: Any = None) -> None:
    """Points this agent's telemetry at stdout and keeps it there.

    Without this, nothing is emitted at all: the root logger defaults to
    WARNING, so every INFO line the rollout writes is dropped before it reaches
    the runtime.

    Owning a handler is not enough to keep it. A host that calls
    ``logging.config.dictConfig`` on startup silences this logger outright --
    ``disable_existing_loggers`` defaults to true and sets ``disabled`` on every
    logger the new configuration does not name, whatever handlers it owns. That
    is why this is a function rather than a block of module-level statements:
    it is re-asserted per request, by which point the host has finished
    starting, and it costs a few attribute assignments.

    Args:
        stream (*optional*):
            Where to write. Defaults to ``sys.stdout``, the stream the runtime
            collects.
    """
    handler = logging.StreamHandler(stream=stream if stream is not None else sys.stdout)
    # No timestamp and no level. `azd ai agent monitor` already prefixes every
    # streamed line with its own clock and stream name, so a second copy cost
    # 15 columns on every line and told a reader nothing twice. In a split
    # terminal those columns are the difference between a line that fits and a
    # line that wraps. WARNING lines mark themselves in their text.
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.handlers[:] = [handler]
    logger.setLevel(os.environ.get("RLE_LOG_LEVEL", "INFO").upper())
    logger.propagate = False
    logger.disabled = False
    # httpx logs one INFO line per tool call already, and openai one per model
    # call. Ours carry the rollout tag and the timing, so keep theirs out of the
    # stream rather than doubling every line in it.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    # The container's own libraries produce four or five lines for every one of
    # ours: the managed-identity credential logs three INFO lines to stderr each
    # time it hands out a token (once per model call), the telemetry exporter
    # announces every five seconds that it had nothing to export, and the agent
    # host narrates each inbound request. Measured on one rollout, 71 of 317
    # streamed lines were these and 39 were the rollout's. A demo pane that is
    # 12% signal is not a demo pane. Failures still surface: this only lifts the
    # floor to WARNING, it does not silence these libraries.
    for noisy in (
        "azure.identity",
        "azure.core",
        "azure.monitor",
        "azure.ai.agentserver",
        "microsoft.opentelemetry",
        "opentelemetry",
        "urllib3",
        "hypercorn.access",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def short(rollout_id: Optional[str]) -> str:
    """The first 8 characters of the rollout id, or a marker when there is none.

    The marker is returned whole: truncating it to the same width produces
    ``no-rollo``, which reads like a real id and hides the thing it exists to
    report -- that this request arrived without rollout headers.
    """
    if not rollout_id:
        return "no-rollout"
    return rollout_id[:8]


def endpoint(url: Optional[str]) -> str:
    """A URL safe to put on a screen.

    Query strings on these endpoints can carry SAS-style credentials, so only
    the scheme, host and path survive. The bearer tokens the agent sends are in
    headers, never in the URL, and are never logged.
    """
    if not url:
        return "(none)"
    return url.split("?", 1)[0]


def brief(url: Optional[str], keep: int = 2) -> str:
    """A URL narrowed to the part that identifies it, for a half-width pane.

    The sandbox tool endpoint is a 400-character ARM resource path. Printed on
    every tool call it wrapped six times in a split terminal, so the rollout's
    own story scrolled past at a fraction of the rate it was written. Only the
    host's first two labels and the last few path segments survive -- enough to
    tell the capture proxy from the sandbox, which is all a reader needs.

    Applies `endpoint` first, so this is also query-string safe.
    """
    safe = endpoint(url)
    if safe == "(none)":
        return safe
    scheme, _, rest = safe.partition("://")
    if not rest:
        rest, scheme = safe, ""
    host, _, path = rest.partition("/")
    labels = host.split(".")
    host_short = ".".join(labels[:2]) + "\u2026" if len(labels) > 2 else host
    segments = [s for s in path.split("/") if s]
    if len(segments) <= keep:
        return host_short + ("/" + "/".join(segments) if segments else "")
    tail = "/".join(s if len(s) <= 16 else s[:8] + "\u2026" for s in segments[-keep:])
    return f"{host_short}/{tail}"


def compact(arguments: dict[str, Any], limit: int = 120) -> str:
    """Tool arguments on one line, truncated to stay readable in a log stream."""
    try:
        rendered = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        rendered = repr(arguments)
    return rendered if len(rendered) <= limit else rendered[: limit - 1] + "\u2026"


def digest(output: str, limit: int = 90) -> str:
    """What a tool actually returned, in one short phrase.

    A byte count says a tool answered; it does not say whether the rollout
    found its evidence. This is the difference between watching a rollout and
    watching a progress bar.

    Two shapes matter here. A list of records is evidence, so it is counted and
    its first item named. A single governed record is a *finding* -- the
    portfolio row whose exposure score and materiality threshold are the whole
    decision -- so its numbers are shown rather than its key names.

    Falls back to a truncated rendering whenever the payload is not JSON or not
    a shape this recognises -- a log line is never worth an exception.
    """
    try:
        parsed = json.loads(output)
    except (TypeError, ValueError):
        flat = " ".join(output.split())
        return _clip(flat, limit)

    if isinstance(parsed, dict):
        for key, value in parsed.items():
            if isinstance(value, list) and value:
                if len(value) == 1:
                    return _clip(_record(value[0]) or f"1 {key}", limit)
                head = _title(value[0])
                return _clip(f"{len(value)} {key}" + (f": {head}" if head else ""), limit)
        # An empty list means the lookup resolved nothing, and the outer record
        # carries the reason ("Exposure is unknown, not zero"). That is a
        # finding too, so fall through to the record's own fields.
        return _clip(_record(parsed) or "{}", limit)
    if isinstance(parsed, list):
        head = _title(parsed[0]) if parsed else ""
        return _clip(f"{len(parsed)} item(s)" + (f": {head}" if head else ""), limit)
    return _clip(str(parsed), limit)


def clip(text: str, limit: int) -> str:
    """Truncates plain text to ``limit`` columns, marking what was dropped.

    For the log lines that carry a list whose length the model chooses -- the
    tools a phase offers, the tools one completion asked for. Those read fine
    with two names and run past 140 columns with four, which wraps the line in
    a split pane and costs more than the names were worth. `compact` is the
    wrong tool for them: it renders JSON, so a string arrives back quoted.
    """
    return text if len(text) <= limit else text[: limit - 1] + "\u2026"


#: Retained so the private spelling used inside this module keeps working.
_clip = clip


#: Short forms for the governed-record keys this environment returns. The
#: portfolio row names its fields in full -- `materiality_threshold` rather
#: than `threshold` -- which is right in a payload the model reads and wrong in
#: a log line 30 columns from the edge of a split pane. Keys not listed here
#: pass through unchanged, so a new field is never hidden, only long.
_SHORT_KEYS = {
    "exposure_score": "exposure",
    "materiality_threshold": "threshold",
    "strategic_alignment": "alignment",
    "row_count": "rows",
    "result_count": "results",
}


def _record(record: Any) -> str:
    """One record as a label and its numbers.

    Prose fields are dropped: they are the part the model reads, and they do
    not fit. The numbers are the part a viewer can check against the verdict.

    A `note` survives, because a lookup that resolved nothing says so there and
    nowhere else -- "Exposure is unknown, not zero" is the distinction this
    environment grades, and reporting it as a bare `row_count=0` would show a
    viewer the opposite of what the tool said.
    """
    if not isinstance(record, dict):
        return str(record)
    label = _title(record)
    numbers = [
        f"{_SHORT_KEYS.get(key, key)}={value}"
        for key, value in record.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    for key in ("note", "message", "error"):
        value = record.get(key)
        if isinstance(value, str) and value:
            return " ".join(([label] if label else []) + [value])
    if not numbers:
        return label or ", ".join(list(record)[:4])
    # With several numbers this is a measurement -- the governed portfolio row,
    # whose exposure and threshold are the finding -- and its label is the
    # company, already on the TASK line. With one it is a record that happens
    # to carry a count, and dropping the label leaves a bare `headcount=41`
    # that names neither the team nor the question.
    if len(numbers) >= 2:
        return " ".join(numbers)
    return " ".join(([label] if label else []) + numbers)


def _title(record: Any, limit: int = 48) -> str:
    """A human label for one record, if it carries anything that reads like one."""
    if not isinstance(record, dict):
        flat = str(record)
        return flat if len(flat) <= limit else flat[: limit - 1] + "\u2026"
    for key in ("title", "company", "name", "entity", "path", "url", "id"):
        value = record.get(key)
        if isinstance(value, str) and value:
            return value if len(value) <= limit else value[: limit - 1] + "\u2026"
    return ""


_FENCED_JSON = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)

_VERDICTS = {True: "MATERIAL", False: "not material", None: "abstained"}

#: Short forms for the decision block's list fields, so the line that carries
#: the rollout's conclusion fits beside the CLI pane rather than wrapping.
_ABBREV = {
    "citations": "cites",
    "stakeholders": "stake",
    "caveats": "caveats",
    "next_actions": "actions",
}


def decision(report: str) -> tuple[str, ...]:
    """The rollout's business outcome, read off the brief's decision block.

    Every other line in the stream is about how the rollout ran. These are
    about what it concluded, which is the thing the environment grades and the
    only part a non-engineer in the room needs to read.

    Returned as the verdict and then the evidence behind it, on separate lines,
    because the two together were 91 columns and wrapped.

    Parsing is deliberately forgiving and never raises: this is a log line, and
    a brief that omitted or mangled its decision block is itself worth seeing
    rather than a reason to lose the rest of the summary. It is a report of the
    agent's own output, not a second implementation of the grader -- the
    environment remains the only thing that scores a rollout.
    """
    payload: Optional[dict[str, Any]] = None
    for raw in reversed(_FENCED_JSON.findall(report or "")):
        try:
            candidate = json.loads(raw)
        except (TypeError, ValueError):
            continue
        if isinstance(candidate, dict):
            payload = candidate
            break
    if payload is None:
        return ("no decision block in the brief",)

    material = payload.get("material")
    verdict = _VERDICTS.get(material, f"material={material!r}")
    exposure = payload.get("exposure_score")
    threshold = payload.get("materiality_threshold")
    call = f"{verdict}  {payload.get('company') or '?'}"
    if exposure is not None:
        call += f"  exposure={exposure}"
        if threshold is not None:
            call += f" vs {threshold}"
    call += f"  conf={payload.get('confidence') or '?'}"

    counts = [
        f"{_ABBREV.get(key, key)}={len(value)}"
        for key in ("citations", "stakeholders", "caveats", "next_actions")
        if isinstance(value := payload.get(key), list)
    ]
    return (call, "  ".join(counts)) if counts else (call,)
    if not isinstance(record, dict):
        flat = str(record)
        return flat if len(flat) <= limit else flat[: limit - 1] + "\u2026"
    for key in ("title", "name", "path", "url", "id"):
        value = record.get(key)
        if isinstance(value, str) and value:
            return value if len(value) <= limit else value[: limit - 1] + "\u2026"
    return ""
