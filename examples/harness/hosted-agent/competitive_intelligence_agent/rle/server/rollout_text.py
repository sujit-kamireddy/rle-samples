"""Turning what RLE hands back into the text the rubric should score.

Duplicated from the legacy harness's ``rle_deprecated/server/env.py`` so this
environment owns every line it runs and ``rle_deprecated/`` can be deleted. The
two copies are identical today and ``tests/test_environment.py`` asserts the
rewards still match while both exist; once ``rle_deprecated/`` goes, this is the
only copy.

These three helpers are not incidental. The first two are the difference
between a run that trains and a run whose reward is constant, and the third is
what makes a grading crash legible through RLE's 200-character error cap.
"""

from __future__ import annotations

import json
import os
import re
import traceback

_OUTPUT_TEXT_START = re.compile(r'"type"\s*:\s*"output_text"\s*,\s*"text"\s*:\s*"')


def salvage_output_text(envelope_text: str) -> list[str]:
    """The ``output_text`` parts of an envelope ``json.loads`` cannot read.

    RLE sanitises the rollout before grading it, rewriting anything that looks
    like a credential to ``[REDACTED]``. Response ids (``caresp_0958111b...``),
    session ids and the model endpoint all match that shape, and the
    substitution lands *inside* the serialised Responses envelope, so the
    envelope stops being valid JSON: on ``ftjob-c78645c32cf846dab90c4aca`` 116
    of 245 rollouts failed to parse.

    Returning the raw envelope in that case -- which is what falling through to
    ``agent_response`` does -- hands ``parse_decision`` a report whose quotes are
    all backslash-escaped, so it never matches. ``format`` scores 0, and every
    dimension that needs a parsed decision collapses to 0 with it. The damage is
    not subtle: rollouts whose verdict was still *exactly right* scored 0.03
    instead of 0.84, and the parse rate tracked the reward exactly -- 100% and
    0.665 at step 0 against 32.5% and 0.340 at step 2. Training was mostly
    ranking whether redaction happened to corrupt the envelope, which is
    independent of anything the policy controls, so the run could not hillclimb.

    Scanning for the text parts directly sidesteps the damage, because the
    corruption is almost always in the ids rather than in the report. That
    recovers 102 of the 116; the remaining 14 are genuine ``Loom sampling
    failed`` 502s whose text is an error message and *should* score near zero.
    """
    chunks: list[str] = []
    for match in _OUTPUT_TEXT_START.finditer(envelope_text):
        index = match.end()
        buffer: list[str] = []
        while index < len(envelope_text):
            char = envelope_text[index]
            if char == "\\":
                buffer.append(envelope_text[index : index + 2])
                index += 2
                continue
            if char == '"':
                break
            buffer.append(char)
            index += 1
        raw = "".join(buffer)
        try:
            chunks.append(json.loads(f'"{raw}"'))
        except json.JSONDecodeError:
            # A redaction inside the report itself can eat an escape sequence.
            # The text is still worth more than the envelope around it.
            chunks.append(raw)
    return chunks


def final_text(agent_response: str) -> str:
    """The agent's report, out of whatever shape RLE handed back.

    For a HostedAgent environment, `agent_response` is the serialised Responses
    object, not the report: ``{"id": "caresp_...", "output": [...]}``. Grading it
    as-is looks for the fenced decision block inside a JSON envelope, where the
    report's own quotes are backslash-escaped, so `parse_decision` never matches.
    Every rollout then scored an identical 0.278 -- the floor from the three
    dimensions that score without a decision -- which is a constant reward, and a
    constant reward is zero GRPO advantage and zero gradient. The run looked
    healthy right up to `grad_norm 0.0`.

    A plain string is passed through, which is what the local end-to-end test and
    a Gym-style caller both send.
    """
    text = agent_response.strip()
    if not text.startswith("{"):
        return agent_response
    try:
        envelope = json.loads(text)
    except json.JSONDecodeError:
        # Sanitisation broke the envelope. Read the text parts out of it anyway
        # rather than grading the envelope; see `salvage_output_text`.
        salvaged = salvage_output_text(text)
        return "\n\n".join(salvaged) if salvaged else agent_response
    if not isinstance(envelope, dict) or "output" not in envelope:
        return agent_response

    chunks: list[str] = []
    for item in envelope.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for part in item.get("content") or []:
            if isinstance(part, dict) and part.get("type") == "output_text":
                chunks.append(str(part.get("text") or ""))
    if not chunks:
        # A failed response carries `output: []` and an error instead. Grade the
        # error text: it scores near zero, and it says why in the metrics.
        error = envelope.get("error")
        if isinstance(error, dict):
            return f"ROLLOUT FAILED: {error.get('code')}: {error.get('message')}"
    return "\n\n".join(chunks)


#: RLE reads a known error field out of a failed response, then caps it at 200
#: characters (`RolloutUpstreamError.MaxDetailLength`) and cuts the tail. So the
#: cause is written shortest-first -- type, then message, then the frame it came
#: from -- and the message is bounded so the frame survives the cut. A whole
#: traceback would arrive as noise; it goes to the log, which this container
#: keeps, rather than into a field sized for a sentence.
_MAX_GRADING_DETAIL = 200
_MAX_GRADING_MESSAGE = 96


def grading_failure_detail(error: BaseException) -> str:
    """Names a grading crash compactly enough to survive RLE's 200-char cap."""
    parts = [f"grading failed: {type(error).__name__}"]
    message = " ".join(str(error).split())
    if message:
        if len(message) > _MAX_GRADING_MESSAGE:
            message = message[:_MAX_GRADING_MESSAGE].rstrip() + "..."
        parts.append(f": {message}")
    frames = traceback.extract_tb(error.__traceback__)
    if frames:
        parts.append(f" at {os.path.basename(frames[-1].filename)}:{frames[-1].lineno}")
    return "".join(parts)[:_MAX_GRADING_DETAIL]
