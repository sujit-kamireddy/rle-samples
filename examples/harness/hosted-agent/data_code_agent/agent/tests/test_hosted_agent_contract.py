"""Tests for the RLE Hosted Agent invocation contract this agent has to speak.

These exist because the contract is invisible at the type level. A Hosted Agent
is called on a standard Responses endpoint, so nothing about the request shape
says "rollout": the rollout context arrives only as headers, and the task
selector arrives only as text inside the one input field. Both can go missing
or change spelling without anything here failing to *parse* -- the symptom is a
rollout that runs against the wrong model, or against task 0 forever, and grades
a plausible-looking number.

The assertions below are RLE's, not ours, and are transcribed from
``HostedAgentRolloutInvoker``: the five ``x-client-rle-*`` header names, the
fact that ``agent_input`` is serialised into a single ``input_text`` (so an
object arrives as a JSON *string*), and that ``agentVersion`` is never sent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rollout_context import RolloutContext  # noqa: E402

HEADERS = {
    "x-client-rle-rollout-id": "rollout-123",
    "x-client-rle-model-endpoint": "https://proxy.invalid/v1",
    "x-client-rle-model-api-key": "session-key",
    "x-client-rle-sandbox-tools-endpoint": "https://tools.invalid/rollouts/rollout-123/tools",
    "x-client-rle-sandbox-tools-token": "tools-token",
}


def _parse_agent_input(text):
    # Imported lazily: `main` builds a Responses host and probes the default
    # model endpoint at import, which the other tests in this file do not need.
    from main import parse_agent_input

    return parse_agent_input(text)


def test_every_rollout_header_is_read():
    rollout = RolloutContext.from_headers(HEADERS)
    assert rollout.rollout_id == "rollout-123"
    assert rollout.model_endpoint == "https://proxy.invalid/v1"
    assert rollout.model_api_key == "session-key"
    assert rollout.sandbox_tools_endpoint == "https://tools.invalid/rollouts/rollout-123/tools"
    assert rollout.sandbox_tools_token == "tools-token"
    assert rollout.in_rollout


def test_headers_are_matched_case_insensitively():
    """HTTP header names are case-insensitive and intermediaries do re-case them."""
    rollout = RolloutContext.from_headers({k.upper(): v for k, v in HEADERS.items()})
    assert rollout.model_endpoint == "https://proxy.invalid/v1"
    assert rollout.sandbox_tools_token == "tools-token"


def test_no_headers_is_not_a_rollout():
    """Ordinary traffic must be distinguishable, not answered against a half-built client."""
    rollout = RolloutContext.from_headers({})
    assert not rollout.in_rollout
    assert rollout.model_endpoint is None
    assert RolloutContext.absent() == rollout


def test_describe_never_reveals_a_credential():
    described = RolloutContext.from_headers(HEADERS).describe()
    assert "session-key" not in described
    assert "tools-token" not in described
    assert "https://proxy.invalid/v1" in described


@pytest.mark.parametrize(
    "text,expected",
    [
        # What RLE actually sends for a configured object `agent_input`: the
        # object serialised into the single `input_text` field.
        ('{"task_index": 7, "split": "FineEnvs/data-agent-harbor-train"}',
         {"task_index": 7, "split": "FineEnvs/data-agent-harbor-train"}),
        # `azd ai rle rollout --agent-input 3`.
        ("3", {"task_index": 3}),
        ('"3"', {"task_index": 3}),
        # No input at all selects the default task rather than failing.
        ("", {}),
    ],
)
def test_agent_input_forms_that_select_a_task(text, expected):
    assert _parse_agent_input(text) == expected


@pytest.mark.parametrize("text", ["true", '"pick a task"', "[1, 2]"])
def test_agent_input_that_names_no_task_is_rejected(text):
    """`True` is an `int` in Python and would have silently selected task 1."""
    with pytest.raises(ValueError):
        _parse_agent_input(text)
