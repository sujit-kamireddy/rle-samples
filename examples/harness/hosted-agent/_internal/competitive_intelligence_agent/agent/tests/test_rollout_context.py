"""RolloutTools' wire format: the one tool-call header RLE will actually check.

These tests exercise the HTTP request ``RolloutTools.call`` builds against a mock
transport, not a mock of ``RolloutTools`` itself. That distinction matters: an earlier
version of ``rollout_context.py`` sent the literal string ``"******"`` as the
Authorization header instead of the rollout's real bearer token, so every tool call
during a real rollout reached the MCP route and was rejected with 403 Forbidden. A test
that only asserted *a* header was present, or mocked ``RolloutTools.call`` away, would
not have caught that; asserting the header's actual value does.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest

from rollout_context import RolloutContext, RolloutTools

_OK_RESULT = {
    "jsonrpc": "2.0",
    "id": 1,
    "result": {"content": [{"type": "text", "text": "ok"}]},
}


def _context(**overrides: Any) -> RolloutContext:
    base: dict[str, Any] = dict(
        rollout_id="rollout-1",
        model_endpoint="https://proxy.example/v1",
        model_api_key="model-key",
        mcp_endpoint="https://harness.example/mcp",
        mcp_bearer_token="secret-token-123",
        mcp_session_id="session-1",
    )
    base.update(overrides)
    return RolloutContext(**base)


def _call_against_mock_transport(
    context: RolloutContext, handler
) -> tuple[str, httpx.Request]:
    """Runs one ``RolloutTools.call`` through a mock transport, capturing its request.

    ``RolloutTools.call`` builds its own ``httpx.AsyncClient`` rather than accepting an
    injected one, so the only way to see the request it sends is to make every
    ``httpx.AsyncClient`` constructed during the call use this mock transport.
    """
    captured: dict[str, httpx.Request] = {}

    def recording_handler(request: httpx.Request) -> httpx.Response:
        captured["request"] = request
        return handler(request)

    real_async_client = httpx.AsyncClient

    def patched_async_client(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs.setdefault("transport", httpx.MockTransport(recording_handler))
        return real_async_client(*args, **kwargs)

    tools = RolloutTools(context)
    original_async_client = httpx.AsyncClient
    httpx.AsyncClient = patched_async_client  # type: ignore[assignment]
    try:
        result = asyncio.run(tools.call("lookup", {"q": "x"}))
    finally:
        httpx.AsyncClient = original_async_client  # type: ignore[assignment]
    return result, captured["request"]


def test_call_sends_the_real_bearer_token_not_the_redacted_placeholder():
    result, request = _call_against_mock_transport(
        _context(mcp_bearer_token="secret-token-123"),
        lambda request: httpx.Response(200, json=_OK_RESULT),
    )
    assert result == "ok"
    assert request.headers["authorization"] == "Bearer secret-token-123"


def test_call_omits_the_header_without_a_bearer_token():
    _, request = _call_against_mock_transport(
        _context(mcp_bearer_token=None),
        lambda request: httpx.Response(200, json=_OK_RESULT),
    )
    assert "authorization" not in request.headers


def test_call_surfaces_a_json_rpc_error_as_a_runtime_error():
    error_result = {"jsonrpc": "2.0", "id": 1, "error": {"message": "no such session"}}
    with pytest.raises(RuntimeError, match="no such session"):
        _call_against_mock_transport(
            _context(),
            lambda request: httpx.Response(200, json=error_result),
        )
