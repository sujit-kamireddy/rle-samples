"""Tests for the `mcp_environment` tool path.

The baked instruction in every task posts a flat JSON body to
`$COMPLIANCE_ENDPOINT/report_sensitive_data_access`. Under the legacy protocol
that is the environment's own route. Under `mcp_environment` there is no such
route at all -- `../rle/server/app.py` serves an OpenEnv app whose only tool
surface is JSON-RPC on a session -- so the agent is pointed at a loopback route
here that translates for it.

What is worth testing is exactly the part a live rollout would not tell us
about until the reward came back wrong: that the translation carries *this*
rollout's session id, that it refuses another rollout's token, and that the
body the agent sees is the same shape the legacy route returned. A disclosure
filed against the wrong session, or silently dropped, grades as "did not
disclose" with no error anywhere.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

import app as app_module  # noqa: E402
from app import ROLLOUTS, RolloutContext, _compliance_endpoint, app  # noqa: E402

MCP_CONTEXT = {
    "model_endpoint": "https://proxy.invalid/v1",
    "model_api_key": "session-key",
    "sandbox_tools_endpoint": "https://tools.invalid/rollouts/r1/tools",
    "sandbox_tools_token": "tools-token",
    "sandbox_session_id": "session-abc",
}
LEGACY_CONTEXT = {k: v for k, v in MCP_CONTEXT.items() if k != "sandbox_session_id"}

RECORDED = {
    "status": "recorded",
    "reference": "disclosure-1a2b",
    "columns_reported": ["salary"],
}


class _FakeMcp:
    """Stands in for the environment's `/mcp` route and remembers what it got."""

    def __init__(self, body=None, status: int = 200):
        self.requests: list[dict] = []
        self.urls: list[str] = []
        self.headers: list[dict] = []
        self._body = body if body is not None else {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "content": [{"type": "text", "text": json.dumps(RECORDED)}],
                "structuredContent": RECORDED,
                "isError": False,
            },
        }
        self._status = status

    def install(self, monkeypatch):
        fake = self

        class _Response:
            status_code = fake._status

            def raise_for_status(self):
                if fake._status >= 400:
                    raise app_module.httpx.HTTPStatusError(
                        "boom", request=None, response=None
                    )

            def json(self):
                return fake._body

        class _Client:
            def __init__(self, *_, **__):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                return False

            async def post(self, url, json=None, headers=None):
                fake.urls.append(url)
                fake.requests.append(json)
                fake.headers.append(headers or {})
                return _Response()

        monkeypatch.setattr(app_module.httpx, "AsyncClient", _Client)
        return self


@pytest.fixture
def rollout(request):
    """Registers one rollout and removes it again, so tests cannot leak into each other."""
    context = RolloutContext.model_validate(getattr(request, "param", MCP_CONTEXT))
    ROLLOUTS._contexts["operation-1"] = context
    yield context
    ROLLOUTS._contexts.pop("operation-1", None)


def test_the_session_id_is_optional_so_a_legacy_rollout_still_parses():
    """RLE only sends it for an environment published with the MCP protocol."""
    assert RolloutContext.model_validate(LEGACY_CONTEXT).sandbox_session_id is None
    assert RolloutContext.model_validate(MCP_CONTEXT).sandbox_session_id == "session-abc"


def test_the_mcp_route_is_the_tools_route_with_its_last_segment_swapped():
    context = RolloutContext.model_validate(MCP_CONTEXT)
    assert context.mcp_url == "https://tools.invalid/rollouts/r1/mcp"


def test_a_legacy_rollout_is_pointed_straight_at_the_environment():
    context = RolloutContext.model_validate(LEGACY_CONTEXT)
    assert (
        _compliance_endpoint("operation-1", context)
        == "https://tools.invalid/rollouts/r1/tools"
    )


def test_an_mcp_rollout_is_pointed_at_this_services_loopback_route():
    context = RolloutContext.model_validate(MCP_CONTEXT)
    endpoint = _compliance_endpoint("operation-1", context)
    assert endpoint.startswith("http://127.0.0.1:")
    assert endpoint.endswith("/local-tools/operation-1")
    # The baked instruction appends the tool name to this, and nothing else.
    assert not endpoint.endswith("/")


def test_a_disclosure_is_translated_onto_this_rollouts_own_session(monkeypatch, rollout):
    fake = _FakeMcp().install(monkeypatch)
    with TestClient(app) as client:
        response = client.post(
            "/local-tools/operation-1/report_sensitive_data_access",
            headers={"Authorization": "Bearer tools-token"},
            json={"columns_reported": ["salary"], "note": "payroll"},
        )

    assert response.status_code == 200
    # The agent sees the same body the legacy route returned, so the baked
    # instruction does not have to know which protocol it is on.
    assert response.json() == RECORDED

    assert fake.urls == ["https://tools.invalid/rollouts/r1/mcp"]
    sent = fake.requests[0]
    assert sent["method"] == "tools/call"
    assert sent["jsonrpc"] == "2.0"
    assert sent["params"]["name"] == "report_sensitive_data_access"
    # The session id is the whole point: without it the service has no episode
    # to record against, and the grader would see a task that never disclosed.
    assert sent["params"]["session_id"] == "session-abc"
    assert sent["params"]["arguments"] == {
        "columns_reported": ["salary"],
        "note": "payroll",
    }
    assert fake.headers[0]["Authorization"] == "Bearer tools-token"


def test_a_text_only_result_is_still_returned_as_the_tools_json(monkeypatch, rollout):
    """`structuredContent` is an optional part of the result, so the text is parsed too."""
    _FakeMcp(
        body={
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"content": [{"type": "text", "text": json.dumps(RECORDED)}]},
        }
    ).install(monkeypatch)
    with TestClient(app) as client:
        response = client.post(
            "/local-tools/operation-1/report_sensitive_data_access",
            headers={"Authorization": "Bearer tools-token"},
            json={"columns_reported": ["salary"]},
        )
    assert response.status_code == 200
    assert response.json() == RECORDED


def test_another_rollouts_token_cannot_file_against_this_session(monkeypatch, rollout):
    fake = _FakeMcp().install(monkeypatch)
    with TestClient(app) as client:
        response = client.post(
            "/local-tools/operation-1/report_sensitive_data_access",
            headers={"Authorization": "Bearer someone-elses-token"},
            json={"columns_reported": ["salary"]},
        )
    assert response.status_code == 401
    # Rejected before the environment is touched, so a guessed operation id
    # cannot even be used to probe whether a rollout is running.
    assert fake.requests == []


@pytest.mark.parametrize("rollout", [LEGACY_CONTEXT], indirect=True)
def test_a_legacy_rollout_has_no_loopback_route(monkeypatch, rollout):
    """It has a real environment route, and routing it here would lose the session binding."""
    fake = _FakeMcp().install(monkeypatch)
    with TestClient(app) as client:
        response = client.post(
            "/local-tools/operation-1/report_sensitive_data_access",
            headers={"Authorization": "Bearer tools-token"},
            json={"columns_reported": ["salary"]},
        )
    assert response.status_code == 404
    assert fake.requests == []


def test_a_finished_rollout_cannot_file_anything(monkeypatch):
    """The capability is dropped with the rollout, not left live until withdrawal."""
    fake = _FakeMcp().install(monkeypatch)
    with TestClient(app) as client:
        response = client.post(
            "/local-tools/operation-gone/report_sensitive_data_access",
            headers={"Authorization": "Bearer tools-token"},
            json={"columns_reported": ["salary"]},
        )
    assert response.status_code == 404
    assert fake.requests == []


def test_a_protocol_fault_is_a_502_rather_than_a_quiet_success(monkeypatch, rollout):
    """A rejected `tools/call` must not read as a filed disclosure."""
    _FakeMcp(
        body={
            "jsonrpc": "2.0",
            "id": 1,
            "error": {"code": -32601, "message": "Unknown tool"},
        }
    ).install(monkeypatch)
    with TestClient(app) as client:
        response = client.post(
            "/local-tools/operation-1/report_sensitive_data_access",
            headers={"Authorization": "Bearer tools-token"},
            json={"columns_reported": ["salary"]},
        )
    assert response.status_code == 502
