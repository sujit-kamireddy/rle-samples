"""Rollout context: the headers RLE attaches to a Hosted Agent call.

RLE posts to this agent version's Responses endpoint with ``agent_session_id``
set to the rollout id, and hands the rollout's runtime context over as request
headers::

    x-client-rle-rollout-id
    x-client-rle-model-endpoint      # capture proxy
    x-client-rle-model-api-key       # its session key
    x-client-rle-mcp-endpoint        # this rollout's tool routes
    x-client-rle-mcp-bearer-token    # bearer for them

They are strictly per-request. At process start they do not exist -- the
container is warm long before any rollout arrives -- and two rollouts served
concurrently by the same process carry different values. Anything derived from
them therefore has to be built per request and passed down the call chain.

RLE has already opened one private OpenEnv session, reset it with this
rollout's task, and will grade the final answer on that same session. The
agent's tool calls have to land on that session or the grader scores an
episode in which the agent did nothing. RLE supplies a session-scoped endpoint;
``RolloutTools`` forwards it unchanged and leaves routing to the environment SDK.

``RolloutContext.absent()`` is the production path: no headers, so the agent
talks to its configured Foundry model and its real toolbox. One build serves
both, which is the whole point of the Hosted Agent subtype -- the thing being
trained is the thing being run.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import count
from typing import Any, Mapping, Optional

import httpx

from telemetry import brief, endpoint

ROLLOUT_ID_HEADER = "x-client-rle-rollout-id"
MODEL_ENDPOINT_HEADER = "x-client-rle-model-endpoint"
MODEL_API_KEY_HEADER = "x-client-rle-model-api-key"
MCP_ENDPOINT_HEADER = "x-client-rle-mcp-endpoint"
MCP_BEARER_TOKEN_HEADER = "x-client-rle-mcp-bearer-token"


@dataclass(frozen=True)
class RolloutContext:
    """One rollout's model proxy and tool routes, or nothing outside a rollout."""

    rollout_id: Optional[str] = None
    model_endpoint: Optional[str] = None
    model_api_key: Optional[str] = None
    mcp_endpoint: Optional[str] = None
    mcp_bearer_token: Optional[str] = None

    @property
    def in_rollout(self) -> bool:
        """True when RLE is driving this call rather than production traffic."""
        return bool(self.model_endpoint)

    @classmethod
    def absent(cls) -> "RolloutContext":
        return cls()

    def describe(self) -> tuple[str, str]:
        """This rollout's wiring, two lines, credentials reduced to a presence.

        The two endpoints are the interesting half -- they name the capture
        proxy this rollout's model calls are recorded by and the MCP endpoint
        its tool calls are answered by, which is what makes the log stream
        legible next to the CLI. They are safe to show: `brief` drops query
        strings, where SAS-style credentials would live.

        The API key and the tool token are the other half and are never shown,
        in any form. Both are live credentials for the length of the rollout
        and a log stream is readable by anyone who can reach the agent, so only
        whether each one arrived is reported. That is the whole diagnostic
        value: a missing header is the failure this catches, and its contents
        would not help.

        The tool endpoint's query contains private session routing metadata.
        It is opaque to this agent and is never shown in diagnostics.

        Returned as two lines because one was 118 columns and wrapped.
        """
        return (
            f"model {brief(self.model_endpoint)}  key {_presence(self.model_api_key)}",
            f"tools {brief(self.mcp_endpoint)}"
            f"  token {_presence(self.mcp_bearer_token)}",
        )

    @classmethod
    def from_headers(cls, headers: Mapping[str, str] | None) -> "RolloutContext":
        """Reads the rollout context out of one request's headers.

        ``ResponseContext.client_headers`` has already lower-cased every key and
        kept only the ``x-client-`` prefix, but this also accepts raw header
        mappings so the agent loop can be exercised directly in tests.
        """
        if not headers:
            return cls.absent()
        lowered = {str(k).lower(): v for k, v in headers.items()}
        return cls(
            rollout_id=lowered.get(ROLLOUT_ID_HEADER),
            model_endpoint=lowered.get(MODEL_ENDPOINT_HEADER),
            model_api_key=lowered.get(MODEL_API_KEY_HEADER),
            mcp_endpoint=lowered.get(MCP_ENDPOINT_HEADER),
            mcp_bearer_token=lowered.get(MCP_BEARER_TOKEN_HEADER),
        )


def _presence(secret: Optional[str]) -> str:
    """Reports whether a credential arrived, never anything about its value."""
    return "ok" if secret else "ABSENT"


class RolloutTools:
    """The rollout's simulated tools, served by the RLE harness container.

    During a rollout every tool the agent would call for real is answered by
    the harness container instead, which records the call so the grader can
    score tool discipline and action restraint from what the agent actually
    did. The bearer token is scoped to this rollout and to its tool routes
    only; it is not a workspace credential and it expires with the rollout.

    Each call is one JSON-RPC ``tools/call``, posted to the rollout's ``/mcp``
    route, forwarding the supplied endpoint unchanged. The SDK routes the
    request to the session RLE opened and will grade on. The result unwraps
    to the tool's output as a JSON string, which is the one thing the agent
    loop above ever sees.
    """

    def __init__(self, context: RolloutContext, timeout_s: float = 30.0) -> None:
        if not context.mcp_endpoint:
            raise ValueError("RolloutTools requires a rollout with an MCP endpoint.")
        self._url = context.mcp_endpoint
        self._headers = (
            {"Authorization": f"Bearer {context.mcp_bearer_token}"}
            if context.mcp_bearer_token
            else {}
        )
        self._timeout_s = timeout_s
        self._request_ids = count(1)

    async def call(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Invokes one simulated tool and returns its output as the model sees it."""
        request = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments,
            },
            "id": next(self._request_ids),
        }
        try:
            async with httpx.AsyncClient(timeout=self._timeout_s) as client:
                response = await client.post(
                    self._url, json=request, headers=self._headers
                )
                response.raise_for_status()
                body = response.json()
        except httpx.HTTPError as error:
            # HTTP error strings include the full private routing URL.
            raise RuntimeError(
                f"MCP tools/call for {tool_name!r} failed: {type(error).__name__}"
            ) from None
        error = body.get("error")
        if error:
            # A JSON-RPC error here is a protocol fault: an unknown tool, a
            # session that was never opened, or one the service already closed.
            # A tool that merely failed comes back as a successful result whose
            # text says so, because the rubric has to see the wasted call.
            raise RuntimeError(
                f"MCP tools/call for {tool_name!r} failed: "
                f"{error.get('message', error)}"
            )
        result = body.get("result")
        return _mcp_text(result if isinstance(result, dict) else {})


def _mcp_text(result: Mapping[str, Any]) -> str:
    """The text of an MCP tool result, which is the tool's JSON string.

    These tools return a single text part, so this is one string in practice.
    The parts are joined rather than indexed so a tool that someday returns
    its output in several does not get silently truncated to the first.
    """
    content = result.get("content")
    if not isinstance(content, list):
        return ""
    return "".join(
        str(part.get("text", ""))
        for part in content
        if isinstance(part, Mapping) and part.get("type") == "text"
    )
