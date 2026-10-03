"""Rollout context: the headers RLE attaches to a Hosted Agent call.

RLE posts to this agent version's Responses endpoint with ``agent_session_id``
set to the rollout id, and hands the rollout's runtime context over as request
headers::

    x-client-rle-rollout-id
    x-client-rle-model-endpoint          # capture proxy
    x-client-rle-model-api-key           # its session key
    x-client-rle-sandbox-tools-endpoint  # this rollout's tool routes
    x-client-rle-sandbox-tools-token     # bearer for them
    x-client-rle-sandbox-session-id      # the MCPEnvironment session, if any

They are strictly per-request. At process start they do not exist -- the
container is warm long before any rollout arrives -- and two rollouts served
concurrently by the same process carry different values. Anything derived from
them therefore has to be built per request and passed down the call chain.

The last header is the protocol discriminator. RLE sends it only when the
published environment declares ``mcp_environment``, in which case the service
has already opened one private OpenEnv session, reset it with this rollout's
task, and will grade the final answer on that same session. The agent's tool
calls have to land on that session or the grader scores an episode in which
the agent did nothing, so the id is threaded straight into ``SandboxTools``.
Without the header the environment is the older one that answers a plain
``POST <tools endpoint>/<tool name>``, and ``SandboxTools`` keeps speaking it.

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
SANDBOX_TOOLS_ENDPOINT_HEADER = "x-client-rle-sandbox-tools-endpoint"
SANDBOX_TOOLS_TOKEN_HEADER = "x-client-rle-sandbox-tools-token"
SANDBOX_SESSION_ID_HEADER = "x-client-rle-sandbox-session-id"


@dataclass(frozen=True)
class RolloutContext:
    """One rollout's model proxy and tool routes, or nothing outside a rollout."""

    rollout_id: Optional[str] = None
    model_endpoint: Optional[str] = None
    model_api_key: Optional[str] = None
    sandbox_tools_endpoint: Optional[str] = None
    sandbox_tools_token: Optional[str] = None
    sandbox_session_id: Optional[str] = None

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
        proxy this rollout's model calls are recorded by and the sandbox its
        tool calls are answered by, which is what makes the log stream legible
        next to the CLI. They are safe to show: `brief` drops query strings,
        where SAS-style credentials would live.

        The API key and the tool token are the other half and are never shown,
        in any form. Both are live credentials for the length of the rollout
        and a log stream is readable by anyone who can reach the agent, so only
        whether each one arrived is reported. That is the whole diagnostic
        value: a missing header is the failure this catches, and its contents
        would not help.

        The session id gets the same treatment. It is not a credential, but
        RLE redacts it from the rollout graph, and paired with the tool token
        it addresses one live session, so this reports only the protocol it
        implies. ``mcp`` or ``http`` is the diagnostic worth having: it says
        which of the two wire formats the tool calls below went out in.

        Returned as two lines because one was 118 columns and wrapped.
        """
        return (
            f"model {brief(self.model_endpoint)}  key {_presence(self.model_api_key)}",
            f"tools {brief(self.sandbox_tools_endpoint)}"
            f"  token {_presence(self.sandbox_tools_token)}"
            f"  protocol {'mcp' if self.sandbox_session_id else 'http'}",
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
            sandbox_tools_endpoint=lowered.get(SANDBOX_TOOLS_ENDPOINT_HEADER),
            sandbox_tools_token=lowered.get(SANDBOX_TOOLS_TOKEN_HEADER),
            sandbox_session_id=lowered.get(SANDBOX_SESSION_ID_HEADER),
        )


def _presence(secret: Optional[str]) -> str:
    """Reports whether a credential arrived, never anything about its value."""
    return "ok" if secret else "ABSENT"


class SandboxTools:
    """The rollout's simulated tools, served by the RLE harness container.

    During a rollout every tool the agent would call for real is answered by
    the harness container instead, which records the call so the grader can
    score tool discipline and action restraint from what the agent actually
    did. The bearer token is scoped to this rollout and to its tool routes
    only; it is not a workspace credential and it expires with the rollout.

    Two wire formats reach the same simulated tools, chosen by whether RLE
    sent a sandbox session id:

    ``mcp``
        ``rle/server/environment.py``. One JSON-RPC ``tools/call`` per tool,
        posted to the rollout's ``/mcp`` route, carrying the session id RLE
        opened and will grade on. This is the converged protocol.

    ``http``
        The legacy harness's own route handler. ``POST <endpoint>/<tool name>``
        with the arguments as the body. Kept because an environment published
        before the convergence sends no session id, and the agent image is the
        same one production runs.

    Both return the tool's output as a JSON string, and it is the identical
    string in both cases: the MCP tool wrapper and the legacy route each
    forward the single message the simulated tool produced. So ``call`` has
    one return contract and the agent loop above does not know which ran.
    """

    def __init__(self, context: RolloutContext, timeout_s: float = 30.0) -> None:
        if not context.sandbox_tools_endpoint:
            raise ValueError("SandboxTools requires a rollout with tool routes.")
        # The header's value already ends in /tools, so for the legacy route
        # only the tool name is appended. Adding another /tools would yield
        # /tools/tools/<name>.
        self._base = context.sandbox_tools_endpoint.rstrip("/")
        self._session_id = context.sandbox_session_id
        self._mcp_url = _mcp_url(self._base)
        self._headers = (
            {"Authorization": f"Bearer {context.sandbox_tools_token}"}
            if context.sandbox_tools_token
            else {}
        )
        self._timeout_s = timeout_s
        self._request_ids = count(1)

    @property
    def protocol(self) -> str:
        """``mcp`` or ``http``: which wire format this rollout's tools speak."""
        return "mcp" if self._session_id else "http"

    async def _post(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            response = await client.post(
                f"{self._base}/{tool_name}",
                json=arguments,
                headers=self._headers,
            )
            response.raise_for_status()
            return response.json()

    async def _call_tool_over_mcp(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """One JSON-RPC ``tools/call`` on this rollout's bound session."""
        request = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments,
                "session_id": self._session_id,
            },
            "id": next(self._request_ids),
        }
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            response = await client.post(
                self._mcp_url, json=request, headers=self._headers
            )
            response.raise_for_status()
            body = response.json()
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
        return result if isinstance(result, dict) else {}

    async def call(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Invokes one simulated tool and returns its output as the model sees it."""
        if self._session_id:
            return _mcp_text(await self._call_tool_over_mcp(tool_name, arguments))
        payload = await self._post(tool_name, arguments)
        return str(payload.get("content", ""))


def _mcp_url(tools_endpoint: str) -> str:
    """The rollout's ``/mcp`` route, derived from its ``/tools`` one.

    RLE hands over one endpoint per rollout and it names the legacy route, so
    the MCP route is reached by swapping the last segment. Both are served by
    the same rollout-scoped path and authorized by the same bearer token, so
    nothing else about the URL changes.
    """
    base = tools_endpoint.rstrip("/")
    suffix = "/tools"
    if base.endswith(suffix):
        base = base[: -len(suffix)]
    return f"{base}/mcp"


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
