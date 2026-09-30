"""Rollout context: the five headers RLE attaches to a Hosted Agent call.

RLE posts to this agent version's Responses endpoint with ``agent_session_id``
set to the rollout id, and hands the rollout's runtime context over as request
headers::

    x-client-rle-rollout-id
    x-client-rle-model-endpoint          # capture proxy
    x-client-rle-model-api-key           # its session key
    x-client-rle-sandbox-tools-endpoint  # this rollout's tool routes
    x-client-rle-sandbox-tools-token     # bearer for them

They are strictly per-request. At process start they do not exist -- the
container is warm long before any rollout arrives -- and two rollouts served
concurrently by the same process carry different values. Anything derived from
them therefore has to be built per request and passed down the call chain.

``RolloutContext.absent()`` is the production path: no headers, so the agent
talks to its configured Foundry model and its real toolbox. One build serves
both, which is the whole point of the Hosted Agent subtype -- the thing being
trained is the thing being run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import httpx

from telemetry import brief, endpoint

ROLLOUT_ID_HEADER = "x-client-rle-rollout-id"
MODEL_ENDPOINT_HEADER = "x-client-rle-model-endpoint"
MODEL_API_KEY_HEADER = "x-client-rle-model-api-key"
SANDBOX_TOOLS_ENDPOINT_HEADER = "x-client-rle-sandbox-tools-endpoint"
SANDBOX_TOOLS_TOKEN_HEADER = "x-client-rle-sandbox-tools-token"


@dataclass(frozen=True)
class RolloutContext:
    """One rollout's model proxy and tool routes, or nothing outside a rollout."""

    rollout_id: Optional[str] = None
    model_endpoint: Optional[str] = None
    model_api_key: Optional[str] = None
    sandbox_tools_endpoint: Optional[str] = None
    sandbox_tools_token: Optional[str] = None

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

        Returned as two lines because one was 118 columns and wrapped.
        """
        return (
            f"model {brief(self.model_endpoint)}  key {_presence(self.model_api_key)}",
            f"tools {brief(self.sandbox_tools_endpoint)}"
            f"  token {_presence(self.sandbox_tools_token)}",
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
        )


def _presence(secret: Optional[str]) -> str:
    """Reports whether a credential arrived, never anything about its value."""
    return "ok" if secret else "ABSENT"


class SandboxTools:
    """The rollout's simulated tools, served by the RLE harness container.

    During a rollout every tool the agent would call for real is answered by
    ``rle/server/env.py`` instead, which records the call so the grader can
    score tool discipline and action restraint from what the agent actually
    did. The bearer token is scoped to this rollout and to its ``/tools``
    routes only; it is not a workspace credential and it expires with the
    rollout.
    """

    def __init__(self, context: RolloutContext, timeout_s: float = 30.0) -> None:
        if not context.sandbox_tools_endpoint:
            raise ValueError("SandboxTools requires a rollout with tool routes.")
        # The header's value already ends in /tools, so only the tool name is
        # appended. Adding another /tools yields /tools/tools/<name>.
        self._base = context.sandbox_tools_endpoint.rstrip("/")
        self._headers = (
            {"Authorization": f"Bearer {context.sandbox_tools_token}"}
            if context.sandbox_tools_token
            else {}
        )
        self._timeout_s = timeout_s

    async def _post(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            response = await client.post(
                f"{self._base}/{tool_name}",
                json=arguments,
                headers=self._headers,
            )
            response.raise_for_status()
            return response.json()

    async def call(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Invokes one simulated tool and returns its output as the model sees it."""
        payload = await self._post(tool_name, arguments)
        return str(payload.get("content", ""))
