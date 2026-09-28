"""Rollout context: the five headers RLE attaches to a Hosted Agent call.

This is the whole difference between the ``HostedAgent`` and ``BYOH`` subtypes
for this sample. BYOH receives the same four values as a ``rollout_context``
object inside a JSON request body it defines itself; a Hosted Agent is called
on its own Responses API, which has no field to put them in, so RLE sends them
as request headers instead::

    x-client-rle-rollout-id
    x-client-rle-model-endpoint          # capture proxy base URL
    x-client-rle-model-api-key           # its session key
    x-client-rle-sandbox-tools-endpoint  # this rollout's tool routes
    x-client-rle-sandbox-tools-token     # bearer for them

The ``x-client-`` prefix is not decorative: ``azure-ai-agentserver`` forwards
exactly that prefix from the incoming request to
``ResponseContext.client_headers``, lower-cased. Headers without it never reach
the handler.

They are strictly per-request. At process start they do not exist -- the
container is warm long before any rollout arrives -- and two rollouts served
concurrently by the same process carry different values. Anything derived from
them therefore has to be built per request and passed down the call chain, never
cached at import.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

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
        """True when RLE is driving this call rather than ordinary traffic."""
        return bool(self.model_endpoint)

    @classmethod
    def absent(cls) -> "RolloutContext":
        return cls()

    @classmethod
    def from_headers(cls, headers: Mapping[str, str] | None) -> "RolloutContext":
        """Reads the rollout context out of one request's headers.

        ``ResponseContext.client_headers`` has already lower-cased every key and
        kept only the ``x-client-`` prefix, but this lower-cases again so the
        agent loop can be exercised directly from a raw header mapping in tests.
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

    def describe(self) -> str:
        """This rollout's wiring, with credentials reduced to a presence.

        The endpoints are safe to log and are the half worth seeing: they name
        the capture proxy this rollout's model calls are recorded by and the
        sandbox its tool calls are answered by. The key and the token are live
        credentials for the length of the rollout, so only whether each arrived
        is reported -- a missing header is the failure this catches, and the
        contents would not help.
        """
        return (
            f"model {self.model_endpoint or 'ABSENT'}"
            f"  key {_presence(self.model_api_key)}"
            f"  tools {self.sandbox_tools_endpoint or 'ABSENT'}"
            f"  token {_presence(self.sandbox_tools_token)}"
        )


def _presence(secret: Optional[str]) -> str:
    """Reports whether a credential arrived, never anything about its value."""
    return "ok" if secret else "ABSENT"
