"""Hands each rollout's own tool credentials to the agent inside its sandbox.

The agent has to reach ``<sandbox_tools_endpoint>/report_sensitive_data_access`` on the RLE
environment container to file a compliance disclosure. Both the URL and the bearer token embed the
rollout id, so they differ per rollout -- and ``run_rollout`` has a fixed parameter list with no
channel for them.

Where the credentials actually enter the sandbox
------------------------------------------------
Not through ``AgentConfig.env``, which is the obvious guess and is wrong for this harness. That
field becomes the agent's ``extra_env``, which Harbor uses only as a *lookup source* when resolving
provider credentials (``agents/base.py`` ``_env_sources``). OpenCode builds the environment it
actually exports into the sandbox from ``self.model_connection.env`` plus three hardcoded XDG/VCS
keys (``agents/installed/opencode.py``, in ``run``), and ``model_connection.env`` is assembled from
a closed allowlist of provider-specific variable names (``agents/model_connection.py``
``resolve_model_connection``). An arbitrary name like ``COMPLIANCE_ENDPOINT`` is silently dropped.

So the injection point is ``OpenCode.model_connection`` itself. It is a plain ``@property``
recomputed on every access -- not a ``cached_property`` -- and OpenCode reads it inside ``run``,
which means the wrapper below executes during the rollout, while that rollout's context is current.
``ResolvedModelConnection`` is a frozen dataclass, so the addition is a ``dataclasses.replace``.

``Seam.resolve`` is wrapped too, but only as a side channel: Harbor scrubs values of
sensitivity-matching keys in ``AgentConfig.env`` out of trial artifacts (``trial/trial.py``
``_scrub_jobs_dir``), and ``COMPLIANCE_TOKEN`` matches. Registering the token there keeps it from
landing in ``opencode.txt`` if the agent ever echoes it. Delivery does not depend on that path --
Harbor may redact or templatize what it stores there, which is exactly why it cannot be the
delivery mechanism.

Why a ContextVar and not a module global
----------------------------------------
This server runs concurrent rollouts in one process. A global holding "the current rollout's token"
is written by rollout B before rollout A's sandbox reads it, and A silently boots with B's
credentials -- no exception, just a disclosure filed against the wrong rollout. A ContextVar gives
each request its own slot, and the value stays visible to the code running on its behalf, including
across ``await`` and ``asyncio.to_thread``.

This works only because ``run_correlated_rollout`` calls ``_run_rollout`` in process; contextvars do
not survive an HTTP hop. openenv relies on the same property for the same reason -- see
``openenv/harbor/proc_env_context.py``, which makes ``os.environ`` reads context-local so that
credential-by-env harnesses stop serialising behind a lock. That mechanism does not cover this case:
it serves harnesses that read ``os.environ`` inside ``run()`` (claude-code, gemini-cli, goose),
whereas opencode takes its configuration through kwargs and the resolved model connection.

Every patch here is best-effort and idempotent. Failing to inject costs the compliance signal for
that rollout; it must never cost the rollout itself.
"""

from __future__ import annotations

import contextlib
import contextvars
import dataclasses
import logging
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)

# Read by the agent inside the sandbox. Named for the fictional internal compliance service the task
# instructions refer to, not for RLE: from the agent's point of view this is an ordinary corporate
# audit API, and the name is part of what makes the disclosure decision a realistic judgement call.
# `tools/bake_compliance_instruction.py` writes these same names into every task prompt.
ENDPOINT_VAR = "COMPLIANCE_ENDPOINT"
TOKEN_VAR = "COMPLIANCE_TOKEN"

_current: contextvars.ContextVar[Optional[dict[str, str]]] = contextvars.ContextVar(
    "rollout_tools", default=None
)


@contextlib.contextmanager
def rollout_tools(endpoint: str, token: str) -> Iterator[None]:
    """Scopes one rollout's tool credentials to the current context.

    An empty endpoint or token scopes nothing, so a caller that was handed no tool capability (a
    local run, or a stale RLE payload) behaves exactly as it did before this feature existed.
    """
    if not endpoint or not token:
        yield
        return
    reset = _current.set({ENDPOINT_VAR: endpoint, TOKEN_VAR: token})
    try:
        yield
    finally:
        _current.reset(reset)


def current() -> dict[str, str]:
    """Returns this context's tool variables, or an empty mapping outside a scoped rollout."""
    return dict(_current.get() or {})


def _patch_model_connection() -> bool:
    """Adds the tool variables to the environment OpenCode exports into its sandbox."""
    try:
        from harbor.agents.installed.opencode import OpenCode
    except ImportError:
        logger.warning("harbor OpenCode agent unavailable; tool credentials not injected")
        return False

    existing = OpenCode.__dict__.get("model_connection")
    if not isinstance(existing, property) or existing.fget is None:
        logger.warning("OpenCode.model_connection is not a property; tool credentials not injected")
        return False
    if getattr(existing.fget, "_injects_rollout_tools", False):
        return True

    original = existing.fget

    def model_connection(self: Any) -> Any:
        access = original(self)
        tools = _current.get()
        if not tools:
            return access
        return dataclasses.replace(access, env={**dict(access.env), **tools})

    model_connection._injects_rollout_tools = True  # type: ignore[attr-defined]
    OpenCode.model_connection = property(model_connection)  # type: ignore[assignment]
    return True


def _patch_seam_resolve() -> bool:
    """Registers the token in `AgentConfig.env` so Harbor scrubs it from trial artifacts."""
    try:
        from openenv.harbor.seams import Seam
    except ImportError:
        logger.warning("openenv seams unavailable; token not registered for scrubbing")
        return False

    if getattr(Seam.resolve, "_injects_rollout_tools", False):
        return True

    original = Seam.resolve

    # `resolve` is keyword-only, so `**kwargs` cannot silently drop a positional argument.
    def resolve(self: Any, **kwargs: Any) -> Any:
        model, agent_kwargs, agent_env, proc_env = original(self, **kwargs)
        tools = _current.get()
        if tools:
            agent_env = {**agent_env, **tools}
        return model, agent_kwargs, agent_env, proc_env

    resolve._injects_rollout_tools = True  # type: ignore[attr-defined]
    Seam.resolve = resolve  # type: ignore[method-assign]
    return True


def install() -> bool:
    """Installs both patches. Idempotent; returns True only if delivery is in place."""
    delivered = _patch_model_connection()
    _patch_seam_resolve()
    return delivered
