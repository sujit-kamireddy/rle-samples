"""ASGI entry point for the agentic-rle-fineenvs harness.

Serves one route. `../agent` POSTs a task to `/correlated-rollouts/{rollout_id}`;
this runs `opencode` against it inside this container (`server/opencode_direct.py`)
and returns the answer text the agent wrote. `../agent` relays that verbatim to
`../rle`'s `/grade`, which computes the reward from it -- this server never
produces one, which is what stops a harness claiming a score it did not earn.

Configuration:

    MODEL_URL          default OpenAI-spec endpoint. Optional: each rollout
                             names its own, and under RLE that is the per-rollout
                             capture proxy rather than a real model endpoint.
    MODEL_ID            served model id. Probed from the endpoint when unset
                             and the endpoint serves exactly one.
    MODEL_API_KEY      credential for the default endpoint.
    MODEL_AUTH_HEADER  header to send it under, when not `Authorization`.
    PORT                     listen port (default 8000).
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx
from fastapi import Body, FastAPI

from . import opencode_direct

logger = logging.getLogger(__name__)

_LLM_URL = os.environ.get("MODEL_URL", "")
_MODEL = os.environ.get("MODEL_ID", "")
_API_KEY = os.environ.get("MODEL_API_KEY", "") or None
_AUTH_HEADER = os.environ.get("MODEL_AUTH_HEADER", "") or "Authorization"


def _served_models(llm_url: str) -> list[str]:
    """Lists the model ids an OpenAI-spec endpoint serves.

    Used only to resolve a default `MODEL_ID` at startup, and only when the
    endpoint serves exactly one model. Under RLE each rollout names its own
    endpoint, so this never runs on the rollout path.
    """
    headers = {_AUTH_HEADER: f"Bearer {_API_KEY}" if _AUTH_HEADER == "Authorization" else _API_KEY}
    response = httpx.get(
        f"{llm_url.rstrip('/')}/models",
        headers={k: v for k, v in headers.items() if v},
        timeout=30.0,
    )
    response.raise_for_status()
    return [m["id"] for m in response.json().get("data", []) if m.get("id")]


# Resolved at import so a misconfigured deployment is visible in the startup log
# rather than on its first rollout. Never fatal: the endpoint named here is only a
# default, and a rollout that names its own works regardless of what this found.
if _LLM_URL and not _MODEL:
    try:
        _served = _served_models(_LLM_URL)
        # Only an unambiguous answer is usable. With several models and none
        # named, `opencode` would be pointed at a model id the endpoint does not
        # recognise, so leaving it empty and failing loudly per rollout is better
        # than guessing one.
        _MODEL = _served[0] if len(_served) == 1 else ""
        if not _MODEL:
            logger.warning(
                "%s serves %d models and MODEL_ID is unset; set it explicitly.",
                _LLM_URL,
                len(_served),
            )
    except Exception as exc:  # noqa: BLE001 - startup must survive an unreachable default endpoint
        logger.warning("could not list models at %s: %s: %s", _LLM_URL, type(exc).__name__, exc)

app = FastAPI(title="data-agent-harness")


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "healthy", "model": _MODEL or None}


# --- Run rollout and surface the agent's own answer text -------------------------------------
#
# `../agent` (untrusted: it runs wherever a customer deploys it) is the only side that can trigger a
# rollout, because only it is handed the model endpoint. But `../rle`'s `/grade` must not trust
# whatever `reward` `../agent` reports, because the grader is public and deterministic: a harness
# that simply echoed back `EXPECTED_ANSWER` would score perfectly without doing any work. So this
# endpoint hands `../agent` the raw *answer text* the agent produced, not a reward -- `/grade`
# computes the reward itself, from that text, using its own vendored copy of the grader (see
# `../rle/server/env.py`). `../agent` still cannot fabricate a better score for itself: it can only
# relay -- or fail to relay -- whatever the agent actually wrote.
@app.post("/correlated-rollouts/{rollout_id}")
async def run_correlated_rollout(
    rollout_id: str, payload: dict[str, Any] = Body(...)
) -> dict[str, Any]:
    """Runs one rollout and hands `../agent` the agent's own answer text.

    The path keeps its historical name -- RLE's `rollout_id` in the URL is useful for log
    correlation -- even though nothing is stored under it.

    `sandbox_tools_endpoint`/`sandbox_tools_bearer_token` are lifted out of the payload rather
    than treated as rollout parameters: they are the capability the agent needs in order to file a
    compliance disclosure, and they reach it as environment variables on the `opencode` process.
    """
    endpoint = str(payload.get("sandbox_tools_endpoint", "") or "")
    token = str(payload.get("sandbox_tools_bearer_token", "") or "")

    try:
        result = await opencode_direct.run_rollout(
            task_index=int(payload.get("task_index", 0)),
            model=_MODEL,
            llm_url=str(payload.get("llm_url") or _LLM_URL),
            api_key=str(payload.get("api_key") or ""),
            compliance_endpoint=endpoint,
            compliance_token=token,
            rollout_id=rollout_id,
        )
    except opencode_direct.RolloutError as exc:
        logger.warning("rollout %s failed: %s", rollout_id, exc)
        return {"ok": False, "error": str(exc), "answer_text": None}
    return result


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))


if __name__ == "__main__":
    main()
