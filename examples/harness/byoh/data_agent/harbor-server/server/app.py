"""ASGI entry point for the agentic-rle-fineenvs harness.

Adapted, nearly verbatim, from `openenv.harbor`'s reference server
(`OpenEnv/envs/harbor_env/server/app.py`) — see that project's docs at
`OpenEnv/docs/source/environments/harbor.md` for the full contract this
implements. The only change here is defaulting `OPENENV_DATASETS` to
`FineEnvs/data-agent-harbor-train` so this harness boots pointed at the
Data Agent dataset without extra configuration; every other env var below
still overrides the default the same way it does upstream.

Everything is read from the environment so the same image serves any dataset and any engine without
a rebuild — which is what makes this deployable to a Space:

    OPENENV_DATASETS     comma-separated dataset specs (HF repo id, local dir, harbor name@version)
    OPENENV_LLM_URL      DEFAULT OpenAI-spec endpoint; optional, since rollouts may name their own
    OPENENV_MAX_OUTPUT_TOKENS  cap on what an agent may request per turn (default 8192)
    OPENENV_MODEL        served model id; read from the engine when it serves exactly one
    OPENENV_LLM_API_KEY  credential for a hosted endpoint; a Space SECRET, never a variable
    OPENENV_LLM_AUTH_HEADER  header to send it under, when not `Authorization`
    E2B_API_KEY / MODAL_TOKEN_ID+MODAL_TOKEN_SECRET   whichever sandboxes you want offered

OPENENV_LLM_URL is OPTIONAL. With no engine the server still comes up serving its datasets, and each
rollout names the engine it wants (`run_rollout(llm_url=...)`), which is probed once and cached. That
is the useful way round: a dataset tree is thousands of files and prebuilt sandbox templates, while an
engine restarts every training run — and a train-tier engine and an eval-tier one are usually both
wanted against the same task suite.

Naming an engine here still works and makes it the default for rollouts that name none.

An endpoint that cannot return token ids is not a boot failure either: the Space comes up as an EVAL
deployment, which is what a hosted provider can honestly offer. `capture_level` says which it is, and
the UI shows it.

The capture proxy rides on this same app rather than on a second port. A Space publishes exactly one
port and one URL, so the proxy is mounted at `/capture` and the sandbox reaches it at
`https://<space>.hf.space/capture`. Nothing is forwarded and no second listener is opened.
"""

from __future__ import annotations

import os

from openenv.harbor.serving import HarborService, build_app

_DATASETS = [
    d.strip()
    for d in os.environ.get(
        "OPENENV_DATASETS", "FineEnvs/data-agent-harbor-train"
    ).split(",")
    if d.strip()
]
_LLM_URL = os.environ.get("OPENENV_LLM_URL", "")
_MODEL = os.environ.get("OPENENV_MODEL", "")
_API_KEY = os.environ.get("OPENENV_LLM_API_KEY", "") or None
_AUTH_HEADER = os.environ.get("OPENENV_LLM_AUTH_HEADER", "") or "Authorization"
_LLM: dict = {}
# "text", not "tokens". This is the value used when the probe never ran or never finished — an
# ambiguous model list, an unset model, an endpoint that raised — and defaulting it optimistically
# meant a Space in exactly that state built its proxy at token level and stamped every rollout it
# produced as trainable. Which is the one failure this whole capture level exists to prevent, so the
# unknown case has to assume the weaker tier and be corrected upward only by evidence.
_CAPTURE_LEVEL = "text"

# Ask the endpoint what it serves when `OPENENV_MODEL` was not set, the same way `harbor serve` does.
# Without this the proxy has no served model id and stops rewriting `model` on the way upstream, so
# whatever name the harness happened to use is forwarded verbatim and the engine rejects it. The
# report is kept so `capabilities()` can state whether capture is actually supported here.
if _LLM_URL:
    try:
        from openenv.core.harness.capture.validate_llm import list_models, validate_llm

        if not _MODEL:
            served = list_models(_LLM_URL, api_key=_API_KEY, auth_header=_AUTH_HEADER)
            _MODEL = served[0] if len(served) == 1 else ""
        if _MODEL:
            _report = validate_llm(
                _LLM_URL, _MODEL, api_key=_API_KEY, auth_header=_AUTH_HEADER
            )
            _CAPTURE_LEVEL = _report.capture_level or "text"
            _LLM = {
                "url": _LLM_URL,
                "model": _report.model,
                "ok": _report.ok,
                "findings": _report.findings,
                "served_models": _report.served_models,
                "capture_level": _report.capture_level,
                "rollout_type": _report.rollout_type,
                "trainable": _report.trainable,
                "reachable": _report.reachable,
                "param_fixes": _report.param_fixes,
                "authenticated": bool(_API_KEY),
            }
    except Exception as exc:  # noqa: BLE001 - a Space must still boot so the UI can show the fault
        _LLM = {
            "url": _LLM_URL,
            "model": _MODEL,
            "ok": False,
            "reachable": False,
            "capture_level": _CAPTURE_LEVEL,
            "findings": [
                f"could not reach the LLM at startup: {type(exc).__name__}: {exc}"
            ],
        }

    if not _MODEL:
        # Reached when the endpoint serves several models and none was named. The proxy then cannot
        # rewrite `model` upstream, so nothing will work anyway — but it must not claim to be
        # trainable while failing.
        _LLM.setdefault("url", _LLM_URL)
        _LLM.setdefault("ok", False)
        _LLM.setdefault("reachable", False)
        _LLM.setdefault(
            "findings",
            [
                "no model resolved: set OPENENV_MODEL, or point at an endpoint that serves "
                "exactly one model"
            ],
        )
        _LLM["capture_level"] = _CAPTURE_LEVEL

# Resolve capture before the app is built. A Space gives no separate boot hook, the UI needs the
# proxy's public URL to exist by the time anyone presses Run, and `build_app` has to see the service
# in order to mount it.
#
# Started unconditionally: the proxy has to be listening and publicly reachable before any rollout
# can name an engine, and it is the SESSION that carries the engine. Gating this on OPENENV_LLM_URL
# was what made an engineless server useless — every rollout answered "server not initialised".
_service = HarborService(
    llm_url=_LLM_URL,
    model=_MODEL,
    datasets=_DATASETS,
    capture_port=int(os.environ.get("OPENENV_CAPTURE_PORT", "8100")),
    expose=os.environ.get("OPENENV_EXPOSE", "gradio"),
    api_key=_API_KEY,
    auth_header=_AUTH_HEADER,
    capture_level=_CAPTURE_LEVEL,
    max_output_tokens=int(os.environ.get("OPENENV_MAX_OUTPUT_TOKENS", "8192")) or None,
)
# On a Space this only computes the public URL and flags the app for mounting; off one it
# publishes the capture port the usual way.
_service.start()
HarborService.set_current(_service)

os.environ.setdefault("ENABLE_WEB_INTERFACE", "true")

app = build_app(datasets=_DATASETS, llm_url=_LLM_URL, model=_MODEL, llm=_LLM)


# --- Correlated rollouts: an authoritative result store keyed by RLE's rollout id -----------------
#
# `../agent` (untrusted: it runs wherever a customer deploys it) is the only side that can trigger a
# rollout, because only it is handed the model endpoint. But it must not also be the only side that
# can report the reward -- a harness that lies about `reward` would otherwise go unnoticed, since
# RLE never re-runs Harbor's own verifier itself. So triggering and reading are split: `../agent`
# POSTs here to start a run, but reads nothing back that RLE will trust. `../rle`'s `/grade` GETs the
# same rollout_id straight from this server -- the one place the verifier's result actually landed --
# instead of trusting whatever `../agent` says.
#
# The POST calls this same server's own `run_rollout` MCP tool over a loopback HTTP connection rather
# than reimplementing engine resolution and capture-level probing here: `HarborEnv` is the same client
# `../agent` would otherwise call directly, so nothing about how a rollout runs changes, only who is
# allowed to read the result afterward.
import asyncio
import threading
from typing import Any

from fastapi import Body
from fastapi.responses import JSONResponse
from openenv.harbor.client import HarborEnv

_CORRELATED_RESULTS: dict[str, dict[str, Any]] = {}
_CORRELATED_RESULTS_LOCK = threading.Lock()
_SELF_URL = f"http://127.0.0.1:{os.environ.get('PORT', '8000')}"


@app.post("/correlated-rollouts/{rollout_id}")
async def run_correlated_rollout(rollout_id: str, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Runs one rollout and records the authoritative result under `rollout_id`.

    Re-posting the same `rollout_id` overwrites the prior result -- there is one rollout per id in
    practice (RLE mints a fresh one per invocation), and overwriting rather than rejecting means a
    harness retry after a transient failure does not have to invent a new id to get an answer through.
    """

    def _call() -> Any:
        with HarborEnv(base_url=_SELF_URL) as env:
            return env.run_rollout(**payload)

    result = await asyncio.to_thread(_call)
    with _CORRELATED_RESULTS_LOCK:
        _CORRELATED_RESULTS[rollout_id] = result.model_dump()
    return result.model_dump()


@app.get("/correlated-rollouts/{rollout_id}")
async def get_correlated_rollout(rollout_id: str) -> JSONResponse:
    """What `../rle`'s `/grade` calls: the verifier's own result, not a harness's retelling of it."""
    with _CORRELATED_RESULTS_LOCK:
        stored = _CORRELATED_RESULTS.get(rollout_id)
    if stored is None:
        return JSONResponse({"error": "not_found"}, status_code=404)
    return JSONResponse(stored)


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))


if __name__ == "__main__":
    main()
