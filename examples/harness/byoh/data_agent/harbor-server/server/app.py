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


# --- Run rollout and surface the sandbox's own answer text ----------------------------------------
#
# `../agent` (untrusted: it runs wherever a customer deploys it) is the only side that can trigger a
# rollout, because only it is handed the model endpoint. But `../rle`'s `/grade` must not trust
# whatever `reward` `../agent` reports, because Harbor's grader is public and deterministic: a
# harness that simply echoed back `EXPECTED_ANSWER` would score perfectly without doing any work. So
# this endpoint hands `../agent` the raw *answer text* Harbor's sandbox produced, not a reward --
# `/grade` computes the reward itself, from that text, using its own vendored copy of the grader (see
# `../rle/server/env.py`). `../agent` still cannot fabricate a better score for itself: it can only
# relay -- or fail to relay -- whatever the sandbox actually wrote.
#
# The answer text is not part of `HarborRolloutResult` (OpenEnv's own result schema has no such
# field): it only ever exists as `/workdir/answer.txt` inside the now-torn-down sandbox. Getting it
# out requires declaring it as a Harbor `artifacts` entry in `task.toml` (see the vendored dataset)
# so Harbor's own `ArtifactHandler` copies it onto this container's local disk, under
# `<trials_dir>/<trial_name>/artifacts/workdir/answer.txt`, before the sandbox is torn down --
# `openenv.harbor.environment._trials_dir()` is the same default (`/tmp/openenv-harbor-trials`,
# overridable via `OPENENV_HARBOR_TRIALS_DIR`) `HarborEnv` already uses internally.
#
# The POST runs the rollout through `HarborEnvironment._run_rollout` -- the method this same
# server's `run_rollout` MCP tool delegates to -- rather than reimplementing engine resolution and
# capture-level probing here. It previously reached that method through `HarborEnv` over a loopback
# HTTP connection; calling it directly changes nothing about how a rollout runs, and keeps the work
# in THIS request's context. That is what lets `rollout_tools` give each sandbox its own tool
# credentials without a process-global, which concurrent rollouts would overwrite for one another.
# The tradeoff is depending on a private method; the openenv wheel is vendored and pinned, so an
# upgrade is the point to re-check it.
import asyncio
from pathlib import Path
from typing import Any

from fastapi import Body
from openenv.harbor.environment import HarborEnvironment
from openenv.harbor.models import HarborRolloutResult

from . import rollout_tools

rollout_tools.install()

# `_run_rollout` declares no defaults for these; `HarborEnv.run_rollout` supplied them client-side,
# so they have to be supplied here instead.
_ROLLOUT_DEFAULTS: dict[str, Any] = {
    "split": "",
    "task_index": 0,
    "harness": "opencode",
    "sandbox": "e2b",
    "reward_key": "",
    "keep_sandbox": False,
    "force_build": False,
}
_TRIALS_DIR = Path(os.environ.get("OPENENV_HARBOR_TRIALS_DIR", "/tmp/openenv-harbor-trials"))


def _read_answer_text(trial_name: str | None) -> str | None:
    """Reads the artifact-collected copy of `/workdir/answer.txt` for a finished trial.

    Returns `None` (never raises) when the trial name is missing or the file was never collected --
    an older, unpatched `task.toml` (no `artifacts` entry), or a rollout that failed before the agent
    wrote an answer. `/grade` treats a missing answer as an ungraded (zero-reward) rollout, the same
    way it already treats a missing `agent_response`.
    """
    if not trial_name:
        return None
    # Harbor's `ArtifactHandler` mirrors an absolute container source path under `artifacts/`,
    # stripping the leading "/": `/workdir/answer.txt` -> `artifacts/workdir/answer.txt`.
    path = _TRIALS_DIR / trial_name / "artifacts" / "workdir" / "answer.txt"
    try:
        return path.read_text().strip()
    except OSError:
        return None


@app.post("/correlated-rollouts/{rollout_id}")
async def run_correlated_rollout(rollout_id: str, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Runs one rollout and hands `../agent` the sandbox's own answer text.

    The path keeps its historical name (RLE's `rollout_id` in the URL is useful for log
    correlation) even though nothing is stored under it anymore -- see the module note above for
    why `../rle`'s `/grade` no longer needs to read anything back from this server.

    `sandbox_tools_endpoint`/`sandbox_tools_bearer_token` are lifted out of the payload rather than
    forwarded to Harbor: they are not rollout parameters but the capability the agent needs in order
    to file a compliance disclosure, and they reach it as sandbox environment variables.
    """
    call_kwargs = {**_ROLLOUT_DEFAULTS, **payload}
    endpoint = str(call_kwargs.pop("sandbox_tools_endpoint", "") or "")
    token = str(call_kwargs.pop("sandbox_tools_bearer_token", "") or "")

    with rollout_tools.rollout_tools(endpoint, token):
        raw = await HarborEnvironment()._run_rollout(**call_kwargs)

    result = HarborRolloutResult.model_validate_json(raw)
    dumped = result.model_dump()
    answer_text = await asyncio.to_thread(_read_answer_text, dumped.get("trial_name"))
    return {**dumped, "answer_text": answer_text}


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))


if __name__ == "__main__":
    main()
