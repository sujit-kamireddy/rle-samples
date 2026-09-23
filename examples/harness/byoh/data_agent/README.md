# BYOH (Bring Your Own Harness) example: FineEnvs data-agent

Wraps Hugging Face's [`FineEnvs/data-agent`](https://huggingface.co/collections/FineEnvs/data-agent)
collection -- a suite of deterministic data-analysis agent tasks (built from
`jupyter-agent`) played through **Harbor**, packaged as `openenv.harbor`. Each
task drops an agent into a sandboxed Jupyter-style workspace with a question
about a dataset; Harbor's own deterministic grader scores the agent's answer.

Unlike a typical BYOH sample, the agent loop, the sandbox, and the grader are
not implemented in this sample at all -- Harbor already owns all three. This
sample is therefore three pieces instead of two:

- [`harbor-server/`](./harbor-server) -- a standalone `openenv.harbor` server.
  Deploy this once; it serves the Data Agent dataset over Harbor's Task API and
  exposes one `run_rollout` MCP tool that boots a sandbox, runs the agent loop
  against whatever model endpoint it is told to call, and grades the result.
  It also exposes `POST /correlated-rollouts/{rollout_id}` -- see the answer
  note below -- and bakes both the dataset (patched so every task declares
  `/workdir/answer.txt` as a Harbor `artifacts` entry) and its own `openenv`
  dependency in as vendored files, so its image builds and runs with no
  network dependency at all.
- [`agent/`](./agent) -- the harness RLE actually invokes (`Harness`/`BYOH`'s
  `agent/`). It implements RLE's `/invoke`/poll/withdraw contract, but instead
  of running its own agent loop it `POST`s to `harbor-server`'s
  `/correlated-rollouts/{rollout_id}`, which triggers a `run_rollout` call
  keyed by that rollout id, passing RLE's capture proxy as the model endpoint
  so every call Harbor's agent makes is recorded in the rollout graph.
- [`rle/`](./rle) -- the RLE container. `/grade` never re-runs Harbor's
  verifier and never trusts a `reward` `agent/` reports; it grades the
  sandbox's own raw answer text itself, with its own vendored copy of
  Harbor's deterministic grader (see the answer note below).

**Why not have `agent/` just report the reward?** `agent/` is untrusted,
customer-hosted code, and Harbor's grader is a public, deterministic function
of `(gold, candidate)` -- nothing stops a misbehaving harness from computing
the winning `candidate` string and simply reporting whatever `reward` it
wants. So `rle/`'s `/grade` computes `reward` itself rather than reading it
out of anything `agent/` sends.

What it does trust is the *raw answer text* the sandbox produced
(`answer_text`): `harbor-server` declares `/workdir/answer.txt` as a Harbor
`artifacts` entry in every task's `task.toml`, which makes Harbor's own
`ArtifactHandler` copy that file onto `harbor-server`'s local disk before the
sandbox is torn down -- OpenEnv's public result schema has no such field, so
this is the only way to get it out at all. `harbor-server` hands that text
back to `agent/`, which relays it verbatim to RLE as `output_text`; RLE
forwards it to `/grade` as `agent_response`, and `/grade` grades it with the
same deterministic grader Harbor itself runs (vendored into
`rle/server/vendor/grader.py`, byte-identical across every task in this
dataset) against a vendored per-task answer key
(`rle/server/vendor/task-meta/`). `agent/` still cannot fabricate a better
score for itself: it can only relay -- or fail to relay -- whatever the
sandbox actually wrote.

Running `azd ai rle init --type Harness --subtype BYOH --sample data_agent`
copies this exact `agent/` + `rle/` pair as your starting point; `harbor-server/`
is deployed separately, once, and referenced by URL.

## Container and registry setup

Choose a container runtime to build, run, and publish this environment:

- **Docker Desktop:** start it in Linux-container mode and check `docker info`.
- **Podman:** install Podman (Podman Desktop is optional) and check `podman info`.
  On Windows/macOS, first run `podman machine list`; run `podman machine init`
  only if no machine exists, then `podman machine start` if it is stopped.
  Native Linux does not need a Podman machine.

Select the runtime in the **same terminal** used for `publish`.
Use `podman` below, or replace it with `docker` for Docker Desktop:

```powershell
$env:AZD_CONTAINER_RUNTIME = "podman"
$env:DOCKER_COMMAND = $env:AZD_CONTAINER_RUNTIME
```

Bash equivalent:

```bash
export AZD_CONTAINER_RUNTIME=podman
export DOCKER_COMMAND="$AZD_CONTAINER_RUNTIME"
```

### Before the first publish

Sign in and authenticate the selected runtime to ACR (use the registry
**name**, without `.azurecr.io`). Your user needs push permission:
`AcrPush`, or `Container Registry Repository Writer` on an ABAC-enabled registry.

```text
az login
azd auth login
az acr login --name "<registry>"
```

Local login does **not** grant the RLE service permission to pull your image.
An administrator must grant that separately to the Foundry **project's
system-assigned managed identity**, not your user or the parent account identity.
Find the project's ARM resource ID in Azure portal (it ends in
`/accounts/<account>/projects/<project>`, not the project's HTTPS endpoint).
Ensure the project's system-assigned identity is enabled, then get its
principal ID and the registry scope:

```text
az resource show --ids "<project-arm-resource-id>" --query identity.principalId --output tsv
az acr show --name "<registry>" --query id --output tsv
az role assignment create --assignee-object-id "<project-principal-id>" --assignee-principal-type ServicePrincipal --role AcrPull --scope "<registry-resource-id>"
```

## 1. Deploy `harbor-server`

Build and deploy the standalone Harbor server first; it has no RLE
dependency and is easiest to prove working on its own. See
[`harbor-server/README.md`](./harbor-server/README.md) for local run, build,
and deploy instructions. Confirm `/health` and `/schema` on the deployed URL
before moving on, and pick a `run_rollout` sandbox backend (`e2b`, `modal`,
`daytona`, ...) you hold credentials for.

## 2. Deploy the agent (harness)

`agent/` is a plain FastAPI service with no Foundry or Azure dependency other
than reaching `harbor-server`. Run it wherever you already run the rest of
your stack:

```bash
cd agent
docker build -t data-agent-byoh:latest .
docker run --rm -p 8080:8080 \
  -e HARBOR_SERVER_URL=https://<your-deployed-harbor-server> \
  data-agent-byoh:latest
```

For Podman, replace `docker` with `podman` in both commands.

Put it behind HTTPS and protect it yourself -- RLE never attaches caller,
workspace, or identity headers to its invocation requests, so require a
shared secret header and validate it in `agent/app.py` before dispatching to
Harbor.

## 3. Author and iterate the RLE side

`rle/server/env.py`'s `/reset` is close to a no-op -- the Harbor task selector
(`task_index`, `split`, `harness`, `sandbox`) travels in `--agent-input`, not
in whatever `/reset` returns. `/grade` doesn't call out anywhere: it parses
`answer_text`, `task_index`, and `split` out of `agent_response` (the same
JSON string `agent/`'s `/invoke` returned as `output_text`, which RLE
forwards to `/grade` verbatim), looks up that task's answer key in the
vendored `rle/server/vendor/task-meta/` metadata, and grades it with the
vendored copy of Harbor's own deterministic grader
(`rle/server/vendor/grader.py`). `agent/` never computes or even sees a
`reward` -- it only relays what the sandbox wrote (see `agent/app.py`'s
`run_harbor_rollout`). Adapt `rle/server/env.py`'s `grade_rollout` if you
want reward shaping other than Harbor's raw verifier score.

`azd ai rle run` is supported only for `Gym: OpenEnv` environments, so iterate
here by publishing a version and running a rollout (steps 4 and 5).

## 4. Register the base URL and publish

Once your deployed agent URL is stable, point the manifest at it and publish
a version. Complete [registry setup](#before-the-first-publish) and set
`FOUNDRY_PROJECT_ENDPOINT` and `AZURE_CONTAINER_REGISTRY_ENDPOINT` in the
same terminal first:

```powershell
$env:FOUNDRY_PROJECT_ENDPOINT = "https://<account>.services.ai.azure.com/api/projects/<project>"
$env:AZURE_CONTAINER_REGISTRY_ENDPOINT = "<registry>.azurecr.io"
```

Set `baseUrl` in `rle/rle.toml` to your deployed agent's invocation URL:

```toml
[rle]
name = "data_agent_byoh"
version = "0.1.0"
type = "Harness"
subtype = "BYOH"
baseUrl = "https://<your-deployed-agent-host>/invoke"
```

Then publish from the folder that holds `rle.toml`:

```bash
cd rle
azd ai rle publish
```

Registered versions are immutable, so bump `version` before republishing.

## 5. Run a rollout

```bash
cd rle
azd ai rle rollout --model Qwen/Qwen3-32B --task '{}' \
  --agent-input '{"task_index": 0, "split": "FineEnvs/data-agent-harbor-train", "harness": "opencode", "sandbox": "e2b"}'
```

This sample's `/reset` ignores `--task`, so `{}` is enough; the Harbor task
selection lives entirely in `--agent-input`.

## Reference: how RLE invokes your harness

The full `/invoke`/poll/withdraw contract is documented in `agent/app.py`'s
module docstring. In short: RLE invokes a harness asynchronously -- the start
request only starts work, and the answer is collected from a per-invocation
resource RLE derives from the registered URL. What `agent/app.py` does with
`rollout_context` once it has it is the one thing worth calling out here:
instead of calling the model directly, it hands
`capture_proxy_endpoint`/`capture_proxy_session_key`
to `harbor-server`'s `/correlated-rollouts/{rollout_id}`, which in turn calls
Harbor's `run_rollout(llm_url=..., api_key=...)` and lets Harbor's agent loop
make the calls.
