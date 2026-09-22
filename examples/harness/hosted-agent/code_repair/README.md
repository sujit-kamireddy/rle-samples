# HostedAgent example: code-repair agent

Same real SWE-bench-Lite instance as the `BYOH` code-repair sample: `psf/requests`
issue `psf__requests-3362` ("`iter_content(decode_unicode=True)` can return
bytes"), pinned at base commit `36453b95b1`. See
[`rle/fixtures/instance.json`](./rle/fixtures/instance.json) for the exact
metadata. The harness reads the real GitHub issue, patches the real
`requests` checkout, and opens a pull request; RLE grades it by running the
real regression test that the real upstream fix made pass.

Two independent pieces make up this `Harness`/`HostedAgent` RLE:

- [`agent/`](./agent) — the agent code that runs as a Foundry Hosted Agent
  version. It reads the RLE-supplied request headers to become rollout-aware;
  otherwise it behaves exactly as it does in production.
- [`rle/`](./rle) — the RLE container: byte-for-byte the same environment the
  `BYOH` sample ships — a real, pinned `requests` checkout baked
  into the image, the hidden regression test patch, mock
  `workspace.apply_patch`/`github.create_pull_request` tools, and a grader
  that runs the real test. RLE does not care which harness subtype invokes
  it, so this piece never changes between subtypes. Running
  `azd ai rle init --type Harness --subtype HostedAgent` copies this exact
  `agent/` + `rle/` pair as your starting point.

`agent/main.py`'s `run_agent_loop` is the same agent loop the `BYOH` sample
runs. Only the invocation transport differs: this one reads runtime context
from request headers on an existing Responses API, while `BYOH` adds an HTTP
invocation endpoint. Scaffold it with
`azd ai rle init --type Harness --subtype BYOH`.

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

`AZD_CONTAINER_RUNTIME` selects the container runtime for RLE.
`DOCKER_COMMAND` selects the runtime used by
[`az acr login`](https://learn.microsoft.com/azure/container-registry/container-registry-authentication#sign-in-by-using-an-alternative-container-tool-instead-of-docker).
Docker Desktop is not required when using Podman.

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

Replace the last command's placeholders with the preceding outputs.
For registries using **RBAC Registry + ABAC Repository Permissions**, replace
`AcrPull` with `"Container Registry Repository Reader"`; `AcrPull` is for
RBAC-only registries. The person assigning the role needs role-assignment
permission at the registry scope. Allow time for propagation before publishing.

After publishing, run `azd ai rle show --output json` and wait for the published
version's `diskImageConversionStatus` to be `Ready` before rollout.
If conversion fails, inspect its error; after fixing the cause, increment
`rle.version` before republishing because registered versions are immutable.

## 1. Deploy the agent as a Hosted Agent version

Publish `agent/` as a Foundry Hosted Agent version through your normal Hosted
Agent deployment path (`azd ai agent ...` or the Foundry portal). Note the
agent's name and the specific version you publish — RLE binds a rollout to
that exact version, so `agentName`/`agentVersion` in `rle.toml` must be
resolvable, callable Hosted Agent identifiers (not `$default`).

## 2. Author and iterate the RLE side

RLE calls exactly four things on a harness container:

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Readiness, polled before `/reset` |
| `POST /reset` | The caller's task, verbatim |
| `POST /tools/*` | The harness's tool calls, proxied per rollout |
| `POST /grade` | `{"rollout": ..., "agent_response": "..."}` to a reward |

There is no `/step` and no OpenEnv `Environment`, `Action`, or `Observation`
here. Those belong to the `Gym: OpenEnv` subtype, where RLE drives the model
through the environment step by step. In a harness RLE the harness owns its
own loop, so the environment only sets the task up, serves the tools, and
scores the result.

The RLE side is identical to the `BYOH` sample's: adapt `rle/server/env.py`'s
`/reset`, mock tool routes, and `/grade` for your own repo, mocks, and
reward function. The taskset is not in the image either. Every Execute
Rollout call carries its own task and RLE posts it to `/reset` unchanged;
`rle/fixtures/instance.json` is one SWE-bench-Lite instance baked in so the
sample runs on its own, and `/reset` rejects a task asking for a different
`instance_id` rather than grading the wrong repo.

`azd ai rle run` is supported only for `Gym: OpenEnv` environments, so iterate
here by publishing a version and running a rollout (steps 3 and 4).

## 3. Register the agent version and publish

Complete [registry setup](#before-the-first-publish) and set the endpoints
in the same terminal first:

```powershell
$env:FOUNDRY_PROJECT_ENDPOINT = "https://<account>.services.ai.azure.com/api/projects/<project>"
$env:AZURE_CONTAINER_REGISTRY_ENDPOINT = "<registry>.azurecr.io"
```

In Bash, use `export NAME="value"` instead of `$env:NAME = "value"`.

Set `agentName` and `agentVersion` in `rle/rle.toml` to the Hosted Agent
version you published in step 1:

```toml
[rle]
name = "code_repair_hosted_agent"
version = "1.1.0"
type = "Harness"
subtype = "HostedAgent"
agentName = "code-repair-agent"
agentVersion = "1"
```

Then publish from the folder that holds `rle.toml`:

```bash
cd rle
azd ai rle publish
```

Registered versions are immutable, so bump `version` before republishing.

Training, evaluation, and optimization jobs against this RLE will now invoke
your Hosted Agent version for every rollout, with one Hosted Agent version
able to serve many rollouts and (during optimization) many candidate
configurations resolved per rollout.

## 4. Run a rollout

Run this from the initialized sample's `rle` folder (where this sample's
`rle.toml` lives) against whichever published name/version you registered.
`rollout` reads `rle.name`/`rle.version` from `rle.toml`, provisions a real
Loom training session and sampler checkpoint for `--model`, calls Execute
Rollout (which forwards `--agent-input` to your Hosted Agent), and prints the
resulting reward. `--task` is this rollout's taskset entry and reaches
`/reset` verbatim, so `{}` and `{"instance_id": "psf__requests-3362"}` both
work here and any other instance is rejected. The agent separately requires
`agent_input.issue`; pull the real SWE-bench issue text out of the bundled
fixture so the command stays copy/paste-ready. Starting from the initialized
sample folder:

```bash
cd rle
AGENT_INPUT=$(python3 -c "import json; print(json.dumps({'issue': json.load(open('fixtures/instance.json'))['problem_statement']}))")
azd ai rle rollout --model Qwen/Qwen3-32B --task '{}' --agent-input "$AGENT_INPUT"
```

## Reference: how RLE invokes your Hosted Agent

The sample already does this wiring. For each rollout, RLE creates a session
for your Hosted Agent version and calls its Responses API with five extra
headers:

```text
x-client-rle-rollout-id
x-client-rle-model-endpoint          # capture proxy, replaces your prod model endpoint
x-client-rle-model-api-key           # capture proxy session key
x-client-rle-sandbox-tools-endpoint  # sandbox tool routes for this rollout
x-client-rle-sandbox-tools-token     # authorizes calls to those routes
```

The tools endpoint is authorized, so a tool call without
`x-client-rle-sandbox-tools-token` as a bearer is refused with 403. The token
is scoped to this rollout and to `/tools/*` and expires with the rollout, so
it is not something to store or reuse.

`agent/main.py`'s `create_model_client` and `call_tool` read these headers
when present and fall back to the agent's normal production model endpoint
and tools when absent (ordinary, non-rollout traffic). Adapt
`handle_responses_request` to however your Hosted Agent hosting framework
exposes incoming request headers to your handler.
