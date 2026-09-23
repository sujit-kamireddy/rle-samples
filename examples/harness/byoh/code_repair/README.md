# BYOH (Bring Your Own Harness) example: code-repair agent

This is a real SWE-bench-Lite instance, not synthetic data: `psf/requests`
issue `psf__requests-3362` ("`iter_content(decode_unicode=True)` can return
bytes"), pinned at base commit `36453b95b1`. See
[`rle/fixtures/instance.json`](./rle/fixtures/instance.json) for the exact
metadata (source: `princeton-nlp/SWE-bench_Lite`). The harness must read the
real GitHub issue, patch the real `requests` checkout, and open a pull
request; RLE grades it by running the real regression test
(`tests/test_requests.py::TestRequests::test_response_decode_unicode`) that
the real upstream fix made pass.

Two independent pieces make up this `Harness`/`BYOH` RLE:

- [`agent/`](./agent) — the production agent (harness), deployable anywhere
  you can reach over HTTPS. Unmodified except for how it builds its model
  client and tools (see "How RLE invokes your harness" below).
- [`rle/`](./rle) — the RLE container: a real, pinned `requests` checkout
  baked into the image, the hidden regression test patch, mock
  `workspace.apply_patch`/`github.create_pull_request` tools, and a grader
  that runs the real test. Running
  `azd ai rle init --type Harness --subtype BYOH` copies this exact
  `agent/` + `rle/` pair as your starting point.

This example and the `HostedAgent` code-repair sample are intentionally
close to identical: `rle/` is byte-for-byte the same environment (RLE does
not care which harness subtype invokes it), and `agent/`'s `run_agent_loop`
is the same agent loop in both. Only the invocation transport differs: this
one adds an HTTP invocation endpoint, while `HostedAgent` reads request
headers on an existing Responses API. Scaffold it with
`azd ai rle init --type Harness --subtype HostedAgent`.

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

## 1. Deploy the agent (harness) anywhere

`agent/` is a plain FastAPI service with no Foundry or Azure dependency. Run
it wherever you already run the rest of your stack — a container in your own
cluster, an App Service, a VM, or a local container. From the initialized
sample folder, use either Docker Desktop or Podman:

```bash
cd agent
docker build -t code-repair-agent:latest .
docker run --rm -p 8080:8080 code-repair-agent:latest
```

For Podman, replace `docker` with `podman` in both commands.

Put it behind HTTPS (a reverse proxy, App Gateway, or your platform's TLS
termination). Since RLE never attaches caller, workspace, or identity
headers to its invocation requests, protect the endpoint yourself — for
example, require a shared secret header and validate it in `agent/app.py`
before dispatching to the agent loop.

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

`rle/server/env.py`'s `/reset` copies the pinned `requests` checkout baked
into the image and applies the hidden test patch, which the harness never
sees. RLE reads only the status code, so that response body is for humans.
`/tools/workspace.apply_patch` and `/tools/github.create_pull_request` mock
the harness's real tools. `/grade` runs the real regression test, checks for
a recorded pull request, and returns `{reward, is_success, info}`. Adapt
these for your own repo, mocks, and reward function.

The taskset is not in the image. Every Execute Rollout call carries its own
task and RLE posts it to `/reset` unchanged; `rle/fixtures/instance.json` is
one SWE-bench-Lite instance baked in so the sample runs on its own. `/reset`
checks the `instance_id` it receives against that fixture and rejects a
mismatch, so a task for another instance fails loudly instead of being
graded against the wrong repo. Scaling to a real taskset means making the
checkout follow `instance_id`, not changing the protocol.

`azd ai rle run` is supported only for `Gym: OpenEnv` environments, so iterate
here by publishing a version and running a rollout (steps 3 and 4).

## 3. Register the base URL and publish

Once your deployed agent URL is stable, point the manifest at it and publish
a version. Complete [registry setup](#before-the-first-publish) and set
`FOUNDRY_PROJECT_ENDPOINT` and `AZURE_CONTAINER_REGISTRY_ENDPOINT` in the
same terminal first:

```powershell
$env:FOUNDRY_PROJECT_ENDPOINT = "https://<account>.services.ai.azure.com/api/projects/<project>"
$env:AZURE_CONTAINER_REGISTRY_ENDPOINT = "<registry>.azurecr.io"
```

In Bash, use `export NAME="value"` instead of `$env:NAME = "value"`.

Set `baseUrl` in `rle/rle.toml` to your deployed agent's invocation URL:

```toml
[rle]
name = "code_repair_byoh"
version = "1.1.0"
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

Training, evaluation, and optimization jobs against this RLE will now invoke
your deployed agent for every rollout.

## 4. Run a rollout

Run this from the initialized sample's `rle` folder (where this sample's `rle.toml`
lives) against whichever published name/version you registered. `rollout`
reads `rle.name`/`rle.version` from `rle.toml`, provisions a real Loom
training session and sampler checkpoint for `--model`, calls Execute Rollout
(which forwards `--agent-input` to your deployed agent's `--base-url`), and
prints the resulting reward. `--task` is this rollout's taskset entry and
reaches `/reset` verbatim, so `{}` and `{"instance_id":
"psf__requests-3362"}` both work here and any other instance is rejected.
The agent separately requires `agent_input.issue`; pull the real SWE-bench
issue text out of the bundled fixture so the command stays copy/paste-ready.
Starting from the initialized sample folder:

```bash
cd rle
AGENT_INPUT=$(python3 -c "import json; print(json.dumps({'issue': json.load(open('fixtures/instance.json'))['problem_statement']}))")
azd ai rle rollout --model Qwen/Qwen3-32B --task '{}' --agent-input "$AGENT_INPUT"
```

## Reference: how RLE invokes your harness

The sample already does this wiring. Only the construction of the agent's
model client and tools becomes RLE-aware — its native agent loop is
unchanged:

```python
model = create_model_client(request.rollout_context)   # -> capture proxy, not prod model endpoint
# tool calls route to request.rollout_context.sandbox_tools_endpoint instead of prod tools
```

See `agent/app.py`'s `create_model_client`, `call_tool`, and
`run_agent_loop` for the full wiring, and `invoke`/`poll`/`withdraw` for the
invocation contract RLE calls.

RLE invokes a harness asynchronously: the start request only starts work, and
the answer is collected from a per-invocation resource RLE derives from the
registered URL. A harness never supplies that address, so RLE only ever calls
back under the path you registered.

```text
POST <base-url>/invoke
{
  "rollout_id": "...",
  "operation_id": "...",
  "agent_input": {"...": "agent-specific input"},
  "rollout_context": {
    "capture_proxy_endpoint": "...",
    "capture_proxy_session_key": "...",
    "sandbox_tools_endpoint": "...",
    "sandbox_tools_bearer_token": "..."
  }
}

202 Accepted
{"retry_after_ms": 500}
```

`rollout_id` correlates; `operation_id` addresses. RLE mints `operation_id`
per invocation and puts it in the poll and cancel URLs, so key your own state on
it. `rollout_id` is unique per project, not globally, and the poll and cancel
legs carry no credential, so the caller-chosen id cannot safely name the
resource.

RLE then polls until the rollout reports an outcome. The first poll is
immediate, so a harness that finishes at once costs one extra round trip rather
than a poll interval:

```text
GET <base-url>/invoke/rollouts/{operation_id}

200 {"status": "running"}
200 {"status": "succeeded", "output_text": "..."}
200 {"status": "failed", "error": {"message": "..."}}
```

If RLE stops waiting — the caller disconnected, or the rollout deadline
passed — it withdraws the rollout so you can stop spending tokens on an answer
nobody will read. RLE checks only the status, and may not send this at all, so
treat it as advisory and keep your own timeout — but do answer 2xx, because RLE
records anything else as a cleanup failure:

```text
DELETE <base-url>/invoke/rollouts/{operation_id}
```

`agent/app.py` tracks rollouts in a process dictionary keyed by `operation_id`,
which is the smallest thing that demonstrates the contract. A harness that runs more than one replica
needs shared state instead: the poll can land on any replica, and a replica that
has never heard of the rollout cannot answer for it.
