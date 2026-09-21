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
  client and tools (see "Wire the harness" below).
- [`rle/`](./rle) — the RLE container: a real, pinned `requests` checkout
  baked into the image, the hidden regression test patch, mock
  `workspace.apply_patch`/`github.create_pull_request` tools, and a grader
  that runs the real test. Running
  `azd ai rle init --type Harness --subtype BYOH --harness-source sample`
  copies this exact `agent/` + `rle/` pair as your starting point (the
  `--harness-source existing` default instead scaffolds a generic,
  empty placeholder for a harness you already built and deployed yourself).

This example and [`../../hosted-agent/code_repair`](../../hosted-agent/code_repair) are intentionally
close to identical: `rle/` is byte-for-byte the same environment (RLE does
not care which harness subtype invokes it), and `agent/`'s `run_agent_loop`
is the same agent loop in both. Only the invocation transport differs — this
one adds an HTTP invocation endpoint; [`hosted-agent`](../../hosted-agent/code_repair)
instead reads request headers on an existing Responses API.

## Container and registry setup

Install the Azure CLI, `azd`, and the RLE extension, then choose a runtime:

- **Docker Desktop:** start it in Linux-container mode and check `docker info`.
- **Podman:** install Podman (Podman Desktop is optional) and check `podman info`.
  On Windows/macOS, first run `podman machine list`; run `podman machine init`
  only if no machine exists, then `podman machine start` if it is stopped.
  Native Linux does not need a Podman machine.

> **Native Podman prerequisite:** Use an RLE CLI build with
> [`AZD_CONTAINER_RUNTIME` support](https://github.com/sujit-kamireddy/azure-dev/tree/fix/rle-podman-runtime/cli/azd/extensions/azure.ai.rle).
> Unpatched `0.8.10-preview` ignores this setting. Docker-compatibility mode
> is an alternative for older builds, but build/export compatibility varies;
> the instructions below use native Podman instead.

Select the runtime in the **same terminal** used for `run` and `publish`.
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

`AZD_CONTAINER_RUNTIME` selects the RLE container executable in compatible
builds; `DOCKER_COMMAND` separately selects the executable used by
[`az acr login`](https://learn.microsoft.com/azure/container-registry/container-registry-authentication#sign-in-by-using-an-alternative-container-tool-instead-of-docker).
The native Podman path does not need Docker Desktop, `DOCKER_HOST`, or Buildx.

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

For Podman, replace `docker` with `podman` in both commands (the direct
`podman build` / `podman run` commands do not depend on the RLE CLI build).

Put it behind HTTPS (a reverse proxy, App Gateway, or your platform's TLS
termination). Since RLE never attaches caller, workspace, or identity
headers to its invocation requests, protect the endpoint yourself — for
example, require a shared secret header and validate it in `agent/app.py`
before dispatching to the agent loop.

## 2. Wire the harness

Per the harness contract, only the construction of the agent's model client
and tools becomes RLE-aware — its native agent loop is unchanged:

```python
model = create_model_client(request.rollout_context)   # -> capture proxy, not prod model endpoint
# tool calls route to request.rollout_context.sandbox_tools_endpoint instead of prod tools
```

See `agent/app.py`'s `create_model_client`, `call_tool`, and
`run_agent_loop` for the full wiring, and `/invoke`/`/health` for the
invocation contract RLE calls:

```text
POST <base-url>/invoke
{
  "rollout_id": "...",
  "agent_input": {"...": "agent-specific input"},
  "rollout_context": {
    "capture_proxy_endpoint": "...",
    "capture_proxy_session_key": "...",
    "sandbox_tools_endpoint": "...",
    "sandbox_tools_bearer_token": "..."
  }
}
```

The agent must respond with `{"output_text": "..."}`.

## 3. Author and iterate the RLE side

From the initialized sample folder, with [container setup](#container-and-registry-setup)
complete:

```bash
cd rle
azd ai rle run
```

`rle/server/env.py`'s `reset()` copies the pinned `requests` checkout baked
into the image, applies the hidden test patch, and returns the real issue
text. `/tools/workspace.apply_patch` and `/tools/github.create_pull_request`
mock the harness's real tools; `/grade` runs the real regression test and
checks for a recorded pull request. Adapt these for your own repo, mocks,
and reward function the same way you'd iterate on a Gym sample.

## 4. Register the base URL and publish

Once your deployed agent URL is stable, scaffold (or update) the manifest
with it and publish a version. Complete [registry setup](#before-the-first-publish)
and set `FOUNDRY_PROJECT_ENDPOINT` and `AZURE_CONTAINER_REGISTRY_ENDPOINT`
in the same terminal first:

```powershell
$env:FOUNDRY_PROJECT_ENDPOINT = "https://<account>.services.ai.azure.com/api/projects/<project>"
$env:AZURE_CONTAINER_REGISTRY_ENDPOINT = "<registry>.azurecr.io"
```

In Bash, use `export NAME="value"` instead of `$env:NAME = "value"`.

```bash
azd ai rle init code_repair_byoh \
  --type Harness --subtype BYOH \
  --harness-source sample \
  --base-url https://<your-deployed-agent-host>/invoke \
  --no-prompt

cd code_repair_byoh/rle
azd ai rle publish
```

Training, evaluation, and optimization jobs against this RLE will now invoke
your deployed agent for every rollout.

## 5. Run a rollout

Run this from the initialized sample's `rle` folder (where this sample's `rle.toml`
lives) against whichever published name/version you registered. `rollout`
reads `rle.name`/`rle.version` from `rle.toml`, provisions a real Loom
training session and sampler checkpoint for `--model`, calls Execute Rollout
(which forwards `--agent-input` to your deployed agent's `--base-url`), and
prints the resulting reward. This sample's `reset()` ignores `--task`, so
`{}` is enough, but the agent requires `agent_input.issue`; pull the real
SWE-bench issue text out of the bundled fixture so the command stays
copy/paste-ready. Starting from the initialized sample folder:

```bash
cd rle
AGENT_INPUT=$(python3 -c "import json; print(json.dumps({'issue': json.load(open('fixtures/instance.json'))['problem_statement']}))")
azd ai rle rollout --model Qwen/Qwen3-32B --task '{}' --agent-input "$AGENT_INPUT"
```
