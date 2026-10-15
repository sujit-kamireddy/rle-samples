# Create, edit and try an RLE

Commands here use PowerShell. On another shell, adapt quoting/environment
syntax and use the platform's path separators. Check installed help before
assuming a preview flag/command is available.

## 1. Prerequisites and discovery

RLE is the `azure.ai.rle` azd extension, not a command group added by toggling
an environment variable. With a compatible azd host, install the public preview
distribution if the extension is missing (respect permissions on shared hosts):

```powershell
azd version
azd extension source add --name rle --type url --location https://raw.githubusercontent.com/Azure/azure-dev/rle-releases/cli/azd/extensions/registry.rle.json
azd extension install azure.ai.rle --source rle
azd ai rle version
azd ai rle --help
azd ai rle init --help
```

For an existing installation, use
`azd extension update azure.ai.rle --source <installed-source-alias>`.
Check `azd extension source list`; don't silently switch a development install
from `rle-dev` to the public `rle` channel. Development source can be newer
than the public release. The verified revisions are in the upstream baseline.

Core commands no longer require `AZD_AI_RLE_ENABLE`. The remaining
`AZD_AI_RLE_ENABLE_ALL=true` gate exposes Harness init targets, hidden samples
and staged commands such as `train`, `jobs` and `cancel`. If `ai` or
`extension` is unknown, use a compatible azd host/installed extension rather
than inventing another gate. Do not modify a shared installation without
permission.

Use a working Linux-container runtime (`docker info` or `podman info`).
If installed Docker Desktop is merely stopped and local container work is
authorized, `docker desktop start --timeout 120` can start it; verify
`docker info` afterward. Do not restart a shared working daemon, reset its
configuration or bypass required setup.
For Podman set the runtime in the same terminal:

```powershell
$env:AZD_CONTAINER_RUNTIME = "podman"
$env:DOCKER_COMMAND = $env:AZD_CONTAINER_RUNTIME
```

On Windows/macOS, start an existing Podman machine; initialize one only if
none exists. For Docker publishing, enable BuildKit where required:

```powershell
$env:DOCKER_BUILDKIT = "1"
```

## 2. Scaffold or inspect an existing project

The current samples and development CLI use `gym`/`byoh` plus
`--protocol`/`--endpoint`. Public `0.8.29-preview` still uses the removed
`--subtype` schema and cannot publish these new manifests. Check installed help
and use an intended compatible release/channel; do not silently downgrade the
current samples to satisfy an older tool. The commands below require the
current schema, not merely a CLI with the same preview version label.

```powershell
azd ai rle init math_lab --type gym --sample math_rl
$env:AZD_AI_RLE_ENABLE_ALL = "true"
azd ai rle init data_lab --type byoh --protocol rle-invocation --endpoint https://your-harness.example.com/invoke --sample data_code_agent
azd ai rle init intel_lab --type byoh --protocol responses --endpoint "https://<account>.services.ai.azure.com/api/projects/<project>/agents/my-agent" --sample competitive_intelligence_agent
```

Use only the command for the chosen type. CLI versions can differ in folder
layout; find the scaffolded `rle.toml` and Dockerfile, then work in that build
context. For Harness, adapt/deploy the paired reference agent separately; don't
assume `init` deploys it or supplies a ready production service.

`init` installs project skills. To refresh them, run the following **from the
initialized RLE build context**, where the CLI can find a valid manifest:

```powershell
azd ai rle skill install --no-prompt
```

The verified installer reads this samples repository's `main` branch; it has
no source/ref override flag. It does not install an uncommitted local edit or
bootstrap an arbitrary empty directory. For an isolated trial, set
`AZD_CONFIG_DIR` to a new disposable profile before extension installation,
so the shared azd profile stays unchanged. Local development of the skill needs
an explicit local copy or a controlled disposable Git source; distinguish that
from testing the currently published skill.

For existing RLEs, preserve task selection/grading with baseline fixtures before
editing. Adapt task data, selector validation, tools, grader, app/agent wiring
and manifest in dependency order. Update the environment README with expected
rewards, selectors and commands.

## 3. Local verification without a model

Install dependencies using the sample's public SDK installer in an isolated
environment (see anatomy). Run tests from their documented working directory.
Gym sample checks from the samples repo root:

```powershell
python -m pytest -q examples\gym\openenv\math_rl\tests examples\gym\openenv\code_rl\tests
```

Check test output for skipped runtime tests: dependency-light contract tests
passing is not proof the SDK app runs. Scaffolded projects may have fewer tests;
add focused tests for changed behavior instead of assuming maintainer suites
were copied.

For Gym, `azd ai rle run` builds/serves the environment; inspect `run --help`
for watch options. It does **not** implement the Harness invocation loop.

For either type, plain container startup works from the environment context:

```powershell
docker build -t my-rle:local .
docker run --rm -p 127.0.0.1:8000:8000 my-rle:local
```

If package downloads fail TLS negotiation, diagnose the failing host first.
The sample Dockerfiles accept a `PIP_INDEX_URL` build argument for an approved
HTTPS package mirror. Use only a credential-free mirror as a build argument;
never bake tokens into image layers or history. Do not disable TLS verification
or replace the checksum-verified RLE wheel with an unrelated SDK to make a
build pass. Keep agent and RLE dependencies in separate images/venvs.

In another terminal:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/schema
```

Then execute the MCP reference's reset/tool/grade smoke test on a known task.
Assert task identity and exact reward for correct, wrong and malformed outputs;
unknown selectors must fail. Run two sessions and verify tool evidence doesn't
cross them. Test cleanup and call-before-reset/call-after-grade.

Exercise invalid selectors before the first valid reset as well as during an
episode; check raw booleans, strings and floats where types are strict. For a
default capacity of one, use sequential sessions; do not demand two simultaneous
sessions unless concurrency is explicitly enabled and proven safe.

The samples repo has a runnable Harness session client at
`examples/harness/byoh/_internal/data_code_agent/rle/scripts/session_demo.py`
and matching live tests; follow that internal README for arguments. Those are
maintainer tools, not files guaranteed in `init` output.

For BYOH, also test 202 acceptance, running/succeeded/failed polling,
string `output_text`, unique operations and idempotent DELETE. For HostedAgent,
test per-request header parsing, serialized first-turn input, background
Responses retrieval (or the deployed revision's synchronous mode), and final
output extraction. Use local fixtures for credentials, never real tokens in
checked-in test data.

Also test concrete model/MCP clients with `httpx.MockTransport` or an equivalent
local transport: assert full URL/query retention, the correct request-local
authorization, successful tool evidence, HTTP errors, HTTP-200 JSON-RPC errors,
and both `is_error` and `isError` tool results. Client-factory fakes alone skip
these boundaries. Do not call real models for these tests.

## 4. Deploy the Harness agent, then publish the environment

Gym skips agent deployment. BYOH deploys a reachable invocation endpoint;
HostedAgent deploys a Foundry agent and configures its endpoint traffic.
Smoke-test it and bind the actual HTTPS endpoint plus invocation protocol in
`[harness]`. A Responses target is the same-project agent URL, not a version.

Check the project's region against the current
[HostedAgent availability list](https://learn.microsoft.com/en-us/azure/foundry/agents/concepts/hosted-agents)
before deploying. A non-CMK account alone is not sufficient: at the verified
trial, Sweden Central was listed and a code-deployed host reached Active;
West US 2 was not listed. This does not prove every earlier 502 had that cause.
RLE access and model capacity need their own preflights. The Harness agent
URL must be in the environment's project; a different sampling
`policy.project_endpoint` does not rebind that agent to another project.

For HostedAgent container deployment from an existing ACR, verify the project's
`ContainerRegistry` connection as well as image-pull permissions. The current
SDK uses `ContainerConfiguration.registry_connection_id` (connection name or
ID); use the installed deployment tool's actual schema. This agent deployment
binding is not another field to invent in `rle.toml`.

Check the account's encryption configuration when diagnosing dependent-resource
authorization. The official
[CMK coverage documentation](https://learn.microsoft.com/en-us/azure/foundry/concepts/customer-managed-keys)
describes HostedAgent limitations, including platform-managed per-session
storage. A coverage limitation is **not proof** that CMK caused a specific 403.
Identify the actual failing resource/identity before proposing permissions;
do not disable CMK, alter firewalls or grant speculative Key Vault roles.

Set deployment context separately from `rle.toml`:

Confirm the intended existing project, registry, model deployment and scope
before publishing or starting billable rollouts. An active subscription alone
is not a chosen deployment target. Do not create infrastructure or grant roles
without approval. Authentication commands below are for the user's terminal;
if login is missing, ask the user to sign in rather than opening an interactive
login flow for them.

```powershell
$env:FOUNDRY_PROJECT_ENDPOINT = "https://<account>.services.ai.azure.com/api/projects/<project>"
$env:AZURE_CONTAINER_REGISTRY_ENDPOINT = "<registry>.azurecr.io"
az login
azd ai rle list --no-prompt
az acr login --name "<registry>"
```

The verified RLE CLI uses `DefaultAzureCredential`; an already authenticated
supported credential, such as Azure CLI, can suffice without `azd auth login`.
The separate azd HostedAgent deployment workflow can require its own login.
Check the applicable workflow rather than treating a missing core azd login
as proof that RLE authentication is unavailable.

Run the read-only RLE list preflight before deploying agents or publishing.
An HTTP 408 here is not a model/grader timeout: no episode has started.
Capture the request ID and confirm the selected endpoint and service access.
Do not create infrastructure or switch projects merely to bypass the error.

If ordinary ACR login fails but Azure CLI can issue a registry token, the
documented token-exchange flow can authenticate Docker without exposing it:

```powershell
$acr = az acr login --name "<registry>" --expose-token --output json --only-show-errors | ConvertFrom-Json
$acr.accessToken | docker login $acr.loginServer --username 00000000-0000-0000-0000-000000000000 --password-stdin
Remove-Variable acr
```

Check both commands succeeded. Never print/save the token object, put the token
in command-line arguments, enable registry admin keys, or disable TLS to retry.

The publishing user needs registry push permission. The Foundry **project's
system-assigned managed identity** needs registry pull permission (`AcrPull`);
local login does not grant it. Verify the project ARM identity and registry
scope before any role assignment, and follow the user's authorization policy.
Do not switch cloud resources or grant roles merely to bypass an error.

From the environment build context:

```powershell
azd ai rle publish
azd ai rle list
azd ai rle show <name>
```

Check image readiness, not just that a version record exists. Update
`rle.version` before publishing contract/image changes; use an immutable new
version. Don't overwrite another user's environment identity.

An ADC 400 followed by a generic image-pull/AcrPull hint is not proof of a
missing role. Verify the actual image, registry permission mode and project
identity; preserve the underlying error if conversion still fails after
authorized corrections. Image-format experiments should use a new immutable
version and compare actual readiness. Do not prescribe disabling provenance
or changing manifest media types unless that comparison establishes a fix.

## 5. Managed rollout: verify the real outcome

First inspect the installed command:

```powershell
azd ai rle version
azd ai rle rollout --help
```

At the verified baseline, **public `0.8.29-preview` is Loom-only** and does not
expose `--mode` or `--loom`; **development `0.8.30-preview`** implements the
backend choices below. Do not infer flag availability from development source.
Update only through the intended extension channel when it supplies the
required feature. Otherwise report CLI eval as unavailable; a separately
verified authenticated REST test is not a successful CLI rollout.
Source support and CLI support also do not establish deployed service support:
a successful WebSocket upgrade can still be followed by `InvalidRolloutRequest`
for a Foundry policy. Preserve that error and verify the deployed capability;
do not rewrite a working environment or silently substitute a Loom policy.
Never drop `--mode eval` and pass a deployment name to the default Loom flow,
or fabricate training-session/checkpoint IDs to make it run.

Choose the rollout backend deliberately:

- `--mode train` (default) creates a Loom training session and sampler checkpoint
  for a base model; capture includes token IDs/logprobs needed for training.
- `--mode eval` uses a Foundry **model deployment name** and no Loom session.
  Capture is text-only and intentionally never trainable.
- `--mode eval --loom` evaluates through Loom instead of a Foundry deployment.

For a Foundry deployment smoke test:

```powershell
azd ai rle rollout --mode eval --model <deployment-name> --task-file .\task.json
```

Run from the environment build context containing `rle.toml`. The current
command reads `rle.name` and `rle.version` from that manifest; it has no
`--version` flag. Check the selected immutable version before execution.

For a trainable Loom rollout:

```powershell
azd ai rle rollout --help
azd ai rle rollout --mode train --model <base-model> --task '{"seed":0,"split":"train"}'
```

That task is for Gym math/code. Use BYOH's actual dataset split/index or
HostedAgent's complete task, not the same seed payload for every sample.
Prefer `--task-file` where supported for complex JSON/Windows native quoting.
Reset's observation supplies Harness first-turn input. `--agent-input` and
`--agent-input-file` have been removed; task is the only caller-supplied input.
Rollout can default its model from `[model].name`, but an eval deployment and a
Loom base-model ID are different identifiers.

Verify several known tasks, including held-out data:

1. Task/episode identity matches the request, including split.
2. Grade is terminal, finite and consistent with the submitted answer and rubric.
3. MCP tool evidence belongs to the same session that was reset and graded.
4. Harness model calls were captured through the supplied proxy.
5. No output truncation or cleanup failure is hidden. Require trainability for
   training trajectories, not text-only Foundry evaluation captures.

At the verified capture-export baseline, model calls are in `rollout_graph.turns`,
not a guessed `nodes` array. Check the actual exporter/schema for the deployed
revision. Loom additionally requires `capture_level: "tokens"`, `trainable: true`
and trainable sequences with aligned `input_ids`, `loss_mask` and `logprobs`;
sampled tokens selected by the mask must have finite logprobs.
Persist a credential-free structural diagnostic before asserting the schema,
then save a validated summary of task, terminal steps, reward, model-turn count
and trainability. Never dump an unknown raw graph or credentials just to debug
its shape. An evidence-parser failure after service completion is not a failed
episode; keep those outcomes separate and don't replay an accepted execution.

A stochastic model need not earn the scripted reference answer's reward.
Compare the actual submitted answer to the local grader rather than expecting
every rollout on that seed to be correct.

After a publish/deploy timeout, look up the unique environment/version or
agent/version before retrying a write: a timeout does not prove it failed to
commit. Version existence alone is not image readiness or a successful rollout.
Report registry pushes, resource registration, readiness, direct model calls
and managed RLE execution separately.

Loom HTTP 503 `engine_busy` / "No engine available" is a sampling-capacity
failure before the environment runs. Honor the reported retry interval with
bounded retries; use only documented available base models, not guessed IDs.
Do not fabricate a session or checkpoint, borrow another user's allocation,
or change the grader to make this pre-episode failure appear successful.

Record owned resources and close any created Loom session on success or failure.
A session-creation timeout can still leave a queued session: compare safe
resource metadata with the preflight snapshot before retrying. Recover and
close only an attributable session through the supported lifecycle API, then
verify the response and resource state; do not guess status semantics or delete
unattributed sessions. Remove only temporary role assignments/connections
created for the trial, and verify their absence. Preserve pre-existing access.

## 6. Training handoff and skill refresh

After verified rollouts, use the checked-in `job_data/` selectors/complete tasks
for training and validation. A typical sample invocation from its directory:

```powershell
$env:AZD_AI_RLE_ENABLE_ALL = "true"
azd ai rle train --model <model> --suffix <suffix> --training-file .\job_data\train.jsonl --validation-file .\job_data\validation.jsonl
```

Inspect `train --help`; keep tuning outside this authoring workflow. Model,
training file, validation file and suffix are flag inputs, not a
`[train.options]` table. Don't silently include held-out examples in the
environment's training picker.

`init` copies the authoring skills; `azd ai rle skill install` refreshes them
where supported. Check `skill --help` and preserve local customizations before
refreshing. Agents can then be prompted, for example:

- "Use the rle skill to create a Gym math environment with one typed helper."
- "Use the rle skill to wrap my existing agent as BYOH without changing its loop."
- "Edit this HostedAgent RLE's tool schema and grader; test session isolation."

If publishing isn't possible, deliver tested local edits plus the exact remaining
commands and blocker. Do not claim cloud verification from static checks.
