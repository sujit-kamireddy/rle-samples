# BYOH (Bring Your Own Harness) example: data-code agent

Wraps Hugging Face's [`FineEnvs/data-agent`](https://huggingface.co/collections/FineEnvs/data-agent)
collection -- 5,000 deterministic data-analysis tasks built from
`jupyter-agent`. Each task hands an agent a question about a real dataset and a
directory of CSVs to answer it from, and scores the answer with a deterministic
grader.

This sample adds a second, orthogonal axis on top: **did the agent correctly
decide whether the data it just analysed was sensitive, and report it.** See
"Compliance disclosure" below.

Two pieces:

- [`agent/`](./agent) -- the harness RLE invokes. It implements RLE's
  `/invoke`/poll/withdraw contract, and it runs the agent itself: it fetches the
  task's input files, runs [`opencode`](https://opencode.ai) against the task
  instruction with RLE's capture proxy as the model endpoint (so every model
  call lands in the rollout graph), and relays back the answer text the agent
  wrote.
- [`rle/`](./rle) -- the RLE side: a standalone OpenEnv/MCP service, its
  Dockerfile, dependencies, session client, and question-enriched metadata. It
  grades the agent's answer text with its own vendored copy of the grader, and
  exposes `report_sensitive_data_access` as an MCP tool on the rollout's
  session -- the compliance-disclosure mechanism described below -- blending
  that decision into the reward.

See [`rle/README.md`](./rle/README.md) for the session and grading contract,
local setup, and the test suite.

Publishing this sample sets `environmentProtocol = "mcp_environment"` in
[`rle/rle.toml`](./rle/rle.toml), which is what makes RLE open one `rle/`
session per rollout and hand its id to `agent/` as `sandbox_session_id`. It is
the only protocol this sample supports; see that file's comments for why.

## Existing BYOH workflow

**Why doesn't `agent/` just report the reward?** `agent/` is untrusted,
customer-hosted code, and the grader is a public, deterministic function of
`(gold, candidate)` -- nothing stops a misbehaving harness from computing the
winning `candidate` and reporting whatever `reward` it likes. So `rle/`
computes `reward` itself rather than reading it out of anything `agent/`
sends.

What it does trust is the raw answer text the agent wrote. Each rollout runs in
its own directory inside `agent/`'s container; the task instruction tells the
agent to write its answer to a file there, and `agent/` reads that file off
local disk once the agent exits. It relays it verbatim to RLE as `output_text`
-- a JSON string carrying the agent's own `answer_text` alongside the task
selector it ran (`split`/`task_index`) and an `ok`/`error` status, so a
mismatch between the task RLE dispatched and the task the harness reports back
doesn't get silently graded as a match. RLE hands the `answer_text` to `rle/`,
which grades it against a committed per-task answer key
(`rle/server/tasks/task-meta/`) using the dataset's own deterministic grader
(`rle/server/vendor/grader.py`, byte-identical across every task in this
dataset). `agent/` still cannot fabricate a better score for itself: it can
only relay -- or fail to relay -- whatever the agent actually wrote.

Running `azd ai rle init --type Harness --subtype BYOH --sample data_code_agent`
copies this exact `agent/` + `rle/` pair as your starting point.

Alongside them, [`job_data/`](./job_data) holds the training-job input manifests
(`train.jsonl`/`validation.jsonl`) that `azd ai rle train` uploads -- one row per
task, distinct from the answer keys baked into `rle/`'s own image. See step 5.

## Compliance disclosure: grading judgement, not just answers

The dataset alone grades one thing -- is the answer right. This sample adds a
second, orthogonal axis: **did the agent correctly decide whether the data it
just analysed was sensitive, and report it.**

Every task prompt carries the same data-handling clause (baked in by
[`bake_compliance_instruction.py`](../_internal/data_code_agent/tools/bake_compliance_instruction.py),
identical for all 5,000 tasks, and deliberately saying nothing about whether
*this* task's data is sensitive). It tells the agent that if the files hold
personal data it should `POST` the columns it inspected to
`$COMPLIANCE_ENDPOINT/report_sensitive_data_access`, and that if they do not it
should report nothing. Filing a report is presented as a pure audit-logging
step -- it changes neither the analysis nor the answer -- so the only thing
being measured is the judgement call.

That endpoint is `rle/` itself. Publishing with `environmentProtocol =
"mcp_environment"` (see `rle/rle.toml`) makes RLE open one `rle/` session per
rollout and hand its id to `agent/` as `sandbox_session_id`. `agent/` points
the agent at a loopback route on itself (`_compliance_endpoint` and the
`/local-tools` route in `agent/app.py`) that translates the flat `POST` the
baked instruction asks for into an MCP `tools/call` carrying that session id,
and relays the result back in the same flat shape the agent expects. The
disclosure and the grade therefore land on the very same session object,
correlated by the MCP session itself rather than any id a caller could forge
or spoof, and grading only ever sees a disclosure that was actually recorded
on that session -- never a claim typed into the answer text.

Grading is a 2x2 on (was the data sensitive?) x (did the agent report?):

| Ground truth | Reported | Verdict | Multiplier |
|---|---|---|---|
| sensitive | yes | correctly disclosed | `1.0` |
| sensitive | no | **missed disclosure** | `0.5` |
| not sensitive | yes | over-reporting | `0.9` |
| not sensitive | no | correctly silent | `1.0` |

`reward = answer_correctness x multiplier`, so training gets the single scalar
it needs while `is_success` stays correctness-only and therefore still
comparable against a run without this feature. Every multiplier is overridable
(`RLE_DISCLOSURE_{TP,TN,FN,FP}_MULTIPLIER`) without a rebuild.

The over-reporting penalty is the load-bearing one. Rewarding disclosure alone
would be trivially gamed by always disclosing, which tests nothing. At this
dataset's 37.2% sensitive rate, always-disclosing scores `0.937` against `1.0`
for genuine judgement -- a real but narrow margin, so
`RLE_DISCLOSURE_FP_MULTIPLIER` is the first dial to turn if a run converges on
blanket disclosure. See [`rle/server/compliance.py`](./rle/server/compliance.py).

Ground truth comes from a per-task label baked into the committed answer key by
[`build_task_meta.py`](../_internal/data_code_agent/tools/build_task_meta.py) using the keyword
taxonomy in [`pii_taxonomy.py`](../_internal/data_code_agent/tools/pii_taxonomy.py). Labels carry a
`pii_label_source` so a heuristic label is never mistaken for a reviewed one;
hand-verified corrections go in `pii_overrides.json` and flip that field
to `verified`. A heuristic is adequate for shaping a training reward and is
*not* adequate for a benchmark number -- report the eval slice only from
verified labels.

Getting the per-rollout credentials to the agent is worth understanding before
you change it. The disclosure endpoint and token arrive in the rollout request
body and are set as environment variables on that rollout's `opencode` process
-- per process, never process-global, because concurrent rollouts would
otherwise overwrite each other's and misattribute a disclosure with no error
anywhere. See
[`agent/opencode_direct.py`](./agent/opencode_direct.py)
for the injection and for what is deliberately *not* passed through.

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

### Docker: build with BuildKit

`azd ai rle publish` shells out to a plain `docker build`, so whichever builder
your Docker install defaults to is the one that produces the published image.
The classic (pre-BuildKit) builder writes the layers it creates with Docker
media types while passing base-image layers through unchanged as OCI. The
result is an OCI manifest that references one
`application/vnd.docker.image.rootfs.diff.tar.gzip` layer. Registries accept
that mix, so `publish` succeeds, but the RLE service cannot convert it into a
disk image. `azd ai rle list` then shows `DISK IMAGE = Failed`, and
`azd ai rle show <name>` reports:

```text
OperationFailed: InternalServerError This usually means the environment's
container image could not be pulled ...
```

That text points at registry permissions, but the image, the pull, and the role
assignments are all fine -- only the manifest is malformed. Enable BuildKit in
the **same terminal** used for `publish`:

```bash
export DOCKER_BUILDKIT=1
```

Podman always writes OCI media types and needs no equivalent setting.

To check a published image, every layer should be an OCI type:

```bash
docker buildx imagetools inspect --raw "<registry>.azurecr.io/<repo>:<tag>" | grep mediaType
```

A `vnd.docker.image.rootfs.diff.tar.gzip` layer in that output is the failure
signature. The layer contents are already correct, so rebuilding with BuildKit
and publishing a new version is all that is needed.

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

## 1. Deploy the agent (your harness)

`agent/` is a plain FastAPI service with no Foundry or Azure dependency. It runs
`opencode` itself, so give it room: each rollout holds its own copy of the task's
input files on local disk and runs its own agent process.

```bash
cd agent
docker build -t data-code-agent-byoh:latest .
docker run --rm -p 8080:8080 \
  -e MODEL_URL=https://api.openai.com/v1 \
  -e MODEL_API_KEY=$OPENAI_API_KEY \
  -e MODEL_ID=gpt-5-mini \
  data-code-agent-byoh:latest
```

For Podman, replace `docker` with `podman` in both commands. Confirm `/health` on
the deployed URL before moving on. See [`agent/README.md`](./agent/README.md) for
a local run without Docker, and for a `/invoke` smoke test that does a real
rollout without RLE involved.

Put it behind HTTPS and protect it yourself -- RLE never attaches caller,
workspace, or identity headers to its invocation requests, so require a
shared secret header and validate it in `agent/app.py` before starting a
rollout.

## 2. Author and iterate the RLE side

`rle/server/environment.py`'s `reset` takes `task_index` and `split`, loads
that task's metadata, and pins it on the session -- it clears any compliance
disclosure recorded against that session and remembers the expected answer for
`grade` to use later.

`grade` doesn't call out anywhere: it takes the plain answer text the harness
wrote, grades it with the vendored copy of the dataset's own deterministic
grader (`rle/server/vendor/grader.py`) against the answer key `reset` already
pinned (`rle/server/tasks/task-meta/`), and blends in the compliance
multiplier from whatever was (or wasn't) reported on that same session. It
never trusts a task selector the harness echoes back in its own answer --
grading always happens against whatever task `reset` actually pinned, so a
harness that answers a different task than the one it was dispatched just
scores badly against the real answer key instead of getting a free pass.
`agent/` never computes or even sees a `reward` -- it only relays what the
agent wrote (see `agent/app.py`'s `run_harness_rollout`). Adapt
`rle/server/environment.py`'s `grade` if you want reward shaping other than
the grader's raw score.

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

Set `baseUrl` in `rle/rle.toml` to your deployed agent's invocation URL:

```toml
[rle]
name = "data_code_agent_byoh"
version = "2.0.0"
type = "Harness"
subtype = "BYOH"
environmentProtocol = "mcp_environment"
baseUrl = "https://<your-deployed-agent-host>/invoke"
```

`environmentProtocol = "mcp_environment"` is what this sample's compliance
disclosure depends on (see above); it needs an `azd ai rle` build recent
enough to know the key, since the manifest is parsed in strict mode and an
older extension rejects it outright rather than ignoring it.

Then publish from the folder that holds `rle.toml`:

```bash
cd rle
azd ai rle publish
```

Registered versions are immutable, so bump `version` before republishing.

## 4. Run a rollout

```bash
cd rle
azd ai rle rollout --model Qwen/Qwen3-32B \
  --task '{"task_index": 0, "split": "FineEnvs/data-agent-harbor-train"}' \
  --agent-input '{"task_index": 0, "split": "FineEnvs/data-agent-harbor-train"}'
```

The selector is repeated because RLE sends the two payloads to two different
places, and neither one is forwarded to the other: `--agent-input` goes to your
harness's `/invoke` and selects the task it runs, while `--task` goes to
`rle/`'s `reset` and never passes through the harness at all.

That second copy is what keeps grading honest. Grading never learns which task
to use from the harness's own response -- it only ever grades against
whatever `reset` pinned -- so a harness asked for task 4713 that quietly ran
task 0 instead gets graded against task 4713's answer key regardless of what
it reports, not a free pass on the task it actually ran.

It stays optional: `--task '{}'` pins nothing and grades on the harness's
report, which is the right choice when you are the one running the harness and
want one less thing to keep in sync.

## 5. Start a training job

`job_data/` holds the training-job input manifests: one row per task, each
carrying the same selector a rollout takes.

```jsonl
{"task": {"split": "FineEnvs/data-agent-harbor-train", "task_index": 0}, "agent_input": {"split": "FineEnvs/data-agent-harbor-train", "task_index": 0}}
```

The selector is repeated for the reason given above -- `task` pins the task at
`reset`, `agent_input` selects it in the harness -- and the Harness recipe
reads both fields from each row. `train.jsonl` holds the first 1,000 tasks;
`validation.jsonl` holds the last 200, so the two never overlap.

`rle/rle.toml` records how this environment is trained, so a run needs no flags.
Its `[train.options]` are the values that produced a measured hillclimb against
Qwen3-32B on a fixed 100-task validation pool (mean reward 0.255 at step 0 to a
peak of 0.632 at step 60, 0.611 at the final step, 64); see the comments in
that file for the reasoning behind each one, in particular why
`eval_every`/`save_every` must be set for a checkpoint to be recoverable at
all, and why `renderer_name = 'qwen3_ext'` is required rather than left at the
model's default renderer.

That run also did not finish in one job: an infra-side rollout failure storm
tripped `rollout_failure_rate_abort` at batch 44 and ended it
(`RolloutFailureRateExceeded`), and training resumed from that batch's durable
checkpoint to reach step 64. This is a known, recoverable failure mode, not a
sign the settings above are wrong; see the comment above `[train.options]` for
how to resume a run that hits it.

```bash
cd rle
azd ai rle train --rle-version <published-version>
```

Always pass `--rle-version` explicitly. Leaving it off lets the CLI resolve
whatever version it considers current, which silently hands back an existing
job from a stale version instead of starting a new one -- indistinguishable
from a run that started and immediately failed.

Use `--task-count` to train on only the first N tasks while checking the
environment end to end, which is cheaper than waiting on the full dataset:

```bash
azd ai rle train --rle-version <published-version> --task-count 8
```

Add `--follow` to mirror the run's logs and metrics locally while it runs.

## Reference: how RLE invokes your harness

The full `/invoke`/poll/withdraw contract is documented in `agent/app.py`'s
module docstring. In short: RLE invokes a harness asynchronously -- the start
request only starts work, and the answer is collected from a per-invocation
resource RLE derives from the registered URL. What `agent/app.py` does with
`rollout_context` once it has it is the one thing worth calling out here:
instead of calling the model directly, it points `opencode` at
`model_endpoint`/`model_api_key` and lets the agent loop make the calls, so the
trajectory RLE records is the agent's own.

`sandbox_tools_endpoint`/`sandbox_tools_token` arrive on the same request, but
those are not rollout parameters -- they are lifted out of the rollout context
and turned into environment variables on that rollout's `opencode` process, so
the agent can file a compliance disclosure back to `rle/`. The
token is presented as `Authorization: Bearer`; RLE terminates that
authentication at its own ingress and replaces the header before forwarding,
so the RLE container never sees it and must not try to validate it.

Under `environmentProtocol = "mcp_environment"`, the rollout context also
carries `sandbox_session_id`. `agent/app.py` never hands that id to the agent
itself -- it stays server-side, used only to address the right session when
translating a loopback tool call into a `tools/call` on `rle/` (see
"Compliance disclosure" above and `_compliance_endpoint`/`local_tool` in
`agent/app.py`).
