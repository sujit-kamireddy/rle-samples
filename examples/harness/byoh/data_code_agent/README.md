# BYOH (Bring Your Own Harness) example: data-code agent

Wraps Hugging Face's [`FineEnvs/data-agent`](https://huggingface.co/collections/FineEnvs/data-agent)
collection -- 5,000 deterministic data-analysis tasks built from
`jupyter-agent`. Each task hands an agent a question about a real dataset and a
directory of CSVs to answer it from, and scores the answer with a deterministic
grader.

It also grades a second, orthogonal axis: did the agent correctly judge
whether the data it analysed was sensitive, and disclose it. See "Compliance
disclosure" below.

Two pieces:

- [`agent/`](./agent) -- the harness RLE invokes. A plain FastAPI service
  implementing RLE's `/invoke`/poll/withdraw contract. It runs
  [`opencode`](https://opencode.ai) against the task instruction with RLE's
  capture proxy as the model endpoint (so every model call lands in the
  rollout graph), and relays back the answer text the agent wrote.
- [`rle/`](./rle) -- the RLE side: an OpenEnv/MCP service that grades that
  answer text with its own vendored copy of the dataset's grader, and exposes
  `report_sensitive_data_access` as an MCP tool for the compliance-disclosure
  check.

See [`rle/README.md`](./rle/README.md) for the session and grading wire
contract, local setup, and the test suite.

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

`azd ai rle publish` runs a plain `docker build`, so whichever builder your
Docker install defaults to produces the published image. The classic
(pre-BuildKit) builder can emit an OCI manifest that registries accept but the
RLE service cannot convert into a disk image: `azd ai rle list` then shows
`DISK IMAGE = Failed`, and `azd ai rle show <name>` blames registry
permissions even though the real cause is the manifest. Avoid it by enabling
BuildKit in the **same terminal** used for `publish`:

```bash
export DOCKER_BUILDKIT=1
```

Podman always writes OCI media types and needs no equivalent setting. If you
already hit this, just rebuild with BuildKit set and publish a new version --
`docker buildx imagetools inspect --raw "<registry>.azurecr.io/<repo>:<tag>" | grep mediaType`
should show only OCI layer types.

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
that task's metadata, and pins it on the session as the expected answer for
`grade` to use later, clearing any prior compliance disclosure on that
session.

`grade` takes the plain answer text the harness wrote, scores it with the
vendored grader (`rle/server/grader.py`) against the answer key `reset`
pinned (`rle/server/tasks/task-meta/`), and blends in the compliance
multiplier from whatever was (or wasn't) reported on that session -- never a
task selector the harness echoes back itself (see step 4 for why).
`agent/` never computes or even sees a `reward`; it only relays what the agent
wrote (see `agent/app.py`'s `run_harness_rollout`). Adapt `grade` if you want
reward shaping other than the grader's raw score.

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
disclosure depends on (see below); it needs an `azd ai rle` build recent
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

`rle/rle.toml` records how this environment is trained, so a run needs no
flags. Its `[train.options]` are pre-tuned for this environment and produced a
validated hillclimb against Qwen3-32B -- see that file's comments for the
reasoning behind each one and for how to resume a run if
`rollout_failure_rate_abort` trips.

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

## Why `rle/` grades, not `agent/`

`agent/` is untrusted, customer-hosted code, and the grader is a public,
deterministic function of `(gold, candidate)` -- nothing stops a misbehaving
harness from computing the winning `candidate` and reporting whatever
`reward` it likes. So `rle/` computes `reward` itself rather than reading it
out of anything `agent/` sends.

What it does trust is the raw answer text the agent wrote. Each rollout runs
in its own directory inside `agent/`'s container; the task instruction tells
the agent to write its answer to a file there, and `agent/` reads that file
off local disk once the agent exits and relays it verbatim to RLE. RLE hands
that text to `rle/`, which grades it against a committed per-task answer key
(`rle/server/tasks/task-meta/`) using the dataset's own deterministic grader
(`rle/server/grader.py`). `agent/` cannot fabricate a better score for
itself -- it can only relay, or fail to relay, whatever the agent actually
wrote.

## Compliance disclosure: grading judgement, not just answers

This sample grades a second, orthogonal axis on top of answer correctness:
**did the agent correctly decide whether the data it just analysed was
sensitive, and report it.**

Every task prompt carries the same data-handling clause (identical for all
5,000 tasks, and deliberately saying nothing about whether *this* task's data
is sensitive): disclose the columns inspected if the files hold personal
data, report nothing if they do not. Filing a report changes neither the
analysis nor the answer, so the only thing being measured is the judgement
call. The agent files it as a plain `POST`; `agent/` translates that into an
MCP `tools/call` on the rollout's `rle/` session, so the disclosure and the
grade land on the same session object and nothing typed into the answer text
can forge one. See [`agent/README.md`](./agent/README.md#rollout-isolation)
for how that plumbing works.

Grading is a 2x2 on (was the data sensitive?) x (did the agent report?):

| Ground truth | Reported | Verdict | Multiplier |
|---|---|---|---|
| sensitive | yes | correctly disclosed | `1.0` |
| sensitive | no | **missed disclosure** | `0.5` |
| not sensitive | yes | over-reporting | `0.9` |
| not sensitive | no | correctly silent | `1.0` |

`reward = answer_correctness x multiplier`, so `is_success` stays
correctness-only and comparable against a run without this feature. Every
multiplier is overridable (`RLE_DISCLOSURE_{TP,TN,FN,FP}_MULTIPLIER`) without a
rebuild -- `RLE_DISCLOSURE_FP_MULTIPLIER` is the first to turn down if a run
converges on blanket disclosure. See
[`rle/server/compliance.py`](./rle/server/compliance.py).

Ground truth comes from a per-task label built from a keyword taxonomy (see
[`../_internal/data_code_agent/tools/`](../_internal/data_code_agent/tools)),
not a human review. It is adequate for shaping a training reward; report an
eval number only from the labels marked `verified` in `pii_overrides.json`.

## Reference: how RLE invokes your harness

The full `/invoke`/poll/withdraw contract is documented in `agent/app.py`'s
module docstring. In short: RLE invokes a harness asynchronously -- the start
request only starts work, and the answer is collected from a per-invocation
resource RLE derives from the registered URL. `agent/app.py` never calls the
model directly: it points `opencode` at `model_endpoint`/`model_api_key` and
lets the agent loop make the calls, so the trajectory RLE records is the
agent's own.

`mcp_endpoint`/`mcp_bearer_token` (and, under
`environmentProtocol = "mcp_environment"`, `mcp_session_id`) arrive on the
same request but are not rollout parameters -- they are lifted out of the
rollout context into environment variables on that rollout's `opencode`
process, so the agent can file a compliance disclosure back to `rle/` (see
"Compliance disclosure" above). The agent itself never sees the session id;
`agent/app.py` uses it server-side to address the right session.
