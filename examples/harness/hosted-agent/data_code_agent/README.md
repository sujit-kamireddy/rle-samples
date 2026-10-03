# Hosted Agent example: data-code agent

Wraps Hugging Face's [`FineEnvs/data-agent`](https://huggingface.co/collections/FineEnvs/data-agent)
collection -- 5,000 deterministic data-analysis tasks built from
`jupyter-agent`. Each task hands an agent a question about a real dataset and a
directory of CSVs to answer it from, and scores the answer with a deterministic
grader.

This sample adds a second, orthogonal axis on top: **did the agent correctly
decide whether the data it just analysed was sensitive, and report it.** See
"Compliance disclosure" below.

Two containers:

- [`agent/`](./agent) -- the Foundry Hosted Agent RLE invokes. It serves the
  Responses protocol and reads each rollout's runtime context from the
  `x-client-rle-*` request headers, and it runs the agent itself: it fetches the
  task's input files, runs [`opencode`](https://opencode.ai) against the task
  instruction with RLE's capture proxy as the model endpoint (so every model
  call lands in the rollout graph), and relays back the answer text the agent
  wrote.
- [`rle_deprecated/`](./rle_deprecated) -- the RLE container. `/grade` scores that answer text with its
  own vendored copy of the grader. It also serves
  `/tools/report_sensitive_data_access`, the compliance-disclosure endpoint
  described below, and blends that decision into the reward.

## Standalone OpenEnv alternative

[`rle/`](./rle) contains the standalone OpenEnv service, its
Dockerfile, dependencies, session client, tests, and question-enriched metadata.
It carries its own copy of the grader and compliance evaluator, kept in
lockstep with the legacy ones by the parity suites, and changes neither the
existing Harness API or deployment.

See its [README](./rle/README.md) for local setup, full-dataset parity
checks, and concurrent-instance validation. The second phase, deployment using
Foundry RLE, remains **TBA**.

## Existing Harness workflow

**Why doesn't `agent/` just report the reward?** `agent/` is untrusted,
customer-hosted code, and the grader is a public, deterministic function of
`(gold, candidate)` -- nothing stops a misbehaving harness from computing the
winning `candidate` and reporting whatever `reward` it likes. So `rle_deprecated/`'s
`/grade` computes `reward` itself rather than reading it out of anything
`agent/` sends.

What it does trust is the raw answer text the agent wrote. Each rollout runs in
its own directory inside `agent/`'s container; the task instruction tells the
agent to write its answer to a file there, and `agent/` reads that file off
local disk once the agent exits. It relays it verbatim to RLE as `output_text`;
RLE forwards it to `/grade` as `agent_response`, and `/grade` grades it against a
vendored per-task answer key (`rle_deprecated/server/vendor/task-meta/`) using the dataset's
own deterministic grader (`rle_deprecated/server/vendor/grader.py`, byte-identical across
every task in this dataset). `agent/` still cannot fabricate a better score for
itself: it can only relay -- or fail to relay -- whatever the agent actually
wrote.

Running
`azd ai rle init --type Harness --subtype HostedAgent --sample data_code_agent`
copies this exact `agent/` + `rle_deprecated/` pair as your starting point. The `BYOH`
copy of this sample runs the same rollout against the same `rle_deprecated/` container and
differs only in how RLE invokes it -- a request body it defines itself, polled
asynchronously, instead of headers on a synchronous Responses call. Scaffold it
with `--subtype BYOH`.

Alongside them, [`job_data/`](./job_data) holds the training-job input manifests
(`train.jsonl`/`validation.jsonl`) that `azd ai rle train` uploads -- one row per
task, distinct from the answer keys baked into `rle_deprecated/`'s own image. See step 5.

## Compliance disclosure: grading judgement, not just answers

The dataset alone grades one thing -- is the answer right. This sample adds a
second, orthogonal axis: **did the agent correctly decide whether the data it
just analysed was sensitive, and report it.**

Every task prompt carries the same data-handling clause (baked in by
[`tools/bake_compliance_instruction.py`](./tools/bake_compliance_instruction.py),
identical for all 5,000 tasks, and deliberately saying nothing about whether
*this* task's data is sensitive). It tells the agent that if the files hold
personal data it should `POST` the columns it inspected to
`$COMPLIANCE_ENDPOINT/report_sensitive_data_access`, and that if they do not it
should report nothing. Filing a report is presented as a pure audit-logging
step -- it changes neither the analysis nor the answer -- so the only thing
being measured is the judgement call.

That endpoint is the RLE container itself. RLE gives every rollout a
per-rollout `sandbox_tools_endpoint` and bearer token, forwards
`/tools/<name>` through to the container that also serves `/reset` and
`/grade`, and stamps every forwarded request with a **server-owned**
`x-rle-rollout-id` that it refuses to let a caller set. So the disclosure and
the grade land in the same process, correlated by an id the agent cannot
forge, and `/grade` verifies the call was actually made rather than believing
a claim in the answer text.

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
blanket disclosure. See [`rle_deprecated/server/compliance.py`](./rle_deprecated/server/compliance.py).

Ground truth comes from a per-task label baked into the vendored answer key by
[`tools/build_task_meta.py`](./tools/build_task_meta.py) using the keyword
taxonomy in [`tools/pii_taxonomy.py`](./tools/pii_taxonomy.py). Labels carry a
`pii_label_source` so a heuristic label is never mistaken for a reviewed one;
hand-verified corrections go in `tools/pii_overrides.json` and flip that field
to `verified`. A heuristic is adequate for shaping a training reward and is
*not* adequate for a benchmark number -- report the eval slice only from
verified labels.

Getting the per-rollout credentials to the agent is worth understanding before
you change it. The disclosure endpoint and token arrive as the
`x-client-rle-sandbox-tools-endpoint` / `x-client-rle-sandbox-tools-token`
request headers and are set as environment variables on that rollout's `opencode` process
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

## 1. Build and register the agent

`agent/` is a plain Responses-protocol service built on
`azure-ai-agentserver-responses`, with no other Foundry or Azure dependency. It
runs `opencode` itself, so give it room: each rollout holds its own copy of the
task's input files on local disk and runs its own agent process.

```bash
cd agent
docker build -t data-code-agent-hosted:latest .
docker run --rm -p 8088:8088 \
  -e MODEL_URL=https://api.openai.com/v1 \
  -e MODEL_API_KEY=$OPENAI_API_KEY \
  -e MODEL_ID=gpt-5-mini \
  data-code-agent-hosted:latest
```

For Podman, replace `docker` with `podman` in both commands. Confirm
`/readiness` before moving on. See [`agent/README.md`](./agent/README.md) for a
local run without Docker, and for a Responses smoke test that does a real
rollout without RLE involved.

Then push the image to the registry your Foundry project can pull from and
register it as a Hosted Agent in the project. Note its **name** and
**version**: those are what `rle_deprecated/rle.toml` points at in step 3.

Unlike BYOH there is nothing to expose publicly and no endpoint of your own to
protect. RLE resolves the agent from the project and calls it with the caller's
own bearer token, so the agent is reachable only through Foundry.

## 2. Author and iterate the RLE side

`rle_deprecated/server/env.py`'s `/reset` returns nothing RLE reads -- only its status
code matters -- but it is not idle. It clears any compliance disclosure left
under this rollout id, and it pins the task selector (`task_index`, `split`)
if `--task` carries one, so `/grade` can check what the harness reports
against what RLE asked for (see step 4).

`/grade` doesn't call out anywhere: it parses
`answer_text`, `task_index`, and `split` out of `agent_response` (the same
JSON string `agent/` returned as its final assistant message, which RLE takes as
`output_text` and forwards to `/grade` verbatim), looks up that task's answer key in the
vendored `rle_deprecated/server/vendor/task-meta/` metadata, and grades it with the
vendored copy of the dataset's own deterministic grader
(`rle_deprecated/server/vendor/grader.py`). Where `/reset` pinned a task, a report naming
a different one scores zero instead. `agent/` never computes or even sees a
`reward` -- it only relays what the agent wrote (see `agent/main.py`'s
`run_harness_rollout`). Adapt `rle_deprecated/server/env.py`'s `grade_rollout` if you
want reward shaping other than the grader's raw score.

`azd ai rle run` is supported only for `Gym: OpenEnv` environments, so iterate
here by publishing a version and running a rollout (steps 3 and 4).

## 3. Register the agent and publish

Once the Hosted Agent is registered in your project, point the manifest at it
and publish a version. Complete [registry setup](#before-the-first-publish) and
set `FOUNDRY_PROJECT_ENDPOINT` and `AZURE_CONTAINER_REGISTRY_ENDPOINT` in the
same terminal first:

```powershell
$env:FOUNDRY_PROJECT_ENDPOINT = "https://<account>.services.ai.azure.com/api/projects/<project>"
$env:AZURE_CONTAINER_REGISTRY_ENDPOINT = "<registry>.azurecr.io"
```

Set `agentName` and `agentVersion` in `rle_deprecated/rle.toml` to the registered agent:

```toml
[rle]
name = "data_code_agent_hosted_agent"
version = "1.0.0"
type = "Harness"
subtype = "HostedAgent"
agentName = "<your-registered-agent-name>"
agentVersion = "1"
```

`baseUrl` is rejected for this subtype -- there is no URL to register. Every
other setting `rle.toml` accepts is listed in that file, commented out where
this environment leaves it unset.

`agentVersion` must be a positive integer, or `draft-{unix-timestamp}` for an
unpublished draft. RLE records it as target metadata and does not send it to the
agent, so the container cannot read its own version.

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
places, and neither one is forwarded to the other: `--agent-input` is serialised into the
Responses input text your agent receives, and selects the task it runs, while `--task` goes to
`rle_deprecated/`'s `/reset` and never passes through the harness at all.

That second copy is what lets `/grade` check the harness's work. `/grade`
otherwise learns which task it is grading from the harness's own response, so a
harness asked for task 4713 could run task 0, report task 0, and be graded
correctly against task 0 -- no forged reward required, just a quietly
re-selected task. `/reset` pins whatever selector `--task` carries, and `/grade`
scores zero if the harness reports a different one.

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
`/reset`, `agent_input` selects it in the harness -- and the Harness recipe
reads both fields from each row. `train.jsonl` holds the first 1,000 tasks;
`validation.jsonl` holds the last 200, so the two never overlap.

`rle_deprecated/rle.toml` records how this environment is trained, so a run needs no flags:

```toml
[train]
model = "qwen3-32b-1"
training_file = "../job_data/train.jsonl"
validation_file = "../job_data/validation.jsonl"

[train.options]
group_size = 4
max_concurrent_rollouts = 8
```

```bash
cd rle
azd ai rle train
```

Use `--task-count` to train on only the first N tasks while checking the
environment end to end, which is cheaper than waiting on the full dataset:

```bash
azd ai rle train --task-count 8
```

Add `--follow` to mirror the run's logs and metrics locally while it runs.

## Reference: how RLE invokes a Hosted Agent

The full contract is documented in `agent/main.py`'s module docstring. In
short, RLE calls the agent's Responses endpoint once per rollout,
synchronously:

```text
POST {project}/agents/{agentName}/endpoint/protocols/openai/responses?api-version=v1
x-client-rle-rollout-id: ...
x-client-rle-model-endpoint: https://.../rle/v1.0/capture-proxy/v1
x-client-rle-model-api-key: ...
x-client-rle-sandbox-tools-endpoint: https://.../rollouts/<rollout-id>/tools
x-client-rle-sandbox-tools-token: ...

{"agent_session_id": "<rollout id>",
 "input": [{"type": "message", "role": "user",
            "content": [{"type": "input_text", "text": "<agent_input>"}]}],
 "background": false, "stream": false, "store": false}
```

There is no acknowledgement, no poll and no withdraw: the response *is* the
result, and the final assistant message text is the rollout's `output_text`.
That is the whole difference from BYOH, which receives the same four context
values inside a request body it defines itself and is polled for the answer.

Two consequences are worth knowing before you change `agent/main.py`.

The context arrives per *request*, not per process. The container is warm long
before any rollout reaches it, and two concurrent rollouts carry different
values, so a model client built once at import would send both rollouts'
calls through whichever proxy happened to build it and corrupt both
trajectories.

What `main.py` does with that context is the other thing worth calling out:
instead of calling the model directly, it points `opencode` at the endpoint and
key from the headers and lets the agent loop make the calls, so the trajectory
RLE records is the agent's own.

`x-client-rle-sandbox-tools-endpoint` / `-token` arrive on the same request, but
those are not rollout parameters -- they are lifted out and turned into
environment variables on that rollout's `opencode` process, so the agent can
file a compliance disclosure back to `rle_deprecated/`. The token is presented as
`Authorization: Bearer`; RLE terminates that authentication at its own ingress
and replaces the header before forwarding, so the RLE container never sees it
and must not try to validate it.
