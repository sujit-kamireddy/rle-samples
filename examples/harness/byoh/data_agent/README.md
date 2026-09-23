# BYOH (Bring Your Own Harness) example: FineEnvs data-agent

Wraps Hugging Face's [`FineEnvs/data-agent`](https://huggingface.co/collections/FineEnvs/data-agent)
collection -- 5,000 deterministic data-analysis tasks built from
`jupyter-agent`. Each task hands an agent a question about a real dataset and a
directory of CSVs to answer it from, and scores the answer with a deterministic
grader.

This sample adds a second, orthogonal axis on top: **did the agent correctly
decide whether the data it just analysed was sensitive, and report it.** See
"Compliance disclosure" below.

Three containers:

- [`harness/`](./harness) -- runs the agent. `POST /correlated-rollouts/{rollout_id}`
  fetches the task's input files, runs [`opencode`](https://opencode.ai) against
  the task instruction, and returns the answer text the agent wrote. Deploy this
  once and reference it by URL.
- [`agent/`](./agent) -- the harness RLE actually invokes (`Harness`/`BYOH`'s
  `agent/`). It implements RLE's `/invoke`/poll/withdraw contract and does no
  agent work of its own: it POSTs to `harness/`, passing RLE's capture proxy as
  the model endpoint so every model call lands in the rollout graph, and relays
  the answer text back.
- [`rle/`](./rle) -- the RLE container. `/grade` scores that answer text with its
  own vendored copy of the grader. It also serves
  `/tools/report_sensitive_data_access`, the compliance-disclosure endpoint
  described below, and blends that decision into the reward.

**Why doesn't `agent/` just report the reward?** `agent/` is untrusted,
customer-hosted code, and the grader is a public, deterministic function of
`(gold, candidate)` -- nothing stops a misbehaving harness from computing the
winning `candidate` and reporting whatever `reward` it likes. So `rle/`'s
`/grade` computes `reward` itself rather than reading it out of anything
`agent/` sends.

What it does trust is the raw answer text the agent wrote. Each rollout runs in
its own directory inside `harness/`; the task instruction tells the agent to
write its answer to a file there, and `harness/` reads that file off local disk
once the agent exits and returns it as `answer_text`. `agent/` relays it
verbatim to RLE as `output_text`; RLE forwards it to `/grade` as
`agent_response`, and `/grade` grades it against a vendored per-task answer key
(`rle/server/vendor/task-meta/`) using the dataset's own deterministic grader
(`rle/server/vendor/grader.py`, byte-identical across every task in this
dataset). `agent/` still cannot fabricate a better score for itself: it can only
relay -- or fail to relay -- whatever the agent actually wrote.

Running `azd ai rle init --type Harness --subtype BYOH --sample data_agent`
copies this exact `agent/` + `rle/` pair as your starting point; `harness/` is
deployed separately, once, and referenced by URL.

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
blanket disclosure. See [`rle/server/compliance.py`](./rle/server/compliance.py).

Ground truth comes from a per-task label baked into the vendored answer key by
[`tools/build_task_meta.py`](./tools/build_task_meta.py) using the keyword
taxonomy in [`tools/pii_taxonomy.py`](./tools/pii_taxonomy.py). Labels carry a
`pii_label_source` so a heuristic label is never mistaken for a reviewed one;
hand-verified corrections go in `tools/pii_overrides.json` and flip that field
to `verified`. A heuristic is adequate for shaping a training reward and is
*not* adequate for a benchmark number -- report the eval slice only from
verified labels.

Getting the per-rollout credentials to the agent is worth understanding before
you change it. The disclosure endpoint and token arrive in the rollout request
body and are set as environment variables on that rollout's `opencode` process
-- per process, never process-global, because concurrent rollouts would
otherwise overwrite each other's and misattribute a disclosure with no error
anywhere. See
[`harness/server/opencode_direct.py`](./harness/server/opencode_direct.py)
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

## 1. Deploy `harness`

Build and deploy the harness container first; it has no RLE
dependency and is easiest to prove working on its own. See
[`harness/README.md`](./harness/README.md) for local run, build,
and deploy instructions. Confirm `/health` on the deployed URL
before moving on.

## 2. Deploy the agent (harness)

`agent/` is a plain FastAPI service with no Foundry or Azure dependency other
than reaching `harness`. Run it wherever you already run the rest of
your stack:

```bash
cd agent
docker build -t data-agent-byoh:latest .
docker run --rm -p 8080:8080 \
  -e HARNESS_SERVER_URL=https://<your-deployed-harness> \
  data-agent-byoh:latest
```

For Podman, replace `docker` with `podman` in both commands.

Put it behind HTTPS and protect it yourself -- RLE never attaches caller,
workspace, or identity headers to its invocation requests, so require a
shared secret header and validate it in `agent/app.py` before dispatching to
`harness/`.

## 3. Author and iterate the RLE side

`rle/server/env.py`'s `/reset` is close to a no-op -- the task selector
(`task_index`, `split`) travels in `--agent-input`, not
in whatever `/reset` returns. `/grade` doesn't call out anywhere: it parses
`answer_text`, `task_index`, and `split` out of `agent_response` (the same
JSON string `agent/`'s `/invoke` returned as `output_text`, which RLE
forwards to `/grade` verbatim), looks up that task's answer key in the
vendored `rle/server/vendor/task-meta/` metadata, and grades it with the
vendored copy of the dataset's own deterministic grader
(`rle/server/vendor/grader.py`). `agent/` never computes or even sees a
`reward` -- it only relays what the agent wrote (see `agent/app.py`'s
`run_harness_rollout`). Adapt `rle/server/env.py`'s `grade_rollout` if you
want reward shaping other than the grader's raw score.

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
  --agent-input '{"task_index": 0, "split": "FineEnvs/data-agent-harbor-train"}'
```

This sample's `/reset` ignores `--task`, so `{}` is enough; task
selection lives entirely in `--agent-input`.

## Reference: how RLE invokes your harness

The full `/invoke`/poll/withdraw contract is documented in `agent/app.py`'s
module docstring. In short: RLE invokes a harness asynchronously -- the start
request only starts work, and the answer is collected from a per-invocation
resource RLE derives from the registered URL. What `agent/app.py` does with
`rollout_context` once it has it is the one thing worth calling out here:
instead of calling the model directly, it hands
`model_endpoint`/`model_api_key`
to `harness`'s `/correlated-rollouts/{rollout_id}`, which points `opencode` at
that endpoint and lets the agent loop make the calls.

It forwards `sandbox_tools_endpoint`/`sandbox_tools_token` on the same
request, but those are not rollout parameters -- `harness` lifts them
back out of the payload and turns them into environment variables on the
`opencode` process, so the agent can file a compliance disclosure back to
`rle/`. The
token is presented as `Authorization: Bearer`; RLE terminates that
authentication at its own ingress and replaces the header before forwarding,
so the RLE container never sees it and must not try to validate it.
