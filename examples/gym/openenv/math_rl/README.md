# `math_rl` OpenEnv environment

Self-hosted, OpenEnv-compatible environment for one Hendrycks MATH problem
per episode, graded by comparing the submitted `\boxed{...}` answer against
the reference answer. It exposes the standard OpenEnv `reset` and `step`
lifecycle, so you can iterate locally with your own OpenEnv-compatible agent
before publishing an environment version. See
[`../../../README.md`](../../../README.md) for the general contract; this
file only covers what's specific to math.

## What's here

| File | Purpose |
| --- | --- |
| `server/schema.py` | `MathAction` (`answer_text: str`), `MathObservation` (`messages`, `problem_id`). |
| `server/math_rl_environment.py` | `MathRLEnvironment`: `reset()` picks a Hendrycks MATH problem. `step()` grades one `\boxed{...}` answer with `grading.safe_grade` and ends the episode. |
| `server/grading.py` | `safe_grade` (sympy by default) and `extract_boxed`. |
| `server/dataset.py` | JSONL loading + `EpisodePicker` (seed -> row, via a fixed shuffled permutation). |
| `server/app.py` | FastAPI app (`create_fastapi_app(...)` from `openenv.core.env_server.http_server`), served with `uvicorn`. One environment instance, built at startup and shared by both handlers. |
| `rle.toml` | Host-agnostic RLE identity and control-plane interface (`Gym` / `OpenEnv`). |
| `env_data/` | Checked-in, gzip-compressed Hendrycks MATH snapshot (`train.jsonl.gz`/`validation.jsonl.gz`) the server reads at rollout time, plus provenance (`source.json`, `NOTICE.md`). |
| `job_data/` | Training-job input manifests (`{"seed": ..., "split": ...}` per row) for a training loop to pass as `reset()` arguments -- distinct from, and not baked into, the server's own `env_data/` snapshot. See `job_data/README.md`. |
| `scripts/dataset_source.py` | Hendrycks MATH loader (`HuggingFaceH4/MATH-500` for validation and the filtered Hendrycks MATH corpus for train). Maintainer-only, not part of the Docker build. |
| `scripts/build_dataset.py` | Maintainer-only CLI that downloads and bakes `env_data/train.jsonl.gz`/`validation.jsonl.gz`. Also (re)generates `job_data/train.jsonl`/`job_data/validation.jsonl` so the two stay in lockstep. |
| `Dockerfile` | Two-stage build: the dataset stage downloads and bakes the data; the runtime stage installs only `openenv`, `sympy`, and `pylatexenc`. No runtime dataset download. |

## How RLE drives this environment

RLE reads the action vocabulary from `MathAction`'s JSON Schema at
`GET /schema`. `MathAction` declares one action, so the model is offered no
tools at all: every completion becomes a single graded step. `rle.toml` names
the field that receives the completion text verbatim:

```toml
[defaults.gym_openenv]
model_response_field = "answer_text"
```

This is the sample to copy when an environment needs no tools. Adding one
means adding a second action variant — see `code_rl`.

RLE sets no `max_tokens` of its own on this path — it is the harness here — so
`rle.toml` also states the per-turn output budget. Left unset the request
inherits the sampler's default (1024 tokens today), which stops a reasoning
model mid-thought and submits a truncated answer that still grades:

```toml
[defaults.reinforcement]
max_completion_tokens = 8192
```

RLE sends the smaller of that and its own ceiling, and the capture proxy applies
its own cap on top — 8192 today, which is why nothing here asks for more.

## Grading answers

`step()` extracts `\boxed{...}` from the submitted answer and the reference,
then grades the values with the `safe_grade` implementation (`server/grading.py`) using
SymPy by default. An answer without `\boxed{...}` receives the formatting
penalty and is not graded as a raw answer.

## Row schema (`train.jsonl`/`validation.jsonl`)

```json
{
  "messages": [
    {"role": "user", "content": "<few-shot question>"},
    {"role": "assistant", "content": "<few-shot answer>"},
    {"role": "user", "content": "<problem> Write your answer in \\boxed{} format."}
  ],
  "problem": "<problem text>",
  "solution": "<reference solution containing \\boxed{...}>"
}
```

## Container and registry setup

Choose a container runtime to build, run, and publish this environment:

- **Docker Desktop:** start it in Linux-container mode and check `docker info`.
- **Podman:** install Podman (Podman Desktop is optional) and check `podman info`.
  On Windows/macOS, first run `podman machine list`; run `podman machine init`
  only if no machine exists, then `podman machine start` if it is stopped.
  Native Linux does not need a Podman machine.

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

## Build and run locally

Complete [container setup](#container-and-registry-setup), then run from the
folder containing `rle.toml` (the folder created by `azd ai rle init`):

```bash
azd ai rle run
```

Builds this Dockerfile, starts an OpenEnv-compatible local runtime at the
printed URL, and opens a local playground.

> **Build-time data:** The dataset stage downloads the selected Hendrycks MATH
> data and bakes 4,000 training tasks and 500 validation tasks into the image.
> The download occurs during the build only; runtime instances read the baked
> JSONL files from disk without dataset egress.

### Copy-paste smoke test

After `azd ai rle run` opens the `rle>` shell, enter the following two
commands separately. Do not type the `rle>` prompt itself.

With the default build data, `seed: 0` selects the modular-arithmetic problem
at `problem_id` `"2314"`. This environment is served over `/ws` only (one
environment instance for the life of the connection), so `step()` always
grades against the row the preceding `reset()` picked -- there is no
`problem_id` field to echo back.

```text
reset {"seed":0}

step {"answer_text":"\\boxed{30}"}
```

The final response has `done: true`, `reward: 1.0`, and
`metadata.correct: true`.

Requires the selected container runtime to be running and a runner/shell that
keeps one `/ws` connection open across both commands -- a runner that sends
`reset` and `step` as separate stateless HTTP requests will not work, since
there is no `problem_id` for `step()` to fall back on.

The checked-in manifest declares the initial `math_rl` release as version
`1.0.0`. Update its name when copying this source outside `azd ai rle init`,
and update its version before publishing a subsequent release.

## Iterate with your agent, run, publish, and rollout

Your agent can use the same OpenEnv lifecycle as the local playground:
send `reset`, use the returned `messages` to construct a `MathAction`, then
send that action to `step`. `MathAction` accepts a `\boxed{...}` `answer_text`.
Requires `/ws` (this environment keeps no `problem_id` fallback for stateless
HTTP transports).

1. **Iterate locally with your agent.** Edit the environment or your agent,
   then start the local runtime with `--watch`. Point your
   OpenEnv-compatible agent at the URL printed by the command. It rebuilds
   and restarts the container when the environment source changes; use the
   `rle>` shell or playground for manual smoke tests alongside your agent.
   Reconnect the agent and begin a new episode with `reset` after a restart.

   ```bash
   azd ai rle run --watch
   ```

2. **Publish an immutable version.** Complete [registry setup](#before-the-first-publish).
   Set your Foundry project endpoint and
   Azure Container Registry endpoint, then publish the name and version from
   `rle.toml`. The command builds the image, pushes it to the registry, and
   registers the environment. The snippet below uses Bash; in PowerShell,
   replace `export NAME="value"` with `$env:NAME = "value"`.

   ```bash
   export FOUNDRY_PROJECT_ENDPOINT="https://<account>.services.ai.azure.com/api/projects/<project>"
   export AZURE_CONTAINER_REGISTRY_ENDPOINT="<registry>.azurecr.io"
   azd ai rle publish
   ```

3. **Execute one rollout of the published environment.** With
    `FOUNDRY_PROJECT_ENDPOINT` still set, `rollout` reads `rle.name` and
   `rle.version` from `rle.toml`, provisions a model session and sampler
   checkpoint for `--model`, calls Execute Rollout with `--task`, and prints
   the resulting reward — no interactive shell, and no session or
   checkpoint identifiers for you to manage.

   ```bash
    azd ai rle rollout --model Qwen/Qwen3-32B --task '{"split": "train"}'
   ```

4. **Start a training job.** Run this from the environment folder: `train`
    reads `rle.name` and `rle.version` from `rle.toml`, and `job_data` holds
    the task rows the job replays. Add
    `--validation-file .\job_data\validation.jsonl` to score a validation
    split as well.

    ```powershell
    azd ai rle train --model qwen3-32b-1 --suffix rle-cli-smoke-20260918 --training-file .\job_data\train.jsonl
    ```

## Use the published environment in a trainer loop

`azd ai rle run` is for iterating on the environment. A trainer connects only
to a published RLE version: it owns policy sampling, trajectories, advantages,
and optimization; RLE owns task selection, environment execution, grading, and
the managed instance pool. See the [RLE OpenEnv/Gym guide](https://aka.ms/rle)
for the complete SDK contract.

Pin the environment that the run uses. The name and version must match the
published `rle.toml`, not necessarily the defaults below if you renamed the
sample while initializing it.

```bash
export FOUNDRY_PROJECT_ENDPOINT="https://<account>.services.ai.azure.com/api/projects/<project>"
export RLE_ENV_NAME="math_rl"
export RLE_ENV_VERSION="1.0.0"
```

Install the RLE-enabled `azure-ai-projects` wheel shown in the guide. The
public PyPI build with the same package version does not expose the `rle`
operation group.

```bash
pip install --force-reinstall \
  "https://rle-onboarding-docs.orangeground-ba9696de.eastus2.azurecontainerapps.io/downloads/azure_ai_projects-2.6.0-py3-none-any.whl" \
  azure-identity aiohttp
```

### Collect a math rollout with the SDK

Create one `OpenEnvClient` for the training run, then lease one instance for
each concurrent rollout. `policy.answer()` below is the integration point for
your renderer and current model weights; it must return an answer containing
`\boxed{...}`.

```python
import asyncio
import os

from azure.ai.projects.aio import AIProjectClient
from azure.identity.aio import DefaultAzureCredential


async def collect_rollout(openenv_client, policy, seed):
    async with openenv_client.get_instance() as instance:
        reset = await instance.reset(seed=seed, split="train")
        observation = reset.observation
        answer_text = await policy.answer(observation["messages"])
        action = {"answer_text": answer_text}
        result = await instance.step(action)
        return {
            "messages": observation["messages"],
            "action": action,
            "reward": result.reward,
            "done": result.done,
            "metadata": result.metadata,
        }


async def collect_batch(policy):
    concurrency = 32
    async with DefaultAzureCredential() as credential:
        async with AIProjectClient(
            endpoint=os.environ["FOUNDRY_PROJECT_ENDPOINT"],
            credential=credential,
            allow_preview=True,
        ) as project_client:
            async with project_client.rle.get_openenv_client(
                name=os.environ["RLE_ENV_NAME"],
                version=os.environ["RLE_ENV_VERSION"],
                max_active_instances=concurrency,
                instance_acquire_timeout=900,
            ) as openenv_client:
                return await asyncio.gather(
                    *(
                        collect_rollout(openenv_client, policy, seed)
                        for seed in range(concurrency)
                    )
                )
```

Feed each returned rollout into the existing trainer's advantage and loss
calculation. Set `max_active_instances` to the number of rollouts you want in
flight and within project quota; extra rollout requests wait for an instance.
Use `split="validation"` with a distinct seed range for held-out evaluation.
Closing the client context releases the run's instance group and its leases.
