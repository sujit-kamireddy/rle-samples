# `code_rl` OpenEnv environment

Self-hosted, OpenEnv-compatible environment for DeepCoder-style
competitive-programming problems, graded by running each submission's real
test cases. It exposes the standard OpenEnv `reset` and `step` lifecycle, so
you can iterate locally with your own OpenEnv-compatible agent before
publishing an environment version.
See [`../../../README.md`](../../../README.md) for the general contract; this file
only covers what's specific to code.

## What's here

| File | Purpose |
| --- | --- |
| `server/schema.py` | `CheckSolutionAction`/`SubmitAnswerAction` and the `CodeAction` union over them; `CodeObservation` (`messages`, `starter_code`, `problem_id`). |
| `server/code_rl_environment.py` | `CodeRLEnvironment`: `reset()` picks a DeepCoder-style competitive-programming problem. `step()` supports `check_solution` tool calls, then grades a fenced ` ```python ``` ` final response with `code_grading.check_correctness`. |
| `server/code_grading.py`, `server/lcb_utils.py` | Grading (`check_correctness`, `extract_code_from_model`): runs each submission's test cases in an isolated subprocess. |
| `server/dataset.py` | JSONL loading + `EpisodePicker` (seed -> row, via a fixed shuffled permutation). |
| `server/app.py` | FastAPI app (`create_fastapi_app(...)` from `openenv.core.env_server.http_server`), served with `uvicorn`. One environment instance, built at startup and shared by both handlers. |
| `rle.toml` | Host-agnostic RLE identity and control-plane interface (`Gym` / `OpenEnv`). |
| `env_data/` | Checked-in, gzip-compressed snapshot with 900 training tasks and 100 validation tasks, plus provenance. Each task retains at most three complete test cases and 12 KiB of test input/output. |
| `job_data/` | Training-job input manifests (`{"seed": ..., "split": ...}` per row) for a training loop to pass as `reset()` arguments -- distinct from, and not baked into, the server's own `env_data/` snapshot. See `job_data/README.md`. |
| `Dockerfile` | Runtime-only image that copies the compact snapshot and installs only what the server imports (`openenv`, `numpy`). No Hugging Face data download, sandbox service, or runtime egress. |

## How RLE drives this environment

RLE reads the action vocabulary from `CodeAction`'s JSON Schema at
`GET /schema` — there is no runtime discovery call and no tool spec to
maintain. The union member declaring `rle.toml`'s `model_response_field`
(`code_text`, on `SubmitAnswerAction`) receives the model's completion text;
every other member is offered to the model as a tool named after its
discriminator. `check_solution` is therefore a tool purely by virtue of being
a second union member.

```toml
[defaults.gym_openenv]
model_response_field = "code_text"
```

Each member's non-`const` fields become that tool's parameters, and its
docstring becomes the tool description — so the tool the model sees is
`check_solution(code: str)`, described as "Execute the proposed solution
against the task's test cases." The inherited OpenEnv `metadata` field and the
`type` discriminator are never shown to the model: RLE supplies the
discriminator itself when it builds the action.

RLE owns each tool call's `tool_call_id` and pairs the result with it, so
`CodeObservation.messages` only needs to carry the text of the result.

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

## Executing submitted code

`step()` grades by calling `run_test` (from `server/lcb_utils.py`) directly,
in the same process as the server. `run_test` executes the submission itself
via `exec()` and enforces its own per-test wall-clock timeout with
`signal.alarm`; there's no subprocess and no rlimits — a bad submission can
crash or wedge the server. That's an accepted tradeoff, not an oversight:
Foundry RLE leases each rollout an isolated managed instance, so a failure is
contained to that rollout rather than the trainer process. Because `run_test`
needs `signal.alarm`, grading must run on the server's main thread, blocking
it for that request.

`check_correctness` builds on the same `lcb_utils.run_test` the local
(`use_rle=False`) recipe path calls, so both paths agree on what counts as a
passing solution — but this environment disables `run_test`'s
`reliability_guard()` step, since that guard permanently mutates process
state and is only safe in the throwaway subprocess it was designed for.

## Row schema (`train.jsonl.gz`/`validation.jsonl.gz`)

```json
{
  "messages": [
    {"role": "system", "content": "You are an expert competitive programmer. ..."},
    {"role": "user", "content": "<problem statement>"}
  ],
  "tests": [{"input": "...", "output": "..."}, ...],
  "starter_code": "<optional starter code, or null>"
}
```

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

## Build and run locally

Complete [container setup](#container-and-registry-setup), then run from the
folder containing `rle.toml` (the folder created by `azd ai rle init`):

```bash
azd ai rle run
```

Builds this Dockerfile, starts an OpenEnv-compatible local runtime at the
printed URL, and opens a local playground.

> **Compact sample data:** The image includes a reproducible 1,000-task
> DeepCoder-Preview snapshot (900 training and 100 validation tasks), so the
> normal build does not download Hugging Face data. It intentionally retains
> at most three complete test cases and 12 KiB of test input/output per task.
> It is for trying the RLE experience, not benchmarking model quality.

### Copy-paste smoke test

After `azd ai rle run` opens the `rle>` shell, enter the following two
commands separately. Do not type the `rle>` prompt itself.

The checked-in compact snapshot makes `seed: 0` deterministic: it selects
the pancake-stack task at `problem_id` `"316"`. This environment is served
over `/ws` only (one environment instance for the life of the connection),
so `step()` always grades against the row the preceding `reset()` picked --
there is no `problem_id` field to echo back.

````text
reset {"seed":0}

step {"type":"submit_answer","code_text":"```python\nimport sys\n\nMOD = 1_000_000_007\nvalues = list(map(int, sys.stdin.read().split()))\nif values:\n    queries = values[1:1 + values[0]]\n    max_n = max(queries, default=0)\n\n    bell = [0] * (max_n + 1)\n    row = [1]\n    for n in range(1, max_n + 1):\n        next_row = [row[-1]]\n        for k in range(1, n + 1):\n            next_row.append((next_row[-1] + row[k - 1]) % MOD)\n        row = next_row\n        bell[n] = row[0]\n\n    sys.stdout.write(chr(10).join(str(bell[n]) if n > 0 else '0' for n in queries))\n```"}
````

The final response has `done: true`, `reward: 1.0`, and
`metadata.passed: true`.

Requires the selected container runtime to be running and a runner/shell that
keeps one `/ws` connection open across both commands -- a runner that sends
`reset` and `step` as separate stateless HTTP requests will not work, since
there is no `problem_id` for `step()` to fall back on.

The checked-in manifest declares the initial `code_rl` release as version
`1.0.0`. Update its name when copying this source outside `azd ai rle init`,
and update its version before publishing a subsequent release.

## Iterate with your agent, run, publish, and rollout

Your agent can use the same OpenEnv lifecycle as the local playground:
send `reset`, use the returned `messages` and `starter_code` to construct a
`CodeAction`, then send that action to `step`. `CodeAction` is a discriminated
union, so every action carries its `type`: `{"type": "check_solution", "code":
...}` to test a candidate, `{"type": "submit_answer", "code_text": ...}` to
submit a fenced Python solution and end the episode. Requires `/ws` (this
environment keeps no `problem_id` fallback for stateless HTTP transports).

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

4. **Start a training job.** `azd ai rle init` downloads `training.jsonl`
    into the initialized RLE folder's `job_data` subfolder. Run the training
    command from that subfolder so the relative training-file path resolves:

    ```powershell
    cd .\job_data
    azd ai rle train --rle-name code_rl --rle-version 1.0.0 --model qwen3-32b-1 --suffix rle-cli-smoke-20260918 --training-file .\training.jsonl
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
export RLE_ENV_NAME="code_rl"
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

### Collect a code rollout with the SDK

Create one `OpenEnvClient` for the training run, then lease one instance for
each concurrent rollout. `policy.next_code_action()` below is the integration
point for your renderer and current model weights: it returns a dictionary
with `code_text` and, when the policy calls `check_solution`, optional
`tool_calls`. Its terminal response must contain a fenced Python solution.

```python
import asyncio
import os

from azure.ai.projects.aio import AIProjectClient
from azure.identity.aio import DefaultAzureCredential


async def collect_rollout(openenv_client, policy, seed):
    async with openenv_client.get_instance() as instance:
        reset = await instance.reset(seed=seed, split="train")
        observation = reset.observation
        messages = list(observation["messages"])
        starter_code = observation.get("starter_code")
        trajectory = []

        while True:
            action = await policy.next_code_action(messages, starter_code)
            result = await instance.step(action)
            trajectory.append(
                {
                    "messages": list(messages),
                    "action": action,
                    "reward": result.reward,
                    "done": result.done,
                    "metadata": result.metadata,
                }
            )
            if result.done:
                return trajectory, result.reward, result.metadata

            # A non-terminal step contains new check_solution tool messages.
            messages.extend(result.observation["messages"])


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

Feed each returned `(trajectory, reward, metadata)` into the existing trainer's
advantage and loss calculation. Set `max_active_instances` to the number of
rollouts you want in flight and within project quota; extra rollout requests
wait for an instance. Use `split="validation"` with a distinct seed range for
held-out evaluation. Closing the client context releases the run's instance
group and its leases.
