# `code_rl` OpenEnv environment

Self-hosted, OpenEnv-compatible environment for
[`code_rl`](../../loom_cookbook/recipes/code_rl/README.md). It exposes the
standard OpenEnv `reset` and `step` lifecycle, so you can iterate locally with
your own OpenEnv-compatible agent before publishing an environment version.
See [`../../../README.md`](../../../README.md) for the general contract; this file
only covers what's specific to code.

## What's here

| File | Purpose |
| --- | --- |
| `models.py` | `CodeAction` (`code_text: str`), `CodeObservation` (`messages`, `starter_code`, `problem_id`). |
| `server/code_rl_environment.py` | `CodeRLEnvironment`: `reset()` picks a DeepCoder-style competitive-programming problem. `step()` supports `check_solution` tool calls, then grades a fenced ` ```python ``` ` final response with `code_grading.check_correctness`. |
| `server/app.py` | FastAPI app (`create_fastapi_app(...)` from `openenv.core.env_server.http_server`), served with `uvicorn`. One environment instance, built at startup and shared by both handlers. |
| `rle.toml` | Host-agnostic RLE identity and control-plane interface (`Gym` / `OpenEnv`). |
| `data/` | Checked-in, gzip-compressed snapshot with 900 training tasks and 100 validation tasks, plus provenance. Each task retains at most three complete test cases and 12 KiB of test input/output. |
| `training/` | Training-job input manifests (`{"seed": ..., "split": ...}` per row) for a training loop to pass as `reset()` arguments -- distinct from, and not baked into, the server's own `data/` snapshot. See `training/README.md`. |
| `Dockerfile` | Runtime-only image that copies the compact snapshot and installs only what the server imports (`openenv`, `numpy`). No Hugging Face data download, `loom_cookbook`, sandbox service, or runtime egress. |

## Executing submitted code

`step()` grades by calling `run_test` (from `grading/lcb_utils.py`) directly,
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

## Build and run locally

```bash
cd examples/gym/openenv/code_rl
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
the pancake-stack task at `problem_id` `"316"`. The value below is intentional,
not a placeholder. Including it also makes the command work with local runners
that send `reset` and `step` as separate HTTP requests.

````text
reset {"seed":0}

step {"problem_id":"316","code_text":"```python\nimport sys\n\nMOD = 1_000_000_007\nvalues = list(map(int, sys.stdin.read().split()))\nif values:\n    queries = values[1:1 + values[0]]\n    max_n = max(queries, default=0)\n\n    bell = [0] * (max_n + 1)\n    row = [1]\n    for n in range(1, max_n + 1):\n        next_row = [row[-1]]\n        for k in range(1, n + 1):\n            next_row.append((next_row[-1] + row[k - 1]) % MOD)\n        row = next_row\n        bell[n] = row[0]\n\n    sys.stdout.write(chr(10).join(str(bell[n]) if n > 0 else '0' for n in queries))\n```"}
````

The final response has `done: true`, `reward: 1.0`, and
`metadata.passed: true`.

Needs Docker running and the `azd` RLE extension (see
[`../../../README.md`](../../../README.md)).

The checked-in manifest declares the initial `code_rl` release as version
`1.0.0`. Update its name when copying this source outside `azd ai rle init`,
and update its version before publishing a subsequent release.

## Iterate with your agent, run, publish, and invoke

Your agent can use the same OpenEnv lifecycle as the local playground:
send `reset`, use the returned `messages`, `starter_code`, and `problem_id`
to construct a `CodeAction`, then send that action to `step`. `CodeAction`
accepts a fenced Python `code_text`, plus optional `tool_calls` and
`problem_id`.

1. **Iterate locally with your agent.** Edit the environment or your agent,
   then start the local runtime with `--watch`. Point your
   OpenEnv-compatible agent at the URL printed by the command. It rebuilds
   and restarts the container when the environment source changes; use the
   `rle>` shell or playground for manual smoke tests alongside your agent.
   Reconnect the agent and begin a new episode with `reset` after a restart.

   ```bash
   azd ai rle run --watch
   ```

2. **Publish an immutable version.** Set your Foundry project endpoint and
   Azure Container Registry endpoint, then publish the name and version from
   `rle.toml`. The command builds the image, pushes it to the registry, and
   registers the environment.

   ```bash
   export FOUNDRY_PROJECT_ENDPOINT="https://<account>.services.ai.azure.com/api/projects/<project>"
   export AZURE_CONTAINER_REGISTRY_ENDPOINT="<registry>.azurecr.io"
   azd ai rle publish
   ```

3. **Execute one rollout of the published environment.** With
   `FOUNDRY_PROJECT_ENDPOINT` still set, `invoke` reads `rle.name` and
   `rle.version` from `rle.toml`, provisions a real Loom training session and
   sampler checkpoint for `--model`, calls Execute Rollout with `--task`, and
   prints the resulting reward — no interactive shell, and no Loom session or
   checkpoint identifiers for you to manage.

   ```bash
   azd ai rle invoke --model Qwen/Qwen3-32B --task '{"split": "train"}'
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
        problem_id = observation["problem_id"]
        starter_code = observation.get("starter_code")
        trajectory = []

        while True:
            action = await policy.next_code_action(messages, starter_code)
            action["problem_id"] = problem_id
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

### Use the Loom adapter

The `loom_cookbook` recipes already implement the adapter between their model
renderer and this environment. From a checkout with its RLE extra and the
RLE-enabled SDK wheel installed, run:

```bash
pip install 'loom-cookbook[rle]'

uv run python -m loom_cookbook.recipes.code_rl.train_azure \
  project_endpoint="$FOUNDRY_PROJECT_ENDPOINT" \
  model_name="Qwen/Qwen3-32B" tokenizer_name="Qwen/Qwen3-32B" \
  use_rle=true rle_env_name="$RLE_ENV_NAME" \
  rle_env_version="$RLE_ENV_VERSION" rle_max_active_instances=32
```

The recipe leases published RLE instances for rollout execution while its
training session retains model state and performs optimization. Set
`rle_project_endpoint` as well when the environment was published to a
different Foundry project than the training session.
