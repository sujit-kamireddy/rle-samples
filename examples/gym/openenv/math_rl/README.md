# `math_rl` OpenEnv environment

Self-hosted, OpenEnv-compatible environment for
[`math_rl`](../../loom_cookbook/recipes/math_rl/README.md). It exposes the
standard OpenEnv `reset` and `step` lifecycle, so you can iterate locally with
your own OpenEnv-compatible agent before publishing an environment version.
See [`../../../README.md`](../../../README.md) for the general contract; this file
only covers what's specific to math.

## What's here

| File | Purpose |
| --- | --- |
| `models.py` | `MathAction` (`answer_text: str`), `MathObservation` (`messages`, `problem_id`). |
| `grading/` | `safe_grade` (sympy by default) and `extract_boxed`/`math_grading.py`, vendored from `loom_cookbook/recipes/math_rl/`. `tests/test_env_grading_parity.py` pins these copies against the recipe originals. |
| `server/math_rl_environment.py` | `MathRLEnvironment`: `reset()` picks a Hendrycks MATH problem. `step()` grades one `\boxed{...}` answer with the vendored `grading.safe_grade` and ends the episode. |
| `server/app.py` | FastAPI app (`create_fastapi_app(...)` from `openenv.core.env_server.http_server`), served with `uvicorn`. One environment instance, built at startup and shared by both handlers. |
| `rle.toml` | Host-agnostic RLE identity and control-plane interface (`Gym` / `OpenEnv`). |
| `dataset_source.py` | Vendored Hendrycks MATH loader (`HuggingFaceH4/MATH-500` for validation and the filtered Hendrycks MATH corpus for train). |
| `build_dataset.py` | Downloads and bakes `train.jsonl`/`validation.jsonl` during `docker build`. |
| `Dockerfile` | Two-stage build: the dataset stage downloads and bakes the data; the runtime stage installs only `openenv`, `sympy`, and `pylatexenc`. No `loom_cookbook` install or runtime dataset download. |

## Grading answers

`step()` extracts `\boxed{...}` from the submitted answer and the reference,
then grades the values with the vendored `safe_grade` implementation using
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

## Build and run locally

```bash
cd examples/gym/openenv/math_rl
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
at `problem_id` `"2314"`. The value below is intentional, not a placeholder.
Including it also makes the command work with local runners that send `reset`
and `step` as separate HTTP requests.

```text
reset {"seed":0}

step {"problem_id":"2314","answer_text":"\\boxed{30}"}
```

The final response has `done: true`, `reward: 1.0`, and
`metadata.correct: true`.

Needs Docker running and the `azd` RLE extension (see
[`../../../README.md`](../../../README.md)).

The checked-in manifest declares the initial `math_rl` release as version
`1.0.0`. Update its name when copying this source outside `azd ai rle init`,
and update its version before publishing a subsequent release.

## Iterate with your agent, run, publish, and invoke

Your agent can use the same OpenEnv lifecycle as the local playground:
send `reset`, use the returned `messages` and `problem_id` to construct a
`MathAction`, then send that action to `step`. `MathAction` accepts a
`\boxed{...}` `answer_text` and an optional `problem_id`.

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

3. **Invoke the published environment.** With
   `FOUNDRY_PROJECT_ENDPOINT` still set, `invoke` reads `rle.name` and
   `rle.version` from `rle.toml`, starts a remote runtime, and opens the
   interactive `rle>` shell. Use the same `reset` and `step` commands as the
   local shell.

   ```bash
   azd ai rle invoke
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
        action = {
            "problem_id": observation["problem_id"],
            "answer_text": answer_text,
        }
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

### Use the Loom adapter

The `loom_cookbook` recipes already implement the adapter between their model
renderer and this environment. From a checkout with its RLE extra and the
RLE-enabled SDK wheel installed, run:

```bash
pip install 'loom-cookbook[rle]'

uv run python -m loom_cookbook.recipes.math_rl.train_azure \
  project_endpoint="$FOUNDRY_PROJECT_ENDPOINT" \
  model_name="Qwen/Qwen3-32B" tokenizer_name="Qwen/Qwen3-32B" env=math \
  use_rle=true rle_env_name="$RLE_ENV_NAME" \
  rle_env_version="$RLE_ENV_VERSION" rle_max_active_instances=32
```

The recipe leases published RLE instances for rollout execution while its
training session retains model state and performs optimization. Set
`rle_project_endpoint` as well when the environment was published to a
different Foundry project than the training session.
