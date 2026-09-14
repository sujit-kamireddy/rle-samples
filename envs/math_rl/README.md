# `math_rl` OpenEnv environment

Self-hosted, OpenEnv-compatible environment for
[`math_rl`](../../loom_cookbook/recipes/math_rl/README.md). It exposes the
standard OpenEnv `reset` and `step` lifecycle, so you can iterate locally with
your own OpenEnv-compatible agent before publishing an environment version.
See [`../../README.md`](../../README.md) for the general contract; this file
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
cd envs/math_rl
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
[`../../README.md`](../../README.md)).

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
