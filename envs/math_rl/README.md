# `math_rl` OpenEnv environment

Self-hosted OpenEnv environment for
[`math_rl`](../../loom_cookbook/recipes/math_rl/README.md). See
[`../README.md`](../README.md) for the general contract; this file only
covers what's specific to math.

## What's here

| File | Purpose |
| --- | --- |
| `models.py` | `MathAction` (`answer_text: str`), `MathObservation` (`messages`, `problem_id`). |
| `grading/` | `safe_grade` (sympy by default) and `extract_boxed`/`math_grading.py`, vendored from `loom_cookbook/recipes/math_rl/`. `tests/test_env_grading_parity.py` pins these copies against the recipe originals. |
| `server/math_rl_environment.py` | `MathRLEnvironment`: `reset()` picks a Hendrycks MATH problem, `step()` extracts `\boxed{...}` from the submission and reference, grades with the vendored `grading.safe_grade`. Single-turn: `step()` always returns `done=True`. |
| `server/app.py` | FastAPI app (`create_fastapi_app(...)` from `openenv.core.env_server.http_server`), served with `uvicorn`. One environment instance, built at startup and shared by both handlers. |
| `dataset_source.py` | Vendored Hendrycks MATH loader (`HuggingFaceH4/MATH-500` for validation, the full MATH train split for train), extracted from `math_env.py`. |
| `build_dataset.py` | Bakes `train.jsonl`/`validation.jsonl` at `docker build` time from `dataset_source.py`. |
| `Dockerfile` | Two-stage build: the `dataset` stage installs only `datasets` to bake the dataset; the runtime stage installs only what the server imports (`openenv`, `sympy`, `pylatexenc`) and runs it. No `loom_cookbook` install. |

## Row schema (`train.jsonl`/`validation.jsonl`)

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful math assistant. ..."},
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

Builds this Dockerfile and opens a local playground at the printed URL.
Validate the environment contract right there:

```
reset {"seed": 0}
state
step {"answer_text": "The answer is \\boxed{5}"}
```

Needs Docker running and the `azd` RLE extension (see
[`../README.md`](../README.md)).

## Train against it

The recipe only talks to a *published* environment — publish this one
to your Foundry project (see [`../README.md`](../README.md)) and name
it, so each rollout leases its own instance:

```bash
uv run python -m loom_cookbook.recipes.math_rl.train_azure \
    project_endpoint="https://<your-project>.services.ai.azure.com/api/projects/<name>" \
    model_name="Qwen/Qwen3-32B" tokenizer_name="Qwen/Qwen3-32B" env=math \
    use_rle=true rle_env_name="math-rl" rle_max_active_instances=32
```
