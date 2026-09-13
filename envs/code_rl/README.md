# `code_rl` OpenEnv environment

Self-hosted OpenEnv environment for
[`code_rl`](../../loom_cookbook/recipes/code_rl/README.md). See
[`../README.md`](../README.md) for the general contract; this file only
covers what's specific to code.

## What's here

| File | Purpose |
| --- | --- |
| `models.py` | `CodeAction` (`code_text: str`), `CodeObservation` (`messages`, `starter_code`, `problem_id`). |
| `server/code_rl_environment.py` | `CodeRLEnvironment`: `reset()` picks a DeepCoder-style competitive-programming problem, `step()` extracts a fenced ` ```python ``` ` block and grades it with `code_grading.check_correctness`, which executes the submission in-process. Single-turn: `step()` always returns `done=True`. |
| `server/app.py` | FastAPI app (`create_fastapi_app(...)` from `openenv.core.env_server.http_server`), served with `uvicorn`. One environment instance, built at startup and shared by both handlers. |
| `rle.toml` | Host-agnostic RLE identity and control-plane interface (`Gym` / `OpenEnv`). |
| `build_dataset.py` | Bakes `train.jsonl`/`validation.jsonl` at `docker build` time from the same DeepCoder-Preview loader `code_env.py` uses locally. |
| `Dockerfile` | Two-stage build: the `dataset` stage bakes the dataset, the runtime stage installs only what the server imports (`openenv`, `numpy`) and runs it. No `loom_cookbook` install. Self-contained: no sandbox service, no egress. |

## Executing submitted code

`step()` grades by calling `run_test` (from `grading/lcb_utils.py`) directly,
in the same process as the server. `run_test` executes the submission itself
via `exec()` and enforces its own per-test wall-clock timeout with
`signal.alarm`; there's no subprocess and no rlimits — a bad submission can
crash or wedge the server. That's an accepted tradeoff, not an oversight:
Foundry RLE gives every episode a fresh container, so losing one episode
never costs shared capacity. Because `run_test` needs `signal.alarm`, grading
must run on the server's main thread, blocking it for that request.

`check_correctness` builds on the same `lcb_utils.run_test` the local
(`use_rle=False`) recipe path calls, so both paths agree on what counts as a
passing solution — but this environment disables `run_test`'s
`reliability_guard()` step, since that guard permanently mutates process
state and is only safe in the throwaway subprocess it was designed for.

## Row schema (`train.jsonl`/`validation.jsonl`)

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
cd envs/code_rl
azd ai rle run
```

Builds this Dockerfile and opens a local playground at the printed URL.
Validate the environment contract right there:

```
reset {"seed": 0}
state
step {"code_text": "a, b = map(int, input().split())\nprint(a + b)"}
```

Needs Docker running and the `azd` RLE extension (see
[`../README.md`](../README.md)).

The checked-in manifest declares the initial `code_rl` release as version
`1.0.0`. Update its name when copying this source outside `azd ai rle init`,
and update its version before publishing a subsequent release.

## Train against it

The recipe only talks to a *published* environment — publish this one
to your Foundry project (see [`../README.md`](../README.md)) and name
it, so each rollout leases its own instance:

```bash
uv run python -m loom_cookbook.recipes.code_rl.train_azure \
    project_endpoint="https://<your-project>.services.ai.azure.com/api/projects/<name>" \
    model_name="Qwen/Qwen3-32B" tokenizer_name="Qwen/Qwen3-32B" \
    use_rle=true rle_env_name="code_rl" rle_max_active_instances=32
```
