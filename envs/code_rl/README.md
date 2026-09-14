# `code_rl` OpenEnv environment

Self-hosted, OpenEnv-compatible environment for
[`code_rl`](../../loom_cookbook/recipes/code_rl/README.md). It exposes the
standard OpenEnv `reset` and `step` lifecycle, so you can iterate locally with
your own OpenEnv-compatible agent before publishing an environment version.
See [`../README.md`](../README.md) for the general contract; this file only
covers what's specific to code.

## What's here

| File | Purpose |
| --- | --- |
| `models.py` | `CodeAction` (`code_text: str`), `CodeObservation` (`messages`, `starter_code`, `problem_id`). |
| `server/code_rl_environment.py` | `CodeRLEnvironment`: `reset()` picks a DeepCoder-style competitive-programming problem. `step()` supports `check_solution` tool calls, then grades a fenced ` ```python ``` ` final response with `code_grading.check_correctness`. |
| `server/app.py` | FastAPI app (`create_fastapi_app(...)` from `openenv.core.env_server.http_server`), served with `uvicorn`. One environment instance, built at startup and shared by both handlers. |
| `rle.toml` | Host-agnostic RLE identity and control-plane interface (`Gym` / `OpenEnv`). |
| `data/` | Checked-in, gzip-compressed snapshot with 900 training tasks and 100 validation tasks, plus provenance. Each task retains at most three complete test cases and 12 KiB of test input/output. |
| `Dockerfile` | Runtime-only image that copies the compact snapshot and installs only what the server imports (`openenv`, `numpy`). No Hugging Face data download, `loom_cookbook`, sandbox service, or runtime egress. |

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
cd envs/code_rl
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
[`../README.md`](../README.md)).

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

3. **Invoke the published environment.** With
   `FOUNDRY_PROJECT_ENDPOINT` still set, `invoke` reads `rle.name` and
   `rle.version` from `rle.toml`, starts a remote runtime, and opens the
   interactive `rle>` shell. Use the same `reset` and `step` commands as the
   local shell.

   ```bash
   azd ai rle invoke
   ```
