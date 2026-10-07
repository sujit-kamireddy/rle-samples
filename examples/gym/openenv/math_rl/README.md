# `math_rl`: SDK MCP Gym environment

`math_rl` serves one deterministic-by-seed Hendrycks MATH problem per episode.
It subclasses the SDK `RLEnvironment`, exposes an optional typed MCP
equivalence checker, and grades the terminal `GradeAction.response` against the
boxed reference answer with the existing safe grader.

## Contract

- `reset(seed, split)` selects from the baked train or validation snapshot.
  `EpisodePicker` retains its fixed shuffled permutation, so the same seed and
  split select the same row.
- `check_equivalence(candidate_expression: str, comparison_expression: str)`
  safely checks two **model-supplied** expressions with `safe_grade`. It never
  receives or reveals the dataset reference answer.
- Final submission is `GradeAction(response="<reasoning and \\boxed{...}>")`.
- A correct boxed answer earns `1.0`, a formatted wrong answer earns `0.0`,
  and an unboxed answer earns `-0.1` (`FORMAT_COEF`).
- The helper is optional. Its use is recorded but is not a reward condition.

`MathState` explicitly records the environment `instance_id`, selected split
and row, helper count and last helper inputs/result, and terminal status.
Because MCP calls and grading share one persistent OpenEnv WebSocket, this
state proves that tool activity belongs to the same episode instance.

The manifest selects this mode explicitly:

```toml
environment_protocol = "mcp_environment"

[defaults.reinforcement]
max_episode_steps = 4
```

The four-step budget allows up to three helper calls plus terminal grading.
There is no `model_response_field`; that setting belongs to schema-driven Gym.

## Layout

| Path | Purpose |
| --- | --- |
| `server/math_rl_environment.py` | SDK environment, MCP tool, reset, and grade |
| `server/models.py` | observation and explicit per-episode state |
| `server/grading.py` | boxed extraction, normalization, and safe grading |
| `server/dataset.py` | local JSONL loading and deterministic seed picker |
| `server/app.py` | single-session SDK/OpenEnv ASGI app |
| `env_data/` | baked train and validation datasets plus provenance |
| `job_data/` | deterministic reset selectors used by training/evaluation |
| `scripts/` | maintainer-only dataset refresh utilities |
| `tests/` | MCP contract and runtime behavior tests |

## Data and evaluation

The image performs no runtime dataset download. The checked-in
`env_data/train.jsonl.gz` and `env_data/validation.jsonl.gz` are loaded locally.
Training rows pass `{"seed": N, "split": "train"}`; validation rows pass the
same selector with `"validation"`. Validation is lazily loaded and cached, so
held-out evaluation remains separate from training.

Each dataset row contains chat `messages`, the original `problem`, and a
reference `solution` containing `\boxed{...}`. Only the boxed content is sent
to `safe_grade`; raw unboxed final text is never accepted as a lucky match.

## Build and run

Choose Docker or Podman and run from this directory:

```bash
docker build -t math-rl .
docker run --rm -p 8000:8000 math-rl
```

The Dockerfile pins `openenv==0.6.0` and immutable SDK source revision
`2520ed7b6606f0e6eaff24affa49e888dbb7732e`. That revision contains
`azure.ai.projects.rle.environments.RLEnvironment`; the released public 2.7.0
wheel does not, so it is not currently an equivalent dependency.

The server must use one worker. RLE owns the model loop and sends MCP
`tools/list`, optional `tools/call`, and terminal `GradeAction` exchanges over
the same WebSocket-backed environment session.

With the RLE CLI configured, local iteration remains:

```bash
azd ai rle run
```

Publish and roll out using the name/version in `rle.toml`:

```bash
azd ai rle publish
azd ai rle rollout --model Qwen/Qwen3-32B --task '{"seed":0,"split":"train"}'
```

Start training with the checked-in selectors:

```powershell
azd ai rle train --model qwen3-32b-1 --suffix math-rl `
  --training-file .\job_data\train.jsonl `
  --validation-file .\job_data\validation.jsonl
```

## Test

Install the same dependencies as the Dockerfile, then run:

```bash
python -m pytest -q tests
python -m compileall -q server tests
```

Contract tests remain dependency-light. Runtime tests skip when the private
RLE SDK is unavailable rather than silently testing a public wheel without
`RLEnvironment`.
