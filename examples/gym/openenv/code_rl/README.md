# `code_rl`: SDK MCP code-grading environment

This Gym/OpenEnv sample presents DeepCoder-style competitive-programming
problems and grades Python submissions against hidden tests. It uses the SDK
`RLEnvironment` MCP contract rather than an action-schema union.

## Contract

- `CodeRLEnvironment` subclasses
  `azure.ai.projects.rle.environments.RLEnvironment`.
- `check_solution(code: str)` is a typed MCP tool registered with
  `self.tool()`. It runs the same hidden-test grader as final submission.
- The model's final response arrives as `GradeAction.response`.
- `[rle] environmentProtocol = "mcp_environment"` selects this protocol;
  there is no `model_response_field`.
- There is no `max_episode_steps` any more; RLE caps every Gym episode at a
  fixed 32-step server ceiling, which comfortably covers a few tool calls
  plus one final grade.

Tool use is optional, matching the prior sample contract: a correct fenced
final answer earns full correctness reward even when `check_solution` was not
called. `CodeState` nevertheless records each tool call, its result, the
selected row/split, and terminal grading state. Because OpenEnv creates one
environment per WebSocket session, reset, MCP calls, and grade all observe the
same instance state.

## Reward and grading

The final answer must contain a fenced code block; the last fenced block is
graded. Correct fenced code scores `1.0`. Incorrect fenced code scores `0.0`.
An unfenced answer is not executed and scores `-0.1` (`FORMAT_COEF`).

Every grading call runs in a fresh disposable child process. The child applies
the LiveCodeBench reliability guard and per-test timeout; the parent enforces
an overall hard timeout and terminates an unresponsive child. A
`check_solution` call returns `passed`, details, and the per-episode call count.
Calls beyond the two-call tool budget return a recoverable error result.

## Data and selectors

`env_data/train.jsonl.gz` contains 900 tasks and
`env_data/validation.jsonl.gz` contains 100. `reset(seed, split)` indexes the
seed into a fixed shuffled permutation. `split` must be `train` or
`validation`; unknown reset selectors fail instead of silently selecting a
random problem.

`job_data/train.jsonl` and `job_data/validation.jsonl` contain only selectors,
not problem content:

```json
{"seed": 0, "split": "train"}
```

## Layout

| Path | Purpose |
| --- | --- |
| `server/code_rl_environment.py` | reset, typed MCP tool, and final grader |
| `server/models.py` | observation and explicit per-episode state |
| `server/code_grading.py`, `server/lcb_utils.py` | fenced parsing and child-process hidden-test execution |
| `server/dataset.py` | dataset loading, split selection, selector validation |
| `server/app.py` | single-session OpenEnv ASGI app |
| `env_data/` | compressed runtime task snapshot |
| `job_data/` | training/validation selector rows |

## Build and run

The Dockerfile pins `openenv==0.6.0` and the immutable SDK source revision
`e56ac36c6b497f74cc7cb1caf1f2fa07bd0cb241`; the public SDK wheel at that
version does not yet contain `RLEnvironment`.

```bash
docker build -t code-rl .
docker run --rm -p 8000:8000 code-rl
```

Run focused tests from the repository root after installing those dependencies:

```bash
python -m pytest -q examples/gym/openenv/code_rl/tests
python -m compileall -q examples/gym/openenv/code_rl/server
```

Publish with the normal Gym/OpenEnv workflow (`azd ai rle publish`), then use
the selector JSONL files for training and validation. Keep the server at one
worker: managed RLE scales containers, while one WebSocket session must retain
one environment instance for the entire episode.
