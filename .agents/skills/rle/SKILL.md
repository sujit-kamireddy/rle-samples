---
name: rle
license: MIT
metadata:
  version: "1.0"
  # Bump major when either subtype's wire contract or rle.toml schema changes.
  # Bump minor when adding a reference or a newly observed failure mode.
description: >-
  **WORKFLOW SKILL** — Authors, validates and ships a Foundry RLE environment of any type: a
  Gym/OpenEnv environment RLE drives itself, or a Harness (BYOH or HostedAgent) wrapping a
  production agent that drives itself. Covers rle.toml, local run/rollout, and publish.

  INVOKES: azd ai rle CLI (init, run, publish, list, show, rollout, train), docker, curl, python,
  git.

  USE FOR: writing any Foundry RLE environment; deciding Gym vs Harness and, for Harness, BYOH vs
  HostedAgent; rle.toml, schema-driven model_response_field, SDK RLEnvironment/GradeAction,
  environment_protocol, max_episode_steps, max_completion_tokens, GET /schema action vocabulary,
  reward shaping, reset selectors, a Harness's /invoke or Responses wire contract,
  agentName/agentVersion vs baseUrl, and debugging EnvironmentContractViolation,
  RolloutDependencyFailed, degenerate_rollout, a wrong episode being graded, or a truncated answer.

  DO NOT USE FOR: training-job hyperparameter tuning, Foundry model deployment, or changes to the
  RLE service itself.
---

# rle

Authors a Foundry RLE environment — Gym/OpenEnv or Harness — that a rollout can actually drive (or
be driven by) and grade.

## Overview

RLE has one job either way: run a rollout against your container(s) and produce a graded episode.
How the model loop is driven is what tells the two types apart, and it decides almost everything
else about how you author the environment:

| | Gym/OpenEnv | Harness (BYOH / HostedAgent) |
| --- | --- | --- |
| Who drives the model loop | RLE itself | Your own agent, which RLE invokes once per rollout |
| RLE talks to your container over | OpenEnv WebSocket `reset`/`step`; SDK MCP tools also use the session's MCP endpoint | a fixed contract: `/health`, `/reset`, `/tools/<name>`, `/grade` |
| What you author | One container implementing either the schema-driven or SDK MCP OpenEnv contract | Two pieces: your `agent/` (always yours) and RLE's own `rle/` container |
| `rle.toml` `[rle]` | `type = "Gym"`, `subtype = "OpenEnv"` | `type = "Harness"`, `subtype = "BYOH"` or `"HostedAgent"` |

If you don't already know which type you're building: pick **Gym/OpenEnv** when you want RLE to run
the model completion loop for you against a plain tool/reward server; pick **Harness** when you
already have (or are building) a production agent — with its own tool use, multi-step planning, or
existing deployment — that should keep driving itself while RLE just scores what it produces.

Start from a sample instead of a blank folder: `azd ai rle init` copies one out of this repo,
prompting for type/subtype/sample. `examples/gym/openenv/math_rl` and `code_rl` demonstrate
schema-driven Gym authoring; `mcp_rl` demonstrates the additive SDK `RLEnvironment` MCP mode.
`examples/harness/{byoh,hosted-agent}/*` are the Harness samples.

RLE never reads your source for either type. Everything it needs, it takes declaratively from
`rle.toml` and (for Gym) your `GET /schema`, so most authoring failures are contract failures, not
logic bugs — and several of them fail *silently*, grading a real reward for the wrong episode.
Follow the references below rather than inferring the contract from one working sample.

## If you are building a Gym/OpenEnv environment

A Gym/OpenEnv environment is a container that speaks the OpenEnv protocol. Foundry RLE leases one
container per rollout, opens a WebSocket to it, and then runs the loop itself:

```
reset(task)  ->  observation  ->  model completion  ->  action  ->  step()  ->  reward, done
```

| RLE needs | It reads it from |
| --- | --- |
| what the model may emit | schema-driven: `GET /schema` `action`; SDK MCP: registered `self.tool()` methods plus `GradeAction` |
| how the contract is selected | schema-driven: `defaults.gym_openenv.model_response_field`; SDK MCP: top-level `environment_protocol = "mcp_environment"` |
| how many steps, how many tokens | `rle.toml` → `defaults.reinforcement`, clamped by server ceilings |
| the episode to run | the job's `task` JSON, forwarded verbatim to `reset()` |

{{ references/anatomy.md }}

{{ references/manifest.md }}

{{ references/action-vocabulary.md }}

{{ references/server-contract.md }}

{{ references/grading-and-budget.md }}

{{ references/workflow.md }}

{{ references/troubleshooting.md }}

### Gym/OpenEnv exit criteria

- Choose exactly one contract. Schema-driven environments declare exactly one action variant
  carrying `model_response_field`; SDK MCP environments subclass `RLEnvironment`, register typed
  tools with `self.tool()`, and grade `GradeAction`.
- `rle.toml` has `schema_version = "1.0.0"`, `[rle] type = "Gym"` / `subtype = "OpenEnv"`,
  and matching reinforcement budgets. Schema-driven mode sets
  `[defaults.gym_openenv] model_response_field`; SDK MCP mode instead sets top-level
  `environment_protocol = "mcp_environment"` and must not set `model_response_field`.
- `reset()` ends in `**kwargs` and rejects selectors it does not understand.
- The episode reaches `done=True` with a real reward on every terminal path — including the
  give-up path — so the rollout is never exported ungraded.
- `azd ai rle run` serves the container, and a scripted `reset`/`step` over `/ws` returns the
  expected reward for a known episode.
- `azd ai rle publish` then `azd ai rle rollout <name> --version <v> --model <m> --task '{"seed": 0}'`
  returns a rollout whose graded episode is the one the task asked for.

## If you are building a Harness environment (BYOH or HostedAgent)

{{ references/harness.md }}
