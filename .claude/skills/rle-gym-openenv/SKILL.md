---
name: rle-gym-openenv
license: MIT
metadata:
  version: "1.0"
  # Bump major when the rle.toml contract or the GET /schema action-vocabulary rules change.
  # Bump minor when adding a reference or a newly observed failure mode.
description: >-
  **WORKFLOW SKILL** — Authors, validates and ships a Foundry RLE Gym/OpenEnv environment:
  OpenEnv server, action schema, rle.toml manifest, grading, then local run and published rollout.

  INVOKES: azd ai rle CLI (init, run, publish, list, show, rollout), docker, curl, python, git.

  USE FOR: writing an OpenEnv environment for Foundry RLE, creating a gym sample, rle.toml,
  model_response_field, max_episode_steps, max_completion_tokens, GET /schema action vocabulary,
  adding a tool to an RL environment, reward shaping, reset selectors, and debugging
  EnvironmentContractViolation, RolloutDependencyFailed, degenerate_rollout, a wrong episode
  being graded, or a truncated model answer.

  DO NOT USE FOR: Harness, HostedAgent or BYOH environments (this covers type=Gym subtype=OpenEnv
  only), training-job hyperparameter tuning, Foundry model deployment, or changes to the RLE
  service itself.
---

# rle-gym-openenv

Authors a Foundry RLE Gym/OpenEnv environment that a rollout can actually drive and grade.

## Overview

A Gym/OpenEnv environment is a container that speaks the OpenEnv protocol. Foundry RLE leases one
container per rollout, opens a WebSocket to it, and then runs the loop itself:

```
reset(task)  ->  observation  ->  model completion  ->  action  ->  step()  ->  reward, done
```

RLE never reads your source. Everything it needs it takes from two places:

| RLE needs | It reads it from |
| --- | --- |
| what the model may emit | your `GET /schema` `action` schema |
| which action property holds the completion | `rle.toml` → `defaults.gym_openenv.model_response_field` |
| how many steps, how many tokens | `rle.toml` → `defaults.reinforcement`, clamped by server ceilings |
| the episode to run | the job's `task` JSON, forwarded verbatim to `reset()` |

Three of the four are declarative, so most authoring failures are contract failures, not logic
bugs — and several of them fail *silently*, grading a real reward for the wrong episode. Follow the
references below rather than inferring the contract from one working sample.

Start from a sample instead of a blank folder: `azd ai rle init` copies one out of this repo.
`examples/gym/openenv/math_rl` is the single-turn, no-tools case; `code_rl` is the case with a tool;
`code_repair` is the repo-checkout case.

{{ references/anatomy.md }}

{{ references/manifest.md }}

{{ references/action-vocabulary.md }}

{{ references/server-contract.md }}

{{ references/grading-and-budget.md }}

{{ references/workflow.md }}

{{ references/troubleshooting.md }}

## Exit Criteria

- `GET /schema` on the running container declares exactly one action variant carrying
  `model_response_field`, and every other variant has a discriminator.
- `rle.toml` has `schema_version = "1.0.0"`, `[rle] type = "Gym"` / `subtype = "OpenEnv"`,
  `[defaults.gym_openenv] model_response_field`, and a `[defaults.reinforcement]`
  `max_episode_steps` / `max_completion_tokens` that match how the episode actually behaves.
- `reset()` ends in `**kwargs` and rejects selectors it does not understand.
- The episode reaches `done=True` with a real reward on every terminal path — including the
  give-up path — so the rollout is never exported ungraded.
- `azd ai rle run` serves the container, and a scripted `reset`/`step` over `/ws` returns the
  expected reward for a known episode.
- `azd ai rle publish` then `azd ai rle rollout <name> --version <v> --model <m> --task '{"seed": 0}'`
  returns a rollout whose graded episode is the one the task asked for.
