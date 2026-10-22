---
name: rle
license: MIT
metadata:
  version: "2.8"
description: >-
  Author, edit, migrate, test and publish Microsoft Foundry Reinforcement Learning
  Environments (RLE containers). Use for Gym/OpenEnv, BYOH or HostedAgent,
  choosing RLE types, rle.toml schema, azure-ai-projects[rle] RLEnvironment,
  GradeAction, OpenEnv sessions, MCP tools and session routing, asynchronous
  invocation protocol, Responses API rollout headers, task selectors, reward
  shaping, Docker/Podman, azd ai rle init/run/publish/rollout, Foundry eval vs
  trainable Loom rollouts, and debugging
  contract violations, missing tools, wrong-episode grading or untrainable captures.
  Not for RLE service implementation, model deployment or training hyperparameter tuning.
---

# RLE authoring

Create or edit one working RLE, preserving its task selection, tools and grading.
Use this skill for all supported types; the environment SDK contract is shared,
but the owner of the agent loop and the agent invocation protocol differ.

## Choose the type before changing code

| Harness type / protocol | Who runs the model loop? | What the user supplies | Working sample |
| --- | --- | --- | --- |
| `gym` | RLE | Environment container with optional MCP tools and grading | `math_rl` or `code_rl` |
| `byoh` / `rle-invocation` | User's agent, hosted anywhere reachable by RLE | Environment container **and** async invocation endpoint | `data_code_agent` |
| `byoh` / `responses` | User's Foundry Hosted Agent | Environment container **and** same-project Foundry agent endpoint | `competitive_intelligence_agent` |

Choose Gym for a model that RLE should drive. Choose Harness when an existing or
new agent must keep its own planning/tool loop; choose its invocation protocol.
HostedAgent is a `byoh` target using `responses`, not a separate manifest type.
Do not convert a production agent into Gym just to reuse a sample.

The current schema uses lowercase `gym`/`byoh`, `endpoint` and `protocol`.
Removed `subtype`, `invocation_url`, `agent_name` and `agent_version` keys fail
validation. `[task] format="harbor"` and harness `type="package"` are reserved,
not runnable. Match installed CLI capability to the current samples before
scaffolding or publishing; older public releases can reject this schema.

All four current samples implement **SDK `RLEnvironment` + OpenEnv MCP**. There is
no environment protocol selector to set. Do not author the obsolete schema-driven
`model_response_field` action-union contract or the old Harness
`/reset`, `/tools/<name>`, `/grade` REST surface.

```text
Gym:     RLE -> session/reset -> model <-> MCP tools -> GradeAction -> reward
Harness: RLE -> session/reset -> rendered observation -> invoke user's agent
                 user's agent -> capture proxy + session-scoped MCP tools
         RLE <- final response -> GradeAction on the SAME session -> reward
```

Harness invocation and environment MCP are **different protocols**. A BYOH
`/invoke` endpoint is not the environment's `/mcp` endpoint. The agent returns
an answer, not a trusted reward; the environment computes the reward.

## Procedure

1. **Inspect before editing.** Read the existing manifest, Dockerfile,
   requirements, environment, app factory, agent adapter (Harness), task/job
   data and tests. Identify harness type/protocol and the actual reset payload. Preserve
   working behavior and build context; do not rewrite an agent unnecessarily.
2. **Start from the closest current sample** for a new RLE. Check
   `azd ai rle --help` and `init --help` before using version-dependent flags.
   Install the `azure.ai.rle` extension; core commands no longer require
   `AZD_AI_RLE_ENABLE`. Harness scaffolding and staged training commands use
   `AZD_AI_RLE_ENABLE_ALL=true`.
   Harness scaffolding supplies the RLE side; the agent is independently
   authored/deployed. Do not assume a scaffold contains the sample's maintainer
   tests or a complete production agent.
3. **Author the shared environment.** Subclass `RLEnvironment`, call
   `super().__init__()` before `self.tool()(self.method)`, implement
   `reset(..., **kwargs)` with explicit task validation and
   `grade(GradeAction, ...)`, and pass `GradeAction` plus observation/state
   models to `create_app`. For HTTP session-scoped MCP use the SDK
   `azure.ai.projects.rle.environments.create_app` wrapper.
4. **Wire Harness, if selected.** Use per-invocation model credentials and the
   complete supplied MCP URL unchanged. Validate the BYOH start/poll/cancel
   contract or HostedAgent Responses contract, including first-turn parsing.
5. **Declare the manifest.** Identity belongs in `[rle]`; type, endpoint and
   invocation protocol belong in `[harness]`. Use `[model]` and `[train.grpo]` only
   for supported defaults. Never put credentials or invented budget/protocol
   fields in the manifest.
6. **Verify locally, then publish and roll out.** Test known correct, wrong and
   malformed submissions; invalid task selectors; session isolation; tool
   evidence; reset-to-first-turn rendering; and terminal grades. Confirm a
   managed rollout captured the intended model calls, graded the requested
   task, and returned a finite reward. Exercise actual HTTP/WebSocket payloads
   and adapter transports, not only direct methods or injected client fakes.
   A successful HTTP status is not enough.
   Use `--mode eval` for a Foundry deployment smoke test **only when installed
   help supports it**; the public CLI can lag development source. Its text-only
   capture is intentionally not trainable. Verify the deployed service accepts
   that policy too. Use Loom for training trajectories.
7. **Report honestly.** Separate local checks from managed rollout verification.
   If CLI, credentials, model or container runtime is unavailable, state the
   exact blocker and the remaining commands; never claim a publish succeeded.

## Read the relevant references

Load these files with a file-reading tool, relative to this skill's directory.
They are normal Markdown links, not template includes.

| Reference | Read when |
| --- | --- |
| [Sample anatomy](./references/anatomy.md) | Selecting/scaffolding a sample, SDK install or Docker context |
| [Manifest schema](./references/manifest.md) | Creating/editing `rle.toml`, identity and endpoint/protocol binding |
| [Environment contract](./references/server-contract.md) | Implementing reset, observation rendering, state and grade |
| [MCP protocol](./references/mcp.md) | Session lifecycle, JSON-RPC tools, query routing and errors |
| [Tool vocabulary](./references/action-vocabulary.md) | Adding/editing model-visible tools or migrating action unions |
| [Harness architecture](./references/harness.md) | Separating the agent from the grading environment |
| [Invocation protocol](./references/invocation.md) | BYOH async start/poll/cancel or HostedAgent Responses |
| [Grading and budgets](./references/grading-and-budget.md) | Reward shaping, terminal output, capture and timeouts |
| [Authoring workflow](./references/workflow.md) | Prerequisites, local checks, publish, rollout and training handoff |
| [Troubleshooting](./references/troubleshooting.md) | Contract, routing, capture, build or rollout failures |
| [Verified upstream baseline](./references/upstream-contracts.md) | Checking source/release revisions and refreshing this skill |

## Completion checklist

- Correct type/endpoint/protocol; no stale subtype, schema-driven, REST-tool or invocation-id wiring.
- Valid manifest and SDK build; no runtime secret baked into the image.
- Task selectors fail explicitly when invalid; held-out data stays held out.
- Model-visible prompt/tool results never expose private reference answers.
- Reset, tools and grading share one session; per-request credentials are not global.
- Expected terminal outputs return `done=True` and a real finite reward;
  Harness also returns boolean `is_success`.
- Tests cover task identity, reward values, tool state and session isolation.
- MCP failures cover OpenEnv `is_error` and standard MCP `isError`; a HostedAgent
  output fragment alone is not a runnable Responses host.
- Harness calls the capture proxy rather than its normal production model endpoint.
- Agent failures retain secret-safe diagnostics; access/client/telemetry output
  does not expose credentials or session/operation capabilities.
- Local smoke check passes; managed publish/rollout verified or explicitly blocked.
