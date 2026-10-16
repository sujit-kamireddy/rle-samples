# Harness architecture

A Harness keeps the production agent's own loop. The environment and agent
are independent deployments:

| Piece | Responsibilities | Not responsible for |
| --- | --- | --- |
| `rle/` environment | Reset task, simulated MCP tools, private state, final grading | Production model planning/agent loop |
| `agent/` | Parse first turn, run its model/tool loop, return final answer | Reporting a trusted reward, opening/resetting the grading session |
| RLE service | Create/reset session, invoke agent, capture model calls, grade and clean up | Reading agent/environment source to infer behavior |

Both `byoh` protocols share the same SDK/OpenEnv/MCP environment shape as Gym.
There is no separate legacy `/tools/<name>` + REST `/grade` environment to add.
What differs is the **invocation adapter**:

- **BYOH / `rle-invocation`**: agent anywhere reachable via `[harness].endpoint`;
  asynchronous POST, poll and advisory cancel. Context in request body.
- **HostedAgent / `responses`**: same-project Foundry agent URL in
  `[harness].endpoint`, not an agent name/version binding.
  The current service uses background Responses and polls the stored result;
  older integrations use synchronous Responses. Context is in per-request
  `x-client-rle-*` headers, not the BYOH envelope.

RLE resets the environment with the task **before** invoking the agent. The
reset observation becomes the first-turn chat messages. The final answer is
submitted through `GradeAction` on the same session that saw the tool calls.

## Wiring rules

- Read model/MCP context per request. Construct the model client with the
  capture proxy URL and key; use the supplied MCP URL and bearer token.
- Do not hardcode a production model endpoint for RLE traffic: a plausible
  answer without captured calls is not a useful training rollout.
- Keep tools production-shaped (names, parameters, behavior), while making
  external effects simulated and deterministic in the environment.
- Separate expected agent mistakes from infrastructure faults. Wrong answers
  are graded; broken task parsing/transport/grading must fail explicitly.
- Preserve request-scoped state under concurrent invocation. Do not cache
  credentials, sessions or task identities in one mutable global client.
- Keep answer keys and scoring logic outside the agent's shell/filesystem.
  Session isolation does not create OS sandbox isolation.
- Give the agent a bounded runtime and cleanup path; advisory cancellation may
  never arrive.

## Editing or porting an existing agent

Change the entrypoint/context/tool adapters rather than rewriting the agent
loop. When porting BYOH to HostedAgent (or vice versa), test first-turn
serialization, output extraction and task identity independently of scoring.
Rebind `[harness]` in the manifest and deploy the agent before publishing.

The samples are intentionally different tasks, not interchangeable agent
packages: BYOH uses `data_code_agent`; HostedAgent uses
`competitive_intelligence_agent`. Their shared environment base does not mean
one can point a manifest at the other's agent and obtain meaningful grades.

Read the skill's invocation, MCP, environment and workflow references for the
actual wire shapes and validation commands.
