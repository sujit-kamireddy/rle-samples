## byoh harness (`responses` / `rle-invocation`)

A byoh harness environment wraps a *production agent* that already owns its own loop. Unlike
gym, RLE never calls `reset`/`step`: it invokes the harness once per rollout, the harness
runs the agent to completion itself, and reports back a single final response for RLE's own
`rle/` container to grade.

Every byoh sample ships two independent pieces that deploy and publish separately:

- `agent/` — the harness/agent code. Always yours to write and deploy; the CLI never scaffolds it.
  For `rle-invocation` this is a small service you host anywhere and point `endpoint` at. For
  `responses` this is the agent code that runs as a Foundry hosted agent.
- `rle/` — the RLE-side container `azd ai rle init --type byoh` scaffolds for you: task setup,
  mock tools, and a grader, behind `rle/server/env.py`, plus `rle/rle.toml` and `rle/Dockerfile`.

`rle/` is identical in shape whichever protocol invokes it — RLE does not care which one called the
harness — so most of what differs between `rle-invocation` and `responses` lives entirely in `agent/`'s
entrypoint and `rle.toml`'s `[harness]` table.

### Choosing a protocol

| | `responses` (Foundry hosted agent) | `rle-invocation` (RLE's own BYOH contract) |
| --- | --- | --- |
| Where the agent runs | A Foundry hosted agent | Anywhere you can serve an https endpoint |
| `rle.toml` identifies it by | `[harness] endpoint` = the agent's responses URL, `protocol = "responses"` | `[harness] endpoint` = your service URL, `protocol = "rle-invocation"` |
| Invocation | One synchronous call | Async: start, then poll, then optional cancel |
| Hosting burden | None — Foundry hosts it | You deploy and protect the endpoint yourself |

Start from a sample instead of a blank folder: `azd ai rle init --type byoh --protocol <protocol>
--sample <name>` copies one out of this repo's `examples/harness/<byoh|hosted-agent>/` and scaffolds
`rle/`. The two protocol samples of the same environment are typically near-identical at the code
level — same `rle/`, same agent loop — with only the entrypoint differing, so porting one to the
other protocol is usually small.

### The RLE-side container's contract (`rle/server/env.py`)

RLE calls a byoh rollout's own container over a small, fixed surface — deliberately not the
OpenEnv `reset`/`step` protocol:

- `GET /health` — readiness probe RLE polls before it will attempt `/reset`.
- `POST /reset` — a liveness formality on the `rle-invocation` path, not a data channel. Called with the
  caller's `--task` before RLE invokes the harness; use it to pin the task by `rollout_id` so
  `/grade` can look it up later even though the harness itself never sees this container.
- `POST /tools/<name>` — any mock tool the task needs, reachable at
  `{sandbox_tools_endpoint}/tools/<name>` as a sibling of `/reset` and `/grade`.
- `POST /grade` — computes `reward` itself from the harness's raw answer, rather than trusting a
  score the harness claims. RLE forwards the harness's final `output_text` (or Responses output)
  here verbatim as `agent_response`; this is the only place a reward is produced.

### `rle-invocation` wire contract

RLE invokes the harness asynchronously against `[harness] endpoint`. Two identifiers arrive
and answer different questions: `rollout_id` is the correlation handle both sides log;
`operation_id` is what RLE polls and cancels, is minted per invocation, and — because the poll and
cancel legs carry no credential — is also what authorizes those calls. Key your own state on
`operation_id`; `rollout_id` is unique per project, not globally.

1. **Start.** RLE POSTs the rollout; the harness acknowledges `202` as soon as it has taken
   ownership, before the rollout has run:

   ```
   POST <base-url>
   {
     "rollout_id": "...", "operation_id": "...",
     "agent_input": { ... },
     "rollout_context": {
       "model_endpoint": "https://.../rle/v1.0/capture-proxy/v1",
       "model_api_key": "...",
       "sandbox_tools_endpoint": "https://.../rollouts/<rollout-id>/tools",
       "sandbox_tools_token": "..."
     }
   }

   202 Accepted
   {"retry_after_ms": 500}
   ```

2. **Poll.** RLE GETs the rollout resource until it reports an outcome — this can take minutes,
   since one rollout is one blocking agent run:

   ```
   GET <base-url>/rollouts/{operation_id}

   200 {"status": "running"}
   200 {"status": "succeeded", "output_text": "..."}
   200 {"status": "failed", "error": {"message": "..."}}
   ```

   `output_text` is forwarded to `rle/`'s `/grade` verbatim as `agent_response` — that string is
   how the agent's answer reaches the grader.

3. **Withdraw (advisory).** If RLE stops waiting, it DELETEs the rollout resource so the harness can
   stop spending tokens. RLE may not send this at all, so keep your own timeout regardless, but
   answer 2xx when it arrives — anything else is recorded as a cleanup failure.

   ```
   DELETE <base-url>/rollouts/{operation_id}
   ```

RLE never attaches caller, workspace, or identity headers to these requests, so an `rle-invocation` endpoint has
no built-in auth and must protect itself (for example, a shared-secret header validated before
starting a rollout).

### `responses` wire contract

One synchronous request per rollout — no acknowledgement, no poll resource, no cancel leg:

```
POST <endpoint>/endpoint/protocols/openai/responses?api-version=v1   # <endpoint> = {project}/agents/{name}; RLE appends the path
Authorization: ******
x-ms-client-request-id: <rollout id>
Foundry-Features: HostedAgents=V1Preview
x-client-rle-rollout-id: ...
x-client-rle-model-endpoint: https://.../rle/v1.0/capture-proxy/v1
x-client-rle-model-api-key: ...
x-client-rle-sandbox-tools-endpoint: https://.../rollouts/<rollout-id>/tools
x-client-rle-sandbox-tools-token: ...

{
  "agent_session_id": "<rollout id>",
  "input": [{"type": "message", "role": "user",
             "content": [{"type": "input_text", "text": "<agent_input>"}]}],
  "background": false, "stream": false, "store": false
}
```

`agent_input` is not a JSON object here: RLE serialises whatever `rle.toml`/the rollout request
configured into that single `input_text` string, so the agent has to parse it back out (and should
tolerate a bare scalar, since `azd ai rle rollout --agent-input <value>` can send one directly). The
agent answers synchronously with a normal Responses API payload; its final assistant message text is
what reaches `rle/`'s `/grade` as `agent_response`.

### `rle.toml` for byoh

Same manifest shape as gym, with a `[harness]` table that points at your endpoint:

```toml
[rle]
name = "my_harness_env"
version = "1.0.0"

[harness]
type = "byoh"
endpoint = "https://your-harness.example.com/invoke"
protocol = "rle-invocation"   # or "responses"; optional
```

`endpoint` is required for `byoh` (absolute `https`, no credentials, query, or fragment) and must be
omitted for `gym`. `protocol` is optional: when omitted the service resolves it to `responses` if
`endpoint` is your own Foundry project agent URL, otherwise `rle-invocation`.

Gym (default, `[harness]` may be omitted):

```toml
[harness]
type = "gym"
```

byoh with a Foundry agent endpoint (`responses`):

```toml
[harness]
type = "byoh"
endpoint = "https://<account>.services.ai.azure.com/api/projects/<project>/agents/<name>"
protocol = "responses"
```

byoh with a custom https service (`rle-invocation`):

```toml
[harness]
type = "byoh"
endpoint = "https://your-harness.example.com/invoke"
protocol = "rle-invocation"
```

The old `type = "Gym"|"Harness"`, `subtype`, `agent_name`, `agent_version`, `invocation_url`, and
`baseUrl` keys are removed and rejected; agent versions are no longer targetable.

- `[defaults.rollout.gym_openenv]` (`model_response_field`) is gym-only and rejected for
  byoh.
- `[defaults.train.grpo]` is still meaningful for byoh — the harness owns its own
  model-call/turn loop, so RLE never reads it at rollout time, but `azd ai rle train` flattens it
  into the training job's hyperparameters, so this is where a byoh sample's tuned GRPO defaults
  live. There is no manifest field for a completion-token budget any more; RLE applies a fixed
  server ceiling regardless of rollout kind.
- There is no `[train]` / `[train.options]` table any more. `model`, `training-file`,
  `validation-file`, and `suffix` are CLI-flag-only on `azd ai rle train` — there is no manifest
  fallback for them.

### Workflow

1. `azd ai rle init --type byoh --protocol <responses|rle-invocation> --sample <name>` scaffolds `rle/`
   from a working sample and copies its paired `agent/` alongside it as a reference to adapt.
2. Build and deploy `agent/` — anywhere reachable over HTTPS for `rle-invocation`, or as a Foundry
   hosted agent for `responses` — independently of the RLE side.
3. Point `rle.toml` at it (`[harness] endpoint`, and `protocol` if not inferable), then
   `azd ai rle publish`.
4. `azd ai rle rollout <name> --version <v> --agent-input '...'` exercises one rollout end to end
   before training: confirm the harness answers, and that `/grade` grades the episode the task
   asked for.
5. `azd ai rle train` as usual — a byoh rollout still trains through the same recipe, it just
   never touches `[defaults.rollout.gym_openenv]` or `model_response_field`, which are
   gym-only.

### Exit criteria

- `rle.toml` has `[harness] type = "byoh"`, an absolute https `endpoint`, and (when the
  endpoint is not your own Foundry agent URL) the correct `protocol`.
- `rle/server/env.py`'s `/reset` pins whatever `/grade` needs to identify and score the right
  episode, keyed so a concurrent rollout with a different task can't cross-contaminate it.
- The harness answers the real wire contract for its protocol (202 + poll + optional cancel for
  `rle-invocation`; one synchronous Responses call for `responses`) — not a guess at it.
- `azd ai rle rollout` against a deployed harness returns a rollout whose graded episode is the one
  the task asked for, not a stale or mismatched one.
