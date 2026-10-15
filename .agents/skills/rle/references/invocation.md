# Harness invocation protocol

Do not confuse invocation with environment MCP. Invocation runs the user's
agent; MCP calls the already-reset environment's tools. The returned answer
is graded by RLE through `GradeAction.response`, not REST `/grade`.

## BYOH: start, poll, withdraw

Set `[harness] type = "byoh", protocol = "rle-invocation"` and register the
**exact** `endpoint`, such as `https://host.example/invoke`.
RLE POSTs there without inventing a `/invoke` suffix. It must be an absolute
HTTPS URL without embedded credentials, query parameters or a fragment.

```http
POST <endpoint>
Content-Type: application/json
```

```json
{
  "rollout_id": "<correlation id>",
  "operation_id": "<service-minted invocation id>",
  "agent_input": [
    {"role":"user","content":"<rendered reset observation>"}
  ],
  "rollout_context": {
    "model_endpoint": "<capture proxy URL>",
    "model_api_key": "<key>",
    "mcp_endpoint": "<complete session-scoped MCP URL>",
    "mcp_bearer_token": "<token>"
  }
}
```

Take ownership and immediately acknowledge, before the agent finishes:

```http
202 Accepted
Content-Type: application/json

{"retry_after_ms":500}
```

Store work/results by **`operation_id`**, not `rollout_id`. Rollout ID is for
correlation and is project-scoped; operation ID is the unique invocation handle.
Do not generate `invocation_id`, return a callback URL, or wait for the agent
inside the start request.

RLE derives the resource by appending `/rollouts/{operation_id}` to the
registered invocation path:

```http
GET <endpoint>/rollouts/{operation_id}
```

Return HTTP 200 with one of these states:

```json
{"status":"running"}
```

```json
{"status":"succeeded","output_text":"<final answer string>"}
```

```json
{"status":"failed","error":{"message":"<safe diagnostic>"}}
```

`output_text` is a **string**. A task-specific JSON answer envelope is valid
only when the paired grader explicitly decodes it. Do not stringify arbitrary
progress logs or append hidden rubric/reference information.

On withdrawal:

```http
DELETE <endpoint>/rollouts/{operation_id}
```

Stop work where possible and return 2xx (the sample uses 204); make late/repeated
withdrawal idempotent. Cancel is advisory and might not arrive. Bound runtime,
storage and result retention independently; use a state store appropriate to
the deployment rather than assuming one in-memory process survives restarts.

RLE does not automatically attach caller/workspace/identity credentials to BYOH
start/poll/cancel. The operation ID is also a capability for uncredentialed poll
and cancel: don't log or expose it publicly. Do not instruct users to require an
arbitrary secret header that RLE has no configured way to send. Use only an
explicitly supported ingress/auth arrangement and verify it with a real start,
poll and cancel; otherwise document this limitation before production exposure.

## HostedAgent: Responses request

Set `[harness] type = "byoh", protocol = "responses"` and bind `endpoint` to
the same-project Foundry agent URL ending in `/agents/<name>`. Do not specify a
version or append the dispatch suffix yourself. RLE sends a standard Responses
request to the agent's configured endpoint:

```http
POST <project>/agents/<agentName>/endpoint/protocols/openai/responses?api-version=v1
x-ms-client-request-id: <rollout id>
Foundry-Features: HostedAgents=V1Preview
x-client-rle-rollout-id: <rollout id>
x-client-rle-model-endpoint: <capture proxy URL>
x-client-rle-model-api-key: <per-rollout key>
x-client-rle-mcp-endpoint: <complete session-scoped MCP URL>
x-client-rle-mcp-bearer-token: <per-rollout token>
```

Platform authorization is supplied by RLE; do not replace it with the capture
proxy key. Context values are request headers, not environment variables set
when the container boots.

```json
{
  "agent_session_id":"<rollout id>",
  "input":[
    {
      "type":"message",
      "role":"user",
      "content":[{"type":"input_text","text":"<serialized rendered chat messages>"}]
    }
  ],
  "background":true,
  "stream":false,
  "store":true
}
```

Latest remote service source uses **background Responses**: `background:true`
and `store:true` let it poll `GET .../responses/{responseId}` while status is
`queued` or `in_progress`, then extract the completed response. `failed` or
`cancelled` is an invocation failure. The agent must support this standard
Responses lifecycle; don't create a BYOH `/rollouts/{operation_id}` resource
for HostedAgent.

Older sample guidance describes synchronous
`background:false, stream:false, store:false`. Treat this as version-dependent,
not a universal HostedAgent guarantee; confirm the deployed service and host
SDK support the requested mode before changing the agent.

The text contains JSON-serialized first-turn messages, not the raw task row.
Parse the Responses text, unwrap the expected chat-message content, then parse
the task-specific prompt if it is JSON. Reject missing/unrecognized task input;
don't accidentally research `"None"` or default to a different task.

Return a normal Responses payload containing the final assistant
`output_text`, for example:

```json
{"output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"<final report>"}]}]}
```

This is the relevant output fragment; use the Responses host SDK to generate
the complete compliant response. Extraction uses top-level `output_text` if
present, otherwise assistant message `content` items of type `output_text`.
The current service can ask for a final answer again in the same agent
session when the completed response has no final text. Keep progress and
function-call items separate from the final report; don't rely on that retry
to compensate for an incorrectly shaped response.

A function returning this fragment is **not a runnable Hosted Agent**. Wire it
into the sample's actual Responses host and start that host with its declared
dependencies. For the current background mode, test POST acceptance with a
response ID/status, GET retrieval by that ID, and completed assistant text.
An `async def` function or a unit test calling it directly does not prove
background Responses support. If the host SDK, deployment or lifecycle cannot
be exercised, label the result an adapter-only prototype and report the blocker;
do not invent a synchronous server and claim HostedAgent completion.

The tested `2.1.0` API imports `ResponsesAgentServerHost`, `ResponsesServerOptions`,
`CreateResponse` and `ResponseContext` from `azure.ai.agentserver.responses`
(not from `azure.ai.agentserver.core`). Register an async
`@host.response_handler(request, context, cancellation_signal)` handler.
`ResponseEventStream` from `azure.ai.agentserver.responses.aio.streaming`
provides `emit_created`, `emit_in_progress`, `output_item_message` and
`emit_completed` events. The SDK host supplies the actual Responses routes,
response IDs, background storage and retrieval; do not recreate them.
Read `context.client_headers` per request, observe cancellation, bound the
agent work, and close owned model clients. Propagate failures to the host
instead of returning a traceback as a successful assistant answer.

Test the real ASGI host with HTTP requests, not only the adapter function.
Then test the built agent and environment containers together: reset one
session, invoke background Responses, verify the agent's MCP calls modify that
session, retrieve the final response, and grade it on the original connection.
A local mock capture proxy can verify URL/key wiring without model costs,
but is not proof of managed capture or a deployed Foundry agent.

There is no **custom BYOH** acknowledgement, poll resource or cancel leg here.
Standard Responses background retrieval is a different protocol.

Use the sample's `ResponsesAgentServerHost`, `ResponseContext.client_headers`,
input parser and request-local model/tool construction as prior art. Preserve
normal production traffic only if the agent actually supports it; the current
sample rejects traffic without RLE headers rather than silently using fallback
clients.
