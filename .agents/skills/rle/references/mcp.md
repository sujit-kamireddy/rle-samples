# OpenEnv MCP protocol and session routing

This is OpenEnv `0.6.0` JSON-RPC with session extensions. Do not assume arbitrary
Streamable HTTP MCP clients implement those extensions or create the right
episode automatically.

## Managed rollout lifecycle

RLE owns session creation, reset, terminal grading and cleanup. Gym tool work
runs through its OpenEnv session; Harness agents get an HTTP MCP endpoint
already scoped to that session. The agent must **not** reset, open a replacement
session or grade itself.

The runtime context carries:

```json
{
  "model_endpoint": "<capture proxy base URL>",
  "model_api_key": "<per-rollout credential>",
  "mcp_endpoint": "https://<service>/rollouts/<id>/mcp?rle_session_id=<encoded-id>",
  "mcp_bearer_token": "<per-rollout tool credential>"
}
```

Use the complete supplied `mcp_endpoint` unchanged, with
`Authorization: Bearer <mcp_bearer_token>`. Do not parse/remove/rebuild its query,
append `/tools/<name>`, or insert your own `session_id` into the request body.
The SDK `create_app` wrapper adapts the query selector into OpenEnv's session
parameter for `tools/list` and `tools/call`.

The query selects state; it is **not authentication**. A local environment is a
trusted/private service, not a multi-tenant security boundary. Keep full URLs,
tokens, operation IDs and model credentials out of logs and source control.

## HTTP tool requests

POST JSON to the scoped endpoint:

```json
{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}
```

```json
{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"check_equivalence","arguments":{"candidate_expression":"1+1","comparison_expression":"2"}}}
```

Discovery returns `result.tools` with `name`, `description` and `inputSchema`.
The tested OpenEnv `0.6.0` HTTP app serializes call results with snake-case
`result.structured_content` and `result.is_error`, alongside `result.content`.
Standard MCP clients/servers may instead use `structuredContent` and `isError`.
Do not assume standard MCP spelling describes this OpenEnv HTTP boundary.
Match the documented tool result instead of assuming every result is a plain
string or a single JSON object.

Check all failure layers: HTTP status, JSON-RPC `error`, both tool-error
spellings, and documented task errors. HTTP 200 can carry a protocol error.
For a small direct HTTP adapter, the essential checks are:

```python
response.raise_for_status()
payload = response.json()
if "error" in payload:
    raise RuntimeError("MCP JSON-RPC failure")
result = payload["result"]
if result.get("is_error") or result.get("isError"):
    raise RuntimeError("MCP tool failure")
```

Use safe diagnostic context rather than logging credentials or entire payloads.
Do not catch these failures and feed their content into the model as successful
tool evidence. Test the actual HTTP adapter against captured local responses
or a mock transport; injected fake tool clients bypass its parsing and headers.
Tool exceptions may arrive as JSON-RPC errors rather than error-marked results;
tests should accept either explicit rejection, not require one error layer.

## Local detached-session smoke test (Harness-style)

This is the author/test-client lifecycle, not something the managed agent runs.
Run it against the SDK wrapper app on localhost:

1. `POST /mcp`:

   ```json
   {"jsonrpc":"2.0","id":1,"method":"openenv/session/create","params":{}}
   ```

   Retain `result.session_id`; do not mistake it for rollout or operation ID.
2. Attach `ws://127.0.0.1:8000/ws?session_id=<URL-encoded-id>`.
3. Send reset with that environment's actual selectors:

   ```json
   {"type":"reset","data":{"split":"FineEnvs/data-agent-harbor-train","task_index":30}}
   ```

4. POST `tools/list` and `tools/call` to
   `/mcp?rle_session_id=<URL-encoded-id>`. Assert tool state is recorded on the
   instance attached in step 2, not a temporary environment.
5. Send final grading on the same WebSocket:

   ```json
   {"type":"step","data":{"response":"<known submission>"}}
   ```

   Check the reply's `data.observation`, `data.reward` and `data.done`.
6. Detach the WebSocket, then POST close in `finally`:

   ```json
   {"jsonrpc":"2.0","id":3,"method":"openenv/session/close","params":{"session_id":"<id>"}}
   ```

HTTP-created sessions survive detach. Close while attached can report
`closed:false, closing:true`; detach to complete cleanup. Idle reclamation
applies to detached sessions, not a substitute for cleaning up attached clients.
Capacity errors are explicit; do not respond by opening more workers.

For a plain Gym test, attach `/ws`, reset, exercise OpenEnv MCP actions using
the sample's client/tests, and submit `GradeAction` on that connection. Preserve
the transport the app supports; don't assume the Gym app has the SDK's HTTP
query-routing wrapper.

Body-only `params.session_id` works locally when the query is absent. With the
SDK wrapper, one nonempty query selector takes precedence for tool methods.
Missing both can create an uninitialized temporary environment upstream, which
is why routing and tool-before-reset guards need tests. No MCP tools named
`reset` or `grade` are required: simulation control stays on the WebSocket.
