# Standalone OpenEnv environment

[`CompetitiveIntelEnvironment`](./server/environment.py) owns one episode per
OpenEnv session. It is built on the `RLEnvironment` base class that
`azure-ai-projects` ships in its `rle` extra, so the tool surface is served over
MCP and the final report is graded through a `GradeAction` step.

This service is **separate from, and additive to**, the existing
[Harness environment](../rle_deprecated). That one is the legacy HTTP tool
route, kept while the grading-parity test still has something to compare
against. This folder carries the [`rle.toml`](./rle.toml) the CLI reads, so
`azd ai rle init`, `publish` and `train` all act on this environment.

Grading is not reimplemented from scratch, and it is not imported from the
legacy harness either. This folder carries its own copy of the world, the
tasks, the simulated tools, the tool-calling primitives they are built on,
and the rubric, plus the response-shaping helpers, in [`server/`](./server).
The tool surface is read off the same `ToolSession` class the Harness
environment uses.

That duplication is a deliberate stopgap. `../rle_deprecated` is scheduled for deletion
once this variant is proven, and nothing in this folder or in its image reads
it, so the delete is a `rm -rf` rather than a migration. While both copies
exist, `test_grade_matches_the_legacy_harness` drives each environment through
its own protocol over its own copy of the rubric and asserts the rewards are
identical, which is what keeps the two from drifting. That test skips
automatically once `../rle_deprecated` is gone.

## Delivery phases

1. **Local phase:** standalone OpenEnv, with grading parity against `../rle_deprecated`
   asserted by a test that drives both services through their own protocols.
2. **Foundry phase using RLE:** the agent in [`../agent`](../agent) now speaks
   both wire formats. It calls tools over this folder's `/mcp` JSON-RPC surface
   when RLE hands it a sandbox session id in the
   `x-client-rle-sandbox-session-id` header, and falls back to the legacy POST
   route when that header is absent, which is exactly the condition the service
   itself applies when it decides whether to send it.

   RLE picks that path from the `environmentProtocol` field in
   [`rle.toml`](./rle.toml), which this folder sets to `mcp_environment`. The
   value is recorded at publish time and is immutable for the version, so
   switching protocols means publishing a new version. `azd ai rle publish`
   prints the protocol the service recorded and fails if it does not match the
   manifest, which is what keeps a silently legacy environment from reaching a
   training run.

## Build and run locally

Run every command below from the parent `competitive_intelligence_agent/`
directory. The **Docker build context is this folder (`rle/`) itself**, not
the sample root: `azd ai rle publish` always builds from the directory
holding `rle.toml`. The agent, its tooling, the job data and `rle_deprecated/`
are sibling folders already outside that context, so the image cannot pick up
a dependency on the folder that is about to be deleted.

```bash
docker build -f rle/Dockerfile -t ci-rle-openenv:local rle/
docker run --rm -p 127.0.0.1:8000:8000 ci-rle-openenv:local
```

Or run it directly, which needs only the sample root on the path:

```bash
python -m pip install -r rle/requirements.txt
PYTHONPATH="$PWD" \
  uvicorn rle.server.app:app --host 127.0.0.1 --port 8000 --workers 1
```

Use **one** Uvicorn worker: session ids belong to the process that minted them.

### Capacity

`OPENENV_MAX_CONCURRENT_ENVS` defaults to `1` and that is the only value this
environment accepts. OpenEnv refuses anything higher unless the environment
class sets `SUPPORTS_CONCURRENT_SESSIONS = True`, and this one does not: the
simulated `WORLD` is built once per process and shared, so concurrent sessions
are not proven safe. One rollout per container also matches what an RLE sandbox
schedules. Setting it higher fails at startup with
`ConcurrencyConfigurationError` rather than corrupting a rollout quietly.

`OPENENV_SESSION_TIMEOUT_SECONDS` defaults to `600`. It bounds how long a
detached session may idle before it is reclaimed, so a rollout that stops
calling tools and never grades cannot hold the only slot for the life of the
container.

## Session and grading contract

This is OpenEnv's JSON-RPC interface with session extensions. It is **not** a
full Streamable HTTP MCP endpoint: there is no `initialize` handshake, so a
stock MCP client will not connect. `POST /mcp` serves exactly four methods:
`openenv/session/create`, `openenv/session/close`, `tools/list` and
`tools/call`.

The plain `POST /reset` and `POST /step` endpoints build and close a fresh
environment per request. They are useful for a smoke test, but a real rollout
must use a session, or the tool calls and the grade land on different instances
and tool discipline scores zero.

1. Create the session and keep `result.session_id`:

   ```json
   {"jsonrpc":"2.0","id":1,"method":"openenv/session/create","params":{}}
   ```

2. Attach `ws://127.0.0.1:8000/ws?session_id=<id>`.
3. Reset through the WebSocket, passing the task row from
   [`job_data/validation.jsonl`](../job_data/validation.jsonl). Rows there are
   wrapped as `{"task": {...}}`; send the inner object:

   ```json
   {"type":"reset","data":{"episode_id":"ep-1","task_id":"...","variant":"...","query":"..."}}
   ```

   A row the environment cannot parse raises rather than grading a task nobody
   asked for. Reset returns the task identity and the tool names for the
   configured surface.
4. Call tools through `POST /mcp`, **always** supplying `params.session_id`:

   ```json
   {"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"session_id":"<id>","name":"web_search","arguments":{"search_query":"..."}}}
   ```

   Every call is recorded on the session the grader reads, which is how tool
   discipline and citation fidelity are scored.
5. Submit the report through the WebSocket:

   ```json
   {"type":"step","data":{"answer":"...the final report..."}}
   ```

   The response carries `reward`, `done: true`, and an `observation.info`
   payload identical to the legacy `/grade` response, including `metrics`,
   `verdict`, `expected_verdict`, `n_tool_calls` and `tools_called`.
6. Detach the WebSocket, then call `openenv/session/close` with
   `params.session_id` from a `finally` block. A leaked session holds the only
   capacity slot until the idle timeout reclaims it.

The reward is clamped to `[0, 1]` as a backstop only. `grade_episode` already
rescales, and that rescale must not be removed.

This is a **trusted local service**, not a multi-tenant one. Session ids route
state; they are not authorization.

## Tests

```bash
python -m pip install -r rle/requirements-test.txt
python -m pytest rle/tests -q
```

The suite asserts the production tool surface, that MCP schemas are generated
from each tool's own argument model, that a tool call before reset is a protocol
error, and that an unreadable task fails loudly. The test that justifies the
port is `test_grade_matches_the_legacy_harness`: it drives this environment and
`../rle_deprecated` through their own public protocols, over two distinct copies of the
rubric, with the same task, the same tool call and the same answer, then asserts
the reward, `is_success` and the whole `info` payload are identical and that the
reward is not trivially zero. It is the gate on the duplicated world, tasks,
tools and rubric, and it skips on its own once `../rle_deprecated` is deleted.

## Dependency note

[`requirements.txt`](./requirements.txt) installs `azure-ai-projects[rle]` from
a GitHub fork. The `rle` subpackage that provides `RLEnvironment` is not in any
released `azure-ai-projects` wheel yet. Swap that line for a normal version pin
once it ships to PyPI; the `TODO` in the file says the same.
