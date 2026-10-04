# Standalone OpenEnv environment

[`CompetitiveIntelEnvironment`](./server/environment.py) owns one episode per
OpenEnv session. It is built on the `RLEnvironment` base class that
`azure-ai-projects` ships in its `rle` extra, so the tool surface is served over
MCP and the final report is graded through a `GradeAction` step. This folder
carries the [`rle.toml`](./rle.toml) the CLI reads, so `azd ai rle init`,
`publish` and `train` all act on this environment.

The world, the tasks, the simulated tools, the tool-calling primitives they
are built on, and the rubric all live in [`server/`](./server), alongside the
response-shaping helpers `environment.py` needs to turn the agent's final
message into the text `grade` scores. There is one copy of each: what decides
a reward and what an RLE training run sees are the same code.

## Delivery phases

1. **Local phase:** run and smoke-test this environment standalone, with no
   RLE or Foundry dependency at all -- see "Build and run locally" below.
2. **Foundry phase using RLE:** the agent in [`../agent`](../agent) calls tools
   over this folder's `/mcp` JSON-RPC surface. RLE hands it the session to call
   them on in the `x-client-rle-mcp-session-id` header, which it only sends
   when the published version's protocol is `mcp_environment`.

   RLE picks that path from the `environmentProtocol` field in
   [`rle.toml`](./rle.toml), which this folder sets to `mcp_environment`. The
   value is recorded at publish time and is immutable for the version, so
   switching protocols means publishing a new version. `azd ai rle publish`
   prints the protocol the service recorded and fails if it does not match the
   manifest, which is what catches a version published on the wrong one before
   it reaches a training run.

## Build and run locally

Run every command below from the parent `competitive_intelligence_agent/`
directory. The **Docker build context is this folder (`rle/`) itself**, not
the sample root: `azd ai rle publish` always builds from the directory
holding `rle.toml`. The agent and the job data are sibling folders already
outside that context, and this sample's maintainer-only tooling and tests live
further out still, in a sibling `_internal/` copy of this sample, so none of
it can leak into the image.

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
   [`job_data/validation.jsonl`](../job_data/validation.jsonl) directly as
   `data`, with an `episode_id` added:

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
   payload with `metrics`, `verdict`, `expected_verdict`, `n_tool_calls` and
   `tools_called`.
6. Detach the WebSocket, then call `openenv/session/close` with
   `params.session_id` from a `finally` block. A leaked session holds the only
   capacity slot until the idle timeout reclaims it.

The reward is clamped to `[0, 1]` as a backstop only. `grade_episode` already
rescales, and that rescale must not be removed.

This is a **trusted local service**, not a multi-tenant one. Session ids route
state; they are not authorization.

## Tests

This sample's maintainer-only tooling (`tools/` and the test suite below)
lives outside the scaffolded tree, in a sibling `_internal/` copy of this
sample -- so `azd ai rle init` does not hand a new user a pile of things that
are only here to keep the sample itself correct. Run the commands below from
`_internal/competitive_intelligence_agent/`, the root of that copy, not from
this sample root.

```bash
python -m pip install -r ../../competitive_intelligence_agent/rle/requirements-test.txt
python -m pytest rle/tests -q
```

The suite asserts the production tool surface, that MCP schemas are generated
from each tool's own argument model, that tool calls land on the session the
rubric reads, that a tool call or a grade before reset is a protocol error,
that an unreadable task fails loudly, and that the reward `grade` returns
always lands inside the `[0, 1]` range RLE requires.

## Dependency note

[`requirements.txt`](./requirements.txt) installs `azure-ai-projects[rle]` from
a GitHub fork. The `rle` subpackage that provides `RLEnvironment` is not in any
released `azure-ai-projects` wheel yet. Swap that line for a normal version pin
once it ships to PyPI; the `TODO` in the file says the same.
