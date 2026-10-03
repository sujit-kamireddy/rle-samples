# MCP environment

[`DataCodeAgentRLEnvironment`](./server/environment.py) extends `RLEnvironment`
from `azure-ai-projects[rle]`, the Foundry RLE authoring base class, and owns
one episode per OpenEnv session. The base class supplies `GradeAction` and
routes a non-MCP step into `grade`. HTTP MCP tools and WebSocket simulation
control share that instance; multiple sessions can run independently in one
container.

This folder is the only RLE service in the sample: it grades a rollout itself
rather than trusting a score relayed back through a harness, using its own
copy of the [grader](./server/vendor/grader.py) and the
[compliance evaluator](./server/compliance.py). The agent runner and CSV
provisioning are unaffected by anything in here.

## Build and run locally

Run all commands below from the parent `data_code_agent/` directory unless
otherwise specified. The **Docker build context is this `rle/` folder**, not
the sample root, matching where `azd ai rle publish` resolves `rle.toml` and
builds from, so paths in the Dockerfile are written relative to here. The
Dockerfile-specific ignore file restricts the build context to this folder,
excluding tests and Python bytecode.

```bash
docker build -f rle/Dockerfile -t byoh-rle-openenv:local rle/
docker run --rm -p 127.0.0.1:8000:8000 byoh-rle-openenv:local
```

Use **one Uvicorn worker**: session IDs belong to that process.

Alternatively, create an isolated Python environment and run from the sample root:

```bash
python -m pip install -r rle/requirements.txt
uvicorn rle.server.app:app --factory --host 127.0.0.1 --port 8000 --workers 1
```

OpenEnv is pinned to `0.6.0`, which supports attaching WebSocket control to an
HTTP-created MCP session. This is OpenEnv's JSON-RPC interface with session
extensions, not a claim of compatibility with every Streamable HTTP MCP client.

### Capacity

`OPENENV_MAX_CONCURRENT_ENVS` defaults to `1`, matching what a real RLE sandbox
schedules per container. Unlike the CI sample, this environment sets
`SUPPORTS_CONCURRENT_SESSIONS = True`: every session gets its own
`DataCodeAgentRLEnvironment` instance with no shared process-level state, so
the default may be raised for higher local throughput, e.g.
`-e OPENENV_MAX_CONCURRENT_ENVS=4`. Once the configured capacity is reached,
further `openenv/session/create` calls fail with a `SessionCapacityError`
(surfaced as a JSON-RPC error carrying `active_sessions` and `max_sessions`)
rather than silently overcommitting.

`OPENENV_SESSION_TIMEOUT_SECONDS` defaults to `600`. It bounds how long a
detached session may idle before it is reclaimed, so a rollout that stops
calling tools and never grades cannot hold its slot for the life of the
container.

Both variables are optional overrides. Set either to an invalid value
(non-integer, non-positive, or non-finite) and startup raises `ValueError`
instead of serving with a broken limit.

## Session and grading contract

1. `POST /mcp` with method `openenv/session/create` and `params: {}`; retain
   `result.session_id`.
2. Attach `ws://127.0.0.1:8000/ws?session_id=<id>`.
3. Reset through the WebSocket:

   ```json
   {"type":"reset","data":{"split":"FineEnvs/data-agent-harbor-train","task_index":30}}
   ```

   `split` is the existing dataset ID, not `"test"`/`"train"` aliases.
   `task_index` must be a non-negative integer. Unknown selectors, booleans,
   missing selectors, and out-of-range indexes fail explicitly. Reset returns
   the question, task identity, and MCP disclosure policy. Provision matching
   CSVs externally; this service neither downloads them nor executes agent code.
4. Invoke `tools/list` and `tools/call` through `POST /mcp`, **always** supplying
   `params.session_id`. For example:

   ```json
   {"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"session_id":"<id>","name":"report_sensitive_data_access","arguments":{"columns_reported":["income"],"note":"Personal data inspected"}}}
   ```

5. Submit `{"type":"step","data":{"answer":"..."}}` through the WebSocket.
   The response's `data` contains `observation`, `reward`, and `done`.
   A valid submission, including an incorrect answer, returns `done: true`.
   `observation.score` equals the compliance-adjusted reward;
   `observation.is_success` measures answer correctness alone.
6. Detach the WebSocket, then call `openenv/session/close` with
   `params.session_id` in a `finally` block.

There are no MCP simulation-control tools. The disclosure acknowledgement never
reveals whether reporting was correct; repeat disclosures replace the prior
record. Per-instance locking serializes reset, disclosure, grading, and close
without a global episode lock.

Before reset and after terminal grading, submissions and mutating tool calls
fail. Reset starts a fresh episode, clearing only that instance's disclosure and
score. Reference answers, sensitivity labels, and detailed grading diagnostics
stay private. Public state snapshots never block behind another thread's grading.

Terminal grading does not release the session. HTTP-created sessions survive
WebSocket detach and can be reattached. HTTP close while attached reports
`closed: false, closing: true`, deferring cleanup until detach. Idle cleanup
reclaims detached sessions only; attached sessions require caller cleanup.
Omitting the HTTP session ID creates a temporary upstream environment; its
uninitialized tool guard rejects disclosure rather than recording it elsewhere.

This is a **trusted local/private service**, not multi-tenant authorization.
Session IDs route state; they are not equivalent to Foundry's authenticated
rollout headers. Instance isolation is not an OS sandbox: do not give an
untrusted agent shell access to this grading container.

## Scripted client

This sample's maintainer-only tooling (this script, `tools/`, and the test
suites below) lives outside the scaffolded tree, in a sibling `_internal/`
copy of this sample -- so `azd ai rle init` does not hand a new user a pile of
things that are only here to keep the sample itself correct. Run the commands
in the rest of this document from `_internal/data_code_agent/`, the root of
that copy, not from this sample root.

```bash
python -m pip install -r rle/requirements-test.txt
python -m rle.scripts.session_demo \
  --url http://127.0.0.1:8000 \
  --split FineEnvs/data-agent-harbor-train \
  --task-index 30 --answer "your submitted answer"
```

Supply `--disclose column1 column2` only when appropriate. Omit it for no
disclosure; `--disclose` alone records an empty list. The script demonstrates
session creation, reset, discovery/call, terminal grading, and cleanup; it does
not run the BYOH harness agent.

## Dataset artifacts

The [task metadata](./server/vendor/task-meta) is generated from the same
upstream source as the rest of the sample's dataset using
[build_task_meta.py](../../_internal/data_code_agent/tools/build_task_meta.py)
(from `_internal/data_code_agent/`, as above):

```bash
python tools/build_task_meta.py --verify  # diff against the committed key
python tools/build_task_meta.py --write   # regenerate the committed key
```

`--verify` fails on any drift in a task's question, expected answer, reward
mode, tolerances, or sensitivity label; `rle/tests/test_metadata.py` pins the
same invariants (plus a hash of every non-question field) so corruption of the
baked dataset is caught in CI, not at grading time.

## Test suite

With test requirements installed, run from `_internal/data_code_agent/`:

```bash
# Unit tests: dataset metadata and the in-process environment/app.
python -m pytest rle/tests/test_metadata.py rle/tests/test_environment.py agent/tests/

# Real container, HTTP MCP, WebSockets, and independent concurrent sessions.
# (build as in "Build and run locally" above, from the sample root, then come back here)
OPENENV_TEST_IMAGE=byoh-rle-openenv:local \
OPENENV_LIVE_REPORT=/tmp/openenv-live.json \
  python -m pytest rle/tests/test_live.py
```

These are plain `pytest` invocations, not `python -m unittest rle.tests...`:
`rle/` here is a placeholder with no `__init__.py` of its own (only `tests/`
and `scripts/` moved), so it cannot resolve `rle.server` the way the real
sample's `rle/` package does; pytest's per-file import handles that, a shared
dotted package path cannot.

Unit tests cover dataset integrity, fixed grading outcomes, tolerances/lists,
input normalization, private-state filtering, and overlapping operations
in-process. Live tests exercise the real image: package boundaries, every
disclosed/correct quadrant for a sensitive and a clean task, separate
concurrent instances, capacity, detach/reattach, deferred close, and idle
cleanup, all over HTTP MCP and WebSocket exactly as Foundry would drive them.
Neither suite establishes end-to-end Foundry rollout parity on its own.
