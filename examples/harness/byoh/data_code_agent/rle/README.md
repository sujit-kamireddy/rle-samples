# Standalone OpenEnv environment

[`ByohRLEEnvironment`](./server/environment.py) extends `RLEnvironment`
from `azure-ai-projects[rle]`, the Foundry RLE authoring base class, and owns
one episode per OpenEnv session. The base class supplies `GradeAction` and
routes a non-MCP step into `grade`. HTTP MCP tools and WebSocket simulation
control share that instance; multiple sessions can run independently in one
container.

This service is separate from the existing [Harness environment](../rle_deprecated).
Grading is not reimplemented from scratch, and it is not imported from the
legacy harness either: this folder carries its own copy of the
[grader](./server/vendor/grader.py) and the
[compliance evaluator](./server/compliance.py). The agent runner, legacy HTTP
API, and CSV provisioning remain unchanged.

That duplication is a deliberate stopgap. `../rle_deprecated` is scheduled for deletion
once this variant is proven, and nothing in this folder or in its image reads
it, so the delete is a `rm -rf` rather than a migration. While both copies
exist, `test_distinct_service_package_carries_its_own_grading_modules` asserts
they are distinct objects loaded from byte-identical files, and the parity
suites drive both services over the whole dataset and require the rewards to
match. Those tests skip automatically once `../rle_deprecated` is gone.

## Delivery phases

1. **Local phase:** standalone OpenEnv, parity with existing RLE grading/tools,
   and independent concurrent environment instances.
2. **Foundry phase using RLE:** deployment, integration, and validation details
   **TBA**, after local validation. Local concurrency does not establish Foundry
   multi-rollout-per-sandbox support.

## Build and run locally

Run all commands below from the parent `data_code_agent/` directory unless
otherwise specified. The **Docker build context is this `rle/` folder**, not
the sample root, matching where `azd ai rle publish` resolves `rle.toml` and
builds from, so paths in the Dockerfile are written relative to here. The
image copies nothing from `../rle_deprecated`: not the agent, not the legacy
HTTP server, and not its grading modules. The Dockerfile-specific ignore file
restricts the build context to this folder, excluding tests and Python
bytecode.

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

The legacy answer key remains in [rle_deprecated/server/vendor/task-meta](../rle_deprecated/server/vendor/task-meta).
OpenEnv's [task metadata](./server/vendor/task-meta) adds question text while
preserving every original field and row position. Both artifacts are generated
from the same upstream source using [build_task_meta.py](../tools/build_task_meta.py):

```bash
python tools/build_task_meta.py --verify
python tools/build_task_meta.py --openenv --verify
# Regenerate only the OpenEnv artifact when updating its source:
python tools/build_task_meta.py --openenv --write
```

The OpenEnv flag is explicit so the legacy builder's default target and format
remain unchanged. Metadata tests compare the artifacts and guard the original
answer key with its pre-conversion hash.

## Parity and concurrency validation

With test requirements installed, run from the sample root:

```bash
# Existing Harness command is unchanged and does not import OpenEnv.
(cd rle_deprecated && python -m unittest discover -s tests -t .)

# Focused local implementation and differential checks.
OPENENV_PARITY_REPORT=/tmp/openenv-focused-parity.json \
  python -m unittest rle.tests.test_metadata \
    rle.tests.test_environment rle.tests.test_parity

# Required full-dataset gate; allow approximately 20 minutes.
OPENENV_PARITY_REPORT=/tmp/openenv-parity.json \
  python -m unittest rle.tests.test_full_dataset

# Real containers, HTTP MCP, WebSockets, and independent concurrent sessions.
docker build -t byoh-rle-legacy:local rle_deprecated
docker build -f rle/Dockerfile -t byoh-rle-openenv:local rle/
LEGACY_TEST_IMAGE=byoh-rle-legacy:local \
OPENENV_TEST_IMAGE=byoh-rle-openenv:local \
OPENENV_LIVE_REPORT=/tmp/openenv-live.json \
  python -m unittest rle.tests.test_live
```

The full gate compares legacy HTTP handlers with real OpenEnv HTTP MCP/WebSocket
handlers: four candidate variants, each with and without disclosure, for every
vendored task (8N cases). Exact rewards, correctness-only success, selected task,
and tool outcomes must agree, with zero unexpected mismatches or required skips.

Focused tests cover fixed grading outcomes, tolerances/lists, input normalization,
private-state filtering, contract differences, and overlapping operations.
Live tests exercise separate concurrent instances, capacity, detach/reattach,
deferred close, and idle cleanup in real containers.

Parity applies to **valid episode outcomes**, not identical protocol envelopes:
the legacy API keeps permissive reset, malformed-report zero scores, and consuming
disclosure semantics; OpenEnv requires explicit selectors and typed actions, then
rejects further mutations after its terminal grade. These tests do not establish
BYOH harness trajectory or Foundry rollout parity.
