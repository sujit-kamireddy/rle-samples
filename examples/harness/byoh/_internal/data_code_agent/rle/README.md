# Maintainer-only `rle/` tooling

The sibling copy referenced from the sample's
[`rle/README.md`](../../../data_code_agent/rle/README.md): the scripted
client, dataset-generation tooling, and test suites that keep the shipped
sample correct, kept out of the scaffolded tree so `azd ai rle init` does not
hand a new user a pile of things only needed to maintain the sample itself.

Run every command below from `_internal/data_code_agent/` (one level up from
this file), not the sample root. `rle/tests/conftest.py` puts the real sample
root on `sys.path` so `rle.server` imports still resolve from there.

## Scripted client

```bash
python -m pip install -r ../../data_code_agent/rle/requirements-test.txt
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

The sample's [task metadata](../../../data_code_agent/rle/server/tasks/task-meta)
is generated from the same upstream source as the rest of the dataset, using
[`build_task_meta.py`](../tools/build_task_meta.py):

```bash
python tools/build_task_meta.py --verify  # diff against the committed key
python tools/build_task_meta.py --write   # regenerate the committed key
```

`--verify` fails on any drift in a task's question, expected answer, reward
mode, tolerances, or sensitivity label; `rle/tests/test_metadata.py` pins the
same invariants (plus a hash of every non-question field) so corruption of the
baked dataset is caught in CI, not at grading time.

## Test suite

With test requirements installed:

```bash
# Unit tests: dataset metadata and the in-process environment/app.
python -m pytest rle/tests/test_metadata.py rle/tests/test_environment.py agent/tests/

# Real container, HTTP MCP, WebSockets, and independent concurrent sessions.
# (build the image as in the sample's "Build and run locally" first)
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
