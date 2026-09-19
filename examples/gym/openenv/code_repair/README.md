# `code_repair` OpenEnv environment

The Gym/OpenEnv sibling of the
[`examples/harness/byoh`](../../harness/byoh) and
[`examples/harness/hosted-agent`](../../harness/hosted-agent) "code repair"
examples: same real SWE-bench-Lite instance
(`psf/requests` issue `psf__requests-3362`, pinned at base commit
`36453b95b1` — see [`fixtures/instance.json`](./fixtures/instance.json)), but
a `Gym`/`OpenEnv` RLE rather than a `Harness` one.

See [`../../../README.md`](../../../README.md) for the general contract; this
file only covers what's specific to code repair. See
[`../code_rl/README.md`](../code_rl/README.md) for another `Gym` example with
a larger dataset and a multi-turn tool-use loop.

## Gym vs. Harness, for this same instance

Both flavors grade the same real regression test
(`tests/test_requests.py::TestRequests::test_response_decode_unicode`)
against the same pinned `psf/requests` checkout. What differs is who calls
whom:

- **Harness** (`examples/harness/*`): RLE invokes an external agent (over
  HTTP for BYOH, or a Foundry Hosted Agent's Responses API) and gives it
  mock `workspace.apply_patch`/`github.create_pull_request` tools to edit the
  checkout turn-by-turn; RLE calls `/grade` once the agent finishes.
- **Gym** (this environment): there is no agent and no tool mocks. An
  external RL trainer/policy calls `reset()`/`step()` on this environment
  directly, over OpenEnv's own HTTP/WS contract. The whole candidate fix —
  not a sequence of tool calls — is the `step()` action itself, and grading
  happens immediately, in that same call.

## What's here

| File | Purpose |
| --- | --- |
| `models.py` | `CodeRepairAction` (`patch: str` — a unified diff), `CodeRepairObservation` (`messages`, `instance_id`). |
| `server/code_repair_environment.py` | `CodeRepairEnvironment`: `reset()` seeds a fresh checkout and returns the real GitHub issue as the observation. `step()` applies the submitted patch and grades it by running the real regression test — always `done=True`. |
| `server/app.py` | FastAPI app (`create_fastapi_app(...)` from `openenv.core.env_server.http_server`), served with `uvicorn`. |
| `server/sitecustomize.py` | Restores `collections.Mapping` and friends (removed in Python 3.10) so the 2016-era pinned `requests` checkout still imports — same fix as `examples/harness/byoh/rle`. |
| `rle.toml` | Host-agnostic RLE identity and control-plane interface (`Gym` / `OpenEnv`). |
| `fixtures/instance.json`, `fixtures/test_patch.diff` | The same real instance metadata and hidden regression-test patch used by the Harness examples. |
| `training/` | Training-job input manifest (a single `{}` row, since `reset()` ignores all per-episode arguments for this single fixed instance). See `training/README.md`. |
| `Dockerfile` | Clones the pinned `psf/requests` checkout at build time and installs `openenv` + the instance's pinned `pytest==6.2.5`. No sandbox service or runtime egress — grading runs `pytest` as a local subprocess against the rollout's own working copy. |

## Grading

Each `reset()` copies the read-only reference checkout baked into the image
(`/opt/base-repo`) into a fresh `/tmp/rollouts/<episode_id>` directory and
applies the hidden regression-test patch there — never shown to the policy,
only used for grading. `step(action)` then:

1. Applies `action.patch` to that same working directory with `git apply`.
   A patch that fails to apply grades `reward: 0.0` immediately.
2. Runs the instance's real `fail_to_pass` test with `pytest` as a
   subprocess (`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, for the same reason as
   the Harness example's `/grade` — third-party pytest plugins pulled in
   transitively by `openenv`/`starlette` crash the instance's pinned
   `pytest==6.2.5`).
3. Returns `reward: 1.0` if the test then passes, `0.0` otherwise.

The episode always ends on this single `step()` call (`done: true`) — there
is no partial credit for an unapplied or partially-correct patch, and no
follow-up turn to revise it.

## Build and run locally

```bash
cd examples/gym/openenv/code_repair
azd ai rle run
```

Builds this Dockerfile, starts an OpenEnv-compatible local runtime at the
printed URL, and opens a local playground.

### Copy-paste smoke test

After `azd ai rle run` opens the `rle>` shell, enter the following two
commands separately. Do not type the `rle>` prompt itself.

```text
reset {}

step {"patch": "diff --git a/requests/utils.py b/requests/utils.py\nindex 8d17b6b2fb..62d023fae9 100644\n--- a/requests/utils.py\n+++ b/requests/utils.py\n@@ -358,13 +358,20 @@ def get_encoding_from_headers(headers):\n \n def stream_decode_response_unicode(iterator, r):\n     \"\"\"Stream decodes a iterator.\"\"\"\n+    encoding = r.encoding\n \n-    if r.encoding is None:\n-        for item in iterator:\n-            yield item\n-        return\n+    if encoding is None:\n+        encoding = r.apparent_encoding\n+\n+    try:\n+        decoder = codecs.getincrementaldecoder(encoding)(errors='replace')\n+    except (LookupError, TypeError):\n+        # A LookupError is raised if the encoding was not found which could\n+        # indicate a misspelling or similar mistake.\n+        #\n+        # A TypeError can be raised if encoding is None\n+        raise UnicodeError(\"Unable to decode contents with encoding %s.\" % encoding)\n \n-    decoder = codecs.getincrementaldecoder(r.encoding)(errors='replace')\n     for chunk in iterator:\n         rv = decoder.decode(chunk)\n         if rv:\n"}
```

The patch above is the real upstream fix (`psf/requests` PR #3362, the
`requests/utils.py` half of the merged change) — not a synthetic example.
Verified locally against a fresh checkout at the pinned base commit plus the
hidden test patch: `git apply` succeeds, then
`tests/test_requests.py::TestRequests::test_response_decode_unicode` passes
(`1 passed`). Your own candidate patch works the same way; a patch that
fails to apply, or applies but doesn't fix the test, grades `reward: 0.0`.

Needs Docker running and the `azd` RLE extension (see
[`../../../README.md`](../../../README.md)).

The checked-in manifest declares the initial `code_repair` release as
version `1.0.0`. Update its name when copying this source outside
`azd ai rle init`, and update its version before publishing a subsequent
release.

## Iterate, publish, and invoke

Same lifecycle as [`code_rl`](../code_rl/README.md#iterate-with-your-agent-run-publish-and-invoke):

```bash
# 1. Iterate locally, rebuilding on source changes:
azd ai rle run --watch

# 2. Publish an immutable version:
export FOUNDRY_PROJECT_ENDPOINT="https://<account>.services.ai.azure.com/api/projects/<project>"
export AZURE_CONTAINER_REGISTRY_ENDPOINT="<registry>.azurecr.io"
azd ai rle publish

# 3. Execute one rollout of the published environment (task is ignored by
#    this environment's reset(), so {} is enough):
azd ai rle invoke --model Qwen/Qwen3-32B --task '{}'
```

## Use the published environment in a trainer loop

A trainer connects to a published RLE version the same way as
[`code_rl`'s trainer example](../code_rl/README.md#use-the-published-environment-in-a-trainer-loop),
using `azure.ai.projects.aio.AIProjectClient.rle.get_openenv_client(...)` to
lease an instance, then calling `reset()`/`step()` directly — there is no
`check_solution`-style tool-call loop to drive here, so each rollout is a
single `reset()` + one `step(patch=...)` call.
