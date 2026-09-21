# `code_repair` OpenEnv environment

The Gym/OpenEnv sibling of the
[`examples/harness/byoh/code_repair`](../../harness/byoh/code_repair) and
[`examples/harness/hosted-agent/code_repair`](../../harness/hosted-agent/code_repair) "code repair"
examples: same real-GitHub-issue task shape, but a `Gym`/`OpenEnv` RLE rather
than a `Harness` one, and with real task diversity -- 40 real
SWE-bench-Verified instances (`django/django`, version `3.2` -- the largest
single-repo, single-environment slice in the dataset), one drawn per episode,
rather than one fixed instance repeated every rollout.

See [`../../../README.md`](../../../README.md) for the general contract; this
file only covers what's specific to code repair. See
[`../code_rl/README.md`](../code_rl/README.md) for another `Gym` example with
a similar baked-dataset/`EpisodePicker` design.

## Gym vs. Harness

Both flavors grade real regression tests against real `django/django`
checkouts. What differs is who calls whom:

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
| `server/schema.py` | `CodeRepairAction` (`patch: str` — a unified diff), `CodeRepairObservation` (`messages`, `instance_id`, `problem_id`, `episode_id`). |
| `server/dataset.py` | `EpisodePicker`/`load_jsonl` — seed-based row selection over the baked `env_data/` snapshot, same as `math_rl`/`code_rl`. |
| `server/code_repair_environment.py` | `CodeRepairEnvironment`: `reset(seed, split)` picks a row, checks out its `base_commit` into a fresh `git worktree`, and returns the real GitHub issue as the observation. `step()` applies the submitted patch and grades it by running the real regression test(s) with Django's own test runner — always `done=True`. |
| `server/app.py` | FastAPI app (`create_fastapi_app(...)` from `openenv.core.env_server.http_server`), served with `uvicorn`. |
| `scripts/dataset_source.py`, `scripts/build_dataset.py` | Maintainer-only tools that regenerate `env_data/`'s baked snapshot from `princeton-nlp/SWE-bench_Verified` (not part of the Docker build — see `.dockerignore`). |
| `rle.toml` | Host-agnostic RLE identity and control-plane interface (`Gym` / `OpenEnv`). |
| `env_data/` | Baked, checked-in snapshot of the 40 usable instances (`train.jsonl.gz`/`validation.jsonl.gz`, plus `source.json`/`NOTICE.md` provenance). Read from local disk at server start — no network access. |
| `job_data/` | Training-job input manifests (`{"seed": .., "split": ..}` per row) — see `job_data/README.md`. |
| `Dockerfile` | Clones the full `django/django` repository at build time (so every instance's commit is a local, network-free `git worktree` checkout) and installs `openenv` + Django's own trimmed test dependencies. No sandbox service or runtime egress — grading runs `tests/runtests.py` as a local subprocess against the rollout's own worktree. |

## How RLE drives this environment

RLE reads the action vocabulary from `CodeRepairAction`'s JSON Schema at
`GET /schema`. `CodeRepairAction` declares one action, so the model is offered
no tools: every completion becomes a single graded step, and `step()` always
ends the episode. `rle.toml` names the field that receives the completion
text verbatim:

```toml
[defaults.gym_openenv]
model_response_field = "patch"
```

The model's whole completion lands in `patch`, so the prompt asks for a
unified diff and nothing else.

RLE sets no `max_tokens` of its own on this path — it is the harness here — so
`rle.toml` also states the per-turn output budget. Left unset the request
inherits the sampler's default (1024 tokens today), which stops a reasoning
model mid-thought and submits a truncated answer that still grades:

```toml
[defaults.reinforcement]
max_completion_tokens = 8192
```

RLE sends the smaller of that and its own ceiling, and the capture proxy applies
its own cap on top — 8192 today, which is why nothing here asks for more.

## Grading

Each `reset(seed, split)` picks a row from the baked dataset the same way as
`math_rl`/`code_rl` (`seed % 40` into a fixed shuffled permutation, per
`split`), then:

1. Checks out that row's `base_commit` into a fresh
   `/tmp/rollouts/<episode_id>` directory via
   `git worktree add --detach` against the full `django/django` clone baked
   into the image (`/opt/base-repo`) — no copy, no network access.
2. Applies the row's hidden regression-test patch there — never shown to
   the policy, only used for grading.

`step(action)` then:

1. Applies `action.patch` to that same working directory with `git apply`.
   A patch that fails to apply grades `reward: 0.0` immediately.
2. Runs the instance's real `fail_to_pass` test(s) with
   `python tests/runtests.py --settings=test_sqlite <dotted test labels>`
   (Django's own test runner puts the checkout on `sys.path` itself, so no
   `pip install -e .` is needed).
3. Returns `reward: 1.0` if the test(s) then pass, `0.0` otherwise.
4. Removes the worktree either way — each episode gets its own, and
   nothing outside this `step()` call needs it again.

The episode always ends on this single `step()` call (`done: true`) — there
is no partial credit for an unapplied or partially-correct patch, and no
follow-up turn to revise it.

## Container and registry setup

Install the Azure CLI, `azd`, and the RLE extension, then choose a runtime:

- **Docker Desktop:** start it in Linux-container mode and check `docker info`.
- **Podman:** install Podman (Podman Desktop is optional) and check `podman info`.
  On Windows/macOS, first run `podman machine list`; run `podman machine init`
  only if no machine exists, then `podman machine start` if it is stopped.
  Native Linux does not need a Podman machine.

Select the runtime in the **same terminal** used for `run` and `publish`.
Use `podman` below, or replace it with `docker` for Docker Desktop:

```powershell
$env:AZD_CONTAINER_RUNTIME = "podman"
$env:DOCKER_COMMAND = $env:AZD_CONTAINER_RUNTIME
```

Bash equivalent:

```bash
export AZD_CONTAINER_RUNTIME=podman
export DOCKER_COMMAND="$AZD_CONTAINER_RUNTIME"
```

`AZD_CONTAINER_RUNTIME` selects the container runtime for RLE.
`DOCKER_COMMAND` selects the runtime used by
[`az acr login`](https://learn.microsoft.com/azure/container-registry/container-registry-authentication#sign-in-by-using-an-alternative-container-tool-instead-of-docker).
Docker Desktop is not required when using Podman.

### Before the first publish

Sign in and authenticate the selected runtime to ACR (use the registry
**name**, without `.azurecr.io`). Your user needs push permission:
`AcrPush`, or `Container Registry Repository Writer` on an ABAC-enabled registry.

```text
az login
azd auth login
az acr login --name "<registry>"
```

Local login does **not** grant the RLE service permission to pull your image.
An administrator must grant that separately to the Foundry **project's
system-assigned managed identity**, not your user or the parent account identity.
Find the project's ARM resource ID in Azure portal (it ends in
`/accounts/<account>/projects/<project>`, not the project's HTTPS endpoint).
Ensure the project's system-assigned identity is enabled, then get its
principal ID and the registry scope:

```text
az resource show --ids "<project-arm-resource-id>" --query identity.principalId --output tsv
az acr show --name "<registry>" --query id --output tsv
az role assignment create --assignee-object-id "<project-principal-id>" --assignee-principal-type ServicePrincipal --role AcrPull --scope "<registry-resource-id>"
```

Replace the last command's placeholders with the preceding outputs.
For registries using **RBAC Registry + ABAC Repository Permissions**, replace
`AcrPull` with `"Container Registry Repository Reader"`; `AcrPull` is for
RBAC-only registries. The person assigning the role needs role-assignment
permission at the registry scope. Allow time for propagation before publishing.

After publishing, run `azd ai rle show --output json` and wait for the published
version's `diskImageConversionStatus` to be `Ready` before rollout.
If conversion fails, inspect its error; after fixing the cause, increment
`rle.version` before republishing because registered versions are immutable.

## Build and run locally

Complete [container setup](#container-and-registry-setup), then run from the
folder containing `rle.toml` (the folder created by `azd ai rle init`):

```bash
azd ai rle run
```

Builds this Dockerfile, starts an OpenEnv-compatible local runtime at the
printed URL, and opens a local playground.

### Copy-paste smoke test

After `azd ai rle run` opens the `rle>` shell, enter the following two
commands separately. Do not type the `rle>` prompt itself. This environment
is served over `/ws` only (one environment instance for the life of the
connection), so `step()` always grades against the row/workspace the
preceding `reset()` set up -- there is no `problem_id`/`episode_id` to echo
back, and no fallback for a stateless-HTTP runner that sends `reset` and
`step` as separate requests.

```text
reset {"seed": 0, "split": "train"}

step {"patch": "diff --git a/django/db/models/sql/compiler.py b/django/db/models/sql/compiler.py\n--- a/django/db/models/sql/compiler.py\n+++ b/django/db/models/sql/compiler.py\n@@ -727,7 +727,12 @@ def find_ordering_name(self, name, opts, alias=None, default_order='ASC',\n         # If we get to this point and the field is a relation to another model,\n         # append the default ordering for that model unless it is the pk\n         # shortcut or the attribute name of the field that is specified.\n-        if field.is_relation and opts.ordering and getattr(field, 'attname', None) != name and name != 'pk':\n+        if (\n+            field.is_relation and\n+            opts.ordering and\n+            getattr(field, 'attname', None) != pieces[-1] and\n+            name != 'pk'\n+        ):\n             # Firstly, avoid infinite loops.\n             already_seen = already_seen or set()\n             join_tuple = tuple(getattr(self.query.alias_map[j], 'join_cols', None) for j in joins)\n"}
```

The patch above is the real upstream fix for `django__django-13033`
(self-referencing foreign key ordering) — not a synthetic example. Verified
end-to-end against a real build of this image: `reset(seed=0,
split="train")` picks this instance, the patch applies cleanly, and
`tests/runtests.py` then reports `test_order_by_self_referential_fk ...
ok` (`reward: 1.0`). Your own candidate patch works the same way; a patch
that fails to apply, or applies but doesn't fix the test, grades
`reward: 0.0`.

Requires the selected container runtime to be running and a runner/shell that
keeps one `/ws` connection open across both commands.

The checked-in manifest declares this release as version `2.0.0` (bumped
from the single-instance `1.0.0` design). Update its name when copying this
source outside `azd ai rle init`, and update its version before publishing
a subsequent release.

## Iterate, publish, and rollout

Same lifecycle as [`code_rl`](../code_rl/README.md#iterate-with-your-agent-run-publish-and-rollout):

Complete [registry setup](#before-the-first-publish) before publishing.
The snippet below uses Bash; in PowerShell, replace `export NAME="value"`
with `$env:NAME = "value"`.

```bash
# 1. Iterate locally, rebuilding on source changes:
azd ai rle run --watch

# 2. Publish an immutable version:
export FOUNDRY_PROJECT_ENDPOINT="https://<account>.services.ai.azure.com/api/projects/<project>"
export AZURE_CONTAINER_REGISTRY_ENDPOINT="<registry>.azurecr.io"
azd ai rle publish

# 3. Execute one rollout of the published environment, picking a row by seed
#    (see server/dataset.py's EpisodePicker):
azd ai rle rollout --model Qwen/Qwen3-32B --task '{"seed": 0, "split": "train"}'

# 4. Start a training job from the job_data subfolder:
cd job_data
azd ai rle train --rle-name code_repair --rle-version 2.0.0 --model qwen3-32b-1 --suffix rle-cli-smoke-20260918 --training-file ./training.jsonl
```

`azd ai rle init` downloads `training.jsonl` into the initialized RLE
folder's `job_data` subfolder, which is why the training command changes to
that directory first.

## Use the published environment in a trainer loop

A trainer connects to a published RLE version the same way as
[`code_rl`'s trainer example](../code_rl/README.md#use-the-published-environment-in-a-trainer-loop),
using `azure.ai.projects.aio.AIProjectClient.rle.get_openenv_client(...)` to
lease an instance, then calling `reset(seed=.., split=..)` +
`step(patch=...)` directly — one `reset()` + one `step()` call per rollout,
with `seed` swept across an epoch the same way `math_rl`/`code_rl` do (see
`job_data/README.md`).
