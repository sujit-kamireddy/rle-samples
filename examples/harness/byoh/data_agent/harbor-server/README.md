# harbor-server

Standalone `openenv.harbor` server for the Hugging Face
[`FineEnvs/data-agent-harbor-*`](https://huggingface.co/collections/FineEnvs/data-agent)
collection: a FastAPI service that serves the Data Agent dataset's tasks over
the Task API, exposes a `run_rollout` MCP tool, and runs a capture proxy so
every model call an agent makes is recorded with token ids and logprobs (when
the endpoint supports it).

Adapted from `openenv.harbor`'s reference server
([`openenv/envs/harbor_env/server/app.py`](https://github.com/huggingface/OpenEnv/blob/main/envs/harbor_env/server/app.py)).
Beyond defaulting `OPENENV_DATASETS` to `FineEnvs/data-agent-harbor-train`,
this version adds:

- `POST /correlated-rollouts/{rollout_id}` -- the reward-hacking fix
  described in `../README.md`. Runs a rollout, reads
  the sandbox's own `/workdir/answer.txt` (collected via a Harbor
  `artifacts` entry patched into every task's `task.toml` -- see
  `vendor/harbor-datasets/`) off local disk before the sandbox is torn down,
  and returns it as `answer_text` alongside the rollout result in the same
  response. `../agent` calls this to trigger a rollout and relays
  `answer_text` on to RLE; `../rle`'s `/grade` grades that text itself
  rather than trusting any score `../agent` reports.

  It reaches Harbor through `HarborEnvironment._run_rollout` -- the method
  this server's own `run_rollout` MCP tool delegates to -- rather than
  through `HarborEnv` over a loopback HTTP connection as it originally did.
  Nothing about how a rollout runs changed; what changed is that the work now
  happens inside the request's own context, which is what `rollout_tools`
  needs.
- `server/rollout_tools.py` -- delivers each rollout's own
  `sandbox_tools_endpoint`/`sandbox_tools_bearer_token` into its sandbox as
  `COMPLIANCE_ENDPOINT`/`COMPLIANCE_TOKEN`, so the agent can file the
  compliance disclosure that `../rle` grades. Worth reading before changing:
  the obvious channel (`AgentConfig.env`) silently does not reach the sandbox
  for this harness, and the credentials are context-scoped rather than
  process-global because concurrent rollouts would otherwise overwrite each
  other's -- misattributing a disclosure with no error anywhere. Covered by
  `tests/test_rollout_tools.py`, which asserts against the real Harbor
  `OpenCode` class rather than a stub.
- Capture-proxy exposure without a public tunnel. The sandbox runs
  off-cluster and has to reach this server's capture proxy over the internet.
  openenv's default route to that is a gradio tunnel, which mints a fresh
  unauthenticated `*.gradio.live` address on every boot -- a third-party URL
  in front of the model route, outside the ingress everything else here is
  protected by. openenv already has a better branch: given a single public
  host and port it mounts the capture app on this server's own app at
  `/capture` and skips forwarding entirely. Azure Container Apps is exactly
  that shape and injects the app's FQDN as `CONTAINER_APP_HOSTNAME`, so on
  ACA this needs no configuration; elsewhere, set `HARBOR_PUBLIC_HOST`. When
  neither resolves, `OPENENV_EXPOSE` decides, and it now defaults to `direct`
  rather than `gradio` so a misconfigured deployment fails visibly instead of
  quietly publishing a tunnel. Local development against an off-cluster
  sandbox can still opt back in with `OPENENV_EXPOSE=gradio`.
- `vendor/harbor-datasets/` -- the full `FineEnvs/data-agent-harbor-train`
  task suite (5,000 tasks, ~196MB), baked into the image via a plain
  Dockerfile `COPY` instead of a build-time `prefetch()` download. A fresh
  container serves every task the moment it passes its health check, with no
  Hugging Face Hub reachability needed at build *or* run time. Re-vendor by
  running `prefetch()` yourself against a different `OPENENV_DATASET_CACHE`
  and copying the result in, if you need a different split.

  `server/app.py`'s `_resolve_dataset` is what makes that guarantee hold.
  openenv resolves a dataset spec by shape: an existing directory is used
  as-is, anything shaped like `org/name` goes to `snapshot_download`
  (`openenv.harbor.tasks`). So the logical id is translated to the baked
  directory *before* openenv sees it, which makes the Hub branch unreachable
  rather than merely disabled, and a spec with no baked copy raises instead
  of quietly falling back to a download. `HF_HUB_OFFLINE=1` is still set, but
  only as a backstop for other callers -- gradio pulls `huggingface_hub` in
  transitively, so the library's presence is not something this image can
  remove. The translation happens here rather than on the wire because
  `../agent` and `../rle` agree on the logical dataset id, and `../rle`
  derives its vendored answer-key filename from it; a container path in that
  field would couple the graded container to this one's filesystem layout.

  Three patches are applied to the upstream tarball, all reproducible and all
  verifiable without a rebuild: the `artifacts` entry above; the
  data-handling clause spliced into every `instruction.md` by
  `../tools/bake_compliance_instruction.py`; and the dataset source repointed
  from the upstream Hugging Face bucket to a public HTTPS object store by
  `../tools/bake_bucket_source.py`. Each script is idempotent and has a
  `--verify` mode that re-reads the archive and reports drift rather than
  assuming its own last run held.

  That third patch is what removes the last credential from the data path.
  Upstream, every task pulled its Kaggle files through `huggingface_hub`
  using an `HF_TOKEN` threaded in from the host -- 5,000 rollouts depending on
  another account staying reachable and the token staying valid. Each
  `task.toml` now names a `BUCKET_BASE_URL` instead, and each task's
  `environment/pull_bucket.py` is replaced by `../tools/pull_bucket.py`, which
  fetches over anonymous HTTPS with nothing but the standard library. The
  store grants anonymous read on blobs but not container listing, so the file
  list for a prefix travels with it as `<prefix>/_manifest.txt`; the fetcher
  decides "already downloaded" against that manifest rather than against "is
  the directory non-empty", so a health-check retry after a partial download
  finishes the job instead of handing the agent a truncated dataset.
  `BUCKET_PREFIX` is untouched -- the upstream prefixes already match the
  store's `<owner>__<dataset>` layout one-for-one.

  That manifest check is also what makes the fetch resumable, which the
  largest prefixes need: 844MB across 2,004 files takes minutes to pull, well
  past the task's 180s health-check timeout, so being killed mid-fetch is the
  normal case rather than the exceptional one. Each file is moved into place
  as soon as it lands rather than publishing the batch at the end, so every
  attempt keeps the work of the one before it and the retry loop converges
  (measured: 420 -> 838 -> 1,806 -> 2,004 files across four attempts, then a
  clean exit). Publishing atomically at the end would instead make each
  attempt discard the last one's progress and the health check would never
  pass. Downloads stage through a hidden sibling directory, so a partial file
  is never visible under a name the manifest lists and `/home/user/input/`
  never accumulates `.part` debris. Set `BUCKET_WORKERS` to widen or narrow
  the fetch concurrency (default 16).
- `vendor/wheels/` -- a pinned `openenv==0.5.0` wheel, downloaded straight
  from PyPI. Needed because this sample requires `openenv>=0.5.0` for
  `openenv.harbor`, but the internal package feed proxy (the
  `PIP_FALLBACK_INDEX_URL` this Dockerfile falls back to when public PyPI is
  unreachable from the build environment) mirrors PyPI with a lag and may
  only have older releases. `--find-links` lets `uv` pick the vendored wheel
  up without ever hitting the network for this one package.

This container owns the sandbox, the agent loop, and Harbor's own grader for
every task. It is almost entirely RLE-unaware -- `../agent` is what bridges it
to RLE's `Harness`/`BYOH` contract, and `../rle` is what RLE itself talks to.
The one exception is `rollout_tools.py`, which has to know that a rollout may
come with a tool capability attached, because Harbor has no concept of one.
See `../README.md` for how the three pieces fit together.

## Run locally

```bash
cd harbor-server
python3.12 -m venv .venv && source .venv/bin/activate
pip install ./vendor/wheels/openenv-0.5.0-py3-none-any.whl
pip install -e .
mkdir -p /tmp/harbor-datasets && tar xzf vendor/harbor-datasets.tar.gz -C /tmp/harbor-datasets
OPENENV_DATASET_CACHE=/tmp/harbor-datasets \
HF_HUB_OFFLINE=1 \
OPENENV_LLM_URL=https://api.openai.com/v1 \
OPENENV_LLM_API_KEY=$OPENAI_API_KEY \
OPENENV_MODEL=gpt-5-mini \
  python -m server.app
```

```bash
curl -s localhost:8000/health
curl -s localhost:8000/schema | jq .
```

`OPENENV_LLM_URL` is optional at boot -- `../agent` overrides it per rollout
with RLE's capture proxy endpoint (`run_rollout(llm_url=...)`) -- but a local
smoke test needs one endpoint to actually call.

## Tests

```bash
cd harbor-server
python -m unittest discover -s tests -t .
```

Needs the same venv as above plus `harbor` itself (`pip install -e .` pulls
it). The tests exercise the real Harbor `OpenCode` class on purpose: the
mechanism `rollout_tools.py` relies on is a Harbor implementation detail, so a
stub would confirm a channel Harbor actually discards.

## Build and deploy

```bash
docker build -t data-agent-harness:local .
docker run --rm -p 8000:8000 \
  -e OPENENV_LLM_URL=https://api.openai.com/v1 \
  -e OPENENV_LLM_API_KEY=$OPENAI_API_KEY \
  -e OPENENV_MODEL=gpt-5-mini \
  data-agent-harness:local
```

Push this image anywhere `../agent` can reach over HTTPS (Azure Container
Apps, your own cluster, a VM) and point `../agent`'s `HARBOR_SERVER_URL` at
it. `run_rollout`'s sandbox backends (`e2b`, `modal`, `daytona`, ...) each need
their own credentials as environment variables on this container; pick one
you already hold credentials for.
