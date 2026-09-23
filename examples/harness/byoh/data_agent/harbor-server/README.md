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
- `vendor/harbor-datasets/` -- the full `FineEnvs/data-agent-harbor-train`
  task suite (5,000 tasks, ~196MB), baked into the image via a plain
  Dockerfile `COPY` instead of a build-time `prefetch()` download. A fresh
  container serves every task the moment it passes its health check, with no
  Hugging Face Hub reachability needed at build *or* run time
  (`HF_HUB_OFFLINE=1`). Re-vendor by running `prefetch()` yourself against a
  different `OPENENV_DATASET_CACHE` and copying the result in, if you need a
  different split.

  Two patches are applied to the upstream tarball, both reproducible and both
  verifiable without a rebuild: the `artifacts` entry above, and the
  data-handling clause spliced into every `instruction.md` by
  `../tools/bake_compliance_instruction.py`. That script is idempotent and
  its `--verify` mode strips the clause back out to prove nothing else in the
  archive moved.
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
