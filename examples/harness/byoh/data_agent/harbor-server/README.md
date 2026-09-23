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

- `POST`/`GET /correlated-rollouts/{rollout_id}` -- the reward-hacking fix
  described in `../README.md`. `POST` loopback-triggers a `run_rollout` call
  and stores the full result keyed by `rollout_id`; `GET` returns it (404 if
  nothing is recorded yet). `../agent` calls `POST` to trigger a rollout;
  `../rle`'s `/grade` calls `GET` to fetch the authoritative result
  independently, rather than trusting whatever `../agent` reports.
- `vendor/harbor-datasets/` -- the full `FineEnvs/data-agent-harbor-train`
  task suite (5,000 tasks, ~196MB), baked into the image via a plain
  Dockerfile `COPY` instead of a build-time `prefetch()` download. A fresh
  container serves every task the moment it passes its health check, with no
  Hugging Face Hub reachability needed at build *or* run time
  (`HF_HUB_OFFLINE=1`). Re-vendor by running `prefetch()` yourself against a
  different `OPENENV_DATASET_CACHE` and copying the result in, if you need a
  different split.
- `vendor/wheels/` -- a pinned `openenv==0.5.0` wheel, downloaded straight
  from PyPI. Needed because this sample requires `openenv>=0.5.0` for
  `openenv.harbor`, but the internal package feed proxy (the
  `PIP_FALLBACK_INDEX_URL` this Dockerfile falls back to when public PyPI is
  unreachable from the build environment) mirrors PyPI with a lag and may
  only have older releases. `--find-links` lets `uv` pick the vendored wheel
  up without ever hitting the network for this one package.

This container owns the sandbox, the agent loop, and Harbor's own grader for
every task. It has no RLE awareness at all -- `../agent` is what bridges it to
RLE's `Harness`/`BYOH` contract, and `../rle` is what RLE itself talks to. See
`../README.md` for how the three pieces fit together.

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
