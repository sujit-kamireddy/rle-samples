# agent

The BYOH harness for the Hugging Face
[`FineEnvs/data-agent-harbor-*`](https://huggingface.co/collections/FineEnvs/data-agent)
collection: one FastAPI service that speaks RLE's harness contract and runs the
Data Agent tasks itself, by driving [`opencode`](https://opencode.ai) inside this
container.

RLE's contract, all of it in `app.py`:

- `POST /invoke` -- starts one rollout. Acknowledges with `202` and runs the
  rollout in the background; RLE polls for the result.
- `GET /invoke/rollouts/{operation_id}` -- the poll. `202` while the rollout is
  still running, `200` with the result once it has finished.
- `DELETE /invoke/rollouts/{operation_id}` -- withdraws a rollout RLE no longer
  wants.
- `GET /health` -- readiness, plus the resolved default model id.

`operation_id` is RLE's, not ours. It is minted per invocation, it is the only
thing authenticating the poll and withdraw calls, and it is therefore what
rollouts are stored under here. `rollout_id`, which arrives on the same request,
is unique only within a project -- a correlation key for logs, not an identity.

What RLE polls for is the answer text the agent wrote, never a reward. `../rle`
computes the reward itself from that text, which is what stops a harness
claiming a score it did not earn.

`opencode` is installed at image build time and runs as a child process of this
service. There is no sandbox provider and no per-rollout agent install. The whole
Python dependency set is `fastapi`, `uvicorn[standard]` and `httpx`.

The trade-off to know about: rollouts share a container, and `opencode` runs with
`--dangerously-skip-permissions`, so the boundary between two concurrent rollouts
is a directory rather than a VM. Two things make that safe. RLE scopes the
capture-proxy session key and the MCP-tools URL and token per rollout, so
neither is usable across them. And this container holds no grading key to find --
which is the next section.

## What the harness is given

`vendor/task-index.json.gz` (0.4MB, built by
`../../_internal/data_code_agent/tools/build_task_index.py`): each task's
instruction, bucket coordinates and agent timeout. Not the task suite (216MB
extracted).

That is a security boundary, not a size optimisation. Every `task.toml` in the
suite carries `metadata.gold_answer` and `verifier.env.EXPECTED_ANSWER`, so an
agent with a shell (which `--dangerously-skip-permissions` grants) could grep
the suite for its own question text and read its own answer without
analysing anything -- confirmed reproducible on sampled tasks. So the answers
do not ship here: the grading key lives in `../rle`'s container, which gives
the agent no shell.

Keep it that way. Anything added to this image is readable by the agent.

## Rollout isolation

A task's instruction names absolute paths -- the CSVs at `/home/user/input`, the
answer at `/workdir/answer.txt`. Those are fine when the rollout owns the whole
container and collide the moment two run side by side. Each rollout gets a
private directory and both paths are rewritten to point inside it. Nothing
downstream depends on the original strings: `/grade` is handed the answer *text*,
never a path. See `opencode_direct.py`.

If OpenCode exits normally without a non-empty `answer.txt`, the harness resumes
the same OpenCode session once, using the remaining task timeout (at most 120
seconds), and asks it to write the file. If the file is still missing, the
harness reports `ok: false` with a missing-answer error; `/grade` records a
zero-reward rollout rather than silently treating it as a successful answer.

The example keeps rollout state in memory. On Cloud Run, keep it to one instance
unless you replace that store with shared state: a poll can reach a different
instance and return 404, and a restart loses in-flight rollouts. Size the
instance for your intended parallelism -- a 16 GiB instance comfortably ran
four concurrent OpenCode rollouts where 4 GiB hit out-of-memory restarts.

The per-rollout compliance credentials (`mcp_endpoint` / `mcp_bearer_token`,
sent on `/invoke`) are lifted out of the rollout context and set as
`COMPLIANCE_ENDPOINT` / `COMPLIANCE_TOKEN` on that rollout's `opencode` process
(per-process, never global, so concurrent rollouts can't overwrite each
other's). `COMPLIANCE_ENDPOINT` isn't `mcp_endpoint` verbatim: `../rle`
publishes under `environmentProtocol = "mcp_environment"`, so reaching its
tools needs a JSON-RPC `tools/call` on `/mcp`, not a flat route. Rather than
teach the baked instruction a second wire format, the rollout is pointed at a
loopback route on this service (`/local-tools/{operation_id}/...` in
`app.py`) that translates the flat POST into that `tools/call`.

## Task inputs

`vendor/pull_bucket.py` (shipped to `/opt/pull_bucket.py`, a generated copy of
`../../_internal/data_code_agent/tools/pull_bucket.py`) fetches a task's input
files once per rollout, as a subprocess of this service.

Each `task.toml` names a `BUCKET_BASE_URL`, fetched over anonymous HTTPS with
nothing but the standard library -- no credentials anywhere in the data path. The
store grants anonymous read on blobs but not container listing, so the file list
for a prefix travels with it as `<prefix>/_manifest.txt`; the fetcher checks
that manifest, not "is the directory non-empty", to decide what's already
downloaded, so a retry after a partial download finishes the job instead of
handing the agent a truncated dataset.

That manifest check is also what makes the fetch resumable, which the largest
prefixes need: the biggest is 844MB across 2,004 files, well past the task's
180s timeout, so being killed mid-fetch is the normal case. Each file is moved
into place as soon as it lands (verified convergent across retries) rather
than publishing the batch at the end, which would instead discard each
attempt's progress. Downloads stage through a hidden sibling directory, so a
partial file is never visible under a name the manifest lists. Set
`BUCKET_WORKERS` to widen or narrow the fetch concurrency (default 16).

## Build-time inputs

`vendor/harbor-datasets.tar.gz` is the upstream task suite. It is **not** copied
into the image -- see "What the harness is given" -- it is the source the
`../../_internal/data_code_agent/tools/` scripts read to generate what *is*
shipped: `vendor/task-index.json.gz`, `vendor/pull_bucket.py` and `../rle`'s
vendored answer key.

Three patches are applied to it, all reproducible and all verifiable without a
rebuild: an `artifacts` entry on every task so the agent's `answer.txt` is
collected; the data-handling clause spliced into every `instruction.md`; and the
dataset source repointed to a public HTTPS object store. Each script is
idempotent and has a `--verify` mode that re-reads the archive and reports
drift rather than assuming its own last run held.

## Run locally

Needs `opencode` on `PATH` (`npm i -g opencode-ai`) and the data-stack packages
every task instruction promises are installed (`pandas`, `numpy`, `matplotlib`,
`seaborn`, `scipy`, `scikit-learn`, `statsmodels`, `tabulate`, `plotly`).

```bash
cd agent
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
PULL_BUCKET_PATH=vendor/pull_bucket.py \
MODEL_URL=https://api.openai.com/v1 \
MODEL_API_KEY=$OPENAI_API_KEY \
MODEL_ID=gpt-5-mini \
  python -m uvicorn app:app --port 8080
```

```bash
curl -s localhost:8080/health

curl -s -X POST localhost:8080/invoke \
  -H 'content-type: application/json' \
  -d '{
        "rollout_id": "local-1",
        "operation_id": "op-local-1",
        "agent_input": [
          {
            "role": "user",
            "content": "{\"task_index\": 0, \"split\": \"FineEnvs/data-agent-harbor-train\"}"
          }
        ],
        "rollout_context": {
          "model_endpoint": "https://api.openai.com/v1",
          "model_api_key": "'"$OPENAI_API_KEY"'"
        }
      }'

# Poll under the operation_id until it stops answering 202.
curl -s localhost:8080/invoke/rollouts/op-local-1 | jq .
```

Locally you supply `operation_id` yourself; under RLE it is generated for you and
must be echoed back exactly as received. `model_endpoint` is RLE's capture proxy
in a real rollout, so a local smoke test is the one case where it is a real model
endpoint. `MODEL_URL` is only the fallback for a rollout that names none.
`MODEL_ID` is probed from the endpoint when unset and the endpoint serves exactly
one model.

## Tests

```bash
cd agent
python -m pytest tests -q
```

No extra dependencies beyond `requirements.txt` and `pytest`; the tests drive
`opencode_direct` with the `opencode` invocation stubbed, so they need neither a
model endpoint nor the task suite.

## Reading the logs

RLE polls `/invoke/rollouts/{operation_id}` every 500ms for the whole rollout, so
the access log would otherwise carry several hundred identical `200`s per rollout
and bury everything the rollout itself reports. Those lines are filtered out
here; the poll that delivers the result and any poll naming an unknown rollout
are still logged. `POST /invoke` and `DELETE /invoke/rollouts/...` are left alone.

On Cloud Run the platform also logs every request a second time as its own
`httpRequest` entry, which no container-side change can suppress. Drop them when
reading, by asking only for this container's streams:

```bash
gcloud logging read \
  'resource.type=cloud_run_revision
   AND resource.labels.service_name=<your-service>
   AND logName:("stdout" OR "stderr")' \
  --limit 100 --format='value(textPayload)'
```

For a live stream, use `gcloud alpha logging tail` with the same resource filter
and no `--format` (older `gcloud` versions reject `logName:(...)` and silence the
stream under `--format='value(textPayload)'`):

```bash
gcloud alpha logging tail \
  'resource.type=cloud_run_revision
   AND resource.labels.service_name=<your-service>'
```

## Build and deploy

```bash
docker build -t data-code-agent:local .
docker run --rm -p 8080:8080 \
  -e MODEL_URL=https://api.openai.com/v1 \
  -e MODEL_API_KEY=$OPENAI_API_KEY \
  -e MODEL_ID=gpt-5-mini \
  data-code-agent:local
```

Push this image anywhere RLE can reach over HTTPS (Azure Container Apps, your own
cluster, a VM) and register that base URL as your BYOH harness. RLE appends
`/invoke` to it, so register the root.

Size the container for concurrency: each rollout holds its own copy of the task's
input files on local disk and runs its own `opencode` process.
