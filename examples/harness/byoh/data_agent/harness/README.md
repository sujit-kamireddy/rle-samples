# harness

The harness container for the Hugging Face
[`FineEnvs/data-agent-harbor-*`](https://huggingface.co/collections/FineEnvs/data-agent)
collection: a small FastAPI service that runs one Data Agent task per request by
driving [`opencode`](https://opencode.ai) inside this container, and hands back
the answer text the agent wrote.

Two routes, and only one of them does anything:

- `GET /health` -- readiness, plus the resolved default model id.
- `POST /correlated-rollouts/{rollout_id}` -- runs a rollout and returns
  `answer_text`. `../agent` calls this and relays `answer_text` verbatim to
  `../rle`'s `/grade`, which computes the reward itself. This server never
  produces a reward, which is what stops a harness claiming a score it did not
  earn. See `server/app.py`.

`opencode` is installed at image build time and runs as a child process of this
server. There is no sandbox provider and no per-rollout agent install. The whole
dependency set is `fastapi`, `uvicorn[standard]` and `httpx`.

The trade-off to know about: rollouts share a container, and `opencode` runs with
`--dangerously-skip-permissions`, so the boundary between two concurrent rollouts
is a directory rather than a VM. Two things make that safe. RLE scopes the
capture-proxy session key and the sandbox-tools URL and token per rollout, so
neither is usable across them. And this container holds no grading key to find --
which is the next section.

## What the harness is given

`vendor/task-index.json.gz` (0.4MB, built by `../tools/build_task_index.py`):
each task's instruction, bucket coordinates and agent timeout. Not the task
suite (216MB extracted).

That is a security boundary, not a size optimisation. Every `task.toml` in the
suite carries `metadata.gold_answer` and `verifier.env.EXPECTED_ANSWER`, and
every `instruction.md` quotes its question verbatim while `task.description`
repeats it. An agent with a shell -- which is exactly what
`--dangerously-skip-permissions` grants -- can therefore
`grep -rlF "<its own question>"` the suite, land on its own task directory and
read its own answer without analysing anything. Measured at 25 of 25 sampled
tasks. So the answers do not ship here: the grading key lives in `../rle`'s
container, which gives the agent no shell.

Keep it that way. Anything added to this image is readable by the agent.

## Rollout isolation

A task's instruction names absolute paths -- the CSVs at `/home/user/input`, the
answer at `/workdir/answer.txt`. Those are fine when the rollout owns the whole
container and collide the moment two run side by side. Each rollout gets a
private directory and both paths are rewritten to point inside it. Nothing
downstream depends on the original strings: `/grade` is handed the answer *text*,
never a path. See `server/opencode_direct.py`.

The per-rollout compliance credentials (`sandbox_tools_endpoint` /
`sandbox_tools_bearer_token`) are lifted out of the request body and set as
`COMPLIANCE_ENDPOINT` / `COMPLIANCE_TOKEN` on that rollout's `opencode` process,
so the agent can file the disclosure that `../rle` grades. They are per-process,
never process-global: concurrent rollouts would otherwise overwrite each other's
and misattribute a disclosure with no error anywhere.

## Task inputs

`vendor/pull_bucket.py` (shipped to `/opt/pull_bucket.py`, a generated copy of
`../tools/pull_bucket.py`) fetches a task's input files once per rollout, as a
subprocess of the server.

Each `task.toml` names a `BUCKET_BASE_URL`, fetched over anonymous HTTPS with
nothing but the standard library -- no credentials anywhere in the data path. The
store grants anonymous read on blobs but not container listing, so the file list
for a prefix travels with it as `<prefix>/_manifest.txt`; the fetcher decides
"already downloaded" against that manifest rather than against "is the directory
non-empty", so a retry after a partial download finishes the job instead of
handing the agent a truncated dataset.

That manifest check is also what makes the fetch resumable, which the largest
prefixes need: 844MB across 2,004 files takes minutes to pull, well past the
task's 180s timeout, so being killed mid-fetch is the normal case rather than the
exceptional one. Each file is moved into place as soon as it lands rather than
publishing the batch at the end, so every attempt keeps the work of the one
before it and the retry loop converges (measured: 420 -> 838 -> 1,806 -> 2,004
files across four attempts, then a clean exit). Publishing atomically at the end
would instead make each attempt discard the last one's progress. Downloads stage
through a hidden sibling directory, so a partial file is never visible under a
name the manifest lists. Set `BUCKET_WORKERS` to widen or narrow the fetch
concurrency (default 16).

## Build-time inputs

`vendor/harbor-datasets.tar.gz` is the upstream task suite. It is **not** copied
into the image -- see "What the harness is given" -- it is the source the
`../tools/` scripts read to generate what *is* shipped:
`vendor/task-index.json.gz`, `vendor/pull_bucket.py` and `../rle`'s vendored
answer key.

Three patches are applied to it, all reproducible and all verifiable without a
rebuild: an `artifacts` entry on every task so the agent's `answer.txt` is
collected; the data-handling clause spliced into every `instruction.md` by
`../tools/bake_compliance_instruction.py`; and the dataset source repointed to a
public HTTPS object store by `../tools/bake_bucket_source.py`. Each script is
idempotent and has a `--verify` mode that re-reads the archive and reports drift
rather than assuming its own last run held.

## Run locally

Needs `opencode` on `PATH` (`npm i -g opencode-ai`) and the data-stack packages
every task instruction promises are installed (`pandas`, `numpy`, `matplotlib`,
`seaborn`, `scipy`, `scikit-learn`, `statsmodels`, `tabulate`, `plotly`).

```bash
cd harness
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e .
PULL_BUCKET_PATH=vendor/pull_bucket.py \
MODEL_URL=https://api.openai.com/v1 \
MODEL_API_KEY=$OPENAI_API_KEY \
MODEL_ID=gpt-5-mini \
  python -m server.app
```

```bash
curl -s localhost:8000/health
curl -s -X POST localhost:8000/correlated-rollouts/local-1 \
  -H 'content-type: application/json' \
  -d '{"task_index": 0, "llm_url": "https://api.openai.com/v1", "api_key": "'"$OPENAI_API_KEY"'"}' | jq .
```

`MODEL_URL` is optional at boot -- `../agent` overrides it per rollout with
RLE's capture proxy endpoint -- but a local smoke test needs one endpoint to
actually call. `MODEL_ID` is probed from the endpoint when unset and the
endpoint serves exactly one model.

## Tests

```bash
cd harness
python -m pytest tests -q
```

No extra dependencies beyond `pip install -e .` and `pytest`; the tests drive
`opencode_direct` with the `opencode` invocation stubbed, so they need neither a
model endpoint nor the task suite.

## Build and deploy

```bash
docker build -t data-agent-harness:local .
docker run --rm -p 8000:8000 \
  -e MODEL_URL=https://api.openai.com/v1 \
  -e MODEL_API_KEY=$OPENAI_API_KEY \
  -e MODEL_ID=gpt-5-mini \
  data-agent-harness:local
```

Push this image anywhere `../agent` can reach over HTTPS (Azure Container Apps,
your own cluster, a VM) and point `../agent`'s `HARNESS_SERVER_URL` at it.

Size the container for concurrency: each rollout holds its own copy of the task's
input files on local disk and runs its own `opencode` process.
