# agent

The Foundry **Hosted Agent** harness for the Hugging Face
[`FineEnvs/data-agent-harbor-*`](https://huggingface.co/collections/FineEnvs/data-agent)
collection: one Responses-protocol agent that RLE calls directly and that runs
the Data Agent tasks itself, by driving [`opencode`](https://opencode.ai) inside
this container.

The `BYOH` copy of this sample
(`azd ai rle init --type Harness --subtype BYOH --sample data_code_agent`) runs
the same rollout against the same `../rle_deprecated` container. Only the invocation
contract differs, and `opencode_direct.py` is byte-identical between the two.

RLE's contract, all of it in `main.py`:

- One `POST` to this agent's Responses endpoint per rollout, synchronously.
  There is no acknowledgement, no poll resource and no withdraw leg -- all of
  which BYOH has to implement, because BYOH is asynchronous and RLE polls a URL
  you registered. Here RLE resolves the agent from the project, so there is no
  URL to register and nothing to host yourself.
- The rollout's runtime context arrives as five request headers, because the
  Responses protocol has no field to put it in:
  `x-client-rle-rollout-id`, `x-client-rle-model-endpoint`,
  `x-client-rle-model-api-key`, `x-client-rle-sandbox-tools-endpoint`,
  `x-client-rle-sandbox-tools-token`. `azure-ai-agentserver` forwards exactly
  the `x-client-` prefix into `ResponseContext.client_headers`, which is why
  RLE chose it; see `rollout_context.py`.
- `agent_input` arrives as *text*, serialised into the single `input_text` of
  the request's one input message. A configured object therefore arrives as a
  JSON string that has to be parsed back out -- `parse_agent_input` does that,
  and also accepts a bare index so `azd ai rle rollout --agent-input 3` works.
- The final assistant message is this rollout's `output_text`. RLE forwards
  that string verbatim to `../rle_deprecated`'s `/grade` as `agent_response`.
- `GET /readiness` -- the Responses host's own probe route. This agent adds no
  `/health` of its own.

`agent_session_id` is set to the rollout id, and `agentVersion` from
`rle_deprecated/rle.toml` is never sent -- RLE records it as target metadata only, so the
container cannot read its own version.

What RLE receives is the answer text the agent wrote, never a reward. `../rle_deprecated`'s
`/grade` computes the reward itself from that text, which is what stops a harness
claiming a score it did not earn. A rollout that fails returns the traceback as
its answer text rather than raising: an exception out of the handler reaches RLE
as a bare `status: failed` with nothing in the reward to tell two unrelated
failures apart, while a returned traceback still scores zero *and* shows up in
metrics and logs.

`opencode` is installed at image build time and runs as a child process of this
service. There is no sandbox provider and no per-rollout agent install. The whole
Python dependency set is `azure-ai-agentserver-responses` and `httpx`.

The trade-off to know about: rollouts share a container, and `opencode` runs with
`--dangerously-skip-permissions`, so the boundary between two concurrent rollouts
is a directory rather than a VM. Two things make that safe. RLE scopes the
capture-proxy session key and the sandbox-tools URL and token per rollout, so
neither is usable across them -- and because they arrive per *request*, nothing
derived from them may be cached at import, or one rollout's proxy would capture
another's calls. And this container holds no grading key to find -- which is the
next section.

## Timeouts

The service-side budget for Execute Rollout is 1800s and the Hosted Agent
invoker sets nothing shorter, so a long `opencode` run is viable here.
`RLE_ROLLOUT_TIMEOUT_S` (default 900) keeps this agent failing itself first: an
agent that times out returns a transcript and an honest zero, while overrunning
the service returns a bare error and loses the episode.

The one exception is the *HTTP* Execute Rollout path, which a Foundry
data-plane gateway cuts at roughly 120s. `azd ai rle rollout` and training both
drive Execute Rollout over a WebSocket, which is not cut. Lower
`RLE_ROLLOUT_TIMEOUT_S` if you are calling the HTTP path directly.

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
tasks. So the answers do not ship here: the grading key lives in `../rle_deprecated`'s
container, which gives the agent no shell.

Keep it that way. Anything added to this image is readable by the agent.

## Rollout isolation

A task's instruction names absolute paths -- the CSVs at `/home/user/input`, the
answer at `/workdir/answer.txt`. Those are fine when the rollout owns the whole
container and collide the moment two run side by side. Each rollout gets a
private directory and both paths are rewritten to point inside it. Nothing
downstream depends on the original strings: `/grade` is handed the answer *text*,
never a path. See `opencode_direct.py`.

The per-rollout compliance credentials (the
`x-client-rle-sandbox-tools-endpoint` / `x-client-rle-sandbox-tools-token`
headers) are lifted out of the rollout context and set as
`COMPLIANCE_ENDPOINT` / `COMPLIANCE_TOKEN` on that rollout's `opencode`
process, so the agent can file the disclosure that `../rle_deprecated`
grades. They are per-process,
never process-global: concurrent rollouts would otherwise overwrite each other's
and misattribute a disclosure with no error anywhere.

## Task inputs

`vendor/pull_bucket.py` (shipped to `/opt/pull_bucket.py`, a generated copy of
`../tools/pull_bucket.py`) fetches a task's input files once per rollout, as a
subprocess of this service.

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
`vendor/task-index.json.gz`, `vendor/pull_bucket.py` and `../rle_deprecated`'s vendored
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
cd agent
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
PULL_BUCKET_PATH=vendor/pull_bucket.py \
MODEL_URL=https://api.openai.com/v1 \
MODEL_API_KEY=$OPENAI_API_KEY \
MODEL_ID=gpt-5-mini \
  python main.py
```

```bash
curl -s localhost:8088/readiness

curl -s -X POST localhost:8088/v1/responses \
  -H 'content-type: application/json' \
  -H 'x-client-rle-rollout-id: local-1' \
  -H "x-client-rle-model-endpoint: https://api.openai.com/v1" \
  -H "x-client-rle-model-api-key: $OPENAI_API_KEY" \
  -d '{
        "agent_session_id": "local-1",
        "input": [{"type": "message", "role": "user",
                   "content": [{"type": "input_text",
                                "text": "{\"task_index\": 0}"}]}],
        "background": false, "stream": false, "store": false
      }' | jq -r '.output[-1].content[0].text'
```

The call blocks for the length of the rollout, which is the contract -- there
is nothing to poll. `x-client-rle-model-endpoint` is RLE's capture proxy in a
real rollout, so a local smoke test is the one case where it is a real model
endpoint. `MODEL_URL` is only the fallback for a rollout that names none.
`MODEL_ID` is probed from the endpoint when unset and the endpoint serves
exactly one model.

Omitting `x-client-rle-model-endpoint` is rejected rather than quietly answered:
without the capture proxy nothing would be recorded, so the rollout would look
like it ran and train on nothing.

## Tests

```bash
cd agent
python -m pytest tests -q
```

No extra dependencies beyond `requirements.txt` and `pytest`; the tests drive
`opencode_direct` with the `opencode` invocation stubbed, so they need neither a
model endpoint nor the task suite. `test_hosted_agent_contract.py` covers the
half of the contract that is invisible at the type level -- the header names and
the fact that `agent_input` arrives as text.

## Build and deploy

```bash
docker build -t data-code-agent:local .
docker run --rm -p 8088:8088 \
  -e MODEL_URL=https://api.openai.com/v1 \
  -e MODEL_API_KEY=$OPENAI_API_KEY \
  -e MODEL_ID=gpt-5-mini \
  data-code-agent:local
```

Push this image to the registry your Foundry project can pull from, register it
as a Hosted Agent, and put that agent's name and version in `rle_deprecated/rle.toml` as
`agentName` / `agentVersion`. There is no base URL to register: RLE resolves the
agent from the project and calls
`{project}/agents/{agentName}/endpoint/protocols/openai/responses` itself.

Size the container for concurrency: each rollout holds its own copy of the task's
input files on local disk and runs its own `opencode` process.
