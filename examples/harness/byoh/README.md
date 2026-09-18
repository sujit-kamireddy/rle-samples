# BYOH (Bring Your Own Harness) example: code-repair agent

This is a real SWE-bench-Lite instance, not synthetic data: `psf/requests`
issue `psf__requests-3362` ("`iter_content(decode_unicode=True)` can return
bytes"), pinned at base commit `36453b95b1`. See
[`rle/fixtures/instance.json`](./rle/fixtures/instance.json) for the exact
metadata (source: `princeton-nlp/SWE-bench_Lite`). The harness must read the
real GitHub issue, patch the real `requests` checkout, and open a pull
request; RLE grades it by running the real regression test
(`tests/test_requests.py::TestRequests::test_response_decode_unicode`) that
the real upstream fix made pass.

Two independent pieces make up this `Harness`/`BYOH` RLE:

- [`agent/`](./agent) — the production agent (harness), deployable anywhere
  you can reach over HTTPS. Unmodified except for how it builds its model
  client and tools (see "Wire the harness" below).
- [`rle/`](./rle) — the RLE container: a real, pinned `requests` checkout
  baked into the image, the hidden regression test patch, mock
  `workspace.apply_patch`/`github.create_pull_request` tools, and a grader
  that runs the real test. This is the piece
  `azd ai rle init --type Harness --subtype BYOH` scaffolds for you; this
  folder is that scaffold filled in for this instance.

This example and [`../hosted-agent`](../hosted-agent) are intentionally
close to identical: `rle/` is byte-for-byte the same environment (RLE does
not care which harness subtype invokes it), and `agent/`'s `run_agent_loop`
is the same agent loop in both. Only the invocation transport differs — this
one adds an HTTP invocation endpoint; [`hosted-agent`](../hosted-agent)
instead reads request headers on an existing Responses API.

## 1. Deploy the agent (harness) anywhere

`agent/` is a plain FastAPI service with no Foundry or Azure dependency. Run
it wherever you already run the rest of your stack — a container in your own
cluster, an App Service, a VM, or `docker run` locally:

```bash
cd examples/harness/byoh/agent
docker build -t code-repair-agent:latest .
docker run --rm -p 8080:8080 code-repair-agent:latest
```

Put it behind HTTPS (a reverse proxy, App Gateway, or your platform's TLS
termination). Since RLE never attaches caller, workspace, or identity
headers to its invocation requests, protect the endpoint yourself — for
example, require a shared secret header and validate it in `agent/app.py`
before dispatching to the agent loop.

## 2. Wire the harness

Per the harness contract, only the construction of the agent's model client
and tools becomes RLE-aware — its native agent loop is unchanged:

```python
model = create_model_client(request.rollout_context)   # -> capture proxy, not prod model endpoint
# tool calls route to request.rollout_context.sandbox_tools_endpoint instead of prod tools
```

See `agent/app.py`'s `create_model_client`, `call_tool`, and
`run_agent_loop` for the full wiring, and `/invoke`/`/health` for the
invocation contract RLE calls:

```text
POST <base-url>/invoke
{
  "rollout_id": "...",
  "agent_input": {"...": "agent-specific input"},
  "rollout_context": {
    "capture_proxy_endpoint": "...",
    "capture_proxy_session_key": "...",
    "sandbox_tools_endpoint": "...",
    "sandbox_tools_bearer_token": "..."
  }
}
```

The agent must respond with `{"output_text": "..."}`.

## 3. Author and iterate the RLE side

```bash
cd examples/harness/byoh/rle
azd ai rle run
```

`rle/server/env.py`'s `reset()` copies the pinned `requests` checkout baked
into the image, applies the hidden test patch, and returns the real issue
text. `/tools/workspace.apply_patch` and `/tools/github.create_pull_request`
mock the harness's real tools; `/grade` runs the real regression test and
checks for a recorded pull request. Adapt these for your own repo, mocks,
and reward function the same way you'd iterate on a Gym sample.

## 4. Register the base URL and publish

Once your deployed agent URL is stable, scaffold (or update) the manifest
with it and publish a version:

```bash
azd ai rle init code_repair_byoh \
  --type Harness --subtype BYOH \
  --base-url https://<your-deployed-agent-host>/invoke \
  --no-prompt

cd code_repair_byoh
azd ai rle publish
```

Training, evaluation, and optimization jobs against this RLE will now invoke
your deployed agent for every rollout.
