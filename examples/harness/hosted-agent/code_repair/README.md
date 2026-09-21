# HostedAgent example: code-repair agent

Same real SWE-bench-Lite instance as [`../../byoh/code_repair`](../../byoh/code_repair): `psf/requests`
issue `psf__requests-3362` ("`iter_content(decode_unicode=True)` can return
bytes"), pinned at base commit `36453b95b1`. See
[`rle/fixtures/instance.json`](./rle/fixtures/instance.json) for the exact
metadata. The harness reads the real GitHub issue, patches the real
`requests` checkout, and opens a pull request; RLE grades it by running the
real regression test that the real upstream fix made pass.

Two independent pieces make up this `Harness`/`HostedAgent` RLE:

- [`agent/`](./agent) — the agent code that runs as a Foundry Hosted Agent
  version. It reads the RLE-supplied request headers to become rollout-aware;
  otherwise it behaves exactly as it does in production.
- [`rle/`](./rle) — the RLE container: byte-for-byte the same environment as
  [`../../byoh/code_repair/rle`](../../byoh/code_repair/rle) — a real, pinned `requests` checkout baked
  into the image, the hidden regression test patch, mock
  `workspace.apply_patch`/`github.create_pull_request` tools, and a grader
  that runs the real test. RLE does not care which harness subtype invokes
  it, so this piece never changes between subtypes. Running
  `azd ai rle init --type Harness --subtype HostedAgent --harness-source sample`
  copies this exact `agent/` + `rle/` pair as your starting point (the
  `--harness-source existing` default instead scaffolds a generic,
  empty placeholder for a harness you already built and deployed yourself).

`agent/main.py`'s `run_agent_loop` is the same agent loop as
[`../../byoh/code_repair/agent/app.py`](../../byoh/code_repair/agent/app.py)'s. Only the invocation
transport differs — this one reads runtime context from request headers on
an existing Responses API; BYOH instead adds an HTTP invocation endpoint.

## 1. Deploy the agent as a Hosted Agent version

Publish `agent/` as a Foundry Hosted Agent version through your normal Hosted
Agent deployment path (`azd ai agent ...` or the Foundry portal). Note the
agent's name and the specific version you publish — RLE binds a rollout to
that exact version, so `agentName`/`agentVersion` in `rle.toml` must be
resolvable, callable Hosted Agent identifiers (not `$default`).

## 2. Wire the harness

For each rollout, RLE creates a session for your Hosted Agent version and
calls its Responses API with four extra headers:

```text
x-client-rle-rollout-id
x-client-rle-model-endpoint          # capture proxy, replaces your prod model endpoint
x-client-rle-model-api-key           # capture proxy session key
x-client-rle-sandbox-tools-endpoint  # sandbox tool routes for this rollout
```

`agent/main.py`'s `create_model_client` and `call_tool` read these headers
when present and fall back to the agent's normal production model endpoint
and tools when absent (ordinary, non-rollout traffic). Adapt
`handle_responses_request` to however your Hosted Agent hosting framework
exposes incoming request headers to your handler.

## 3. Author and iterate the RLE side

```bash
cd examples/harness/hosted-agent/code_repair/rle
azd ai rle run
```

Identical to the [BYOH example](../../byoh/code_repair)'s RLE side — adapt
`rle/server/env.py`'s `reset()`, mock tool routes, and `/grade` for your own
repo, mocks, and reward function.

## 4. Register the agent version and publish

```bash
azd ai rle init code_repair_hosted_agent \
  --type Harness --subtype HostedAgent \
  --harness-source sample \
  --agent-name code-repair-agent --agent-version 1 \
  --no-prompt

cd code_repair_hosted_agent/rle
azd ai rle publish
```

Training, evaluation, and optimization jobs against this RLE will now invoke
your Hosted Agent version for every rollout, with one Hosted Agent version
able to serve many rollouts and (during optimization) many candidate
configurations resolved per rollout.

## 5. Run a rollout

Run this from `examples/harness/hosted-agent/code_repair/rle` (where this sample's
`rle.toml` lives) against whichever published name/version you registered.
`rollout` reads `rle.name`/`rle.version` from `rle.toml`, provisions a real
Loom training session and sampler checkpoint for `--model`, calls Execute
Rollout (which forwards `--agent-input` to your Hosted Agent), and prints the
resulting reward. This sample's `reset()` ignores `--task`, so `{}` is
enough, but the agent requires `agent_input.issue`; pull the real SWE-bench
issue text out of the bundled fixture so the command stays copy/paste-ready:

```bash
cd examples/harness/hosted-agent/code_repair/rle
AGENT_INPUT=$(python3 -c "import json; print(json.dumps({'issue': json.load(open('fixtures/instance.json'))['problem_statement']}))")
azd ai rle rollout --model Qwen/Qwen3-32B --task '{}' --agent-input "$AGENT_INPUT"
```
