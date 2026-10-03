# Harness RLE examples

The samples under `examples/gym/openenv/` are `Gym`/`OpenEnv` environments: the caller (a
training, evaluation, or optimization job) owns the agent loop and drives the
environment through `reset`/`step`. They are copied directly by
`azd ai rle init <sample-name>`.

`examples/harness/` covers the other RLE type: `Harness`. A harness RLE
wraps a *production agent* that owns its own loop. RLE does not call
`reset`/`step` on it; instead it invokes the harness once per rollout and the
harness reports back a final response. `azd ai rle init --type Harness`
scaffolds the RLE-side container for you (task setup, mock tools, grader),
but the harness side is always specific to your agent, so these folders are
worked reference examples to copy from and adapt rather than something the
CLI templates automatically.

Two harness subtypes exist:

| Subtype | Where the agent runs | Samples |
| --- | --- | --- |
| `HostedAgent` | A Foundry Hosted Agent version | [`hosted-agent/`](./hosted-agent) |
| `BYOH` (Bring Your Own Harness) | Anywhere you register a base URL | [`byoh/`](./byoh) |

Each subtype directory holds one directory per named sample — today both
hold a single [`data_code_agent/`](./byoh/data_code_agent) — alongside a
`catalog.toml` listing which samples `azd ai rle init` offers. Pick one with
`--sample <name>`, or let the CLI prompt: it always does, even when only one
sample is visible.

Each sample ships two independent pieces that get deployed and published
separately:

- `agent/` — the harness/agent code. For `BYOH` this is a small service you
  deploy anywhere and register with RLE by base URL. For `HostedAgent` this is
  the agent code that runs as a Foundry Hosted Agent version.
- `rle/` — the RLE side: the environment container, with its concrete task
  setup, tool surface and grader, plus its `rle.toml` and `Dockerfile`.

`rle/` implements the OpenEnv protocol, which the RLE environment contract is
converging on across every type and subtype. The environment it describes is
unchanged: the harness still drives your `agent/`, the agent still calls the
environment's tools, and the environment still owns grading. Only the wire
format between RLE and the environment differs.

Each sample also keeps `rle_deprecated/`, the original hand-rolled
`azd ai rle init --type Harness` scaffold that serves `/reset`,
`/tools/*` and `/grade` from `rle_deprecated/server/env.py`. It is the
published, end-to-end validated path today, and the sample READMEs still
walk through it. It is retained so the two can be graded side by side, and
goes once `rle/` is validated against Foundry.

[`byoh/data_code_agent`](./byoh/data_code_agent) wraps Hugging Face's
`FineEnvs/data-agent` collection — 5,000 data-analysis tasks over real CSVs —
and grades a second axis on top of answer correctness: whether the agent
correctly judged the data it analysed to be sensitive, and filed a disclosure.
Its `agent/` runs the agent loop itself (`opencode`, in the same container) and
its `rle/` vendors the dataset's own grader, so the two pieces are
self-contained — see its own README for how they fit
together.

The two `data_code_agent` samples are the same environment under both
subtypes, and are intentionally near-identical at the code level: `rle/` is
the same in both, since RLE does not care which harness subtype invokes it,
and `agent/opencode_direct.py` — the agent loop itself — is shared verbatim.
Only the entrypoint differs. `BYOH` adds an HTTP invocation endpoint RLE
POSTs to and then polls, and reads the rollout context out of the request
body; `HostedAgent` serves the Responses API and reads the same context off
`x-client-rle-*` request headers, answering synchronously.

See each sample's own README for the deploy → wire → register → publish
flow.
