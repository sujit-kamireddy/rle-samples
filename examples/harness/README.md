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

Each subtype directory holds one directory per named sample — today
`hosted-agent/` holds a single `code_repair/`, and `byoh/` holds both
`code_repair/` and [`data_code_agent/`](./byoh/data_code_agent) — alongside a
`catalog.toml` listing which samples `azd ai rle init` offers. Pick one with
`--sample <name>`; when a subtype has only one visible sample the CLI
scaffolds it without prompting.

Each sample ships two independent pieces that get deployed and published
separately:

- `agent/` — the harness/agent code. For `BYOH` this is a small service you
  deploy anywhere and register with RLE by base URL. For `HostedAgent` this is
  the agent code that runs as a Foundry Hosted Agent version.
- `rle/` — the RLE side: the container `azd ai rle init --type Harness`
  scaffolds, filled in with a concrete task setup, mock tools, and grader
  (`rle/server/env.py`), plus `rle/rle.toml` and `rle/Dockerfile`.

[`byoh/data_code_agent`](./byoh/data_code_agent) wraps Hugging Face's
`FineEnvs/data-agent` collection — 5,000 data-analysis tasks over real CSVs —
and grades a second axis on top of answer correctness: whether the agent
correctly judged the data it analysed to be sensitive, and filed a disclosure.
Its `agent/` runs the agent loop itself (`opencode`, in the same container) and
its `rle/` vendors the dataset's own grader, so the two pieces are
self-contained — see its own README for how they fit
together.

Both `code_repair` samples wrap the same real SWE-bench-Lite instance
(`psf__requests-3362`, a real `psf/requests` issue, pinned repo checkout, and
real regression test — see either sample's README for details), and are
intentionally near-identical at the code level: `rle/` is byte-for-byte the
same in both, since RLE does not care which harness subtype invokes it, and
each `agent/`'s core `run_agent_loop` is the same agent loop. `BYOH` differs
only by adding an HTTP invocation endpoint for RLE to call; `HostedAgent`
instead reads rollout context from request headers on its existing Responses
API.

See each sample's own README for the deploy → wire → register → publish
flow.

For a PowerShell walkthrough of the published data-code BYOH harness, training
job, and Loom dashboard, see [`RLE-BYOH-training.md`](./RLE-BYOH-training.md).
