# Harness RLE examples

The samples under `envs/` are `Gym`/`OpenEnv` environments: the caller (a
training, evaluation, or optimization job) owns the agent loop and drives the
environment through `reset`/`step`. They are copied directly by
`azd ai rle init <sample-name>`.

`harness-examples/` covers the other RLE type: `Harness`. A harness RLE
wraps a *production agent* that owns its own loop. RLE does not call
`reset`/`step` on it; instead it invokes the harness once per rollout and the
harness reports back a final response. `azd ai rle init --type Harness`
scaffolds the RLE-side container for you (task setup, mock tools, grader),
but the harness side is always specific to your agent, so these folders are
worked reference examples to copy from and adapt rather than something the
CLI templates automatically.

Two harness subtypes exist:

| Subtype | Where the agent runs | Example |
| --- | --- | --- |
| `HostedAgent` | A Foundry Hosted Agent version | [`hosted-agent/`](./hosted-agent) |
| `BYOH` (Bring Your Own Harness) | Anywhere you register a base URL | [`byoh/`](./byoh) |

Each example ships two independent pieces that get deployed and published
separately:

- `agent/` — the harness/agent code. For `BYOH` this is a small service you
  deploy anywhere and register with RLE by base URL. For `HostedAgent` this is
  the agent code that runs as a Foundry Hosted Agent version.
- `rle/` — the RLE side: the container `azd ai rle init --type Harness`
  scaffolds, filled in with a concrete task setup, mock tools, and grader
  (`rle/server/env.py`), plus `rle/rle.toml` and `rle/Dockerfile`.

Both examples wrap the same real SWE-bench-Lite instance
(`psf__requests-3362`, a real `psf/requests` issue, pinned repo checkout, and
real regression test — see either example's README for details), and are
intentionally near-identical at the code level: `rle/` is byte-for-byte the
same in both, since RLE does not care which harness subtype invokes it, and
each `agent/`'s core `run_agent_loop` is the same agent loop. `BYOH` differs
only by adding an HTTP invocation endpoint for RLE to call; `HostedAgent`
instead reads rollout context from request headers on its existing Responses
API.

See each example's own README for the deploy → wire → register → publish
flow.
