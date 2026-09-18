# RLE samples

RLE has two implementation types: `Gym` and `Harness`. Every environment
carries a host-agnostic `rle.toml` manifest declaring its immutable identity
and callable interface:

```toml
[rle]
name = "code_rl"
version = "1.0.0"
type = "Gym"
subtype = "OpenEnv"
```

The `type` and `subtype` values match the RLE control plane. The manifest
intentionally does not contain Foundry project endpoints, registry
locations, service IDs, or credentials.

## `envs/` — Gym/OpenEnv samples

[`envs/`](./envs) holds `Gym`/`OpenEnv` samples, where the caller drives the
environment through `reset`/`step`. These are copied directly by
`azd ai rle init <sample-name>`.

## `harness-examples/` — Harness samples (HostedAgent, BYOH)

[`harness-examples/`](./harness-examples) holds worked reference examples
for the `Harness` type, where RLE wraps a production agent that owns its own
loop: [`harness-examples/hosted-agent/`](./harness-examples/hosted-agent)
for agents running as a Foundry Hosted Agent version, and
[`harness-examples/byoh/`](./harness-examples/byoh) for bring-your-own
harnesses deployed anywhere and registered by base URL. Each includes both
the agent/harness code and the paired RLE code (task setup, mock tools,
grader), plus a README covering deploy → wire → publish. Because the agent
side is always specific to your own harness, `azd ai rle init --type Harness`
scaffolds only the RLE side for you — these folders show how to build and
wire the other half.

When a sample is initialized through `azd ai rle init <folder-name>`, the CLI
updates `rle.name` to the target folder name. Before publishing another
version, update `rle.version` to the direct next major, minor, or patch
release.

## `code_rl` compact dataset maintenance

`examples/gym/openenv/code_rl` ships its 900-training-task and 100-validation-task snapshot
under `data/`, so normal Docker builds and runtime instances do not download
Hugging Face data. The snapshot generator is intentionally outside the sample
at `tools/generate_code_rl_compact_dataset.py`; it is not copied by
`azd ai rle init`.

Maintainers can intentionally refresh the revision-pinned snapshot and its
checksums from the repository root:

```bash
python tools/generate_code_rl_compact_dataset.py
```

The generator calls the Hugging Face datasets server, preserves no more than
three complete tests and 12 KiB of test input/output per task, and updates
`examples/gym/openenv/code_rl/data/source.json` with the selected source and file hashes.
