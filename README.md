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

## `.agents/skills/` and `.claude/skills/` — the authoring skills

The skills carry the RLE authoring contract: the server surface, the action
vocabulary behind `GET /schema`, the manifest fields, grading and step budget,
and the failure modes worth recognizing. A coding agent working in a project
that has them can author or extend an environment without being told the
contract first.

They ship twice because no single directory reaches every agent:

| Agent          | Reads                                              |
| -------------- | -------------------------------------------------- |
| Claude Code    | `.claude/skills/` only                             |
| OpenAI Codex   | `.agents/skills/`                                  |
| GitHub Copilot | `.github/skills/`, `.claude/skills/`, `.agents/skills/` |

The two trees must stay byte-identical. `.agents/skills/` is the source of
truth and `.github/workflows/skills-parity.yml` fails the build if they
diverge, so after editing a skill mirror it across:

```bash
rm -rf .claude/skills && cp -R .agents/skills .claude/skills
```

`azd ai rle init` copies both trees into a new environment, and
`azd ai rle skill install` refreshes them in place from this repository.

## `examples/gym/openenv/` — Gym/OpenEnv samples

[`examples/gym/openenv/`](./examples/gym/openenv) holds `Gym`/`OpenEnv`
samples, where the caller drives the environment through `reset`/`step`.
These are copied directly by `azd ai rle init <sample-name>`.

[`examples/gym/openenv/catalog.toml`](./examples/gym/openenv/catalog.toml)
controls which sample directories `azd ai rle init` offers. A new sample
directory is visible automatically; add an entry with `visible = false` to
stage a sample before announcing it.

## `examples/harness/` — Harness samples (HostedAgent, BYOH)

[`examples/harness/`](./examples/harness) holds worked reference examples
for the `Harness` type, where RLE wraps a production agent that owns its own
loop: [`examples/harness/hosted-agent/`](./examples/harness/hosted-agent)
for agents running as a Foundry Hosted Agent version, and
[`examples/harness/byoh/`](./examples/harness/byoh) for bring-your-own
harnesses deployed anywhere and registered by base URL. Each subtype holds
one directory per sample — today
[`code_repair`](./examples/harness/byoh/code_repair) under both, plus
[`spreadsheet_query_agent`](./examples/harness/byoh/spreadsheet_query_agent) under `byoh` — and each
sample includes the agent/harness code and the paired RLE code (task setup,
mock tools, grader), plus a README covering deploy → wire → publish. Because
the agent side is always specific to your own harness,
`azd ai rle init --type Harness` scaffolds only the RLE side for you — these
folders show how to build and wire the other half.

Each subtype has its own
[`catalog.toml`](./examples/harness/byoh/catalog.toml), with the same
visibility rules as the Gym/OpenEnv one. Pick a sample with
`azd ai rle init --sample <name>`; with only one visible sample, that subtype
scaffolds it without prompting.

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
