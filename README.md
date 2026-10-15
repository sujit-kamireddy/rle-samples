# RLE samples

RLE uses `gym` when RLE drives the model loop, or `byoh` when an external agent
drives its own loop. Every environment
carries a host-agnostic `rle.toml` manifest declaring its immutable identity
and callable interface:

```toml
[rle]
name = "code_rl"
version = "1.0.0"

[harness]
type = "gym"
```

For `byoh`, `[harness]` also binds an HTTPS `endpoint` and invocation `protocol`:
`rle-invocation` for async start/poll/cancel, or `responses` for a same-project
Foundry agent URL ending in `/agents/<name>`. Removed subtype/name/version
binding keys are not accepted. The manifest excludes registry locations,
service IDs and credentials; a Responses harness endpoint includes its project.
Task format defaults to `json`; Harbor tasks and harness packages are reserved,
not yet runnable.

## RLE SDK installation

All four RLE container samples install `azure-ai-projects[rle]` wheels from the
[public SDK manifest](https://raw.githubusercontent.com/Azure/azure-sdk-for-python/rle-releases/sdk/ai/azure-ai-projects/rle/registry.json),
hosted by [Azure/azure-sdk-for-python#49488](https://github.com/Azure/azure-sdk-for-python/pull/49488).
The SDK is installed inside the container during `azd ai rle publish`; no host
SDK installation, SDK source-fork access, or GitHub credentials are required.
Each sample includes `install_rle_sdk.py` in its own Docker build context.
The Dockerfile copies the installer locally; it is not downloaded during builds.
The installer selects the requested version and verifies its SHA-256 before
running pip. Docker fetches the registry JSON over HTTPS, then the installer
downloads only the selected wheel from its registry URL. Local development uses
the same download flow. Approved version paths must not be overwritten, and a
checksum mismatch always fails installation.
Keep the four installer copies identical when updating them. Each sample
retains its own build context and includes its installer when scaffolded by the CLI.

Each sample's Dockerfile sets `ENV AZURE_AI_PROJECTS_VERSION=latest` before
SDK installation. To pin a version, edit that line in your local Dockerfile to
an exact version listed in the manifest. A missing version fails; it never falls
back. No SDK setting is needed on the `azd` CLI side.
The Dockerfiles use HTTP `ADD` for the public registry JSON. When its content
changes, Docker invalidates the SDK installation layer. GitHub's CDN can briefly
serve an older registry; this delay is accepted. No Git fetch is needed.

The default Dockerfile setting is:

```dockerfile
ENV AZURE_AI_PROJECTS_VERSION=latest
```

To pin the SDK version, change that line locally before rebuilding:

```dockerfile
ENV AZURE_AI_PROJECTS_VERSION=2.7.0b1
```

For local server development outside Docker, run from the folder containing
the sample's Dockerfile:

```powershell
python install_rle_sdk.py --requirements requirements.txt
```

The local installer also accepts `--version <version>` for a version listed in
the manifest. Docker builds use the environment value defined in the Dockerfile.
Runtime container environment variables cannot change the SDK baked into an
image.
[`tools/test_rle_sdk_distribution.py`](./tools/test_rle_sdk_distribution.py)
checks that all sample installers match, along with version selection, checksum
verification, and pip installation.

## `.agents/skills/` and `.claude/skills/` — the authoring skills

One unified [`rle` skill](./.agents/skills/rle/SKILL.md) covers creating, editing
and migrating all three supported paths: `gym`, `byoh/rle-invocation` and
`byoh/responses` (HostedAgent). It explains the shared SDK/OpenEnv
environment contract, [`rle.toml`](./.agents/skills/rle/references/manifest.md),
[MCP sessions and tools](./.agents/skills/rle/references/mcp.md),
[agent invocation](./.agents/skills/rle/references/invocation.md), grading,
local verification and publish/rollout workflows. References load on demand
through normal Markdown links.

They ship twice because no single directory reaches every agent:

| Agent          | Reads                                              |
| -------------- | -------------------------------------------------- |
| Claude Code    | `.claude/skills/` only                             |
| OpenAI Codex   | `.agents/skills/`                                  |
| GitHub Copilot | `.github/skills/`, `.claude/skills/`, `.agents/skills/` |

The two trees must stay byte-identical. `.agents/skills/` is the source of
truth and `.github/workflows/skills-parity.yml` fails the build if they
diverge, so after editing a skill mirror it across:

```powershell
Copy-Item -Path .agents\skills\rle -Destination .claude\skills -Recurse -Force
python -m unittest discover -s tools -p test_rle_skill.py
```

`azd ai rle init` copies both trees into a new environment, and
`azd ai rle skill install` refreshes them in place from this repository.
Run it from an initialized RLE build context with a valid manifest. It reads
the published `main` branch, not uncommitted local skill edits. See the
[isolated authoring workflow](./.agents/skills/rle/references/workflow.md)
for testing without changing a shared azd profile.
Check installed `azd ai rle --help` first: a compatible azd host needs the
`azure.ai.rle` extension installed from the intended source. Core commands no
longer need `AZD_AI_RLE_ENABLE`; Harness init and staged training commands use
`AZD_AI_RLE_ENABLE_ALL`. See the
[verified remote source/release baseline](./.agents/skills/rle/references/upstream-contracts.md)
for the CLI, SDK and service revisions used to refresh the skill.

### Try authoring with an agent

Open this repo or an initialized RLE project in your coding agent, then ask:

- "Use the rle skill to create a Gym/OpenEnv environment with a typed MCP
  helper and test correct, wrong and malformed answers."
- "Use the rle skill to wrap my existing agent as BYOH. Preserve its loop and
  implement async start/poll/cancel plus session-scoped MCP tools."
- "Use the rle skill to edit this HostedAgent RLE's manifest, tool schema and
  grader, then verify task identity and session isolation."

The skill guides local smoke tests before a managed rollout and reports any
missing runtime, CLI or cloud credentials instead of claiming a publish worked.
See the [step-by-step workflow](./.agents/skills/rle/references/workflow.md).
The dependency-free skill checks validate links, examples, discovery metadata
and byte parity; CI runs them alongside the existing parity checks.

## `examples/gym/openenv/` — Gym/OpenEnv samples

[`examples/gym/openenv/`](./examples/gym/openenv) holds `Gym`/`OpenEnv`
samples, where the caller drives the environment through `reset`/`step`.
These are copied directly by `azd ai rle init <sample-name>`.

Both Gym samples use the SDK `RLEnvironment` MCP contract:

- [`code_rl`](./examples/gym/openenv/code_rl) registers the typed
  `check_solution` tool and grades the final fenced Python submission.
- [`math_rl`](./examples/gym/openenv/math_rl) registers the typed
  `check_equivalence` helper and grades the final boxed answer.

Both run under RLE's MCP environment protocol (the only one), and carry reset,
MCP tool calls, and terminal `GradeAction` grading over one persistent OpenEnv
session.

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
[`data_code_agent`](./examples/harness/byoh/data_code_agent) for BYOH and
[`competitive_intelligence_agent`](./examples/harness/hosted-agent/competitive_intelligence_agent)
for HostedAgent — and each
sample includes the agent/harness code and the paired RLE code (task setup,
mock tools, grader), plus a README covering deploy → wire → publish. Because
the agent side is always specific to your own harness,
`azd ai rle init --type Harness` scaffolds only the RLE side for you — these
folders show how to build and wire the other half.

Each subtype has its own
[`catalog.toml`](./examples/harness/byoh/catalog.toml), with the same
visibility rules as the Gym/OpenEnv one. Pick a sample with
`azd ai rle init --sample <name>`, or let the CLI prompt — it always does,
even when only one sample is visible.

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
