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

## RLE SDK installation

All four RLE container samples install `azure-ai-projects[rle]` wheels from the
[public SDK manifest](https://raw.githubusercontent.com/Azure/azure-sdk-for-python/rle-releases/sdk/ai/azure-ai-projects/rle/registry.json),
hosted by [Azure/azure-sdk-for-python#49488](https://github.com/Azure/azure-sdk-for-python/pull/49488).
During `azd ai rle publish`, the environment image is built and its Dockerfile
installs the SDK inside that container. The SDK is a container dependency, not
a dependency of the RLE CLI executable; no local SDK installation is needed
to publish an environment.
No source fork access, Git installation, or GitHub credentials are required.
Each sample includes `install_rle_sdk.py` in its own Docker build context.
The Dockerfile copies the installer locally; it is not downloaded during builds.
The installer selects the requested version and verifies its SHA-256 before
running pip. Docker uses the wheel already fetched beside the registry, without
downloading it again. Local development using the remote registry downloads the
selected wheel instead. Approved version paths must not be overwritten, and a
checksum mismatch always fails installation.
For now, the four sample copies and the repository-root
[`tools/install_rle_sdk.py`](./tools/install_rle_sdk.py) are kept identical.
Update all five copies together. Each sample retains its own build context
and includes its installer when scaffolded by the CLI.

Each sample's Dockerfile sets `ENV AZURE_AI_PROJECTS_VERSION=latest` before
SDK installation. To pin a version, edit that line in your local Dockerfile to
an exact version listed in the manifest. A missing version fails; it never falls
back. No SDK version setting, environment forwarding, or SDK installation is
needed on the `azd` CLI side.
The Dockerfiles use Git-source `ADD` scoped to the public distribution
subdirectory. A builder supporting Git-source `ADD`
(such as current Docker with BuildKit)
resolves the branch commit on each build, so new releases invalidate the SDK
installation layer without relying on a CDN-cached raw manifest or GitHub REST
API rate limits. Git is not installed in the runtime image.

The default Dockerfile setting is:

```dockerfile
ENV AZURE_AI_PROJECTS_VERSION=latest
```

For reproducible builds, change it locally before rebuilding:

```dockerfile
ENV AZURE_AI_PROJECTS_VERSION=2.7.0b1
```

Only for local server development outside Docker, run from this repository's
root, choosing the sample requirements:

```powershell
python tools\install_rle_sdk.py --requirements examples\gym\openenv\math_rl\requirements.txt
```

Preview releases use beta versions such as `2.7.0b1` and `2.7.1b1`.
Only versions listed in the public manifest are available for pinning.

The local Python installer also accepts `--version <version>`. Docker builds use
only the environment value defined in the Dockerfile, defaulting to latest.
Runtime container environment variables cannot change the SDK baked into an
image. The public SDK PR contains only the manifest and wheels; release tooling
and release validation stay in the SDK fork, while consumer installer copies
live in this repository.
[`tools/test_rle_sdk_distribution.py`](./tools/test_rle_sdk_distribution.py)
checks that all sample installers match the reference copy in `tools`, as well
as version selection, artifact verification
and pip installation.

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
[`data_code_agent`](./examples/harness/byoh/data_code_agent) under both — and each
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
