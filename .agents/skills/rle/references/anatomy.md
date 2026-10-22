# Sample anatomy and SDK installation

## Current samples

Paths below are relative to the samples repository, not to the skill folder.
Scaffolded projects need not retain that outer `examples/` hierarchy.

| Path | Harness type / protocol | Demonstrates |
| --- | --- | --- |
| `examples/gym/openenv/math_rl` | `gym` | Seed/split selection, optional typed equivalence helper, boxed-answer grading |
| `examples/gym/openenv/code_rl` | `gym` | Typed hidden-test helper, fenced Python submission and safe execution |
| `examples/harness/byoh/data_code_agent` | `byoh` / `rle-invocation` | Async agent invocation, MCP disclosure evidence and independent answer grading |
| `examples/harness/hosted-agent/competitive_intelligence_agent` | `byoh` / `responses` | Responses headers, production-shaped simulated tools and rubric grading |

Catalogs control sample visibility; their folder names are not TOML enum values.
Do not use `_internal/`
maintainer directories as scaffolding templates.

## Shared environment layout

```text
<environment build context>/
  rle.toml
  Dockerfile
  install_rle_sdk.py
  requirements.txt
  README.md
  server/
    app.py
    models.py
    <environment>.py
    <dataset/task helpers and grader>
```

Gym samples also have baked `env_data/`, selector-only `job_data/` and tests.
Harness samples separate `agent/`, `rle/` and `job_data/`; **`rle/` is the
environment Docker build context**, not the parent sample or agent folder.
Publish the environment and deploy the agent independently.

Dataset strategy is task-specific:

- Gym math/code load baked train/validation snapshots without runtime downloads.
- BYOH data-code pins a dataset split/index; the agent separately provisions CSVs.
- Hosted competitive intelligence receives the complete task in reset and builds
  a deterministic simulated world. Do not bake its answer-key taskset into an
  agent-accessible filesystem just to imitate a Gym sample.

Keep data provenance/licensing with redistributed data. Keep answer keys private
to grading, regardless of whether tasks are baked or supplied in reset.

## Install the RLE-enabled SDK, not a random same-version wheel

Each environment carries its own identical `install_rle_sdk.py`. The installer
reads the public RLE SDK registry, selects a wheel, verifies SHA-256, then installs
it with the sample requirements. No private SDK fork or GitHub credentials are
needed. Do not replace the installer with a `git+` dependency or silently fall
back to an ordinary wheel that lacks `azure.ai.projects.rle.environments`.

From the environment's build context, in an isolated Python environment:

```powershell
python install_rle_sdk.py --requirements requirements.txt
```

Dockerfiles set `ENV AZURE_AI_PROJECTS_VERSION=latest`. To reproduce an exact
SDK build, edit that line to a registry-listed version before rebuilding.
For local Python, the installer accepts `--version <version>`. Runtime
environment variables do not replace the SDK baked into the image.

The samples pin OpenEnv `0.6.0`. Preserve compatible SDK/OpenEnv pins and the
existing app entrypoint when editing; do not independently upgrade protocol
dependencies without a session-routing regression test.

Use **separate virtual environments and images** for the RLE and Harness agent.
In the verified public distributions, RLE SDK `2.7.0b1` requires
`openai>=3.0.0`, while the reference HostedAgent pins `openai==2.54.0` with
`azure-ai-agentserver-responses==2.1.0` and core `2.1.0`. Installing both sets
into one venv can produce a successful pip exit with a dependency-conflict
warning. Run `python -m pip check` in each runtime; a successful install or
import alone is not proof the dependencies are consistent.
Preserve the agent's tested pins separately from the grading SDK.

Serve on `0.0.0.0` and the expected container port (samples use 8000), with one
Uvicorn worker. Keep default capacity at one. Raising capacity requires proven
per-session isolation and `SUPPORTS_CONCURRENT_SESSIONS = True`; increasing
workers does not provide shared session state.
