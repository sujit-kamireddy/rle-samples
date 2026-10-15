# Verified upstream contract baseline

Checked on **2026-10-15** from remote repositories, not dirty local files.
Remote Git refs were resolved/fetched and source was read at the exact commits
below. No existing branch was checked out, merged, rebased or reset.

## Source revisions

| Component | Repository / ref | Verified commit |
| --- | --- | --- |
| RLE CLI development source | `sujit-kamireddy/azure-dev`, `main` | `f015fbfb25fbf90545afa128cd8157146ee7a6ac` |
| RLE SDK source | `sujit-kamireddy/azure-sdk-for-python`, `main` | `01090d46d354937c855c875e2602908cf0fc6e60` |
| Vienna RLE service source | Vienna repository, `master` | `bb36536014db5628f421f6086374ae792e841b4d` |

The CLI source still identifies itself as `0.8.30-preview`, but now implements
the Harness schema redesign also merged in samples PR #83. Version labels
alone do not identify the manifest contract. A local feature worktree is not
a substitute for checking these remote refs.

Current authoring uses lowercase `gym`/`byoh`, `endpoint` and `protocol`.
HostedAgent uses `byoh/responses` with a same-project agent URL; version binding
is removed. `[task].format = "harbor"` and harness `type = "package"` are
reserved and rejected as not yet supported. The SDK environment modules are
unchanged from the preceding source check; service routing and CLI manifests
changed together.

## Public distributions are separate from development source

| Distribution | Public repository / ref | Verified commit | Latest listed version |
| --- | --- | --- | --- |
| CLI registry | `Azure/azure-dev`, `rle-releases` | `a62ebb0444c40d1f76110233e42862eefc8989a3` | `0.8.29-preview` |
| SDK registry | `Azure/azure-sdk-for-python`, `rle-releases` | `17415c9e72407a2e112b10d86d6e541afaae9abb` | `2.7.0b1` |

CLI public release was one version behind development source at this check.
The two public registry refs are unchanged from the October 14 check.
This is a capability difference, not just a version label: the installed public
`0.8.29-preview` rollout help is Loom-only and has no `--mode`/`--loom` flags.
The verified development `0.8.30-preview` source adds Foundry eval mode.
It also uses the new Harness schema and `init --protocol`/`--endpoint`;
public `0.8.29-preview` still uses the removed `--subtype` contract. Do not use
that public CLI to publish the current sample manifests.
Versions `0.8.28-preview` and `0.8.29-preview` are marked breaking in the public
registry. Use the intended installation source and its update workflow, not an
arbitrary binary or unmerged branch.

The public SDK wheel's SHA-256 was verified against the pinned registry.
Its `environments/__init__.py`, `_environment.py` and `_app.py` were compared
with remote SDK main and matched after line-ending normalization. Thus the
documented authoring base class and query-session wrapper are present in the
published RLE SDK, not merely in a local SDK feature branch.

## Implementation evidence to recheck

CLI paths are under `cli/azd/extensions/azure.ai.rle/`:

- `internal/project/rle_config.go`: strict TOML schema, endpoint/protocol binding,
  schema version and GRPO/default validation.
- `internal/project/identifiers.go`: environment-name syntax.
- `internal/cmd/root.go`, `init.go`, `run.go`, `publish.go`: core vs staged
  commands, Harness scaffolding gate, local-run constraints and build/publish.
- `internal/cmd/rollout.go`: train/eval backends, task-only input and model
  default handling.
- `internal/cmd/train.go`: training flag inputs.
- `README.md`, `extension.yaml`: extension sources, updates and source version.

SDK paths are under
`sdk/ai/azure-ai-projects/azure/ai/projects/rle/environments/`:

- `_environment.py`: `RLEnvironment`, defensive state, `GradeAction` and
  optional `rollout_graph`.
- `_app.py`: `create_app` query adaptation for `tools/list` and `tools/call`.

Vienna paths are under `src/azureml-api/src/RLE/`:

- `EntryPoints/Validation/RleEnvironmentDefaultsValidator.cs`: current
  `model`, `train.grpo`, `rollout.gym_openenv` defaults, not the older
  `reinforcement` defaults model.
- `EntryPoints/Configuration/RleExecuteRolloutOptions.cs`: service ceilings.
- `EntryPoints/Services/ExecuteRollout/HostedAgentRolloutInvoker.cs`:
  background Responses request, retrieval and final-text extraction.
- `McpEnvironmentHostedAgentRolloutTargetInvoker.cs` in that same directory:
  reset-observation input and same-session grading.
- `RolloutTargetResolver.cs`: same-project agent URL validation for Responses;
  no agent-version selection and no cross-project platform-token forwarding.
- `ByohRolloutTargetInvoker.cs`: 202 start, service-owned operation ID,
  derived poll/cancel resource, string final output and HTTPS URL validation.
- `McpEnvironmentRolloutSession.cs`: terminal Harness grade requires
  `done`, finite reward and boolean `is_success`.
- `CaptureProxy/src/capture_proxy/server.py`: the model proxy's catch-all
  supports `/v1/chat/completions` and `/v1/responses` (plus other dialects).
  Responses is supported in this source snapshot; still verify the actual
  deployed proxy and capture rather than assuming source equals deployment.

## Limits of this verification

This is **source and distribution verification**, not proof that a particular
Foundry deployment is running those commits. No managed environment was
published and no cloud rollout was performed during the refresh. Deployed
service configuration can also differ from source option defaults.

The October 14 isolated trials used the preceding manifest schema. They
published a Gym version that reached Ready and
completed real managed train and validation episodes with reward 1, two model
turns, one recorded helper call, token-level trainable capture and completed
rollout cleanup. These used a real Sweden Central Loom session/checkpoint
against the ready East US 2 environment. The verifier was corrected against
`CaptureProxy/src/capture_proxy/export.py`: exported calls are in `turns`,
with training arrays in `sequences`, not `nodes`.

A code-deployed HostedAgent in Sweden Central reached Active, and an actual
background Responses request executed the adapter's missing-context rejection;
scoped runtime logs confirmed the error masked by the client API. This is not
proof of deploying the user-built image or of a positive managed Harness
episode. The tested host/environment container loop remains local with a mock
model proxy. The owned cloud host/session and sampler were released afterward.

Other targets rejected Foundry policy support or failed provisioning/capacity.
These outcomes do not identify a deployed commit or invalidate this snapshot.
Keep publishing, image readiness, agent deployment, policy acceptance, capture,
rollout completion and cleanup as separate verification claims.
Those runtime results do not claim a fresh managed rollout using the October
15 schema redesign. Current-schema verification covers the remote source,
sample manifests and local skill regression checks.

When updating this skill, resolve the current remote refs again, read the
active implementation rather than stale README examples, distinguish public
release from development main, update both skill trees, and run the skill
checks. Refresh this baseline with the new exact revisions; don't leave
unqualified claims of "latest" without a dated source snapshot.
