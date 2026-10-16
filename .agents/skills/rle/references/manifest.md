# `rle.toml` authoring schema

Use the current sample/CLI TOML shape below. Do not copy control-plane JSON
models literally into TOML: publishing translates the authoring manifest into
the service's type configuration and stored defaults.

## Minimal manifests

Gym:

```toml
schema_version = "1.0.0"

[rle]
name = "my_math_rl"
version = "1.0.0"

[harness]
type = "gym"
```

BYOH:

```toml
schema_version = "1.0.0"

[rle]
name = "my_byoh_rl"
version = "1.0.0"

[harness]
type = "byoh"
endpoint = "https://your-harness.example.com/invoke"
protocol = "rle-invocation"
```

HostedAgent:

```toml
schema_version = "1.0.0"

[rle]
name = "my_hosted_rl"
version = "1.0.0"

[harness]
type = "byoh"
endpoint = "https://<account>.services.ai.azure.com/api/projects/<project>/agents/my-deployed-agent"
protocol = "responses"
```

## Fields and target constraints

| Table / field | Value / meaning |
| --- | --- |
| `schema_version` | Use `"1.0.0"`; required with any model/train/rollout table; recommended otherwise |
| `[rle].name` | Environment identity matching `^[a-z][a-z0-9_]*$`; scaffold may rename it |
| `[rle].version` | Immutable `major.minor.patch` version, no prerelease/build suffix |
| `[rle].imageTag` | Optional string accepted for image-tag metadata compatibility; not required by samples |
| `[task].format` | Lowercase `json` (default); `harbor` is reserved and rejected as not yet supported |
| `[harness].type` | Lowercase `gym` (default), `byoh`, or reserved/not-yet-runnable `package` |
| `[harness].endpoint` | Required for `byoh`: absolute HTTPS URL, at most 2048 characters; no credentials/query/fragment |
| `[harness].protocol` | For `byoh`: `responses` or `rle-invocation`; set explicitly rather than relying on inference |
| `[harness].name` | Package name such as `opencode`, valid only with reserved `type = "package"` |
| `[model].name` | Optional model default, at most 256 characters; not an endpoint or credential |
| `[model].renderer_name` | Optional renderer default, at most 256 characters |
| `[train.grpo]` | Optional typed training defaults, not environment turn limits |
| `[rollout.gym_openenv].model_response_field` | Accepted optional string (maximum 128 characters) for legacy schema routing; omit for SDK MCP environments |

`gym` has no endpoint, protocol or package name. Both externally hosted async
agents and Foundry HostedAgents use `byoh`; the protocol selects the invocation
wire contract, not the environment MCP contract.

For `responses`, use the same-project Foundry **agent URL ending in
`/agents/<name>`**, not a version URL or `/endpoint/protocols/openai/responses`.
The service builds the Responses dispatch URL and rejects cross-project agent
URLs before forwarding platform credentials. Agent versions are no longer
targetable in this schema. Deploy the agent and configure its endpoint traffic
before publishing.

Removed `subtype`, `agent_name`, `agent_version`, `invocation_url`, `base_url`
and their legacy aliases are rejected, not silently translated. Replace an
old BYOH `invocation_url` with `endpoint` and `protocol = "rle-invocation"`;
replace HostedAgent name/version binding with the agent URL and
`protocol = "responses"`.

Identity and interface fields are not interchangeable. `[rle]` does **not**
contain `type`/`subtype` in the current authoring manifest. Never put the agent
name in `rle.name` merely because that agent is being optimized.
Lengths above use the validators' UTF-16 character count after trimming.
Unknown TOML fields are rejected by strict decoding. `imageTag` is camel-case;
do not introduce an `image_tag` alias or assume it changes `publish` behavior.
A new environment starts at `1.0.0`; later versions follow the direct next
major, minor or patch release, rather than skipping versions.

## Training defaults carried by the samples

These are the fields in the verified current CLI schema as well as the samples:

| `[train.grpo]` field | TOML type |
| --- | --- |
| `learning_rate` | float |
| `lora_rank` | integer |
| `group_size` | integer |
| `groups_per_batch` | integer |
| `num_epochs` | integer |
| `max_steps` | integer |
| `eval_every` | integer |
| `save_every` | integer |
| `remove_constant_reward_groups` | boolean |
| `sampling_seed` | integer |
| `min_trainable_group_members` | integer |

All these integer defaults must be positive when supplied, **including
`sampling_seed`**. `learning_rate` must be finite and positive;
`remove_constant_reward_groups` is a boolean. Do not use zero to mean "disabled"
unless the installed schema explicitly supports it.

Copy tuned values only when intentionally preserving the sample's training
setup. `train.grpo.max_steps` is the **training** step count, not the Gym
episode length. Task records and training/validation file paths are job inputs,
not tool schemas. Use the installed `train --help` for model/file/suffix flags;
don't invent a `[train.options]` free-form table.

The four shipped manifests do not contain `[rollout]`. The current service
accepts `rollout.gym_openenv.model_response_field`, but SDK MCP environments do
not need it. Do not invent additional rollout fields from service options or
old documentation.

## Patterns not to generate for current samples

- `[rle] type = ...` / `subtype = ...`: use `[harness]`.
- `[defaults.rollout]`, `[defaults.train.grpo]`: past documentation shapes,
  not current sample authoring tables.
- `[rollout.gym_openenv] model_response_field`: old schema-driven action
  routing; current environments use SDK `GradeAction.response` and MCP tools.
- `environment_protocol`: not an authoring selector. `[harness].protocol`
  selects agent invocation, not the environment's MCP protocol.
- `max_episode_steps` / `max_completion_tokens`: not represented in current
  sample manifests. Do not confuse them with `train.grpo.max_steps`.

### Current service defaults

Latest remote Vienna source normalizes `model`, `train` and `rollout` defaults,
with the fields listed above. The older `reinforcement` defaults model and its
environment-authored `max_episode_steps` / `max_completion_tokens` are no
longer active members. Gym limits are service-owned options, not authoring
fields. Do not copy those obsolete members back from an older local checkout.

Runtime session timeouts/capacity belong in app configuration; production
agent timeouts belong in the agent. Neither implies a supported TOML key.

## Validation and publication

Parse TOML locally, then use the installed CLI/service for semantic validation:

```powershell
python -c "import pathlib,tomllib; tomllib.loads(pathlib.Path('rle.toml').read_text(encoding='utf-8')); print('TOML syntax OK')"
azd ai rle publish
```

TOML parsing does not validate enum combinations, deployed agent versions,
HTTPS reachability or model/renderer compatibility. Inspect publish errors;
don't suppress unknown-field failures or delete fields blindly until it passes.

Keep registry locations, access tokens, API keys and service IDs out of this
manifest. The `byoh` harness endpoint is an intentional target binding; a
Responses agent URL necessarily includes its project. Set other deployment
context separately. Publish a
new version for contract/image changes; do not mutate an already published
version. `show` is a typed convenience view, not proof that every stored key
was rendered.
