## The `rle.toml` manifest

`rle.toml` is the only file the RLE service reads directly. It is validated at publish time and
re-read at rollout time.

Schema-driven Gym mode:

```toml
schema_version = "1.0.0"

[rle]
name = "math_rl"
version = "1.0.0"
type = "Gym"
subtype = "OpenEnv"

[defaults.rollout]
max_completion_tokens = 8192

[defaults.rollout.gym_openenv]
model_response_field = "answer_text"
```

SDK `RLEnvironment` MCP mode:

```toml
schema_version = "1.0.0"

[rle]
name = "code_rl"
version = "1.0.0"
type = "Gym"
subtype = "OpenEnv"
environmentProtocol = "mcp_environment"

[defaults.rollout]
max_completion_tokens = 8192
```

### Rules

**`schema_version` must be exactly `"1.0.0"`,** and it is *required whenever any `[defaults.*]`
table is present. Omitting it while supplying defaults is a publish-time validation error, not a
silent default.

**`type` / `subtype` must be `Gym` / `OpenEnv`** for this skill's contract to apply. Other
combinations select entirely different rollout paths.

**Choose one Gym contract.** In schema-driven mode,
`[defaults.rollout.gym_openenv] model_response_field` is mandatory and must name a property on
exactly one action variant (maximum 128 characters). In SDK MCP mode, set
`[rle] environmentProtocol = "mcp_environment"` and omit `model_response_field`. Combining the two
selectors asks RLE to apply incompatible response contracts.

**There is no manifest field for the step budget any more.** RLE now caps every Gym episode at a
fixed 32 `step()` calls server-side; the old `[defaults.reinforcement] max_episode_steps` field is
gone from the schema entirely (a publish-time error if present) rather than defaulted. Design the
environment to terminate well before 32 steps regardless.

**`max_completion_tokens` is the per-call `max_tokens` on the model request, under
`[defaults.rollout]`.** Server ceiling: **8192**. Effective value is `min(yours, 8192)` when you set
a positive value, and **8192** when you leave it unset — the rollout always sends an explicit
budget, so the sampler's own default never applies. Asking for more than 8192 is not an error; it is
silently clamped, because the capture proxy applies the same ceiling downstream.

Do not treat the ceiling as a free default to copy blindly: a budget far above what the episode
needs costs tokens on every call, and a budget below it truncates the model mid-reasoning and
submits a half-formed answer that grades as wrong. Measure it — `math_rl` documents in its own
`rle.toml` that 4096 truncated 2 of 3 sampled problems against Qwen3-32B while the longest
self-terminating run used 6252, which is why it sits at 8192.

### What the manifest does *not* cover

There is no manifest syntax for individual tools, the reset signature, the observation shape, or
the step budget (now a fixed server ceiling). Schema-driven tools come from action variants. SDK MCP
tools come from typed methods registered with `RLEnvironment.tool()`. The reset signature is
invisible to RLE in both modes.

There is also no `[train]` / `[train.options]` table any more. `model`, `training-file`,
`validation-file`, and `suffix` are CLI-flag-only on `azd ai rle train`; GRPO and rollout
hyperparameters that should travel with the published environment live under
`[defaults.train.grpo]` and `[defaults.rollout]` instead (see the Harness reference and the
`examples/harness/*/rle/rle.toml` samples for the full field list).
