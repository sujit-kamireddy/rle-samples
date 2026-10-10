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

**There is no manifest field for the completion-token budget any more.** Every model call's
`max_tokens` is now a fixed server ceiling (**8192**) regardless of what the environment would
otherwise ask for; the old `[defaults.rollout] max_completion_tokens` field is gone from the schema
entirely (a publish-time error if present) rather than defaulted, following the same pattern as
`max_episode_steps`. This may be re-exposed in the manifest later once there is feedback on which
environments actually need a different budget.

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
