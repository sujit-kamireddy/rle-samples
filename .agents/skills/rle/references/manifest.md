## The `rle.toml` manifest

`rle.toml` is the only file the RLE service reads directly. It is validated at publish time and
re-read at rollout time.

Schema-driven Gym mode:

```toml
schema_version = "1.0.0"

[rle]
name = "math_rl"
version = "1.0.0"

[harness]
type = "gym"

[rollout.gym_openenv]
model_response_field = "answer_text"
```

SDK `RLEnvironment` MCP mode:

```toml
schema_version = "1.0.0"

[rle]
name = "code_rl"
version = "1.0.0"

[harness]
type = "gym"
```

### `[task]` and `[harness] type = "package"`

The schema also accepts `[task] format = "json" | "harbor"` (default `json`) and
`[harness] type = "package"` with `name = "opencode"` (`name` is valid only with `package`). All
values are lowercase and unknown values are rejected. Harbor tasks and harness packages are **not
yet runnable**: the `azd ai rle` extension and the RLE service reject `format = "harbor"` and
`type = "package"` with a "not yet supported" error until Harbor execution ships. Do not use them
in samples yet.

```toml
[task]
format = "json"   # json (default) | harbor

[harness]
type = "gym"      # gym (default) | byoh | package
```

### Rules

**`schema_version` must be exactly `"1.0.0"`,** and it is *required whenever any of `[model]`,
`[train]`, or `[rollout]` is present. Omitting it while supplying those is a publish-time validation
error, not a silent default.

**`[harness] type` must be `gym`** (the default when omitted) for this skill's contract to apply.
`byoh` selects an entirely different rollout path. The old `type = "Gym"`/`"Harness"` and `subtype`
keys are rejected.

**Choose one Gym contract.** In schema-driven mode,
`[rollout.gym_openenv] model_response_field` is mandatory and must name a property on
exactly one action variant (maximum 128 characters). In SDK MCP mode, there is no protocol
field to set, and `model_response_field` must be omitted. Combining the two selectors asks
RLE to apply incompatible response contracts.

**There is no manifest field for the step budget any more.** RLE now caps every Gym episode at a
fixed 32 `step()` calls server-side; there is no `max_episode_steps` field in the schema (a
publish-time error if present) rather than defaulted. Design the environment to terminate well
before 32 steps regardless.

**There is no manifest field for the completion-token budget any more.** Every model call's
`max_tokens` is now a fixed server ceiling (**8192**) regardless of what the environment would
otherwise ask for; there is no `max_completion_tokens` field in the schema (a publish-time error if
present) rather than defaulted, following the same pattern as `max_episode_steps`. This may be
re-exposed in the manifest later once there is feedback on which environments actually need a
different budget.

### What the manifest does *not* cover

There is no manifest syntax for individual tools, the reset signature, the observation shape, or
the step budget (now a fixed server ceiling). Schema-driven tools come from action variants. SDK MCP
tools come from typed methods registered with `RLEnvironment.tool()`. The reset signature is
invisible to RLE in both modes.

There is also no `[train]` / `[train.options]` free-form table any more. `model`, `training-file`,
`validation-file`, and `suffix` are CLI-flag-only on `azd ai rle train`; GRPO and rollout
hyperparameters that should travel with the published environment live under `[train.grpo]` and
`[rollout]` instead (see the byoh harness reference and the `examples/harness/*/rle/rle.toml` samples
for the full field list).
