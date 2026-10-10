## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| HTTP 400 `EnvironmentContractViolation` on a schema-driven rollout | the published version has no `[defaults.rollout.gym_openenv] model_response_field` | add it and publish a new version |
| SDK MCP environment is treated as schema-driven | `[rle] environmentProtocol` is absent or not `mcp_environment` | set `[rle] environmentProtocol = "mcp_environment"`, remove `model_response_field`, and republish |
| `azure.ai.projects.rle` cannot be imported | the installed public SDK build does not contain the additive RLE package | install the pinned RLE-enabled artifact documented by the sample; equal public package versions may not have the same modules |
| MCP tool is missing or has an untyped schema | it was registered before `RLEnvironment.__init__`, not registered, or uses untyped parameters | call `super().__init__()` first, then `self.tool()(self.method)`, with concrete annotations |
| Correct SDK MCP answer gets reward 0 | tool evidence was stored globally or on another instance | update `self.state` inside the tool and read that same state in `grade(GradeAction)` |
| Conformance error: "does not declare the configured model_response_field" | the field name in `rle.toml` does not match any action property | read the error — it lists every property it found; usually a typo or a renamed field |
| Conformance error: declared "on more than one action variant" | two union members share the response property | rename one, or make the non-response variant carry a differently-named field |
| Conformance error: variant "with no discriminator" | a union member has no `Literal` discriminator field | add `type: Literal["…"]` and declare the discriminator on the `RootModel` |
| The model is offered a tool you did not intend | every non-response variant becomes a tool | remove the union member, or make it the response variant |
| The model is offered **no** tools | single-variant action schema — this is correct for a single-turn environment | add a union member to add a tool |
| HTTP 500 `RolloutDependencyFailed` | a dependency of the rollout (commonly the capture proxy's model completion) failed; the error body carries no inner detail | reproduce locally with `azd ai rle run` first to rule out the environment, then escalate with the rollout id — this class of failure is usually service-side, not authoring |
| Rollout export reports `degenerate_rollout` (FATAL) | exactly one model call **and** the episode came back ungraded | make every terminal path set a reward and `done=True`; never end by exhausting the step budget |
| Reward is real but for the wrong problem | `reset()` dropped an unknown selector and the picker drew a random row | end `reset()`'s signature in `**kwargs` and call `reject_unknown_selectors(kwargs)` first |
| Answers look truncated / format score is always negative | completion hit RLE's fixed 8192-token server ceiling (no longer a manifest field to raise) | shorten the required reasoning so it fits inside the ceiling |
| Episode stops earlier than expected | the episode is hitting RLE's fixed 32-step server ceiling (no longer a manifest field to raise) | redesign the environment to terminate with a grade well inside 32 steps |
| Connection drops mid-rollout | `step()` blocked the event loop past the WebSocket keepalive interval | move long grading work off-thread |
| Rollout aborts citing an unexpected frame | the environment sent a WebSocket frame when nothing was awaiting a response | only ever reply; never push |
| Works locally, fails in a rollout | something is fetched at runtime that the container cannot reach | bake it into `env_data/` |
| Local container serves a stale image | `azd ai rle run` rebuilt from a cached layer | re-run with `--watch`, or rebuild explicitly |

### Reading a failure in the right order

1. **Did it fail before any model call?** Then it is a contract failure — manifest or action schema.
   The message names the exact key or property.
2. **Did it fail during the loop?** Then it is the environment or a dependency. Reproduce with
   `azd ai rle run` and a scripted `/ws` exchange; if that passes, it is not the environment.
3. **Did it fail at export?** Then the loop ran but produced something untrainable — almost always
   an ungraded episode.
