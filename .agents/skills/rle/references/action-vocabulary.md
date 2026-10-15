# Model-visible tools

The current vocabulary comes from MCP `tools/list`, not discriminator members
in a custom OpenEnv action union. `GET /schema` is useful for inspecting
observation/state/action serialization, but does not replace MCP tool discovery.
There is no `model_response_field` to configure.

## Adding or editing a tool

1. Add a named method with concrete argument types and a useful docstring.
2. Register it after `super().__init__()` with `self.tool()(self.method)`.
3. Check its published `tools/list` name, description and `inputSchema`.
4. Call it on an active session and assert its result and state changes.
5. Grade that same session and verify the rubric sees the evidence.

```python
from azure.ai.projects.rle.environments import RLEnvironment

class ExampleEnvironment(RLEnvironment):
    def __init__(self):
        super().__init__()
        self.tool()(self.check_candidate)

    def check_candidate(self, candidate: str) -> dict[str, bool]:
        """Check a candidate without revealing the private reference answer."""
        # Implement task-specific validation/checking here.
        raise NotImplementedError
```

This fragment shows registration only; a concrete environment must also
implement `reset` and `grade`. Do not leave a placeholder in the shipped app.

Names and descriptions are policy-facing instructions. Preserve production
tool names/schemas for Harness environments instead of substituting easier
training-only tools. The competitive-intelligence sample deliberately publishes
production Pydantic parameter schemas while allowing malformed calls to reach
the rubric; copy that advanced pattern only when grading invalid calls is an
explicit requirement, not as an excuse to use untyped schemas everywhere.

## State and result rules

- Store per-episode tool evidence on the environment instance, never a process
  global or a second independently created environment.
- SDK `state` is a defensive copy. If using `_state`, mutate the snapshot then
  call `_set_state(state)`; modifying `self.state` alone does not persist.
- Check lifecycle: tools before reset or after terminal grading must not silently
  mutate an unrelated episode.
- Return JSON-serializable values or text; the SDK produces the MCP result.
  Do not manufacture model tool-call IDs or assistant messages in tool results.
- Keep labels/reference answers out of descriptions, public state and outputs.
- Distinguish protocol faults from task-level tool failures. If a rubric needs
  to score a rejected action, return the documented task-level error result.
  Do not turn session/transport faults into successful acknowledgements.
- Offload expensive blocking work appropriately; don't block the event loop
  through its keepalive or rollout deadline.

## Migrating an old schema-driven environment

Preserve selectors, grader and reward semantics, then:

- Replace the action-union tool members with registered MCP methods.
- Replace the final response action with SDK `GradeAction`; grade
  `action.response`, and accept optional `action.rollout_graph`.
- Replace custom ordinary `step()` dispatch with `RLEnvironment`'s dispatch.
- Wire `GradeAction`, observation/state models and the suitable `create_app`.
- Remove `model_response_field`, `[rollout.gym_openenv]` and environment
  protocol selectors from the manifest. Preserve `[harness].protocol` for
  `byoh` invocation. Do not retain obsolete keys as comments in a template.
- Test the same known tasks and tool evidence before and after migration.

For old Harness REST tools, move task initialization to `reset`, scoring to
`grade`, and each `/tools/<name>` operation to an MCP tool. Change the agent's
tool adapter to use the supplied session-scoped MCP endpoint.
