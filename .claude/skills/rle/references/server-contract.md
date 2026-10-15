# Shared environment contract

Gym and both `byoh` protocols use an SDK `RLEnvironment`, OpenEnv control and
MCP tools. The **agent loop**, not the grader container's base class, differs.
The SDK provides routing and health; don't implement a parallel REST grader.

## App factory

For HTTP tools with Foundry's query-based session routing:

```python
from azure.ai.projects.rle.environments import GradeAction, create_app
from openenv.core.env_server.types import ConcurrencyConfig

app = create_app(
    ExampleEnvironment,
    GradeAction,
    ExampleObservation,
    state_cls=ExampleState,
    env_name="example_rl",
    concurrency_config=ConcurrencyConfig(
        max_concurrent_envs=1,
        session_timeout=600.0,
    ),
)
```

Supply real environment/observation/state classes. The Gym samples use upstream
OpenEnv `create_app` for their WebSocket-driven loop. Harness samples use the
SDK wrapper: it adapts `rle_session_id` query routing for HTTP MCP. Do not swap
the Harness wrapper for upstream OpenEnv while keeping query-scoped tool URLs.

Use one worker. RLE creates one environment per session; detached HTTP sessions
and the WebSocket must resolve to that same instance. A warm process can receive
different rollout credentials, so none belong in module-level mutable clients.

## Reset and private task selection

Implement `reset(seed=None, episode_id=None, ..., **kwargs)`. Capture unknown
selectors and reject them, or validate the complete task through its explicit
task model before doing work. OpenEnv can filter undeclared reset arguments;
without `**kwargs`, a typo may silently pick a different task.

Do not impose Gym's seed picker on every environment:

- Math/code support deterministic `seed` and separate `split` snapshots.
  Reuse their fixed-permutation `EpisodePicker` for an epoch-like seed traversal.
- BYOH data-code requires explicit dataset `split` and integer `task_index`;
  it rejects seeds and boolean indexes.
- Hosted competitive intelligence validates the complete caller-supplied task.

Reset must clear previous tool evidence and terminal status, store the private
task/reference on the instance, and return only policy-visible information.
Test missing, unknown and out-of-range selectors and reset after grading.
Do not trust a task selector the model puts into its final answer over the task
that reset already pinned.

Test strict selector types over the actual managed transport on a **fresh session**
before any valid reset, and again after grading. Direct Python tests do not prove
wire validation; an active-episode guard can also mask invalid input in a test.
Use boolean, string and float payloads when an integer-only selector is required.
Framework request models can coerce types: OpenEnv's HTTP `ResetRequest` accepts
some values that a strict integer check would reject. This does not imply its
WebSocket reset path does the same; test the path RLE actually uses. Preserve
valid task identity and state after rejection. Do not globally monkeypatch
framework request models to change validation.

## The reset observation is the agent's first turn

Harness agents do not receive the raw dataset row as `agent_input`. RLE resets
the environment, renders that observation as chat messages, and invokes the
agent with those messages. Use an explicit string `prompt` where the agent must
consume a precise first-turn shape. The samples use:

- BYOH: JSON text with `split` and `task_index`.
- HostedAgent: JSON text with `topic` and `execution_mode`.
- Gym math/code: structured chat `messages` describing the problem.

Keep the parser and observation renderer in agreement. Test that the rendered
input names the exact reset task, and never fall back to task zero or a string
like `"None"` when parsing fails. Do not expose reference solutions by
serializing the entire private task into the observation.

## Grade and state

Implement:

```python
def grade(self, action: GradeAction, timeout_s=None, **kwargs):
    ...
```

`action.response` is the final answer string. `action.rollout_graph` is optional
sanitized capture evidence; tolerate `None` for versions/targets that do not
provide it. Keep grading independent of the agent's claimed score.

- Return a terminal observation with `done=True` and a finite numeric reward
  for correct, incorrect, malformed and deliberate give-up submissions.
- For Harness, also expose a boolean `is_success` on the terminal observation;
  the service validates it along with termination and the finite reward.
  Define it according to task correctness, independently of reward shaping.
- Reject grade-before-reset, invalid lifecycle and unexpected options explicitly.
- Treat infrastructure/grader faults as errors, not fabricated zero-reward
  successes. Log safe diagnostic context and retain enough state to investigate.
- Put task ID and grading metrics in diagnostics without leaking reference
  answers to subsequent model turns.
- Persist SDK state snapshots through `_set_state`; override `state` only if
  deliberately implementing equivalent snapshot/isolation behavior.

Do not override ordinary tool `step()` dispatch: `RLEnvironment` handles MCP
actions and routes non-MCP `GradeAction` into `grade()`. Passing a bare `Action`
class to the app factory instead of `GradeAction` breaks terminal grading.

Terminal grading does not itself close a detached OpenEnv session. The caller
must detach and close it; implement `close()` to release task resources and
invalidate mutable tools. Never push unsolicited WebSocket frames.
