# Grading, capture and budgets

## Grade the task that reset selected

Use the environment's private task and the same session's tool evidence. The
agent's answer, claimed reward or embedded selector cannot replace those.
Grade ordinary wrong/malformed answers deterministically, but surface genuine
grader/transport/setup faults explicitly rather than hiding them as reward zero.

Test exact values, not merely `reward is not None`:

| Sample | Correct | Wrong but formatted | Malformed |
| --- | --- | --- | --- |
| Gym math | `1.0` | `0.0` | `-0.1` (no boxed answer) |
| Gym code | `1.0` | `0.0` (fenced, tests fail) | Negative format coefficient (unfenced) |
| BYOH data-code | Correctness times disclosure multiplier | Task/rubric-specific | Task/rubric-specific |
| Hosted competitive intelligence | Rubric rescaled to `[0,1]` | Rubric-specific | Rubric-specific |

The math helper is optional; absence of tool use alone is not a penalty there.
Disclosure/tool-discipline requirements in other samples are task-specific,
not a universal RLE rule.

Use finite numeric rewards. Choose a bounded, comparable task scale and preserve
ordering between malformed, wrong and correct responses. Do not clamp all
format penalties to zero merely to imitate a different sample's scale.
The Hosted sample's `[0,1]` mapping is not evidence that every RLE environment
must have that range; the Gym samples deliberately use negative penalties.

Keep scoring separately testable; avoid hidden-test execution on unusable
submissions. Protect the grader from untrusted code and bound subprocess,
expression-evaluation and external-grader runtimes.

## Terminal episodes and captured model calls

Return `done=True` with a reward for every expected terminal submission,
including giving up. A legitimate single-turn Gym episode is valid when it
terminates graded; don't add dummy model calls to avoid graph validation.

`degenerate_rollout` can indicate an ungraded episode or missing model calls.
For Harness, verify every model call uses the supplied capture proxy;
successful answer grading does not prove the policy trajectory was captured.
Inspect tool/call counts and task identity in diagnostics.

Foundry-backed `rollout --mode eval` produces text-only captures that are
intentionally not trainable. Do not diagnose that as a failure or add artificial
turns. Training trajectories require the Loom-backed path, with token
IDs/logprobs; `--mode eval --loom` is also available for Loom evaluation.

Use optional `GradeAction.rollout_graph` for additional capture-based rubric
evidence where available; don't require it on every target/SDK version.
Do not reconstruct missing capture by trusting an agent-submitted transcript.

## Separate the budgets

Latest remote service options default to Gym ceilings of **32 turns** and
**8192 completion tokens**, and an Execute Rollout deadline of **1800 seconds**.
These are service behavior, not configurable authoring manifest fields, and can
differ by deployment configuration. Verify deployed limits when diagnosing
exhaustion.

- Design a Gym episode to finish well inside its service turn budget.
- Don't treat training `max_steps` as an episode/tool budget.
- A token cutoff can break an otherwise correct fenced/boxed submission.
  Inspect finish reasons/usage before concluding the grader is wrong.
- OpenEnv `session_timeout` bounds detached-session idleness, not total rollout
  runtime; attached clients still require explicit cleanup.
- Harness agents control their own model calls/tool loops. Bound individual
  model requests and total agent runtime separately and below service deadlines.
- BYOH cancel is advisory. Always have independent timeout/cleanup.
- Gateway HTTP deadlines and WebSocket rollout deadlines need not match.
  A multi-minute agent should not copy a short HTTP timeout into its entire
  loop without confirming the transport.

Do not advertise a manifest token-budget knob when none is represented in the
current authoring schema. Shorten expected output or change the appropriate
agent/service configuration only when that surface actually supports it.
