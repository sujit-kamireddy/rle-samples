## Grading, token budget, and the ungraded-rollout trap

### Reward shaping

Both samples use the same two-part shape: a small format term that is cheap to satisfy, plus the
real correctness term.

`math_rl` (`server/math_rl_environment.py`, `FORMAT_COEF = 0.1`):

```python
reward = FORMAT_COEF * (format_score - 1.0) + correct_score
```

- answer without `\boxed{}` → `-0.1`, and correctness is **not even graded**
- correct, well-formed answer → `1.0`

`code_rl`: `1.0` correct and fenced, `0.0` fenced but failing tests, `-FORMAT_COEF` unfenced.

The pattern to copy: **a malformed response is worse than a wrong one**, and grading a malformed
response is skipped entirely (it is also the expensive path — do not run hidden tests on output you
already know is unusable).

Keep the scale bounded and comparable across episodes. Rewards that vary in magnitude with the
difficulty of the row teach the policy about the dataset, not about the task.

### The ungraded-rollout trap

After the loop finishes, the rollout graph is exported and validated. A rollout with **exactly one
model call** is checked against a rule named `degenerate_rollout`, which is **FATAL** — the export
refuses to stand behind it.

That rule exists to catch an agent that stopped after its first response. The graph cannot tell that
apart from a legitimate one-question/one-answer episode, because both produce the same shape: one
root, one turn, nothing to stitch. The difference is only knowable from the environment, so RLE
passes an `episode_terminated` assertion when — and only when — **the Gym episode came back graded**.

The practical consequence for you:

> A single-turn environment is completely fine, **provided the episode terminates with a grade.** An
> episode that ends ungraded — the step budget ran out, or a code path returned without setting a
> reward — is exported as `degenerate_rollout` FATAL and the rollout is wasted.

So:

- Set `max_episode_steps` to what the episode actually needs. A single-turn environment declaring
  `max_episode_steps = 1` and returning `done=True` on its first `step()` is the correct design, not
  a workaround.
- Never let the loop end by exhausting the step budget. If the policy has not submitted by the last
  step, grade it as a failure and set `done=True` yourself.
- Never return from `step()` on an error path without a reward and `done=True`.

### Token budget

`max_completion_tokens` in `rle.toml` becomes `max_tokens` on every model call, clamped to the 8192
server ceiling (see the manifest reference). Two failure modes, both quiet:

- **Too low:** a reasoning model is cut off mid-thought and submits a truncated answer, which the
  format term then scores as malformed. It looks like a bad policy; it is a bad budget.
- **Too high:** every call pays for headroom nobody uses.

Measure it. Run several episodes locally against the real model, look at the longest completion that
terminated on its own, and set the budget above that — then record the measurement in a comment in
`rle.toml` the way `math_rl` does, so the next person does not have to re-derive it.
