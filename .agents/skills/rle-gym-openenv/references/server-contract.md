## The OpenEnv server contract

### Entry point

`server/app.py` is the whole server:

```python
app = create_fastapi_app(
    MathEnvironment,
    MathAction,
    MathObservation,
    max_concurrent_envs=1,
    env_name="math_rl",
)
```

**Serve it with a single worker, and keep `max_concurrent_envs=1`.** A rollout container is leased
to one RLE instance for the life of the rollout. Extra workers do not add throughput; they add
processes that hold a different episode's state than the one being graded.

### RLE talks to you over `/ws`, not over plain HTTP

The Gym/OpenEnv rollout path uses the WebSocket endpoint exclusively. That has three consequences:

1. **The environment instance persists for the connection.** State set in `reset()` is still there in
   `step()`. This is why actions carry no episode id.
2. **Never send an unsolicited frame.** The session correlates each response to the request it has
   outstanding. A frame sent when nothing is awaiting one is a protocol violation, and the session
   is entitled to abort the rollout over it. Emit frames only as replies (the pre-loop error frames
   OpenEnv itself sends before the loop starts are the one exception, and they are fatal by design).
3. **Keep the connection alive.** The server pings on an interval (uvicorn's default WebSocket
   keepalive is 20s); long silent gaps get the connection dropped mid-rollout. Do not block the
   event loop for minutes inside `step()` — run long grading work off-thread.

### `reset()` — the silent-failure footgun

This is the single most dangerous thing in the contract, because it fails by **grading the wrong
episode and reporting success**.

```python
def reset(
    self,
    seed: Optional[int] = None,
    episode_id: Optional[str] = None,
    split: str = "train",
    **kwargs: Any,
) -> MathObservation:
    reject_unknown_selectors(kwargs)
    ...
```

Two rules, both mandatory:

- **The signature must end in `**kwargs`.**
- **The first statement must reject unknown selectors.**

Why: RLE forwards the job's `task` JSON to `reset()` verbatim and *cannot* validate it —
`GET /schema` describes actions, observations and state, and never the reset signature. OpenEnv's
`ResetRequest` is `extra="allow"`, and its server then filters the payload down to the parameters
`reset` actually declares (`_get_valid_kwargs` in `openenv/core/env_server/http_server.py`). So a
task like `{"instance_id": "abc"}` against a `reset(seed=...)` signature is dropped twice over,
`seed` stays `None`, and the episode picker draws a **random** row. The rollout completes, the reward
is real, and it belongs to a different episode than the one the job asked for. Nothing in the
response marks it wrong.

`reject_unknown_selectors` (see `math_rl/server/dataset.py`) turns that into a loud failure. Copy it.

### Episode selection

Select by **seed**, not by content. A seed should index into a fixed shuffled permutation of the
dataset rather than drawing independently, so that a training run using consecutive seeds
(`base_seed + i`) traverses distinct rows — independent random draws leave roughly 1/e of the rows
unvisited in what is meant to be one epoch, and duplicate others. `EpisodePicker` in
`math_rl/server/dataset.py` implements this; reuse it.

Support a `split` selector so training and validation draw from different rows.

### `step()`

- Read the concrete action (`.root` for a union), apply it, grade, return an observation.
- **Always reach `done=True` on every terminal path**, including the give-up path — an episode that
  simply runs out of steps without ever being graded is treated as a failed rollout downstream (see
  the grading-and-budget reference).
- Put the reward on the step that ends the episode.
- Return only what the policy should see. For a tool step, return the tool's result text; RLE — not
  the environment — owns the tool call's id, so return text, not a synthesized tool-call envelope.
