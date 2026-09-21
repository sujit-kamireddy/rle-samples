# Job data (training-job input)

This directory is **not** part of the Docker image (see `.dockerignore`) and
is not read by the server at rollout time. It is the training job's own
input: `train.jsonl`/`validation.jsonl` here list, one per line, the exact
`task` payload a training job passes to this sample's `reset()` for one
episode.

For `code_rl`, `reset(seed, split)` picks a row from the dataset already
baked into the image (`../env_data/`) by indexing `seed % len(rows)` into a
fixed shuffled permutation -- so a row here carries no problem/test-case
content, only enough to make one episode's pick reproducible and to
identify which split it draws from:

```json
{"seed": 0, "split": "train"}
```

`train.jsonl` has one line per row in `../env_data/train.jsonl.gz` (900 lines,
`seed` 0..899); `validation.jsonl` mirrors `../env_data/validation.jsonl.gz`
(100 lines). Regenerate both if the baked dataset's row counts change --
they must stay in lockstep with `../env_data/source.json`'s `rows` values.

> **The `task` payload is a selector, not the episode's content.** Every row
> here must *name* the episode it means. A row carrying the problem itself --
> `{"messages": [...], "problem": ...}` -- supplies no `seed`, and the episode
> that gets graded is then unrelated to the row.
>
> Nothing upstream catches this for you. OpenEnv's `ResetRequest` is
> `extra="allow"` and its server filters the payload down to `reset()`'s own
> signature, so an unrecognised key is dropped twice over without erroring.
> RLE cannot catch it either: it passes the job's task through verbatim, and
> `GET /schema` describes actions, observations and state -- never the reset
> signature. This environment therefore ends `reset()` in `**kwargs` and calls
> `reject_unknown_selectors()` on the leftovers, so a mis-shaped row fails the
> rollout instead of silently mis-grading it.

Other samples with a different `reset()` signature (for example one that
takes a specific instance ID, or no per-episode input at all) shape this
file's rows differently -- see that sample's own `job_data/README.md`.
