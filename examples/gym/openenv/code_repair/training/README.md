# Training-job input

This directory is **not** part of the Docker image (see `.dockerignore`) and
is not read by the server at rollout time. It is the training job's own
input: `train.jsonl` here lists, one per line, the exact `task` payload a
training job passes to this sample's `reset()` for one episode.

`code_repair` wraps exactly one fixed SWE-bench-Lite instance (baked into
the image as `fixtures/instance.json`) -- `reset(seed=None, **kwargs)`
ignores every per-episode argument (`del seed, kwargs`), so there is no
seed/index to vary. `train.jsonl` is therefore a single line, the empty
task object:

```json
{}
```

There is no `validation.jsonl`: with one instance there is no separate
held-out split. A sample with several instances would instead give each
line whatever field `reset()` uses to pick one (for example
`{"instance_id": "..."}`) -- see `math_rl`/`code_rl`'s `training/README.md`
for the seed-based version of that.
