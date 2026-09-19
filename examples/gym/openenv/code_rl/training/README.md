# Training-job input

This directory is **not** part of the Docker image (see `.dockerignore`) and
is not read by the server at rollout time. It is the training job's own
input: `train.jsonl`/`validation.jsonl` here list, one per line, the exact
`task` payload a training job passes to this sample's `reset()` for one
episode.

For `code_rl`, `reset(seed, split)` picks a row from the dataset already
baked into the image (`../data/`) by indexing `seed % len(rows)` into a
fixed shuffled permutation -- so a row here carries no problem/test-case
content, only enough to make one episode's pick reproducible and to
identify which split it draws from:

```json
{"seed": 0, "split": "train"}
```

`train.jsonl` has one line per row in `../data/train.jsonl.gz` (900 lines,
`seed` 0..899); `validation.jsonl` mirrors `../data/validation.jsonl.gz`
(100 lines). Regenerate both if the baked dataset's row counts change --
they must stay in lockstep with `../data/source.json`'s `rows` values.

Other samples with a different `reset()` signature (for example one that
takes a specific instance ID, or no per-episode input at all) shape this
file's rows differently -- see that sample's own `training/README.md`.
