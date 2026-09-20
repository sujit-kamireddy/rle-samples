# Job data (training-job input)

This directory is **not** part of the Docker image (see `.dockerignore`) and
is not read by the server at rollout time. It is the training job's own
input: `train.jsonl`/`validation.jsonl` here list, one per line, the exact
`task` payload a training job passes to this sample's `reset()` for one
episode.

For `code_repair`, `reset(seed, split)` picks a row from the dataset already
baked into the image (`../env_data/`) by indexing `seed % len(rows)` into a
fixed shuffled permutation -- so a row here carries no instance/patch
content, only enough to make one episode's pick reproducible and to
identify which split it draws from:

```json
{"seed": 0, "split": "train"}
```

`train.jsonl` has one line per row in `../env_data/train.jsonl.gz` (32 lines,
`seed` 0..31); `validation.jsonl` mirrors `../env_data/validation.jsonl.gz`
(8 lines). Regenerate both if the baked dataset's row counts change -- they
must stay in lockstep with `../env_data/source.json`'s `rows` values. See
`math_rl`/`code_rl`'s `job_data/README.md` for the same convention.
