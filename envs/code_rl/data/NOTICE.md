# Compact `code_rl` Dataset Snapshot

`train.jsonl.gz` and `validation.jsonl.gz` are a filtered, reformatted subset
of [`agentica-org/DeepCoder-Preview-Dataset`](https://huggingface.co/datasets/agentica-org/DeepCoder-Preview-Dataset).
The upstream dataset card declares the MIT license.

The snapshot intentionally excludes the upstream `solutions` field. It
contains 900 TACO training tasks and 100 Codeforces validation tasks from the
revision recorded in `source.json`; each task has at most three complete test
cases and 12 KiB of test input/output. It is designed for a fast,
self-contained RLE sample—not a benchmark.

To refresh it from the repository root, run:

```bash
python envs/code_rl/tools/generate_compact_dataset.py
```
