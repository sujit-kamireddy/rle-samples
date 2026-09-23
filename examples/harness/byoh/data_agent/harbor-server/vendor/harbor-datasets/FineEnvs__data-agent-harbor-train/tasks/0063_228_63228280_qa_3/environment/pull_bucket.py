"""Pull this task's bucket prefix into /home/user/input/.

Invoked by Harbor's [environment.healthcheck] command (declared in task.toml)
— runs after container start, before the agent. Idempotent: skips work if
files are already present from a prior pull.
"""

import os
import sys
from pathlib import Path

from huggingface_hub import download_bucket_files, list_bucket_tree


def main() -> int:
    bucket = os.environ.get("HF_BUCKET") or "AdithyaSK/jupyter-agent-kaggle-all"
    prefix = os.environ["BUCKET_PREFIX"].rstrip("/") + "/"
    dest = Path("/home/user/input")
    dest.mkdir(parents=True, exist_ok=True)

    existing = [p for p in dest.iterdir() if p.is_file()]
    if existing:
        print(f"[pull_bucket] {dest}/ already has {len(existing)} file(s); skipping", flush=True)
        return 0

    targets = [
        (it.path, str(dest / Path(it.path).name))
        for it in list_bucket_tree(bucket, prefix=prefix, recursive=True)
        if getattr(it, "type", None) == "file"
    ]
    if not targets:
        print(f"[pull_bucket] FATAL: no files at hf://buckets/{bucket}/{prefix}", flush=True)
        return 2

    download_bucket_files(bucket, files=targets)
    print(f"[pull_bucket] staged {len(targets)} file(s) from {bucket}/{prefix}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
