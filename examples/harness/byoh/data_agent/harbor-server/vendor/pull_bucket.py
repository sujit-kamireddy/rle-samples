"""Pull this task's dataset files into /home/user/input/.

Invoked by Harbor's [environment.healthcheck] command (declared in task.toml) -- runs after
container start, before the agent.

Reads from a plain public HTTPS object store instead of a Hugging Face bucket: no SDK, no
credentials, no HF_TOKEN, nothing to install beyond the standard library. The store grants
anonymous read on individual blobs but deliberately not container listing, so the set of files
belonging to a prefix is published alongside them as <prefix>/_manifest.txt.

Idempotency is decided against that manifest rather than against "is the directory non-empty".
The healthcheck retries, and a run that was cut off partway leaves a directory that is non-empty
but incomplete -- which the old check accepted, handing the agent a truncated dataset and turning a
fetch failure into a wrong answer. Comparing the manifest to what is on disk makes a resumed run
finish the job instead.

That also dictates how files are published. The largest prefix here is 844MB across 2,004 files and
takes ~5 minutes to pull, well past the 180s healthcheck timeout, so being cut off mid-fetch is the
normal case rather than the exceptional one. Each file is therefore moved into place as soon as it
lands, not at the end of the batch: a kill leaves every completed file where the next attempt can
see it, and the retry downloads only the remainder. Publishing the batch atomically instead would
be pointless here -- the manifest, not the directory's emptiness, is what decides completeness --
and actively harmful, because each attempt would discard the work of the one before it and the
healthcheck would never converge.

Downloads stage through a hidden sibling directory on the same filesystem, so a partially written
file is never visible under a name the manifest lists, and `/home/user/input/` never accumulates
`.part` debris for the agent to trip over.
"""

from __future__ import annotations

import concurrent.futures
import os
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

# Absolute by default because the baked-in copy runs as a task container's healthcheck, where
# `/home/user/input` is the path the task instruction names. `BUCKET_DEST` exists for the opposite
# case -- `server/opencode_direct.py` runs many rollouts side by side inside one container, so each
# needs its own directory rather than a single shared one.
DEST = Path(os.environ.get("BUCKET_DEST", "/home/user/input"))
STAGE = Path(os.environ.get("BUCKET_STAGE", str(DEST.parent / ".input-stage")))
ATTEMPTS = 4
TIMEOUT_SEC = 120
# One prefix in this collection holds 4,330 files and another 844MB, so the fetch is parallel.
# These are object-store round trips rather than compute, so the 1-CPU container is not the limit;
# tunable because the useful width depends on the deployment's egress more than on the task.
WORKERS = int(os.environ.get("BUCKET_WORKERS", "16"))


def _fetch(url: str, dest: Path) -> None:
    """Streams one blob to `dest`, retrying transient failures with a backoff."""
    last: Exception | None = None
    for attempt in range(ATTEMPTS):
        try:
            with urllib.request.urlopen(url, timeout=TIMEOUT_SEC) as response:
                tmp = dest.parent / (dest.name + ".part")
                with tmp.open("wb") as handle:
                    # Streamed, not `.read()`: some of these are multi-hundred-MB HDF5 files and
                    # the container has 1GB of RAM.
                    shutil.copyfileobj(response, handle)
                tmp.replace(dest)
                return
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            last = exc
            # A 404 is a manifest/store mismatch, not a blip; retrying only delays the real error.
            if isinstance(exc, urllib.error.HTTPError) and exc.code == 404:
                break
            time.sleep(2**attempt)
    raise RuntimeError(f"{url}: {last}")


def _read_text(url: str) -> str:
    for attempt in range(ATTEMPTS):
        try:
            with urllib.request.urlopen(url, timeout=TIMEOUT_SEC) as response:
                return response.read().decode("utf-8")
        except (urllib.error.URLError, OSError, TimeoutError):
            if attempt == ATTEMPTS - 1:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def main() -> int:
    base = os.environ["BUCKET_BASE_URL"].rstrip("/")
    prefix = os.environ["BUCKET_PREFIX"].strip("/")
    root = f"{base}/{urllib.parse.quote(prefix)}"

    try:
        manifest = _read_text(f"{root}/_manifest.txt")
    except Exception as exc:  # noqa: BLE001
        print(f"[pull_bucket] FATAL: no manifest at {root}/_manifest.txt: {exc}", flush=True)
        return 2

    names = sorted({line.strip() for line in manifest.splitlines() if line.strip()})
    if not names:
        print(f"[pull_bucket] FATAL: manifest at {root} lists no files", flush=True)
        return 2
    # Every prefix in this collection is flat, and the task prompt tells the agent so. Rather than
    # flattening a nested name and risking two files landing on one path, refuse it.
    nested = [name for name in names if "/" in name]
    if nested:
        print(f"[pull_bucket] FATAL: manifest lists subdirectories: {nested[:3]}", flush=True)
        return 2

    DEST.mkdir(parents=True, exist_ok=True)
    missing = [name for name in names if not (DEST / name).is_file()]
    if not missing:
        print(f"[pull_bucket] {DEST}/ already has all {len(names)} file(s); skipping", flush=True)
        return 0

    STAGE.mkdir(parents=True, exist_ok=True)
    done = 0
    try:
        def _one(name: str) -> None:
            staged = STAGE / name
            _fetch(f"{root}/{urllib.parse.quote(name)}", staged)
            # Published immediately, so a run killed by the healthcheck timeout keeps everything it
            # already fetched and the next attempt resumes rather than restarting.
            staged.replace(DEST / name)

        with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
            for _ in pool.map(_one, missing):
                done += 1
    except Exception as exc:  # noqa: BLE001
        print(f"[pull_bucket] FATAL after {done}/{len(missing)} file(s): {exc}", flush=True)
        return 2
    finally:
        shutil.rmtree(STAGE, ignore_errors=True)

    print(f"[pull_bucket] fetched {done} file(s) from {root}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
