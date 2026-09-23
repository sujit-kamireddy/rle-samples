#!/usr/bin/env python3
"""Repoints every task's dataset fetch from the Hugging Face bucket to a public HTTPS store.

The upstream tasks pull their Kaggle files from `AdithyaSK/jupyter-agent-kaggle-all` using
`huggingface_hub` and an `HF_TOKEN` threaded in from the host. That is a dependency on someone
else's account for every one of 5000 rollouts. This rewrite swaps it for a plain anonymous object
store that we own, fetched with nothing but the standard library -- see `tools/pull_bucket.py` for
why the file list travels as a per-prefix `_manifest.txt`.

Three edits per task, all inside the vendored tarball because the task directory *is* the contract
Harbor reads:

  * `task.toml`  -- `HF_BUCKET` becomes `BUCKET_BASE_URL`; the `HF_TOKEN` passthrough is dropped.
  * `environment/pull_bucket.py` -- replaced wholesale with the stdlib fetcher.
  * `environment/Dockerfile` -- gains a `COPY` that installs that fetcher over the one baked into
    the base image.

That third edit is the one that makes the other two take effect, and it is easy to miss. The task
`environment/` directory is *not* uploaded into the sandbox: Harbor skips the upload whenever an
`environment/Dockerfile` exists (`harbor.environments.definition.should_upload_environment_dir`),
and these tasks all have one. The upstream Dockerfile is `FROM ...:base` plus a `WORKDIR` and
copies nothing, so the `/opt/pull_bucket.py` the healthcheck runs is the one baked into
`savatar101/env-data-agent-train:base` -- and that copy imports `huggingface_hub` and falls back to
a hardcoded `HF_BUCKET` default when the variable is absent. Rewriting only `task.toml` would
therefore not fail; it would quietly keep fetching from the upstream Hugging Face bucket. The
`environment/` directory is the Docker build context for both the docker and E2B backends, so
copying the file in is a one-liner.

`KAGGLE_DATASET_NAME` and `BUCKET_PREFIX` are left alone: the prefixes already match the store's
`<owner>__<dataset>` layout one-for-one, which is what made this swap a rename rather than a
migration.

    python tools/bake_bucket_source.py --verify   # report drift, write nothing
    python tools/bake_bucket_source.py --write    # rewrite the tarball in place
"""

from __future__ import annotations

import argparse
import io
import shutil
import sys
import tarfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TARBALL = REPO_ROOT / "harbor-server" / "vendor" / "harbor-datasets.tar.gz"
FETCHER = REPO_ROOT / "tools" / "pull_bucket.py"

BASE_URL = "https://sujit-hf-datasets-d0dgfkeee4fxepga.b01.azurefd.net"

OLD_BUCKET_LINE = 'HF_BUCKET = "AdithyaSK/jupyter-agent-kaggle-all"'
NEW_BUCKET_LINE = f'BUCKET_BASE_URL = "{BASE_URL}"'
OLD_TOKEN_LINE = 'HF_TOKEN = "${HF_TOKEN}"'

# Overwrites the copy baked into the base image. `environment/` is the build context, so the
# source path is a bare filename.
COPY_LINE = "COPY pull_bucket.py /opt/pull_bucket.py"

# Left in place by design; asserted so a future upstream re-bake that drops them is caught here
# rather than at rollout time.
KEEP_KEYS = ("BUCKET_PREFIX = ", "KAGGLE_DATASET_NAME = ")


class DriftError(RuntimeError):
    """Raised when a task.toml does not have the shape this rewrite assumes."""


def rewrite_toml(text: str) -> str:
    """Returns `text` pointing at `BASE_URL`, with the token passthrough removed.

    Handles both the original upstream shape (an `HF_BUCKET` line plus an `HF_TOKEN` passthrough)
    and an already-migrated file whose `BUCKET_BASE_URL` simply names a different host -- moving
    the collection behind a CDN, say. Raises `DriftError` rather than silently skipping, because a
    task that quietly kept the old value would fail only once a rollout tried to fetch it.
    """
    if NEW_BUCKET_LINE in text:
        return text
    lines = text.split("\n")
    for key in KEEP_KEYS:
        if not any(line.startswith(key) for line in lines):
            raise DriftError(f"missing {key.strip()}")

    existing = [line for line in lines if line.startswith("BUCKET_BASE_URL = ")]
    if existing:
        if len(existing) != 1:
            raise DriftError("expected exactly one BUCKET_BASE_URL line")
        return "\n".join(NEW_BUCKET_LINE if line == existing[0] else line for line in lines)

    if lines.count(OLD_BUCKET_LINE) != 1:
        raise DriftError(f"expected exactly one {OLD_BUCKET_LINE!r}")
    if lines.count(OLD_TOKEN_LINE) != 1:
        raise DriftError(f"expected exactly one {OLD_TOKEN_LINE!r}")
    out = [NEW_BUCKET_LINE if line == OLD_BUCKET_LINE else line for line in lines]
    out.remove(OLD_TOKEN_LINE)
    return "\n".join(out)


def rewrite_dockerfile(text: str) -> str:
    """Returns `text` with the fetcher copied in over the base image's own.

    Appended last so it lands after the `FROM`, and asserted to be the only `COPY`, so a future
    upstream Dockerfile that already installs something at that path is caught rather than
    silently double-copied.
    """
    if COPY_LINE in text:
        return text
    if "/opt/pull_bucket.py" in text:
        raise DriftError("Dockerfile already writes /opt/pull_bucket.py")
    if not text.lstrip().startswith("FROM "):
        raise DriftError("Dockerfile does not start with FROM")
    return text.rstrip("\n") + "\n" + COPY_LINE + "\n"


def _assert_only_expected_lines_changed(before: str, after: str) -> None:
    """Guards the rewrite: only the known line edits, and nothing else.

    Two accepted shapes -- the initial migration off Hugging Face (drops two lines, adds one) and a
    later repoint of an existing `BUCKET_BASE_URL` to a different host (one for one).
    """
    removed = [line for line in before.split("\n") if line not in after.split("\n")]
    added = [line for line in after.split("\n") if line not in before.split("\n")]
    if added != [NEW_BUCKET_LINE]:
        raise DriftError(f"unexpected delta: removed={removed!r} added={added!r}")
    from_hf = sorted(removed) == sorted([OLD_BUCKET_LINE, OLD_TOKEN_LINE])
    repoint = len(removed) == 1 and removed[0].startswith("BUCKET_BASE_URL = ")
    if not (from_hf or repoint):
        raise DriftError(f"unexpected delta: removed={removed!r} added={added!r}")


def _rewrite(src: Path, dst: Path, fetcher: bytes) -> tuple[int, int, int, int]:
    """Copies `src` to `dst`, rewriting the three files. Returns (tomls, pys, dockerfiles, total)."""
    tomls = pys = dockerfiles = total = 0
    with tarfile.open(src, "r:gz") as tin, tarfile.open(dst, "w:gz") as tout:
        for member in tin:
            total += 1
            is_toml = member.isfile() and member.name.endswith("/task.toml")
            is_py = member.isfile() and member.name.endswith("/pull_bucket.py")
            is_docker = member.isfile() and member.name.endswith("/environment/Dockerfile")
            if not (is_toml or is_py or is_docker):
                tout.addfile(member, tin.extractfile(member) if member.isfile() else None)
                continue

            if is_py:
                payload = fetcher
                pys += 1
            elif is_docker:
                before = tin.extractfile(member).read().decode("utf-8")
                try:
                    after = rewrite_dockerfile(before)
                except DriftError as exc:
                    raise DriftError(f"{member.name}: {exc}") from exc
                if after != before:
                    dockerfiles += 1
                payload = after.encode("utf-8")
            else:
                before = tin.extractfile(member).read().decode("utf-8")
                try:
                    after = rewrite_toml(before)
                except DriftError as exc:
                    raise DriftError(f"{member.name}: {exc}") from exc
                if after != before:
                    _assert_only_expected_lines_changed(before, after)
                    tomls += 1
                payload = after.encode("utf-8")

            # Copy the member so mode/mtime/ownership survive; only the size changes.
            out = tarfile.TarInfo(member.name)
            out.mode, out.mtime, out.uid, out.gid = member.mode, member.mtime, member.uid, member.gid
            out.uname, out.gname, out.type = member.uname, member.gname, member.type
            out.size = len(payload)
            tout.addfile(out, io.BytesIO(payload))
    return tomls, pys, total


def verify(tarball: Path, fetcher: bytes) -> int:
    """Checks every task points at the new store and carries the current fetcher verbatim."""
    bad_toml: list[str] = []
    stale_py: list[str] = []
    bad_docker: list[str] = []
    tomls = pys = dockerfiles = 0
    with tarfile.open(tarball, "r:gz") as tin:
        for member in tin:
            if not member.isfile():
                continue
            if member.name.endswith("/task.toml"):
                tomls += 1
                text = tin.extractfile(member).read().decode("utf-8")
                ok = (
                    NEW_BUCKET_LINE in text
                    and OLD_BUCKET_LINE not in text
                    and "HF_TOKEN" not in text
                    and all(key in text for key in KEEP_KEYS)
                )
                if not ok:
                    bad_toml.append(member.name)
            elif member.name.endswith("/pull_bucket.py"):
                pys += 1
                if tin.extractfile(member).read() != fetcher:
                    stale_py.append(member.name)
            elif member.name.endswith("/environment/Dockerfile"):
                dockerfiles += 1
                if COPY_LINE not in tin.extractfile(member).read().decode("utf-8"):
                    bad_docker.append(member.name)

    print(f"task.toml: {tomls}   pull_bucket.py: {pys}   environment/Dockerfile: {dockerfiles}")
    for label, names in (
        ("task.toml not repointed at the new store", bad_toml),
        ("pull_bucket.py differing from tools/pull_bucket.py", stale_py),
        ("Dockerfile not installing the fetcher", bad_docker),
    ):
        print(f"  {label}: {len(names)}")
        for name in names[:5]:
            print(f"    {name}")
    if bad_toml or stale_py or bad_docker or not tomls == pys == dockerfiles:
        return 1
    print(f"OK: every task fetches from {BASE_URL} with the vendored stdlib fetcher.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--verify", action="store_true", help="report drift, write nothing")
    group.add_argument("--write", action="store_true", help="rewrite the tarball in place")
    parser.add_argument("--tarball", type=Path, default=TARBALL)
    parser.add_argument("--fetcher", type=Path, default=FETCHER)
    args = parser.parse_args()

    for path in (args.tarball, args.fetcher):
        if not path.exists():
            print(f"not found: {path}", file=sys.stderr)
            return 1
    fetcher = args.fetcher.read_bytes()

    if args.verify:
        return verify(args.tarball, fetcher)

    tmp = args.tarball.with_suffix(".tar.gz.new")
    try:
        tomls, pys, total = _rewrite(args.tarball, tmp, fetcher)
    except DriftError as exc:
        tmp.unlink(missing_ok=True)
        print(f"aborted, tarball untouched: {exc}", file=sys.stderr)
        return 1
    # Round-trip the result before overwriting: a corrupt archive here breaks every rollout.
    with tarfile.open(tmp, "r:gz") as check:
        members = sum(1 for _ in check)
    if members != total:
        tmp.unlink()
        print(f"member count changed: {total} -> {members}", file=sys.stderr)
        return 1
    shutil.move(str(tmp), str(args.tarball))
    print(f"rewrote {tomls} task.toml and {pys} pull_bucket.py ({total} members preserved)")
    return verify(args.tarball, fetcher)


if __name__ == "__main__":
    raise SystemExit(main())
