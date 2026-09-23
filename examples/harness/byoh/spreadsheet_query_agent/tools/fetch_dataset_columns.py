"""Fetches the real column names of every dataset behind the task suite.

`pii_taxonomy.py` labels a task by keyword-matching the Kaggle slug, the question text and the
file names, because when it was written the CSVs lived on Hugging Face and were pulled only at
container start -- so the columns did not exist at labelling time. They do now: the suite's inputs
are served from our own CDN, and an HTTP range request over the first few KB of a file is enough
to read its header row.

That matters because slug matching is wrong in both directions, and measurably so. Adjudicating a
20-task sample against real headers put 15% of labels wrong: `pankrzysiu__weather-archive-jena` is
labelled `health` (it is temperature and air pressure), `tmdb__tmdb-movie-metadata` is labelled
`financial` (it is box-office budget and revenue, which is corporate, not personal), and
`uciml__sms-spam-collection-dataset` is labelled clean despite a column of raw SMS message text.
Columns separate those cases; a slug cannot.

Only 471 distinct datasets sit behind the 5000 tasks, so this is one cheap pass, and the result is
cached in `vendor/dataset-columns.json` for `build_task_meta.py` to label from.

What this deliberately does not do is download whole files. The suite is 216MB extracted and some
single datasets run to 844MB; a header row is a few hundred bytes. Servers that ignore `Range` are
detected by checking for `206`, and the read is capped either way.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INDEX = ROOT / "agent" / "vendor" / "task-index.json.gz"
DEFAULT_OUT = ROOT / "tools" / "vendor" / "dataset-columns.json"

HEADER_BYTES = 8192
TIMEOUT_SEC = 60
# Reading every file of a 30-file dataset buys nothing: the sensitive columns that decide a label
# are in the tables the question is asked about, and those are the ones listed first.
MAX_FILES = 6
TABULAR_SUFFIXES = (".csv", ".tsv", ".txt", ".data")


def _fetch_head(url: str) -> tuple[str, bool]:
    """Returns the first `HEADER_BYTES` of `url` and whether the server honoured the range."""
    req = urllib.request.Request(url, headers={"Range": f"bytes=0-{HEADER_BYTES - 1}"})
    with urllib.request.urlopen(req, timeout=TIMEOUT_SEC) as response:
        raw = response.read(HEADER_BYTES)
        return raw.decode("utf-8", "replace"), response.status == 206


def _header_row(text: str) -> list[str]:
    """Parses the first line as a delimited header, sniffing the delimiter.

    Falls back to a comma rather than raising: a mis-sniffed delimiter yields one long pseudo-column,
    which still carries the words a keyword pass needs, whereas an exception loses the dataset.
    """
    first = text.splitlines()[0] if text.splitlines() else ""
    if not first:
        return []
    try:
        delimiter = csv.Sniffer().sniff(first, delimiters=",;\t|").delimiter
    except csv.Error:
        delimiter = ","
    row = next(csv.reader(io.StringIO(first), delimiter=delimiter), [])
    return [c.strip().strip('"').strip()[:64] for c in row]


def fetch_dataset(base_url: str, prefix: str) -> dict:
    # The prefix is a Kaggle slug and is usually URL-safe, but not always
    # ("kkanda__communities and crime unnormalized data set"), so quote it like any other segment.
    root = f"{base_url.rstrip('/')}/{urllib.parse.quote(prefix.strip('/'))}"
    out: dict = {"prefix": prefix, "files": {}, "skipped": [], "error": None}
    try:
        with urllib.request.urlopen(f"{root}/_manifest.txt", timeout=TIMEOUT_SEC) as response:
            manifest = response.read().decode("utf-8", "replace")
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        out["error"] = f"manifest: {exc}"
        return out

    names = sorted({line.strip() for line in manifest.splitlines() if line.strip()})
    tabular = [n for n in names if n.lower().endswith(TABULAR_SUFFIXES)]
    out["skipped"] = [n for n in names if n not in tabular]
    for name in tabular[:MAX_FILES]:
        # Manifest names are raw file names, and plenty contain spaces ("IMDB Horror movies.csv").
        # urllib rejects those outright, so quote the segment rather than trusting it.
        url = f"{root}/{urllib.parse.quote(name)}"
        try:
            text, ranged = _fetch_head(url)
        except Exception as exc:  # noqa: BLE001 - one unreadable file must not lose the dataset
            out["files"][name] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        out["files"][name] = {"columns": _header_row(text), "ranged": ranged}
    return out


def _fetch_dataset_safe(base_url: str, prefix: str) -> dict:
    """Never raises: `ThreadPoolExecutor.map` aborts the whole pass on the first exception, and a
    471-dataset run is too expensive to lose to one malformed name."""
    try:
        return fetch_dataset(base_url, prefix)
    except Exception as exc:  # noqa: BLE001
        return {"prefix": prefix, "files": {}, "skipped": [], "error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--limit", type=int, default=0, help="only fetch the first N datasets")
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT / "harness"))
    import gzip

    rows = json.loads(gzip.decompress(args.index.read_bytes()))
    rows = rows["tasks"] if isinstance(rows, dict) else rows
    datasets = sorted({(r["bucket_base_url"], r["bucket_prefix"]) for r in rows})
    if args.limit:
        datasets = datasets[: args.limit]

    # Resume rather than restart: a partial run costs 471 round trips to redo, and the CDN is the
    # slow part. Anything already recorded without an error is kept.
    done: dict = {}
    if args.out.exists():
        done = {k: v for k, v in json.loads(args.out.read_text()).items() if not v.get("error")}
    todo = [d for d in datasets if d[1] not in done]
    print(f"{len(datasets)} datasets, {len(done)} cached, {len(todo)} to fetch", flush=True)

    results = dict(done)
    if todo:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for i, res in enumerate(pool.map(lambda d: _fetch_dataset_safe(*d), todo), 1):
                results[res["prefix"]] = res
                if i % 25 == 0 or i == len(todo):
                    print(f"  {i}/{len(todo)}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1, sort_keys=True) + "\n")

    failed = [k for k, v in results.items() if v.get("error")]
    empty = [k for k, v in results.items() if not v.get("error") and not any(
        f.get("columns") for f in v["files"].values())]
    print(f"\nwrote {args.out} ({len(results)} datasets)")
    print(f"  failed manifests: {len(failed)}")
    print(f"  no tabular header: {len(empty)}")
    for k in (failed + empty)[:10]:
        print(f"    {k}: {results[k].get('error') or 'no columns'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
