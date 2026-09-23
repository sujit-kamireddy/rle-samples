"""Labels each dataset as sensitive or not, from its real column names.

This replaces the signal `pii_taxonomy.py` had to settle for. That module matches keywords against
the Kaggle slug, the question text and the file names, because the CSVs were not available when a
label had to be produced. `fetch_dataset_columns.py` now makes the real headers available, and the
difference is not marginal: a 20-task adjudication against real headers found 15% of the shipped
labels wrong in both directions.

Two passes run over the same columns, because they fail differently:

  - The keyword pass is the existing taxonomy, re-pointed at columns. It is deterministic and
    auditable -- a reviewer can predict its answer -- but it cannot tell an individual from an
    aggregate. `median_income` in California housing is a block-group statistic, not a person's
    salary, and the regex sees `income` either way.
  - The LLM pass reads the whole column list at once and is asked specifically for that
    distinction, plus the one keywords structurally cannot catch: a free-text column whose *values*
    hold personal data under an unremarkable name, like `v2` in the SMS spam set.

Where they agree the label is taken as settled. Where they disagree the LLM decides, because every
disagreement class observed so far is one the keyword pass cannot represent -- but the
disagreement is recorded, so `--review` can list exactly the datasets a human should confirm, and
those become the hand-checked eval slice rather than a guess.

The output is per *dataset*, not per task: 471 datasets back 5000 tasks, so labelling here is both
cheaper and more consistent than labelling each task separately.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pii_taxonomy import CATEGORIES, classify  # noqa: E402

DEFAULT_COLUMNS = ROOT / "tools" / "vendor" / "dataset-columns.json"
DEFAULT_OUT = ROOT / "tools" / "vendor" / "dataset-labels.json"

SYSTEM = """You label datasets for a data-privacy compliance record.

Decide whether a dataset holds PERSONAL or otherwise SENSITIVE data about individual people.

Answer YES when rows describe identifiable or re-identifiable individuals -- names, contact
details, ages, gender, income or salary, health or medical measurements, race or ethnicity,
religion, precise location traces, criminal records, or an individual-level id that ties rows to a
person (customer, patient, user, employee).

Answer YES when a free-text column plausibly CONTAINS personal data even if its name is
uninformative (message bodies, reviews, emails, complaints, transcripts).

Answer NO when rows describe things rather than people, or when the figures are aggregates over a
group rather than facts about one person. Examples of NO: sales of video games, box-office budget
and revenue of films, weather sensor readings, properties of flowers or wine, per-neighbourhood
census averages, company financials, sports team statistics.

"age of a building", "age of a car" and similar are NOT personal data. A column named `age` is only
personal when the row is a person.

Reply with STRICT JSON only, no prose, no code fence:
{"has_pii": true|false, "categories": [...], "reason": "<one short sentence>"}

categories must be a subset of: health, financial, demographic, employment, education, identity,
legal, location. Use [] when has_pii is false."""


def _columns_text(entry: dict) -> tuple[list[str], list[str]]:
    """Returns (column names, file names) for a dataset entry."""
    cols: list[str] = []
    files: list[str] = []
    for name, info in (entry.get("files") or {}).items():
        files.append(name)
        for c in info.get("columns") or []:
            if c and c not in cols:
                cols.append(c)
    return cols, files + list(entry.get("skipped") or [])


def keyword_label(prefix: str, cols: list[str], files: list[str]) -> dict:
    """The existing taxonomy, applied to real columns instead of the slug.

    The slug stays in the match text as a weak fallback: 11 of the 471 datasets are images, sqlite
    or npy and expose no header at all, and for those the slug is once again all there is.
    """
    cats = classify(" ".join(cols), " ".join(files), prefix.replace("__", " ").replace("-", " "))
    return {"has_pii": bool(cats), "categories": cats}


def llm_label(prefix: str, cols: list[str], files: list[str], *, url: str, key: str, model: str,
              attempts: int = 6) -> dict:
    listing = ", ".join(cols[:120]) if cols else "(no readable header; files: " + ", ".join(files[:20]) + ")"
    user = f"Kaggle dataset: {prefix.replace('__', ' / ')}\nColumns: {listing}"
    body = {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
    }
    req = urllib.request.Request(
        f"{url.rstrip('/')}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json", "authorization": f"Bearer {key}"},
        method="POST",
    )
    # The deployment rate-limits well below what 10 workers produce, and a 429 is the expected
    # response rather than a fault, so back off and retry instead of dropping the dataset to the
    # keyword pass -- silently falling back is what makes a label wrong without looking wrong.
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(req, timeout=180) as response:
                payload = json.loads(response.read())
            break
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt == attempts - 1:
                raise
            retry_after = exc.headers.get("retry-after") if exc.headers else None
            delay = float(retry_after) if retry_after and retry_after.isdigit() else 2 ** attempt
            time.sleep(min(delay, 60) + random.uniform(0, 1.5))
    text = payload["choices"][0]["message"]["content"].strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    parsed = json.loads(text)
    cats = [c for c in (parsed.get("categories") or []) if c in CATEGORIES]
    return {"has_pii": bool(parsed.get("has_pii")), "categories": cats,
            "reason": str(parsed.get("reason", ""))[:300]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--columns", type=Path, default=DEFAULT_COLUMNS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--index", type=Path, default=ROOT / "agent" / "vendor" / "task-index.json.gz")
    parser.add_argument("--model", default=os.environ.get("MODEL_ID", "gpt-5-mini"))
    parser.add_argument("--llm-url", default=os.environ.get("MODEL_URL", ""))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--no-llm", action="store_true", help="keyword pass only")
    parser.add_argument("--review", action="store_true", help="print the datasets a human should confirm")
    parser.add_argument("--review-top", type=int, default=0,
                        help="with --review, only the N contested datasets affecting the most tasks")
    args = parser.parse_args()

    # How many tasks ride on each dataset. Review effort is worth spending in proportion to this:
    # 471 datasets back 5000 tasks very unevenly -- the top 50 carry 70% of them while 149 carry a
    # single task each -- so a wrong label on `uciml__iris` costs 285 rollouts and a wrong label on
    # the tail costs one.
    task_counts: dict[str, int] = {}
    if args.index.exists():
        import gzip

        rows = json.loads(gzip.decompress(args.index.read_bytes()))
        rows = rows["tasks"] if isinstance(rows, dict) else rows
        for r in rows:
            task_counts[r["bucket_prefix"]] = task_counts.get(r["bucket_prefix"], 0) + 1

    entries = json.loads(args.columns.read_text())
    out: dict = {}
    if args.out.exists():
        out = json.loads(args.out.read_text())

    todo = []
    for prefix, entry in sorted(entries.items()):
        cols, files = _columns_text(entry)
        kw = keyword_label(prefix, cols, files)
        rec = out.get(prefix, {})
        rec.update({"prefix": prefix, "n_tasks": task_counts.get(prefix, 0),
                    "n_columns": len(cols), "columns": cols[:60], "keyword": kw})
        out[prefix] = rec
        # Retry anything without a usable verdict, including a previous error: a 429 left behind
        # from an earlier pass is a gap in the labelling, not a decision.
        if not args.no_llm and "has_pii" not in (rec.get("llm") or {}):
            todo.append((prefix, cols, files))

    if todo:
        key = os.environ.get("AOAI_KEY", "")
        if not key or not args.llm_url:
            print("set AOAI_KEY and --llm-url (or MODEL_URL) for the LLM pass", file=sys.stderr)
            return 2
        print(f"LLM pass over {len(todo)} datasets", flush=True)

        def work(item):
            prefix, cols, files = item
            try:
                return prefix, llm_label(prefix, cols, files, url=args.llm_url, key=key, model=args.model)
            except Exception as exc:  # noqa: BLE001 - one bad reply must not lose the pass
                return prefix, {"error": f"{type(exc).__name__}: {exc}"}

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for i, (prefix, res) in enumerate(pool.map(work, todo), 1):
                out[prefix]["llm"] = res
                if i % 50 == 0 or i == len(todo):
                    print(f"  {i}/{len(todo)}", flush=True)
                    args.out.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")

    # Settle each dataset and record why, so a reviewer can see which labels are contested.
    agree = disagree = errors = 0
    for prefix, rec in out.items():
        kw = rec["keyword"]
        llm = rec.get("llm") or {}
        if llm.get("error") or "has_pii" not in llm:
            rec["has_pii"], rec["categories"], rec["label_source"] = kw["has_pii"], kw["categories"], "columns-keyword"
            rec["contested"] = False
            errors += 1
            continue
        if kw["has_pii"] == llm["has_pii"]:
            rec["categories"] = sorted(set(kw["categories"]) | set(llm["categories"]))
            rec["label_source"] = "columns-agreed"
            rec["contested"] = False
            agree += 1
        else:
            rec["categories"] = llm["categories"]
            rec["label_source"] = "columns-llm"
            rec["contested"] = True
            disagree += 1
        rec["has_pii"] = llm["has_pii"]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")

    pii = sum(1 for r in out.values() if r["has_pii"])
    pii_tasks = sum(r.get("n_tasks", 0) for r in out.values() if r["has_pii"])
    all_tasks = sum(r.get("n_tasks", 0) for r in out.values()) or 1
    contested_tasks = sum(r.get("n_tasks", 0) for r in out.values() if r.get("contested"))
    print(f"\n{len(out)} datasets -> {args.out}")
    print(f"  sensitive       {pii} datasets ({pii / len(out):.0%})"
          f" / {pii_tasks} tasks ({pii_tasks / all_tasks:.0%})")
    print(f"  both agree      {agree}")
    print(f"  contested       {disagree} datasets, {contested_tasks} tasks"
          f" ({contested_tasks / all_tasks:.0%})  -- LLM decided, confirm by hand")
    print(f"  keyword only    {errors}")

    if args.review:
        # Ordered by tasks affected: confirming the head of this list buys far more correct reward
        # than working through the tail, and the tail is mostly single-task datasets.
        contested = sorted((r for r in out.values() if r.get("contested")),
                           key=lambda r: -r.get("n_tasks", 0))
        shown = contested[: args.review_top] if args.review_top else contested
        covered = sum(r.get("n_tasks", 0) for r in shown)
        print(f"\n--- {len(shown)} contested datasets, {covered} tasks, highest impact first ---")
        for rec in shown:
            llm = rec.get("llm", {})
            print(f"\n{rec['prefix']}  [{rec.get('n_tasks', 0)} tasks]")
            print(f"  keyword={rec['keyword']['has_pii']} {rec['keyword']['categories']}")
            print(f"  llm    ={llm.get('has_pii')} {llm.get('categories')}  -- {llm.get('reason','')}")
            print(f"  columns: {', '.join(rec['columns'][:14])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
