"""Proves a locally running RLE container grades the way it should.

Start the container first::

    cd rle && docker build -t competitive-intelligence-rle:latest .
    docker run --rm -p 8000:8000 competitive-intelligence-rle:latest

then, from the sample root::

    python tools/smoke_grade.py

This is worth running before a training job, because the failure it catches is
expensive and quiet. An environment that answers ``/health`` and returns a
well-formed reward can still be grading against the wrong thing, and RL will
happily optimise whatever it is actually measuring. A 15-hour run is a slow way
to discover that.

For each variant in the evaluation set it resets a rollout, then grades three
briefs that differ only in their verdict: material, immaterial, and abstain. It
asserts that the verdict matching the task's answer key earns the highest
reward of the three. It also checks the rubric's deliberate asymmetry: where a
verdict was available, abstaining scores above asserting the wrong one, because
over-caution is a smaller failure than fabrication.

No model is involved. The briefs are synthetic, so this measures the
environment and nothing else.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VERDICTS = {"material": True, "immaterial": False, "abstain": None}


def post(base: str, path: str, body: dict, headers: dict | None = None) -> dict:
    request = urllib.request.Request(
        base + path,
        json.dumps(body).encode(),
        {"Content-Type": "application/json", **(headers or {})},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read())


def brief(material: bool | None) -> str:
    """A syntactically valid brief carrying the given verdict.

    The grader reads the last fenced JSON object in the final assistant
    message, so the prose around it is irrelevant but the fence is not.
    """
    decision = {
        "material": material,
        "confidence": "medium",
        "company": "Example Corp",
        "internal_product": "Example Product",
        "exposure_score": 0.5,
        "citations": ["https://example.com/a"],
        "stakeholders": ["product lead"],
        "caveats": [],
        "next_actions": ["brief the team"],
    }
    return "Summary of findings.\n\n```json\n" + json.dumps(decision, indent=2) + "\n```\n"


def expected_verdict(task: dict) -> str:
    if task.get("expects_abstain") or task.get("material") is None:
        return "abstain"
    return "material" if task.get("material") else "immaterial"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--eval", dest="eval_", type=Path,
                        default=ROOT / "job_data" / "validation.jsonl")
    args = parser.parse_args()

    try:
        urllib.request.urlopen(args.base + "/health", timeout=10)
    except (urllib.error.URLError, OSError) as error:
        print(f"no RLE container at {args.base}: {error}")
        print("start one with: cd rle && docker run --rm -p 8000:8000 <image>")
        return 2

    rows = [json.loads(line) for line in args.eval_.read_text().splitlines() if line.strip()]
    by_variant: dict[str, dict] = {}
    for row in rows:
        by_variant.setdefault(row["task"]["variant"], row)

    print(f"{'variant':<18}{'expected':<12}{'said':<12}{'reward':>8}{'verdict':>9}")
    failures = []

    for variant, row in sorted(by_variant.items()):
        task = row["task"]
        rollout_id = f"smoke-{variant}"
        post(args.base, "/reset", task, {"x-rle-rollout-id": rollout_id})
        want = expected_verdict(task)

        scores = {}
        for label, value in VERDICTS.items():
            result = post(args.base, "/grade", {
                "rollout": {"rollout_id": rollout_id},
                "agent_response": brief(value),
            })
            scores[label] = result["reward"]
            exact = result["info"]["metrics"]["verdict_exact"]
            flag = "  <- answer key" if label == want else ""
            print(f"{variant:<18}{want:<12}{label:<12}"
                  f"{result['reward']:>8.3f}{exact:>9.0f}{flag}")

        best = max(scores, key=scores.get)
        if best != want:
            failures.append(
                f"{variant}: '{best}' outscored the answer key '{want}' "
                f"({scores[best]:.3f} vs {scores[want]:.3f})")
        if want != "abstain" and scores["abstain"] <= min(
                v for k, v in scores.items() if k not in (want, "abstain")):
            failures.append(
                f"{variant}: abstaining ({scores['abstain']:.3f}) did not beat "
                f"the wrong verdict, so over-caution is not being credited")
        print()

    if failures:
        print("FAILED")
        for failure in failures:
            print("  " + failure)
        return 1

    print(f"OK: the answer key scored highest on all {len(by_variant)} variants")
    return 0


if __name__ == "__main__":
    sys.exit(main())
