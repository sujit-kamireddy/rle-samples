"""Proves a locally running RLE container grades the way it should.

From this sample's root, start the container first::

    cd rle && docker build -t competitive-intelligence-rle:latest .
    docker run --rm -p 8000:8000 competitive-intelligence-rle:latest

then, from the sibling `_internal/` copy of this sample (see its own
``tools/``)::

    cd ../_internal/competitive_intelligence_agent
    pip install -r ../../competitive_intelligence_agent/rle/requirements.txt
    python tools/smoke_grade.py

The install step pulls in the same MCP client library (``openenv``) and
``RLEnvironment``/``GradeAction`` types (``azure-ai-projects[rle]``) that the
container itself depends on, since this script is a real MCP client rather
than a bare HTTP caller.

This is worth running before a training job, because the failure it catches is
expensive and quiet. An environment that answers ``/health`` and returns a
well-formed reward can still be grading against the wrong thing, and RL will
happily optimise whatever it is actually measuring. A 15-hour run is a slow way
to discover that.

For each variant in the evaluation set it opens one MCP session, resets a
rollout, then grades three briefs that differ only in their verdict: material,
immaterial, and abstain. It asserts that the verdict matching the task's
answer key earns the highest reward of the three. It also checks the rubric's
deliberate asymmetry: where a verdict was available, abstaining scores above
asserting the wrong one, because over-caution is a smaller failure than
fabrication.

No model is involved. The briefs are synthetic, so this measures the
environment and nothing else.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from azure.ai.projects.rle.environments import GradeAction
from openenv.core.env_client import StepResult
from openenv.core.mcp_client import GenericMCPObservation, MCPToolClient

# This file was relocated to `_internal/` (see `../rle/tests/conftest.py`), so
# its own directory is no longer inside the sample that ships `job_data/`.
# Reach the real sample root the same way those tests do, rather than
# deriving it from this file's own (now relocated) position.
SAMPLE_ROOT = Path(__file__).resolve().parents[3] / "competitive_intelligence_agent"
VERDICTS = {"material": True, "immaterial": False, "abstain": None}


class SmokeTestClient(MCPToolClient):
    """An ``MCPToolClient`` with observation parsing fixed for this environment.

    The base client's ``_parse_result`` sniffs the observation payload for a
    ``tools`` key to decide whether it is a tool listing rather than a plain
    observation. ``CompetitiveIntelObservation`` also happens to have a
    ``tools: list[str]`` field -- the surface the rollout used, not a tool
    listing -- so the sniff always misfires here, and the base client raises
    trying to parse each tool name as a tool description. This script never
    calls ``list_tools``, so every observation it receives really is a
    ``CompetitiveIntelObservation``; this always takes the generic passthrough
    branch, which is what the base client would do anyway were the field
    named anything else.
    """

    def _parse_result(self, payload: dict[str, Any]) -> StepResult:
        obs_data = payload.get("observation", {})
        custom_fields = {
            key: value for key, value in obs_data.items()
            if key not in {"done", "reward", "metadata"}
        }
        observation = GenericMCPObservation(
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=payload.get("metadata", obs_data.get("metadata", {})),
            **custom_fields,
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
            metadata=payload.get("metadata"),
        )


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


async def grade_variant(base: str, task: dict) -> dict[str, dict]:
    """Resets one rollout and grades all three candidate verdicts against it.

    One MCP session covers the reset and every grade call below, so they
    share the one live environment instance a real rollout would -- unlike a
    bare HTTP ``/reset`` or ``/step`` call, which each get a fresh environment
    of their own and never see each other's state.
    """
    scores = {}
    infos = {}
    async with SmokeTestClient(base_url=base) as env:
        await env.reset(**task)
        for label, value in VERDICTS.items():
            result = await env.step(GradeAction(answer=brief(value)))
            scores[label] = result.reward
            infos[label] = result.observation.info
    return {"scores": scores, "infos": infos}


async def run(base: str, eval_path: Path) -> int:
    rows = [json.loads(line) for line in eval_path.read_text().splitlines() if line.strip()]
    by_variant: dict[str, dict] = {}
    for row in rows:
        by_variant.setdefault(row["task"]["variant"], row)

    print(f"{'variant':<18}{'expected':<12}{'said':<12}{'reward':>8}{'verdict':>9}")
    failures = []

    for variant, row in sorted(by_variant.items()):
        task = row["task"]
        want = expected_verdict(task)

        graded = await grade_variant(base, task)
        scores = graded["scores"]
        for label in VERDICTS:
            exact = graded["infos"][label]["metrics"]["verdict_exact"]
            flag = "  <- answer key" if label == want else ""
            print(f"{variant:<18}{want:<12}{label:<12}"
                  f"{scores[label]:>8.3f}{exact:>9.0f}{flag}")

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--eval", dest="eval_", type=Path,
                        default=SAMPLE_ROOT / "job_data" / "validation.jsonl")
    args = parser.parse_args()

    try:
        urllib.request.urlopen(args.base + "/health", timeout=10)
    except (urllib.error.URLError, OSError) as error:
        print(f"no RLE container at {args.base}: {error}")
        print("start one with: cd rle && docker run --rm -p 8000:8000 <image>")
        return 2

    return asyncio.run(run(args.base, args.eval_))


if __name__ == "__main__":
    sys.exit(main())
