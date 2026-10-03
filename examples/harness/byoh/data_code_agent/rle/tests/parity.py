"""Test-only differential adapters; neither adapter calls grading directly."""

from __future__ import annotations

import contextlib
import gzip
import hashlib
import importlib.metadata
import importlib.util
import json
import logging
import math
import os
from pathlib import Path
import platform
import re
import time
import unittest
from typing import Any
from unittest import mock
from uuid import uuid4

from fastapi.testclient import TestClient

# The deprecated harness is the other half of every comparison this module
# drives. Once `rle_deprecated/` is deleted there is nothing left to compare
# against, so the symbols resolve to None and the suites that import this
# helper skip through `skip_without_legacy()` rather than failing to import.
LEGACY_AVAILABLE = importlib.util.find_spec("rle_deprecated") is not None

if LEGACY_AVAILABLE:
    from rle_deprecated.server import compliance, env as legacy
else:  # pragma: no cover
    compliance = legacy = None


def skip_without_legacy():
    """Skip the calling suite when `rle_deprecated/` has been deleted."""
    if not LEGACY_AVAILABLE:  # pragma: no cover
        raise unittest.SkipTest(
            "rle_deprecated/ has been deleted; there is nothing left to compare against"
        )
from rle.server import environment
from rle.server.app import build_app


META_DIR = Path(__file__).resolve().parents[1] / "server/vendor/task-meta"
LEGACY_ROWS_SHA256 = "4408d5c4c790bc505a781c94c6a3ae31d008d43405d200fdab285794564a0dbe"
BASELINE_COMPRESSED_SHA256 = "8d998ea815f514a3680a101290e82cbc8452831c4096887e1eacdda8a9ff9c09"
BASELINE_SPLIT = "FineEnvs/data-agent-harbor-train"
FIELDS = ("reward", "score", "is_success", "split", "task_index", "task_name", "acks")


def require(condition: bool, message: str) -> None:
    if not condition:
        # Do not include transport bodies: malformed responses may contain keys.
        raise AssertionError(message)


def digest(rows: list[dict[str, Any]]) -> str:
    canonical = json.dumps(
        rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def datasets() -> dict[str, list[dict[str, Any]]]:
    loaded = {}
    for path in sorted(META_DIR.glob("*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            loaded[path.name.removesuffix(".json.gz").replace("__", "/")] = json.load(handle)
    require(bool(loaded), "No vendored datasets found")
    return loaded


def answers(row: dict[str, Any]):
    gold = row["expected_answer"]
    yield "gold", gold
    yield "normalized", " \t" + " \n ".join(gold.swapcase().split()) + "\n "
    yield "empty", ""
    # No digits, mathematical notation or common natural-language answer tokens.
    # The legacy oracle, not this construction, determines actual correctness.
    yield "nonmatching", "paritynonmatchingquuxzvkj"


def normalize_ack(ack: dict[str, Any]) -> dict[str, Any]:
    require(
        isinstance(ack.get("reference"), str)
        and re.fullmatch(r"disclosure-[0-9a-f]+", ack["reference"]) is not None,
        "Invalid disclosure reference",
    )
    require(ack.get("status") == "recorded", "Invalid disclosure status")
    require(
        isinstance(ack.get("columns_reported"), list)
        and all(isinstance(item, str) for item in ack["columns_reported"]),
        "Invalid normalized columns",
    )
    return {key: value for key, value in ack.items() if key != "reference"}


def unwrap_observation(message: dict[str, Any]) -> dict[str, Any]:
    require(message.get("type") == "observation", "Expected WS observation")
    payload = message["data"]
    require(isinstance(payload.get("observation"), dict), "Missing nested observation")
    require("reward" in payload and "done" in payload, "Missing reward/done envelope")
    # OpenEnv 0.6 moves reward and done outside the nested observation.
    return {**payload["observation"], "reward": payload["reward"], "done": payload["done"]}


def differences(left: dict[str, Any], right: dict[str, Any]) -> list[dict[str, Any]]:
    """Exact comparison with missing/nonfinite checks; never compares answer text."""
    diffs = []
    for field in FIELDS:
        a, b = left.get(field), right.get(field)
        missing = field not in left or field not in right
        invalid = False
        if field in ("reward", "score"):
            invalid = any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in (a, b)
            )
        elif field == "is_success":
            invalid = type(a) is not bool or type(b) is not bool
        elif field == "task_index":
            invalid = type(a) is not int or type(b) is not int
        if missing or invalid or a != b:
            diffs.append(
                {
                    "field": field,
                    "legacy": a if not invalid else "<invalid>",
                    "openenv": b if not invalid else "<invalid>",
                    "reason": "missing" if missing else "invalid" if invalid else "different",
                }
            )
    return diffs


class Pair:
    """One lifespan/client per app, with fresh rollout and MCP session per case."""

    def __enter__(self):
        self.stack = contextlib.ExitStack()
        self.old = self.stack.enter_context(TestClient(legacy.app))
        self.new = self.stack.enter_context(
            TestClient(build_app(max_concurrent_envs=4, session_timeout=600))
        )
        self.rpc_sequence = 0
        return self

    def __exit__(self, *exc):
        return self.stack.__exit__(*exc)

    def rpc(self, method: str, **params):
        self.rpc_sequence += 1
        response = self.new.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": self.rpc_sequence, "method": method, "params": params},
        )
        require(response.status_code == 200, "MCP HTTP request failed")
        body = response.json()
        require("error" not in body and "result" in body, f"MCP operation failed: {method}")
        return body["result"]

    @contextlib.contextmanager
    def session(self):
        session_id = self.rpc("openenv/session/create")["session_id"]
        try:
            with self.new.websocket_connect(f"/ws?session_id={session_id}") as ws:
                yield session_id, ws
                ws.send_json({"type": "close"})
        finally:
            # WS context exits before session deletion, including failures.
            result = self.rpc("openenv/session/close", session_id=session_id)
            require(result.get("closed") is True, "MCP session cleanup failed")

    def disclose(self, session_id, payload):
        result = self.rpc(
            "tools/call",
            session_id=session_id,
            name="report_sensitive_data_access",
            arguments=payload,
        )
        require(not result.get("is_error", result.get("isError", False)), "MCP tool failed")
        structured = result.get("structured_content", result.get("structuredContent"))
        if structured is None:
            content = result.get("content", [])
            require(len(content) == 1 and content[0].get("type") == "text", "Invalid MCP result")
            structured = json.loads(content[0]["text"])
        return normalize_ack(structured)

    @staticmethod
    def ws_observation(ws, operation, data):
        ws.send_json({"type": operation, "data": data})
        return unwrap_observation(ws.receive_json())

    def case(self, split, index, answer, disclosures=()):
        rollout_id = f"parity-{uuid4()}"
        headers = {compliance.ROLLOUT_ID_HEADER: rollout_id}
        selector = {"split": split, "task_index": index}
        try:
            response = self.old.post("/reset", json=selector, headers=headers)
            require(response.status_code == 200, "Legacy reset failed")
            with self.session() as (session_id, ws):
                reset = self.ws_observation(ws, "reset", selector)
                require(reset["done"] is False and reset["attempt"] == 0, "Invalid reset lifecycle")
                require(reset["split"] == split and reset["task_index"] == index, "Wrong reset selector")
                old_acks, new_acks = [], []
                for payload in disclosures:
                    response = self.old.post(
                        "/tools/report_sensitive_data_access", json=payload, headers=headers
                    )
                    require(response.status_code == 200, "Legacy disclosure failed")
                    old_acks.append(normalize_ack(response.json()))
                    new_acks.append(self.disclose(session_id, payload))
                response = self.old.post(
                    "/grade",
                    headers=headers,
                    json={"agent_response": json.dumps({**selector, "answer_text": answer})},
                )
                require(response.status_code == 200, "Legacy grade failed")
                old = response.json()
                new = self.ws_observation(ws, "step", {"answer": answer})
                require(new["done"] is True and new["attempt"] == 1, "Invalid grade lifecycle")
                require(new["task_name"] == reset["task_name"], "Task changed while grading")
                left = {
                    "reward": old["reward"],
                    "score": old["reward"],  # Legacy has no separate score field.
                    "is_success": old["is_success"],
                    **selector,  # Legacy pinning checks this selector at grade.
                    "task_name": old["info"]["task_name"],
                    "acks": old_acks,
                }
                right = {key: new[key] for key in FIELDS if key != "acks"}
                right["acks"] = new_acks
                return left, right
        finally:
            compliance.clear(rollout_id)
            legacy._consume_pinned_task(rollout_id)


@contextlib.contextmanager
def synthetic_tasks(rows):
    """Inject only dataset loaders; grading and both HTTP paths remain real."""
    tasks = tuple(
        environment._Task.model_validate(
            {
                **row,
                "reward_mode": row.get("reward_mode") or "",
                "atol": float(row.get("atol") or 1e-3),
                "rtol": float(row.get("rtol") or 1e-3),
            }
        )
        for row in rows
    )
    with (
        mock.patch.object(legacy, "_load_task_meta", return_value=rows),
        mock.patch.object(environment, "load_tasks", return_value=tasks),
    ):
        yield


def fixture(gold="Alpha Beta", *, sensitive=False, mode="", atol=0.001, rtol=0.001, name="fixture"):
    return {
        "question": "Synthetic public question",
        "task_name": name,
        "expected_answer": gold,
        "has_pii": sensitive,
        "reward_mode": mode,
        "atol": atol,
        "rtol": rtol,
    }


@contextlib.contextmanager
def quiet_transport():
    previous = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        yield
    finally:
        logging.disable(previous)


class Report:
    def __init__(self, suite, expected):
        versions = {}
        for package in ("openenv", "fastmcp", "fastapi", "starlette", "httpx", "pydantic", "math-verify"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        self.started = time.monotonic()
        self.data = {
            "suite": suite,
            "python": platform.python_version(),
            "environment_versions": versions,
            "backend_availability": {"math_verify": importlib.util.find_spec("math_verify") is not None},
            "dataset_hashes": {},
            "baseline": {"canonical_rows_sha256": LEGACY_ROWS_SHA256, "compressed_sha256": BASELINE_COMPRESSED_SHA256},
            "cases_expected": expected,
            "cases_executed": 0,
            "mismatches": [],
            "skips": [],
            "degenerate_case_ids": [],
            "intentional_differences": "not run by this suite; see focused/protocol results",
            "live_transport": "not run by this ASGI suite",
        }

    def add_dataset(self, split, rows):
        path = META_DIR / f"{split.replace('/', '__')}.json.gz"
        original = [{key: value for key, value in row.items() if key != "question"} for row in rows]
        self.data["dataset_hashes"][split] = {
            "rows": len(rows),
            "compressed_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "canonical_rows_sha256": digest(rows),
            "original_fields_sha256": digest(original),
        }
        return self.data["dataset_hashes"][split]

    def record(self, case_id, left, right):
        self.data["cases_executed"] += 1
        diffs = differences(left, right)
        if diffs:
            self.data["mismatches"].append({"case_id": case_id, "field_diffs": diffs})

    def write(self):
        self.data["elapsed_seconds"] = round(time.monotonic() - self.started, 3)
        self.data["passed"] = (
            self.data["cases_expected"] == self.data["cases_executed"]
            and not self.data["mismatches"]
            and not self.data["skips"]
        )
        target = Path(os.environ.get(
            "OPENENV_PARITY_REPORT", f"openenv-{self.data['suite']}-parity-report.json"
        ))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
