"""Tests for keeping RLE's poll loop out of this harness's logs.

RLE polls `/invoke/rollouts/{operation_id}` every `RETRY_AFTER_MS`, so a two-minute rollout
produces hundreds of identical access-log lines. On a shared log pane that buries everything the
rollout itself reports, which is the only reason those logs are being read.

These tests pin the two halves of the fix: the access-log line for a poll is dropped, and the
polls that carry information -- the one holding the result, and one naming a rollout this process
does not have -- are logged exactly once each by the handler instead.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from app import (  # noqa: E402
    app,
    PollAccessNoiseFilter,
    ROLLOUTS,
    RolloutStore,
)

from tests.test_byoh_contract import MASTER_CONTEXT  # noqa: E402


def _access_record(method: str, path: str, status: int) -> logging.LogRecord:
    """Builds the record uvicorn emits for one request.

    Uvicorn logs `'%s - "%s %s HTTP/%s" %d'` with
    `(client_addr, method, full_path, http_version, status_code)`, so a test that invented its own
    shape would pass while the filter never matched anything in production.
    """
    return logging.LogRecord(
        name="uvicorn.access",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg='%s - "%s %s HTTP/%s" %d',
        args=("127.0.0.1:1234", method, path, "1.1", status),
        exc_info=None,
    )


def test_the_poll_access_line_is_dropped():
    noise = _access_record("GET", "/invoke/rollouts/op-1", 200)
    assert PollAccessNoiseFilter().filter(noise) is False


def test_every_other_access_line_survives():
    """The filter has to be surgical: silencing the whole access log would hide the dispatch."""
    keep = [
        _access_record("POST", "/invoke", 202),
        _access_record("GET", "/health", 200),
        _access_record("DELETE", "/invoke/rollouts/op-1", 204),
    ]
    assert [PollAccessNoiseFilter().filter(r) for r in keep] == [True, True, True]


def test_a_record_shaped_differently_is_left_alone():
    """If uvicorn ever changes its args, the access log must go quiet-by-accident nowhere."""
    odd = logging.LogRecord(
        name="uvicorn.access",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="GET /invoke/rollouts/op-1 200",
        args=None,
        exc_info=None,
    )
    assert PollAccessNoiseFilter().filter(odd) is True


def test_the_filter_is_installed_on_the_access_logger():
    """Asserting on the class alone would pass even if nothing ever attached it."""
    installed = logging.getLogger("uvicorn.access").filters
    assert any(isinstance(f, PollAccessNoiseFilter) for f in installed)


def test_the_harness_logger_can_actually_emit_info():
    """`logger.info` reaches nothing on the default config -- `lastResort` is fixed at WARNING."""
    harness = logging.getLogger("byoh-data-code-agent")
    assert harness.isEnabledFor(logging.INFO)
    assert harness.handlers


def test_an_outcome_is_claimed_once_per_kind():
    store = RolloutStore()
    assert store.claim_report("op-1", "outcome") is True
    assert store.claim_report("op-1", "outcome") is False
    # A different kind, and a different rollout, are their own claims.
    assert store.claim_report("op-1", "unknown") is True
    assert store.claim_report("op-2", "outcome") is True


def test_withdrawing_releases_the_claim():
    """An operation id RLE re-uses after a withdraw has to be able to report again."""
    store = RolloutStore()
    store.claim_report("op-1", "outcome")
    store.withdraw("op-1")
    assert store.claim_report("op-1", "outcome") is True


def test_the_running_polls_are_silent_and_the_result_is_logged_once(monkeypatch, caplog):
    async def succeeds(rollout_id, rollout_context, agent_input):
        return "Web Development"

    monkeypatch.setattr("app.run_harness_rollout", succeeds)
    with caplog.at_level(logging.INFO, logger="byoh-data-code-agent"):
        with TestClient(app) as client:
            client.post(
                "/invoke",
                json={
                    "rollout_id": "rollout-noise",
                    "operation_id": "operation-noise",
                    "agent_input": {"task_index": 0},
                    "rollout_context": MASTER_CONTEXT,
                },
            )
            for _ in range(200):
                body = client.get("/invoke/rollouts/operation-noise").json()
                if body["status"] != "running":
                    break
            assert body["status"] == "succeeded"
            # Keep polling after the result, the way a client that has not stopped yet would.
            for _ in range(5):
                client.get("/invoke/rollouts/operation-noise")

    delivered = [r for r in caplog.records if "result delivered" in r.getMessage()]
    assert len(delivered) == 1
    assert "succeeded" in delivered[0].getMessage()


def test_an_unknown_rollout_is_reported_once_however_often_it_is_polled(caplog):
    with caplog.at_level(logging.INFO, logger="byoh-data-code-agent"):
        with TestClient(app) as client:
            for _ in range(5):
                assert client.get("/invoke/rollouts/operation-missing").status_code == 404

    unknown = [r for r in caplog.records if "unknown rollout" in r.getMessage()]
    assert len(unknown) == 1


def teardown_function() -> None:
    ROLLOUTS.__init__()  # noqa: PLC2801 - the store is module state shared across tests
