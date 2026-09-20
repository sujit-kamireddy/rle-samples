"""
Code grading for the ``code_rl`` environment.

Grades a submission in a fresh, disposable child process per call -- not
in-process, and not a persistent worker pool. ``lcb_utils.TEST_UTIL`` is
source code (not an importable module: it's designed to be written out as
``testing_util.py`` and run by a throwaway subprocess), so it's ``exec()``'d
in the child (see ``_grade_in_subprocess``) to get a live ``run_test`` we
call directly. That ``run_test`` executes the submission itself via
``exec()``, applies ``reliability_guard()`` (neuters dangerous builtins --
``os.system``, ``subprocess.Popen``, ``shutil.rmtree``, ...) first, and
enforces its own per-test timeout with ``signal.alarm``.

``reliability_guard()``'s edits land on the real, shared ``os``/``sys``/
``shutil``/``subprocess`` modules and are never undone -- safe only in a
process that exits right after, which is exactly what the child here is.
Running it in-process in the long-lived server (as this module used to do,
with the guard disabled to compensate) would have been safe *only* if every
episode got its own fresh container for its one and only submission -- but
Foundry RLE instances are pooled and reused across many episodes/rollouts,
so a submission that
crashes, wedges, or (with the guard disabled) tampers with the process
could affect every later episode that instance goes on to serve, not just
its own. Isolating each submission in its own child process -- torn down
(and forcibly killed if it overruns a hard wall-clock ceiling, since
``signal.alarm`` can't interrupt something the guard blocks from unwinding
cleanly) after every single grading call -- restores real per-submission
isolation regardless of instance reuse, and lets ``reliability_guard()`` run
for real again.
"""

from __future__ import annotations

import asyncio
import json
import logging
import multiprocessing as mp
import re
import types
from typing import Any

from .lcb_utils import TEST_UTIL

logger = logging.getLogger(__name__)

# Per-test timeout (run_test's own signal.alarm) is enforced inside the
# child; this is the *parent's* backstop in case the child never reports
# back at all (killed guard-blocked syscall, native crash mid-write, ...).
# Comfortably above what a well-behaved run should ever take.
_HARD_TIMEOUT_GRACE_S = 10.0


def _grade_in_subprocess(
    test_cases: dict[str, Any],
    generation: str,
    timeout: int,
    overall_timeout: float | None,
    result_queue: "mp.Queue[tuple[str, Any, dict[str, Any]]]",
) -> None:
    """Entry point for the disposable child process (see module docstring).

    `TEST_UTIL` is `exec()`'d fresh here, in the child, every call -- unlike
    the old in-process approach, there is no shared, long-lived namespace
    for `reliability_guard()`'s changes to leak out of, so it runs
    unmodified.
    """
    try:
        ns = types.ModuleType("_code_rl_testing_util_child")
        exec(compile(TEST_UTIL, "<testing_util>", "exec"), ns.__dict__)
        result, metadata = ns.run_test(
            test_cases,
            test=generation,
            debug=False,
            timeout=timeout,
            overall_timeout=overall_timeout,
        )
        result_queue.put(("ok", result, metadata))
    except BaseException as e:  # noqa: BLE001 - report *any* child failure back rather than let the parent hang
        result_queue.put(("error", str(e), {}))


def extract_code_from_model(model_response: str) -> str | None:
    """Extract the last fenced code block from a model response."""
    code_blocks = re.findall(r"```(?:\w+)?\n(.*?)```", model_response, re.DOTALL)
    if not code_blocks:
        return None
    return code_blocks[-1].strip()


def postprocess_lcb_sample(sample: list[dict[str, Any]]) -> dict[str, str]:
    """Convert test cases to LiveCodeBench format for the test runner."""
    sample_inputs = [item["input"] for item in sample]
    sample_outputs = [item["output"] for item in sample]

    sample_dict: dict[str, Any] = {
        "inputs": sample_inputs,
        "outputs": sample_outputs,
    }

    if sample[0].get("testtype") == "functional":
        metadata = sample[0].get("metadata", {})
        fn_name = metadata.get("func_name")
        if fn_name is None:
            raise AssertionError(f"Function name missing in metadata: {metadata}. Sample: {sample}")
        sample_dict["fn_name"] = fn_name

    return {
        "input_output": json.dumps(sample_dict),
    }


async def check_correctness(
    sample: list[dict[str, Any]],
    generation: str,
    timeout: int = 6,
    overall_timeout: float | None = None,
) -> tuple[bool, dict[str, Any]]:
    """
    Check correctness of generated code by executing it, in its own
    disposable child process (see module docstring), against ``sample``.

    Args:
        sample: List of test cases in LiveCodeBench format
        generation: Generated code to test
        timeout: Per-test timeout in seconds, enforced by `run_test` itself
            (inside the child) via `signal.alarm` -- this must run on the
            child's main thread, which it does: `multiprocessing.Process`
            gives every child a fresh main thread of its own.
        overall_timeout: Optional wall-clock budget in seconds for the whole
            call, across all of `sample`'s test cases. A submission that
            never trips the per-test `timeout` but has many test cases can
            otherwise run for minutes; the RLE gateway (or other proxies
            in front of this container) has its own, shorter timeout and
            returns a 502/504 to the client once it gives up waiting, which
            looks like a container/infra failure even though this process is
            still working. Bounding total grading time keeps every `step()`
            call safely under that window instead. Also sets this call's own
            hard kill ceiling (plus `_HARD_TIMEOUT_GRACE_S`) for the rare
            case the child never reports back at all.

    Returns:
        Tuple of (all_passed: bool, details: dict)
    """
    assert len(sample) >= 1, "Sample must contain at least one test case"

    test_cases = postprocess_lcb_sample(sample)

    ctx = mp.get_context()
    result_queue: "mp.Queue[tuple[str, Any, dict[str, Any]]]" = ctx.Queue()
    proc = ctx.Process(
        target=_grade_in_subprocess,
        args=(test_cases, generation, timeout, overall_timeout, result_queue),
        daemon=True,
    )
    proc.start()

    hard_timeout = (overall_timeout if overall_timeout is not None else float(timeout) * 5) + (
        _HARD_TIMEOUT_GRACE_S
    )
    loop = asyncio.get_event_loop()
    deadline = loop.time() + hard_timeout
    # Poll instead of a single blocking join(): this runs on the shared
    # event-loop thread (see step_async's docstring), and yielding between
    # polls keeps it responsive to other bookkeeping (e.g. the WebSocket
    # transport's own keepalive) while the child does the real work.
    while proc.is_alive() and loop.time() < deadline:
        await asyncio.sleep(0.05)

    if proc.is_alive():
        logger.error(
            "Code execution child process exceeded its %.1fs hard timeout; killing it.",
            hard_timeout,
        )
        proc.terminate()
        await asyncio.sleep(0.2)
        if proc.is_alive():
            proc.kill()
        proc.join(timeout=2)
        return False, {"error": f"Execution exceeded hard timeout of {hard_timeout:.1f}s"}

    proc.join()

    if result_queue.empty():
        error = f"Execution child process exited without a result (exitcode={proc.exitcode})"
        logger.error("Code execution failed: %s", error)
        return False, {"error": error}

    kind, payload, metadata = result_queue.get()
    if kind == "error":
        logger.error("Code execution failed: %r", payload)
        return False, {"error": payload}

    result = payload
    passed = not any(x != True for x in result)  # noqa: E712 - mirrors run_test's own sentinel check
    details = {"result": result, "metadata": metadata}
    logger.info("Execution result: passed=%s details=%r", passed, details)
    return passed, details


def taco_to_lcb_format(tests: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert TACO-style tests to LiveCodeBench format."""
    inputs = tests.get("inputs", [])
    outputs = tests.get("outputs", [])

    n = max(len(inputs), len(outputs))

    test_cases: list[dict[str, Any]] = []
    for i in range(n):
        inp = inputs[i] if i < len(inputs) else (inputs[0] if inputs else "")
        out = outputs[i] if i < len(outputs) else (outputs[0] if outputs else "")
        if isinstance(out, list):
            out = out[0] if out else ""
        case: dict[str, Any] = {
            "input": inp,
            "output": out,
            "metadata": {},
        }
        if "fn_name" in tests:
            case["testtype"] = "functional"
            case["metadata"]["func_name"] = tests["fn_name"]
        else:
            case["testtype"] = "stdin_stdout"
        test_cases.append(case)

    return test_cases
