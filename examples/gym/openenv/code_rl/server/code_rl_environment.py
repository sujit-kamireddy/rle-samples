"""``code_rl`` OpenEnv environment: one competitive-programming problem per
episode.

``reset()`` selects a problem. ``step()`` may be called up to ``max_turns``
times per episode: a response with a ``check_solution`` tool call runs that
call for real (same grader as the final answer) and keeps the episode going
(``done=False``) by returning the tool's result as a ``"tool"``-role
message; a response with no tool call, or one at ``max_turns``, grades
``code_text`` and ends the episode (``done=True``).

Grading runs ``check_correctness`` from ``server/code_grading.py``, entirely
in this container (see ``examples/gym/openenv/README.md``).

``FORMAT_COEF`` is the format-penalty coefficient: code-block extraction
uses ``extract_code_from_model`` (last fenced block, or ``None``), and an
unfenced submission scores ``-FORMAT_COEF`` without attempting to grade the
raw text at all -- rewarding "submitted correctly-formatted code" strictly
more than "got lucky with unformatted text".

The submitted solution is ``exec()``'d in-process (see
``server/code_grading.py``) rather than farmed out to an external sandbox
service, so the image needs no sandbox dependency and no egress. Grading
(``check_correctness``) is only ever awaited from ``step_async`` -- see that
method's docstring for why it must run on the event loop's own thread rather
than a threadpool worker.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from .code_grading import check_correctness, extract_code_from_model
from .dataset import EpisodePicker, load_jsonl
from .schema import CodeAction, CodeObservation

# Format-penalty coefficient -- see the module docstring.
FORMAT_COEF = 0.1

# Default number of check_solution turns allowed before a final answer is
# required.
DEFAULT_MAX_TURNS = 2

# The check_solution tool's spec, as a literal dict rather than generated
# from a decorator. Handed back in ``reset()``'s observation metadata (see
# ``CodeRLEnvironment.reset()``) rather than a separate schema fetch, so a
# training client never needs its own copy of this spec -- OpenEnv's ``/ws``
# protocol has no distinct "schema" message type.
CHECK_SOLUTION_TOOL_SPEC: dict[str, Any] = {
    "name": "check_solution",
    "description": (
        "Execute the proposed solution against the task's test cases.\n\n"
        "Use this to test your code before providing your final answer."
    ),
    "parameters": {
        "properties": {
            "code": {
                "description": "Python code implementing the solution.",
                "title": "Code",
                "type": "string",
            }
        },
        "required": ["code"],
        "title": "check_solution_params",
        "type": "object",
    },
}

# Same tool, reshaped for RLE's Gym tool-discovery response (CodeObservation.tools --
# see its field docstring in schema.py): "parameters" -> "input_schema", matching
# GymOpenEnvRolloutTargetInvoker.ParseChatCompletionTools's expected key name.
CHECK_SOLUTION_TOOL_DISCOVERY: dict[str, Any] = {
    "name": CHECK_SOLUTION_TOOL_SPEC["name"],
    "description": CHECK_SOLUTION_TOOL_SPEC["description"],
    "input_schema": CHECK_SOLUTION_TOOL_SPEC["parameters"],
}


class CodeRLEnvironment(Environment[CodeAction, CodeObservation, State]):
    """One-shot code-grading environment: reset -> problem, step -> reward.

    A real ``openenv`` ``Environment`` subclass, served by ``openenv``'s own
    ``HTTPEnvServer`` (see ``examples/gym/openenv/code_rl/server/app.py``). Its ``/ws`` route
    builds one instance per WebSocket connection and keeps it for that
    connection's whole lifetime, so ``step()`` grades against the row
    ``reset()`` picked as long as both calls land on the same connection --
    which they do, since a training client is expected to open exactly one
    connection per leased instance and reuse it for every episode that
    instance goes on to serve.
    """

    def __init__(
        self,
        dataset_path: Optional[str] = None,
        grading_timeout: int = 6,
        grading_overall_timeout: float = 30.0,
        max_turns: int = DEFAULT_MAX_TURNS,
        validation_dataset_path: Optional[str] = None,
    ):
        super().__init__()
        default_dataset_path = Path(__file__).resolve().parents[1] / "env_data" / "train.jsonl.gz"
        dataset_path = dataset_path or os.environ.get("CODE_RL_DATASET_PATH", str(default_dataset_path))
        self._rows = EpisodePicker(load_jsonl(dataset_path))
        # Resolved lazily (see _validation_rows()), not loaded here: most
        # callers (including every existing test) construct this with only
        # a train file on disk and never request split="validation", and
        # eagerly loading here would make that an unconditional
        # FileNotFoundError. Defaults to the matching sibling validation file
        # for either plain JSONL or the checked-in gzip snapshot.
        default_validation_name = (
            "validation.jsonl.gz" if str(dataset_path).endswith(".gz") else "validation.jsonl"
        )
        self._validation_dataset_path = validation_dataset_path or os.environ.get(
            "CODE_RL_VALIDATION_DATASET_PATH",
            str(Path(dataset_path).with_name(default_validation_name)),
        )
        self._validation_rows: Optional[EpisodePicker] = None
        self._grading_timeout = int(os.environ.get("CODE_RL_GRADING_TIMEOUT", grading_timeout))
        # Bounds total grading time across all of a sample's test cases (see
        # check_correctness's overall_timeout docstring) -- keeps a single
        # step() call well under the RLE gateway's own timeout even for
        # submissions/problems that never trip the per-test grading_timeout
        # but have many test cases. Comfortably below the ~45-120s Bad
        # Gateway/timeout window observed from the RLE data plane.
        self._grading_overall_timeout = float(
            os.environ.get("CODE_RL_GRADING_OVERALL_TIMEOUT", grading_overall_timeout)
        )
        self._max_turns = int(os.environ.get("CODE_RL_MAX_TURNS", max_turns))
        self._state = State(episode_id=None, step_count=0)
        self._current_row: Optional[dict] = None

    def _rows_for_split(self, split: str) -> EpisodePicker:
        """Resolve which baked dataset file a ``reset()`` picks from.

        ``split="validation"`` picks from the matching validation JSONL
        snapshot (loaded lazily and cached on first use) instead of the
        training file. Without this, validation metrics would be computed
        against different rows of the same training file.
        """
        if split == "train":
            return self._rows
        if split == "validation":
            if self._validation_rows is None:
                self._validation_rows = EpisodePicker(load_jsonl(self._validation_dataset_path))
            return self._validation_rows
        raise ValueError(f"Unknown split {split!r}; expected 'train' or 'validation'.")

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        split: str = "train",
        **kwargs: Any,
    ) -> CodeObservation:
        index, row = self._rows_for_split(split).pick(seed)
        self._current_row = row
        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            extra={"row_index": index},
        )
        return CodeObservation(
            done=False,
            reward=None,
            messages=row["messages"],
            starter_code=row.get("starter_code"),
            problem_id=str(index),
            # See CHECK_SOLUTION_TOOL_SPEC's comment: handed back here
            # rather than over a separate schema call, since OpenEnv's
            # ``/ws`` protocol has no such call.
            metadata={"tool_specs": [CHECK_SOLUTION_TOOL_SPEC]},
        )

    def step(
        self,
        action: CodeAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> CodeObservation:
        return asyncio.run(self._step_impl(action, timeout_s=timeout_s, **kwargs))

    async def step_async(
        self,
        action: CodeAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> CodeObservation:
        """Overrides the base (default-to-``step``) ``step_async`` for real,
        so ``openenv``'s ``/ws`` route ``await``s it directly on its own
        event-loop thread instead of dispatching ``step()`` onto a
        threadpool worker (see ``_run_in_session_executor`` in
        ``openenv.core.env_server.http_server``). That distinction matters
        here specifically: grading's per-test timeout
        (``check_correctness``'s ``timeout``) is enforced by
        ``lcb_utils.run_test`` via ``signal.alarm``, and ``SIGALRM`` is only
        ever delivered to a process's *main* thread -- on a threadpool
        worker it would instead land in whichever thread owns the signal
        (typically the event loop thread), leaving the worker's submission
        running unbounded while raising a stray, unrelated
        ``TimeoutException`` elsewhere. Running this on the event loop
        thread directly is safe precisely because each container serves one
        connection/episode at a time (``max_concurrent_envs=1``, see
        ``examples/gym/openenv/code_rl/server/app.py``), so blocking it for the bounded
        ``grading_overall_timeout`` here is the same tradeoff the
        single-worker/single-session deployment already makes elsewhere.
        """
        return await self._step_impl(action, timeout_s=timeout_s, **kwargs)

    async def _step_impl(
        self,
        action: CodeAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> CodeObservation:
        if action.type == "list_tools":
            # Stand-in for OpenEnv's own ListToolsAction handling -- see
            # CodeAction's docstring in schema.py. Echoes the same tool reset()
            # already hands back in metadata.tool_specs, reshaped for RLE's expected
            # top-level `tools` schema (see CHECK_SOLUTION_TOOL_DISCOVERY).
            return CodeObservation(
                done=False,
                reward=None,
                messages=[],
                tools=[CHECK_SOLUTION_TOOL_DISCOVERY],
            )

        self._state.step_count += 1
        row = self._rows.row_for_episode(action.problem_id, self._current_row)

        tests = row.get("tests")

        if not tests:
            # A data problem, not a wrong answer -- surface it as a failed
            # (but non-crashing) episode rather than raising, so a single
            # bad row can't take down a rollout batch.
            return CodeObservation(
                done=True,
                reward=0.0,
                messages=[],
                metadata={"error": "row has no 'tests' field to grade against"},
            )

        tool_calls = action.tool_calls or []
        no_tool_calls = len(tool_calls) == 0
        max_turns_reached = self._state.step_count >= self._max_turns
        done = no_tool_calls or max_turns_reached

        tool_messages: list[dict[str, Any]] = []
        if tool_calls:
            try:
                tool_messages = [
                    await self._run_check_solution_call(tc, tests) for tc in tool_calls
                ]
            except Exception as exc:  # pragma: no cover - defensive
                # A malformed tool call shouldn't take down a rollout batch
                # any more than a malformed final submission does.
                return CodeObservation(
                    done=True,
                    reward=0.0,
                    messages=[],
                    metadata={"error": f"tool call failed: {exc}"},
                )

        if not done:
            # Episode continues: hand back the tool result(s), don't grade
            # yet -- only the episode's final step grades and assigns
            # reward.
            return CodeObservation(done=False, reward=0.0, messages=tool_messages)

        code = extract_code_from_model(action.code_text)
        has_code_block = code is not None

        passed = False
        details = None
        if has_code_block:
            try:
                passed, details = await check_correctness(
                    tests,
                    code,
                    timeout=self._grading_timeout,
                    overall_timeout=self._grading_overall_timeout,
                )
            except Exception as exc:  # pragma: no cover - defensive
                # check_correctness catches its own execution failures; this
                # only guards a genuinely unexpected error (e.g. malformed
                # tests) so a single bad row can't take down a rollout batch.
                return CodeObservation(
                    done=True,
                    reward=0.0,
                    messages=[],
                    metadata={"error": f"grading failed: {exc}"},
                )

        format_score = 1.0 if has_code_block else 0.0
        correct_score = 1.0 if passed else 0.0
        reward = FORMAT_COEF * (format_score - 1.0) + correct_score

        metadata: dict[str, Any] = {"passed": passed, "format": has_code_block, "details": details}
        if max_turns_reached and not no_tool_calls:
            metadata["max_turns"] = True

        return CodeObservation(
            done=True,
            reward=reward,
            # Tool result(s) from this same turn are included in the message
            # history even though the episode ends here, so the transcript
            # shows the full final turn.
            messages=tool_messages,
            metadata=metadata,
        )

    async def _run_check_solution_call(self, tool_call: dict[str, Any], tests: Any) -> dict[str, Any]:
        """Run one ``check_solution`` tool call for real and return its
        ``"tool"``-role result message.
        """
        call_id = tool_call.get("id") or ""
        name = tool_call.get("name") or "check_solution"
        try:
            arguments = json.loads(tool_call.get("arguments") or "{}")
            code = arguments.get("code", "")
        except (TypeError, ValueError) as exc:
            return {
                "role": "tool",
                "content": json.dumps({"error": f"invalid arguments: {exc}", "passed": False}),
                "tool_call_id": call_id,
                "name": name,
            }

        try:
            passed, details = await check_correctness(
                tests,
                code,
                timeout=self._grading_timeout,
                overall_timeout=self._grading_overall_timeout,
            )
            content = json.dumps({"passed": passed, "details": details}, ensure_ascii=False)
        except Exception as exc:  # pragma: no cover - defensive
            content = json.dumps({"error": str(exc), "passed": False})

        return {"role": "tool", "content": content, "tool_call_id": call_id, "name": name}

    @property
    def state(self) -> State:
        return self._state
