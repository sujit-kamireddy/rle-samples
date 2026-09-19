"""``code_rl`` OpenEnv environment: one competitive-programming problem per
episode.

``reset()`` selects a problem. ``step()`` may be called up to ``max_turns``
times per episode: a response with a ``check_solution`` tool call runs that
call for real (same grader as the final answer) and keeps the episode going
(``done=False``) by returning the tool's result as a ``"tool"``-role
message; a response with no tool call, or one at ``max_turns``, grades
``code_text`` and ends the episode (``done=True``). This mirrors
``loom_cookbook.tool_use.agent_tool_message_env.AgentToolMessageEnv.step()``
and ``loom_cookbook.recipes.code_rl.deepcoder_tool.DeepcoderReward``, which
together give the same turn-continuation/final-grading semantics on the
``use_rle=False`` path.

Grading runs ``check_correctness`` from ``examples/gym/openenv/code_rl/grading/`` -- a copy
of the recipe's grader, vendored so this image installs no ``loom_cookbook``
(see ``examples/gym/openenv/README.md``). ``tests/test_env_grading_parity.py`` pins the copy
against the original, which is what keeps the ``use_rle=True`` and
``use_rle=False`` paths scoring solutions identically.

The reward *composition* is pinned the same way: ``FORMAT_COEF`` below must
equal ``loom_cookbook.recipes.code_rl.deepcoder_tool.DeepcoderReward``'s
``format_coef`` default, code-block extraction uses the same vendored
``extract_code_from_model`` (last fenced block, or ``None``) rather than a
local regex, and an unfenced submission scores ``-FORMAT_COEF`` without
attempting to grade the raw text -- exactly like ``DeepcoderReward.__call__``.
``test_env_grading_parity.py`` pins both.

The submitted solution is ``exec()``'d in-process (see
``grading/code_grading.py``) rather than farmed out to an external sandbox
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

try:
    from .._common.dataset import EpisodePicker, load_jsonl
except ImportError:  # pragma: no cover - standalone container import path
    from _common.dataset import EpisodePicker, load_jsonl

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..grading import check_correctness, extract_code_from_model
except ImportError:  # pragma: no cover - standalone container import path
    from grading import check_correctness, extract_code_from_model

try:
    from ..models import CodeAction, CodeObservation
except ImportError:  # pragma: no cover - standalone container import path
    from models import CodeAction, CodeObservation

# Must equal loom_cookbook.recipes.code_rl.deepcoder_tool.DeepcoderReward's
# format_coef default -- see the module docstring.
FORMAT_COEF = 0.1

# Must equal loom_cookbook.recipes.code_rl.train_azure.CLIConfig's
# max_turns default -- see the module docstring.
DEFAULT_MAX_TURNS = 2

# Byte-for-byte what @tool-decorating
# loom_cookbook.recipes.code_rl.deepcoder_tool.DeepcoderTool.check_solution
# generates via FunctionTool.to_spec() (pinned by
# tests/test_env_grading_parity.py) -- vendored as a literal dict rather
# than imported, since this image installs no loom_cookbook (see module
# docstring). Handed back in ``reset()``'s observation metadata (see
# ``CodeRLEnvironment.reset()``) rather than a separate schema fetch, so
# the client (``loom_cookbook/rl/rle_env.py``) never needs its own copy
# of this spec and OpenEnv's ``/ws`` protocol -- which has no distinct
# "schema" message type -- never needs one either.
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


class CodeRLEnvironment(Environment[CodeAction, CodeObservation, State]):
    """One-shot code-grading environment: reset -> problem, step -> reward.

    A real ``openenv`` ``Environment`` subclass, served by ``openenv``'s own
    ``HTTPEnvServer`` (see ``examples/gym/openenv/code_rl/server/app.py``). Its ``/ws`` route
    builds one instance per WebSocket connection and keeps it for that
    connection's whole lifetime, so ``step()`` grades against the row
    ``reset()`` picked as long as both calls land on the same connection --
    which they do, since ``loom_cookbook/rl/rle_env.py`` opens exactly one
    connection per leased instance and reuses it for every episode that
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
        default_dataset_path = Path(__file__).resolve().parents[1] / "data" / "train.jsonl.gz"
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
        (``check_correctness``'s ``timeout``) is enforced by the vendored
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
            # CodeAction's docstring in ../models.py. Echoes the same spec reset()
            # already hands back in metadata.tool_specs, so both paths agree.
            return CodeObservation(
                done=False,
                reward=None,
                messages=[],
                metadata={"tool_specs": [CHECK_SOLUTION_TOOL_SPEC]},
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
            # yet -- mirrors AgentToolMessageEnv.step()'s reward_fn only
            # firing once the episode is done.
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
            # Tool result(s) from this same turn are included for parity
            # with the local (use_rle=False) path's message history, even
            # though the episode ends here -- see AgentToolMessageEnv.step().
            messages=tool_messages,
            metadata=metadata,
        )

    async def _run_check_solution_call(self, tool_call: dict[str, Any], tests: Any) -> dict[str, Any]:
        """Run one ``check_solution`` tool call for real and return its
        ``"tool"``-role result message -- mirrors
        ``loom_cookbook.recipes.code_rl.deepcoder_tool.DeepcoderTool.check_solution``
        and ``loom_cookbook.tool_use.tools.simple_tool_result``.
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
