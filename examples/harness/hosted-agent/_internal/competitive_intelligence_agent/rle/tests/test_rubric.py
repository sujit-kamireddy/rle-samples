"""Direct coverage for the rollout-graph error-count helper in `grading/rubric.py`.

This is the one piece of the rubric that reads data from outside the
simulated tool world (`Task`, `ToolSession`), so unlike the rest of the
rubric's scoring functions -- exercised end-to-end through
`test_environment.py`'s `grade` contract tests -- it is worth pinning
directly against the range of shapes a `rollout_graph` can actually take:
absent, fully populated, and partially stripped the way a grader-facing
sanitizer can strip one today.
"""

from __future__ import annotations

from rle.server.grading.rubric import _rollout_graph_error_counts


def test_a_missing_rollout_graph_counts_as_no_errors():
    # `None` is the common case: see `environment.grade`'s docstring for why.
    assert _rollout_graph_error_counts(None) == (0, 0)
    assert _rollout_graph_error_counts({}) == (0, 0)


def test_counts_prefer_the_always_present_count_fields_over_the_detailed_lists():
    rollout_graph = {
        "turns": [{"n_tool_call_errors": 2}, {"n_tool_call_errors": 0}],
        "model_call_errors": [{"error_code": "timeout"}, {"error_code": "http_error"}],
    }

    assert _rollout_graph_error_counts(rollout_graph) == (2, 2)


def test_counts_fall_back_to_the_detailed_lists_when_a_count_field_is_missing():
    # A grader-facing sanitizer can legitimately strip the top-level
    # `model_call_errors` list while leaving the nested `stats` count alone
    # (see `rubric.py`'s module docstring), and the per-turn detailed
    # `tool_call_errors` list is gated independently of its own count.
    rollout_graph = {
        "turns": [{"tool_call_errors": [{"error_code": "invalid_syntax"}]}],
        "stats": {"n_model_call_errors": 3},
    }

    assert _rollout_graph_error_counts(rollout_graph) == (1, 3)


def test_a_turn_that_is_not_a_dict_is_skipped_rather_than_raising():
    rollout_graph = {"turns": [None, {"n_tool_call_errors": 1}, "not-a-turn"]}

    assert _rollout_graph_error_counts(rollout_graph) == (1, 0)
