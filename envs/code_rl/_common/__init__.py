"""Small, shared plumbing used by every ``envs/<recipe>/`` OpenEnv server.

Nothing reward-relevant or recipe-specific lives here -- just JSONL loading
and episode-index picking -- so each recipe's
``server/<recipe>_environment.py`` can focus on its own
``reset``/``step``/``state`` logic (the actual RL environment
implementation, meant to be read end-to-end by repo users) instead of
re-deriving "how do I read a baked JSONL file" every time.
"""
