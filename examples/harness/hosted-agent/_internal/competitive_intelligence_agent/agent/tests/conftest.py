"""Puts the real sample's `agent/` directory on `sys.path` so flat imports resolve.

These tests exercise
`examples/harness/hosted-agent/competitive_intelligence_agent/agent/rollout_context.py`
via a flat import (``from rollout_context import ...``), matching how the agent
container itself imports it (see ``agent/main.py``); only the tests themselves live
here, relocated out of the user-facing sample tree so `azd ai rle init` does not
scaffold maintainer-only content.

Without this, pytest's own import-path insertion would walk up only as far as this
placeholder `agent/` directory (it has no `__init__.py`, so pytest stops there), which
does not contain the real `rollout_context.py`. Inserting the real sample's `agent/`
directory ahead of that gives Python a path that actually has it.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REAL_SAMPLE_AGENT_ROOT = (
    Path(__file__).resolve().parents[4] / "competitive_intelligence_agent" / "agent"
)

sys.path.insert(0, str(_REAL_SAMPLE_AGENT_ROOT))
