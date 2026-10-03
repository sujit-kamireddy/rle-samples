"""Puts the real sample's `agent/` directory on `sys.path`.

These tests exercise `examples/harness/byoh/data_code_agent/agent/app.py` and `opencode_direct.py`
directly via a bare `import app` / `import opencode_direct`; only the tests themselves live here,
relocated out of the user-facing sample tree so `azd ai rle init` does not scaffold maintainer-only
content. Centralised here instead of a per-file `sys.path.insert` so there is one place to fix if
this directory ever moves again.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REAL_AGENT_DIR = Path(__file__).resolve().parents[4] / "data_code_agent" / "agent"
sys.path.insert(0, str(_REAL_AGENT_DIR))
