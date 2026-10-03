"""Puts the real sample root on `sys.path` so `rle.server.*` imports resolve.

These tests exercise `examples/harness/byoh/data_code_agent/rle/server/...` via dotted imports
like `from rle.server import compliance`; only the tests themselves live here, relocated out of
the user-facing sample tree so `azd ai rle init` does not scaffold maintainer-only content.

Without this, pytest's own import-path insertion would walk up only as far as this placeholder
`rle/` directory (it has no `__init__.py`, so pytest stops there), which does not contain
`server/` and leaves `rle.server` unresolvable. Inserting the real sample root ahead of that gives
Python a path that actually has `rle/server/` under it.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REAL_SAMPLE_ROOT = Path(__file__).resolve().parents[4] / "data_code_agent"
sys.path.insert(0, str(_REAL_SAMPLE_ROOT))
