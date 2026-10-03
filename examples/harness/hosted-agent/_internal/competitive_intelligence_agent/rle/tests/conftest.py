"""Puts the real sample root on `sys.path` so `rle.server.*` imports resolve.

These tests exercise `examples/harness/hosted-agent/competitive_intelligence_agent/rle/server/...`
via dotted imports like `from rle.server.environment import ...`; only the tests themselves live
here, relocated out of the user-facing sample tree so `azd ai rle init` does not scaffold
maintainer-only content.

Without this, pytest's own import-path insertion would walk up only as far as this placeholder
`rle/` directory (it has no `__init__.py`, so pytest stops there), which does not contain
`server/` and leaves `rle.server` unresolvable. Inserting the real sample root ahead of that gives
Python a path that actually has `rle/server/` under it.

`rle_deprecated/` moved here too, alongside the one test that still needs it
(`test_grade_matches_the_legacy_harness`). `rle_deprecated` itself has no
`__init__.py` (only its `server/` and `rl/` subpackages do), so
``from rle_deprecated.server.env import app`` only resolves once
`rle_deprecated`'s own *parent* -- this directory, not `rle_deprecated/`
itself -- is on `sys.path`; that is `_INTERNAL_SAMPLE_ROOT`, inserted below
alongside the `vendor/` entry its code expects for its own bare imports.
Doing this only if `rle_deprecated/` is present is what lets that test skip
itself, rather than error, once it is deleted for good.
"""

from __future__ import annotations

import sys
from pathlib import Path

_INTERNAL_SAMPLE_ROOT = Path(__file__).resolve().parents[2]
_REAL_SAMPLE_ROOT = Path(__file__).resolve().parents[4] / "competitive_intelligence_agent"

sys.path.insert(0, str(_REAL_SAMPLE_ROOT))

if (_INTERNAL_SAMPLE_ROOT / "rle_deprecated").exists():
    sys.path.insert(0, str(_INTERNAL_SAMPLE_ROOT))
    vendor = str(_INTERNAL_SAMPLE_ROOT / "rle_deprecated" / "vendor")
    if vendor not in sys.path:
        sys.path.append(vendor)
