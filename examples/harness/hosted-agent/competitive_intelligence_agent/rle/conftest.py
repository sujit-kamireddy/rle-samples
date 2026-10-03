"""Makes the sample's import roots visible to pytest.

``rle`` is self-contained: ``server/`` carries its own tasks, world, tools and
rubric so the container never needs ``rle_deprecated/``. Importing the
environment needs only the sample root, which is exactly what the container
sets with ``PYTHONPATH``.

``rle_deprecated/`` is appended afterwards purely so ``tests/test_environment.py``
can stand the legacy harness up beside this one and assert the two still grade
identically. That test is the gate on the duplication. It and this last entry
go when ``rle_deprecated/`` is deleted.
"""

from __future__ import annotations

import sys
from pathlib import Path

SAMPLE_ROOT = Path(__file__).resolve().parent.parent

entry = str(SAMPLE_ROOT)
if entry in sys.path:
    sys.path.remove(entry)
sys.path.insert(0, entry)

for path in (SAMPLE_ROOT / "rle_deprecated", SAMPLE_ROOT / "rle_deprecated" / "vendor"):
    entry = str(path)
    if path.exists() and entry not in sys.path:
        sys.path.append(entry)
