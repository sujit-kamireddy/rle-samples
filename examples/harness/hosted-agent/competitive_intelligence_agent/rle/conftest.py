"""Makes the sample's import roots visible to pytest.

``rle`` is self-contained: it carries its own ``rl/`` and ``vendor/``
so the container never needs ``rle_deprecated/``. Importing the environment
needs only the sample root and this folder's vendored ``loom_cookbook``, which
is exactly what the container sets with ``PYTHONPATH``.

``rle_deprecated/`` is appended afterwards purely so ``tests/test_environment.py``
can stand the legacy harness up beside this one and assert the two still grade
identically. That test is the gate on the duplication. It and these last two
entries go when ``rle_deprecated/`` is deleted.
"""

from __future__ import annotations

import sys
from pathlib import Path

SAMPLE_ROOT = Path(__file__).resolve().parent.parent
VENDOR = SAMPLE_ROOT / "rle" / "vendor"

# Order matters: `loom_cookbook` must resolve to this folder's copy, not the
# legacy one, so the tests exercise the code the image actually ships.
for position, path in enumerate((SAMPLE_ROOT, VENDOR)):
    entry = str(path)
    if entry in sys.path:
        sys.path.remove(entry)
    sys.path.insert(position, entry)

for path in (SAMPLE_ROOT / "rle_deprecated", SAMPLE_ROOT / "rle_deprecated" / "vendor"):
    entry = str(path)
    if path.exists() and entry not in sys.path:
        sys.path.append(entry)
