"""Makes the sample's three import roots visible to pytest.

The environment imports ``rl.*`` and ``rle.server.env`` from the sample root and
``loom_cookbook`` from ``rle/vendor``, which the container sets up with
``PYTHONPATH``. Doing it here too means ``pytest`` runs from the sample root
with no environment variables.
"""

from __future__ import annotations

import sys
from pathlib import Path

SAMPLE_ROOT = Path(__file__).resolve().parent.parent

for path in (SAMPLE_ROOT, SAMPLE_ROOT / "rle", SAMPLE_ROOT / "rle" / "vendor"):
    entry = str(path)
    if entry not in sys.path:
        sys.path.insert(0, entry)
