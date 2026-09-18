"""Restores collections.abc aliases removed from the top-level `collections`
module in Python 3.10, so the pinned 2016-era ``requests`` checkout (which
still does ``collections.Mapping``) imports unmodified on a modern
interpreter. Auto-imported by every Python process in this image (including
the ``pytest`` subprocess spawned by ``/grade``) because ``sitecustomize``
is a magic module name Python imports at startup when present on
``sys.path``. This file changes no application behavior; it only restores
removed stdlib aliases.
"""

import collections
import collections.abc as _abc

for _name in ("Mapping", "MutableMapping", "Sequence", "Iterable", "Callable"):
    if not hasattr(collections, _name):
        setattr(collections, _name, getattr(_abc, _name))
