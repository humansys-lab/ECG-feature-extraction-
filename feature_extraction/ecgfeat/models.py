"""Legacy model module path (silent alias).

The measurement carriers live in ``ecgfeat._engine.foundation.models``; the
supported legacy names are re-exported by ``ecgfeat.compat.models_v0``.  Plain
import of this module is intentionally silent (docs/library_design/03_public_api.md,
"Deprecation mechanism"): it resolves to the same module object, so class
identity, ``isinstance`` checks and pickles are unchanged.
"""

import importlib as _importlib
import sys as _sys

_sys.modules[__name__] = _importlib.import_module("._engine.foundation.models", __name__.rpartition(".")[0])
