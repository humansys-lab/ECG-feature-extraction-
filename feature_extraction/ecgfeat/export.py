"""Legacy export module path (silent alias).

The historical ``to_dict``/``prepare_json_export`` exporter lives in
``ecgfeat.compat.export_v0``.  Plain import stays silent
(docs/library_design/03_public_api.md, "Deprecation mechanism"); this path
resolves to the same module object.
"""

import importlib as _importlib
import sys as _sys

_sys.modules[__name__] = _importlib.import_module(".compat.export_v0", __name__.rpartition(".")[0])
