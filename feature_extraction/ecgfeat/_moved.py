"""Aliases for private modules relocated under ``ecgfeat._engine``.

An old module path is bound to the *same* module object as its new location,
so attribute access, ``mock.patch`` targets and private helpers keep working.
Each alias warns once per process, attributed to the importing code.
"""

from __future__ import annotations

import importlib
import sys
import warnings

#: First release allowed to drop the aliases (two minor releases after the
#: first warning release, docs/library_design/05_migration_plan.md section A).
REMOVAL_RELEASE = "0.3.0"


def _caller_stacklevel() -> int:
    """Stack level of the first frame outside import machinery and shims.

    ``warnings.warn`` itself skips ``importlib._bootstrap`` frames when it walks
    ``stacklevel`` frames, so those frames are skipped without being counted.
    """

    frame = sys._getframe(2)  # the shim module body, i.e. warn(stacklevel=2)
    level = 2
    while frame is not None:
        filename = frame.f_code.co_filename
        internal = "importlib" in filename and "_bootstrap" in filename
        if not internal:
            shim = (frame.f_globals.get("alias_module") is alias_module
                    or frame.f_globals.get("__name__") == "feature_extraction"  # repository alias loader
                    or filename.endswith(("importlib/__init__.py", "importlib\\__init__.py")))
            if not shim:
                return level
            level += 1
        frame = frame.f_back
    return 2


def alias_module(name: str, relative_target: str) -> None:
    package = name.rpartition(".")[0]
    target = importlib.import_module(relative_target, package)
    warnings.warn(
        f"{name} moved to the private module {target.__name__}; the old path is not "
        f"public API and will be removed no earlier than ecg-records {REMOVAL_RELEASE}. "
        "Use the documented ecgfeat record/extraction API instead.",
        DeprecationWarning,
        stacklevel=_caller_stacklevel(),
    )
    sys.modules[name] = target
