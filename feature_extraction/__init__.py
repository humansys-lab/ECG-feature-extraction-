"""Repository-local alias: ``feature_extraction.ecgfeat`` *is* ``ecgfeat``.

Code and tests in this repository historically imported the package both as
``feature_extraction.ecgfeat`` (from the checkout) and as ``ecgfeat`` (the
installed distribution), which loaded two independent copies of every module
and class.  This finder maps every ``feature_extraction.ecgfeat[.x]`` import to
the ``ecgfeat[.x]`` module object, so there is one copy, ``isinstance`` works
across spellings, and ``mock.patch`` targets patch the real module.  It is
migration debt (docs/library_design/05_migration_plan.md, "repository-local
test imports") and is not part of any distribution.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys

_ALIAS = __name__ + ".ecgfeat"


class _AliasLoader(importlib.abc.Loader):
    def __init__(self, target: str) -> None:
        self._target = target
        self._original_spec = None

    def create_module(self, spec):
        module = importlib.import_module(self._target)
        self._original_spec = module.__spec__
        return module

    def exec_module(self, module) -> None:
        # The target module is already executed.  importlib unconditionally set
        # module.__spec__ to the alias spec; restore the real one so that
        # importlib.resources, pickling and reloads keep seeing ``ecgfeat``.
        module.__spec__ = self._original_spec


class _AliasFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != _ALIAS and not fullname.startswith(_ALIAS + "."):
            return None
        real = "ecgfeat" + fullname[len(_ALIAS):]
        return importlib.util.spec_from_loader(fullname, _AliasLoader(real))


if not any(isinstance(finder, _AliasFinder) for finder in sys.meta_path):
    sys.meta_path.insert(0, _AliasFinder())
