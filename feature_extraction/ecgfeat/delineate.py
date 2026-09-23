"""Moved: the implementation now lives in ``ecgfeat._engine.delineation.core``.

This private engine module was relocated during the ``_engine`` migration
(docs/library_design/05_migration_plan.md, Phase 1).  The old import path is a
temporary alias to the same module object and emits ``DeprecationWarning``.
"""

from ._moved import alias_module

alias_module(__name__, "._engine.delineation.core")
