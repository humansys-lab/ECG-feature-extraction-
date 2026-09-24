"""Moved: the implementation now lives in ``ecgfeat._engine.measurement.rhythm``.

Pacing, beat-selection and measurement-availability helpers stay in the core
engine; diagnostic statement evidence moved to ``ecginterpret.rhythm``
(docs/library_design/05_migration_plan.md, Phase 5).  The old import path is a
temporary alias to the same module object and emits ``DeprecationWarning``.
"""

from ._moved import alias_module

alias_module(__name__, "._engine.measurement.rhythm")
