"""Patch one name in every module that makes the call during extraction.

Before the Phase 3 decomposition, ``patch("feature_extraction.ecgfeat.api.X", ...)``
intercepted every call to ``X`` made by ``ECGFeatureExtractor.extract`` and by the
private helpers that lived in ``api.py``.  Those calls are now made from several
stage, policy and compatibility-hook modules.  ``patch_calls`` applies one mock (same
arguments, one shared object) to all of them, so a single ``with`` item keeps the
original interception semantics.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from typing import Any, Iterator
from unittest.mock import patch


@contextmanager
def patch_calls(*targets: str, **kwargs: Any) -> Iterator[Any]:
    first, *rest = targets
    with patch(first, **kwargs) as mock, ExitStack() as stack:
        for target in rest:
            stack.enter_context(patch(target, new=mock))
        yield mock
