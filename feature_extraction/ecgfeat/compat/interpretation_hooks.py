"""Interpretation calls made by the legacy extraction entry point.

The pre-decomposition ``ECGFeatureExtractor.extract`` called interpretation code at
three fixed points:

1. ``build_rhythm_statement_candidates`` after measurement availability was applied,
   stored as ``rule_summary["statement_evidence"]``;
2. ``interpret(features)`` once the legacy ``ECGFeatures`` was assembled;
3. ``analyze_clinical(features, prior_features=...).to_dict()`` stored as
   ``metadata["clinical_interpretation"]``.

Interpretation belongs to the compatibility layer, not to pipeline stages, so the
stages call these through an injected :class:`LegacyInterpretationHooks` at exactly
those points (see ``ecgfeat.pipeline.context.InterpretationHooks``).  The legacy entry
point always injects :data:`LEGACY_INTERPRETATION_HOOKS`, which preserves the legacy
output, including metadata key insertion order.
"""

from __future__ import annotations

from typing import Any

from ..clinical_rules.engine import analyze_clinical
from ..interpret import interpret
from ..rhythm_statements import build_rhythm_statement_candidates


class LegacyInterpretationHooks:
    """Default hooks: the exact calls the pre-decomposition ``extract`` made."""

    __slots__ = ()

    def rhythm_statement_evidence(self, **evidence: Any) -> Any:
        return build_rhythm_statement_candidates(**evidence)

    def interpret(self, features: Any) -> Any:
        return interpret(features)

    def clinical_interpretation(self, features: Any, *, prior_features: Any = None) -> Any:
        return analyze_clinical(
            features,
            prior_features=prior_features,
        ).to_dict()


LEGACY_INTERPRETATION_HOOKS = LegacyInterpretationHooks()

__all__ = ["LegacyInterpretationHooks", "LEGACY_INTERPRETATION_HOOKS"]
