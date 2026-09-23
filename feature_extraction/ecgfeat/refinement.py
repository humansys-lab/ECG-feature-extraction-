"""Explicit, independently ablatable accuracy candidates.

No candidate is enabled implicitly. Default extraction retains its established
measurements; callers opt into candidates whose cohort results they reviewed.
"""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RefinementConfig:
    p_model_arbitration: bool = False
    p_multiple_candidates: bool = False
    p_boundary_correction: bool = False
    t_onset_change_point: bool = False
    t_bidirectional: bool = False
    t_correlated_fusion: bool = False
    qrs_terminal_multiscale: bool = False
    qrs_quality_reference: bool = False
    representative_robust: bool = False
    grouping_outliers: bool = False
    p_phasor_candidates: bool = False
    t_boundary_projection: bool = False
    t_sequence_selection: bool = False
    t_projection_offset_only: bool = False
    t_sequence_offset_only: bool = False
    qrs_adaptive_consensus: bool = False
    p_pathology_candidates: bool = False
    atrial_event_validation: bool = False

    def __post_init__(self):
        if any(type(value) is not bool for value in asdict(self).values()):
            raise ValueError("refinement options must be booleans")

    @classmethod
    def experimental(cls, *names: str, enabled: bool = True):
        """All candidates, for evaluation; not a validated clinical preset."""
        if not names:
            names = tuple(cls.__dataclass_fields__)
        unknown = set(names) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown refinement flags: {', '.join(sorted(unknown))}")
        if type(enabled) is not bool:
            raise ValueError("enabled must be a boolean")
        return cls(**{name: enabled for name in names})

    @property
    def experimental_flags(self):
        """Effective flags; a property keeps legacy asdict() JSON-compatible."""
        return frozenset(name for name, value in asdict(self).items() if value)

    @property
    def enabled(self):
        return any(asdict(self).values())
