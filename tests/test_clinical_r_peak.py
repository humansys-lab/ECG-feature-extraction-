"""Clinical R selection in negative-dominant complexes (_clinical_r_peak_local).

Standard nomenclature: R is the *first* positive deflection of the QRS, and a
later positive component is R'. The global positive maximum relabels R' as R
whenever the later component is taller, which is the rSR' / rS-with-overshoot
case in V1/V2/aVR.
"""
from __future__ import annotations

import numpy as np

from feature_extraction.ecgfeat.delineate import (
    _R_FIRST_POSITIVE_FRACTION,
    _clinical_r_peak_local,
)


def _lobe(centre, amplitude, width, n=200):
    x = np.arange(n, dtype=float)
    return amplitude * np.exp(-((x - centre) ** 2) / (2.0 * width**2))


def test_taller_terminal_r_prime_does_not_displace_the_initial_r() -> None:
    """rSR': the first lobe is the R wave even when R' is taller."""
    seg = _lobe(40, 0.50, 5) - _lobe(60, 0.90, 6) + _lobe(85, 0.60, 5)

    index = _clinical_r_peak_local(seg)

    assert abs(index - 40) <= 3, f"expected the initial r near 40, got {index}"
    assert index != int(np.argmax(seg))


def test_a_small_early_bump_does_not_displace_a_dominant_later_r() -> None:
    """Below the fraction gate the early lobe is treated as noise, not an r."""
    seg = _lobe(40, 0.08, 4) - _lobe(60, 0.90, 6) + _lobe(85, 0.60, 5)

    index = _clinical_r_peak_local(seg)

    assert abs(index - 85) <= 3
    assert index == int(np.argmax(seg))


def test_single_positive_lobe_is_returned_unchanged() -> None:
    seg = _lobe(70, 0.60, 6) - _lobe(95, 0.90, 6)

    assert _clinical_r_peak_local(seg) == int(np.argmax(seg))


def test_pure_qs_complex_falls_back_to_the_global_maximum() -> None:
    """No positive deflection at all: nothing to promote, do not invent one."""
    seg = -_lobe(60, 0.90, 8)

    assert _clinical_r_peak_local(seg) == int(np.argmax(seg))


def test_absolute_floor_blocks_promotion_of_sub_millivolt_noise() -> None:
    """Two tiny lobes: the fraction gate alone would promote the first."""
    seg = _lobe(30, 0.030, 4) - _lobe(60, 0.90, 6) + _lobe(85, 0.032, 4)

    index = _clinical_r_peak_local(seg)

    assert abs(index - 85) <= 3, "a 0.03 mV bump is below the absolute floor"


def test_threshold_is_relative_to_the_tallest_positive_deflection() -> None:
    tall = 0.80
    just_over = tall * _R_FIRST_POSITIVE_FRACTION + 0.02
    just_under = tall * _R_FIRST_POSITIVE_FRACTION - 0.02

    over = _lobe(40, just_over, 5) - _lobe(60, 1.2, 6) + _lobe(85, tall, 5)
    under = _lobe(40, just_under, 5) - _lobe(60, 1.2, 6) + _lobe(85, tall, 5)

    assert abs(_clinical_r_peak_local(over) - 40) <= 4
    assert abs(_clinical_r_peak_local(under) - 85) <= 4


def test_empty_segment_is_handled() -> None:
    assert _clinical_r_peak_local(np.array([])) == 0
