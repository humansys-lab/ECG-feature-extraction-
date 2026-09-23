"""Deterministic synthetic 12-lead reference ECG (10 s, 500 Hz, 10 beats).

No third-party data: every sample is computed here, so the fixture can be
redistributed with the repository.  Each beat is a sum of Gaussian P, Q, R,
S and T components; a fixed cardiac dipole direction per wave is projected
onto approximate frontal/horizontal lead vectors.  The shape is only meant to
be ECG-like enough for stable delineation; it is not physiological evidence.

Run ``python make_signal.py`` to rewrite ``signal.npz`` (float64, mV).
"""

from pathlib import Path

import numpy as np

FS = 500
SECONDS = 10
LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
# Lead axes as unit vectors in (x: left, y: inferior, z: anterior).
_ANGLES = {"I": 0, "II": 60, "III": 120, "aVR": -150, "aVL": -30, "aVF": 90}
_PRECORDIAL = {"V1": (-0.35, 0.1, 0.93), "V2": (-0.1, 0.1, 0.99), "V3": (0.25, 0.15, 0.95),
               "V4": (0.55, 0.2, 0.8), "V5": (0.8, 0.2, 0.55), "V6": (0.95, 0.2, 0.2)}
# (centre offset from R in s, width s, amplitude mV, dipole direction)
_WAVES = (
    (-0.160, 0.022, 0.15, (0.6, 0.7, 0.2)),     # P
    (-0.028, 0.008, -0.10, (0.2, -0.3, 0.9)),   # Q
    (0.000, 0.011, 1.30, (0.55, 0.75, -0.35)),  # R
    (0.030, 0.010, -0.30, (-0.4, -0.5, 0.8)),   # S
    (0.300, 0.050, 0.35, (0.6, 0.7, 0.35)),     # T
)


def lead_vector(name):
    if name in _ANGLES:
        angle = np.deg2rad(_ANGLES[name])
        return np.array([np.cos(angle), np.sin(angle), 0.0])
    vector = np.array(_PRECORDIAL[name], dtype=float)
    return vector / np.linalg.norm(vector)


def make_signal():
    t = np.arange(FS * SECONDS) / FS
    r_times = 0.5 + np.arange(10) * 1.0
    signal = np.zeros((len(LEADS), t.size))
    for row, name in enumerate(LEADS):
        axis = lead_vector(name)
        for r in r_times:
            for offset, width, amplitude, direction in _WAVES:
                d = np.asarray(direction) / np.linalg.norm(direction)
                signal[row] += amplitude * float(axis @ d) * np.exp(-0.5 * ((t - r - offset) / width) ** 2)
    rng = np.random.default_rng(20260923)
    signal += rng.normal(0.0, 0.005, signal.shape)  # 5 uV white noise, fixed seed
    return signal


if __name__ == "__main__":
    np.savez_compressed(Path(__file__).with_name("signal.npz"), signal=make_signal())
