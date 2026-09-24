"""Colours, marker shapes and artist ids shared by every plot."""

from __future__ import annotations

from dataclasses import dataclass

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES_1 = "#2a78d6"

# Status colours are reserved for record quality and never reused for series.
STATUS_COLOURS = {"usable": "#0ca30c", "limited": "#fab219"}
STATUS_UNKNOWN = MUTED

#: Prefix of the ``gid`` of every artist this package adds to an Axes.
GID = "ecgfeat.viz:"


@dataclass(frozen=True, slots=True)
class MarkerStyle:
    label: str
    colour: str
    marker: str
    hollow: bool = False


# One categorical hue per wave, in the fixed slot order of the default palette;
# marker shape is the secondary encoding for onset/offset pairs.
FIDUCIAL_STYLES = {
    "p_onset": MarkerStyle("P onset", "#2a78d6", ">"),
    "p_offset": MarkerStyle("P offset", "#2a78d6", "<"),
    "qrs_onset": MarkerStyle("QRS onset", "#eb6834", ">"),
    "r_peak": MarkerStyle("R peak", "#1baf7a", "^"),
    "qrs_offset": MarkerStyle("QRS offset", "#eb6834", "<"),
    # Hollow and larger: the J point often coincides with the QRS offset.
    "j_point": MarkerStyle("J point", "#eda100", "D", hollow=True),
    "t_offset": MarkerStyle("T offset", "#e87ba4", "s"),
}


def fiducial_gid(name: str) -> str:
    return f"{GID}fiducial:{name}"


BEATS_GID = f"{GID}beats"
BEAT_LABEL_GID = f"{GID}beat_label"
BEAT_OVERLAY_GID = f"{GID}beat_overlay"
MEDIAN_BEAT_GID = f"{GID}median_beat"
TRACE_GID = f"{GID}trace"
RMS_GID = f"{GID}rms"


def status_gid(status: str) -> str:
    return f"{GID}quality_status:{status}"
