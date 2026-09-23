"""Plots of a raw ECG signal together with the ECG Record measured from it.

Every function takes ``(signal, record)`` first: ``signal`` is the exact
channel-major array (one row per record lead, same values, amplitude unit and
row order) that produced ``record``; ``record`` is an
:class:`ecgfeat.record.ECGRecord` or its JSON mapping. The pair is verified
against ``/acquisition`` and ``/artifacts/raw_signal/sha256`` before anything
is drawn, and a mismatch raises :class:`VisualizationInputError`.

Annotations come only from the record (native sample indices); nothing is
measured here. Absent cells (JSON ``null``, whatever the absence reason) are
not drawn.

Only the object-oriented Matplotlib API is used: ``pyplot`` is never imported
and no global figure state is read or changed. With ``figure=None`` a new
:class:`~matplotlib.figure.Figure` (constrained layout) is created; a supplied
``Figure`` or ``SubFigure`` only receives new axes, and figure-level
decorations (suptitle, figure legend) are added only to figures the function
created. Showing, saving and closing the figure is the caller's job.

Artists added by this package carry a ``gid`` starting with
``"ecgrecords_viz:"`` (for example ``"ecgrecords_viz:fiducial:r_peak"``) so
callers can find and restyle them.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure, FigureBase
from matplotlib.lines import Line2D
from numpy.typing import ArrayLike

from . import _style as style
from ._inputs import INTERVALS, WAVE_FIDUCIALS, Beat, RecordLike, SignalRecord, member
from .errors import VisualizationInputError

if TYPE_CHECKING:
    from matplotlib.axes import Axes

AnnotationMode = Literal["none", "beats", "waves", "all"]
_ANNOTATION_MODES = ("none", "beats", "waves", "all")

#: Window around each beat's ``r_sample`` used for single-beat views and the
#: median representative beat, in milliseconds.
BEAT_WINDOW_MS = (-250.0, 450.0)
_LANDMARK_PAD_MS = 40.0


# -- shared helpers ---------------------------------------------------------


def _figure(figure: Any, figsize: tuple[float, float]) -> tuple[Any, bool]:
    if figure is None:
        return Figure(figsize=figsize, layout="constrained"), True
    if not isinstance(figure, FigureBase):
        raise TypeError(f"figure must be a matplotlib Figure or SubFigure, not {type(figure).__name__}")
    return figure, False


def _style_axes(ax: Axes) -> None:
    ax.grid(True, color=style.GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(style.BASELINE)
    ax.tick_params(labelsize=7, colors=style.INK_SECONDARY)


def _marker_kwargs(name: str) -> dict[str, Any]:
    spec = style.FIDUCIAL_STYLES[name]
    kwargs: dict[str, Any] = {"linestyle": "none", "marker": spec.marker, "color": spec.colour, "label": spec.label}
    if spec.hollow:
        kwargs.update(markersize=8, markerfacecolor="none", markeredgecolor=spec.colour, markeredgewidth=1.4)
    else:
        kwargs.update(markersize=6, markeredgecolor=style.SURFACE, markeredgewidth=0.8)
    return kwargs


def _draw_markers(ax: Axes, name: str, xs: Sequence[float] | np.ndarray, ys: Sequence[float] | np.ndarray) -> Line2D:
    (line,) = ax.plot(xs, ys, zorder=4, gid=style.fiducial_gid(name), **_marker_kwargs(name))
    return line


def _legend_handles(names: Collection[str], *, beats: bool = False) -> list[Line2D]:
    handles = [Line2D([], [], **_marker_kwargs(name)) for name in WAVE_FIDUCIALS if name in names]
    if beats:
        handles.append(Line2D([], [], color=style.MUTED, linestyle=":", linewidth=0.8, label="beat (R sample)"))
    return handles


def _add_legend(fig: Any, axes: Sequence[Axes], handles: list[Line2D], created: bool) -> None:
    if not handles:
        return
    if created:
        fig.legend(handles=handles, loc="outside lower center", ncols=min(len(handles), 8), fontsize=7,
                   frameon=False)
    else:
        axes[0].legend(handles=handles, loc="upper right", ncols=min(len(handles), 4), fontsize=6,
                       framealpha=0.85)


def _short_id(record_id: str) -> str:
    return record_id if len(record_id) <= 20 else record_id[:16] + "…"


def _finite(value: Any, name: str) -> float:
    if isinstance(value, (bool, str, bytes)):
        raise VisualizationInputError(f"{name} must be a number of seconds, got {value!r}", code="invalid_time_window")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise VisualizationInputError(f"{name} must be a number of seconds, got {value!r}",
                                      code="invalid_time_window") from exc
    if not math.isfinite(number):
        raise VisualizationInputError(f"{name} must be finite, got {value!r}", code="invalid_time_window")
    return number


def _time_window(pair: SignalRecord, start_s: float | None, end_s: float | None) -> tuple[int, int]:
    duration = pair.sample_count / pair.fs
    start = 0.0 if start_s is None else _finite(start_s, "start_s")
    end = duration if end_s is None else _finite(end_s, "end_s")
    if start < 0 or end > duration + 0.5 / pair.fs or start >= end:
        raise VisualizationInputError(
            f"time window [{start:g}, {end:g}] s must satisfy 0 <= start_s < end_s <= {duration:g}",
            code="invalid_time_window",
        )
    first = int(math.ceil(start * pair.fs - 1e-9))
    stop = min(pair.sample_count, int(math.floor(end * pair.fs + 1e-9)) + 1)
    if stop - first < 2:
        raise VisualizationInputError(f"time window [{start:g}, {end:g}] s contains fewer than two samples",
                                      code="invalid_time_window")
    return first, stop


def _anchored(beat: Beat) -> Beat:
    if beat.r_sample is None:
        raise VisualizationInputError(f"beat {beat.index} ({beat.id}) has no r_sample on /axes/beats",
                                      code="beat_without_r_sample")
    return beat


def _beat_window(pair: SignalRecord, r_sample: int, landmarks: Iterable[int]) -> tuple[int, int]:
    """Samples ``[first, stop)`` shown for one beat: -250..+450 ms around R, widened to its landmarks."""

    first = r_sample + int(round(BEAT_WINDOW_MS[0] * pair.fs / 1000.0))
    last = r_sample + int(round(BEAT_WINDOW_MS[1] * pair.fs / 1000.0))
    samples = list(landmarks)
    if samples:
        pad = int(round(_LANDMARK_PAD_MS * pair.fs / 1000.0))
        first = min(first, min(samples) - pad)
        last = max(last, max(samples) + pad)
    return max(0, first), min(pair.sample_count - 1, last) + 1


def _draw_beat(ax: Axes, pair: SignalRecord, lead: str, beat: Beat, first: int, stop: int,
               landmarks: Mapping[str, int]) -> None:
    row = pair.row(lead)
    r_sample = beat.r_sample
    assert r_sample is not None
    ms = (np.arange(first, stop) - r_sample) * 1000.0 / pair.fs
    _style_axes(ax)
    ax.plot(ms, pair.signal[row, first:stop], color=style.INK, linewidth=1.0, zorder=2, gid=style.TRACE_GID)
    for name in WAVE_FIDUCIALS:
        if name in landmarks:
            sample = landmarks[name]
            _draw_markers(ax, name, [(sample - r_sample) * 1000.0 / pair.fs], [pair.signal[row, sample]])
    if ms.size > 1:
        ax.set_xlim(ms[0], ms[-1])


def _interval_text(pair: SignalRecord, beat: Beat, lead: str) -> list[str]:
    parts = []
    for name, label in (("pr_interval_ms", "PR"), ("qrs_duration_ms", "QRS"), ("qt_interval_ms", "QT")):
        if not pair.has_field(INTERVALS, name):
            continue
        value = pair.cell(INTERVALS, name, beat.index, lead)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            unit = member(pair.document, *INTERVALS, name, "unit")
            parts.append(f"{label} {value:g} {unit or ''}".rstrip())
    return parts


# -- public API ---------------------------------------------------------------


def plot_record(
    signal: ArrayLike,
    record: RecordLike,
    *,
    leads: Sequence[str] | None = None,
    start_s: float | None = None,
    end_s: float | None = None,
    annotations: AnnotationMode = "waves",
    figure: Figure | None = None,
) -> tuple[Figure, tuple[Axes, ...]]:
    """Plot selected ECG leads with record-backed annotations and return the figure and axes.

    One axes per lead (stacked, shared time axis in seconds), in the order of
    ``leads`` (default: every record lead in signal-row order). ``start_s`` and
    ``end_s`` select a time window (default: the whole record).

    ``annotations`` selects what is drawn from the record:

    * ``"waves"`` -- per lead, a marker on the trace at each published
      landmark sample: P onset/offset, QRS onset/offset, R peak, J point and
      T offset (``/delineation/fiducials/<name>/values[beat][lead]``);
    * ``"beats"`` -- a dotted line at every ``/axes/beats[*].r_sample`` on each
      axes, labelled with its beat index (the ``beat=`` selector of
      :func:`plot_beat`) on the top axes;
    * ``"all"`` -- both; ``"none"`` -- the traces only.

    Returns ``(figure, axes)`` with ``axes`` in lead order.
    """

    pair = SignalRecord(signal, record)
    selected = pair.select_leads(leads)
    if annotations not in _ANNOTATION_MODES:
        raise VisualizationInputError(
            f"annotations must be one of {', '.join(_ANNOTATION_MODES)}; got {annotations!r}",
            code="invalid_annotations",
        )
    first, stop = _time_window(pair, start_s, end_s)
    show_waves = annotations in ("waves", "all")
    show_beats = annotations in ("beats", "all")
    beats_in_view = [b for b in pair.beats if b.r_sample is not None and first <= b.r_sample < stop]

    fig, created = _figure(figure, (11.0, max(2.4, 0.85 * len(selected) + 1.2)))
    grid = fig.subplots(len(selected), 1, sharex=True, squeeze=False)
    axes: tuple[Axes, ...] = tuple(grid[:, 0])
    seconds = np.arange(first, stop) / pair.fs
    drawn: set[str] = set()
    for ax, lead in zip(axes, selected):
        row = pair.row(lead)
        _style_axes(ax)
        ax.plot(seconds, pair.signal[row, first:stop], color=style.INK, linewidth=0.8, zorder=2,
                gid=style.TRACE_GID)
        ax.set_ylabel(lead, rotation=0, ha="right", va="center", fontsize=8)
        if show_beats and beats_in_view:
            ax.vlines([b.r_sample / pair.fs for b in beats_in_view], 0, 1,  # type: ignore[operator]
                      transform=ax.get_xaxis_transform(), colors=style.MUTED, linestyles=":", linewidth=0.8,
                      zorder=1, gid=style.BEATS_GID)
        if show_waves:
            for name, cells in pair.lead_fiducials(lead).items():
                samples = np.array([sample for _, sample in cells if first <= sample < stop], dtype=np.intp)
                if samples.size:
                    _draw_markers(ax, name, samples / pair.fs, pair.signal[row, samples])
                    drawn.add(name)
    labelled = show_beats and bool(beats_in_view)
    if labelled:
        # Just above the top spine, where R-peak markers cannot hide them.
        for beat in beats_in_view:
            axes[0].text(beat.r_sample / pair.fs, 1.0, str(beat.index),  # type: ignore[operator]
                         transform=axes[0].get_xaxis_transform(), ha="center", va="bottom", fontsize=6,
                         color=style.MUTED, gid=style.BEAT_LABEL_GID)
    axes[0].set_xlim(first / pair.fs, (stop - 1) / pair.fs)
    axes[-1].set_xlabel("Time (s)", fontsize=8)
    axes[0].set_title(
        f"{_short_id(pair.record_id)} · {pair.fs:g} Hz · amplitude in {pair.amplitude_unit}"
        + (" · numbers mark beat indices" if labelled else ""),
        loc="left", fontsize=9, pad=12 if labelled else 6,
    )
    _add_legend(fig, axes, _legend_handles(drawn, beats=show_beats and bool(beats_in_view)), created)
    return fig, axes


def plot_beat(
    signal: ArrayLike,
    record: RecordLike,
    *,
    beat: int,
    lead: str,
    figure: Figure | None = None,
) -> tuple[Figure, Axes]:
    """Plot one beat on one lead using boundaries and measurements from the record.

    ``beat`` is a zero-based index into ``/axes/beats``. The view spans
    -250..+450 ms around that beat's ``r_sample`` (time 0), widened so every
    published landmark of the beat on ``lead`` is visible. Landmarks are drawn
    as markers on the trace; PR, QRS and QT from ``/measurements/intervals``
    for the same cell are shown in the title when the record has them.
    """

    pair = SignalRecord(signal, record)
    lead = pair.require_lead(lead)
    chosen = _anchored(pair.require_beat(beat))
    assert chosen.r_sample is not None
    landmarks = pair.beat_fiducials(chosen.index, lead)
    first, stop = _beat_window(pair, chosen.r_sample, landmarks.values())

    fig, _created = _figure(figure, (7.5, 3.2))
    ax = fig.add_subplot()
    _draw_beat(ax, pair, lead, chosen, first, stop, landmarks)
    ax.set_xlabel("Time from beat R (ms)", fontsize=8)
    ax.set_ylabel(pair.amplitude_unit, fontsize=8)
    title = [f"Lead {lead} · beat {chosen.index} ({chosen.id}) · R at {chosen.r_sample / pair.fs:.3f} s"]
    ax.set_title(" · ".join(title + _interval_text(pair, chosen, lead)), loc="left", fontsize=9)
    handles = _legend_handles(landmarks)
    if handles:
        ax.legend(handles=handles, loc="best", fontsize=7, framealpha=0.85)
    return fig, ax


def plot_beat_all_leads(
    signal: ArrayLike,
    record: RecordLike,
    *,
    beat: int,
    leads: Sequence[str] | None = None,
    figure: Figure | None = None,
) -> tuple[Figure, tuple[Axes, ...]]:
    """Plot one beat across selected leads and return the figure and axes.

    Leads fill a grid column by column, three per column, so the twelve
    standard leads take the conventional I-III / aVR-aVF / V1-V3 / V4-V6
    layout. All axes share one time axis in milliseconds from the beat's
    ``r_sample``; the window is -250..+450 ms widened to the beat's landmarks
    on every selected lead. Returns the axes in lead order.
    """

    pair = SignalRecord(signal, record)
    selected = pair.select_leads(leads)
    chosen = _anchored(pair.require_beat(beat))
    assert chosen.r_sample is not None
    landmarks = {lead: pair.beat_fiducials(chosen.index, lead) for lead in selected}
    first, stop = _beat_window(pair, chosen.r_sample,
                               [sample for found in landmarks.values() for sample in found.values()])

    count = len(selected)
    nrows = min(3, count)
    ncols = math.ceil(count / nrows)
    fig, created = _figure(figure, (3.0 * ncols + 0.6, 2.1 * nrows + 0.9))
    grid = fig.subplots(nrows, ncols, sharex=True, squeeze=False)
    axes = []
    for position, lead in enumerate(selected):
        ax = grid[position % nrows, position // nrows]
        _draw_beat(ax, pair, lead, chosen, first, stop, landmarks[lead])
        ax.set_title(lead, loc="left", fontsize=8)
        axes.append(ax)
    for position in range(count, nrows * ncols):
        grid[position % nrows, position // nrows].remove()
    for column in range(ncols):
        last = min(count, (column + 1) * nrows) - 1
        bottom = grid[last % nrows, column]
        bottom.xaxis.set_tick_params(labelbottom=True)
        bottom.set_xlabel("ms from R", fontsize=7)
    if created:
        fig.suptitle(f"Beat {chosen.index} ({chosen.id}) · R at {chosen.r_sample / pair.fs:.3f} s",
                     fontsize=10)
    drawn = {name for found in landmarks.values() for name in found}
    _add_legend(fig, axes, _legend_handles(drawn), created)
    return fig, tuple(axes)


def plot_representative_beat(
    signal: ArrayLike,
    record: RecordLike,
    *,
    lead: str,
    figure: Figure | None = None,
) -> tuple[Figure, Axes]:
    """Plot the representative beat for one lead: the median of the record's beats.

    An ECG Record stores no representative waveform, so this view is computed
    here from the signal, using only record-defined alignment:

    * every beat on ``/axes/beats`` is aligned on its ``r_sample`` and cut to
      -250..+450 ms; beats whose window does not fit inside the signal (and
      beats without an ``r_sample``) are left out;
    * the sample-wise median of those segments is drawn over the individual
      aligned beats (thin grey lines);
    * for each published landmark (P onset/offset, QRS onset/offset, R peak,
      J point, T offset) the offset from R of every included beat's value on
      ``lead`` is taken from the record, and the marker is placed at the
      median offset on the median waveform. Absent cells are ignored, and a
      landmark with no present cell is not drawn.

    Every beat on the beat axis is included regardless of morphology (the
    record does not classify beats here), so ectopic beats are part of the
    median. The markers summarize record values; they are not new
    measurements of the median beat.
    """

    pair = SignalRecord(signal, record)
    lead = pair.require_lead(lead)
    row = pair.row(lead)
    before = int(round(-BEAT_WINDOW_MS[0] * pair.fs / 1000.0))
    after = int(round(BEAT_WINDOW_MS[1] * pair.fs / 1000.0))
    used = [b for b in pair.beats
            if b.r_sample is not None and b.r_sample - before >= 0 and b.r_sample + after < pair.sample_count]
    if not used:
        raise VisualizationInputError(
            f"no beat on /axes/beats has a complete {BEAT_WINDOW_MS[0]:g}..+{BEAT_WINDOW_MS[1]:g} ms window "
            "inside the signal",
            code="no_complete_beat_window",
        )
    offsets = np.arange(-before, after + 1)
    ms = offsets * 1000.0 / pair.fs
    segments = np.stack([pair.signal[row, b.r_sample - before:b.r_sample + after + 1]  # type: ignore[operator]
                         for b in used])
    median = np.median(segments, axis=0)

    fig, _created = _figure(figure, (7.5, 3.4))
    ax = fig.add_subplot()
    _style_axes(ax)
    ax.add_collection(LineCollection([np.column_stack([ms, segment]) for segment in segments],
                                     colors=style.BASELINE, linewidths=0.6, zorder=1, gid=style.BEAT_OVERLAY_GID))
    ax.plot(ms, median, color=style.INK, linewidth=1.4, zorder=2, gid=style.MEDIAN_BEAT_GID)
    r_of = {b.index: b.r_sample for b in used}
    drawn = set()
    for name, cells in pair.lead_fiducials(lead, used).items():
        if not cells:
            continue
        offset = float(np.median([sample - r_of[index] for index, sample in cells]))  # type: ignore[operator]
        if -before <= offset <= after:
            _draw_markers(ax, name, [offset * 1000.0 / pair.fs], [float(np.interp(offset, offsets, median))])
            drawn.add(name)
    ax.autoscale_view()
    ax.set_xlim(ms[0], ms[-1])
    ax.set_xlabel("Time from R (ms)", fontsize=8)
    ax.set_ylabel(pair.amplitude_unit, fontsize=8)
    ax.set_title(f"Median beat · lead {lead} · {len(used)} of {len(pair.beats)} beats aligned on R",
                 loc="left", fontsize=9)
    handles = [Line2D([], [], color=style.INK, linewidth=1.4, label="median beat"),
               Line2D([], [], color=style.BASELINE, linewidth=0.6, label="aligned beats")]
    ax.legend(handles=handles + _legend_handles(drawn), loc="best", fontsize=7, framealpha=0.85)
    return fig, ax


def plot_quality_summary(
    signal: ArrayLike,
    record: RecordLike,
    *,
    leads: Sequence[str] | None = None,
    figure: Figure | None = None,
) -> tuple[Figure, tuple[Axes, ...]]:
    """Plot signal-quality measurements already present in the record.

    Returns ``(figure, (status_axes, rms_axes))``:

    * ``status_axes`` places one marker per lead at its
      ``/quality/leads/<lead>/status`` (``usable`` / ``limited``; a lead the
      record does not grade is shown as ``unknown``); the title carries the
      record grade and status from ``/quality/record``;
    * ``rms_axes`` shows each lead's signal RMS about its mean (the standard
      deviation of the samples, in the record's amplitude unit), computed from
      the verified signal.
    """

    pair = SignalRecord(signal, record)
    selected = pair.select_leads(leads)
    quality = member(pair.document, "quality")
    quality = quality if isinstance(quality, Mapping) else {}
    by_lead = quality.get("leads")
    by_lead = by_lead if isinstance(by_lead, Mapping) else {}
    statuses = []
    for lead in selected:
        entry = by_lead.get(lead)
        status = entry.get("status") if isinstance(entry, Mapping) else None
        statuses.append(status if isinstance(status, str) and status else "unknown")
    levels = (["unknown"] if "unknown" in statuses else []) + sorted(
        set(statuses) - {"unknown", "limited", "usable"}) + ["limited", "usable"]
    rms = np.array([float(np.std(pair.signal[pair.row(lead)])) for lead in selected])
    x = np.arange(len(selected))

    fig, _created = _figure(figure, (max(6.0, 0.5 * len(selected) + 2.5), 4.4))
    grid = fig.subplots(2, 1, sharex=True, squeeze=False, gridspec_kw={"height_ratios": [1, 2]})
    status_ax, rms_ax = grid[0, 0], grid[1, 0]
    _style_axes(status_ax)
    _style_axes(rms_ax)
    for level_index, status in enumerate(levels):
        members = [i for i, value in enumerate(statuses) if value == status]
        if members:
            status_ax.scatter(x[members], np.full(len(members), level_index), s=60, zorder=3,
                              color=style.STATUS_COLOURS.get(status, style.STATUS_UNKNOWN),
                              edgecolors=style.SURFACE, linewidths=0.8, label=status,
                              gid=style.status_gid(status))
    status_ax.set_yticks(range(len(levels)), levels, fontsize=7)
    status_ax.set_ylim(-0.6, len(levels) - 0.4)
    record_quality = quality.get("record")
    record_quality = record_quality if isinstance(record_quality, Mapping) else {}
    grade, status = record_quality.get("grade"), record_quality.get("status")
    headline = f"Record quality {grade}" if grade is not None else "Record quality grade not reported"
    if status is not None:
        headline += f" ({status})"
    status_ax.set_title(f"{headline} · {_short_id(pair.record_id)}", loc="left", fontsize=9)
    status_ax.set_ylabel("lead status", fontsize=8)

    rms_ax.bar(x, rms, width=0.6, color=style.SERIES_1, zorder=2, gid=style.RMS_GID)
    rms_ax.set_ylabel(f"RMS about mean ({pair.amplitude_unit})", fontsize=8)
    rms_ax.set_xticks(x, list(selected), fontsize=8)
    rms_ax.set_xlim(-0.6, len(selected) - 0.4)
    return fig, (status_ax, rms_ax)


__all__ = [
    "plot_record",
    "plot_beat",
    "plot_beat_all_leads",
    "plot_representative_beat",
    "plot_quality_summary",
]
