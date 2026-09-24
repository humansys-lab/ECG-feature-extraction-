"""Behaviour of the (signal, record) plotting API on the pinned reference record."""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.figure import Figure

import ecgrecords_viz
from ecgrecords_viz import (
    VisualizationInputError,
    plot_beat,
    plot_beat_all_leads,
    plot_quality_summary,
    plot_record,
    plot_representative_beat,
)

PREFIX = "ecgrecords_viz:"
LANDMARKS = ("p_onset", "p_offset", "qrs_onset", "r_peak", "qrs_offset", "j_point", "t_offset")
FS = 500.0


# -- helpers -------------------------------------------------------------------



def _suptitle(fig):
    """Figure.get_suptitle() exists from Matplotlib 3.8; the package supports 3.7."""
    if hasattr(fig, "get_suptitle"):
        return fig.get_suptitle()
    return fig._suptitle.get_text() if getattr(fig, "_suptitle", None) is not None else ""


def _labels_bottom(ax):
    """Whether bottom tick labels are shown (portable across Matplotlib 3.7+)."""
    return any(tick.label1.get_visible() for tick in ax.xaxis.get_major_ticks())

def fiducial_artists(ax: Axes) -> dict:
    tag = PREFIX + "fiducial:"
    found = {}
    for line in ax.lines:
        gid = line.get_gid() or ""
        if gid.startswith(tag):
            name = gid[len(tag):]
            assert name not in found, f"duplicate artist for {name}"
            found[name] = line
    return found


def gid_artists(ax: Axes, gid: str) -> list:
    return [artist for artist in ax.get_children() if artist.get_gid() == PREFIX + gid]


def lead_cells(document: dict, lead: str, first: int = 0, stop: int = 5000) -> dict[str, list[int]]:
    column = document["axes"]["leads"].index(lead)
    cells = {}
    for name in LANDMARKS:
        values = document["delineation"]["fiducials"][name]["values"]
        samples = [row[column] for row in values if row[column] is not None and first <= row[column] < stop]
        if samples:
            cells[name] = samples
    return cells


def beat_cells(document: dict, beat: int, lead: str) -> dict[str, int]:
    column = document["axes"]["leads"].index(lead)
    cells = {}
    for name in LANDMARKS:
        value = document["delineation"]["fiducials"][name]["values"][beat][column]
        if value is not None:
            cells[name] = value
    return cells


def annotation_data(axes) -> dict:
    data = {}
    for index, ax in enumerate(axes):
        for name, line in fiducial_artists(ax).items():
            data[(index, name)] = (tuple(np.asarray(line.get_xdata(), float)), tuple(np.asarray(line.get_ydata(), float)))
    return data


# -- public surface ---------------------------------------------------------------------


def test_public_names_and_version():
    assert set(ecgrecords_viz.__all__) == {
        "plot_record", "plot_beat", "plot_beat_all_leads", "plot_representative_beat", "plot_quality_summary",
        "VisualizationInputError", "__version__",
    }
    for name in ecgrecords_viz.__all__:
        assert hasattr(ecgrecords_viz, name)
    assert ecgrecords_viz.__version__ == "0.1.0"


def test_version_matches_pyproject():
    tomllib = pytest.importorskip("tomllib")
    pyproject = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["name"] == "ecg-records-viz"
    assert pyproject["project"]["version"] == ecgrecords_viz.__version__


def test_error_is_a_public_ecg_input_error_defined_in_the_viz_package():
    import ecgfeat.errors

    assert issubclass(VisualizationInputError, ecgfeat.errors.ECGInputError)
    assert issubclass(VisualizationInputError, ValueError)
    assert VisualizationInputError.__module__ == "ecgrecords_viz.errors"
    assert not hasattr(ecgfeat.errors, "VisualizationInputError")
    error = VisualizationInputError("boom", code="some_code", lead="II")
    assert error.to_dict() == {"code": "some_code", "message": "boom", "details": {"lead": "II"}}


EMPTY = inspect.Parameter.empty
SIGNATURES = {
    plot_record: {"leads": None, "start_s": None, "end_s": None, "annotations": "waves", "figure": None},
    plot_beat: {"beat": EMPTY, "lead": EMPTY, "figure": None},
    plot_beat_all_leads: {"beat": EMPTY, "leads": None, "figure": None},
    plot_representative_beat: {"lead": EMPTY, "figure": None},
    plot_quality_summary: {"leads": None, "figure": None},
}


@pytest.mark.parametrize("function", list(SIGNATURES), ids=lambda f: f.__name__)
def test_signature_matches_document_03(function):
    parameters = list(inspect.signature(function).parameters.values())
    assert [(p.name, p.kind) for p in parameters[:2]] == [
        ("signal", inspect.Parameter.POSITIONAL_OR_KEYWORD), ("record", inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    keyword_only = parameters[2:]
    assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in keyword_only)
    assert {p.name: p.default for p in keyword_only} == SIGNATURES[function]
    assert [p.name for p in keyword_only] == list(SIGNATURES[function])


# -- plot_record -----------------------------------------------------------------


def test_plot_record_draws_every_published_landmark_on_every_lead(signal, record, document):
    fig, axes = plot_record(signal, record)
    assert isinstance(fig, Figure)
    assert isinstance(axes, tuple) and len(axes) == 12 and all(isinstance(ax, Axes) for ax in axes)
    assert list(fig.axes) == list(axes)
    leads = document["acquisition"]["leads"]
    total_points = 0
    for row, (ax, lead) in enumerate(zip(axes, leads)):
        assert ax.get_ylabel() == lead
        traces = [line for line in ax.lines if line.get_gid() == PREFIX + "trace"]
        assert len(traces) == 1
        np.testing.assert_array_equal(traces[0].get_ydata(), signal[row])
        expected = lead_cells(document, lead)
        drawn = fiducial_artists(ax)
        assert set(drawn) == set(expected)
        for name, samples in expected.items():
            np.testing.assert_allclose(drawn[name].get_xdata(), np.array(samples) / FS)
            np.testing.assert_array_equal(drawn[name].get_ydata(), signal[row, samples])
            total_points += len(samples)
        assert not gid_artists(ax, "beats")
    # 7 landmarks x 10 beats x 12 leads, minus the 6 + 6 absent P onset/offset cells.
    assert total_points == 7 * 10 * 12 - 12
    assert not [t for t in axes[0].texts if t.get_gid() == PREFIX + "beat_label"]


@pytest.mark.parametrize("mode,waves,beats", [("none", False, False), ("beats", False, True),
                                              ("waves", True, False), ("all", True, True)])
def test_plot_record_annotation_modes(signal, record, document, mode, waves, beats):
    _, axes = plot_record(signal, record, annotations=mode)
    for ax, lead in zip(axes, document["acquisition"]["leads"]):
        assert bool(fiducial_artists(ax)) is waves
        markers = gid_artists(ax, "beats")
        if beats:
            assert len(markers) == 1 and isinstance(markers[0], LineCollection)
            assert len(markers[0].get_segments()) == 10
            np.testing.assert_allclose([segment[0, 0] for segment in markers[0].get_segments()],
                                       [beat["r_sample"] / FS for beat in document["axes"]["beats"]])
        else:
            assert markers == []
    labels = [t.get_text() for t in axes[0].texts if t.get_gid() == PREFIX + "beat_label"]
    assert labels == ([str(i) for i in range(10)] if beats else [])
    for ax in axes[1:]:
        assert not [t for t in ax.texts if t.get_gid() == PREFIX + "beat_label"]


def test_plot_record_window_and_lead_order(signal, record, document):
    fig, axes = plot_record(signal, record, leads=["V1", "II"], start_s=2.0, end_s=4.0, annotations="all")
    assert [ax.get_ylabel() for ax in axes] == ["V1", "II"]
    rows = {lead: i for i, lead in enumerate(document["acquisition"]["leads"])}
    for ax, lead in zip(axes, ["V1", "II"]):
        trace = next(line for line in ax.lines if line.get_gid() == PREFIX + "trace")
        np.testing.assert_array_equal(trace.get_ydata(), signal[rows[lead], 1000:2001])
        assert ax.get_xlim() == pytest.approx((2.0, 4.0))
        expected = lead_cells(document, lead, 1000, 2001)
        drawn = fiducial_artists(ax)
        assert {name: list(np.round(np.asarray(line.get_xdata()) * FS).astype(int)) for name, line in drawn.items()} \
            == expected
        (markers,) = gid_artists(ax, "beats")
        assert len(markers.get_segments()) == 2  # beats at samples 1250 and 1750
    assert [t.get_text() for t in axes[0].texts if t.get_gid() == PREFIX + "beat_label"] == ["2", "3"]


def test_record_mapping_and_ecgrecord_draw_identically(signal, record, document):
    _, from_record = plot_record(signal, record)
    _, from_mapping = plot_record(signal, document)
    assert annotation_data(from_record) == annotation_data(from_mapping)


def test_absent_cells_are_not_drawn(signal, document):
    column = document["axes"]["leads"].index("II")
    fiducials = document["delineation"]["fiducials"]
    fiducials["p_onset"]["values"][0][column] = None
    for row in fiducials["p_offset"]["values"]:
        row[column] = None
    _, axes = plot_record(signal, document, leads=["II"])
    drawn = fiducial_artists(axes[0])
    assert "p_offset" not in drawn
    assert len(drawn["p_onset"].get_xdata()) == len(lead_cells(document, "II")["p_onset"])
    assert len(drawn["p_onset"].get_xdata()) == sum(v[column] is not None for v in fiducials["p_onset"]["values"])


def test_fields_missing_from_the_record_are_not_drawn(signal, document):
    del document["delineation"]["fiducials"]["j_point"]
    _, axes = plot_record(signal, document, leads=["II"])
    assert "j_point" not in fiducial_artists(axes[0])
    _, ax = plot_beat(signal, document, beat=0, lead="II")
    assert "j_point" not in fiducial_artists(ax)


def test_inline_reads_agree_with_query_measurement(signal, record, document):
    """The inline fast path returns exactly what the public query API returns."""

    from ecgfeat.record import query_measurement

    from ecgrecords_viz._inputs import SignalRecord

    pair = SignalRecord(signal, record)
    leads = document["axes"]["leads"]
    cells = set()
    for name in LANDMARKS:
        for beat, row in enumerate(document["delineation"]["fiducials"][name]["values"]):
            for column, value in enumerate(row):
                if value is None or beat in (0, 9):
                    cells.add((name, beat, leads[column]))
    assert any(document["delineation"]["fiducials"][n]["values"][b][leads.index(lead)] is None for n, b, lead in cells)
    for name, beat, lead in sorted(cells):
        assert pair.fiducial(name, beat, lead) == query_measurement(record, name, lead=lead, beat=beat).value


# -- single beat views ------------------------------------------------------------


def test_plot_beat_marks_the_cell_landmarks_relative_to_r(signal, record, document):
    fig, ax = plot_beat(signal, record, beat=3, lead="II")
    assert isinstance(fig, Figure) and isinstance(ax, Axes) and fig.axes == [ax]
    r_sample = document["axes"]["beats"][3]["r_sample"]
    expected = beat_cells(document, 3, "II")
    drawn = fiducial_artists(ax)
    assert set(drawn) == set(expected) == set(LANDMARKS)
    for name, sample in expected.items():
        assert list(drawn[name].get_xdata()) == pytest.approx([(sample - r_sample) * 1000.0 / FS])
        assert list(drawn[name].get_ydata()) == [signal[1, sample]]
    title = ax.get_title(loc="left")
    assert "beat 3 (b0004)" in title
    column = document["axes"]["leads"].index("II")
    intervals = document["measurements"]["intervals"]
    for label, name in (("PR", "pr_interval_ms"), ("QRS", "qrs_duration_ms"), ("QT", "qt_interval_ms")):
        assert f"{label} {intervals[name]['values'][3][column]:g} ms" in title
    low, high = ax.get_xlim()
    assert low <= -250 + 1e-9 and high >= 450 - 1e-9


def test_plot_beat_all_leads_uses_the_standard_column_layout(signal, record, document):
    fig, axes = plot_beat_all_leads(signal, record, beat=5)
    leads = document["acquisition"]["leads"]
    assert len(axes) == 12 and set(fig.axes) == set(axes)
    assert [ax.get_title(loc="left") for ax in axes] == leads
    for position, ax in enumerate(axes):
        spec = ax.get_subplotspec()
        assert (spec.rowspan.start, spec.colspan.start) == (position % 3, position // 3)
        assert set(fiducial_artists(ax)) == set(beat_cells(document, 5, leads[position]))
    assert _suptitle(fig).startswith("Beat 5 (b0006)")
    limits = {ax.get_xlim() for ax in axes}
    assert len(limits) == 1  # one shared time axis


def test_plot_beat_all_leads_subset_removes_unused_cells(signal, record):
    fig, axes = plot_beat_all_leads(signal, record, beat=0, leads=["V1", "V2", "V3", "V4"])
    assert [ax.get_title(loc="left") for ax in axes] == ["V1", "V2", "V3", "V4"]
    assert len(fig.axes) == 4
    # V3 ends column one and V4 is alone in column two: both show their time ticks.
    assert _labels_bottom(axes[2]) and _labels_bottom(axes[3])


def test_plot_representative_beat_is_the_median_of_r_aligned_beats(signal, record, document):
    fig, ax = plot_representative_beat(signal, record, lead="V3")
    assert isinstance(fig, Figure) and isinstance(ax, Axes)
    row = document["acquisition"]["leads"].index("V3")
    rs = [beat["r_sample"] for beat in document["axes"]["beats"]]
    segments = np.stack([signal[row, r - 125:r + 226] for r in rs])
    (median,) = gid_artists(ax, "median_beat")
    np.testing.assert_allclose(median.get_xdata(), np.arange(-125, 226) * 2.0)
    np.testing.assert_array_equal(median.get_ydata(), np.median(segments, axis=0))
    (overlay,) = gid_artists(ax, "beat_overlay")
    assert len(overlay.get_segments()) == 10
    drawn = fiducial_artists(ax)
    column = document["axes"]["leads"].index("V3")
    for name in LANDMARKS:
        offsets = [row_values[column] - r for row_values, r in
                   zip(document["delineation"]["fiducials"][name]["values"], rs) if row_values[column] is not None]
        assert list(drawn[name].get_xdata()) == pytest.approx([np.median(offsets) * 2.0])
        assert len(drawn[name].get_ydata()) == 1
    assert "10 of 10 beats" in ax.get_title(loc="left")


def test_plot_representative_beat_skips_beats_without_a_full_window(signal, document):
    document["axes"]["beats"][0]["r_sample"] = 60  # -250 ms would start before sample 0
    _, ax = plot_representative_beat(signal, document, lead="II")
    assert "9 of 10 beats" in ax.get_title(loc="left")
    assert len(gid_artists(ax, "beat_overlay")[0].get_segments()) == 9
    for beat in document["axes"]["beats"]:
        beat["r_sample"] = 4990
    with pytest.raises(VisualizationInputError) as info:
        plot_representative_beat(signal, document, lead="II")
    assert info.value.code == "no_complete_beat_window"


def test_plot_quality_summary(signal, record, document):
    fig, axes = plot_quality_summary(signal, record)
    assert isinstance(axes, tuple) and len(axes) == 2 and all(isinstance(ax, Axes) for ax in axes)
    status_ax, rms_ax = axes
    (usable,) = [c for c in status_ax.collections if c.get_gid() == PREFIX + "quality_status:usable"]
    assert isinstance(usable, PathCollection) and len(usable.get_offsets()) == 12
    assert [t.get_text() for t in status_ax.get_yticklabels()] == ["limited", "usable"]
    assert "Q0" in status_ax.get_title(loc="left") and "usable" in status_ax.get_title(loc="left")
    bars = [patch for patch in rms_ax.patches if patch.get_gid() == PREFIX + "rms"]
    assert len(bars) == 12
    np.testing.assert_allclose([bar.get_height() for bar in bars], signal.std(axis=1))
    assert [t.get_text() for t in rms_ax.get_xticklabels()] == document["acquisition"]["leads"]


def test_plot_quality_summary_statuses_and_subset(signal, document):
    document["quality"]["leads"]["V1"] = {"status": "limited", "reason": "quality_gate"}
    del document["quality"]["leads"]["V2"]
    document["quality"]["record"] = {"status": "limited", "grade": "Q2"}
    _, (status_ax, rms_ax) = plot_quality_summary(signal, document, leads=["V1", "V2", "V3"])
    by_status = {c.get_gid().rsplit(":", 1)[1]: [tuple(o) for o in c.get_offsets()]
                 for c in status_ax.collections if (c.get_gid() or "").startswith(PREFIX + "quality_status:")}
    assert [t.get_text() for t in status_ax.get_yticklabels()] == ["unknown", "limited", "usable"]
    assert by_status == {"limited": [(0.0, 1.0)], "unknown": [(1.0, 0.0)], "usable": [(2.0, 2.0)]}
    assert "Q2 (limited)" in status_ax.get_title(loc="left")
    assert len([p for p in rms_ax.patches if p.get_gid() == PREFIX + "rms"]) == 3


# -- figures -----------------------------------------------------------------------

CALLS = {
    "plot_record": lambda s, r, **kw: plot_record(s, r, leads=["I", "II"], **kw),
    "plot_beat": lambda s, r, **kw: plot_beat(s, r, beat=1, lead="II", **kw),
    "plot_beat_all_leads": lambda s, r, **kw: plot_beat_all_leads(s, r, beat=1, leads=["I", "II", "V1"], **kw),
    "plot_representative_beat": lambda s, r, **kw: plot_representative_beat(s, r, lead="II", **kw),
    "plot_quality_summary": lambda s, r, **kw: plot_quality_summary(s, r, **kw),
}


def returned_axes(result) -> list:
    _, axes = result
    return list(axes) if isinstance(axes, tuple) else [axes]


@pytest.mark.parametrize("name", list(CALLS))
def test_supplied_figure_receives_the_axes(signal, record, name):
    figure = Figure()
    result = CALLS[name](signal, record, figure=figure)
    assert result[0] is figure
    first = returned_axes(result)
    assert first and all(ax.figure is figure for ax in first) and set(figure.axes) == set(first)
    assert _suptitle(figure) == "" and figure.legends == []
    again = CALLS[name](signal, record, figure=figure)
    assert again[0] is figure
    assert len(figure.axes) == 2 * len(first)


@pytest.mark.parametrize("name", list(CALLS))
def test_new_figure_and_subfigure(signal, record, name):
    created = CALLS[name](signal, record)
    assert isinstance(created[0], Figure) and created[0].get_layout_engine() is not None
    parent = Figure()
    left, _right = parent.subfigures(1, 2)
    result = CALLS[name](signal, record, figure=left)
    assert result[0] is left
    assert all(ax in left.axes for ax in returned_axes(result))


@pytest.mark.parametrize("name", list(CALLS))
def test_figure_argument_must_be_a_matplotlib_figure(signal, record, name):
    with pytest.raises(TypeError):
        CALLS[name](signal, record, figure="not a figure")


# -- identity checks -------------------------------------------------------------


def _mutate_one_sample(signal):
    changed = signal.copy()
    changed[4, 2500] += 1e-9
    return changed


BAD_SIGNALS = {
    "one_dimensional": (lambda s: s[0], "signal_shape_mismatch"),
    "three_dimensional": (lambda s: s[None], "signal_shape_mismatch"),
    "sample_major": (lambda s: s.T, "signal_shape_mismatch"),
    "missing_lead": (lambda s: s[:11], "signal_shape_mismatch"),
    "extra_lead": (lambda s: np.vstack([s, s[:1]]), "signal_shape_mismatch"),
    "short": (lambda s: s[:, :4999], "signal_shape_mismatch"),
    "one_sample_changed": (_mutate_one_sample, "signal_fingerprint_mismatch"),
    "rows_swapped": (lambda s: s[[1, 0, *range(2, 12)]], "signal_fingerprint_mismatch"),
    "microvolts": (lambda s: s * 1000.0, "signal_fingerprint_mismatch"),
    "float32": (lambda s: s.astype(np.float32), "signal_fingerprint_mismatch"),
    "strings": (lambda s: s.astype(str), "signal_type"),
    "objects": (lambda s: s.astype(object), "signal_type"),
}


@pytest.mark.parametrize("case", list(BAD_SIGNALS))
@pytest.mark.parametrize("name", list(CALLS))
def test_mismatched_signal_is_rejected(signal, record, case, name):
    transform, code = BAD_SIGNALS[case]
    with pytest.raises(VisualizationInputError) as info:
        CALLS[name](transform(signal), record)
    assert info.value.code == code


def test_sample_major_signal_gets_a_transpose_hint(signal, record):
    with pytest.raises(VisualizationInputError, match="transpose"):
        plot_record(signal.T, record)


def test_errors_are_catchable_as_the_public_root(signal, record):
    from ecgfeat.errors import ECGInputError

    with pytest.raises(ECGInputError):
        plot_record(signal[:, :100], record)


def _strided_view(signal):
    padded = np.zeros((signal.shape[0], 2 * signal.shape[1]))
    padded[:, ::2] = signal
    view = padded[:, ::2]
    assert not view.flags.c_contiguous
    return view


@pytest.mark.parametrize("convert", [
    lambda s: s.tolist(),
    np.asfortranarray,
    lambda s: s.astype(">f8"),
    _strided_view,
], ids=["list", "fortran", "big_endian", "strided"])
def test_equal_values_in_another_layout_are_accepted(signal, record, convert):
    _, axes = plot_record(convert(signal), record, leads=["II"])
    assert fiducial_artists(axes[0])


def test_digest_prefix_is_optional_but_the_digest_is_required(signal, document):
    raw = document["artifacts"]["raw_signal"]
    raw["sha256"] = raw["sha256"].removeprefix("sha256:").upper()
    plot_beat(signal, document, beat=0, lead="I")
    del raw["sha256"]
    with pytest.raises(VisualizationInputError) as info:
        plot_beat(signal, document, beat=0, lead="I")
    assert info.value.code == "record_identity_incomplete"


def test_declared_raw_signal_shape_must_agree(signal, document):
    document["artifacts"]["raw_signal"]["shape"] = [6, 10000]
    with pytest.raises(VisualizationInputError) as info:
        plot_record(signal, document)
    assert info.value.code == "signal_shape_mismatch"


@pytest.mark.parametrize("pointer", ["sample_rate_hz", "sample_count"])
def test_incomplete_acquisition_is_rejected(signal, document, pointer):
    del document["acquisition"][pointer]
    with pytest.raises(VisualizationInputError) as info:
        plot_record(signal, document)
    assert info.value.code == "record_identity_incomplete"


def test_fiducial_outside_the_signal_is_rejected(signal, document):
    document["delineation"]["fiducials"]["t_offset"]["values"][9][1] = 5000
    with pytest.raises(VisualizationInputError) as info:
        plot_record(signal, document, leads=["II"])
    assert info.value.code == "sample_outside_acquisition"


def test_record_must_be_a_record_or_mapping(signal):
    with pytest.raises(TypeError):
        plot_record(signal, ["not", "a", "record"])


# -- selectors -----------------------------------------------------------------


@pytest.mark.parametrize("call", [
    lambda s, r: plot_record(s, r, leads=["V7"]),
    lambda s, r: plot_record(s, r, leads="II"),
    lambda s, r: plot_record(s, r, leads=[]),
    lambda s, r: plot_record(s, r, leads=["II", "II"]),
    lambda s, r: plot_beat(s, r, beat=0, lead="ii"),
    lambda s, r: plot_beat(s, r, beat=0, lead=None),
    lambda s, r: plot_beat_all_leads(s, r, beat=0, leads=["I", "aVX"]),
    lambda s, r: plot_representative_beat(s, r, lead="MLII"),
    lambda s, r: plot_quality_summary(s, r, leads=["X"]),
], ids=["record", "string", "empty", "duplicate", "beat_case", "beat_none", "all_leads", "representative",
        "quality"])
def test_unknown_lead_selectors_raise(signal, record, call):
    with pytest.raises(VisualizationInputError) as info:
        call(signal, record)
    assert info.value.code in {"unknown_lead", "duplicate_lead"}


@pytest.mark.parametrize("beat", [10, -1, True, 1.0, "3", None])
@pytest.mark.parametrize("function", [plot_beat, plot_beat_all_leads], ids=lambda f: f.__name__)
def test_unknown_beat_selectors_raise(signal, record, function, beat):
    kwargs = {"lead": "II"} if function is plot_beat else {}
    with pytest.raises(VisualizationInputError) as info:
        function(signal, record, beat=beat, **kwargs)
    assert info.value.code == "unknown_beat"


def test_numpy_integer_beat_is_accepted(signal, record):
    _, ax = plot_beat(signal, record, beat=np.int64(2), lead="V2")
    assert "beat 2" in ax.get_title(loc="left")


@pytest.mark.parametrize("kwargs,code", [
    ({"annotations": "fiducials"}, "invalid_annotations"),
    ({"start_s": 4.0, "end_s": 4.0}, "invalid_time_window"),
    ({"start_s": -0.5}, "invalid_time_window"),
    ({"end_s": 10.5}, "invalid_time_window"),
    ({"start_s": float("nan")}, "invalid_time_window"),
    ({"end_s": "4"}, "invalid_time_window"),
])
def test_invalid_record_plot_arguments(signal, record, kwargs, code):
    with pytest.raises(VisualizationInputError) as info:
        plot_record(signal, record, **kwargs)
    assert info.value.code == code


# -- NPZ sidecar ----------------------------------------------------------------


def test_sidecar_backed_mapping_asks_for_a_loaded_record(signal, sidecar_files):
    import json

    document = json.loads((sidecar_files / "record.json").read_text(encoding="utf-8"))
    assert document["delineation"]["fiducials"]["r_peak"]["values"] is None
    with pytest.raises(VisualizationInputError, match="load_record") as info:
        plot_record(signal, document, leads=["II"])
    assert info.value.code == "sidecar_requires_loaded_record"
    # Traces alone need no dense fields.
    _, axes = plot_record(signal, document, leads=["II"], annotations="beats")
    assert gid_artists(axes[0], "beats")


def test_sidecar_backed_record_is_read_through_query_measurement(signal, record, sidecar_files):
    from ecgfeat.record import load_record

    loaded = load_record(sidecar_files / "record.json")
    assert loaded.delineation["fiducials"]["r_peak"]["values"] is None
    _, inline_axes = plot_record(signal, record, leads=["II", "V3"])
    _, sidecar_axes = plot_record(signal, loaded, leads=["II", "V3"])
    assert annotation_data(sidecar_axes) == annotation_data(inline_axes)
    _, inline_beat = plot_beat(signal, record, beat=4, lead="V3")
    _, sidecar_beat = plot_beat(signal, loaded, beat=4, lead="V3")
    assert annotation_data([sidecar_beat]) == annotation_data([inline_beat])
    assert sidecar_beat.get_title(loc="left") == inline_beat.get_title(loc="left")


def test_sidecar_backed_record_without_its_sidecar(signal, sidecar_files):
    import json

    from ecgfeat.record import load_record

    document = json.loads((sidecar_files / "record.json").read_text(encoding="utf-8"))
    detached = load_record(document, validate="none")
    with pytest.raises(VisualizationInputError) as info:
        plot_beat(signal, detached, beat=0, lead="II")
    assert info.value.code == "sidecar_requires_loaded_record"
