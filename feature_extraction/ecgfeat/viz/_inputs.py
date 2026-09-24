"""Verify ``(signal, record)`` pairs and read record annotations.

Nothing here measures the ECG. Sample indices come from the record, and the
signal is only checked against the acquisition identity that the record
declares.
"""

from __future__ import annotations

import hashlib
import math
import operator
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ecgfeat.record import ECGRecord, SidecarError, query_measurement

from .errors import VisualizationInputError

RecordLike = Union[ECGRecord, Mapping[str, Any]]

#: The seven final landmarks of the ECG Record ``summary`` tier, in drawing order.
WAVE_FIDUCIALS = ("p_onset", "p_offset", "qrs_onset", "r_peak", "qrs_offset", "j_point", "t_offset")

FIDUCIALS = ("delineation", "fiducials")
INTERVALS = ("measurements", "intervals")


@dataclass(frozen=True, slots=True)
class Beat:
    index: int
    id: str
    r_sample: int | None


def member(document: Mapping[str, Any], *path: str) -> Any:
    value: Any = document
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value


def _incomplete(pointer: str, what: str) -> VisualizationInputError:
    return VisualizationInputError(
        f"the record does not declare a valid {pointer} ({what}); the signal cannot be matched to it",
        code="record_identity_incomplete",
        pointer=pointer,
    )


def _lead_list(value: Any, pointer: str) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise _incomplete(pointer, "a list of lead names")
    leads = tuple(value)
    if not leads or any(not isinstance(lead, str) for lead in leads) or len(set(leads)) != len(leads):
        raise _incomplete(pointer, "a non-empty list of unique lead names")
    return leads


class SignalRecord:
    """A signal verified against the acquisition identity of one ECG Record."""

    def __init__(self, signal: ArrayLike, record: RecordLike) -> None:
        if isinstance(record, ECGRecord):
            self.record: ECGRecord | None = record
            document: Mapping[str, Any] = record.as_dict()
        elif isinstance(record, Mapping):
            self.record = None
            document = record
        else:
            raise TypeError(
                "record must be an ecgfeat.record.ECGRecord or an ECG Record mapping, "
                f"not {type(record).__name__}"
            )
        self.document = document
        self.record_id = str(document.get("record_id", ""))

        rate = member(document, "acquisition", "sample_rate_hz")
        if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0:
            raise _incomplete("/acquisition/sample_rate_hz", "a positive sampling rate")
        self.fs = float(rate)
        count = member(document, "acquisition", "sample_count")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise _incomplete("/acquisition/sample_count", "a positive integer")
        self.sample_count = count

        acquisition_leads = _lead_list(member(document, "acquisition", "leads"), "/acquisition/leads")
        axis_leads = _lead_list(member(document, "axes", "leads"), "/axes/leads")
        if acquisition_leads is None and axis_leads is None:
            raise _incomplete("/acquisition/leads", "the lead order of the signal rows")
        if acquisition_leads is not None and axis_leads is not None and set(acquisition_leads) != set(axis_leads):
            raise VisualizationInputError(
                "/acquisition/leads and /axes/leads name different leads; the record is inconsistent",
                code="record_lead_axes_inconsistent",
            )
        #: Lead names in signal-row order.
        self.leads: tuple[str, ...] = acquisition_leads or axis_leads  # type: ignore[assignment]
        self._row = {lead: index for index, lead in enumerate(self.leads)}
        self._column = {lead: index for index, lead in enumerate(axis_leads or self.leads)}
        unit = member(document, "acquisition", "amplitude_unit")
        self.amplitude_unit = unit if isinstance(unit, str) and unit else "mV"

        self.signal = self._verified_signal(signal, document)
        self.beats = self._beat_axis(document)
        self._sidecar_cells: dict[tuple[str, int, str], Any] = {}

    # -- identity ---------------------------------------------------------

    def _verified_signal(self, signal: ArrayLike, document: Mapping[str, Any]) -> NDArray[np.float64]:
        try:
            array = np.asarray(signal)
        except (TypeError, ValueError) as exc:
            raise VisualizationInputError("signal must be a numeric array", code="signal_type") from exc
        if array.dtype.kind not in "biuf":
            raise VisualizationInputError(
                f"signal must be a real numeric array, got dtype {array.dtype}", code="signal_type"
            )
        n_leads = len(self.leads)
        if array.ndim != 2:
            raise VisualizationInputError(
                f"signal must be 2-D channel-major with shape ({n_leads}, {self.sample_count}); "
                f"got a {array.ndim}-D array",
                code="signal_shape_mismatch",
            )
        if array.shape != (n_leads, self.sample_count):
            problems = []
            if array.shape[0] != n_leads:
                problems.append(
                    f"{array.shape[0]} rows but the record declares {n_leads} leads ({', '.join(self.leads)})"
                )
            if array.shape[1] != self.sample_count:
                problems.append(f"{array.shape[1]} samples but /acquisition/sample_count is {self.sample_count}")
            hint = ""
            if array.shape == (self.sample_count, n_leads):
                hint = "; it looks sample-major, pass the transpose (one row per lead)"
            raise VisualizationInputError(
                "signal must be channel-major (one row per record lead): it has " + " and ".join(problems) + hint,
                code="signal_shape_mismatch",
                shape=list(array.shape),
                expected_shape=[n_leads, self.sample_count],
            )
        declared_shape = member(document, "artifacts", "raw_signal", "shape")
        if declared_shape is not None and list(declared_shape) != list(array.shape):
            raise VisualizationInputError(
                f"signal shape {list(array.shape)} differs from /artifacts/raw_signal/shape {list(declared_shape)}",
                code="signal_shape_mismatch",
            )
        declared = member(document, "artifacts", "raw_signal", "sha256")
        if not isinstance(declared, str) or not declared:
            raise _incomplete("/artifacts/raw_signal/sha256", "the raw-signal fingerprint")
        expected = declared.removeprefix("sha256:").lower()
        values = np.ascontiguousarray(array, dtype="<f8")
        actual = hashlib.sha256(values.tobytes()).hexdigest()
        if actual != expected:
            raise VisualizationInputError(
                "signal does not match the record: its SHA-256 "
                f"{actual[:12]}... differs from /artifacts/raw_signal/sha256 {expected[:12]}...; pass the exact "
                "array (same values, amplitude unit and row order) that produced this record",
                code="signal_fingerprint_mismatch",
                expected=f"sha256:{expected}",
                actual=f"sha256:{actual}",
            )
        return values

    def _beat_axis(self, document: Mapping[str, Any]) -> tuple[Beat, ...]:
        items = member(document, "axes", "beats")
        if items is None:
            return ()
        if isinstance(items, (str, bytes)) or not isinstance(items, Sequence):
            raise _incomplete("/axes/beats", "a list of beats")
        beats = []
        for index, item in enumerate(items):
            if not isinstance(item, Mapping):
                raise _incomplete(f"/axes/beats/{index}", "a beat object")
            beat_id = item.get("id")
            r_sample = self._sample(item.get("r_sample"), f"/axes/beats/{index}/r_sample")
            beats.append(Beat(index, str(beat_id) if beat_id is not None else f"b{index + 1:04d}", r_sample))
        return tuple(beats)

    def _sample(self, value: Any, pointer: str) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise VisualizationInputError(f"{pointer} is not an integer sample index: {value!r}",
                                          code="invalid_sample_index")
        value = int(value)
        if not 0 <= value < self.sample_count:
            raise VisualizationInputError(
                f"{pointer} = {value} lies outside the signal (0..{self.sample_count - 1})",
                code="sample_outside_acquisition",
            )
        return value

    # -- selectors --------------------------------------------------------

    def row(self, lead: str) -> int:
        return self._row[self.require_lead(lead)]

    def require_lead(self, lead: Any) -> str:
        if not isinstance(lead, str) or lead not in self._row:
            raise VisualizationInputError(
                f"unknown lead {lead!r}; the record declares {', '.join(self.leads)}",
                code="unknown_lead",
                lead=repr(lead),
            )
        return lead

    def select_leads(self, leads: Sequence[str] | None) -> tuple[str, ...]:
        if leads is None:
            return self.leads
        if isinstance(leads, (str, bytes)):
            raise VisualizationInputError(
                f"leads must be a sequence of lead names such as [{leads!r}], not a single string",
                code="unknown_lead",
            )
        try:
            selected = tuple(leads)
        except TypeError as exc:
            raise VisualizationInputError("leads must be a sequence of lead names", code="unknown_lead") from exc
        if not selected:
            raise VisualizationInputError("leads must name at least one lead", code="unknown_lead")
        for lead in selected:
            self.require_lead(lead)
        if len(set(selected)) != len(selected):
            raise VisualizationInputError(f"leads contains duplicates: {list(selected)}", code="duplicate_lead")
        return selected

    def require_beat(self, beat: Any) -> Beat:
        if isinstance(beat, bool):
            index = None
        else:
            try:
                index = operator.index(beat)
            except TypeError:
                index = None
        if index is None:
            raise VisualizationInputError(
                f"beat must be an integer index into /axes/beats, got {beat!r}", code="unknown_beat"
            )
        if not 0 <= index < len(self.beats):
            span = f"0..{len(self.beats) - 1}" if self.beats else "none"
            raise VisualizationInputError(
                f"beat {index} does not exist; the record has {len(self.beats)} beats (indices {span})",
                code="unknown_beat",
                beat=index,
            )
        return self.beats[index]

    # -- record fields ----------------------------------------------------

    def has_field(self, group: tuple[str, str], name: str) -> bool:
        return isinstance(member(self.document, *group, name), Mapping)

    def cell(self, group: tuple[str, str], name: str, beat: int, lead: str) -> Any:
        """Value of a ``["beat", "lead"]`` field cell, or ``None`` when absent."""

        pointer = f"/{group[0]}/{group[1]}/{name}"
        entry = member(self.document, *group, name)
        if not isinstance(entry, Mapping):
            return None
        if list(entry.get("axes", ())) != ["beat", "lead"]:
            raise VisualizationInputError(f"{pointer} is not a beat x lead field", code="record_field_axes")
        if "sidecar" in entry:
            return self._sidecar_cell(pointer, name, beat, lead)
        try:
            return entry["values"][beat][self._column[lead]]
        except (KeyError, IndexError, TypeError) as exc:
            raise VisualizationInputError(
                f"{pointer}/values does not match the /axes beat x lead shape", code="record_field_shape"
            ) from exc

    def _sidecar_cell(self, pointer: str, name: str, beat: int, lead: str) -> Any:
        if self.record is None:
            raise VisualizationInputError(
                f"{pointer} is stored in an NPZ sidecar, which a plain record mapping cannot resolve; "
                "load the record with ecgfeat.record.load_record(path) (or load_record(document, sidecar=...)) "
                "and pass the resulting ECGRecord",
                code="sidecar_requires_loaded_record",
                pointer=pointer,
            )
        key = (name, beat, lead)
        if key not in self._sidecar_cells:
            try:
                result = query_measurement(self.record, name, lead=lead, beat=beat)
            except SidecarError as exc:
                if exc.code != "sidecar_missing":
                    raise
                raise VisualizationInputError(
                    f"{pointer} is stored in an NPZ sidecar that is not attached to this ECGRecord; load it with "
                    "ecgfeat.record.load_record(path) next to its sidecar, or pass sidecar=",
                    code="sidecar_requires_loaded_record",
                    pointer=pointer,
                ) from exc
            self._sidecar_cells[key] = result.value
        return self._sidecar_cells[key]

    def fiducial(self, name: str, beat: int, lead: str) -> int | None:
        value = self.cell(FIDUCIALS, name, beat, lead)
        return self._sample(value, f"/delineation/fiducials/{name}/values/{beat}/{self._column[lead]}")

    def beat_fiducials(self, beat: int, lead: str) -> dict[str, int]:
        """Published landmarks present for one beat on one lead."""

        found = {}
        for name in WAVE_FIDUCIALS:
            if self.has_field(FIDUCIALS, name):
                sample = self.fiducial(name, beat, lead)
                if sample is not None:
                    found[name] = sample
        return found

    def lead_fiducials(self, lead: str, beats: Iterable[Beat] | None = None) -> dict[str, list[tuple[int, int]]]:
        """``name -> [(beat index, sample), ...]`` for present cells on one lead."""

        chosen = tuple(self.beats if beats is None else beats)
        found: dict[str, list[tuple[int, int]]] = {}
        for name in WAVE_FIDUCIALS:
            if not self.has_field(FIDUCIALS, name):
                continue
            cells = []
            for beat in chosen:
                sample = self.fiducial(name, beat.index, lead)
                if sample is not None:
                    cells.append((beat.index, sample))
            found[name] = cells
        return found


__all__ = ["Beat", "FIDUCIALS", "INTERVALS", "RecordLike", "SignalRecord", "WAVE_FIDUCIALS", "member"]
