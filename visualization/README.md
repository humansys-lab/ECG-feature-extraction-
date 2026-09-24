# ecg-records-viz

Matplotlib plots of an ECG Record together with the raw signal it was measured
from. This is the optional visualization distribution of `ecg-records`
(import name `ecgfeat`), released and versioned separately so that the core
measurement package never depends on Matplotlib.

The plots read what the record already contains: fiducial sample indices,
the beat axis, and per-lead quality. They never run a measurement, change the
record, or import the interpretation engine. Every call takes `(signal, record)`
and returns `(figure, axes)`. Only the object-oriented Matplotlib API is used:
`matplotlib.pyplot` is never imported, and showing, saving or closing the figure
is up to you.

## Install

```bash
pip install ecg-records-viz
# or, through the core package's extra
pip install "ecg-records[viz]"
```

Python 3.10 to 3.13; requires `ecg-records>=0.1,<0.3`, `matplotlib>=3.7,<4`,
and `numpy>=1.26,<3`.

> This software is research and engineering software. It is not a medical device and is not intended to diagnose, treat, cure, or prevent disease. Outputs require independent validation for the intended use and must not be used as a substitute for professional medical judgment.

## Usage

```python
import numpy as np
from ecgfeat import ecg_record
from ecgrecords_viz import plot_beat, plot_record

signal = np.load("signal.npz")["signal"]        # (12, n_samples), mV, one row per lead
record = ecg_record(signal, sampling_rate=500, lead_names=[
    "I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6",
], profile="all")

fig, axes = plot_record(signal, record, leads=["II", "V1", "V5"],
                        start_s=1.0, end_s=4.0, annotations="all")
fig.savefig("record.png", dpi=150)

fig, ax = plot_beat(signal, record, beat=3, lead="II")
```

A record saved to disk works the same way. Use `ecgfeat.record.load_record(path)`
when the record keeps its dense matrices in an NPZ sidecar: the loaded
`ECGRecord` resolves the sidecar, but a plain `json.load` mapping cannot.

The same functions are available as `ecgfeat.viz.plot_record` and so on. They
are imported lazily, so `import ecgfeat` never loads Matplotlib.

| Function | Returns | Draws |
| --- | --- | --- |
| `plot_record(signal, record, *, leads=None, start_s=None, end_s=None, annotations="waves", figure=None)` | `(Figure, tuple[Axes, ...])` | stacked leads with `"waves"` (landmarks), `"beats"` (R samples with beat indices), `"all"` or `"none"` |
| `plot_beat(signal, record, *, beat, lead, figure=None)` | `(Figure, Axes)` | one beat on one lead, with the landmarks of that cell and PR/QRS/QT in the title |
| `plot_beat_all_leads(signal, record, *, beat, leads=None, figure=None)` | `(Figure, tuple[Axes, ...])` | one beat on every selected lead, in the standard 3 x 4 layout |
| `plot_representative_beat(signal, record, *, lead, figure=None)` | `(Figure, Axes)` | the median of all beats aligned on the record's R samples (-250 to +450 ms), with the median landmark offsets |
| `plot_quality_summary(signal, record, *, leads=None, figure=None)` | `(Figure, tuple[Axes, ...])` | per-lead quality status and signal RMS, with the record grade in the title |

Landmarks drawn: P onset and offset, QRS onset and offset, R peak, J point, and
T offset, each at its sample on the trace. Cells the record leaves absent are
not drawn. Every artist the package adds carries a `gid` starting with
`ecgrecords_viz:`, so you can find and restyle it.

### The signal must match the record

`signal` must be the exact array that produced the record: channel-major, one
row per record lead in `/acquisition/leads` order, `/acquisition/sample_count`
columns, and the same values and amplitude unit. The SHA-256 of
`np.ascontiguousarray(signal, dtype="<f8").tobytes()` is compared with
`/artifacts/raw_signal/sha256`. If they differ, or a lead or beat selector names
something the record does not have, the call raises
`ecgrecords_viz.VisualizationInputError`. It is a subclass of
`ecgfeat.errors.ECGInputError`, so code that already catches that error keeps
working.

`figure=None` creates a new `Figure` with constrained layout. If you pass a
`Figure` or `SubFigure`, the function only adds axes to it. The suptitle and
figure legend are added only to figures the function creates.

## Legacy helpers

The pre-record helpers that plot legacy `ECGFeatures` objects (`plot_beat`,
`plot_rep_beat`, `plot_beat_all_leads`, `plot_quality_summary`) live unchanged in
`ecgrecords_viz.legacy`. The old import path `ecgfeat.visualize` still resolves
to that module but emits a `DeprecationWarning`. It will be removed no earlier
than `ecg-records` 0.3.0. These helpers use `pyplot` and are imported only when
requested.
