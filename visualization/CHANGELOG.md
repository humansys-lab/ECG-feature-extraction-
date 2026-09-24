# Changelog — ecg-records-viz

## [Unreleased]

## [0.1.0] - 2026-09-24

### Added
- Record-based plotting: `plot_record`, `plot_beat`, `plot_beat_all_leads`,
  `plot_representative_beat`, `plot_quality_summary`, all taking
  `(signal, record)` and returning `(Figure, axes)` through the object-oriented
  Matplotlib API; the signal is checked against the record's raw-signal SHA-256.
- `ecgrecords_viz.legacy`: the previous `ECGFeatures`-based plots (deprecated).
