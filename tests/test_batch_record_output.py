"""batch_extract_ecgfeat.py --output-format legacy|record|both (document 05, section E)."""

import json
from pathlib import Path

import numpy as np
import pytest

import batch_extract_ecgfeat as batch_script
from tests.fixtures.golden.reference_10s_12lead.make_signal import make_signal


def _job(tmp_path, name):
    return batch_script.BatchJob(disease_label="ref", source_name="real", input_path=tmp_path / "real.pt",
                                 output_dir=tmp_path / name, metadata_path=None)


def _run(tmp_path, name, output_format, extractor, record_writer=None):
    from ecgfeat.export import prepare_json_export, to_dict

    return batch_script.process_job(
        job=_job(tmp_path, name), extractor=extractor, to_dict_fn=to_dict,
        report_writer=lambda record_id, header, ecg, result, path: path.write_text(record_id),
        sampling_rate=500, load_pt_fn=lambda path: np.stack([make_signal()]), save_pt_fn=lambda path, payload: None,
        prepare_json_export_fn=prepare_json_export, export_profile="summary", save_pt_output=False,
        output_format=output_format, record_writer=record_writer, record_profile="summary")


@pytest.fixture(scope="module")
def extractor():
    from ecgfeat.api import ECGFeatureExtractor

    return ECGFeatureExtractor(mains_freq=50)


def test_both_mode_keeps_the_legacy_json_identical_and_adds_a_valid_record(tmp_path, extractor):
    from ecgfeat.record import loads_record

    legacy = _run(tmp_path, "legacy", "legacy", extractor)
    both = _run(tmp_path, "both", "both", extractor, batch_script.build_record_writer("summary"))
    assert legacy["processed"] == both["processed"] == 1 and both["failed"] == 0
    from benchmarks.golden.runner import normalize_legacy

    # Only the interpretation's wall-clock generated_at timestamps may differ (Phase 0 finding).
    legacy_json, both_json = ((tmp_path / name / "000_features.json").read_text() for name in ("legacy", "both"))
    assert json.dumps(normalize_legacy(json.loads(legacy_json))) == json.dumps(normalize_legacy(json.loads(both_json)))
    assert not (tmp_path / "legacy/000_record.json").exists()
    record = loads_record((tmp_path / "both/000_record.json").read_bytes())
    assert record.profile == "summary" and len(record.axes["beats"]) == 10
    manifest = json.loads((tmp_path / "both/manifest.json").read_text())
    assert manifest["output_format"] == "both" and manifest["record_schema_version"] == record.schema_version
    assert manifest["record_config_sha256"] == record.provenance["config_hash"]


def test_record_mode_writes_no_legacy_feature_json(tmp_path, extractor):
    summary = _run(tmp_path, "record", "record", extractor, batch_script.build_record_writer("all"))
    assert summary["processed"] == 1
    assert not (tmp_path / "record/000_features.json").exists()
    assert json.loads((tmp_path / "record/000_record.json").read_bytes())["profile"] == "all"
    assert (tmp_path / "record/000_report.txt").exists()


def test_record_output_requires_a_writer_and_a_known_format(tmp_path, extractor):
    with pytest.raises(ValueError):
        _run(tmp_path, "x", "record", extractor)
    with pytest.raises(ValueError):
        _run(tmp_path, "y", "records", extractor, batch_script.build_record_writer("summary"))


def test_cli_exposes_the_modes():
    args = batch_script.build_arg_parser().parse_args(["--output-format", "both", "--record-profile", "all"])
    assert (args.output_format, args.record_profile) == ("both", "all")
    assert batch_script.build_arg_parser().parse_args([]).output_format == "legacy"
