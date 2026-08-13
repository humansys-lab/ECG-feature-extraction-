from __future__ import annotations

import numpy as np
import pytest
from scipy.io import savemat

from feature_extraction.ecgfeat.io import load_wfdb_mat, parse_wfdb_header


def test_parse_wfdb_header_reads_record_and_comment_metadata(tmp_path) -> None:
    header = tmp_path / "sample.hea"
    header.write_text(
        "\n".join(
            [
                "sample 12 500/1000 5000",
                "sample.mat 16+24 1000/mV 16 0 0 0 I",
                "#Age: 47",
                "#Sex: Female",
                "#Dx: 426783006, 164934002",
                "#Rx: none",
            ]
        ),
        encoding="utf-8",
    )

    metadata = parse_wfdb_header(header)

    assert metadata == {
        "record": "sample",
        "n_leads": 12,
        "fs": 500,
        "n_samples": 5000,
        "age": 47,
        "sex": "Female",
        "dx": ["426783006", "164934002"],
        "rx": "none",
    }


def test_parse_wfdb_header_marks_unknown_age_as_missing(tmp_path) -> None:
    header = tmp_path / "sample.hea"
    header.write_text("sample 12 500 5000\n#Age: NaN\n", encoding="utf-8")

    assert parse_wfdb_header(header)["age"] is None


def test_load_wfdb_mat_converts_adc_values_to_millivolts(tmp_path) -> None:
    mat_path = tmp_path / "sample.mat"
    adc = np.array([[1000, -500], [250, 0]], dtype=np.int16)
    savemat(mat_path, {"val": adc})

    ecg = load_wfdb_mat(mat_path, gain=1000.0)

    assert ecg.dtype == np.float64
    np.testing.assert_allclose(ecg, [[1.0, -0.5], [0.25, 0.0]])


def test_load_wfdb_mat_rejects_missing_signal_array(tmp_path) -> None:
    mat_path = tmp_path / "sample.mat"
    savemat(mat_path, {"other": np.zeros((2, 2))})

    with pytest.raises(KeyError, match="'val'"):
        load_wfdb_mat(mat_path)
