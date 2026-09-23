"""Real regressions: optional local data, never downloaded during tests."""
from pathlib import Path
import numpy as np
import pytest
from feature_extraction.ecgfeat import ECGFeatureExtractor, RefinementConfig


@pytest.mark.parametrize("record",["324","343"])
def test_offset_candidates_do_not_change_final_repaired_t_onset(record):
    root=Path(__file__).resolve().parents[1]/"data/external_ecg/isp/isp_delineation_dataset/train_data"
    paths=list(root.rglob(record+".hea"))
    if not paths:
        pytest.skip("ISP regression data not installed")
    import wfdb
    raw=wfdb.rdrecord(str(paths[0].with_suffix("")))
    leads=["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
    lookup={name.lower():i for i,name in enumerate(raw.sig_name)}
    signal=raw.p_signal[:,[lookup[name.lower()] for name in leads]].T/1000
    baseline=ECGFeatureExtractor(fs_internal=500,mains_freq=50).extract(signal,raw.fs)
    before={(f.beat_id,f.lead):(f.t.onset,f.t.peak) for f in baseline.beat_features}
    before_offsets={(f.beat_id,f.lead):f.t.offset for f in baseline.beat_features}
    for sequence in (False,True):
        result=ECGFeatureExtractor(fs_internal=500,mains_freq=50,refinement=RefinementConfig(
            t_projection_offset_only=True,t_sequence_offset_only=sequence)).extract(signal,raw.fs)
        assert {(f.beat_id,f.lead):(f.t.onset,f.t.peak) for f in result.beat_features}==before
        for f in result.beat_features:
            # Legacy delineation can already have onset==peak. This regression
            # checks changed candidates rather than claiming to repair it.
            if (f.t.offset != before_offsets[(f.beat_id,f.lead)]
                    and f.t.offset is not None and f.t.onset is not None and f.t.peak is not None):
                assert f.t.onset < f.t.peak < f.t.offset
                assert np.isclose(f.t_dur_ms,(f.t.offset-f.t.onset)*2)
                if f.qrs.onset is not None:
                    assert np.isclose(f.qt_ms,(f.t.offset-f.qrs.onset)*2)
