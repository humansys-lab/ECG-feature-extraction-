"""Guard independent-data evaluation against unit, scope and label mistakes."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import wfdb

from scripts.evaluate_ecgfeat_cross_dataset import (
    adapt_record, annotations, in_cores, pipeline_events, select_windows, source_overlaps,
)
from scripts.evaluate_ecgfeat_external import WaveEvent, score


@pytest.mark.parametrize("fs", [128, 257, 360])
def test_full_windows_own_each_sample_once(fs):
    length = round(1805.56 * fs)
    bounds = select_windows(length, fs, "pipeline_full")
    assert bounds[0][0] == 0 and bounds[-1][1] == length
    assert all(left[1] == right[0] for left, right in zip(bounds, bounds[1:]))
    assert all(0 <= c <= a < b <= d <= length for a, b, c, d in bounds)
    assert sum(b-a for a, b, _, _ in bounds) == length


def test_sample_selection_depends_only_on_length():
    bounds = select_windows(1800 * 257, 257, "pipeline_sampled")
    assert [(a/257, b/257) for a,b,_,_ in bounds] == [(0,30),(885,915),(1770,1800)]
    assert sum(b-a for a,b,_,_ in bounds) == 90 * 257
    assert not in_cores(60*257, bounds)
    assert not in_cores(30*257, bounds)
    assert in_cores(885*257, bounds)


def test_incart_augmented_names_are_normalized_without_gain_rescaling():
    names = ['I','II','III','AVR','AVL','AVF','V1','V2','V3','V4','V5','V6']
    values = np.arange(120, dtype=float).reshape(10,12)/306
    record = SimpleNamespace(n_sig=12, units=['mV']*12, sig_name=names, p_signal=values)
    signal, canonical = adapt_record(record, 'incartdb')
    assert canonical[3:6] == ['aVR','aVL','aVF']
    np.testing.assert_array_equal(signal, values.T)


def test_limited_channels_keep_measured_names():
    record = SimpleNamespace(n_sig=2, units=['mV']*2, sig_name=['MLII','V1'], p_signal=np.ones((360,2)))
    signal, names = adapt_record(record, 'pwave')
    assert signal.shape == (2,360) and names == ['MLII','V1']
    record.units = ['uV','mV']
    with pytest.raises(ValueError, match='unexpected input'):
        adapt_record(record, 'pwave')


def test_beat_truth_excludes_rhythm_flutter_and_blocked_p(tmp_path):
    wfdb.wrann('sample', 'atr', np.array([0,20,40,60,80,100,120]),
               symbol=['+','N','V','!','x','A','N'], write_dir=str(tmp_path))
    truth, symbols, excluded = annotations(tmp_path/'sample', 'atr', 110, 'QRS')
    assert [e.peak for e in truth] == [20,40,100]
    assert symbols == {20:'N',40:'V',100:'A'}
    assert len(excluded) == 4


def test_p_truth_uses_p_symbol_and_keeps_record_edge(tmp_path):
    wfdb.wrann('sample', 'pwave', np.array([2,200,250]), symbol=['p','p','+'], write_dir=str(tmp_path))
    truth, _, excluded = annotations(tmp_path/'sample', 'pwave', 300, 'P')
    assert [e.peak for e in truth] == [2,200]
    assert len(excluded) == 1


def test_identical_p_reference_duplicates_count_once_with_audit(tmp_path):
    wfdb.wrann('sample', 'pwave', np.array([2,200,200]), symbol=['p','p','p'], write_dir=str(tmp_path))
    truth, _, excluded = annotations(tmp_path/'sample', 'pwave', 300, 'P')
    assert [e.peak for e in truth] == [2,200]
    assert excluded == [dict(sample=200,symbol='p',action='duplicate_identical_reference')]


def test_conflicting_beat_reference_duplicates_raise(tmp_path):
    wfdb.wrann('sample', 'atr', np.array([20,20]), symbol=['N','V'], write_dir=str(tmp_path))
    with pytest.raises(ValueError, match='conflicting reference'):
        annotations(tmp_path/'sample', 'atr', 300, 'QRS')


def test_failure_is_kept_as_false_negatives():
    row, matches = score('failed', 'default', 'QRS', 'QRS',
                         [WaveEvent(None,20,None),WaveEvent(None,100,None)], [], 128,
                         'partial_failure', tolerance=75)
    assert (row['gt_count'],row['tp'],row['fp'],row['fn']) == (2,0,0,2)
    assert not matches


def test_p_outputs_separate_candidate_and_accepted():
    features = SimpleNamespace(
        beats=[SimpleNamespace(r_index=150)],
        p_wave_assessments=[SimpleNamespace(beat_id=1,accepted=True)],
        beat_features=[SimpleNamespace(beat_id=1,lead='II',p=SimpleNamespace(peak=50)),
                       SimpleNamespace(beat_id=2,lead='II',p=SimpleNamespace(peak=250)),
                       SimpleNamespace(beat_id=1,lead='V1',p=SimpleNamespace(peak=55))],
        metadata={'rhythm_analysis':{'atrial_events':[{'sample':48},{'sample':248}]}})
    assert pipeline_events(features,'II') == {'QRS':[150], 'P_native':[50,250],
                                             'P_accepted':[50], 'P_atrial':[48,248]}


def test_overlap_excludes_whole_source_record(tmp_path):
    but = tmp_path/'data/external_ecg/but-pdb'; but.mkdir(parents=True)
    (but/'README.txt').write_text('MIT-BIH Arrhythmia Database 231 [30,70]\n'
                                  'MIT-BIH Supraventricular Arrhythmia Database 801 [26,41]')
    qt = tmp_path/'data/qtdb'; qt.mkdir(parents=True); (qt/'sel100.hea').touch()
    assert source_overlaps('pwave','231',tmp_path) == ['BUT source record']
    assert source_overlaps('pwave','100',tmp_path) == ['QTDB source record']
    assert source_overlaps('pwave','119',tmp_path) == ['NSTDB source record']
    assert source_overlaps('svdb','801',tmp_path) == ['BUT source record']
    assert source_overlaps('pwave','122',tmp_path) == []
    assert source_overlaps('incartdb','I01',tmp_path) == []


def test_cluster_bootstrap_groups_records_by_patient():
    from scripts.summarize_ecgfeat_cross_dataset import paired_bootstrap
    rows=[]
    for record,patient in [('I01','1'),('I02','1'),('I03','2')]:
        for method in ('default','enhanced'):
            rows.append(dict(record=record,patient=patient,dataset='incartdb',scope='detector_full',wave='QRS',
                             lead='QRS',protocol='peak_75ms',method=method,tp=10,fp=1,fn=1))
    result=paired_bootstrap(rows)
    assert len(result)==1 and result[0]['clusters']==2
    assert result[0]['delta_f1_pp']==0 and result[0]['ci95_pp']==[0,0]


def test_cluster_bootstrap_rejects_unpaired_configs():
    from scripts.summarize_ecgfeat_cross_dataset import paired_bootstrap
    row=dict(patient='1',dataset='svdb',scope='detector_full',wave='QRS',lead='QRS',protocol='peak_75ms',
             method='default',tp=10,fp=1,fn=1)
    with pytest.raises(ValueError,match='unpaired cohort'):
        paired_bootstrap([row])
