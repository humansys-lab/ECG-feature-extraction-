from types import SimpleNamespace

import numpy as np
import pytest

from feature_extraction.ecgfeat.adaptive_qrs import detect_adaptive_qrs
from feature_extraction.ecgfeat.adaptive_qrs import guard_weak_additions
from feature_extraction.ecgfeat.adaptive_qrs import _nearest_distances
from feature_extraction.ecgfeat.atrial_validation import p_waveform_evidence, validate_atrial_events
from feature_extraction.ecgfeat.models import WaveBounds
from feature_extraction.ecgfeat.preprocess import lowpass_filter
from feature_extraction.ecgfeat.qrs import detect_qrs_multilead_with_meta
from tests.test_limited_leads import signal


@pytest.mark.parametrize("fs", [128, 250, 500, 1000])
def test_adaptive_qrs_handles_polarity_and_flat_placeholders(fs):
    x = signal(fs)
    full = np.zeros((12, x.shape[1]));full[1], full[7] = x[0], -x[1]
    result = detect_adaptive_qrs(full, fs, [0, 1, 7])
    truth = np.arange(.6, 9.7, .8)*fs
    assert len(result.r_locs) == len(truth)
    assert np.max(abs(np.array(result.r_locs)-truth)) < .035*fs
    assert all(np.diff(result.r_locs) >= .2*fs)
    assert detect_adaptive_qrs(full*0, fs, [1,7]).r_locs == []


def test_high_agreement_qrs_retains_established_detector():
    x = np.zeros((12, 5000));x[1],x[7]=signal()
    qualities = {"II": SimpleNamespace(b_sqi=1.), "V2": SimpleNamespace(b_sqi=1.)}
    a = detect_qrs_multilead_with_meta(x, 500, leads=[1,7])
    b = detect_qrs_multilead_with_meta(x, 500, leads=[1,7], quality=qualities, adaptive_consensus=True)
    assert a.r_locs == b.r_locs


def test_adaptive_qrs_validates_explicit_fiducial_before_branching():
    with pytest.raises(ValueError, match="fiducial_lead"):
        detect_qrs_multilead_with_meta(np.zeros((12, 5000)),500,leads=[1,7],
                                     fiducial_lead=12,adaptive_consensus=True)


@pytest.mark.parametrize("polarity", [-1, 1])
def test_local_p_evidence_survives_gain_and_polarity_but_rejects_flat(polarity):
    t = np.arange(500)/500
    x = polarity*.12*np.exp(-.5*((t-.5)/.025)**2)
    result = p_waveform_evidence(x, lowpass_filter(x,500,15),250,500)
    assert result is not None
    assert abs(result["peak"]-250)<=2
    assert p_waveform_evidence(x*0,x*0,250,500) is None


def test_atrial_validation_keeps_repeatable_p_and_rejects_qrs_overlap():
    x=np.zeros((12,5000));x[1],x[7]=signal()
    r=np.arange(.6,9.7,.8)*500
    features=[SimpleNamespace(beat_id=i,lead="II",t=None,qrs=WaveBounds(int(p-20),int(p),int(p+30))) for i,p in enumerate(r)]
    events=[dict(sample=int(p-90),time_ms=(p-90)*2,confidence=.8) for p in r]
    events += [dict(sample=int(p),time_ms=p*2,confidence=1.) for p in r]
    quality={"II":SimpleNamespace(reliable_for_p=True),"V2":SimpleNamespace(reliable_for_p=True)}
    audit={};result=validate_atrial_events(events,x,500,features,quality,audit=audit)
    assert len(result)==len(r)
    assert audit["rejected"]["ventricular_overlap"]==len(r)
    assert all("validation" in e for e in result)


def test_isolated_nonconducted_p_is_not_deleted_for_lacking_a_template():
    t=np.arange(3000)/500
    x=np.zeros((12,len(t)))
    x[1]=.08*np.exp(-.5*((t-3)/.035)**2)
    event=dict(sample=1500,confidence=.8,association_type="blocked")
    result=validate_atrial_events([event],x,500,[],{"II":SimpleNamespace(reliable_for_p=True)})
    assert len(result)==1 and result[0]["association_type"]=="blocked"


@pytest.mark.parametrize("gain", [.1, 1., 5.])
@pytest.mark.parametrize("variable_peer", [False,True])
def test_qrs_guard_removes_faint_secondary_waves_without_losing_strong_ectopy(gain,variable_peer):
    fs=500;t=np.arange(7000)/fs
    anchors=np.arange(500,6500,800)
    extra=anchors[:-1]+300
    strong=int(extra[-1]+200)
    x=np.zeros((12,len(t)))
    for channel,amplitude in [(1,1.2),(7,.8)]:
        for i,p in enumerate(anchors):
            scale = (1.7 if i%2 else .6) if channel==7 and variable_peer else 1.
            x[channel]+=scale*amplitude*np.exp(-.5*((t-p/fs)/.012)**2)
        for p in extra:
            x[channel]+=.04*np.exp(-.5*((t-p/fs)/.035)**2)
        x[channel]+=.8*amplitude*np.exp(-.5*((t-strong/fs)/.025)**2)
    candidates=[SimpleNamespace(refined_r_index=int(p)) for p in sorted([*anchors,*extra,strong])]
    result=guard_weak_additions(candidates,SimpleNamespace(r_locs=anchors.tolist()),gain*x,fs,[1,7])
    assert [r.refined_r_index for r in result]==sorted([*anchors,strong])


def test_nearest_peak_search_matches_brute_force_with_edges_and_ties():
    rng=np.random.default_rng(43)
    for n,m in [(1,1),(100,31),(5,150)]:
        reference=np.sort(rng.integers(0,1000,n))
        values=rng.integers(-10,1010,m)
        assert np.array_equal(_nearest_distances(values,reference),
                              np.min(abs(values[:,None]-reference[None,:]),axis=1))
