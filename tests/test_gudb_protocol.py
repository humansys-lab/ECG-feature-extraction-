import numpy as np
import pytest
from scripts.evaluate_ecgfeat_gudb import load_stream


def test_gudb_physical_units_channel_mapping_and_zero_based_annotations(tmp_path):
    data=np.zeros((100,6))
    data[:,:3]=[.001,-.002,.003]
    data[:,3:]=999
    np.savetxt(tmp_path/"ECG.tsv",data)
    np.savetxt(tmp_path/"annotation_cs.tsv",[0,50,99],fmt="%d")
    signal,truth=load_stream(tmp_path,"chest_strap")
    assert signal[1,0]==1
    assert np.array_equal(truth,[0,50,99])
    assert np.count_nonzero(signal[:,0])==1
    with pytest.raises(FileNotFoundError):
        load_stream(tmp_path,"cables")
    np.savetxt(tmp_path/"annotation_cables.tsv",[20,70],fmt="%d")
    signal,truth=load_stream(tmp_path,"cables")
    assert np.array_equal(signal[[1,7],0],[-2,3])
    assert np.array_equal(truth,[20,70])


@pytest.mark.parametrize("samples",[[50,50],[.5],[100],[-1]])
def test_gudb_invalid_truth_is_not_silently_clipped(tmp_path,samples):
    np.savetxt(tmp_path/"ECG.tsv",np.zeros((100,6)))
    np.savetxt(tmp_path/"annotation_cs.tsv",samples)
    with pytest.raises(ValueError,match="invalid R"):
        load_stream(tmp_path,"chest_strap")
