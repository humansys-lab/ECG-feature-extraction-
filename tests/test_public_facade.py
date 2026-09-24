"""Phase 4 public surface: the documented API, silent imports, use-time deprecations."""

import subprocess
import sys
import warnings

import numpy as np
import pytest

import ecgfeat
from ecgfeat.errors import ECGDeprecationWarning

DOC03_ALL = [
    "ecg_record", "ecg_prepare", "ecg_measure", "ecg_emit", "ecg_dump", "load_record", "validate_record",
    "record_to_dict", "parse_address", "resolve_address", "query_measurement", "query_many", "select_measurements",
    "dump_measurements", "iter_leads", "iter_beats", "ECGRecord", "ECGInput", "ECGMeasurements", "RecordAddress",
    "AddressResolution", "MeasurementQuery", "MeasurementResult", "MeasurementSelection", "NullAbsence",
    "UnmeasurableAbsence", "NotApplicableAbsence", "MeasurementProvenance", "ValidationStatus", "ECGConfig",
    "RefinementConfig", "PWaveConfig", "PatientMeta", "STANDARD_12_LEADS", "ECGInputError", "SignalShapeError",
    "LeadNameError", "SamplingRateError", "AmplitudeUnitError", "ConfigurationError", "RecordValidationError",
    "RecordIdentityError", "AddressSyntaxError", "AddressNotFoundError", "MeasurementNotFoundError",
    "MeasurementSelectorError", "ComputationInvariantError", "ECGWarning", "ECGCompatibilityWarning",
    "ECGDeprecationWarning",
]
LEGACY_FUNCTIONS = ["to_dict", "interpret", "load_wfdb_mat", "parse_wfdb_header", "plot_beat",
                    "plot_beat_all_leads", "plot_rep_beat", "plot_quality_summary"]


def test_all_is_the_documented_surface_plus_grouped_legacy_names():
    assert set(DOC03_ALL) <= set(ecgfeat.PUBLIC_API)
    assert set(ecgfeat.PUBLIC_API) - set(DOC03_ALL) == {
        "ECGRecordExtractor", "dumps_record", "loads_record", "serialize_record", "materialize_record", "SidecarError"}
    assert set(ecgfeat.__all__) == set(ecgfeat.PUBLIC_API) | set(ecgfeat.LEGACY_API)
    for name in ecgfeat.__all__:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert getattr(ecgfeat, name) is not None, name
    assert ecgfeat.STANDARD_12_LEADS == ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")


def test_importing_the_facade_loads_no_numerical_plotting_or_interpretation_code():
    code = ("import sys, ecgfeat; ecgfeat.ECGRecord; ecgfeat.load_record; "
            "heavy = [m for m in ('numpy', 'scipy', 'matplotlib', 'ecginterpret', 'ecgfeat.viz.plots') if m in sys.modules]; "
            "assert not heavy, heavy")
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize("name", LEGACY_FUNCTIONS)
def test_legacy_top_level_functions_warn_on_use_not_on_import(name):
    # Importing the old *module* path ecgfeat.interpret binds that module over the
    # top-level function of the same name (a pre-existing ambiguity); reset it.
    vars(ecgfeat).pop(name, None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        function = getattr(ecgfeat, name)
    assert not [w for w in caught if issubclass(w.category, ECGDeprecationWarning)]
    with pytest.warns(ECGDeprecationWarning, match=r"removed no earlier than ecg-records 0\.3\.0; use "):
        try:
            function()
        except TypeError:
            pass  # called without arguments; the warning precedes the call
        except ImportError as exc:  # optional extra not installed (viz: Matplotlib; interpret: ecginterpret)
            assert "pip install" in str(exc)


def test_legacy_extractor_spelling_warns_and_compat_is_silent():
    from ecgfeat.compat.api_v0 import ECGFeatureExtractor as Compat

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Compat()
    with pytest.warns(ECGDeprecationWarning, match="ecgfeat.ecg_record"):
        legacy = ecgfeat.ECGFeatureExtractor()
    assert isinstance(legacy, Compat)


def test_legacy_export_spelling_warns_on_use_and_matches_compat():
    from ecgfeat import export as legacy_export
    from ecgfeat.compat import export_v0

    from ecgfeat.compat.api_v0 import ECGFeatureExtractor

    signal = np.load("tests/fixtures/golden/reference_10s_12lead/signal.npz")["signal"]
    features = ECGFeatureExtractor().extract(signal, 500)
    with pytest.warns(ECGDeprecationWarning, match="record_to_dict"):
        payload = legacy_export.to_dict(features, profile="summary")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert export_v0.to_dict(features, profile="summary").keys() == payload.keys()
    assert legacy_export.clinical_fingerprint is export_v0.clinical_fingerprint  # other names forward


def test_legacy_result_types_are_the_compat_types():
    from ecgfeat.compat import models_v0

    assert ecgfeat.ECGFeatures is models_v0.ECGFeatures
    assert ecgfeat.ECGInterpretation is models_v0.ECGInterpretation
