"""ecgfeat: versioned ECG Record extraction and measurement (distribution ``ecg-records``).

Research and engineering software; not a medical device.  The supported
surface is ``__all__``: the ``ecg_*`` functions, the ECG Record read/query
API, configuration types and the error hierarchy.  Imports are lazy, so
record-only use loads no numerical engine, plotting or interpretation code.

Legacy names (``ECGFeatureExtractor``, ``to_dict``, ``interpret``, ``plot_*``,
WFDB helpers and the legacy result dataclasses) remain importable for the
compatibility window: legacy *functions* warn on use, and the exact legacy
contract is available without warnings from ``ecgfeat.compat``.
"""

from importlib import import_module

from ._moved import install_interpretation_aliases as _install_interpretation_aliases

_install_interpretation_aliases()

_EXPORTS = {
    "ECGFeatureExtractor": (".api", "ECGFeatureExtractor"),
    "ECGFeatures": (".compat.models_v0", "ECGFeatures"),
    "ECGInterpretation": (".compat.models_v0", "ECGInterpretation"),
    "GlobalFeatures": (".compat.models_v0", "GlobalFeatures"),
    "GroupFeatures": (".compat.models_v0", "GroupFeatures"),
    "LeadBeatFeatures": (".compat.models_v0", "LeadBeatFeatures"),
    "LeadQuality": (".compat.models_v0", "LeadQuality"),
    "PWaveBeatAssessment": (".compat.models_v0", "PWaveBeatAssessment"),
    "PWaveLeadBoundary": (".compat.models_v0", "PWaveLeadBoundary"),
    "PatientMeta": (".models", "PatientMeta"),
    "QRSDetectorResult": (".compat.models_v0", "QRSDetectorResult"),
    "QRSCandidateWindow": (".compat.models_v0", "QRSCandidateWindow"),
    "RepresentativeLeadFeatures": (".compat.models_v0", "RepresentativeLeadFeatures"),
    "STANDARD_12_LEADS": (".config", "STANDARD_12_LEADS"),
    "WaveBounds": (".compat.models_v0", "WaveBounds"),
    "PWaveConfig": ("._engine.atrial.p_wave", "PWaveConfig"),
    "ECGInputError": (".errors", "ECGInputError"),
    "RefinementConfig": (".refinement", "RefinementConfig"),
    "ECGConfig": (".config", "ECGConfig"),
    "ExtractionConfig": (".config", "ExtractionConfig"),
    "LimitedInput": (".config", "LimitedInput"),
    "StandardInput": (".config", "StandardInput"),
    "AnalysisST": (".config", "AnalysisST"),
    "CalibratedPRST": (".config", "CalibratedPRST"),
    "AdaptivePRTPST": (".config", "AdaptivePRTPST"),
    "RecordPWaveConfig": (".config", "PWaveConfig"),
    "STAmplitudeSource": (".config", "STAmplitudeSource"),
    "config_provenance": (".config", "config_provenance"),
    "ECGInput": (".pipeline", "ECGInput"),
    "ECGMeasurements": (".pipeline", "ECGMeasurements"),
    "ECGRecordExtractor": (".pipeline", "ECGRecordExtractor"),
    "ecg_dump": (".pipeline", "ecg_dump"),
    "ecg_emit": (".pipeline", "ecg_emit"),
    "ecg_measure": (".pipeline", "ecg_measure"),
    "ecg_prepare": (".pipeline", "ecg_prepare"),
    "ecg_record": (".pipeline", "ecg_record"),
    "extract_ecg_record": (".pipeline", "extract_ecg_record"),
    "extract_record": (".pipeline", "extract_record"),
    "AddressResolution": (".record", "AddressResolution"),
    "AlgorithmFingerprint": (".record", "AlgorithmFingerprint"),
    "BeatView": (".record", "BeatView"),
    "ECGRecord": (".record", "ECGRecord"),
    "FieldValidation": (".record", "FieldValidation"),
    "InputFingerprint": (".record", "InputFingerprint"),
    "LeadView": (".record", "LeadView"),
    "MeasurementQuery": (".record", "MeasurementQuery"),
    "MeasurementProvenance": (".record", "MeasurementProvenance"),
    "MeasurementResult": (".record", "MeasurementResult"),
    "MeasurementSelection": (".record", "MeasurementSelection"),
    "Measured": (".record", "Measured"),
    "NotApplicable": (".record", "NotApplicable"),
    "NotApplicableAbsence": (".record", "NotApplicableAbsence"),
    "NullAbsence": (".record", "NullAbsence"),
    "RecordAddress": (".record", "RecordAddress"),
    "RecordSidecar": (".record", "RecordSidecar"),
    "SerializedRecord": (".record", "SerializedRecord"),
    "Unavailable": (".record", "Unavailable"),
    "UnmeasurableAbsence": (".record", "UnmeasurableAbsence"),
    "ValidationStatus": (".record", "ValidationStatus"),
    "build_record": (".record", "build_record"),
    "build_provenance": (".record", "build_provenance"),
    "dumps_record": (".record", "dumps_record"),
    "dump_measurements": (".record", "dump_measurements"),
    "encode_npz_sidecar": (".record", "encode_npz_sidecar"),
    "fingerprint_input": (".record", "fingerprint_input"),
    "iter_beats": (".record", "iter_beats"),
    "iter_leads": (".record", "iter_leads"),
    "load_record": (".record", "load_record"),
    "loads_record": (".record", "loads_record"),
    "make_record_address": (".record", "make_record_address"),
    "measured": (".record", "measured"),
    "not_applicable": (".record", "not_applicable"),
    "parse_address": (".record", "parse_address"),
    "parse_record_address": (".record", "parse_record_address"),
    "query_many": (".record", "query_many"),
    "query_measurement": (".record", "query_measurement"),
    "record_to_dict": (".record", "record_to_dict"),
    "resolve_address": (".record", "resolve_address"),
    "resolve_pointer": (".record", "resolve_pointer"),
    "select_measurements": (".record", "select_measurements"),
    "serialize_record": (".record", "serialize_record"),
    "materialize_record": (".record", "materialize_record"),
    "SidecarError": (".record", "SidecarError"),
    "to_json_obj": (".record", "to_json_obj"),
    "unavailable": (".record", "unavailable"),
    "validate_record": (".record", "validate_record"),
    "AddressNotFoundError": (".errors", "AddressNotFoundError"),
    "AddressSyntaxError": (".errors", "AddressSyntaxError"),
    "AmplitudeUnitError": (".errors", "AmplitudeUnitError"),
    "ComputationInvariantError": (".errors", "ComputationInvariantError"),
    "ConfigurationError": (".errors", "ConfigurationError"),
    "LeadNameError": (".errors", "LeadNameError"),
    "MeasurementNotFoundError": (".errors", "MeasurementNotFoundError"),
    "MeasurementSelectorError": (".errors", "MeasurementSelectorError"),
    "RecordIdentityError": (".errors", "RecordIdentityError"),
    "RecordValidationError": (".errors", "RecordValidationError"),
    "SamplingRateError": (".errors", "SamplingRateError"),
    "SignalShapeError": (".errors", "SignalShapeError"),
    "ECGWarning": (".errors", "ECGWarning"),
    "ECGCompatibilityWarning": (".errors", "ECGCompatibilityWarning"),
    "ECGDeprecationWarning": (".errors", "ECGDeprecationWarning"),
}

#: Supported public surface (document 03, "Typing and public-surface policy").
PUBLIC_API = ('ecg_record', 'ecg_prepare', 'ecg_measure', 'ecg_emit', 'ecg_dump', 'load_record', 'validate_record', 'record_to_dict', 'parse_address', 'resolve_address', 'query_measurement', 'query_many', 'select_measurements', 'dump_measurements', 'iter_leads', 'iter_beats', 'ECGRecord', 'ECGInput', 'ECGMeasurements', 'RecordAddress', 'AddressResolution', 'MeasurementQuery', 'MeasurementResult', 'MeasurementSelection', 'NullAbsence', 'UnmeasurableAbsence', 'NotApplicableAbsence', 'MeasurementProvenance', 'ValidationStatus', 'ECGConfig', 'RefinementConfig', 'PWaveConfig', 'PatientMeta', 'STANDARD_12_LEADS', 'ECGInputError', 'SignalShapeError', 'LeadNameError', 'SamplingRateError', 'AmplitudeUnitError', 'ConfigurationError', 'RecordValidationError', 'RecordIdentityError', 'AddressSyntaxError', 'AddressNotFoundError', 'MeasurementNotFoundError', 'MeasurementSelectorError', 'ComputationInvariantError', 'ECGWarning', 'ECGCompatibilityWarning', 'ECGDeprecationWarning', 'ECGRecordExtractor', 'dumps_record', 'loads_record', 'serialize_record', 'materialize_record', 'SidecarError')
#: Deprecated legacy names kept for the compatibility window (removal no earlier than 0.3.0).
LEGACY_API = ('ECGFeatureExtractor', 'to_dict', 'interpret', 'load_wfdb_mat', 'parse_wfdb_header', 'plot_beat', 'plot_beat_all_leads', 'plot_rep_beat', 'plot_quality_summary', 'ECGFeatures', 'ECGInterpretation', 'GlobalFeatures', 'GroupFeatures', 'LeadBeatFeatures', 'LeadQuality', 'PWaveBeatAssessment', 'PWaveLeadBoundary', 'QRSDetectorResult', 'QRSCandidateWindow', 'RepresentativeLeadFeatures', 'WaveBounds')

__all__ = list(PUBLIC_API) + list(LEGACY_API)
__version__ = "0.1.0"


def __getattr__(name: str):
    if name in _LEGACY_FUNCTION_NAMES:
        from ._deprecated import LEGACY_FUNCTIONS

        value = LEGACY_FUNCTIONS[name]
    else:
        try:
            module_name, symbol = _EXPORTS[name]
        except KeyError:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
        value = getattr(import_module(module_name, __name__), symbol)
    globals()[name] = value
    return value


_LEGACY_FUNCTION_NAMES = frozenset(('to_dict', 'interpret', 'load_wfdb_mat', 'parse_wfdb_header', 'plot_beat', 'plot_beat_all_leads', 'plot_rep_beat', 'plot_quality_summary'))


def __dir__():
    return sorted(set(globals()) | set(__all__))
