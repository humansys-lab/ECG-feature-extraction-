"""The dependency-light ECG Record contract.

Importing this package only imports standard-library record code.  Extraction
and interpretation are deliberately kept out of this namespace.
"""

from .availability import (
    Availability,
    Measured,
    NotApplicable,
    Unavailable,
    is_measured,
    measured,
    not_applicable,
    unavailable,
)
from .model import ECGRecord, RecordSidecar, build_record, record_from_document
from .provenance import (
    AlgorithmFingerprint,
    InputFingerprint,
    PolicyDecisionRecord,
    Provenance,
    build_provenance,
    clinical_fingerprint,
    fingerprint_input,
)
from .query import (
    AddressResolution,
    BeatView,
    LeadView,
    MeasurementQuery,
    MeasurementProvenance,
    MeasurementResult,
    MeasurementSelection,
    NotApplicableAbsence,
    NullAbsence,
    RecordAddress,
    UnmeasurableAbsence,
    ValidationStatus,
    iter_beats,
    iter_leads,
    dump_measurements,
    make_record_address,
    parse_address,
    parse_record_address,
    query_many,
    query_measurement,
    resolve_address,
    resolve_pointer,
    select_measurements,
)
from .serialize import (
    RecordProfile,
    SerializedRecord,
    dumps_record,
    encode_npz_sidecar,
    from_json_obj,
    load_record,
    loads_record,
    record_to_dict,
    serialize_record,
    to_json_obj,
    validate_record,
)
from .validation import (
    EvidenceReference,
    FieldValidation,
    ValidationTier,
    validation_for,
    validate_publishable_field,
)

__all__ = [
    "Availability", "Measured", "Unavailable", "NotApplicable", "measured",
    "unavailable", "not_applicable", "is_measured", "ECGRecord", "RecordSidecar",
    "build_record", "record_from_document", "InputFingerprint", "AlgorithmFingerprint",
    "PolicyDecisionRecord", "Provenance", "fingerprint_input", "clinical_fingerprint",
    "build_provenance", "RecordAddress", "AddressResolution", "MeasurementQuery",
    "MeasurementProvenance", "MeasurementResult", "MeasurementSelection", "ValidationStatus", "NullAbsence", "UnmeasurableAbsence",
    "NotApplicableAbsence", "LeadView", "BeatView", "parse_address",
    "parse_record_address", "make_record_address", "resolve_pointer", "resolve_address",
    "query_measurement", "query_many", "select_measurements", "iter_leads", "iter_beats",
    "RecordProfile", "SerializedRecord", "to_json_obj", "from_json_obj", "dumps_record",
    "loads_record", "load_record", "validate_record", "record_to_dict", "serialize_record", "encode_npz_sidecar", "dump_measurements", "EvidenceReference",
    "FieldValidation", "ValidationTier", "validation_for", "validate_publishable_field",
]
