"""Finite migration namespace for legacy object and export names.

Submodules load lazily so that importing ``ecgfeat.compat.models_v0`` (as the
interpretation distribution does) never pulls in the legacy exporter.
"""

from importlib import import_module

_EXPORTS = {
    "ECGFeatureExtractor": ".api_v0",
    "ECGFeatures": ".models_v0",
    "features_from_record": ".models_v0",
    "to_dict": ".export_v0",
    "prepare_json_export": ".export_v0",
    "build_structured_payload": ".export_v0",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    try:
        module = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value
