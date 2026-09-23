"""Citable, bounded reads of the diagnosis-neutral raw measurement review."""
from __future__ import annotations

from ..evidence.store import EvidenceStore
from .registry import ToolResult, ToolSpec

MAX_RENDERED_OBSERVATIONS = 6
MAX_ALL_PROFILE_OBSERVATIONS = 2
_NOTE = (
    "Raw remeasurement of the same acquisition; not independent clinical validation. "
    "Exported timing is still a shared dependency. observations_available does not "
    "establish measurement correctness or a diagnosis. Missing review is not agreement."
)


def get_waveform_review(store: EvidenceStore, profile: str = "all") -> ToolResult:
    """Read approved artifact atoms; never quote arbitrary source diagnostic text."""
    if profile not in {"p_av", "qrs", "st", "all"}:
        return ToolResult.error("profile must be p_av, qrs, st, or all")
    if "waveform_review" not in store.document:
        return ToolResult.unavailable("Original waveform review artifact is unavailable. " + _NOTE)
    # Also sanitize full/adjudication stores. The tool's namespace is always the
    # physically isolated diagnosis contract, regardless of the caller's view.
    view = store.diagnostic_view()
    root = "/waveform_review"
    artifact = view.document["waveform_review"]
    requested = ("p_av", "qrs", "st") if profile == "all" else (profile,)
    lines: list[str] = []
    citations: list[str] = []

    def show(pointer: str) -> None:
        value = view.try_resolve(pointer)
        original = store.try_resolve(pointer)
        if value is not None and original is not None and value.value == original.value:
            label = pointer.removeprefix(root + "/").replace("profiles/", "").replace("/observations/", "/")
            lines.append(f"{label}: {value.render()}")
            citations.append(value.pointer)

    show(root + "/status")
    show(root + "/schema_version")
    for key in artifact["provenance"]:
        show(root + "/provenance/" + key)
    truncated = False
    row_limit = MAX_ALL_PROFILE_OBSERVATIONS if profile == "all" else MAX_RENDERED_OBSERVATIONS
    for name in requested:
        path = root + "/profiles/" + name
        data = artifact["profiles"][name]
        for key in ("status", "conflict_reasons", "missing_requirements", "input_count", "reviewed_count", "truncated"):
            show(path + "/" + key)
        rows = data["observations"]
        # Surface contradictions first; always preserve their original pointers.
        order = sorted(range(len(rows)), key=lambda i: rows[i].get("status") != "conflict")
        for index in order[:row_limit]:
            for key in rows[index]:
                show(f"{path}/observations/{index}/{key}")
        if len(rows) > row_limit:
            truncated = True
            lines.append(f"{name}: showing {row_limit} of {len(rows)} observations; remaining atoms are readable by exact pointer.")
    unavailable = all(artifact["profiles"][name]["status"] == "unavailable" for name in requested)
    if unavailable:
        lines.append("Requested raw waveform review is unavailable.")
        lines.append(_NOTE)
    return ToolResult(
        ok=not unavailable, text="\n".join(lines), citations=tuple(citations),
        truncated=truncated, note="measurement_unavailable" if unavailable else _NOTE,
    )


SPECS = (
    ToolSpec(
        name="get_waveform_review",
        description=(
            "Read bounded raw waveform measurement observations or explicit unavailability: "
            "P-candidate morphology/QRS-T overlap (p_av), signed QRS extrema (qrs), "
            "or raw PR-baselined ST (st). Values cite /waveform_review atoms. "
            "Same acquisition and shared exported timing; no clinical validation or diagnosis."
        ),
        parameters={"profile": {"type": "string", "enum": ["p_av", "qrs", "st", "all"], "default": "all"}},
        handler=get_waveform_review,
    ),
)


__all__ = ["get_waveform_review", "SPECS"]
