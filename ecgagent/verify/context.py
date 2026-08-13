"""Binding a written quantity to the measurement it claims to be.

A payload holds tens of thousands of numeric leaves, so matching a written
number against all of them on magnitude alone proves nothing: a plausible
`168 ms` will collide with something no matter where it came from.  To make
traceability mean anything, a match must also agree with *what the sentence
says the number is* - the measurement term next to it, and the lead it names.

Both signals are read off the claim text, so this is a heuristic; it is used
to make a match stricter, never to manufacture one.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Measurement term -> substrings a matching field name must contain.
# Ordered longest-keyword-first at match time so `QTc` beats `QT`.
_TERMS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    # key: (keywords found in prose, field-name substrings)
    "qtc": (("qtc", "corrected qt"), ("qtc_",)),
    "qt": (("qt interval", "qt duration", "qt "), ("qt_ms", "qt_consensus", "qt_robust", "qt_latest")),
    "qrs": (("qrs",), ("qrs_ms", "qrs_wide", "qrs_consensus", "qrs_duration")),
    "pr": (("pr interval", "p-r interval", "pr segment", "pr "), ("pr_ms", "pr_consensus", "pr_segment")),
    "rr": (("rr interval", "rr ", "r-r"), ("rr_", "_rr")),
    "rate": (("heart rate", "ventricular rate", "atrial rate", "bpm"), ("rate_bpm", "heart_rate")),
    "axis": (("axis",), ("_axis_deg",)),
    "st": (("st segment", "st elevation", "st depression", "st-segment", "j point", "j-point", "st "),
           ("st_", "_stj_", "_stm_", "_ste_")),
    "qwave": (("q wave", "q-wave", "q duration", "pathological q"), ("q_duration", "q_amp", "q_area", "q_r_ratio")),
    "ramp": (("r wave", "r-wave", "r amplitude"), ("r_amp", "r_duration", "r_prime")),
    "samp": (("s wave", "s-wave", "s amplitude"), ("s_amp", "s_duration", "s_prime")),
    "pwave": (("p wave", "p-wave", "p duration", "p terminal", "terminal p"),
              ("p_dur", "p_amp", "p_terminal", "p_initial", "p_area", "ptf_")),
    "twave": (("t wave", "t-wave", "t amplitude", "t duration"), ("t_amp", "t_dur", "t_area", "tpe", "tpte")),
    "jt": (("jt interval", "jt "), ("jt_",)),
    "duration": (("duration sec", "record length", "recording"), ("duration_sec",)),
}

_LEAD_RE = re.compile(r"\b(I{1,3}|aVR|aVL|aVF|V[1-9]R?)\b")

# How far before a quantity a term may sit and still be considered its subject.
_BACKWARD_WINDOW = 90
_FORWARD_WINDOW = 40


@dataclass(frozen=True)
class ClaimContext:
    """Terms and leads a claim mentions, with their positions."""

    terms: tuple[tuple[int, int, str], ...]
    leads: frozenset[str]

    def term_for(self, position: int) -> str | None:
        """The measurement term a quantity at `position` most likely refers to."""
        best: tuple[int, str] | None = None
        for start, end, key in self.terms:
            if end <= position:
                distance = position - end
                if distance <= _BACKWARD_WINDOW and (best is None or distance < best[0]):
                    best = (distance, key)
            else:
                distance = start - position
                # A term after the number is weaker evidence; only accept it
                # when nothing precedes ("... 126 ms QRS duration").
                if distance <= _FORWARD_WINDOW and best is None:
                    best = (distance + _BACKWARD_WINDOW, key)
        return None if best is None else best[1]


def parse_claim(claim: str) -> ClaimContext:
    lowered = claim.lower()
    hits: list[tuple[int, int, str]] = []
    for key, (keywords, _) in _TERMS.items():
        for keyword in keywords:
            start = 0
            while True:
                found = lowered.find(keyword, start)
                if found < 0:
                    break
                hits.append((found, found + len(keyword), key))
                start = found + 1
    # Prefer the most specific term when two overlap (qtc over qt).
    hits.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    deduped: list[tuple[int, int, str]] = []
    for start, end, key in hits:
        if deduped and start < deduped[-1][1] and (end - start) <= (deduped[-1][1] - deduped[-1][0]):
            continue
        deduped.append((start, end, key))

    leads = frozenset(match.group(1) for match in _LEAD_RE.finditer(claim))
    return ClaimContext(terms=tuple(deduped), leads=leads)


def field_matches_term(pointer: str, term: str | None) -> bool:
    """Whether a pointer's field name is consistent with the claimed measurement."""
    if term is None:
        return True
    substrings = _TERMS.get(term, ((), ()))[1]
    if not substrings:
        return True
    field_name = pointer.rsplit("/", 1)[-1].lower()
    return any(fragment in field_name for fragment in substrings)


def lead_matches(leaf_lead: str | None, leads: frozenset[str]) -> bool:
    """Whether a lead-scoped value sits in one of the leads the claim names.

    A value with no lead (a global measurement) is never excluded by a lead
    mention: "QRS is 126 ms in V1 and V2" still refers to the global QRS.
    """
    if not leads or leaf_lead is None:
        return True
    return leaf_lead in leads
