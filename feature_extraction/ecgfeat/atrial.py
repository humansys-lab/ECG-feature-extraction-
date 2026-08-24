from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import correlate, decimate, find_peaks, welch

from .preprocess import lowpass_filter


PREFERRED_ATRIAL_LEADS = ("I", "II", "III", "V1", "V2")
# Reddy/Farrell raw-rhythm P-wave detector: one limb + one precordial lead.
ATRIAL_LIMB_LEADS = ("I", "II")
ATRIAL_PRECORDIAL_LEADS = ("V1", "V2", "V3", "V4", "V5", "V6")
ATRIAL_RESIDUAL_LP_HZ = 23.0              # Farrell 2003 zero-phase low-pass
COMPOSITE_P_DETECTION_METHOD = "composite_qrst_residual_derivative"
# Farrell/Xue/Young 2003 spectral atrial-flutter detection parameters.
FLUTTER_SPECTRAL_TARGET_FS = 125.0        # decimation target (samples/s)
FLUTTER_SPECTRAL_NFFT = 1024              # FFT length for PSD
# Flutter PSD-peak search band. The lower edge used to be 220 bpm, which is
# above drug-slowed flutter: a labelled AFLT record (PTB-XL 2430) has its
# dominant peak at 212 bpm and could not produce a candidate at all. The
# literature band for the dominant F-wave peak is 3-10 Hz (180-600 bpm);
# 180 bpm is taken as the lower edge, the upper stays at 400 bpm because
# beyond that the band starts admitting QRS-rate harmonics.
FLUTTER_SPECTRAL_MIN_BPM = 180.0          # flutter PSD-peak search band (lower)
FLUTTER_SPECTRAL_MAX_BPM = 400.0          # flutter PSD-peak search band (upper)
# Narrowness of the dominant peak: power within +/-0.25 Hz of it over the
# power in a 2.5 Hz envelope around it. Flutter is one macro-reentrant
# wavefront and so gives a single narrow peak; AF's multiple wavefronts give a
# broad one (Aliot/Ndrepepa, JACC 2005). Measured here over 34 AFLT vs 120
# AFIB+sinus records, the mean of leads II and V1 separates them at AUC 0.832
# (AFLT median 0.527 vs 0.363), by far the best single discriminator
# available -- `spectral_peak_ratio` alone is not just weak but inverted,
# scoring higher on sinus (median 4.48) than on flutter (2.95).
FLUTTER_SPECTRAL_NARROW_PEAK_HALF_HZ = 0.25
FLUTTER_SPECTRAL_NARROW_ENVELOPE_HZ = 2.5
# 0.52 on the measured curve gives 50% flutter detection at 5% false
# positives (AFIB 8%, sinus 2%). JACC's 0.44 reaches 79% but at 30% false
# positives, which is far too loose for a flag that outranks AF.
FLUTTER_SPECTRAL_NARROWNESS_MIN = 0.52
# Regular flutter needs a second route, on the *un*subtracted signal.
# `flutter_analysis_signal` cancels the QRST by averaging beats, which works
# only while the F waves drift in phase relative to the QRS. Under fixed-ratio
# conduction they do not: the F wave sits at the same offset in every beat,
# the median template absorbs it, and the subtraction erases the very signal
# being sought. Measured over 34 labelled flutter records split by RR
# regularity: on the 23 regular ones (rr_cv < 0.10) narrowness falls from 0.960
# raw to 0.471 subtracted -- 96% of them get worse, and the count reaching the
# flutter gate collapses from 22/23 to 5/23. On the 11 irregular ones the same
# subtraction is neutral (0.743 -> 0.758). Regular flutter is the common,
# textbook presentation, so it cannot be the case that is dropped.
#
# The raw signal is not free of a confound: any periodic rhythm puts harmonics
# in the 3-10 Hz band, and 20% of sinus records reach narrowness 0.60 that way.
# Three constraints together separate them (measured 34 AFLT / 50 AFIB /
# 50 sinus): flutter 53%, AF 0%, sinus 4%.
_FLUTTER_RAW_NARROWNESS_MIN = 0.80
# Dominant frequency over ventricular rate. Real flutter conducts at a small
# whole-number ratio (2:1 typically, measured median 2.01); a sinus harmonic
# usually lands higher (measured median 3.92). This only narrows the confound,
# it does not remove it -- half the high-narrowness sinus records also sit at
# 2-3 -- so it is used alongside the other two, never alone.
_FLUTTER_RAW_MAX_CONDUCTION_RATIO = 3.0
_FLUTTER_RAW_RATIO_INTEGER_TOLERANCE = 0.15
# Organised-P level above which a narrow spectral peak is read as a harmonic
# of that P sequence rather than as flutter. Sits above `_P_ABSENCE_RATIO_GATE`
# because vetoing a positive finding needs firmer ground than merely failing
# to establish P absence -- see the note at its use site for the measurement.
_FLUTTER_VETO_ORGANISED_P_RATIO = 0.75
FLUTTER_SPECTRAL_PEAK_RATIO = 6.0         # peak-vs-floor ratio to call a candidate
ATRIAL_SPECTRAL_MIN_HZ = 0.5
ATRIAL_SPECTRAL_MAX_HZ = 15.0
FLUTTER_SPECTRAL_HALF_WIDTH_HZ = 0.30
FLUTTER_MULTILEAD_BPM_TOLERANCE = 30.0
FLUTTER_MULTILEAD_MIN_CONFIDENCE = 0.55
# Normalising references for the F-wave shape descriptors. A clean sawtooth
# runs well above both; these set where the score saturates, not a pass mark.
_F_WAVE_ASYMMETRY_REFERENCE = 1.0
_F_WAVE_CONTINUITY_REFERENCE = 0.80
# Morphology consensus: how many leads must show flutter-shaped residual, and
# how strong each has to be, before morphology is treated as established
# independently of QRST-template validation.
_F_WAVE_MORPHOLOGY_MIN_SCORE = 0.55
_F_WAVE_MORPHOLOGY_MIN_LEADS = 3
# Typical (counterclockwise) flutter is negative in the inferior leads and
# positive in V1. Opposition across those groups is the specific sign.
_F_WAVE_INFERIOR_LEADS = ("II", "III", "aVF")
FIBRILLATORY_MIN_CONFIDENCE = 0.55
# RR irregularity required before AF can be called, and the band just below it
# that abstains instead. Calibrated against 150 AFIB and 100 sinus records:
# rr_cv separates them at AUC 0.913, but the former 0.15 sat well inside the
# AF distribution (AFIB p25 0.128, median 0.173) and was the single reason
# 58 of the 73 missed AF records were missed -- their rr_cv clusters right
# underneath it (median 0.126, p75 0.148). Moving to 0.12 takes the share of
# AFIB meeting the criterion from 61% to 81% while sinus goes 10% -> 12%, and
# widens the separation from 51 to 69 points. Flutter is unaffected: it is a
# regular rhythm, median rr_cv 0.036.
_AF_RR_CV_MIN = 0.12
_AF_RR_CV_NEAR_MIN = 0.09
# Organized-P gate for AF.
#
# Do NOT try to stabilise this gate with a dead band / indeterminate zone --
# measured and rejected 2026-08-08. A P-wave detector change moves
# organized_p_ratio on 51% of records with median |delta| 0.091 (p90 0.209),
# i.e. 95% of the movement exceeds any plausible band half-width. A band does
# not remove the boundary, it replaces one boundary with two, so records still
# cross it: flips went UP (5 -> 7 per 284 records) and sinus false-AF doubled.
#
# Root cause is quantisation, not noise: the ratio is an integer beat count
# over ~13 beats in a 10 s record, so its resolution floor is 1/13 = 0.077 and
# the gate sits between 7/13 = 0.538 and 8/13 = 0.615 -- one beat decides AF.
# Confidence weighting cannot smooth it either (90% of conducted events carry
# confidence exactly 1.00).
_P_ABSENCE_RATIO_GATE = 0.60
# Where the ratio stops carrying the decision. Measured AUC for AFIB-vs-sinus
# separation inside this band collapses to 0.665 (0.524 at its centre) while
# the ratio scores 0.872 globally -- i.e. it is close to a coin flip exactly
# where it is being asked to decide. PR-interval dispersion is independent,
# continuous (no 1/n floor) and holds AUC 0.809 in the same band, so it
# adjudicates there instead of the ratio.
_P_ABSENCE_RATIO_AMBIGUOUS_LO = 0.45
_P_ABSENCE_RATIO_AMBIGUOUS_HI = 0.75
# Beat-to-beat PR scatter (MAD, robust sigma) above which the "conducted P"
# sequence is not a real sinus sequence. Sinus median in-band is 12.6 ms
# against 28.2 ms for AF.
_PR_DISPERSION_DISORGANISED_MS = 20.0
# MAD needs a few beats to mean anything; below this the tiebreaker abstains
# and the ratio keeps the decision.
_PR_DISPERSION_MIN_BEATS = 4
STANDARD_12_LEAD_ORDER = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
QRST_RESIDUAL_SCAFFOLD_METHOD = "baseline_subtracted_qrst_window_scaffold"
QRST_TEMPLATE_SUBTRACTION_METHOD = "r_centered_qrst_template_subtraction"
QRST_TEMPLATE_RESIDUAL_RMS_RATIO_MAX = 0.25
QRST_TEMPLATE_HIGH_FREQUENCY_RATIO_MAX = 1.25
QRST_TEMPLATE_MIN_RR_MS_FOR_P_PROTECTION = 520.0
# Internal marker for a deflection that only the T wave can account for. It is
# never published: `extract_atrial_events` drops these instead of asserting a
# retrograde or blocked mechanism the detector cannot support.
_T_WAVE_ARTIFACT = "t_wave_artifact"
# Widen each beat's measured T bounds slightly before testing containment: the
# residual peak of an imperfectly cancelled T sits near, not exactly on, the
# measured onset/offset.
_T_WAVE_ARTIFACT_GUARD_MS = 20.0
# PR bounds that define the `conducted` association window. Named so the T-wave
# gate can reserve the same region and never delete a real conducted P.
_CONDUCTED_P_MIN_PR_MS = 60.0
_CONDUCTED_P_MAX_PR_MS = 280.0
# A published atrial event has to be large enough to be a P wave at all. The
# QRST-subtraction residual routinely produces few-microvolt deflections that
# were previously exported with confidence 1.0.
_ATRIAL_EVENT_MIN_AMPLITUDE_MV = 0.020
# Amplitude at which an atrial deflection is credible enough to carry the
# detector's full confidence; smaller events are derated proportionally.
_ATRIAL_EVENT_REFERENCE_AMPLITUDE_MV = 0.050


def _cluster_atrial_candidates(
    candidates: List[Dict[str, Any]],
    merge_samples: int,
) -> List[Dict[str, Any]]:
    if not candidates:
        return []
    candidates = sorted(candidates, key=lambda item: int(item["sample"]))
    groups: List[List[Dict[str, Any]]] = [[candidates[0]]]
    for cand in candidates[1:]:
        if int(cand["sample"]) - int(groups[-1][-1]["sample"]) <= merge_samples:
            groups[-1].append(cand)
        else:
            groups.append([cand])

    out: List[Dict[str, Any]] = []
    for idx, group in enumerate(groups):
        samples = [int(item["sample"]) for item in group]
        confs = [float(item["confidence"]) for item in group]
        source_leads = sorted({str(item["lead"]) for item in group})
        winner = max(group, key=lambda item: float(item["confidence"]))
        event: Dict[str, Any] = {
            "event_id": idx,
            "sample": int(round(float(np.mean(samples)))),
            "confidence": float(np.mean(confs)),
            "source_leads": source_leads,
        }
        for key in (
            "onset_sample",
            "offset_sample",
            "amplitude_mv",
            "area_mv_ms",
            "signed_area_mv_ms",
            "detection_function_peak",
        ):
            if key in winner:
                event[key] = winner[key]
        if any("onset_sample" in item for item in group):
            event["onset_sample"] = int(
                min(int(item["onset_sample"]) for item in group if "onset_sample" in item)
            )
        if any("offset_sample" in item for item in group):
            event["offset_sample"] = int(
                max(int(item["offset_sample"]) for item in group if "offset_sample" in item)
            )
        out.append(event)
    return out


def _beat_t_windows(
    beat_features,
    fs: int,
    beat_count: int,
    qrs_onsets: Optional[np.ndarray] = None,
) -> Dict[int, Tuple[int, int]]:
    """Per-beat T extent, as the cross-lead median of the measured T bounds.

    Leads disagree on T onset/offset by tens of milliseconds; the median is used
    rather than the union so a single mis-delineated lead cannot swallow the
    whole ST-T segment and suppress genuine atrial events.
    """
    onsets: Dict[int, List[int]] = {}
    offsets: Dict[int, List[int]] = {}
    for bf in beat_features:
        beat_id = int(bf.beat_id)
        if beat_id >= beat_count:
            continue
        t_on = getattr(bf.t, "onset", None)
        t_off = getattr(bf.t, "offset", None)
        if t_on is None or t_off is None or int(t_off) <= int(t_on):
            continue
        onsets.setdefault(beat_id, []).append(int(t_on))
        offsets.setdefault(beat_id, []).append(int(t_off))

    guard = int(round(_T_WAVE_ARTIFACT_GUARD_MS * fs / 1000.0))
    # Never let a T window reach into the region where the next beat's P wave is
    # expected. An over-estimated T offset would otherwise silently delete real
    # conducted P waves, which is the failure this whole gate exists to avoid.
    p_region = int(round(_CONDUCTED_P_MAX_PR_MS * fs / 1000.0))
    onsets_arr = (
        np.asarray(qrs_onsets, dtype=int) if qrs_onsets is not None else None
    )
    windows: Dict[int, Tuple[int, int]] = {}
    for beat_id, on_values in onsets.items():
        off_values = offsets.get(beat_id)
        if not off_values:
            continue
        start = int(round(float(np.median(on_values)))) - guard
        stop = int(round(float(np.median(off_values)))) + guard
        if onsets_arr is not None and onsets_arr.size:
            later = onsets_arr[onsets_arr > start]
            if later.size:
                stop = min(stop, int(later.min()) - p_region)
        if stop > start:
            windows[beat_id] = (start, stop)
    return windows


def _event_in_any_t_wave(
    event_sample: int,
    t_windows: Optional[Dict[int, Tuple[int, int]]],
) -> bool:
    if not t_windows:
        return False
    return any(
        start <= int(event_sample) <= stop for start, stop in t_windows.values()
    )


def _below_atrial_event_amplitude_floor(event: Dict[str, Any]) -> bool:
    """True when a measured event is too small to be a P wave.

    Only applies to events that carry a measured amplitude; events without one
    are left to the timing gates rather than being dropped for missing data.
    """
    amplitude = event.get("amplitude_mv")
    if amplitude is None:
        return False
    try:
        return abs(float(amplitude)) < _ATRIAL_EVENT_MIN_AMPLITUDE_MV
    except (TypeError, ValueError):
        return False


def _associate_event_to_qrs(
    event_sample: int,
    qrs_onsets: np.ndarray,
    qrs_offsets: np.ndarray,
    fs: int,
    t_windows: Optional[Dict[int, Tuple[int, int]]] = None,
) -> Tuple[Optional[int], str, Optional[float]]:
    # A deflection sitting inside a measured T wave carries no evidence that it
    # is atrial at all: the 0-240 ms post-QRS-offset "retrograde" window is the
    # ST-T segment, so an imperfectly cancelled T in the QRST-subtraction
    # residual lands there. Reject such an event outright instead of asserting
    # any mechanism for it. This test must come before the candidate search --
    # merely dropping the retrograde option would let the same artifact fall
    # through to the `conducted` window and be published as an organised P wave,
    # which is worse than the retrograde label it replaced.
    if _event_in_any_t_wave(event_sample, t_windows):
        return None, _T_WAVE_ARTIFACT, None

    candidates: List[Tuple[float, int, int, str, Optional[float]]] = []
    for beat_id, qrs_on in enumerate(qrs_onsets):
        dt_ms = (int(qrs_on) - int(event_sample)) * 1000.0 / fs
        if _CONDUCTED_P_MIN_PR_MS <= dt_ms <= _CONDUCTED_P_MAX_PR_MS:
            candidates.append((abs(float(dt_ms)), 0, int(beat_id), "conducted", float(dt_ms)))
    for beat_id, qrs_off in enumerate(qrs_offsets):
        dt_ms = (int(event_sample) - int(qrs_off)) * 1000.0 / fs
        if 0.0 <= dt_ms <= 240.0:
            candidates.append((abs(float(dt_ms)), 1, int(beat_id), "retrograde", None))
    if candidates:
        # Proximity first, interpretation second. This ordering is load-bearing
        # and deliberately NOT "conducted wins ties": see
        # docs/诊断层待修改清单.md item C. Preferring `conducted` whenever both
        # windows admit the same deflection does raise organized_p_ratio on
        # sinus tachycardia (measured 0.10 -> 0.90 on JS00960), but it also
        # reads flutter F waves as conducted P waves (JS01040, a confirmed
        # flutter, went 0.05 -> 0.84). That defeats the disorganized-atrial
        # -activity guard and would let flutter be reported as sinus, so the
        # naive reordering is a net loss. Disambiguating the two needs
        # evidence beyond proximity -- PR-cluster stability across beats --
        # which is not implemented yet.
        _, _, beat_id, assoc, pr_ms = min(candidates, key=lambda item: item[:3])
        return beat_id, assoc, pr_ms
    return None, "blocked", None


def _scan_interbeat_atrial_candidates(
    ecg: np.ndarray,
    fs: int,
    r_locs: np.ndarray,
    qrs_offsets: np.ndarray,
    quality,
    preferred_leads: Sequence[str] = PREFERRED_ATRIAL_LEADS,
) -> List[Dict[str, Any]]:
    if len(r_locs) < 2:
        return []

    ecg_arr = np.asarray(ecg, dtype=float)
    if ecg_arr.ndim != 2:
        return []

    r_samples = np.asarray(r_locs, dtype=int)
    qrs_off_samples = np.asarray(qrs_offsets, dtype=int)
    lead_to_idx = {lead: idx for idx, lead in enumerate(STANDARD_12_LEAD_ORDER)}
    lead_indices = [
        (lead, lead_to_idx[lead])
        for lead in preferred_leads
        if lead in lead_to_idx
        and lead_to_idx[lead] < ecg_arr.shape[0]
        and getattr(quality.get(lead), "reliable_for_p", False)
    ]
    if not lead_indices:
        return []

    guard = int(round(0.040 * fs))
    min_window = int(round(0.080 * fs))
    peak_distance = max(1, int(round(0.080 * fs)))
    noise_window = max(1, int(round(0.120 * fs)))
    epsilon = 1e-6
    candidates: List[Dict[str, Any]] = []

    for beat_idx in range(len(r_samples) - 1):
        prev_qrs_off = (
            int(qrs_off_samples[beat_idx])
            if beat_idx < len(qrs_off_samples)
            else int(r_samples[beat_idx])
        )
        start = max(0, prev_qrs_off + guard)
        stop = min(ecg_arr.shape[1], int(r_samples[beat_idx + 1]) - guard)
        if stop - start < min_window:
            continue

        for lead, lead_idx in lead_indices:
            segment = ecg_arr[lead_idx, start:stop]
            if segment.size < min_window:
                continue
            # Estimate noise robustly over the whole inter-beat window rather
            # than its first 120 ms. That head is the ST segment and early T
            # wave, so on a flat ST the estimate collapsed and the floor below
            # was all that stopped T/U/baseline ripple being harvested as atrial
            # events. The window spans ST, T, TP and P, but T and P occupy a
            # minority of its samples, so a MAD over the whole span tracks the
            # isoelectric level without having to guess where it sits.
            noise = float(np.median(np.abs(segment - np.median(segment)))) * 1.4826 + epsilon
            height = max(_ATRIAL_EVENT_MIN_AMPLITUDE_MV, 3.0 * noise)
            peaks, props = find_peaks(segment, height=height, distance=peak_distance)
            for peak_idx, peak_height in zip(peaks, props.get("peak_heights", [])):
                candidates.append(
                    {
                        "sample": int(start + int(peak_idx)),
                        "lead": lead,
                        "confidence": float(min(1.0, float(peak_height) / (height * 3.0))),
                    }
                )

    merged = _cluster_atrial_candidates(
        candidates,
        merge_samples=max(12, int(0.040 * fs)),
    )
    return [
        {
            "p_event_id": int(event["event_id"]),
            "sample": int(event["sample"]),
            "time_ms": float(event["sample"]) * 1000.0 / fs,
            "confidence": float(event["confidence"]),
            "associated_qrs_beat_id": None,
            "association_type": "blocked",
            "pr_ms": None,
            "source_leads": list(event["source_leads"]),
        }
        for event in merged
    ]


def _deduplicate_atrial_events(
    events: List[Dict[str, Any]],
    merge_samples: int,
) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    for event in sorted(events, key=lambda item: (int(item["sample"]), float(item["time_ms"]))):
        overlap_idx = next(
            (
                idx
                for idx, existing in enumerate(deduped)
                if abs(int(existing["sample"]) - int(event["sample"])) <= merge_samples
            ),
            None,
        )
        if overlap_idx is None:
            deduped.append(dict(event))
            continue

        existing = deduped[overlap_idx]
        existing_nonblocked = existing.get("association_type") != "blocked"
        event_nonblocked = event.get("association_type") != "blocked"
        winner = event if event_nonblocked and not existing_nonblocked else existing
        loser = existing if winner is event else event

        # The raw inter-beat scan is intentionally sensitive; overlapping scanner-only
        # blocked hits must not conflict with a qrs-associated P event from delineation.
        winner["confidence"] = max(float(winner["confidence"]), float(loser["confidence"]))
        winner["source_leads"] = sorted(
            set(winner.get("source_leads", [])) | set(loser.get("source_leads", []))
        )
        deduped[overlap_idx] = dict(winner)
    return deduped


def _spectral_flutter_analysis(residual: np.ndarray, fs: int) -> Dict[str, Any]:
    """Characterize organized F waves and disorganized f-wave activity.

    The old implementation used one FFT peak-to-floor ratio.  That is very
    sensitive to QRST-subtraction remnants.  Use a Welch PSD and require both
    a peak in the physiologic flutter band and concentration of power around
    the fundamental/harmonics.  Spectral entropy and the complementary broad-
    band score are exported as evidence for fibrillatory (lower-case f) waves;
    they never diagnose AF without RR/P-wave context.
    """
    unavailable = {
        "spectral_flutter_candidate": False,
        "spectral_flutter_bpm": None,
        "spectral_peak_ratio": None,
        "spectral_flutter_confidence": 0.0,
        "spectral_entropy": None,
        "organized_spectral_power_ratio": None,
        "flutter_harmonic_power_ratio": None,
        "fibrillatory_spectral_confidence": 0.0,
    }
    if fs <= 0:
        return dict(unavailable)
    sig = np.asarray(residual, dtype=float).reshape(-1)
    sig = sig[np.isfinite(sig)]
    if sig.size < int(round(1.5 * fs)):
        return dict(unavailable)
    sig = sig - float(np.mean(sig))
    if float(np.dot(sig, sig)) <= 1e-12:
        return dict(unavailable)

    # Decimate toward the target rate; fall back gracefully for short segments.
    q = max(1, int(round(float(fs) / FLUTTER_SPECTRAL_TARGET_FS)))
    eff_fs = float(fs)
    if q > 1 and sig.size > 27 * q:
        try:
            sig = np.asarray(decimate(sig, q, ftype="fir", zero_phase=True), dtype=float)
            eff_fs = float(fs) / float(q)
        except ValueError:
            eff_fs = float(fs)

    nperseg = min(sig.size, max(128, int(round(4.0 * eff_fs))))
    if nperseg < 32:
        return dict(unavailable)
    freqs, psd = welch(
        sig,
        fs=eff_fs,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        nfft=max(FLUTTER_SPECTRAL_NFFT, nperseg),
        detrend="constant",
        scaling="density",
    )

    atrial_band = (freqs >= ATRIAL_SPECTRAL_MIN_HZ) & (freqs <= ATRIAL_SPECTRAL_MAX_HZ)
    if not np.any(atrial_band):
        return dict(unavailable)
    atrial_psd = np.asarray(psd[atrial_band], dtype=float)
    total_power = float(np.sum(atrial_psd))
    if total_power <= 1e-18:
        return dict(unavailable)
    probabilities = atrial_psd / total_power
    nonzero = probabilities[probabilities > 0]
    spectral_entropy = float(
        -np.sum(nonzero * np.log(nonzero)) / np.log(max(2, probabilities.size))
    )

    band = (freqs >= FLUTTER_SPECTRAL_MIN_BPM / 60.0) & (freqs <= FLUTTER_SPECTRAL_MAX_BPM / 60.0)
    if not np.any(band):
        return dict(unavailable)
    band_psd = psd[band]
    band_freqs = freqs[band]
    peak_i = int(np.argmax(band_psd))
    peak_power = float(band_psd[peak_i])
    peak_bpm = float(band_freqs[peak_i]) * 60.0

    # Compare with the atrial-band floor, not high-frequency bins that can make
    # a small subtraction artefact appear infinitely prominent.
    floor = float(np.median(atrial_psd))
    peak_ratio = peak_power / (floor + 1e-12)

    fundamental_hz = float(band_freqs[peak_i])
    fundamental_mask = np.abs(freqs - fundamental_hz) <= FLUTTER_SPECTRAL_HALF_WIDTH_HZ
    fundamental_power = float(np.sum(psd[fundamental_mask]))
    harmonic_power = 0.0
    harmonic_count = 0
    for multiple in (2.0, 3.0):
        harmonic_hz = fundamental_hz * multiple
        if harmonic_hz > ATRIAL_SPECTRAL_MAX_HZ:
            continue
        harmonic_mask = np.abs(freqs - harmonic_hz) <= FLUTTER_SPECTRAL_HALF_WIDTH_HZ
        harmonic_power += float(np.sum(psd[harmonic_mask]))
        harmonic_count += 1
    organized_power_ratio = float(
        np.clip((fundamental_power + harmonic_power) / (total_power + 1e-12), 0.0, 1.0)
    )
    harmonic_ratio = float(
        np.clip(harmonic_power / (fundamental_power + 1e-12), 0.0, 1.0)
    )

    # Narrowness of the dominant peak, normalised against a *local* 2.5 Hz
    # envelope rather than the whole atrial band. `organized_power_ratio`
    # above divides by total 0.5-15 Hz power, so drift, muscle and residual
    # QRS dilute it; measured on labelled flutter it lands at 0.09-0.17, i.e.
    # entirely below its own 0.18 gate, which is why the spectral detector
    # fired on 0 of 34 flutter records.
    narrow_mask = np.abs(freqs - fundamental_hz) <= FLUTTER_SPECTRAL_NARROW_PEAK_HALF_HZ
    envelope_mask = (
        np.abs(freqs - fundamental_hz) <= FLUTTER_SPECTRAL_NARROW_ENVELOPE_HZ / 2.0
    )
    envelope_power = float(np.sum(psd[envelope_mask]))
    peak_narrowness = float(
        np.clip(float(np.sum(psd[narrow_mask])) / (envelope_power + 1e-18), 0.0, 1.0)
    )

    peak_score = float(np.clip((peak_ratio - 4.0) / 16.0, 0.0, 1.0))
    concentration_score = float(np.clip((organized_power_ratio - 0.12) / 0.38, 0.0, 1.0))
    organization_score = float(np.clip((0.85 - spectral_entropy) / 0.55, 0.0, 1.0))
    narrowness_score = float(
        np.clip((peak_narrowness - 0.35) / 0.30, 0.0, 1.0)
    )
    confidence = float(
        np.clip(
            0.30 * peak_score
            + 0.25 * concentration_score
            + 0.10 * organization_score
            + 0.35 * narrowness_score,
            0.0,
            1.0,
        )
    )
    # Two independent ways to qualify. The original height/concentration test
    # is kept as-is so nothing that used to be detected stops being detected;
    # the narrowness test is the measured one and is what actually fires on
    # real flutter.
    height_path = bool(
        peak_ratio >= FLUTTER_SPECTRAL_PEAK_RATIO
        and organized_power_ratio >= 0.18
        and confidence >= 0.50
    )
    narrowness_path = bool(
        peak_narrowness >= FLUTTER_SPECTRAL_NARROWNESS_MIN
    )
    candidate = bool(height_path or narrowness_path)

    # AF f waves are broad-band and poorly organized.  This score is supportive
    # only; muscle/noise can look similar, so downstream logic also requires an
    # irregular ventricular response, absent consistent P waves, and quality.
    broadband_score = float(np.clip((spectral_entropy - 0.45) / 0.45, 0.0, 1.0))
    nonconcentration_score = float(np.clip((0.45 - organized_power_ratio) / 0.35, 0.0, 1.0))
    fibrillatory_confidence = float(
        np.clip(0.65 * broadband_score + 0.35 * nonconcentration_score, 0.0, 1.0)
    )
    return {
        "spectral_flutter_candidate": candidate,
        "spectral_flutter_bpm": peak_bpm if candidate else None,
        "spectral_peak_ratio": float(peak_ratio),
        "spectral_peak_narrowness": peak_narrowness,
        "spectral_flutter_qualified_by": (
            "narrowness" if narrowness_path and not height_path
            else "height" if height_path and not narrowness_path
            else "both" if candidate else "none"
        ),
        "spectral_flutter_confidence": confidence if candidate else 0.0,
        "spectral_entropy": spectral_entropy,
        "organized_spectral_power_ratio": organized_power_ratio,
        "flutter_harmonic_power_ratio": harmonic_ratio if harmonic_count else 0.0,
        "fibrillatory_spectral_confidence": fibrillatory_confidence,
    }


def _flutter_wave_morphology(
    signal: np.ndarray, fs: int, dominant_lag: int
) -> Dict[str, Any]:
    """Shape descriptors for flutter waves in a QRST-subtracted residual.

    The spectral detector answers "is there a periodic component in the
    flutter band". It cannot answer "does that component look like a flutter
    wave", which is what separates true F waves from a periodic artefact or a
    poorly cancelled T wave. Two shape properties do:

    * **Sawtooth asymmetry** -- a flutter wave has a slow limb and a fast
      limb, so the residual amplitude distribution is skewed. A sinusoidal
      artefact is symmetric.
    * **Continuity** -- flutter has no isoelectric interval between waves, so
      the residual spends little time near baseline. Discrete P waves or
      isolated artefact spend most of the cycle at baseline.

    Both are polarity-signed so callers can test the classic typical-flutter
    signature (inferior leads negative, V1 positive). This function judges one
    lead only; the cross-lead check lives in the multilead aggregator.
    """
    result: Dict[str, Any] = {
        "F_wave_sawtooth_asymmetry": None,
        "F_wave_continuity": None,
        "F_wave_polarity": None,
        "F_wave_morphology_score": 0.0,
    }
    if signal.size < 3 or dominant_lag <= 0:
        return result
    amplitude = float(np.max(np.abs(signal)))
    if amplitude <= 1e-9:
        return result

    # Skewness of the amplitude distribution. Signed so that a predominantly
    # negative sawtooth (inferior leads in typical flutter) reads negative.
    centred = signal - float(np.mean(signal))
    sd = float(np.std(centred))
    if sd <= 1e-12:
        return result
    skewness = float(np.mean((centred / sd) ** 3))
    # A sawtooth's slow/fast limbs also make the slope distribution asymmetric,
    # which is more robust than amplitude skew when the wave is clipped.
    slopes = np.diff(centred)
    slope_sd = float(np.std(slopes))
    slope_skew = (
        float(np.mean((slopes - float(np.mean(slopes))) ** 3) / (slope_sd**3 + 1e-12))
        if slope_sd > 1e-12
        else 0.0
    )
    asymmetry = float(np.clip(abs(skewness) * 0.5 + abs(slope_skew) * 0.5, 0.0, 5.0))

    # Fraction of the record spending meaningful excursion from baseline.
    near_baseline = np.abs(centred) < (0.15 * amplitude)
    continuity = float(1.0 - (np.count_nonzero(near_baseline) / float(centred.size)))

    result["F_wave_sawtooth_asymmetry"] = asymmetry
    result["F_wave_continuity"] = continuity
    result["F_wave_polarity"] = (
        "negative" if skewness < 0 else "positive" if skewness > 0 else None
    )
    # Kept deliberately coarse: this is a screening descriptor feeding a
    # consensus vote, not a calibrated probability.
    result["F_wave_morphology_score"] = float(
        np.clip(
            0.5 * min(1.0, asymmetry / _F_WAVE_ASYMMETRY_REFERENCE)
            + 0.5 * min(1.0, continuity / _F_WAVE_CONTINUITY_REFERENCE),
            0.0,
            1.0,
        )
    )
    return result


def _summarize_atrial_residual(residual: np.ndarray, fs: int) -> Dict[str, Any]:
    """Summarize organization in an atrial residual-like signal.

    This is an engineering scaffold for rhythm-rule inputs, not a validated
    clinical atrial activity estimator.
    """
    if fs <= 0:
        return {"available": False, "reason": "invalid_fs"}

    signal = np.asarray(residual, dtype=float).reshape(-1)
    signal = signal[np.isfinite(signal)]
    min_samples = int(round(1.5 * fs))
    if signal.size < min_samples:
        return {"available": False, "reason": "residual_too_short"}

    signal = signal - float(np.mean(signal))
    rms = float(np.sqrt(np.mean(signal * signal))) if signal.size else 0.0
    energy = float(np.dot(signal, signal))
    if energy <= 1e-12:
        return {
            "available": True,
            "residual_rms_mv": rms,
            "repetitiveness": 0.0,
            "stability": 0.0,
            "dominant_cycle_ms": None,
        }

    lag_min = max(1, int(round(0.120 * fs)))
    lag_max = min(signal.size - 1, int(round(0.400 * fs)))
    if lag_max <= lag_min:
        return {"available": False, "reason": "residual_too_short_for_cycle_scan"}

    # Compute every numerator in one FFT correlation and every denominator
    # from prefix energies.  The former implementation sliced the full signal
    # and ran three BLAS reductions for every lag, which dominated AF/AFL
    # analysis on otherwise short 10-second records.
    lags = np.arange(lag_min, lag_max + 1, dtype=int)
    autocorrelation = correlate(signal, signal, mode="full", method="fft")
    numerators = np.asarray(
        autocorrelation[signal.size - 1 + lags],
        dtype=float,
    )
    squared_prefix = np.concatenate(
        ([0.0], np.cumsum(signal * signal, dtype=float))
    )
    first_energy = squared_prefix[signal.size - lags]
    second_energy = squared_prefix[signal.size] - squared_prefix[lags]
    denominators = np.sqrt(np.maximum(first_energy * second_energy, 0.0))
    lag_scores_array = np.divide(
        numerators,
        denominators + 1e-12,
        out=np.zeros_like(numerators),
        where=denominators > 0.0,
    )

    dominant_offset = int(np.argmax(lag_scores_array))
    dominant_lag = lag_min + dominant_offset
    repetitiveness = float(
        np.clip(lag_scores_array[dominant_offset], 0.0, 1.0)
    )

    shifted_a = signal[:-dominant_lag]
    shifted_b = signal[dominant_lag:]
    denom = float(np.linalg.norm(shifted_a) * np.linalg.norm(shifted_b))
    cycle_corr = float(np.dot(shifted_a, shifted_b) / (denom + 1e-12)) if denom > 0 else 0.0
    stability = float(max(0.0, min(1.0, cycle_corr)))

    summary = {
        "available": True,
        "residual_rms_mv": rms,
        "repetitiveness": repetitiveness,
        "stability": stability,
        "dominant_cycle_ms": float(dominant_lag) * 1000.0 / float(fs),
    }
    summary.update(_spectral_flutter_analysis(signal, fs))
    summary.update(_flutter_wave_morphology(signal, fs, dominant_lag))
    spectral_f = float(summary.get("fibrillatory_spectral_confidence") or 0.0)
    temporal_disorganization = float(np.clip(1.0 - repetitiveness, 0.0, 1.0))
    summary["fibrillatory_wave_confidence"] = float(
        np.clip(0.70 * spectral_f + 0.30 * temporal_disorganization, 0.0, 1.0)
    )
    return summary


def _aggregate_multilead_atrial_activity(
    per_lead: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Require reproducible atrial activity across independent ECG leads."""
    available = {
        lead: row
        for lead, row in per_lead.items()
        if isinstance(row, dict) and bool(row.get("available"))
    }
    if not available:
        return {
            "atrial_activity_lead_count": 0,
            "flutter_supporting_leads": [],
            "flutter_multilead_consensus": False,
            "F_wave_confidence": 0.0,
            "fibrillatory_supporting_leads": [],
            "f_wave_multilead_consensus": False,
            "f_wave_confidence": 0.0,
        }

    flutter_rows = [
        (lead, row)
        for lead, row in available.items()
        if bool(row.get("spectral_flutter_candidate"))
        and float(row.get("spectral_flutter_confidence") or 0.0)
        >= FLUTTER_MULTILEAD_MIN_CONFIDENCE
        and row.get("spectral_flutter_bpm") is not None
    ]
    best_cluster: List[Tuple[str, Dict[str, Any]]] = []
    for _, center_row in flutter_rows:
        center = float(center_row["spectral_flutter_bpm"])
        cluster = [
            (lead, row)
            for lead, row in flutter_rows
            if abs(float(row["spectral_flutter_bpm"]) - center)
            <= FLUTTER_MULTILEAD_BPM_TOLERANCE
        ]
        if len(cluster) > len(best_cluster):
            best_cluster = cluster

    minimum_support = 2 if len(available) >= 2 else 1
    flutter_fraction = float(len(best_cluster)) / float(len(available))
    flutter_consensus = bool(
        len(best_cluster) >= minimum_support and flutter_fraction >= 0.40
    )
    flutter_confidences = [
        float(row.get("spectral_flutter_confidence") or 0.0)
        for _, row in best_cluster
    ]
    flutter_confidence = (
        float(np.median(flutter_confidences)) * min(1.0, flutter_fraction / 0.60)
        if flutter_consensus and flutter_confidences
        else 0.0
    )
    flutter_bpms = [float(row["spectral_flutter_bpm"]) for _, row in best_cluster]

    f_rows = [
        (lead, row)
        for lead, row in available.items()
        if float(row.get("fibrillatory_wave_confidence") or 0.0)
        >= FIBRILLATORY_MIN_CONFIDENCE
        and not bool(row.get("spectral_flutter_candidate"))
    ]
    f_fraction = float(len(f_rows)) / float(len(available))
    f_consensus = bool(len(f_rows) >= minimum_support and f_fraction >= 0.60)
    f_confidences = [
        float(row.get("fibrillatory_wave_confidence") or 0.0)
        for _, row in f_rows
    ]
    f_confidence = (
        float(np.median(f_confidences)) * min(1.0, f_fraction / 0.60)
        if f_consensus and f_confidences
        else 0.0
    )

    # Morphology consensus, independent of the spectral vote. This is what
    # lets the flutter rule reach a decision when QRST-template validation is
    # unavailable, instead of reporting `unavailable` and letting downstream
    # P-wave rules measure F waves as P waves.
    morphology_rows = [
        (lead, row)
        for lead, row in available.items()
        if float(row.get("F_wave_morphology_score") or 0.0)
        >= _F_WAVE_MORPHOLOGY_MIN_SCORE
    ]
    morphology_leads = sorted(lead for lead, _ in morphology_rows)
    inferior_polarities = {
        str(row.get("F_wave_polarity"))
        for lead, row in morphology_rows
        if lead in _F_WAVE_INFERIOR_LEADS
    }
    v1_row = dict(available.get("V1") or {})
    v1_polarity = str(v1_row.get("F_wave_polarity") or "")
    # Typical flutter: inferior sawtooth negative while V1 is upright.
    polarity_opposition = bool(
        inferior_polarities == {"negative"} and v1_polarity == "positive"
    )
    morphology_consensus = bool(
        len(morphology_rows) >= min(_F_WAVE_MORPHOLOGY_MIN_LEADS, len(available))
    )
    morphology_scores = [
        float(row.get("F_wave_morphology_score") or 0.0)
        for _, row in morphology_rows
    ]
    morphology_confidence = (
        float(np.median(morphology_scores)) if morphology_scores else 0.0
    )
    if polarity_opposition:
        morphology_confidence = float(np.clip(morphology_confidence + 0.15, 0.0, 1.0))

    return {
        "atrial_activity_lead_count": len(available),
        "flutter_supporting_leads": sorted(lead for lead, _ in best_cluster),
        "flutter_support_fraction": flutter_fraction,
        "flutter_multilead_consensus": flutter_consensus,
        "F_wave_morphology_leads": morphology_leads,
        "F_wave_morphology_consensus": morphology_consensus,
        "F_wave_morphology_confidence": morphology_confidence,
        "F_wave_polarity_opposition": polarity_opposition,
        "F_wave_confidence": float(np.clip(flutter_confidence, 0.0, 1.0)),
        "F_wave_rate_bpm": float(np.median(flutter_bpms)) if flutter_bpms else None,
        "fibrillatory_supporting_leads": sorted(lead for lead, _ in f_rows),
        "fibrillatory_support_fraction": f_fraction,
        "f_wave_multilead_consensus": f_consensus,
        "f_wave_confidence": float(np.clip(f_confidence, 0.0, 1.0)),
    }


def _classify_af_afl(
    rr_ms: List[float],
    residual_summary: Dict[str, Any],
    organized_p_ratio: float,
    pr_dispersion_ms: Optional[float] = None,
) -> Dict[str, Any]:
    rr_clean: List[float] = []
    for value in rr_ms:
        try:
            rr = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(rr) and rr > 0:
            rr_clean.append(rr)
    rr_vals = np.asarray(rr_clean, dtype=float)
    rr_cv = (
        float(np.std(rr_vals) / (float(np.mean(rr_vals)) + 1e-6))
        if rr_vals.size >= 2
        else None
    )

    residual_available = bool(residual_summary.get("available", False))
    repetitiveness = float(residual_summary.get("repetitiveness") or 0.0) if residual_available else 0.0
    stability = float(residual_summary.get("stability") or 0.0) if residual_available else 0.0
    dominant_cycle_ms = residual_summary.get("dominant_cycle_ms")
    cycle_in_flutter_range = (
        dominant_cycle_ms is not None
        and 120.0 <= float(dominant_cycle_ms) <= 400.0
    )
    time_domain_confidence = (
        0.5 * (repetitiveness + stability)
        if residual_available and cycle_in_flutter_range
        else 0.0
    )
    # Farrell 2003: spectral flutter evidence runs in parallel with the
    # time-domain cycle test; take whichever is stronger.
    spectral_candidate = bool(residual_summary.get("spectral_flutter_candidate"))
    spectral_confidence = (
        float(residual_summary.get("spectral_flutter_confidence") or 0.0)
        if residual_available and spectral_candidate
        else 0.0
    )
    legacy_flutter_confidence = max(time_domain_confidence, spectral_confidence)
    has_multilead_result = "flutter_multilead_consensus" in residual_summary
    # A narrow dominant F-wave peak is independent evidence of flutter and, on
    # this data, the only one that actually fires: the existing consensus path
    # was false on all 34 labelled flutter records. It corroborates rather than
    # replaces -- either route can establish flutter, and the confidence taken
    # forward is the stronger of the two.
    # An organised P sequence and flutter are mutually exclusive descriptions
    # of the same atrial activity, so this condition belongs to both spectral
    # routes, not just the raw one: whichever route found the narrow peak, on
    # a record that also shows organised P waves that peak is a harmonic of
    # those P waves rather than an F wave.
    #
    # It vetoes at a higher ratio than the AF gate on purpose. Reusing 0.60
    # here was measured and is too blunt: it removed one sinus false flutter
    # but cost two true flutter detections and pushed three more flutter
    # records into being called AF, a worse error. Real organised P in sinus
    # measures far above the gate (sinus median 0.826), whereas the spurious
    # value a flutter record picks up sits well below it (0.667 on PTB-XL
    # 1773) -- so the veto is placed between the two.
    organised_p_present = organized_p_ratio > _FLUTTER_VETO_ORGANISED_P_RATIO
    narrow_peak_consensus = bool(
        residual_summary.get("flutter_narrow_peak_consensus")
        and not organised_p_present
    )
    narrow_peak_confidence = (
        float(residual_summary.get("flutter_peak_narrowness") or 0.0)
        if narrow_peak_consensus
        else 0.0
    )
    # Phase-locked (regular) flutter, read off the unsubtracted signal. The
    # third constraint lives here rather than in the residual builder: a
    # narrow peak on a record that also shows an organised P sequence is a
    # harmonic of that sequence, not an F wave.
    raw_narrow_peak = bool(
        residual_summary.get("flutter_raw_narrow_peak")
        and not organised_p_present
    )
    if raw_narrow_peak:
        narrow_peak_consensus = True
        narrow_peak_confidence = max(
            narrow_peak_confidence,
            float(residual_summary.get("flutter_raw_peak_narrowness") or 0.0),
        )
    flutter_consensus = (
        (
            bool(residual_summary.get("flutter_multilead_consensus"))
            or narrow_peak_consensus
        )
        if has_multilead_result
        else bool(spectral_candidate or cycle_in_flutter_range)
    )
    flutter_wave_confidence = max(
        (
            float(residual_summary.get("F_wave_confidence") or 0.0)
            if has_multilead_result
            else legacy_flutter_confidence
        ),
        narrow_peak_confidence,
    )
    f_wave_confidence = float(
        residual_summary.get("f_wave_confidence") or 0.0
        if "f_wave_confidence" in residual_summary
        else residual_summary.get("fibrillatory_wave_confidence") or 0.0
    )
    f_wave_consensus = bool(
        residual_summary.get("f_wave_multilead_consensus", f_wave_confidence >= 0.55)
    )
    validated_residual = bool(residual_summary.get("validated_qrst_subtraction"))

    strong_f_wave = bool(f_wave_consensus and f_wave_confidence >= 0.55)
    p_ratio_overruled_by_f_wave = bool(
        validated_residual
        and strong_f_wave
        and organized_p_ratio < 0.98
    )
    # Inside the ambiguous band the ratio is ~a coin flip, so PR scatter
    # adjudicates instead: P waves that do not march with the ventricles at a
    # fixed interval are not an organized atrial rhythm, however many of them
    # were detected. Outside the band the ratio is strong and keeps the call.
    p_ratio_in_ambiguous_band = bool(
        _P_ABSENCE_RATIO_AMBIGUOUS_LO
        <= organized_p_ratio
        <= _P_ABSENCE_RATIO_AMBIGUOUS_HI
    )
    pr_dispersion_decisive = bool(
        p_ratio_in_ambiguous_band and pr_dispersion_ms is not None
    )
    if pr_dispersion_decisive:
        p_wave_absent_by_measurement = bool(
            float(pr_dispersion_ms) > _PR_DISPERSION_DISORGANISED_MS
        )
    else:
        p_wave_absent_by_measurement = bool(
            organized_p_ratio <= _P_ABSENCE_RATIO_GATE
        )
    p_wave_absence_or_overruled = bool(
        p_wave_absent_by_measurement or p_ratio_overruled_by_f_wave
    )
    classic_af_evidence = bool(
        rr_cv is not None
        and rr_cv >= _AF_RR_CV_MIN
        and p_wave_absence_or_overruled
        and repetitiveness < 0.55
    )
    strong_F_wave = bool(flutter_consensus and flutter_wave_confidence >= 0.60)

    # Variable AV block can make flutter RR intervals highly irregular.  A
    # reproducible, narrow-band F-wave signal across leads therefore outranks
    # RR irregularity.  Conversely, broad-band f-wave activity supports AF but
    # never substitutes for the RR/P-wave criteria.  If both morphologies are
    # genuinely strong and close, abstain instead of forcing a diagnosis.
    conflicting_morphology = bool(
        strong_F_wave
        and strong_f_wave
        and abs(flutter_wave_confidence - f_wave_confidence) < 0.10
    )
    near_af_threshold = bool(
        rr_cv is not None
        and _AF_RR_CV_NEAR_MIN <= rr_cv < _AF_RR_CV_MIN
        and p_wave_absent_by_measurement
        and not strong_F_wave
    )
    unvalidated_f_wave_pattern = bool(
        rr_cv is not None
        and rr_cv >= _AF_RR_CV_MIN
        and not p_wave_absent_by_measurement
        and strong_f_wave
        and not validated_residual
        and not strong_F_wave
    )
    borderline_F_wave_pattern = bool(
        flutter_consensus
        and 0.50 <= flutter_wave_confidence < 0.60
        and not strong_f_wave
    )
    indeterminate_reasons: List[str] = []
    if conflicting_morphology:
        indeterminate_reasons.append("conflicting_f_and_F_wave_morphology")
    if near_af_threshold:
        indeterminate_reasons.append("mild_rr_irregularity_with_absent_organized_p")
    if unvalidated_f_wave_pattern:
        indeterminate_reasons.append("unvalidated_f_wave_pattern_with_irregular_rr")
    if borderline_F_wave_pattern:
        indeterminate_reasons.append("borderline_multilead_F_wave_pattern")
    af_afl_indeterminate = bool(indeterminate_reasons)
    probable_flutter = bool(strong_F_wave and not af_afl_indeterminate)
    probable_af = bool(
        classic_af_evidence
        and not probable_flutter
        and not af_afl_indeterminate
    )

    rr_score = float(
        np.clip(((rr_cv or 0.0) - 0.12) / 0.18, 0.0, 1.0)
    )
    absent_p_score = float(np.clip((0.70 - organized_p_ratio) / 0.60, 0.0, 1.0))
    if validated_residual and strong_f_wave:
        # A reproducible broad-band f-wave residual means apparent conducted-P
        # detections are likely QRST/atrial-fragment associations rather than a
        # true stable P sequence.  Reflect that in confidence without discarding
        # the original organized-P measurement from exported evidence.
        absent_p_score = max(absent_p_score, 0.80 * f_wave_confidence)
    af_confidence = float(
        np.clip(
            0.45 * rr_score
            + 0.35 * absent_p_score
            + 0.20 * (f_wave_confidence if validated_residual else 0.5),
            0.0,
            1.0,
        )
    )
    diagnostic_confidence = (
        min(flutter_wave_confidence, 1.0)
        if probable_flutter
        else af_confidence if probable_af
        else min(flutter_wave_confidence, f_wave_confidence)
        if af_afl_indeterminate
        else 0.0
    )
    classification = (
        "atrial_flutter"
        if probable_flutter
        else "atrial_fibrillation"
        if probable_af
        else "af_afl_indeterminate"
        if af_afl_indeterminate
        else "none"
    )

    return {
        "rr_cv": rr_cv,
        "organized_p_ratio": float(organized_p_ratio),
        "pr_dispersion_ms": (
            float(pr_dispersion_ms) if pr_dispersion_ms is not None else None
        ),
        "p_absence_decided_by": (
            "pr_dispersion" if pr_dispersion_decisive else "organized_p_ratio"
        ),
        "probable_af": probable_af,
        "probable_flutter": probable_flutter,
        "af_afl_indeterminate": af_afl_indeterminate,
        "indeterminate_reasons": indeterminate_reasons,
        "atrial_rhythm_classification": classification,
        "diagnostic_confidence": diagnostic_confidence,
        "classic_af_evidence": classic_af_evidence,
        "f_wave_confidence": f_wave_confidence,
        "f_wave_multilead_consensus": f_wave_consensus,
        "F_wave_confidence": flutter_wave_confidence,
        "F_wave_multilead_consensus": flutter_consensus,
        "F_wave_rate_bpm": residual_summary.get("F_wave_rate_bpm"),
        "F_wave_supporting_leads": list(
            residual_summary.get("flutter_supporting_leads") or []
        ),
        "f_wave_supporting_leads": list(
            residual_summary.get("fibrillatory_supporting_leads") or []
        ),
        "validated_qrst_subtraction": validated_residual,
        "flutter_wave_confidence": float(max(0.0, min(1.0, flutter_wave_confidence))),
        "flutter_detection_source": (
            "spectral_and_time_domain"
            if spectral_candidate and cycle_in_flutter_range
            else "spectral" if spectral_candidate
            else "time_domain" if cycle_in_flutter_range
            else "none"
        ),
        "spectral_flutter_bpm": residual_summary.get("spectral_flutter_bpm"),
    }


def _t_aligned_residual(
    segment: np.ndarray,
    template: np.ndarray,
    fs: int,
    r_index: int,
    max_shift_ms: float = 40.0,
) -> np.ndarray:
    """Subtract a smoothly joined, beat-adaptive QRST template.

    Farrell/Xue/Young 2003: subtracting an R-triggered ST-T template leaves a
    residual artefact because T amplitude/duration track the coupling interval.
    The QRS is kept R-aligned.  The ST-T part is shifted by derivative
    correlation, then fitted with a robust per-beat amplitude and offset.  A
    raised-cosine transition joins the QRS and ST-T models, while edge tapers
    prevent window-boundary steps.  This remains deterministic DSP; it does not
    claim to separate a P wave that is phase-locked inside every T template.
    """
    seg = np.asarray(segment, dtype=float)
    tmpl = np.asarray(template, dtype=float)
    n = int(min(seg.size, tmpl.size))
    if n == 0:
        return seg - tmpl
    seg = seg[:n]
    tmpl = tmpl[:n]
    stt = int(r_index) + int(round(0.060 * fs))  # ST-T begins ~60 ms after R
    if fs <= 0 or r_index < 0 or stt >= n - 3:
        return seg - tmpl

    seg_d = np.gradient(seg[stt:])
    tmpl_d = np.gradient(tmpl[stt:])
    max_shift = max(1, int(round(max_shift_ms * float(fs) / 1000.0)))
    best_shift, best_corr = 0, -np.inf
    for shift in range(-max_shift, max_shift + 1):
        if shift >= 0:
            a, b = seg_d[shift:], tmpl_d[: seg_d.size - shift]
        else:
            a, b = seg_d[: seg_d.size + shift], tmpl_d[-shift:]
        if a.size < 3:
            continue
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        corr = float(np.dot(a, b) / (denom + 1e-12)) if denom > 0 else 0.0
        if corr > best_corr:
            best_corr, best_shift = corr, shift

    aligned = tmpl.copy()
    stt_indices = np.arange(stt, n, dtype=int)
    source_indices = stt_indices - int(best_shift)
    valid = (source_indices >= 0) & (source_indices < n)
    aligned[stt_indices[valid]] = tmpl[source_indices[valid]]

    # Robust affine adaptation over ST-T.  Respiratory axis changes commonly
    # appear as a smooth per-lead amplitude modulation; fitting it beat by beat
    # is materially safer than subtracting a fixed median waveform.
    fit_indices = stt_indices[valid]
    alpha = 1.0
    beta = 0.0
    if fit_indices.size >= max(8, int(round(0.040 * fs))):
        x = aligned[fit_indices]
        y = seg[fit_indices]
        design = np.column_stack((x, np.ones_like(x)))
        weights = np.ones_like(x)
        coefficients = np.asarray([1.0, 0.0], dtype=float)
        for _ in range(5):
            root_weight = np.sqrt(weights)
            try:
                coefficients, *_ = np.linalg.lstsq(
                    design * root_weight[:, None],
                    y * root_weight,
                    rcond=None,
                )
            except np.linalg.LinAlgError:
                coefficients = np.asarray([1.0, 0.0], dtype=float)
                break
            fit_residual = y - design @ coefficients
            center = float(np.median(fit_residual))
            sigma = 1.4826 * float(np.median(np.abs(fit_residual - center)))
            if sigma <= 1e-9:
                break
            limit = 1.5 * sigma
            absolute = np.abs(fit_residual - center)
            weights = np.ones_like(absolute)
            outliers = absolute > limit
            weights[outliers] = limit / np.maximum(absolute[outliers], 1e-12)
        alpha = float(np.clip(coefficients[0], 0.50, 1.50))
        beta_limit = max(
            0.02,
            0.25 * float(np.ptp(seg[fit_indices])),
        )
        beta = float(np.clip(coefficients[1], -beta_limit, beta_limit))

    fixed_model = tmpl.copy()
    adaptive_model = alpha * aligned + beta
    model = fixed_model.copy()
    transition = max(2, int(round(0.020 * fs)))
    transition_stop = min(n, stt + transition)
    if transition_stop > stt:
        phase = np.linspace(0.0, 1.0, transition_stop - stt)
        blend = 0.5 - 0.5 * np.cos(np.pi * phase)
        model[stt:transition_stop] = (
            (1.0 - blend) * fixed_model[stt:transition_stop]
            + blend * adaptive_model[stt:transition_stop]
        )
    if transition_stop < n:
        model[transition_stop:] = adaptive_model[transition_stop:]

    # Taper only the subtraction model; the observed segment remains untouched.
    # This prevents hard steps when the residual is written back into a rhythm
    # strip and therefore avoids manufacturing high-frequency P-like edges.
    edge = max(2, int(round(0.020 * fs)))
    window = np.ones(n, dtype=float)
    edge = min(edge, max(1, n // 3))
    phase = np.linspace(0.0, 1.0, edge)
    ramp = 0.5 - 0.5 * np.cos(np.pi * phase)
    window[:edge] = ramp
    window[-edge:] = ramp[::-1]
    return seg - window * model


def _subtraction_high_frequency_ratio(
    segment: np.ndarray,
    residual: np.ndarray,
) -> float:
    """Derivative-energy guard against subtraction-created sharp artefacts."""
    original = np.asarray(segment, dtype=float)
    result = np.asarray(residual, dtype=float)
    n = min(original.size, result.size)
    if n < 5:
        return 1.0
    original_hf = np.diff(original[:n], n=2)
    residual_hf = np.diff(result[:n], n=2)
    original_rms = float(np.sqrt(np.mean(original_hf * original_hf)))
    residual_rms = float(np.sqrt(np.mean(residual_hf * residual_hf)))
    if original_rms <= 1e-9:
        return 0.0 if residual_rms <= 1e-9 else 10.0
    return float(min(10.0, residual_rms / original_rms))


def _template_subtraction_summary(
    segments_by_lead: Dict[str, Sequence[np.ndarray]],
    fs: int,
    r_index: int,
) -> Tuple[float, Dict[str, float], List[np.ndarray], float, float]:
    correlations: List[float] = []
    correlations_by_lead: Dict[str, float] = {}
    residuals: List[np.ndarray] = []
    pre_energy = 0.0
    residual_energy = 0.0
    sample_count = 0
    high_frequency_ratios: List[float] = []

    for lead, segments in segments_by_lead.items():
        if len(segments) < 2:
            continue
        matrix = np.vstack([np.asarray(segment, dtype=float).reshape(1, -1) for segment in segments])
        template = np.median(matrix, axis=0)
        template_centered = template - float(np.mean(template))
        template_norm = float(np.linalg.norm(template_centered))
        lead_correlations: List[float] = []
        for segment in matrix:
            residual = _t_aligned_residual(segment, template, fs, r_index)
            residuals.append(residual)
            high_frequency_ratios.append(
                _subtraction_high_frequency_ratio(segment, residual)
            )
            pre_energy += float(np.dot(segment, segment))
            residual_energy += float(np.dot(residual, residual))
            sample_count += int(segment.size)
            centered = segment - float(np.mean(segment))
            denom = float(np.linalg.norm(centered)) * template_norm
            if denom <= 1e-12:
                continue
            lead_correlations.append(float(np.dot(centered, template_centered) / denom))
        if lead_correlations:
            correlations_by_lead[lead] = float(np.median(lead_correlations))
            correlations.extend(lead_correlations)

    pre_rms = float(np.sqrt(pre_energy / float(sample_count))) if sample_count else 0.0
    residual_rms = float(np.sqrt(residual_energy / float(sample_count))) if sample_count else 0.0
    return (
        float(np.median(correlations)) if correlations else 0.0,
        correlations_by_lead,
        residuals,
        (residual_rms / pre_rms) if pre_rms > 1e-12 else 1.0,
        (
            float(np.median(high_frequency_ratios))
            if high_frequency_ratios
            else 1.0
        ),
    )


def build_qrst_subtracted_residual(
    ecg,
    fs: int,
    r_locs,
    beat_features,
    quality,
    preferred_leads: Sequence[str] = PREFERRED_ATRIAL_LEADS,
) -> Dict[str, Any]:
    """Build a simple QRST-window residual summary for AF/AFL rule inputs.

    This is a baseline-subtracted QRST segment scaffold, not a full validated
    Philips-style QRST subtraction implementation.
    """
    def unavailable(reason: str) -> Dict[str, Any]:
        return {
            "available": False,
            "reason": reason,
            "method": QRST_RESIDUAL_SCAFFOLD_METHOD,
            "validated_qrst_subtraction": False,
            "validation_metrics": {
                "beat_count": 0,
                "template_correlation": 0.0,
                "template_residual_rms_ratio": None,
                "template_high_frequency_ratio": None,
                "usable_lead_count": 0,
                "usable_leads": [],
                "failure_reasons": [reason],
            },
        }

    if ecg is None:
        return unavailable("ecg_not_available")
    if beat_features is None or quality is None:
        return unavailable("missing_qrst_inputs")

    ecg_arr = np.asarray(ecg, dtype=float)
    if fs <= 0:
        return unavailable("invalid_fs")
    if ecg_arr.ndim != 2:
        return unavailable("invalid_ecg_shape")

    beat_rows = list(beat_features or [])
    lead_to_idx = {lead: idx for idx, lead in enumerate(STANDARD_12_LEAD_ORDER)}
    quality_fallback = not quality and not beat_rows
    reliable_leads = [
        lead
        for lead in preferred_leads
        if lead in lead_to_idx
        and lead_to_idx[lead] < ecg_arr.shape[0]
        and (
            getattr(quality.get(lead), "reliable_for_p", False)
            or quality_fallback
        )
    ]
    if not reliable_leads:
        return unavailable("no_reliable_qrst_segments")

    by_beat: Dict[int, List[Any]] = {}
    for bf in beat_rows:
        by_beat.setdefault(int(bf.beat_id), []).append(bf)

    pre_pad = int(round(0.040 * fs))
    post_pad = int(round(0.080 * fs))
    baseline_len = max(1, int(round(0.030 * fs)))
    segments: List[np.ndarray] = []
    template_segments_by_lead: Dict[str, List[np.ndarray]] = {}
    segment_leads = set()
    r_samples = np.asarray(r_locs if r_locs is not None else [], dtype=int)
    r_count = len(r_samples) if r_locs is not None else len(by_beat)

    for beat_id in range(r_count):
        lead_items = {bf.lead: bf for bf in by_beat.get(int(beat_id), [])}
        for lead in reliable_leads:
            bf = lead_items.get(lead)
            if bf is None or bf.qrs.onset is None or bf.t.offset is None:
                continue
            lead_idx = lead_to_idx[lead]
            start = max(0, int(bf.qrs.onset) - pre_pad)
            stop = min(ecg_arr.shape[1], int(bf.t.offset) + post_pad)
            if stop - start < baseline_len * 2:
                continue
            segment = np.asarray(ecg_arr[lead_idx, start:stop], dtype=float)
            baseline = float(np.median(segment[:baseline_len]))
            segments.append(segment - baseline)
            segment_leads.add(lead)

    template_pre = int(round(0.120 * fs))
    template_post = int(round(0.240 * fs))
    template_len = template_pre + template_post
    for lead in reliable_leads:
        lead_idx = lead_to_idx[lead]
        for r_sample in r_samples:
            start = int(r_sample) - template_pre
            stop = int(r_sample) + template_post
            if start < 0 or stop > ecg_arr.shape[1] or stop - start != template_len:
                continue
            segment = np.asarray(ecg_arr[lead_idx, start:stop], dtype=float)
            template_segments_by_lead.setdefault(lead, []).append(segment - float(np.mean(segment)))
            segment_leads.add(lead)

    (
        template_correlation,
        template_correlation_by_lead,
        template_residuals,
        template_residual_rms_ratio,
        template_high_frequency_ratio,
    ) = _template_subtraction_summary(
        template_segments_by_lead,
        fs,
        template_pre,
    )
    template_validated = bool(
        template_correlation >= 0.85
        and len(r_samples) >= 4
        and template_residuals
        and template_residual_rms_ratio <= QRST_TEMPLATE_RESIDUAL_RMS_RATIO_MAX
        and template_high_frequency_ratio
        <= QRST_TEMPLATE_HIGH_FREQUENCY_RATIO_MAX
    )
    validation_failure_reasons: List[str] = []
    if len(r_samples) < 4:
        validation_failure_reasons.append("insufficient_template_beats")
    if template_correlation < 0.85:
        validation_failure_reasons.append("template_correlation_below_threshold")
    if not template_residuals:
        validation_failure_reasons.append("template_residuals_unavailable")
    if template_residual_rms_ratio > QRST_TEMPLATE_RESIDUAL_RMS_RATIO_MAX:
        validation_failure_reasons.append("template_residual_ratio_above_threshold")
    if (
        template_high_frequency_ratio
        > QRST_TEMPLATE_HIGH_FREQUENCY_RATIO_MAX
    ):
        validation_failure_reasons.append(
            "template_high_frequency_artifact_ratio_above_threshold"
        )
    residual_segments = template_residuals if template_residuals else segments
    if not residual_segments:
        return unavailable("no_reliable_qrst_segments")

    residual = np.concatenate(residual_segments)
    summary = _summarize_atrial_residual(residual, fs)
    per_lead_activity: Dict[str, Dict[str, Any]] = {}
    for lead in reliable_leads:
        lead_idx = lead_to_idx.get(lead)
        if lead_idx is None or lead_idx >= ecg_arr.shape[0]:
            continue
        lead_residual, lead_ok = _atrial_lead_residual(
            ecg_arr, lead_idx, fs, r_samples
        )
        if not lead_ok or lead_residual is None:
            continue
        per_lead_activity[lead] = _summarize_atrial_residual(lead_residual, fs)
    multilead = _aggregate_multilead_atrial_activity(per_lead_activity)
    summary["per_lead_atrial_activity"] = per_lead_activity
    summary.update(multilead)

    # Flutter spectrometry runs on its own continuous signal (see
    # `flutter_analysis_signal`) rather than on the shared residual, whose
    # per-beat concatenation and QRS zeroing destroy exactly the continuity a
    # spectrum needs. Narrowness is averaged over II and V1, the leads where
    # the sawtooth is most visible and the pair measured to separate best.
    narrowness_by_lead: Dict[str, float] = {}
    flutter_bpm_by_lead: Dict[str, float] = {}
    for lead in ("II", "V1"):
        lead_idx = lead_to_idx.get(lead)
        if lead_idx is None or lead_idx >= ecg_arr.shape[0]:
            continue
        analysis = flutter_analysis_signal(ecg_arr[lead_idx], fs, r_samples)
        if analysis is None:
            continue
        measures = _spectral_flutter_analysis(analysis, fs)
        narrowness = measures.get("spectral_peak_narrowness")
        if narrowness is None:
            continue
        narrowness_by_lead[lead] = float(narrowness)
        bpm = measures.get("spectral_flutter_bpm")
        if bpm is not None:
            flutter_bpm_by_lead[lead] = float(bpm)
    if narrowness_by_lead:
        mean_narrowness = float(np.mean(list(narrowness_by_lead.values())))
        summary["flutter_peak_narrowness_by_lead"] = narrowness_by_lead
        summary["flutter_peak_narrowness"] = mean_narrowness
        summary["flutter_narrow_peak_consensus"] = bool(
            mean_narrowness >= FLUTTER_SPECTRAL_NARROWNESS_MIN
        )
        if flutter_bpm_by_lead:
            summary["flutter_narrow_peak_bpm"] = float(
                np.median(list(flutter_bpm_by_lead.values()))
            )
    else:
        summary["flutter_peak_narrowness"] = None
        summary["flutter_narrow_peak_consensus"] = False

    # Second route, for flutter whose F waves are phase-locked to the QRS and
    # therefore cancelled by the subtraction above. Reads the recorded signal
    # directly and pays for that with a much stricter narrowness plus a
    # conduction-ratio sanity check; the caller adds the P-absence condition,
    # which is the third of the three measured constraints.
    raw_narrowness_by_lead: Dict[str, float] = {}
    raw_bpm_by_lead: Dict[str, float] = {}
    for lead in ("II", "V1"):
        lead_idx = lead_to_idx.get(lead)
        if lead_idx is None or lead_idx >= ecg_arr.shape[0]:
            continue
        measures = _spectral_flutter_analysis(ecg_arr[lead_idx], fs)
        narrowness = measures.get("spectral_peak_narrowness")
        if narrowness is None:
            continue
        raw_narrowness_by_lead[lead] = float(narrowness)
        bpm = measures.get("spectral_flutter_bpm")
        if bpm is not None:
            raw_bpm_by_lead[lead] = float(bpm)

    raw_ratio: Optional[float] = None
    if raw_bpm_by_lead and r_samples.size >= 2:
        rr_samples = np.diff(np.asarray(r_samples, dtype=float))
        median_rr = float(np.median(rr_samples)) if rr_samples.size else 0.0
        if median_rr > 0:
            ventricular_bpm = 60.0 * float(fs) / median_rr
            if ventricular_bpm > 0:
                raw_ratio = float(
                    np.median(list(raw_bpm_by_lead.values())) / ventricular_bpm
                )
    if raw_narrowness_by_lead:
        raw_mean = float(np.mean(list(raw_narrowness_by_lead.values())))
        summary["flutter_raw_peak_narrowness"] = raw_mean
        summary["flutter_raw_conduction_ratio"] = raw_ratio
        ratio_plausible = bool(
            raw_ratio is not None
            and abs(raw_ratio - round(raw_ratio))
            <= _FLUTTER_RAW_RATIO_INTEGER_TOLERANCE
            and round(raw_ratio) <= _FLUTTER_RAW_MAX_CONDUCTION_RATIO
            and round(raw_ratio) >= 1
        )
        summary["flutter_raw_narrow_peak"] = bool(
            raw_mean >= _FLUTTER_RAW_NARROWNESS_MIN and ratio_plausible
        )
    else:
        summary["flutter_raw_peak_narrowness"] = None
        summary["flutter_raw_narrow_peak"] = False
    summary["method"] = QRST_TEMPLATE_SUBTRACTION_METHOD if template_validated else QRST_RESIDUAL_SCAFFOLD_METHOD
    summary["validated_qrst_subtraction"] = template_validated
    # NOTE: the F_wave_* shape descriptors produced by
    # `_flutter_wave_morphology` are exported as evidence but deliberately do
    # NOT gate this flag. Measured on WFDBRecords/01/019 they were not
    # discriminative -- sinus records scored 0.62-0.65 against 0.68 for a
    # confirmed flutter record -- so admitting them here would widen flutter
    # calls without evidence. The binding constraint is the sensitivity of the
    # spectral detector (`flutter_multilead_consensus` was false on 4 of 5
    # labelled flutter records), not the absence of a morphology term.
    # See docs/诊断层待修改清单.md item A.2.
    summary["F_wave_morphology_validated"] = bool(
        template_validated and multilead.get("flutter_multilead_consensus")
    )
    summary["f_wave_morphology_validated"] = bool(
        template_validated and multilead.get("f_wave_multilead_consensus")
    )
    summary["template_correlation"] = template_correlation
    summary["template_correlation_by_lead"] = template_correlation_by_lead
    summary["template_residual_rms_ratio"] = template_residual_rms_ratio
    summary["template_high_frequency_ratio"] = template_high_frequency_ratio
    summary["template_adaptation_method"] = (
        "r_aligned_qrs_shifted_robust_affine_stt_cosine_taper"
    )
    summary["segment_count"] = len(residual_segments)
    summary["source_leads"] = sorted(segment_leads)
    summary["validation_metrics"] = {
        "beat_count": int(len(r_samples)),
        "template_correlation": template_correlation,
        "template_residual_rms_ratio": template_residual_rms_ratio,
        "template_high_frequency_ratio": template_high_frequency_ratio,
        "template_adaptation_method": (
            "r_aligned_qrs_shifted_robust_affine_stt_cosine_taper"
        ),
        "usable_lead_count": int(len(segment_leads)),
        "usable_leads": sorted(segment_leads),
        "failure_reasons": [] if template_validated else validation_failure_reasons,
    }
    if not template_validated and "reason" not in summary:
        summary["reason"] = validation_failure_reasons[0] if validation_failure_reasons else "qrst_subtraction_not_validated"
    return summary


def _qrst_subtracted_rhythm(
    sig_lp: np.ndarray, fs: int, r_locs: np.ndarray
) -> Tuple[np.ndarray, bool]:
    """Median-QRST template subtraction along a whole single-lead rhythm strip.

    Builds an R-centred median QRST template, subtracts it (ST-T time-aligned per
    beat, see _t_aligned_residual) at each R, and zeros the QRS core so only
    atrial activity remains (Reddy 1992: QRS-region residual set to zero).
    """
    residual = np.asarray(sig_lp, dtype=float).copy()
    r = np.asarray(r_locs, dtype=int)
    if fs <= 0 or r.size < 3:
        return residual, False
    rr_ms = np.diff(r).astype(float) * 1000.0 / float(fs)
    if (
        rr_ms.size
        and float(np.median(rr_ms))
        < QRST_TEMPLATE_MIN_RR_MS_FOR_P_PROTECTION
    ):
        # The fixed QRST support would intrude into the next atrial search
        # region.  In a phase-locked high-rate rhythm a median R-template can
        # contain and erase the very P-on-T wave being sought; this problem is
        # not identifiable from the template alone.
        return residual, False
    pre = int(round(0.120 * fs))
    post = int(round(0.240 * fs))
    windows = [(int(rr) - pre, int(rr) + post) for rr in r]
    segs = [sig_lp[a:b] for a, b in windows if a >= 0 and b <= len(sig_lp)]
    if len(segs) < 3:
        return residual, False
    matrix = np.vstack(segs)
    template = np.median(matrix, axis=0)
    template_centered = template - float(np.mean(template))
    template_norm = float(np.linalg.norm(template_centered))
    correlations: List[float] = []
    candidate_residuals: List[np.ndarray] = []
    hf_ratios: List[float] = []
    pre_energy = 0.0
    residual_energy = 0.0
    sample_count = 0
    for segment in matrix:
        candidate = _t_aligned_residual(segment, template, fs, pre)
        candidate_residuals.append(candidate)
        hf_ratios.append(_subtraction_high_frequency_ratio(segment, candidate))
        centered = segment - float(np.mean(segment))
        denom = float(np.linalg.norm(centered)) * template_norm
        if denom > 1e-12:
            correlations.append(
                float(np.dot(centered, template_centered) / denom)
            )
        # Score the subtraction where the ventricular complex is, not over the
        # whole R-120..R+240 ms window. Outside the QRS core the residual is
        # *supposed* to still hold atrial activity -- that is the signal being
        # extracted -- so judging the fit by total leftover energy amounts to
        # requiring that no atrial activity be present. Measured whole-window
        # ratios against the 0.25 limit: 0.46 and 0.28 on labelled flutter,
        # 0.72 on AF, 0.55 even on sinus. The gate therefore essentially never
        # passed, and both consumers drop a lead whose subtraction is not
        # validated, so multilead flutter consensus was false on all 34
        # flutter records and `_composite_raw_p_detection` never ran at all.
        core_lo = max(0, pre - int(round(0.040 * fs)))
        core_hi = min(int(segment.size), pre + int(round(0.080 * fs)))
        if core_hi > core_lo:
            core_segment = segment[core_lo:core_hi]
            core_candidate = candidate[core_lo:core_hi]
            pre_energy += float(np.dot(core_segment, core_segment))
            residual_energy += float(np.dot(core_candidate, core_candidate))
            sample_count += int(core_segment.size)
    template_correlation = (
        float(np.median(correlations)) if correlations else 0.0
    )
    residual_ratio = (
        float(np.sqrt(residual_energy / pre_energy))
        if pre_energy > 1e-12 and sample_count
        else 1.0
    )
    high_frequency_ratio = (
        float(np.median(hf_ratios)) if hf_ratios else 1.0
    )
    if (
        template_correlation < 0.85
        or residual_ratio > QRST_TEMPLATE_RESIDUAL_RMS_RATIO_MAX
        or high_frequency_ratio > QRST_TEMPLATE_HIGH_FREQUENCY_RATIO_MAX
    ):
        return residual, False

    candidate_index = 0
    for a, b in windows:
        if a >= 0 and b <= len(sig_lp):
            residual[a:b] = candidate_residuals[candidate_index]
            candidate_index += 1
    q_pre = int(round(0.040 * fs))
    q_post = int(round(0.080 * fs))
    for rr in r:
        a = max(0, int(rr) - q_pre)
        b = min(len(residual), int(rr) + q_post)
        residual[a:b] = 0.0
    return residual, True


def _atrial_detection_function(residual: np.ndarray) -> np.ndarray:
    """Reddy/Farrell P detection function: |first derivative| + |second derivative|."""
    r = np.asarray(residual, dtype=float)
    if r.size < 3:
        return np.zeros_like(r)
    d1 = np.abs(np.gradient(r))
    d2 = np.abs(np.gradient(d1))
    return d1 + d2


def _atrial_normalized_snippet(
    signal: np.ndarray,
    center: int,
    half_width: int,
) -> Optional[np.ndarray]:
    lo = int(center) - int(half_width)
    hi = int(center) + int(half_width) + 1
    if lo < 0 or hi > len(signal) or hi <= lo:
        return None
    snippet = np.asarray(signal[lo:hi], dtype=float)
    snippet = snippet - float(np.median(snippet))
    scale = float(np.max(np.abs(snippet)))
    if scale <= 1e-9:
        return None
    return snippet / scale


def _atrial_template_similarity(
    snippet: Optional[np.ndarray],
    template: Optional[np.ndarray],
) -> float:
    if snippet is None or template is None:
        return 0.5
    if len(snippet) != len(template) or len(snippet) < 3:
        return 0.5
    a = np.asarray(snippet, dtype=float) - float(np.mean(snippet))
    b = np.asarray(template, dtype=float) - float(np.mean(template))
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-9:
        return 0.5
    return float(np.clip(abs(float(np.dot(a, b) / denom)), 0.0, 1.0))


def flutter_analysis_signal(
    sig: np.ndarray, fs: int, r_locs: np.ndarray
) -> Optional[np.ndarray]:
    """Continuous atrial signal built specifically for flutter spectrometry.

    Deliberately separate from `_atrial_lead_residual`, which serves the P/beat
    consumers and for their sake validates the subtraction, zeroes the QRS core
    and drops the lead when the template does not fit. Every one of those is
    wrong for a *spectral* view of flutter:

    * its validation demands the residual keep under 25% of the segment energy,
      which for an atrial arrhythmia is demanding that the atrial activity it
      is supposed to expose not be there (measured: 0.46 and 0.28 on labelled
      flutter, 0.72 on AF, 0.55 even on sinus -- the gate essentially never
      passes, so no lead ever reached the multilead flutter consensus);
    * zeroing the QRS core punches a rectangular hole into every beat, and the
      resulting comb moved the dominant peak of a 212 bpm flutter onto its
      425 bpm harmonic and cut measured narrowness from 0.845 to 0.218.

    Flutter is one continuous wavefront running *through* the QRST, so what it
    needs is simply the ventricular complex removed with the record left
    continuous: subtract a median QRST template, amplitude-fitted per beat.
    Verified against 34 labelled flutter and 120 AF/sinus records -- this
    signal separates them at AUC 0.83, while concatenating beat segments (what
    the shared residual does) scores below 0.5, i.e. inverted.
    """
    values = np.asarray(sig, dtype=float)
    r = np.asarray(r_locs, dtype=int)
    if fs <= 0 or r.size < 4 or values.size < fs:
        return None
    pre, post = int(round(0.25 * fs)), int(round(0.45 * fs))
    segments = [
        values[int(x) - pre : int(x) + post]
        for x in r
        if int(x) - pre >= 0 and int(x) + post <= values.size
    ]
    if len(segments) < 4:
        return None
    template = np.median(np.vstack(segments), axis=0)
    template_energy = float(np.dot(template, template))
    if template_energy <= 1e-18:
        return None
    out = values.copy()
    for x in r:
        lo, hi = int(x) - pre, int(x) + post
        if lo < 0 or hi > values.size:
            continue
        beat = values[lo:hi]
        # Per-beat gain so amplitude drift does not leave a scaled copy of the
        # QRST behind, which would dominate the spectrum.
        gain = float(np.dot(beat, template) / template_energy)
        out[lo:hi] = beat - gain * template
    return out


def _atrial_lead_residual(
    ecg_arr: np.ndarray, lead_idx: int, fs: int, r_locs: np.ndarray
) -> Tuple[Optional[np.ndarray], bool]:
    if lead_idx >= ecg_arr.shape[0]:
        return None, False
    sig = np.asarray(ecg_arr[lead_idx], dtype=float)
    sig_lp = lowpass_filter(sig, fs, cutoff_hz=ATRIAL_RESIDUAL_LP_HZ)
    residual, ok = _qrst_subtracted_rhythm(sig_lp, fs, r_locs)
    return residual, ok


def _atrial_lead_snr(residual: np.ndarray, fs: int) -> float:
    """SNR-like score for a QRST-subtracted lead: detection-function peakiness."""
    f = _atrial_detection_function(residual)
    nz = f[f > 0]
    if nz.size < int(round(0.5 * fs)):
        return 0.0
    floor = float(np.median(nz)) + 1e-9
    peak = float(np.percentile(nz, 95))
    return peak / floor


def select_atrial_leads(
    ecg_arr: np.ndarray,
    fs: int,
    r_locs: np.ndarray,
    quality,
    limb_leads: Sequence[str] = ATRIAL_LIMB_LEADS,
    precordial_leads: Sequence[str] = ATRIAL_PRECORDIAL_LEADS,
) -> Dict[str, Any]:
    """Score every candidate lead and pick the best limb + best precordial lead.

    Farrell 2003: rather than fixing II and V1, all leads are scored (here by the
    QRST-subtracted detection-function SNR, down-weighted when a lead is not
    reliable for P) and the best limb (I/II) and best precordial (V1-V6) chosen.
    """
    lead_to_idx = {lead: idx for idx, lead in enumerate(STANDARD_12_LEAD_ORDER)}
    quality_fallback = not quality

    def best(group: Sequence[str]) -> Tuple[Optional[str], float, Optional[np.ndarray]]:
        best_lead, best_score, best_res = None, -1.0, None
        for lead in group:
            idx = lead_to_idx.get(lead)
            if idx is None or idx >= ecg_arr.shape[0]:
                continue
            reliable = quality_fallback or getattr(quality.get(lead), "reliable_for_p", False)
            residual, ok = _atrial_lead_residual(ecg_arr, idx, fs, r_locs)
            if not ok or residual is None:
                continue
            score = _atrial_lead_snr(residual, fs) * (1.0 if reliable else 0.35)
            if score > best_score:
                best_lead, best_score, best_res = lead, score, residual
        return best_lead, best_score, best_res

    limb_lead, limb_score, limb_res = best(limb_leads)
    prec_lead, prec_score, prec_res = best(precordial_leads)
    return {
        "limb_lead": limb_lead,
        "limb_score": float(limb_score) if limb_lead else None,
        "limb_residual": limb_res,
        "precordial_lead": prec_lead,
        "precordial_score": float(prec_score) if prec_lead else None,
        "precordial_residual": prec_res,
    }


def _composite_raw_p_detection(
    ecg: np.ndarray,
    fs: int,
    r_locs: np.ndarray,
    qrs_offsets: np.ndarray,
    quality,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Reddy/Farrell composite raw-rhythm P / flutter / fibrillatory detection.

    Selects the best limb and precordial leads, forms a QRST-subtracted composite,
    and detects atrial waves in each inter-QRS region using the derivative
    detection function with a region-relative threshold.
    """
    meta: Dict[str, Any] = {"method": COMPOSITE_P_DETECTION_METHOD, "available": False}
    ecg_arr = np.asarray(ecg, dtype=float)
    r = np.asarray(r_locs, dtype=int)
    if ecg_arr.ndim != 2 or fs <= 0 or r.size < 3:
        return [], meta

    selection = select_atrial_leads(ecg_arr, fs, r, quality)
    limb_res = selection["limb_residual"]
    prec_res = selection["precordial_residual"]
    residuals = [res for res in (limb_res, prec_res) if res is not None]
    if not residuals:
        return [], meta

    length = min(res.size for res in residuals)
    detection = np.zeros(length, dtype=float)
    composite = np.zeros(length, dtype=float)
    for res in residuals:
        seg = res[:length]
        std = float(np.std(seg)) or 1.0
        normed = (seg - float(np.mean(seg))) / std
        composite += normed
        detection += _atrial_detection_function(normed)

    qrs_off = np.asarray(qrs_offsets, dtype=int)
    guard = int(round(0.040 * fs))
    min_window = int(round(0.080 * fs))
    peak_distance = max(1, int(round(0.070 * fs)))
    candidates: List[Dict[str, Any]] = []
    source_leads = sorted(
        lead for lead in (selection["limb_lead"], selection["precordial_lead"]) if lead
    )

    def measure_candidate(
        detection_peak: int,
        height: float,
        edge_threshold: float,
        start: int,
        stop: int,
    ) -> Dict[str, Any]:
        max_half_width = max(2, int(round(0.120 * fs)))
        lo = int(detection_peak)
        while (
            lo > start
            and detection[lo] > edge_threshold
            and int(detection_peak) - lo < max_half_width
        ):
            lo -= 1
        hi = int(detection_peak)
        while (
            hi < stop - 1
            and detection[hi] > edge_threshold
            and hi - int(detection_peak) < max_half_width
        ):
            hi += 1
        min_width = max(2, int(round(0.020 * fs)))
        if hi - lo < min_width:
            lo = max(start, int(detection_peak) - min_width)
            hi = min(stop - 1, int(detection_peak) + min_width)

        wave = composite[lo: hi + 1]
        if wave.size:
            peak = lo + int(np.argmax(np.abs(wave - float(np.median(wave)))))
        else:
            peak = int(detection_peak)
        if peak <= lo:
            lo = max(start, int(peak) - min_width)
        if peak >= hi:
            hi = min(stop - 1, int(peak) + min_width)

        best_amp = 0.0
        best_area = 0.0
        best_signed_area = 0.0
        trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        for residual in residuals:
            lead_segment = np.asarray(residual[lo: hi + 1], dtype=float)
            if lead_segment.size == 0:
                continue
            baseline = float(np.median(lead_segment))
            centered = lead_segment - baseline
            peak_local = int(np.clip(peak - lo, 0, lead_segment.size - 1))
            amp = float(centered[peak_local])
            area = float(trapz(np.abs(centered)) * 1000.0 / float(fs))
            signed_area = float(trapz(centered) * 1000.0 / float(fs))
            if abs(amp) > abs(best_amp):
                best_amp = amp
                best_area = area
                best_signed_area = signed_area

        # The detection function is a derivative, so a sharp few-microvolt
        # residual ripple scores as highly as a real P wave and saturates this
        # ratio at 1.0. Derate by how plausible the measured amplitude is for an
        # atrial deflection, otherwise every QRST-subtraction artifact is
        # published with full confidence.
        detection_confidence = float(min(1.0, height / max(edge_threshold * 2.0, 1e-9)))
        amplitude_plausibility = float(
            min(1.0, abs(best_amp) / _ATRIAL_EVENT_REFERENCE_AMPLITUDE_MV)
        )
        return {
            "sample": int(peak),
            "lead": "composite",
            "confidence": float(detection_confidence * amplitude_plausibility),
            "onset_sample": int(lo),
            "offset_sample": int(hi),
            "amplitude_mv": float(best_amp),
            "area_mv_ms": float(best_area),
            "signed_area_mv_ms": float(best_signed_area),
            "detection_function_peak": int(detection_peak),
        }

    def scan_window(start: int, stop: int, threshold_factor: float) -> int:
        if stop - start < min_window:
            return 0
        region = detection[start:stop]
        if region.size < 3 or float(np.max(region)) <= 0.0:
            return 0
        # Region-relative threshold from the local maximum (12SL inter-QRS rule),
        # floored by the robust noise level so flat regions do not spawn peaks.
        region_max = float(np.max(region))
        med = float(np.median(region))
        noise = med + 1.4826 * float(np.median(np.abs(region - med)))
        threshold = max(threshold_factor * region_max, 3.0 * noise)
        peaks, props = find_peaks(region, height=threshold, distance=peak_distance)
        max_events = max(1, int((stop - start) / max(1, int(round(0.070 * fs)))))
        order = np.argsort(props.get("peak_heights", np.zeros(len(peaks))))[::-1][:max_events]
        for k in order:
            peak_idx = int(peaks[int(k)])
            height = float(props["peak_heights"][int(k)])
            edge_threshold = max(1.5 * noise, 0.18 * height)
            candidates.append(
                measure_candidate(
                    detection_peak=int(start + peak_idx),
                    height=height,
                    edge_threshold=edge_threshold,
                    start=start,
                    stop=stop,
                )
            )
        return len(order)

    # First pass: standard region-relative threshold in every inter-QRS gap.
    for beat_idx in range(r.size - 1):
        prev_off = int(qrs_off[beat_idx]) if beat_idx < qrs_off.size else int(r[beat_idx])
        start = max(0, prev_off + guard)
        stop = min(length, int(r[beat_idx + 1]) - guard)
        scan_window(start, stop, 0.45)

    # Contextual second pass (Reddy 1994): where the P sequence has a gap of about
    # two modal PP intervals, re-search that gap at a lower threshold to recover a
    # P wave that the first pass missed (blocked/low-amplitude atrial activity).
    second_pass_recovered = 0
    first_pass_samples = sorted(int(c["sample"]) for c in candidates)
    if len(first_pass_samples) >= 3:
        pp = np.diff(first_pass_samples)
        modal_pp = float(np.median(pp))
        if modal_pp > 0:
            for left, right in zip(first_pass_samples[:-1], first_pass_samples[1:]):
                if (right - left) >= 1.6 * modal_pp:
                    g_start = max(0, left + int(round(0.5 * modal_pp)) - int(round(0.080 * fs)))
                    g_stop = min(length, right - int(round(0.5 * modal_pp)) + int(round(0.080 * fs)))
                    second_pass_recovered += scan_window(g_start, g_stop, 0.25)

    merged = _cluster_atrial_candidates(candidates, merge_samples=max(12, int(0.040 * fs)))
    snippets: List[Optional[np.ndarray]] = []
    half_width = max(2, int(round(0.080 * fs)))
    for event in merged:
        snippets.append(_atrial_normalized_snippet(composite, int(event["sample"]), half_width))
    valid_snippets = [snippet for snippet in snippets if snippet is not None]
    template = (
        np.median(np.vstack(valid_snippets), axis=0)
        if len(valid_snippets) >= 3
        else None
    )
    sorted_samples = sorted(int(event["sample"]) for event in merged)
    pp_by_sample: Dict[int, Optional[float]] = {sample: None for sample in sorted_samples}
    for prev, current in zip(sorted_samples, sorted_samples[1:]):
        pp_by_sample[current] = (int(current) - int(prev)) * 1000.0 / float(fs)
    events = [
        {
            "p_event_id": int(event["event_id"]),
            "sample": int(event["sample"]),
            "time_ms": float(event["sample"]) * 1000.0 / fs,
            "onset_sample": int(event["onset_sample"]) if event.get("onset_sample") is not None else None,
            "offset_sample": int(event["offset_sample"]) if event.get("offset_sample") is not None else None,
            "onset_ms": (
                float(event["onset_sample"]) * 1000.0 / fs
                if event.get("onset_sample") is not None else None
            ),
            "offset_ms": (
                float(event["offset_sample"]) * 1000.0 / fs
                if event.get("offset_sample") is not None else None
            ),
            "duration_ms": (
                (int(event["offset_sample"]) - int(event["onset_sample"])) * 1000.0 / fs
                if event.get("onset_sample") is not None
                and event.get("offset_sample") is not None
                else None
            ),
            "amplitude_mv": event.get("amplitude_mv"),
            "area_mv_ms": event.get("area_mv_ms"),
            "signed_area_mv_ms": event.get("signed_area_mv_ms"),
            "template_similarity": _atrial_template_similarity(
                snippets[idx] if idx < len(snippets) else None,
                template,
            ),
            "pp_ms": pp_by_sample.get(int(event["sample"])),
            "detection_function_peak": event.get("detection_function_peak"),
            "confidence": float(event["confidence"]),
            "associated_qrs_beat_id": None,
            "association_type": "blocked",
            "pr_ms": None,
            "source_leads": source_leads or list(event["source_leads"]),
            "detection_method": COMPOSITE_P_DETECTION_METHOD,
        }
        for idx, event in enumerate(merged)
    ]
    meta.update(
        {
            "available": True,
            "limb_lead": selection["limb_lead"],
            "precordial_lead": selection["precordial_lead"],
            "limb_score": selection["limb_score"],
            "precordial_score": selection["precordial_score"],
            "event_count": len(events),
            "second_pass_recovered": int(second_pass_recovered),
        }
    )
    return events, meta


def contextual_atrial_analysis(
    events: List[Dict[str, Any]], r_locs: np.ndarray, fs: int
) -> Dict[str, Any]:
    """Reddy 1994 contextual rhythm analysis over detected atrial events.

    Summarises PP-interval regularity, PR-interval consistency for conducted
    beats, the count of blocked P waves and of QRS complexes with no preceding P,
    which together let downstream rhythm logic weigh whether atrial activity is
    organised (sinus/flutter) or disorganised (AF) and whether AV block is present.
    """
    r = np.asarray(r_locs, dtype=int)
    summary: Dict[str, Any] = {
        "atrial_event_count": len(events),
        "pp_median_ms": None,
        "pp_cv": None,
        "pr_median_ms": None,
        "pr_consistency": None,
        "blocked_p_count": 0,
        "conducted_p_count": 0,
        "qrs_without_p_count": 0,
        "regular_atrial_rhythm": False,
    }
    if fs <= 0 or not events:
        return summary

    samples = sorted(int(ev["sample"]) for ev in events)
    if len(samples) >= 2:
        pp = np.diff(samples) * 1000.0 / float(fs)
        pp = pp[pp > 0]
        if pp.size:
            pp_median = float(np.median(pp))
            summary["pp_median_ms"] = pp_median
            summary["pp_cv"] = (
                float(np.std(pp) / pp_median) if pp_median > 0 else None
            )

    pr_values = [
        float(ev["pr_ms"])
        for ev in events
        if ev.get("association_type") == "conducted" and ev.get("pr_ms") is not None
    ]
    summary["conducted_p_count"] = len(pr_values)
    summary["blocked_p_count"] = sum(
        1 for ev in events if ev.get("association_type") == "blocked"
    )
    if pr_values:
        pr_median = float(np.median(pr_values))
        summary["pr_median_ms"] = pr_median
        # PR consistency in [0, 1]: 1.0 = identical PR across conducted beats.
        pr_spread = float(np.std(pr_values))
        summary["pr_consistency"] = float(max(0.0, 1.0 - pr_spread / (pr_median + 1e-6)))

    # QRS complexes with no atrial event in a plausible preceding PR window.
    if r.size:
        conducted_beats = {
            int(ev["associated_qrs_beat_id"])
            for ev in events
            if ev.get("association_type") == "conducted"
            and ev.get("associated_qrs_beat_id") is not None
        }
        summary["qrs_without_p_count"] = int(
            sum(1 for beat_id in range(r.size) if beat_id not in conducted_beats)
        )

    pp_cv = summary["pp_cv"]
    summary["regular_atrial_rhythm"] = bool(
        pp_cv is not None and pp_cv < 0.15 and len(samples) >= 3
    )
    return summary


def extract_atrial_events(
    beat_features,
    quality,
    r_locs: np.ndarray,
    fs: int,
    preferred_leads: Sequence[str] = PREFERRED_ATRIAL_LEADS,
    ecg: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    qrs_onsets: List[int] = []
    qrs_offsets: List[int] = []
    per_beat_first_qrs: Dict[int, Tuple[int, int]] = {}

    for bf in beat_features:
        if (
            bf.beat_id not in per_beat_first_qrs
            and bf.qrs.onset is not None
            and bf.qrs.offset is not None
        ):
            per_beat_first_qrs[int(bf.beat_id)] = (int(bf.qrs.onset), int(bf.qrs.offset))
        if (
            bf.lead in preferred_leads
            and bf.p.peak is not None
            and bf.p_confidence >= 0.30
            and getattr(quality.get(bf.lead), "reliable_for_p", False)
        ):
            candidates.append(
                {
                    "sample": int(bf.p.peak),
                    "lead": bf.lead,
                    "confidence": float(bf.p_confidence),
                }
            )

    for beat_id in range(len(r_locs)):
        qrs_on, qrs_off = per_beat_first_qrs.get(
            beat_id,
            (int(r_locs[beat_id]), int(r_locs[beat_id])),
        )
        qrs_onsets.append(qrs_on)
        qrs_offsets.append(qrs_off)

    qrs_onsets_arr = np.asarray(qrs_onsets, dtype=int)
    qrs_offsets_arr = np.asarray(qrs_offsets, dtype=int)

    t_windows = _beat_t_windows(
        beat_features, fs, len(r_locs), qrs_onsets=qrs_onsets_arr
    )

    merged = _cluster_atrial_candidates(candidates, merge_samples=max(12, int(0.035 * fs)))
    out: List[Dict[str, Any]] = []
    for event in merged:
        beat_id, assoc, pr_ms = _associate_event_to_qrs(
            event["sample"],
            qrs_onsets_arr,
            qrs_offsets_arr,
            fs,
            t_windows=t_windows,
        )
        if assoc == _T_WAVE_ARTIFACT:
            continue
        out.append(
            {
                "p_event_id": int(event["event_id"]),
                "sample": int(event["sample"]),
                "time_ms": float(event["sample"]) * 1000.0 / fs,
                "confidence": float(event["confidence"]),
                "associated_qrs_beat_id": beat_id,
                "association_type": assoc,
                "pr_ms": pr_ms,
                "source_leads": list(event["source_leads"]),
            }
        )
    interbeat_merge_samples = max(12, int(0.040 * fs))
    if ecg is not None:
        # Prefer the Reddy/Farrell composite QRST-residual detector (best limb +
        # precordial lead, derivative detection function); fall back to the crude
        # per-lead raw scan when the composite path is unavailable (<3 QRS, etc.).
        composite_events, composite_meta = _composite_raw_p_detection(
            np.asarray(ecg, dtype=float),
            fs,
            r_locs,
            qrs_offsets_arr,
            quality,
        )
        if composite_meta.get("available") and composite_events:
            kept: List[Dict[str, Any]] = []
            for event in composite_events:
                beat_id, assoc, pr_ms = _associate_event_to_qrs(
                    event["sample"],
                    qrs_onsets_arr,
                    qrs_offsets_arr,
                    fs,
                    t_windows=t_windows,
                )
                if assoc == _T_WAVE_ARTIFACT:
                    continue
                if _below_atrial_event_amplitude_floor(event):
                    continue
                event["associated_qrs_beat_id"] = beat_id
                event["association_type"] = assoc
                event["pr_ms"] = pr_ms
                kept.append(event)
            out.extend(kept)
        else:
            # This fallback labels every hit `blocked` without consulting
            # `_associate_event_to_qrs`, so the T-wave gate has to be applied
            # here as well or the same artifacts reappear as blocked P waves.
            out.extend(
                event
                for event in _scan_interbeat_atrial_candidates(
                    ecg,
                    fs=fs,
                    r_locs=r_locs,
                    qrs_offsets=qrs_offsets_arr,
                    quality=quality,
                    preferred_leads=preferred_leads,
                )
                if not _event_in_any_t_wave(event["sample"], t_windows)
            )
    out = _deduplicate_atrial_events(out, merge_samples=interbeat_merge_samples)
    out.sort(key=lambda event: (float(event["time_ms"]), int(event["sample"])))
    for event_id, event in enumerate(out):
        event["p_event_id"] = int(event_id)
    return out


def compute_organized_p_ratio(
    atrial_events: Sequence[Dict[str, Any]], beat_count: int
) -> float:
    """Fraction of beats in the dominant conducted-P/PR cluster.

    `atrial_events` can carry more than one qualifying event per beat:
    independent per-lead detections and the whole-signal composite scan are
    not fully deduplicated against each other. Counting events directly can
    therefore exceed `beat_count` and push the ratio above 1.0. This is a
    per-beat criterion. Retrograde events do not establish an organized
    antegrade atrial rhythm, and random candidate P waves in AF can appear on
    nearly every beat. Count only conducted events whose PR interval belongs
    to the dominant +/-30 ms cluster.
    """
    if beat_count <= 0:
        return 0.0
    candidates = [
        (int(event["associated_qrs_beat_id"]), float(event["pr_ms"]))
        for event in atrial_events
        if event.get("association_type") == "conducted"
        and float(event.get("confidence") or 0.0) >= 0.30
        and event.get("associated_qrs_beat_id") is not None
        and event.get("pr_ms") is not None
        and 80.0 <= float(event["pr_ms"]) <= 320.0
    ]
    if not candidates:
        return 0.0
    dominant_beats: set[int] = set()
    for _, center_pr in candidates:
        cluster_beats = {
            beat_id
            for beat_id, pr_ms in candidates
            if abs(pr_ms - center_pr) <= 30.0
        }
        if len(cluster_beats) > len(dominant_beats):
            dominant_beats = cluster_beats
    return min(1.0, float(len(dominant_beats)) / float(beat_count))


def compute_pr_dispersion_ms(
    atrial_events: Sequence[Dict[str, Any]]
) -> Optional[float]:
    """Beat-to-beat scatter of the conducted-P PR interval, as a robust sigma.

    Companion to `compute_organized_p_ratio`, which is an integer beat count
    over ~13 beats and therefore cannot resolve finer than ~0.077. This is
    continuous, and it answers a different question: not *how many* beats
    carry a conducted P, but whether those P waves march with the ventricles
    at a fixed interval. A real sinus sequence does; scattered candidate P
    waves picked out of fibrillatory baseline do not.

    One PR per beat (median across leads) so that multilead duplicates of the
    same P wave cannot make a single beat dominate the spread. Returns None
    when too few beats contribute for a MAD to mean anything.
    """
    per_beat: Dict[int, List[float]] = {}
    for event in atrial_events:
        if event.get("association_type") != "conducted":
            continue
        if float(event.get("confidence") or 0.0) < 0.30:
            continue
        beat_id = event.get("associated_qrs_beat_id")
        pr_ms = event.get("pr_ms")
        if beat_id is None or pr_ms is None:
            continue
        pr = float(pr_ms)
        if not (80.0 <= pr <= 320.0):
            continue
        per_beat.setdefault(int(beat_id), []).append(pr)

    if len(per_beat) < _PR_DISPERSION_MIN_BEATS:
        return None
    pr_per_beat = np.asarray(
        [float(np.median(values)) for values in per_beat.values()],
        dtype=float,
    )
    median_pr = float(np.median(pr_per_beat))
    return float(np.median(np.abs(pr_per_beat - median_pr)) * 1.4826)
