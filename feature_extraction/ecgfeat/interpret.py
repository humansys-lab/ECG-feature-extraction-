"""ECG Clinical Interpretation Layer
=====================================
Derives clinical conclusions from ECGFeatures using thresholds from
DXL_Threshold_Reference.xlsx.

Implements the five-step clinical ECG reading workflow:
  Step 1 – Technical quality (signal / lead-reversal sanity)
  Step 2 – Rhythm & rate
  Step 3 – Axis & intervals
  Step 4 – Wave morphology
  Step 5 – Anatomical localisation, reciprocal changes

NOTE: Research scaffold only – not a validated medical device.
NOTE: Q-wave duration criteria require q_dur_ms, which is not currently
      stored in beat_features. Amplitude-ratio criteria are used instead.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from ._engine.measurement.features import estimate_initial_qrs_axis_deg
from .glasgow_rules.models import GlasgowConfig
from .glasgow_rules.rate import bradycardia_limit, tachycardia_limit
from .mi import build_mi_evidence, build_mi_statement_candidates
from .models import (
    BeatAnnotation,
    ECGFeatures,
    ECGInterpretation,
    LeadBeatFeatures,
    PatientMeta,
    RepresentativeLeadFeatures,
    STANDARD_12_LEADS,
    resolve_patient_age,
)
from .pediatric_rules import build_pediatric_hypertrophy_evidence

# ============================================================================
# Threshold constants – sourced from DXL_Threshold_Reference.xlsx
# ============================================================================

# ── Rhythm ───────────────────────────────────────────────────────────────────
HR_TACHYCARDIA_BPM  = 100.0
HR_BRADYCARDIA_BPM  =  50.0   # default; complete AVB landmark <45
HR_COMPLETE_AVB_BPM =  45.0

# Age-adjusted tachy/brady curve reused from the Glasgow engine (verified
# against Physician's Guide chapter 5, PDF page 15). Its adult defaults are
# numerically identical to HR_TACHYCARDIA_BPM/HR_BRADYCARDIA_BPM above, so
# reusing it only changes behaviour for patients with a known pediatric age.
_HR_AGE_CONFIG = GlasgowConfig(
    adult_tachycardia_bpm=HR_TACHYCARDIA_BPM,
    adult_bradycardia_bpm=HR_BRADYCARDIA_BPM,
)

# DXL_Threshold_Reference.xlsx Morphology_Thresholds: "Extreme tachy
# (Critical) | > 220-age | bpm | Critical Value alert" -- the classic
# age-predicted maximal heart rate used here as an implausibility/critical
# upper bound rather than an exercise-testing target.
EXTREME_TACHYCARDIA_AGE_OFFSET_BPM = 220.0

# RR irregularity (coefficient of variation)
RR_CV_IRREGULAR       = 0.15   # consistent with AF / marked irregularity
RR_CV_MILD_IRREGULAR  = 0.08

PVC_RR_SHORTENING_PCT = 15.0  # ≥15% shortening from mean RR
RR_PAUSE_FACTOR       =  1.40  # RR > 140% of background mean RR = pause

# DXL Table 2-1: PR interval AVB-1 threshold (ms) by age group × HR band.
# Keys: (age_group, hr_band)  age_group = "young" (≤60 yr) | "older" (>60 yr)
#                              hr_band   = <50 | 51-90 | 91-120 | >120 bpm
#
# NOTE ON CROSS-ENGINE DIVERGENCE: glasgow_rules/intervals.py's 1st-degree AVB
# rule uses a different methodology entirely -- a continuous age-only formula
# (163 + 0.0087*age_days for <=18y, else a flat 220 ms), per GAN Physician's
# Guide section 6.1 -- rather than this age x heart-rate 2D table. Both are
# faithful to their own cited source; they are simply two distinct vendor
# algorithms for the same clinical question, so the two engines can disagree
# on whether a given tracing shows 1st-degree AV block. Neither overrides the
# other; there is no unified arbitration between them in this codebase.
_PR_AVB1_TABLE: Dict[Tuple[str, str], float] = {
    ("young",  "very_slow"):      210.0,
    ("young",  "normal"):         200.0,
    ("young",  "moderate_tachy"): 180.0,
    ("young",  "tachy"):          170.0,
    ("older",  "very_slow"):      220.0,
    ("older",  "normal"):         210.0,
    ("older",  "moderate_tachy"): 190.0,
    ("older",  "tachy"):          180.0,
}

# ── QRS / T / P axis ─────────────────────────────────────────────────────────
QRS_AXIS_NORMAL_LOW  = -30.0
QRS_AXIS_NORMAL_HIGH =  90.0
QRS_AXIS_LAD_CUTOFF  = -30.0   # LAD: < -30°
QRS_AXIS_LAFB_CUTOFF = -40.0   # LAFB: ≤ -40° (Excel: -40 to -90)
QRS_AXIS_RAD_CUTOFF  =  90.0   # RAD: ≥ 90°
QRS_AXIS_LPFB_LOW    = 120.0   # LPFB: 120–210°
QRS_AXIS_LPFB_HIGH   = 210.0

T_AXIS_NORMAL_LOW    = -10.0
T_AXIS_NORMAL_HIGH   = 100.0
QRS_T_ANGLE_ABNORMAL =  90.0   # > 90° → nonspecific T abnormality

P_AXIS_SINUS_LOW     = -30.0   # DXL: sinus P axis range -30° to +120°
P_AXIS_SINUS_HIGH    = 120.0

# ── Intervals / Conduction ───────────────────────────────────────────────────
PR_SHORT_MS              = 120.0   # WPW short PR interval
PR_AVB1_MS               = 200.0   # 1st-degree AVB

QRS_BORDERLINE_LOW_MS    = 100.0
QRS_BORDERLINE_HIGH_MS   = 110.0
QRS_NONSPECIFIC_HIGH_MS  = 120.0
QRS_BBB_MS               = 120.0   # definite BBB / IVCD

WPW_QRS_MIN_MS           = 100.0
WPW_DELTA_PRESENT        = True    # delta_present flag in beat_features

# Bundle branch specific (RBBB: R' in V1)
RBBB_V1_RPRIME_MIN_MV    = 0.10   # R' must be positive in V1
RBBB_LATERAL_S_MIN_MV    = 0.10   # wide S in I or V6 (absolute)
LBBB_V1_R_MAX_MV         = 0.05   # V1 R almost absent in LBBB (<= 0.05 mV)

# ── QTc ─────────────────────────────────────────────────────────────────────
# Short-QTc tiers mirror clinical_rules/intervals.py's classify_adult_qt
# (AHA/ACCF/HRS 2009) so this DXL-style classification agrees with the
# authoritative CLIN-INTERVAL-QT-01 rule instead of only recognising a single
# "short" cutoff.
QTC_SHORT_MS            = 340.0
QTC_POSSIBLY_SHORT_MS    = 360.0
QTC_BORDERLINE_SHORT_MS  = 390.0
QTC_BORDERLINE_PROL_MS = 465.0
QTC_PROLONGED_MS       = 485.0
QTC_SIG_PROLONGED_MS   = 520.0

# ── Atrial enlargement ───────────────────────────────────────────────────────
# RAE
RAE_P_AMP_CONSIDER_MV = 0.24   # in any limb lead → consider RAE
RAE_P_AMP_CONFIRM_MV  = 0.25   # in ≥2 limb leads → RAE confirmed
RAE_V1_BIPHASIC_MV    = 0.20   # biphasic P in V1 positive terminal

# LAE
# P duration bar for LAE.  DXL_Threshold_Reference.xlsx says 110 ms, but that
# number cannot be diagnostic on a correctly scaled measurement: the LUDB expert
# annotations put the *median* lead-II P duration at 110 ms, and once the
# criteria read the consensus duration (see `_median_p_dur_per_lead`) 53.2% of
# PTB-XL 09000 records reach 110 ms against 28.6% at 120 ms.  120 ms is also
# what this codebase's authoritative layer already uses
# (`clinical_rules/config.py: p_duration_prolonged_ms`), so the two engines now
# apply the same bar instead of two different ones.
LAE_P_DUR_MS          = 120.0
# Classic bifid-P criterion: the two peaks must be separated by ≥ 40 ms.  The
# notch detector in p_morphology.py accepts a 20 ms separation because it also
# serves morphology description, so the diagnostic bar has to be applied here:
# among the leads that were triggering the LAE notch path, the median notch
# interval was 26 ms and only 29.7% reached 40 ms.
LAE_P_NOTCH_INTERVAL_MS = 40.0
LAE_V1_NEG_AMP_MV     = 0.09   # |V1 negative P terminal| ≥ 0.09 mV
LAE_V1_NEG_DUR_MS     = 30.0   # V1 negative terminal ≥ 30 ms
# Same table's V1 negative-area row: 0.6 Ashman, where 1 Ashman = 40 ms × 0.1 mV
# = 4 mV·ms.  Used as an alternative to the amplitude+duration pair, which is
# what the terminal-component grading falls back on when one of the two is
# unmeasured.
LAE_V1_NEG_AREA_MV_MS = -2.4
LAE_V1_NEG_PROB_MV    = 0.10   # probable LAE: |V1 neg| > 0.10 mV, terminal dur > 50 ms
LAE_V1_NEG_PROB_DUR_MS = 50.0
LAE_V1_NEG_DEF_MV     = 0.15   # definite LAE: |V1 neg| > 0.15 mV, terminal dur > 60 ms
LAE_V1_NEG_DEF_DUR_MS = 60.0

# BAE (biatrial enlargement).  DXL_Threshold_Reference.xlsx lists BAE as two
# rows: a P-duration row (80 ms) and the combined row that names the actual
# criterion (V1 terminal negative <= -0.15 mV together with P > 0.30 mV in two
# limb leads).  The duration row is the qualifying minimum for that
# combination -- structurally the same kind of row as "RAE | P min duration |
# 60 ms", implemented as a gate in RAE_P_DUR_MIN_MS -- not a sufficient
# criterion on its own.  Read as sufficient it made BAE *easier* to reach than
# LAE (110 ms) and fired on 781 of 1000 PTB-XL 09000 records: all 781 came from
# the duration path alone, only 2 also satisfied RAE+LAE, 335 never even
# reached the LAE duration bar, and 391 of the records called biatrial were
# labelled completely normal.
BAE_P_DUR_MIN_MS      = 80.0   # minimum P duration for the BAE combination to count
BAE_V1_NEG_AMP_MV     = 0.15   # V1 terminal negative component <= -0.15 mV
BAE_LIMB_P_AMP_MV     = 0.30   # with P > 0.30 mV in >= 2 limb leads

# PTF-V1 (P terminal force, mV·ms; negative = posterior component)
# Morris criterion: |PTF-V1| >= 0.04 mm·s = 0.004 mV·s = 4 mV·ms (in code units).
# Probable LAE uses half the definite threshold as a borderline zone.
PTF_V1_PROBABLE_MV_MS = -2.0
PTF_V1_DEFINITE_MV_MS = -4.0

# ── Q waves (amplitude / ratio criteria only; duration = future work) ─────────
# Inferior (II, III, aVF): |Q| > 1/6 R
Q_INF_R_RATIO  = 1.0 / 6.0

# Lateral (I, aVL, V5, V6): |Q| > 0.10 mV AND > 20 % R
Q_LAT_AMP_MV   = 0.10
Q_LAT_R_RATIO  = 0.20

# Anterior (V1–V4): |Q| > 0.07 mV AND > 20 % R
# (ratio guard added: without it, a trivial/noise-level Q next to a very
# large R -- e.g. LVH-driven precordial voltage -- was flagged pathological)
Q_ANT_AMP_MV   = 0.07
Q_ANT_R_RATIO  = 0.20

# Any MI: |Q| > 1/5 R
Q_MI_R_RATIO   = 1.0 / 5.0

# ── ST segment ───────────────────────────────────────────────────────────────
ST_DEP_SIGNIFICANT_MV  = 0.03   # smallest clinically significant depression
ST_ELE_BORDERLINE_MV   = 0.05   # borderline elevation (~0.5 mm)
ST_ELE_ABNORMAL_MV     = 0.10   # abnormal elevation (>1 mm)

# Territory-specific elevation thresholds (acute infarct)
ST_ELE_ANTERIOR_MV     = 0.25   # V2–V5 acute
ST_ELE_ANTERIOR_PROB_MV= 0.15   # V2–V5 probable/possible
ST_ELE_INFERIOR_MV     = 0.10   # II/III/aVF acute
ST_ELE_ANTS_MV         = 0.20   # V1–V2 anteroseptal acute
ST_ELE_EXTENSIVE_MV    = 0.20   # V1–V6 extensive anterior
ST_ELE_LATERAL_MV      = 0.10   # V5/V6/I/aVL lateral
ST_DEP_POSTERIOR_MV    = -0.10  # V1–V3 depression → posterior reciprocal

# ── LVH voltage (mV) ─────────────────────────────────────────────────────────
LVH_R_AVL_M   = 1.2
LVH_R_AVL_F   = 1.1
LVH_RI_SIII   = 2.5
LVH_R_V5V6    = 2.6
LVH_SOKOLOW_M = 3.5
LVH_SOKOLOW_F = 3.25
LVH_CORNELL_M = 2.8
LVH_CORNELL_F = 2.2
LVH_CORNELL_PROD_M = 280.0  # mV·ms
LVH_CORNELL_PROD_F = 300.0

# ── Low voltage ───────────────────────────────────────────────────────────────
LOW_V_FRONTAL_BORDER  = 0.60   # borderline: max frontal QRS peak-to-peak < 0.60 mV
LOW_V_FRONTAL_DEF     = 0.50   # definite
LOW_V_PRECORDIAL_DEF  = 1.00   # precordial peak-to-peak definite

# ── RVH ──────────────────────────────────────────────────────────────────────
RVH_R_V1_MIN_MV   = 0.30   # R' in V1 ≥ 0.30 mV for RVH
RVH_QS_RATIO      = 0.75   # R > 75 % of Q+S in V1

# ── Tall T ───────────────────────────────────────────────────────────────────
TALL_T_ABS_MV = 1.2
TALL_T_REL_MV = 0.5   # AND > 50 % of peak-to-peak QRS voltage

# ── Dextrocardia ─────────────────────────────────────────────────────────────
DEXTRO_V5V6_MAX_PP_MV  = 0.50   # small QRS in V5/V6: peak-to-peak < 0.50 mV

# ── RAE P-duration gate ───────────────────────────────────────────────────────
RAE_P_DUR_MIN_MS       = 60.0   # P duration must also be ≥ 60 ms for RAE amplitude to count
# RAE is a *tall* P, not a wide one; a tall-and-wide P is left atrial (or both).
# Same bar as `clinical_rules/config.py: p_duration_prolonged_ms`.
RAE_P_DUR_MAX_MS       = 120.0

# ── LAE combined criterion ────────────────────────────────────────────────────
LAE_P_AMP_LIMB_MV      = 0.10   # limb-lead P amplitude > 0.10 mV combined with duration

# ── RVH graded scoring ────────────────────────────────────────────────────────
RVH_LAT_NEG_AMP_MV     = 0.20   # Q/S/S' > 0.20 mV in I or V6 (lateral negative criterion)
RVH_LAT_DUR_MS         = 40.0   # Q/S/S' in I or V6 must also last >= 40 ms (when measured)
RVH_R_PRIME_DUR_MS     = 20.0   # R' in V1 must also last >= 20 ms (when measured)
RVH_REPOL_ST_MV        = -0.05  # ST depression threshold for RVH repolarization abnormality

# ── LVH enhanced scoring ──────────────────────────────────────────────────────
LVH_AGE_MIN            = 35.0   # LVH codes ignored for patients < 35 years
LVH_VAT_MS             = 45.0   # VAT in V5 or V6 > 45 ms → QRS:VAT:WIDE flag
LVH_STRAIN_ST_MV       = -0.05  # anterolateral ST depression for LV strain
LVH_STRAIN_T_MV        =  0.0   # T must be negative (< 0) for LV strain pattern
LVH_LAD_DEG            = -30.0  # same LAD cutoff used in LVH scoring

# ── Posterior MI R-dominance ──────────────────────────────────────────────────
PMI_MIN_T_MV           =  0.02  # T must be upright for posterior MI R-dominant pattern
PMI_MAX_Q_MV           =  0.07  # Q must be insignificant (< 0.07 mV)

# ── Rate-related ST depression ────────────────────────────────────────────────
ST_RATE_RELATED_BASE   = 190.0  # HR > (190 − patient_age) → ST dep probably rate-related

# ── QTc electrolyte thresholds ────────────────────────────────────────────────
QTC_HYPERCALCEMIA_MS   = 310.0  # QTc < 310 ms → suggest hypercalcemia
QTC_HYPOKALEMIA_ST_THR = -0.05  # ST dep needed alongside QTc > 520 for hypokalemia hint

# ── Pediatric algorithm (DXL Chapter 4) ───────────────────────────────────────
# NOTE ON CROSS-ENGINE DIVERGENCE: this pediatric cutoff is intentionally
# different from glasgow_rules/context.py's PatientRoute.pediatric flag
# (< 18 years). Each is faithful to its own cited source -- this one to the
# Philips DXL pediatric morphology spec / DXL_Threshold_Reference.xlsx (whose
# age tables stop at "12-15 years"), the Glasgow one to the GAN Physician's
# Guide section 4.3.6 ("the patient is under 18 years of age") -- but the two
# vendor spec families define "pediatric" differently, so a 16-17 year old
# patient is routed through DXL-style adult morphology rules here while the
# Glasgow report for the same tracing states it used pediatric criteria. This
# is a known, deliberate divergence rather than a bug in either engine.
PEDS_MAX_AGE_YEARS        = 16.0   # Chapter 4 applies to patients 0–<16 years
PEDS_DEXTRO_P_AXIS_LOW    = 90.0   # Frontal P axis 90°–180° → dextrocardia
PEDS_DEXTRO_P_AXIS_HIGH   = 180.0
PEDS_DEXTRO_S_MV          = 0.6    # Large S in I AND V6 for dextrocardia
# DXL_Threshold_Reference.xlsx Morphology_Thresholds: pediatric RAE limb-lead
# amplitude threshold is 0.20 mV, lower than the adult 0.24 mV
# (RAE_P_AMP_CONSIDER_MV) because children's P waves are smaller. A previous
# revision removed this constant as "unused"; restored and wired into
# _p_wave_morphology's pediatric branch below.
PEDS_RAE_P_AMP_MV         = 0.20   # limb P amplitude threshold for pediatric RAE
PEDS_RBBB_R_PRIME_MV      = 0.15   # R' in V1 ≥ 0.15 mV for pediatric RBBB
PEDS_RBBB_R_PRIME_DUR_MS  = 20.0   # R' in V1 must also last ≥ 20 ms (gated when measured)
# QTc thresholds (age- and sex-dependent)
PEDS_QTC_SHORT_MS         = 340.0
PEDS_QTC_BPROL_U5_MS      = 450.0   # borderline prolonged, < 5 years
PEDS_QTC_BPROL_5TO12_MS   = 454.0   # 5–12 years
PEDS_QTC_BPROL_BOY_MS     = 458.0   # boys ≥ 13 years
PEDS_QTC_BPROL_GIRL_MS    = 465.0   # girls ≥ 13 years
PEDS_QTC_PROL_U5_MS       = 470.0   # prolonged, < 5 years
PEDS_QTC_PROL_5TO12_MS    = 474.0   # 5–12 years
PEDS_QTC_PROL_BOY_MS      = 478.0   # boys ≥ 13 years
PEDS_QTC_PROL_GIRL_MS     = 485.0   # girls ≥ 13 years
PEDS_QTC_HYPOCALCEMIA_MS  = 520.0   # QTc > 520 → suggest hypocalcemia (pediatric)
# LSH (Left Septal Hypertrophy)
PEDS_LSH_R_V1_MV          = 1.0    # prominent R in V1 → LSH (proxy for >98th percentile)
PEDS_LSH_CONSIDER_R_V1_MV = 0.7    # moderate R in V1 → consider LSH
# Pericarditis / early repolarization age gates
PEDS_PERICARDITIS_AGE_LOW = 5.0
PEDS_PERICARDITIS_AGE_HIGH= 15.0
PEDS_EARLY_REPOL_AGE_LOW  = 13.0
PEDS_EARLY_REPOL_AGE_HIGH = 15.0
# NOTE: pediatric pericarditis is decided by ST-elevation territory membership
# (see _st_analysis / terr_ele in the pediatric branch below), not by a direct
# J-point cutoff.  A former PEDS_ST_ELE_PERI_MV = 0.15 constant was unused and
# misrepresented the actual rule, so it was removed.

# ── Pediatric age-dependent lookup tables ─────────────────────────────────────
# Each entry: (age_low_years_inclusive, age_high_years_exclusive, value)
# Age is in decimal years (e.g., 0 hours = 0, 1 day ≈ 1/365, 1 month ≈ 1/12).

# Table 4-5: Mean QRS duration normal limits (ms)
_PEDS_QRS_NORMAL_MS: List[Tuple] = [
    (0,         1/365,    70),   # 0–23 hours
    (1/365,     4/365,    70),   # 1–3 days
    (4/365,     7/365,    70),   # 4–6 days
    (7/365,     30/365,   70),   # 7–29 days
    (30/365,    91/365,   84),   # 1–2 months
    (91/365,    182/365,  84),   # 3–5 months
    (182/365,   1.0,      84),   # 6–11 months
    (1.0,       3.0,      78),   # 1–2 years
    (3.0,       5.0,      88),   # 3–4 years
    (5.0,       8.0,      88),   # 5–7 years
    (8.0,       12.0,     88),   # 8–11 years
    (12.0,      16.0,     100),  # 12–15 years
]

# Table 4-1: LAD threshold — axis < this value → LAD (standard ±180° notation)
_PEDS_LAD_THRESHOLD: List[Tuple] = [
    (0,         1/365,    54),
    (1/365,     4/365,    54),
    (4/365,     7/365,    54),
    (7/365,     30/365,   54),
    (30/365,    91/365,   20),
    (91/365,    182/365,  -6),
    (182/365,   1.0,      -6),
    (1.0,       3.0,      -6),
    (3.0,       5.0,      -10),
    (5.0,       8.0,      -10),
    (8.0,       12.0,     -10),
    (12.0,      16.0,     -15),
]

# Table 4-2: Borderline LAD upper boundary (standard ±180°)
# axis in [lad_thr, blad_upper) → borderline LAD
_PEDS_BLAD_UPPER: List[Tuple] = [
    (0,         1/365,    65),
    (1/365,     4/365,    65),
    (4/365,     7/365,    65),
    (7/365,     30/365,   65),
    (30/365,    91/365,   30),
    (91/365,    182/365,  1),
    (182/365,   1.0,      1),
    (1.0,       3.0,      1),
    (3.0,       5.0,      1),
    (5.0,       8.0,      1),
    (8.0,       12.0,     1),
    (12.0,      16.0,     1),
]

# Table 4-3: RAD threshold (0–360° clockwise from I+; > 180 → negative standard)
# a_360 in [rad_thr_360, 269] → RAD
_PEDS_RAD_THRESHOLD_360: List[Tuple] = [
    (0,         1/365,    216),
    (1/365,     4/365,    216),
    (4/365,     7/365,    216),
    (7/365,     30/365,   216),
    (30/365,    91/365,   131),
    (91/365,    182/365,  131),
    (182/365,   1.0,      131),
    (1.0,       3.0,      131),
    (3.0,       5.0,      146),
    (5.0,       8.0,      201),
    (8.0,       12.0,     151),
    (12.0,      16.0,     161),
]

# Table 4-4: Borderline RAD lower boundary (0–360°)
# a_360 in [brad_thr_360, rad_thr_360) → borderline RAD
_PEDS_BRAD_THRESHOLD_360: List[Tuple] = [
    (0,         1/365,    205),
    (1/365,     4/365,    205),
    (4/365,     7/365,    205),
    (7/365,     30/365,   200),
    (30/365,    91/365,   115),
    (91/365,    182/365,  115),
    (182/365,   1.0,      115),
    (1.0,       3.0,      115),
    (3.0,       5.0,      126),
    (5.0,       8.0,      160),
    (8.0,       12.0,     135),
    (12.0,      16.0,     145),
]

# ── Lead groupings ────────────────────────────────────────────────────────────
INFERIOR_LEADS      = ["II", "III", "aVF"]
ANTERIOR_LEADS      = ["V1", "V2", "V3", "V4"]
ANTEROSEPTAL_LEADS  = ["V1", "V2"]
EXTENSIVE_ANT_LEADS = ["V1", "V2", "V3", "V4", "V5", "V6"]
HIGH_LATERAL_LEADS  = ["I", "aVL"]
LOW_LATERAL_LEADS   = ["V5", "V6"]
LATERAL_LEADS       = ["I", "aVL", "V5", "V6"]
ANTEROLATERAL_LEADS = ["V2", "V3", "V4", "V5", "V6", "I", "aVL"]
POSTERIOR_LEADS     = ["V1", "V2", "V3"]   # depression here = posterior
FRONTAL_LEADS       = ["I", "II", "III", "aVR", "aVL", "aVF"]
LIMB_LEADS_P        = ["I", "II", "III", "aVL", "aVF"]   # for P-wave amplitude
PRECORDIAL_LEADS    = ["V1", "V2", "V3", "V4", "V5", "V6"]


# ============================================================================
# Internal helpers
# ============================================================================

def _lp(
    representative_leads: Dict[str, RepresentativeLeadFeatures],
    lead: str,
    param: str,
    required_flags: Tuple[str, ...] = (),
) -> Optional[float]:
    """Retrieve a scalar param from representative_leads, returning None if absent."""
    rep = representative_leads.get(lead)
    if rep is None:
        return None
    if required_flags and not all(bool(rep.params.get(flag, False)) for flag in required_flags):
        return None
    val = rep.params.get(param)
    if val is None:
        return None
    try:
        f = float(val)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _lead_param_bool(
    representative_leads: Dict[str, RepresentativeLeadFeatures],
    lead: str,
    param: str,
    required_flags: Tuple[str, ...] = (),
) -> bool:
    rep = representative_leads.get(lead)
    if rep is None:
        return False
    if required_flags and not all(bool(rep.params.get(flag, False)) for flag in required_flags):
        return False
    return bool(rep.params.get(param, False))


def _v1_terminal_negative_component_clears_lae_bars(
    representative_leads: Dict[str, RepresentativeLeadFeatures],
) -> bool:
    """Whether V1's terminal negative component clears the DXL LAE bars.

    Amplitude (≥ 0.09 mV) *and* duration (≥ 30 ms), per
    DXL_Threshold_Reference.xlsx.  This is the strict grading, required before
    PTF-V1 alone may be promoted to definite LAE; the suspicion tier uses
    `_v1_terminal_negative_component_supports_lae_suspicion`, which also accepts
    the table's negative-area row.
    """
    v1_rep = representative_leads.get("V1")
    if v1_rep is None:
        return False
    if v1_rep.params.get("p_measurement_suppressed") is not None:
        return False
    terminal_amp = _lp(
        representative_leads,
        "V1",
        "p_terminal_amp_mv",
        required_flags=("reliable_for_p",),
    )
    if terminal_amp is None:
        terminal_amp = _lp(
            representative_leads,
            "V1",
            "p_terminal_amplitude_mV",
            required_flags=("reliable_for_p",),
        )
    terminal_dur = _lp(
        representative_leads,
        "V1",
        "p_terminal_duration_ms",
        required_flags=("reliable_for_p",),
    )
    terminal_area = _lp(
        representative_leads,
        "V1",
        "p_terminal_area_mv_ms",
        required_flags=("reliable_for_p",),
    )
    if terminal_amp is None or terminal_dur is None:
        return False
    if terminal_amp > -LAE_V1_NEG_AMP_MV:
        return False
    if terminal_dur < LAE_V1_NEG_DUR_MS:
        return False
    return terminal_area is None or terminal_area < 0.0


def _v1_terminal_negative_component_supports_lae_suspicion(
    representative_leads: Dict[str, RepresentativeLeadFeatures],
) -> bool:
    """Whether V1's terminal negative component is large enough to suspect LAE.

    A biphasic P in V1 with a negative terminal component is the ordinary V1 P
    morphology, so the observation alone is not a finding: on the PTB-XL 09000
    shard it matched 87.1% of records and made `lae_suspected` a near-constant
    95.5%.  DXL_Threshold_Reference.xlsx grades that component either on
    amplitude+duration or on its negative area (0.6 Ashman = 2.4 mV·ms), so a
    record whose terminal amplitude or duration was not measured is still graded
    on the area it did produce.  The area row is deliberately *not* accepted by
    the strict grading above, which promotes PTF-V1 to definite LAE.
    """
    terminal_area = _lp(
        representative_leads,
        "V1",
        "p_terminal_area_mv_ms",
        required_flags=("reliable_for_p",),
    )
    if terminal_area is not None and terminal_area <= LAE_V1_NEG_AREA_MV_MS:
        v1_rep = representative_leads.get("V1")
        if v1_rep is not None and v1_rep.params.get("p_measurement_suppressed") is None:
            return True
    return _v1_terminal_negative_component_clears_lae_bars(representative_leads)


def _collect_rr_ms(beats: List[BeatAnnotation]) -> List[float]:
    seen: List[float] = []
    for b in beats:
        if b.rr_next_ms is not None and np.isfinite(b.rr_next_ms) and b.rr_next_ms > 0:
            seen.append(float(b.rr_next_ms))
    return seen


def _lead_has_flags(
    representative_leads: Dict[str, RepresentativeLeadFeatures],
    lead: str,
    *required_flags: str,
) -> bool:
    rep = representative_leads.get(lead)
    if rep is None:
        return False
    return all(bool(rep.params.get(flag, False)) for flag in required_flags)


def _st_j_lp(
    representative_leads: Dict[str, RepresentativeLeadFeatures],
    lead: str,
) -> Optional[float]:
    rep = representative_leads.get(lead)
    if rep is not None and bool(rep.params.get("st_j_unreliable", False)):
        return None
    if rep is not None:
        confidence = rep.params.get("qt_confidence_mean")
        if confidence is not None and float(confidence) < 0.45:
            return None
    return _lp(representative_leads, lead, "st_on_mv", required_flags=("reliable_for_qt",))


def _qrs_peak_to_peak_mv(
    representative_leads: Dict[str, RepresentativeLeadFeatures],
    lead: str,
    required_flags: Tuple[str, ...] = ("reliable_for_qrs",),
) -> Optional[float]:
    vals = [
        _lp(representative_leads, lead, "q_amp_mv", required_flags=required_flags),
        _lp(representative_leads, lead, "r_amp_mv", required_flags=required_flags),
        _lp(representative_leads, lead, "s_amp_mv", required_flags=required_flags),
    ]
    finite_vals = [float(v) for v in vals if v is not None and np.isfinite(v)]
    if not finite_vals:
        return None
    return max([0.0] + finite_vals) - min([0.0] + finite_vals)


def _median_r_prime_per_lead(
    beat_features: List[LeadBeatFeatures],
) -> Dict[str, Optional[float]]:
    """Aggregate r_prime_amp_mv (not in representative_leads) by lead median."""
    by_lead: Dict[str, List[float]] = defaultdict(list)
    for bf in beat_features:
        if bf.r_prime_amp_mv is not None and np.isfinite(bf.r_prime_amp_mv):
            by_lead[bf.lead].append(float(bf.r_prime_amp_mv))
    return {
        lead: float(np.median(vals)) if vals else None
        for lead, vals in by_lead.items()
    }


def _median_r_prime_dur_per_lead(
    beat_features: List[LeadBeatFeatures],
) -> Dict[str, Optional[float]]:
    """Aggregate r_prime_duration_ms (not in representative_leads) by lead median."""
    by_lead: Dict[str, List[float]] = defaultdict(list)
    for bf in beat_features:
        if bf.r_prime_duration_ms is not None and np.isfinite(bf.r_prime_duration_ms):
            by_lead[bf.lead].append(float(bf.r_prime_duration_ms))
    return {
        lead: float(np.median(vals)) if vals else None
        for lead, vals in by_lead.items()
    }


def _median_q_dur_per_lead(
    beat_features: List[LeadBeatFeatures],
    fs: int,
) -> Dict[str, Optional[float]]:
    """
    Approximate Q duration from (qrs.peak - qrs.onset) only when q_amp_mv < 0.
    This overestimates Q duration but serves as a rough gate.
    """
    by_lead: Dict[str, List[float]] = defaultdict(list)
    for bf in beat_features:
        if (
            bf.q_amp_mv is not None
            and bf.q_amp_mv < 0
            and bf.qrs.onset is not None
            and bf.qrs.peak is not None
        ):
            dur_ms = (bf.qrs.peak - bf.qrs.onset) * 1000.0 / fs
            if 0 < dur_ms < 300:
                by_lead[bf.lead].append(dur_ms)
    return {
        lead: float(np.median(vals)) if vals else None
        for lead, vals in by_lead.items()
    }


def _median_p_dur_per_lead(
    beat_features: List[LeadBeatFeatures],
    fs: int,
    representative_leads: Optional[Dict[str, RepresentativeLeadFeatures]] = None,
) -> Dict[str, Optional[float]]:
    """Per-lead P duration for the atrial-enlargement criteria.

    Prefers the cross-beat consensus duration when the representative lead
    carries one, falling back to the median of the raw per-beat measurements.
    The published duration bars (RAE ≥ 60 ms, BAE ≥ 80 ms, LAE ≥ 110 ms) assume
    a properly measured P duration, and the raw per-beat value is not on that
    scale: against the LUDB expert annotations it runs 22 ms short in lead II
    (median 78 ms vs 110 ms), while the consensus value lands within a few ms of
    the expert.  `clinical_rules/hypertrophy.py` already reads
    `p_dur_consensus_ms` for the same criteria, so this also stops the two
    engines from applying the same number to two different scales.
    """
    by_lead: Dict[str, List[float]] = defaultdict(list)
    for bf in beat_features:
        if bf.p_dur_ms is not None and np.isfinite(bf.p_dur_ms) and bf.p_dur_ms > 0:
            by_lead[bf.lead].append(float(bf.p_dur_ms))
    raw = {
        lead: float(np.median(vals)) if vals else None
        for lead, vals in by_lead.items()
    }
    if not representative_leads:
        return raw
    resolved = dict(raw)
    for lead in set(raw) | set(representative_leads):
        consensus = _lp(representative_leads, lead, "p_dur_consensus_ms")
        if consensus is not None and consensus > 0:
            resolved[lead] = float(consensus)
    return resolved


def _angle_diff(a_deg: float, b_deg: float) -> float:
    """Absolute angular difference in [0, 180]."""
    d = abs(a_deg - b_deg) % 360
    return d if d <= 180 else 360 - d


# ============================================================================
# Step 2 – Rhythm & rate
# ============================================================================

def _classify_heart_rate(hr_bpm: Optional[float], age_days: Optional[float] = None) -> str:
    """Classify heart rate against age-appropriate tachy/brady limits.

    Neonates and infants have much higher normal heart rates than adults
    (DXL_Threshold_Reference.xlsx Pediatric_Normals, e.g. a healthy newborn
    averages ~123 bpm), so the fixed adult 100/50 bpm cutoffs below must not
    be applied unconditionally. When age is known, reuse the Glasgow-manual
    age curve (verified against Physician's Guide chapter 5) which reduces
    to these same adult thresholds by age 18.
    """
    if hr_bpm is None:
        return "indeterminate"
    if age_days is not None:
        tachy_limit = tachycardia_limit(age_days, _HR_AGE_CONFIG)
        brady_limit = bradycardia_limit(age_days, _HR_AGE_CONFIG)
    else:
        tachy_limit = HR_TACHYCARDIA_BPM
        brady_limit = HR_BRADYCARDIA_BPM
    if hr_bpm < HR_COMPLETE_AVB_BPM:
        return "extreme_bradycardia"
    if hr_bpm < brady_limit:
        return "bradycardia"
    if hr_bpm < tachy_limit:
        return "normal"
    return "tachycardia"


def _rr_irregularity(
    beats: List[BeatAnnotation],
    representative_leads: Dict[str, RepresentativeLeadFeatures],
) -> Tuple[Optional[float], str, bool]:
    """
    Returns (rr_cv, irregularity_class, probable_af).

    probable_af requires:
      - rr_cv > RR_CV_IRREGULAR (irregularity threshold)
      - low P-wave confidence across available P-reliable leads
    """
    rr = _collect_rr_ms(beats)
    if len(rr) < 3:
        return None, "indeterminate", False
    mean_rr = float(np.mean(rr))
    std_rr  = float(np.std(rr))
    cv = std_rr / mean_rr if mean_rr > 0 else 0.0

    if cv > RR_CV_IRREGULAR:
        cls = "irregular"
    elif cv > RR_CV_MILD_IRREGULAR:
        cls = "mildly_irregular"
    else:
        cls = "regular"

    # AF: irregular + no clear organised P waves in any P-reliable lead.
    p_conf_candidates = [
        conf
        for lead in STANDARD_12_LEADS
        for conf in [
            _lp(
                representative_leads,
                lead,
                "p_confidence_mean",
                required_flags=("reliable_for_p",),
            )
        ]
        if conf is not None
    ]
    low_p = not p_conf_candidates or max(p_conf_candidates) < 0.30
    probable_af = (cls == "irregular") and low_p

    return round(cv, 4), cls, probable_af


# ============================================================================
# Step 3 – Axis & intervals
# ============================================================================

def _classify_qrs_axis(qrs_axis_deg: Optional[float]) -> str:
    if qrs_axis_deg is None:
        return "indeterminate"
    a = float(qrs_axis_deg)
    # Normalise to (-180, 180]
    a = ((a + 180) % 360) - 180
    if QRS_AXIS_NORMAL_LOW <= a <= QRS_AXIS_NORMAL_HIGH:
        return "normal"
    if -90 <= a < QRS_AXIS_LAD_CUTOFF:
        if a <= QRS_AXIS_LAFB_CUTOFF:
            return "LAFB"    # left-axis deviation in LAFB range
        return "LAD"
    if QRS_AXIS_RAD_CUTOFF <= a <= 180:
        if QRS_AXIS_LPFB_LOW <= a <= QRS_AXIS_LPFB_HIGH:
            return "LPFB"
        return "RAD"
    # a < -90: the LPFB zone (120 deg to 210 deg, 0-360 clockwise) wraps past
    # +/-180 deg. 210 deg == -150 deg in this standard +/-180 notation, so the
    # wedge from -180 to -150 deg is still LPFB, not "extreme"/indeterminate.
    if a <= QRS_AXIS_LPFB_HIGH - 360.0:
        return "LPFB"
    return "ERAD"


def _classify_t_axis(
    t_axis_deg: Optional[float],
    qrs_axis_deg: Optional[float],
) -> Tuple[str, Optional[float]]:
    if t_axis_deg is None:
        return "indeterminate", None
    t = float(t_axis_deg)
    t_norm = ((t + 180) % 360) - 180
    t_class = "normal" if T_AXIS_NORMAL_LOW <= t_norm <= T_AXIS_NORMAL_HIGH else "abnormal"
    angle = None
    if qrs_axis_deg is not None:
        angle = round(_angle_diff(float(qrs_axis_deg), t), 1)
    return t_class, angle


def _pr_avb1_threshold(hr_bpm: Optional[float], age: Optional[float]) -> float:
    """DXL Table 2-1: age×HR two-dimensional PR threshold for 1st-degree AVB."""
    hr = float(hr_bpm) if hr_bpm is not None else 70.0
    age_group = "older" if (age is not None and age > 60) else "young"
    if hr <= 50:
        hr_band = "very_slow"
    elif hr <= 90:
        hr_band = "normal"
    elif hr <= 120:
        hr_band = "moderate_tachy"
    else:
        hr_band = "tachy"
    return _PR_AVB1_TABLE.get((age_group, hr_band), 200.0)


def _classify_pr(
    pr_ms: Optional[float],
    delta_present_any: bool,
    hr_bpm: Optional[float] = None,
    age: Optional[float] = None,
) -> Tuple[str, Optional[int]]:
    """Returns (pr_class, avb_grade) using DXL Table 2-1 dynamic AVB-1 threshold."""
    if pr_ms is None:
        return "indeterminate", None
    if pr_ms < PR_SHORT_MS:
        return "short", None            # WPW territory
    avb1_thr = _pr_avb1_threshold(hr_bpm, age)
    if pr_ms > avb1_thr:
        return "avb1", 1
    return "normal", None


def _classify_qrs_width(
    qrs_ms: Optional[float],
    representative_leads: Dict[str, RepresentativeLeadFeatures],
    r_prime_by_lead: Dict[str, Optional[float]],
) -> Tuple[str, Optional[str]]:
    """Returns (width_class, bundle_branch_block_label)."""
    if qrs_ms is None:
        return "indeterminate", None

    if qrs_ms < QRS_BORDERLINE_LOW_MS:
        return "normal", None

    # ── Incomplete / borderline (100–120 ms) ─────────────────────────────────
    if qrs_ms < QRS_BBB_MS:
        # Fetch lateral and V1 amplitudes (needed for both iRBBB and iLBBB)
        _v1_r_bl  = _lp(representative_leads, "V1", "r_amp_mv", required_flags=("reliable_for_qrs",))
        _i_r_bl   = _lp(representative_leads, "I",  "r_amp_mv", required_flags=("reliable_for_qrs",))
        _v6_r_bl  = _lp(representative_leads, "V6", "r_amp_mv", required_flags=("reliable_for_qrs",))

        # Incomplete RBBB: R' in V1 ≥ 0.10 mV
        v1_rprime = r_prime_by_lead.get("V1") if _lead_has_flags(representative_leads, "V1", "reliable_for_qrs") else None
        if v1_rprime is not None and v1_rprime >= RBBB_V1_RPRIME_MIN_MV:
            return "borderline_ivcd", "incomplete_RBBB"

        # Incomplete LBBB (110–120 ms): absent/tiny R in V1 + broad R in I or V6
        if qrs_ms >= QRS_BORDERLINE_HIGH_MS:
            _ilbbb_v1  = (_v1_r_bl is None or _v1_r_bl <= LBBB_V1_R_MAX_MV)
            _ilbbb_lat = (
                (_v6_r_bl is not None and _v6_r_bl > 0.30) or
                (_i_r_bl  is not None and _i_r_bl  > 0.10)
            )
            if _ilbbb_v1 and _ilbbb_lat:
                return "nonspecific_ivcd", "incomplete_LBBB"

        if QRS_BORDERLINE_LOW_MS <= qrs_ms < QRS_BORDERLINE_HIGH_MS:
            return "borderline_ivcd", "IVCD"
        return "nonspecific_ivcd", "IVCD"

    # ── Definite BBB (≥ 120 ms) ──────────────────────────────────────────────
    v1_rprime = r_prime_by_lead.get("V1") if _lead_has_flags(representative_leads, "V1", "reliable_for_qrs") else None
    v1_r      = _lp(representative_leads, "V1", "r_amp_mv", required_flags=("reliable_for_qrs",))
    i_r       = _lp(representative_leads, "I",  "r_amp_mv", required_flags=("reliable_for_qrs",))
    v6_r      = _lp(representative_leads, "V6", "r_amp_mv", required_flags=("reliable_for_qrs",))
    i_s       = _lp(representative_leads, "I",  "s_amp_mv", required_flags=("reliable_for_qrs",))
    v6_s      = _lp(representative_leads, "V6", "s_amp_mv", required_flags=("reliable_for_qrs",))

    # RBBB: R' in V1 + wide S in I and/or V6
    rbbb_v1 = (v1_rprime is not None and v1_rprime >= RBBB_V1_RPRIME_MIN_MV)
    rbbb_lat = (
        (i_s  is not None and i_s  < -RBBB_LATERAL_S_MIN_MV) or
        (v6_s is not None and v6_s < -RBBB_LATERAL_S_MIN_MV)
    )
    if rbbb_v1 and rbbb_lat:
        return "bbb", "RBBB"

    # LBBB: absent/tiny R in V1 + dominant R in I and V6
    lbbb_v1 = (v1_r is None or v1_r <= LBBB_V1_R_MAX_MV)
    lbbb_lat = (
        (v6_r is not None and v6_r > 0.30) or
        (i_r is not None and i_r > 0.10)
    )
    if lbbb_v1 and lbbb_lat:
        return "bbb", "LBBB"

    return "bbb", "IVCD"


def _resolve_qrs_width_class(
    qrs_ms: Optional[float],
    qrs_wide_ms: Optional[float],
    representative_leads: Dict[str, RepresentativeLeadFeatures],
    r_prime_by_lead: Dict[str, Optional[float]],
) -> Tuple[str, Optional[str], bool]:
    """Classify QRS width, escalating via the looser P90 consensus when it
    disagrees with the primary measurement channel.

    ``qrs_ms`` (P75 offset consensus) requires >=2 corroborating leads before
    a late QRS offset counts toward the global measurement, so a genuinely
    wide QRS visible robustly in the P90 channel but not the stricter P75
    channel (e.g. a small reliable-lead pool, as in fast/irregular AFib) can
    be under-called as narrow/borderline even though its morphology is
    classic BBB. ``qrs_wide_ms`` is already computed for QT-lead filtering
    and is a looser consensus that catches this.

    Escalate only when qrs_wide_ms independently reaches definite BBB
    territory *and* the amplitude morphology corroborates a specific
    RBBB/LBBB call -- this keeps the escalation from firing on generic IVCD
    disagreement, which is much weaker evidence of a real effect.
    """
    qrs_width_class, bbb = _classify_qrs_width(qrs_ms, representative_leads, r_prime_by_lead)
    if (
        qrs_width_class != "bbb"
        and qrs_wide_ms is not None
        and qrs_ms is not None
        and qrs_wide_ms > qrs_ms
    ):
        wide_width_class, wide_bbb = _classify_qrs_width(qrs_wide_ms, representative_leads, r_prime_by_lead)
        if wide_width_class == "bbb" and wide_bbb in ("RBBB", "LBBB"):
            return wide_width_class, wide_bbb, True
    return qrs_width_class, bbb, False


def _classify_qtc(
    qtc_ms: Optional[float],
    st_depression_leads: Optional[Dict[str, float]] = None,
    rvh_class: Optional[str] = None,
    lvh_class: Optional[str] = None,
    bbb: Optional[str] = None,
    qrs_width_class: Optional[str] = None,
) -> Tuple[str, Optional[str]]:
    """Returns (qtc_class, electrolyte_hint).

    Suppression: a genuine ventricular conduction defect (BBB / non-specific
    IVCD) and confirmed RVH/LVH suppress the QTc prolongation statement.  A
    merely borderline IVCD (QRS 100–110 ms) is too mild to count and does not
    suppress.
    Electrolyte hints:
      - QTc < 310 ms → "hypercalcemia"
      - QTc > 520 ms + ST depression in ≥ 2 leads → "hypokalemia"
    """
    if qtc_ms is None:
        return "indeterminate", None

    # QTc classification
    if qtc_ms < QTC_SHORT_MS:
        qtc_class = "short"
    elif qtc_ms <= QTC_POSSIBLY_SHORT_MS:
        qtc_class = "possible_short"
    elif qtc_ms <= QTC_BORDERLINE_SHORT_MS:
        qtc_class = "borderline_short"
    elif qtc_ms > QTC_SIG_PROLONGED_MS:
        qtc_class = "significantly_prolonged"
    elif qtc_ms > QTC_PROLONGED_MS:
        qtc_class = "prolonged"
    elif qtc_ms > QTC_BORDERLINE_PROL_MS:
        qtc_class = "borderline_prolonged"
    else:
        qtc_class = "normal"

    # Suppression of prolongation by confounders.  Only confirmed hypertrophy
    # (probable/definite) suppresses — a weak "consider"-level RVH/LVH must not
    # mask a genuinely prolonged QTc — and a merely borderline IVCD (QRS
    # 100–110 ms) is too mild a conduction defect to count.
    vcd_active = bbb is not None and qrs_width_class != "borderline_ivcd"
    if qtc_class in ("borderline_prolonged", "prolonged", "significantly_prolonged"):
        if (
            vcd_active
            or rvh_class in ("probable", "definitive")
            or lvh_class in ("probable", "definite")
        ):
            qtc_class = "normal"

    # Electrolyte hints
    electrolyte_hint: Optional[str] = None
    if qtc_ms < QTC_HYPERCALCEMIA_MS:
        electrolyte_hint = "hypercalcemia"
    elif qtc_ms > QTC_SIG_PROLONGED_MS:
        n_dep = len(st_depression_leads) if st_depression_leads else 0
        any_dep_mv = any(
            v <= QTC_HYPOKALEMIA_ST_THR for v in (st_depression_leads or {}).values()
        )
        if n_dep >= 2 and any_dep_mv:
            electrolyte_hint = "hypokalemia"

    return qtc_class, electrolyte_hint


def _detect_wpw(
    global_features,
    representative_leads: Dict[str, RepresentativeLeadFeatures],
    beat_features: List[LeadBeatFeatures],
) -> bool:
    """WPW: short PR (<120 ms) + delta wave present + QRS ≥ 100 ms."""
    pr = global_features.pr_ms
    qrs = global_features.qrs_ms
    delta_any = any(bf.delta_present for bf in beat_features)
    if pr is None or qrs is None:
        return False
    return pr < PR_SHORT_MS and qrs >= WPW_QRS_MIN_MS and delta_any


# ============================================================================
# Step 4a – P-wave morphology / atrial enlargement
# ============================================================================

def _p_wave_morphology(
    representative_leads: Dict[str, RepresentativeLeadFeatures],
    p_dur_by_lead: Dict[str, Optional[float]],
    ptf_v1: Optional[float],
    is_pediatric: bool = False,
) -> Tuple[Optional[str], List[str], bool, bool, Optional[str]]:
    """
    Returns:
      (p_morphology_class, rae_leads, lae_suspected, lae_definite, ptf_v1_class)

    p_morphology_class: None | "normal" | "rae" | "lae" | "bae"
    """
    # ── RAE ──────────────────────────────────────────────────────────────────
    # DXL: BOTH P-duration ≥ 60 ms AND P-amplitude ≥ threshold must be exceeded.
    # Pediatric patients use a lower amplitude threshold (0.20 mV vs adult
    # 0.24 mV) per DXL_Threshold_Reference.xlsx, since children's P waves are
    # smaller. If P-duration data is unavailable for a lead we still count
    # amplitude alone (conservative fallback so data gaps don't suppress true RAE).
    rae_amp_consider_mv = PEDS_RAE_P_AMP_MV if is_pediatric else RAE_P_AMP_CONSIDER_MV
    rae_leads: List[str] = []
    rae_leads_confirmed: List[str] = []
    for lead in LIMB_LEADS_P:
        amp   = _lp(representative_leads, lead, "p_amp_mv", required_flags=("reliable_for_p",))
        p_dur = p_dur_by_lead.get(lead)
        if amp is None or amp < rae_amp_consider_mv:
            continue
        # Duration gate: if measured, must also be ≥ 60 ms
        if p_dur is not None and p_dur < RAE_P_DUR_MIN_MS:
            continue
        rae_leads.append(lead)
        # The "≥2 leads" confirmation tier requires the stricter 0.25 mV bar
        # (RAE_P_AMP_CONFIRM_MV), not the lower "consider" threshold.
        if amp >= RAE_P_AMP_CONFIRM_MV:
            rae_leads_confirmed.append(lead)

    # V1 positive biphasic terminal (RAE pattern in V1)
    v1_p_amp = _lp(representative_leads, "V1", "p_amp_mv", required_flags=("reliable_for_p",))
    v1_biphasic = _lead_param_bool(
        representative_leads,
        "V1",
        "p_biphasic",
        required_flags=("reliable_for_p",),
    )
    v1_terminal_amp = _lp(representative_leads, "V1", "p_terminal_amp_mv", required_flags=("reliable_for_p",))
    if v1_terminal_amp is None:
        v1_terminal_amp = _lp(
            representative_leads,
            "V1",
            "p_terminal_amplitude_mV",
            required_flags=("reliable_for_p",),
        )
    v1_terminal_dur = _lp(representative_leads, "V1", "p_terminal_duration_ms", required_flags=("reliable_for_p",))
    v1_terminal_area = _lp(representative_leads, "V1", "p_terminal_area_mv_ms", required_flags=("reliable_for_p",))
    v1_negative_terminal = (
        (v1_terminal_amp is not None and v1_terminal_amp < 0.0)
        or (v1_terminal_area is not None and v1_terminal_area < 0.0)
    )
    v1_positive_terminal = (
        (v1_terminal_amp is not None and v1_terminal_amp > 0.0)
        or (v1_terminal_area is not None and v1_terminal_area > 0.0)
    )
    v1_rae = (
        (v1_p_amp is not None and v1_p_amp >= RAE_V1_BIPHASIC_MV)
        or (v1_biphasic and v1_positive_terminal)
    )

    # The classic criterion is a single lead: P >= 0.25 mV (2.5 mm) in II, tall
    # rather than wide, which is what this codebase's authoritative layer applies
    # (`clinical_rules/hypertrophy.py`: `p_ii > 0.25 and p_ii_duration <= 120`).
    # Requiring two limb leads is a stricter DXL tier, and on PTB-XL 09000 it
    # dropped real RAE evidence: record 09237 measures 264 µV in II with definite
    # LAE, i.e. biatrial, and was reported as plain LAE because aVF stopped at
    # 206 µV.  Accept the single-lead criterion too so the two engines agree.
    rae_from_lead_ii = False
    _ii_amp = _lp(representative_leads, "II", "p_amp_mv", required_flags=("reliable_for_p",))
    _ii_dur = p_dur_by_lead.get("II")
    # A *measured* duration is required: RAE is "tall and not wide", so with the
    # width unknown the criterion is not satisfied -- which is also how
    # `clinical_rules/hypertrophy.py` treats a missing `p_dur_consensus_ms`.
    if _ii_amp is not None and _ii_amp >= RAE_P_AMP_CONFIRM_MV:
        rae_from_lead_ii = _ii_dur is not None and _ii_dur <= RAE_P_DUR_MAX_MS

    rae_confirmed = (
        len(rae_leads_confirmed) >= 2
        or (len(rae_leads) >= 1 and v1_rae)
        or rae_from_lead_ii
    )

    # ── LAE ──────────────────────────────────────────────────────────────────
    # PTF-V1 based
    ptf_v1_class: Optional[str] = None
    lae_from_ptf = False
    # Only Morris' published bar (-4 mV·ms) carries a record-level LAE call.
    # PTF_V1_PROBABLE_MV_MS is half of it -- a borderline zone this file invented,
    # not a criterion from Morris or from DXL_Threshold_Reference.xlsx -- and on
    # the PTB-XL 09000 shard the two bars are 19.7% and 43.4% of records
    # respectively, so the invented half-bar was contributing more record-level
    # LAE calls than the real criterion.  It stays as a PTF *grading* in
    # `ptf_v1_class` (that field answers "how big is the terminal force", which
    # is worth reporting); it just no longer decides `lae_suspected` on its own.
    # Plotting the V1 P waves first is what kept this from becoming a
    # measurement change: the deep terminal components are real deflections
    # (median 80 µV × 78 ms at the definite bar, and a P-free baseline moves PTF
    # by <= 23 µV), so the looseness is in the borrowed threshold, not the
    # integral.
    if ptf_v1 is not None and _lead_has_flags(representative_leads, "V1", "reliable_for_p"):
        if ptf_v1 <= PTF_V1_DEFINITE_MV_MS:
            ptf_v1_class = (
                "definite_lae"
                if _v1_terminal_negative_component_clears_lae_bars(representative_leads)
                else "probable_lae"
            )
            lae_from_ptf = True
        elif ptf_v1 <= PTF_V1_PROBABLE_MV_MS:
            ptf_v1_class = "probable_lae"
        else:
            ptf_v1_class = "normal"

    # P-duration + amplitude joint criterion (DXL: both must exceed threshold in same lead)
    lae_from_dur = False
    for _ld in LIMB_LEADS_P:
        if not _lead_has_flags(representative_leads, _ld, "reliable_for_p"):
            continue
        _pdur = p_dur_by_lead.get(_ld)
        if _pdur is None or _pdur < LAE_P_DUR_MS:
            continue
        _pamp = _lp(representative_leads, _ld, "p_amp_mv")
        # Both halves of the DXL criterion are required, and the amplitude must
        # actually have been measured.  A missing `p_amp_mv` no longer means "not
        # computed": `features._p_amplitude_if_measurable` withholds it when the P
        # is no larger than the drift it sits on, and accepting duration alone
        # there turned unmeasurability into a positive finding -- it pushed
        # lae_suspected from 387 to 406 records and lae_definite from 98 to 111 on
        # PTB-XL 09000, in the same direction as the amplitude being large.
        if _pamp is not None and abs(_pamp) >= LAE_P_AMP_LIMB_MV:
            lae_from_dur = True
            break

    # Notched-P criterion, restricted to the leads it is defined in.  Scanning
    # all 12 leads with `any()` turned a measure whose per-lead MAE is 28.5 ms
    # (LUDB) into a max-of-12: the triggering leads were aVL/III/I/V4 rather than
    # II, the max-of-12 duration median was 110 ms against 82 ms in lead II, and
    # the path fired on 40.3% of records / 29.8% of NORM-labelled ones.
    lae_from_notched_duration = any(
        _lead_param_bool(representative_leads, lead, "p_notched", required_flags=("reliable_for_p",))
        and (p_dur_by_lead.get(lead) is not None and p_dur_by_lead[lead] >= LAE_P_DUR_MS)
        and (
            (_notch_interval := _lp(
                representative_leads,
                lead,
                "p_notch_interval_ms",
                required_flags=("reliable_for_p",),
            )) is not None
            and _notch_interval >= LAE_P_NOTCH_INTERVAL_MS
        )
        for lead in LIMB_LEADS_P
    )
    # The bare "biphasic P with a negative terminal" observation matched 87.1%
    # of PTB-XL 09000 records; require the negative component to reach the DXL
    # amplitude+duration or negative-area bars before it counts.
    lae_from_v1_biphasic = bool(
        v1_biphasic
        and v1_negative_terminal
        and _v1_terminal_negative_component_supports_lae_suspicion(representative_leads)
    )
    lae_from_fine_morphology = lae_from_notched_duration or lae_from_v1_biphasic

    # V1 terminal-negative amplitude + duration grading, independent of the
    # PTF-V1 area-based grading above (DXL_Threshold_Reference.xlsx: probable
    # -0.10 mV with terminal duration > 50 ms; definite -0.15 mV with
    # terminal duration > 60 ms).
    v1_amp_dur_class: Optional[str] = None
    if v1_terminal_amp is not None and v1_terminal_dur is not None:
        if v1_terminal_amp <= -LAE_V1_NEG_DEF_MV and v1_terminal_dur > LAE_V1_NEG_DEF_DUR_MS:
            v1_amp_dur_class = "definite_lae"
        elif v1_terminal_amp <= -LAE_V1_NEG_PROB_MV and v1_terminal_dur > LAE_V1_NEG_PROB_DUR_MS:
            v1_amp_dur_class = "probable_lae"
    lae_from_v1_amp_dur = v1_amp_dur_class is not None

    lae_suspected = lae_from_ptf or lae_from_dur or lae_from_fine_morphology or lae_from_v1_amp_dur
    lae_definite  = (
        (ptf_v1_class == "definite_lae")
        or (lae_from_dur and lae_from_ptf)
        or (v1_amp_dur_class == "definite_lae")
    )

    # ── BAE ──────────────────────────────────────────────────────────────────
    # Direct BAE path independent of "RAE and LAE both separately confirmed":
    # V1 terminal negative ≤ -0.15 mV together with limb P > 0.30 mV in at
    # least 2 limb leads, qualified by a P duration ≥ 80 ms.  As with the RAE
    # duration gate, an unmeasured duration must not suppress the criterion --
    # only a measured duration below the bar disqualifies it.
    _bae_measured_limb_durs = [
        p_dur_by_lead[lead] for lead in LIMB_LEADS_P
        if p_dur_by_lead.get(lead) is not None
    ]
    bae_p_duration_qualified = (
        not _bae_measured_limb_durs
        or any(dur >= BAE_P_DUR_MIN_MS for dur in _bae_measured_limb_durs)
    )
    bae_limb_leads_gt_030 = [
        lead for lead in LIMB_LEADS_P
        if (_lp(representative_leads, lead, "p_amp_mv", required_flags=("reliable_for_p",)) or 0.0)
        > BAE_LIMB_P_AMP_MV
    ]
    bae_from_v1_and_limb = bool(
        v1_terminal_amp is not None
        and v1_terminal_amp <= -BAE_V1_NEG_AMP_MV
        and len(bae_limb_leads_gt_030) >= 2
        and bae_p_duration_qualified
    )
    bae_direct = bae_from_v1_and_limb

    # ── Overall classification ────────────────────────────────────────────────
    if (rae_confirmed and lae_suspected) or bae_direct:
        p_class: Optional[str] = "bae"
    elif lae_definite:
        p_class = "lae"
    elif lae_suspected:
        p_class = "probable_lae"
    elif rae_confirmed:
        p_class = "rae"
    elif len(rae_leads) >= 1 or v1_rae:
        p_class = "probable_rae"
    else:
        p_class = "normal"

    return p_class, rae_leads, lae_suspected, lae_definite, ptf_v1_class


# ============================================================================
# Step 4b – Q-wave analysis
# ============================================================================

def _pathological_q_waves(
    representative_leads: Dict[str, RepresentativeLeadFeatures],
    q_dur_by_lead: Dict[str, Optional[float]],
) -> Tuple[Dict[str, bool], List[str]]:
    """
    Per-lead pathological Q flag using amplitude/ratio criteria.
    Duration criteria applied when q_dur_by_lead is available.

    Territory grouping:
      inferior  – II, III, aVF
      anterior  – V1, V2, V3, V4
      lateral   – I, aVL, V5, V6

    A lead is only reported as pathological when >=2 leads within the same
    territory meet criteria. An isolated single-lead Q is a common normal
    variant (septal Q, positional Q in III/aVL) and is not diagnostic on its
    own (Minnesota code / AHA-ACC-HRS convention) -- this mirrors the >=2-lead
    threshold mi.py already uses for its q_wave_mi_pattern determination.
    """
    raw_flags: Dict[str, bool] = {}

    for lead in STANDARD_12_LEADS:
        q = _lp(representative_leads, lead, "q_amp_mv", required_flags=("reliable_for_qrs",))
        r = _lp(representative_leads, lead, "r_amp_mv", required_flags=("reliable_for_qrs",))
        q_dur = _lp(representative_leads, lead, "q_duration_ms", required_flags=("reliable_for_qrs",))
        if q_dur is None:
            q_dur = q_dur_by_lead.get(lead)

        if q is None or q >= 0:
            raw_flags[lead] = False
            continue

        abs_q = abs(q)
        abs_r = abs(r) if (r is not None and r != 0) else None

        flagged = False

        if lead in INFERIOR_LEADS:
            # |Q| > 1/6 R and duration ≥ 25 ms (DXL: significant), ≥ 35 ms (infarct)
            ratio_ok = (abs_r is not None and abs_q > Q_INF_R_RATIO * abs_r)
            dur_ok = (q_dur is None) or (q_dur >= 25.0)
            flagged = ratio_ok and dur_ok

        elif lead in LATERAL_LEADS:
            # |Q| > 0.10 mV AND > 20% R AND duration ≥ 35 ms
            amp_ok   = (abs_q > Q_LAT_AMP_MV)
            ratio_ok = (abs_r is not None and abs_q > Q_LAT_R_RATIO * abs_r)
            dur_ok   = (q_dur is None) or (q_dur >= 35.0)
            flagged  = amp_ok and ratio_ok and dur_ok

        elif lead in ANTERIOR_LEADS:
            # |Q| > 0.07 mV AND > 20% R AND duration ≥ 30 ms
            amp_ok   = (abs_q > Q_ANT_AMP_MV)
            ratio_ok = (abs_r is not None and abs_q > Q_ANT_R_RATIO * abs_r)
            dur_ok   = (q_dur is None) or (q_dur >= 30.0)
            flagged  = amp_ok and ratio_ok and dur_ok

        else:
            # aVR – not assessed for MI Q waves
            flagged = False

        raw_flags[lead] = flagged

    # Territory attribution: require >=2 corroborating leads per territory
    # before reporting either the territory or its member leads.
    path_q: Dict[str, bool] = {lead: False for lead in STANDARD_12_LEADS}
    territories: List[str] = []
    for territory, group in (
        ("inferior", INFERIOR_LEADS),
        ("anterior", ANTERIOR_LEADS),
        ("lateral", LATERAL_LEADS),
    ):
        hits = [l for l in group if raw_flags.get(l, False)]
        if len(hits) >= 2:
            for l in hits:
                path_q[l] = True
            territories.append(territory)

    return path_q, territories


# ============================================================================
# Step 4c – R-wave progression (V1 → V6)
# ============================================================================

def _r_wave_progression(
    representative_leads: Dict[str, RepresentativeLeadFeatures],
) -> Tuple[str, Optional[str]]:
    """
    Classify R-wave progression across precordial leads.

    Returns (class, transition_lead):
      class: "normal" | "poor" | "reverse" | "indeterminate"
      transition_lead: first lead where R > S (the RS transition)
    """
    precordial_order = ["V1", "V2", "V3", "V4", "V5", "V6"]
    r_vals = [_lp(representative_leads, ld, "r_amp_mv", required_flags=("reliable_for_qrs",)) for ld in precordial_order]
    s_vals = [_lp(representative_leads, ld, "s_amp_mv", required_flags=("reliable_for_qrs",)) for ld in precordial_order]

    # Need at least V1–V4 to assess
    if any(v is None for v in r_vals[:4]):
        return "indeterminate", None

    # R-S transition: first lead where R >= |S|
    transition_lead: Optional[str] = None
    for i, lead in enumerate(precordial_order):
        ri = r_vals[i]
        si = s_vals[i]
        if ri is None:
            continue
        si_abs = abs(si) if si is not None else 0.0
        if ri >= si_abs:
            transition_lead = lead
            break

    # Poor R-wave progression: R in V3 < 0.3 mV (or no transition by V4)
    r_v3 = r_vals[2]  # V3 index
    r_v4 = r_vals[3]  # V4 index
    no_transition_by_v4 = (transition_lead is None or
                           precordial_order.index(transition_lead) >= 4)

    # Reverse progression: R decreases from V1/V2 across V1→V6
    non_null = [(i, v) for i, v in enumerate(r_vals) if v is not None]
    reverse = False
    if len(non_null) >= 3:
        # Check if R decreases for ≥3 consecutive leads after V1
        v1_r = r_vals[0]
        if v1_r is not None and all(
            r_vals[i] is not None and r_vals[i] < v1_r  # type: ignore[operator]
            for i in range(1, min(4, len(r_vals)))
            if r_vals[i] is not None
        ):
            reverse = True

    if reverse:
        return "reverse", transition_lead
    if no_transition_by_v4 or (r_v3 is not None and r_v3 < 0.30):
        return "poor", transition_lead
    return "normal", transition_lead


# ============================================================================
# Step 5a – ST analysis, territory, reciprocal changes
# ============================================================================

def _t_upright_in_elevated_leads(
    representative_leads: Dict[str, RepresentativeLeadFeatures],
    leads: List[str],
    ele: Dict[str, float],
) -> bool:
    """T is treated as upright unless explicitly measured non-positive in one
    of the leads carrying the qualifying ST elevation. Leads where T-wave
    data is unavailable never block the call (missing companion evidence
    should not suppress an otherwise-supported finding)."""
    for lead in leads:
        if lead not in ele:
            continue
        t_amp = _lp(representative_leads, lead, "t_amp_mv", required_flags=("reliable_for_t",))
        if t_amp is not None and t_amp <= 0.0:
            return False
    return True


def _st_analysis(
    representative_leads: Dict[str, RepresentativeLeadFeatures],
    hr_bpm: Optional[float] = None,
    age: Optional[float] = None,
    exclude_leads: Tuple[str, ...] = (),
) -> Tuple[
    Dict[str, float],      # st_elevation_leads
    Dict[str, float],      # st_depression_leads
    List[str],             # st_territories_elevated
    List[str],             # st_territories_depressed
    List[str],             # stemi_suspected_codes
    bool,                  # reciprocal_change_detected
    List[List[str]],       # reciprocal_pairs
    bool,                  # st_rate_related
]:
    """
    Uses st_on_mv (J-point) as primary ST measurement.
    Codes are named after DXL_Threshold_Reference.xlsx STEMI_Critical_Values.
    Rate-related ST depression: HR > (190 − age) when age is known.
    exclude_leads: leads whose ST measurements are unreliable (e.g. precordial
    leads when a precordial reversal is detected); they are dropped from the
    elevation/depression sets so no territory, STEMI code, or reciprocal pair is
    derived from them.
    """
    excluded = set(exclude_leads)
    ele: Dict[str, float] = {}
    dep: Dict[str, float] = {}

    for lead in STANDARD_12_LEADS:
        if lead in excluded:
            continue
        st = _st_j_lp(representative_leads, lead)
        if st is None:
            continue
        if st >= ST_ELE_BORDERLINE_MV:
            ele[lead] = round(st, 4)
        elif st <= -ST_DEP_SIGNIFICANT_MV:
            dep[lead] = round(st, 4)

    # ── Territory classification ──────────────────────────────────────────────
    def _count_ele_in(leads: List[str], threshold: float) -> int:
        return sum(1 for ld in leads if ele.get(ld, 0.0) >= threshold)

    def _count_dep_in(leads: List[str], threshold: float = ST_DEP_SIGNIFICANT_MV) -> int:
        return sum(1 for ld in leads if dep.get(ld, 0.0) <= -abs(threshold))

    def _append_unique(dst: List[str], value: str) -> None:
        if value not in dst:
            dst.append(value)

    terr_ele: List[str] = []
    terr_dep: List[str] = []
    codes: List[str] = []

    # Inferior
    if _count_ele_in(INFERIOR_LEADS, ST_ELE_INFERIOR_MV) >= 2:
        _append_unique(terr_ele, "inferior")
        # The plain "acute" tier (IMIA, and its RCA/LCx refinements IMIAR/
        # IMIAX) requires an upright T wave per DXL_Threshold_Reference.xlsx;
        # without it the finding only supports the "probable" tier (IMIAP,
        # which carries no additional criteria of its own).
        if not _t_upright_in_elevated_leads(representative_leads, INFERIOR_LEADS, ele):
            codes.append("IMIAP")
        else:
            # Culprit differentiation hints (Excel: IMIAR / IMIAX)
            st_iii = _st_j_lp(representative_leads, "III") or 0.0
            st_ii  = _st_j_lp(representative_leads, "II") or 0.0
            if st_iii > st_ii:
                codes.append("IMIAR")   # ST III > ST II → RCA
            elif _count_dep_in(["V1", "V2", "V3"]) >= 2:
                codes.append("IMIAX")   # ST dep V1-V3 → LCx
            else:
                codes.append("IMIA")

    # Anteroseptal
    if _count_ele_in(ANTEROSEPTAL_LEADS, ST_ELE_ANTS_MV) >= 2:
        _append_unique(terr_ele, "anteroseptal")
        codes.append("ASMIA")
    elif _count_ele_in(ANTEROSEPTAL_LEADS, ST_ELE_ANTERIOR_PROB_MV) >= 2:
        _append_unique(terr_ele, "anteroseptal")
        # ASMIAP (probable tier) requires an upright T wave per the spreadsheet.
        if _t_upright_in_elevated_leads(representative_leads, ANTEROSEPTAL_LEADS, ele):
            codes.append("ASMIAP")

    # Anterior
    if _count_ele_in(["V2", "V3", "V4", "V5"], ST_ELE_ANTERIOR_MV) >= 2:
        _append_unique(terr_ele, "anterior")
        codes.append("AMIA")
    elif _count_ele_in(["V2", "V3", "V4", "V5"], ST_ELE_ANTERIOR_PROB_MV) >= 2:
        _append_unique(terr_ele, "anterior")
        # AMIAP (probable tier) requires an upright T wave per the spreadsheet.
        if _t_upright_in_elevated_leads(representative_leads, ["V2", "V3", "V4", "V5"], ele):
            codes.append("AMIAP")

    # Extensive anterior (V1–V6)
    if _count_ele_in(EXTENSIVE_ANT_LEADS, ST_ELE_EXTENSIVE_MV) >= 4:
        if "anterior" not in terr_ele:
            terr_ele.append("anterior")
        codes.append("EAMIA")

    # Lateral
    if _count_ele_in(LATERAL_LEADS, ST_ELE_LATERAL_MV) >= 2:
        _append_unique(terr_ele, "lateral")
        codes.append("LMIA")

    # Anterolateral
    if (
        _count_ele_in(["V2", "V3", "V4", "V5", "V6"], ST_ELE_ANTERIOR_PROB_MV) >= 2
        and _count_ele_in(LATERAL_LEADS, ST_ELE_LATERAL_MV) >= 2
    ):
        if "anterolateral" not in terr_ele:
            terr_ele.append("anterolateral")
        codes.append("ALIA")

    # Posterior (>=2 of V1–V3 with ST depression of at least 0.10 mV)
    if _count_dep_in(POSTERIOR_LEADS, abs(ST_DEP_POSTERIOR_MV)) >= 2:
        _append_unique(terr_dep, "posterior_reciprocal")
        codes.append("PMIA")

    # Depression territories
    if _count_dep_in(INFERIOR_LEADS) >= 2:
        _append_unique(terr_dep, "inferior")
    if _count_dep_in(ANTERIOR_LEADS) >= 2:
        _append_unique(terr_dep, "anterior")
    if _count_dep_in(LATERAL_LEADS) >= 2:
        _append_unique(terr_dep, "lateral")

    if not terr_ele:
        ele = {
            lead: value
            for lead, value in ele.items()
            if value >= ST_ELE_ABNORMAL_MV
        }

    # ── Reciprocal changes ────────────────────────────────────────────────────
    # Reciprocal pairs: (elevated_group, expected_reciprocal_depression_group)
    _reciprocal_map = [
        (INFERIOR_LEADS,      HIGH_LATERAL_LEADS),    # inferior ↑ → high-lateral ↓
        (HIGH_LATERAL_LEADS,  INFERIOR_LEADS),        # high-lateral ↑ → inferior ↓
        (ANTERIOR_LEADS,      INFERIOR_LEADS),        # anterior ↑ → inferior ↓
        (ANTEROSEPTAL_LEADS,  ["V5", "V6", "I", "aVL"]),  # anteroseptal ↑ → lateral ↓
        (INFERIOR_LEADS,      ANTERIOR_LEADS),        # inferior ↑ → ant ↓ (LCx)
    ]
    found_pairs: List[List[str]] = []
    for ele_group, dep_group in _reciprocal_map:
        elevated_leads = [ld for ld in ele_group if ele.get(ld, 0.0) >= ST_ELE_ABNORMAL_MV]
        depressed_leads = [ld for ld in dep_group if ld in dep]
        if elevated_leads and depressed_leads:
            found_pairs.append([
                f"+{','.join(elevated_leads)}",
                f"-{','.join(depressed_leads)}",
            ])

    reciprocal = len(found_pairs) > 0

    # ── Rate-related ST depression ─────────────────────────────────────────────
    # Flag when HR exceeds the age-adjusted tachycardia threshold and ST depression
    # is present in ≥ 2 leads — likely physiological rather than ischaemic.
    st_rate_related = False
    if hr_bpm is not None and age is not None and dep:
        rate_threshold = ST_RATE_RELATED_BASE - float(age)
        if hr_bpm > rate_threshold and len(dep) >= 2:
            st_rate_related = True

    return ele, dep, terr_ele, terr_dep, codes, reciprocal, found_pairs, st_rate_related


# ============================================================================
# Step 5b – LVH voltage criteria
# ============================================================================

def _lvh_criteria(
    representative_leads: Dict[str, RepresentativeLeadFeatures],
    qrs_ms: Optional[float],
    is_female: Optional[bool],
    age: Optional[float] = None,
    qrs_axis_class: str = "indeterminate",
    lae_suspected: bool = False,
    lv_strain: bool = False,
    bbb: Optional[str] = None,
) -> Tuple[List[str], Optional[str], bool]:
    """
    DXL LVH scoring.  Returns (criteria_met, lvh_class, secondary_repol_abnormality).
    lvh_class: None | "by_voltage" | "consider" | "probable" | "definite"

    Point rules (DXL_Threshold_Reference.xlsx LVH_Voltage sheet -- a decision
    tree, not a simple additive score):
      1 voltage criterion                      -> by_voltage (borderline)
      >=2 voltage criteria                      -> consider
      voltage + LAD, or voltage + LAE            -> probable
      >=2 voltage criteria + (LAD or LAE)        -> definite
      Cornell Product alone                     -> at least probable
      voltage + anterolateral LV strain          -> secondary_repol_abnormality
                                                     flag, reported alongside
                                                     whichever grade above applies
      QRS/VAT widening in V5 or V6 (no BBB)      -> weak corroborating evidence,
                                                     can only lift a lone voltage
                                                     criterion to "consider"

    Suppressions: age < 35 → None; right axis present → None
    """
    female = bool(is_female)

    # ── Age suppression ───────────────────────────────────────────────────────
    if age is not None and float(age) < LVH_AGE_MIN:
        return [], None, False

    # ── Right-axis suppression ────────────────────────────────────────────────
    if qrs_axis_class in ("RAD", "LPFB", "ERAD"):
        return [], None, False

    r_avl = _lp(representative_leads, "aVL", "r_amp_mv", required_flags=("reliable_for_qrs",))
    r_i   = _lp(representative_leads, "I",   "r_amp_mv", required_flags=("reliable_for_qrs",))
    s_iii = _lp(representative_leads, "III", "s_amp_mv", required_flags=("reliable_for_qrs",))
    r_v5  = _lp(representative_leads, "V5",  "r_amp_mv", required_flags=("reliable_for_qrs",))
    r_v6  = _lp(representative_leads, "V6",  "r_amp_mv", required_flags=("reliable_for_qrs",))
    s_v1  = _lp(representative_leads, "V1",  "s_amp_mv", required_flags=("reliable_for_qrs",))
    s_v2  = _lp(representative_leads, "V2",  "s_amp_mv", required_flags=("reliable_for_qrs",))
    s_v3  = _lp(representative_leads, "V3",  "s_amp_mv", required_flags=("reliable_for_qrs",))

    criteria: List[str] = []

    # ── Step 1: Voltage criteria ──────────────────────────────────────────────
    thr_r_avl = LVH_R_AVL_F if female else LVH_R_AVL_M
    if r_avl is not None and r_avl >= thr_r_avl:
        criteria.append("R_aVL")

    if r_i is not None and s_iii is not None and r_i + abs(s_iii) >= LVH_RI_SIII:
        criteria.append("RI_SIII")

    max_rv5v6 = max((v for v in [r_v5, r_v6] if v is not None), default=None)
    if max_rv5v6 is not None and max_rv5v6 >= LVH_R_V5V6:
        criteria.append("R_V5V6")

    s_v1v2 = max((abs(v) for v in [s_v1, s_v2] if v is not None), default=None)
    thr_sok = LVH_SOKOLOW_F if female else LVH_SOKOLOW_M
    if s_v1v2 is not None and max_rv5v6 is not None and s_v1v2 + max_rv5v6 >= thr_sok:
        criteria.append("Sokolow_Lyon")

    thr_cornell = LVH_CORNELL_F if female else LVH_CORNELL_M
    if r_avl is not None and s_v3 is not None and r_avl + abs(s_v3) >= thr_cornell:
        criteria.append("Cornell_Voltage")

    if r_avl is not None and s_v3 is not None and qrs_ms is not None:
        thr_prod = LVH_CORNELL_PROD_F if female else LVH_CORNELL_PROD_M
        if (r_avl + abs(s_v3)) * qrs_ms >= thr_prod:
            criteria.append("Cornell_Product")

    n_volt = len(criteria)
    if n_volt == 0:
        return criteria, None, False

    # Step 2: LAD (not LAFB — LAFB is a separate conduction diagnosis)
    lad_present = qrs_axis_class == "LAD"
    # Step 3: LAE
    lad_or_lae = lad_present or bool(lae_suspected)

    # Step 5: VAT prolonged in V5 or V6 (no BBB required). The source
    # spreadsheet's point rules don't tie this to probable/definite the way
    # LAD/LAE and Cornell Product are, so it is only ever weak corroborating
    # evidence for the borderline->consider step, never sufficient on its own
    # to reach probable/definite.
    vat_wide = False
    if bbb is None:
        vat_v5 = _lp(representative_leads, "V5", "vat_ms", required_flags=("reliable_for_qrs",))
        vat_v6 = _lp(representative_leads, "V6", "vat_ms", required_flags=("reliable_for_qrs",))
        vat_wide = bool(
            (vat_v5 is not None and vat_v5 > LVH_VAT_MS)
            or (vat_v6 is not None and vat_v6 > LVH_VAT_MS)
        )

    if n_volt >= 2 and lad_or_lae:
        lvh_class = "definite"
    elif lad_or_lae:
        lvh_class = "probable"
    elif n_volt >= 2 or vat_wide:
        lvh_class = "consider"
    else:
        lvh_class = "by_voltage"

    # Cornell Product alone → at least "probable", regardless of the rest of
    # the score.
    if "Cornell_Product" in criteria and lvh_class in ("by_voltage", "consider"):
        lvh_class = "probable"

    # Step 4: anterolateral LV strain is reported as its own distinct
    # complication label rather than folded into the consider/probable/
    # definite ladder.
    secondary_repol_abnormality = bool(lv_strain)

    return criteria, lvh_class, secondary_repol_abnormality


# ============================================================================
# Step 5c – Low voltage
# ============================================================================

def _low_voltage(
    representative_leads: Dict[str, RepresentativeLeadFeatures],
) -> Optional[str]:
    frontal_qrs = {
        ld: _qrs_peak_to_peak_mv(representative_leads, ld)
        for ld in FRONTAL_LEADS
    }
    precordial_qrs = {
        ld: _qrs_peak_to_peak_mv(representative_leads, ld)
        for ld in PRECORDIAL_LEADS
    }

    frontal_complete = all(v is not None for v in frontal_qrs.values())
    precordial_complete = all(v is not None for v in precordial_qrs.values())
    frontal_max = max(frontal_qrs.values()) if frontal_complete else None
    precordial_max = max(precordial_qrs.values()) if precordial_complete else None

    if precordial_max is not None and precordial_max < LOW_V_PRECORDIAL_DEF:
        return "precordial_definite"
    if frontal_max is not None and frontal_max < LOW_V_FRONTAL_DEF:
        return "frontal_definite"
    if frontal_max is not None and frontal_max < LOW_V_FRONTAL_BORDER:
        return "frontal_borderline"
    return None


# ============================================================================
# Step 5d – RVH (graded: Consider / Probable / Definitive)
# ============================================================================

def _rvh_scored(
    representative_leads: Dict[str, RepresentativeLeadFeatures],
    r_prime_by_lead: Dict[str, Optional[float]],
    rae_confirmed: bool,
    qrs_axis_class: str,
    r_prime_dur_by_lead: Optional[Dict[str, Optional[float]]] = None,
) -> Optional[str]:
    """
    DXL graded RVH scoring.

    Voltage criteria (each counts 1 point unless noted):
      V1a – R/V1 ≥ 0.30 mV (prominent R)
      V1b – R > 75 % of (Q+S) in V1  (may overlap with V1a)
      V1c – QRS V1 net positive (R > Q+S)  → 2 pts (highly significant)
      V1d – R' in V1 ≥ 0.30 mV AND, when measured, R' duration ≥ 20 ms
      Lat – |Q|, |S|, or |S'| > 0.20 mV in I or V6, each with a ≥ 40 ms
            duration gate (applied only when that component's duration was
            actually measured)
      Lat2– QRS I or V6 net negative (S > R)  → 2 pts (highly significant)

    Non-voltage modifiers (+1 each):
      +rae   – RAE already confirmed
      +rad   – right axis deviation (RAD or LPFB)
      +repol – ST depression ≤ -0.05 mV AND T < 0 in ≥ 2 of {II, aVF, V1, V2, V3}

    Grading:
      total = 0           → None
      total ≥ 1 (volt ≥ 1) → "consider"
      total ≥ 3 or (volt ≥ 2 and total ≥ 2) → "probable"
      total ≥ 5           → "definitive"
    """
    r_v1 = _lp(representative_leads, "V1", "r_amp_mv", required_flags=("reliable_for_qrs",))
    q_v1 = _lp(representative_leads, "V1", "q_amp_mv", required_flags=("reliable_for_qrs",))
    s_v1 = _lp(representative_leads, "V1", "s_amp_mv", required_flags=("reliable_for_qrs",))
    r_v6 = _lp(representative_leads, "V6", "r_amp_mv", required_flags=("reliable_for_qrs",))
    s_v6 = _lp(representative_leads, "V6", "s_amp_mv", required_flags=("reliable_for_qrs",))
    r_i  = _lp(representative_leads, "I",  "r_amp_mv", required_flags=("reliable_for_qrs",))
    s_i  = _lp(representative_leads, "I",  "s_amp_mv", required_flags=("reliable_for_qrs",))
    q_i  = _lp(representative_leads, "I",  "q_amp_mv", required_flags=("reliable_for_qrs",))
    q_v6 = _lp(representative_leads, "V6", "q_amp_mv", required_flags=("reliable_for_qrs",))
    s_prime_i  = _lp(representative_leads, "I",  "s_prime_amp_mv", required_flags=("reliable_for_qrs",))
    s_prime_v6 = _lp(representative_leads, "V6", "s_prime_amp_mv", required_flags=("reliable_for_qrs",))
    r_prime_v1 = (r_prime_by_lead.get("V1")
                  if _lead_has_flags(representative_leads, "V1", "reliable_for_qrs") else None)
    r_prime_v1_dur = (r_prime_dur_by_lead or {}).get("V1")

    if r_v1 is None:
        return None

    qs_v1 = (abs(q_v1) if q_v1 is not None else 0.0) + (abs(s_v1) if s_v1 is not None else 0.0)

    volt_score = 0

    # V1a: R ≥ 0.30 mV
    if r_v1 >= RVH_R_V1_MIN_MV:
        volt_score += 1

    # V1b: R > 75 % of Q+S (only adds if V1a didn't fire — avoids double counting)
    if volt_score == 0 and qs_v1 > 0 and r_v1 / qs_v1 > RVH_QS_RATIO:
        volt_score += 1

    # V1c: net positive V1 → highly significant (2 pts)
    if qs_v1 > 0 and r_v1 > qs_v1:
        volt_score += 1   # +1 on top of V1a/b

    # V1d: R' in V1 ≥ 0.30 mV, and when duration is measured it must also be ≥ 20 ms
    if (
        r_prime_v1 is not None
        and r_prime_v1 >= RVH_R_V1_MIN_MV
        and (r_prime_v1_dur is None or r_prime_v1_dur >= RVH_R_PRIME_DUR_MS)
    ):
        volt_score += 1

    # Lat: |Q|, |S|, or |S'| > 0.20 mV in I or V6, each gated on ≥ 40 ms
    # duration when that component's duration was actually measured.
    def _lat_component(amp: Optional[float], dur: Optional[float]) -> bool:
        return bool(
            amp is not None
            and abs(amp) > RVH_LAT_NEG_AMP_MV
            and (dur is None or dur >= RVH_LAT_DUR_MS)
        )

    q_dur_i  = _lp(representative_leads, "I",  "q_duration_ms", required_flags=("reliable_for_qrs",))
    q_dur_v6 = _lp(representative_leads, "V6", "q_duration_ms", required_flags=("reliable_for_qrs",))
    s_dur_i  = _lp(representative_leads, "I",  "s_duration_ms", required_flags=("reliable_for_qrs",))
    s_dur_v6 = _lp(representative_leads, "V6", "s_duration_ms", required_flags=("reliable_for_qrs",))
    s_prime_dur_i  = _lp(representative_leads, "I",  "s_prime_duration_ms", required_flags=("reliable_for_qrs",))
    s_prime_dur_v6 = _lp(representative_leads, "V6", "s_prime_duration_ms", required_flags=("reliable_for_qrs",))

    lat_neg_amp = (
        _lat_component(s_i, s_dur_i)
        or _lat_component(s_v6, s_dur_v6)
        or _lat_component(q_i, q_dur_i)
        or _lat_component(q_v6, q_dur_v6)
        or _lat_component(s_prime_i, s_prime_dur_i)
        or _lat_component(s_prime_v6, s_prime_dur_v6)
    )
    if lat_neg_amp:
        volt_score += 1

    # Lat2: net negative I or V6 → highly significant
    net_neg_lat = (
        (r_i  is not None and s_i  is not None and abs(s_i)  > r_i) or
        (r_v6 is not None and s_v6 is not None and abs(s_v6) > r_v6)
    )
    if net_neg_lat:
        volt_score += 1   # +1 on top of Lat

    if volt_score == 0:
        return None

    # Non-voltage modifiers
    repol_count = 0
    for _rl in ["II", "aVF", "V1", "V2", "V3"]:
        _st = _st_j_lp(representative_leads, _rl)
        _t  = _lp(representative_leads, _rl, "t_amp_mv",  required_flags=("reliable_for_t",))
        if _st is not None and _t is not None and _st <= RVH_REPOL_ST_MV and _t < 0:
            repol_count += 1
    repol_abn = repol_count >= 2

    nv = int(rae_confirmed) + int(qrs_axis_class in ("RAD", "LPFB", "ERAD")) + int(repol_abn)
    total = volt_score + nv

    if total >= 5 or (volt_score >= 3 and nv >= 1):
        return "definitive"
    if total >= 3 or (volt_score >= 2 and total >= 2):
        return "probable"
    return "consider"


# ============================================================================
# Step 5e – Tall T
# ============================================================================

def _tall_t_leads(
    representative_leads: Dict[str, RepresentativeLeadFeatures],
) -> List[str]:
    tall: List[str] = []
    for lead in STANDARD_12_LEADS:
        t_amp = _lp(representative_leads, lead, "t_amp_mv", required_flags=("reliable_for_t",))
        if t_amp is None:
            continue
        if abs(t_amp) >= TALL_T_ABS_MV:
            tall.append(lead)
            continue
        # Relative criterion: T > 50% of peak-to-peak QRS
        r = _lp(representative_leads, lead, "r_amp_mv", required_flags=("reliable_for_qrs",)) or 0.0
        s = _lp(representative_leads, lead, "s_amp_mv", required_flags=("reliable_for_qrs",)) or 0.0
        qrs_pp = abs(r) + abs(s)
        if qrs_pp > 0 and abs(t_amp) >= TALL_T_REL_MV and abs(t_amp) > 0.5 * qrs_pp:
            tall.append(lead)
    return tall


# ============================================================================
# Step 2b – Advanced rhythm helpers (DXL Chapter 2)
# ============================================================================

def _aggregate_per_beat(
    beats: List[BeatAnnotation],
    beat_features: List[LeadBeatFeatures],
) -> Dict[int, Dict]:
    """
    Collect per-beat median QRS width, PR interval, and max P confidence
    aggregated across all leads for that beat.
    """
    by_bid: Dict[int, Dict] = {}
    for b in beats:
        by_bid[b.beat_id] = {
            "r_index": b.r_index,
            "rr_prev_ms": b.rr_prev_ms,
            "rr_next_ms": b.rr_next_ms,
            "_qrs": [],
            "_pr": [],
            "_pc": [],
        }
    for bf in beat_features:
        d = by_bid.get(bf.beat_id)
        if d is None:
            continue
        if bf.qrs_ms is not None and np.isfinite(bf.qrs_ms):
            d["_qrs"].append(float(bf.qrs_ms))
        if bf.pr_ms is not None and np.isfinite(bf.pr_ms):
            d["_pr"].append(float(bf.pr_ms))
        if np.isfinite(bf.p_confidence):
            d["_pc"].append(float(bf.p_confidence))
    for d in by_bid.values():
        d["qrs_ms_median"] = float(np.median(d["_qrs"])) if d["_qrs"] else None
        d["pr_ms_median"]  = float(np.median(d["_pr"]))  if d["_pr"]  else None
        d["p_conf_max"]    = float(np.max(d["_pc"]))     if d["_pc"]  else 0.0
    return by_bid


def _detect_premature_complexes(
    beats: List[BeatAnnotation],
    per_beat: Dict[int, Dict],
) -> Dict:
    """
    Classify each beat as Normal (N), Ventricular (V), Atrial (A), or Junctional (J).

    DXL criteria:
      Premature  – rr_prev < (1 − PVC_RR_SHORTENING_PCT/100) × median_RR
      VPC        – premature + wide QRS (≥ QRS_BBB_MS = 120 ms)
      APC        – premature + narrow QRS + P present (p_confidence ≥ 0.30)
      JPC        – premature + narrow QRS + no P wave (p_confidence < 0.30)
    """
    def _empty_result() -> Dict:
        return {
            "vpcs": [],
            "apcs": [],
            "jpcs": [],
            "pairs": False,
            "triplets": False,
            "nsVT": False,
            "beat_types": [],
            "premature_complexes": [],
        }

    if not beats:
        return _empty_result()

    rr_vals = [b.rr_prev_ms for b in beats
               if b.rr_prev_ms is not None and b.rr_prev_ms > 0]
    if not rr_vals:
        return _empty_result()

    mean_rr = float(np.median(rr_vals))
    prem_thr = mean_rr * (1.0 - PVC_RR_SHORTENING_PCT / 100.0)

    vpcs: List[int] = []
    apcs: List[int] = []
    jpcs: List[int] = []
    beat_types: List[str] = []

    for b in beats:
        d = per_beat.get(b.beat_id, {})
        rr_prev = b.rr_prev_ms
        qrs_ms  = d.get("qrs_ms_median")
        p_conf  = d.get("p_conf_max", 0.0)

        is_prem  = rr_prev is not None and rr_prev > 0 and rr_prev < prem_thr
        is_wide  = qrs_ms is not None and qrs_ms >= QRS_BBB_MS
        has_p    = p_conf >= 0.30

        if is_prem and is_wide:
            vpcs.append(b.beat_id)
            beat_types.append("V")
        elif is_prem and not is_wide and has_p:
            apcs.append(b.beat_id)
            beat_types.append("A")
        elif is_prem and not is_wide and not has_p:
            jpcs.append(b.beat_id)
            beat_types.append("J")
        else:
            beat_types.append("N")

    # Consecutive VPC runs → pairs, triplets, nsVT
    pairs = False
    triplets = False
    nsVT = False
    v_run = 0
    for i, bt in enumerate(beat_types):
        if bt == "V":
            v_run += 1
            if v_run >= 2:
                pairs = True
            if v_run >= 3:
                triplets = True
                # nsVT: ≥3 consecutive VPCs at rate > 100 bpm, < 30 beats sustained
                rr_run = [
                    beats[j].rr_prev_ms
                    for j in range(max(0, i - v_run + 1), i + 1)
                    if j < len(beats) and beats[j].rr_prev_ms is not None
                       and beats[j].rr_prev_ms > 0
                ]
                if rr_run:
                    rate = 60000.0 / float(np.median(rr_run))
                    if rate > 100 and v_run < 30:
                        nsVT = True
        else:
            v_run = 0

    pc_list: List[str] = []
    if vpcs:
        pc_list.append(f"VPC×{len(vpcs)}")
    if apcs:
        pc_list.append(f"APC×{len(apcs)}")
    if jpcs:
        pc_list.append(f"JPC×{len(jpcs)}")

    return {
        "vpcs": vpcs,
        "apcs": apcs,
        "jpcs": jpcs,
        "pairs": pairs,
        "triplets": triplets,
        "nsVT": nsVT,
        "beat_types": beat_types,
        "premature_complexes": pc_list,
    }


def _detect_beat_patterns(
    beat_types: List[str],
) -> Tuple[Optional[str], bool]:
    """
    Detect bigeminy (NV or NA alternation ≥2 pairs) and trigeminy (NNV or NNA ≥2 triplets).
    Returns (bigeminy_type, trigeminy).
    """
    def _count_bigem(seq: List[str], ptype: str) -> int:
        best = cur = 0
        i = 0
        while i < len(seq) - 1:
            if seq[i] == "N" and seq[i + 1] == ptype:
                cur += 1
                best = max(best, cur)
                i += 2
            else:
                cur = 0
                i += 1
        return best

    def _count_trigem(seq: List[str], ptype: str) -> int:
        best = cur = 0
        i = 0
        while i < len(seq) - 2:
            if seq[i] == "N" and seq[i + 1] == "N" and seq[i + 2] == ptype:
                cur += 1
                best = max(best, cur)
                i += 3
            else:
                cur = 0
                i += 1
        return best

    bigeminy_type: Optional[str] = None
    if _count_bigem(beat_types, "V") >= 2:
        bigeminy_type = "ventricular"
    elif _count_bigem(beat_types, "A") >= 2:
        bigeminy_type = "atrial"

    trigeminy = (
        _count_trigem(beat_types, "V") >= 2
        or _count_trigem(beat_types, "A") >= 2
    )
    return bigeminy_type, trigeminy


def _detect_pauses(
    beats: List[BeatAnnotation],
) -> Tuple[bool, Optional[float]]:
    """
    RR pause: any RR interval > RR_PAUSE_FACTOR × median_RR.
    Returns (pauses_detected, longest_pause_ms).
    """
    rr_vals = [b.rr_next_ms for b in beats
               if b.rr_next_ms is not None and b.rr_next_ms > 0]
    if len(rr_vals) < 3:
        return False, None
    mean_rr = float(np.median(rr_vals))
    thr = mean_rr * RR_PAUSE_FACTOR
    pauses = [rr for rr in rr_vals if rr > thr]
    if not pauses:
        return False, None
    return True, round(float(max(pauses)), 1)


def _detect_wenckebach(pr_vals: List[float]) -> bool:
    """
    Mobitz I heuristic: find ≥3 consecutive PR increases (≥5 ms each)
    followed by a PR drop of ≥20 ms (dropped beat resets PR).
    """
    if len(pr_vals) < 4:
        return False
    for i in range(len(pr_vals) - 3):
        w = pr_vals[i: i + 4]
        if all(w[j + 1] > w[j] + 5 for j in range(len(w) - 2)) and w[-1] < w[-2] - 20:
            return True
    return False


def _detect_avb_and_dissociation(
    beats: List[BeatAnnotation],
    per_beat: Dict[int, Dict],
    gf,
) -> Tuple[bool, bool, Optional[str]]:
    """
    Returns (complete_avb, av_dissociation, second_degree_avb_str).

    complete_avb     – ventricular rate < HR_COMPLETE_AVB_BPM (45 bpm) AND
                        independent evidence of AV dissociation (P waves
                        marching through the cycle, or an atrial rate
                        distinctly faster than the ventricular rate). A slow
                        rate alone is also seen in marked sinus bradycardia,
                        so rate is a necessary but not sufficient condition.
    av_dissociation  – PR range > 80 ms AND PR SD > 30 ms (P independent of QRS)
    second_degree    – "mobitz_i" (Wenckebach) detected from per-beat PR sequence
    """
    hr = gf.heart_rate_bpm

    # Per-beat median PR in beat order
    ordered_pr = []
    for b in beats:
        d = per_beat.get(b.beat_id, {})
        pr = d.get("pr_ms_median")
        if pr is not None and np.isfinite(pr) and 60 < pr < 600:
            ordered_pr.append(float(pr))

    av_dissociation = False
    if len(ordered_pr) >= 3:
        pr_std   = float(np.std(ordered_pr))
        pr_range = float(np.max(ordered_pr) - np.min(ordered_pr))
        av_dissociation = (pr_range > 80.0 and pr_std > 30.0)

    # Classic distinguishing feature of complete heart block vs. a merely
    # slow sinus rhythm: the atrial rate is distinctly faster than the
    # ventricular rate. Use a small margin so a trivial fallback (atrial
    # rate estimation failing and defaulting to the ventricular rate itself)
    # never counts as "faster".
    atrial_rate = getattr(gf, "atrial_rate_bpm", None)
    atrial_faster_than_ventricular = bool(
        hr is not None and atrial_rate is not None and atrial_rate > hr + 5.0
    )

    complete_avb = bool(
        hr is not None
        and hr < HR_COMPLETE_AVB_BPM
        and (av_dissociation or atrial_faster_than_ventricular)
    )

    second_degree: Optional[str] = None
    if len(ordered_pr) >= 4 and _detect_wenckebach(ordered_pr):
        second_degree = "mobitz_i"

    return complete_avb, av_dissociation, second_degree


# ============================================================================
# Step 0 – Dextrocardia (must be tested first; bypass all morphology if +ve)
# ============================================================================

def _detect_dextrocardia(
    gf,
    representative_leads: Dict[str, RepresentativeLeadFeatures],
) -> bool:
    """
    DXL: Dextrocardia when ALL of:
      - P-axis and QRS-axis deviated rightward in frontal plane
      - Horizontal-plane QRS directed rightward (R < |S| in V5 and V6)
      - Small QRS complexes in V5 and V6 (peak-to-peak < 0.50 mV)
    """
    p_axis  = gf.p_axis_deg
    qrs_axis = gf.qrs_axis_deg

    if p_axis is None or qrs_axis is None:
        return False

    p_norm   = ((float(p_axis)   + 180) % 360) - 180
    qrs_norm = ((float(qrs_axis) + 180) % 360) - 180

    p_rightward   = p_norm   > 90
    qrs_rightward = qrs_norm > 90

    def _r_less_than_s(lead: str) -> bool:
        r = _lp(representative_leads, lead, "r_amp_mv", required_flags=("reliable_for_qrs",))
        s = _lp(representative_leads, lead, "s_amp_mv", required_flags=("reliable_for_qrs",))
        if r is None or s is None:
            return False
        return r < abs(s)

    horiz_rightward = _r_less_than_s("V5") and _r_less_than_s("V6")
    small_v5v6      = (
        (_qrs_peak_to_peak_mv(representative_leads, "V5") or 999.0) < DEXTRO_V5V6_MAX_PP_MV
        and
        (_qrs_peak_to_peak_mv(representative_leads, "V6") or 999.0) < DEXTRO_V5V6_MAX_PP_MV
    )

    return p_rightward and qrs_rightward and horiz_rightward and small_v5v6


# ============================================================================
# Step 3b – LV strain, posterior MI R-dominance, COPD pattern helpers
# ============================================================================

def _lv_strain_present(
    representative_leads: Dict[str, RepresentativeLeadFeatures],
) -> bool:
    """
    LV strain (anterolateral): ST depression ≤ LVH_STRAIN_ST_MV AND T negative
    in ≥ 2 of {I, aVL, V5, V6}.
    """
    strain_leads = ["I", "aVL", "V5", "V6"]
    count = 0
    for ld in strain_leads:
        st = _st_j_lp(representative_leads, ld)
        t  = _lp(representative_leads, ld, "t_amp_mv",  required_flags=("reliable_for_t",))
        if st is not None and t is not None and st <= LVH_STRAIN_ST_MV and t < LVH_STRAIN_T_MV:
            count += 1
    return count >= 2


def _posterior_mi_r_dominant(
    representative_leads: Dict[str, RepresentativeLeadFeatures],
) -> bool:
    """
    Posterior MI R-dominant pattern: ≥ 2 of V1-V3 show dominant R (R > |S|),
    upright T (T > PMI_MIN_T_MV), and insignificant Q (|Q| < PMI_MAX_Q_MV).
    """
    count = 0
    for lead in POSTERIOR_LEADS:
        r = _lp(representative_leads, lead, "r_amp_mv", required_flags=("reliable_for_qrs",))
        s = _lp(representative_leads, lead, "s_amp_mv", required_flags=("reliable_for_qrs",))
        q = _lp(representative_leads, lead, "q_amp_mv", required_flags=("reliable_for_qrs",))
        t = _lp(representative_leads, lead, "t_amp_mv",  required_flags=("reliable_for_t",))
        if r is None:
            continue
        s_abs = abs(s) if s is not None else 0.0
        r_dominant = r > s_abs
        t_upright  = t is not None and t > PMI_MIN_T_MV
        q_insig    = q is None or abs(q) < PMI_MAX_Q_MV
        if r_dominant and t_upright and q_insig:
            count += 1
    return count >= 2


def _copd_pattern(
    low_v_class: Optional[str],
    gf,
    rae_confirmed: bool,
    qrs_axis_class: str,
) -> bool:
    """
    COPD pattern: low voltage (any class) + right-deviated P axis
    + rightward QRS axis + RAE.
    """
    if low_v_class is None:
        return False
    p_axis = gf.p_axis_deg
    p_rightward = p_axis is not None and float(p_axis) > 75
    qrs_rightward = qrs_axis_class in ("RAD", "LPFB", "ERAD")
    return p_rightward and qrs_rightward and rae_confirmed


# ============================================================================
# Pediatric helpers (DXL Chapter 4)
# ============================================================================

def _peds_lookup(table: List[Tuple], age_years: float) -> Optional[float]:
    """Return the value for the age row containing age_years, or None if out of range."""
    for low, high, val in table:
        if low <= age_years < high:
            return val
    return None


def _peds_classify_qrs_axis(
    qrs_axis_deg: Optional[float],
    age_years: float,
) -> str:
    """
    Age-adjusted QRS axis classification for pediatric patients (Tables 4-1 to 4-4).

    Uses 0–360° clockwise (from I+) for RAD zone comparison (values > 180° in the
    tables correspond to negative angles in standard ±180° notation).
    Returns one of: "normal" | "LAD" | "borderline_LAD" | "LAFB" |
                    "borderline_RAD" | "RAD" | "indeterminate"
    """
    if qrs_axis_deg is None:
        return "indeterminate"
    a = float(qrs_axis_deg)
    a_std = ((a + 180) % 360) - 180    # normalise to (−180, +180]
    a_360 = a_std % 360                # 0–360 clockwise: −144° → 216°, 131° → 131°

    lad_thr      = _peds_lookup(_PEDS_LAD_THRESHOLD,       age_years)
    blad_upper   = _peds_lookup(_PEDS_BLAD_UPPER,          age_years)
    rad_thr_360  = _peds_lookup(_PEDS_RAD_THRESHOLD_360,   age_years)
    brad_thr_360 = _peds_lookup(_PEDS_BRAD_THRESHOLD_360,  age_years)

    if lad_thr is None or rad_thr_360 is None:
        return _classify_qrs_axis(qrs_axis_deg)   # fall back to adult

    _RAD_ZONE_MAX = 269   # 269° in 0–360 ≈ −91° standard (upper bound of RAD zone)

    # RAD zone checks (in 0–360 notation)
    in_rad  = (rad_thr_360  <= a_360 <= _RAD_ZONE_MAX)
    in_brad = (brad_thr_360 is not None and brad_thr_360 <= a_360 < rad_thr_360)

    if in_rad:
        return "RAD"
    if in_brad:
        return "borderline_RAD"

    # LAD zone checks (in standard ±180° notation)
    if a_std < lad_thr:
        # Pediatric LAFB: axis between −60° and −90° (absent LBBB — caller checks BBB)
        if -90.0 <= a_std <= -60.0:
            return "LAFB"
        return "LAD"
    if blad_upper is not None and a_std < blad_upper:
        return "borderline_LAD"

    return "normal"


def _peds_classify_qrs_width(
    qrs_ms: Optional[float],
    age_years: float,
    representative_leads: Dict[str, RepresentativeLeadFeatures],
    r_prime_by_lead: Dict[str, Optional[float]],
    r_prime_dur_by_lead: Optional[Dict[str, Optional[float]]] = None,
) -> Tuple[str, Optional[str]]:
    """
    Pediatric age-adjusted QRS classification (Table 4-5, Chapter 4).
    Borderline IVCD: > 110% of normal limit.
    Non-specific IVCD: > 120% of normal limit (= BBB range).
    Pediatric RBBB R' criterion: R' ≥ 0.15 mV (vs adult 0.10 mV) and, when the
    R' duration is measured, R' ≥ 20 ms.
    """
    if qrs_ms is None:
        return "indeterminate", None

    normal_ms = _peds_lookup(_PEDS_QRS_NORMAL_MS, age_years)
    if normal_ms is None:
        return _classify_qrs_width(qrs_ms, representative_leads, r_prime_by_lead)

    borderline_ms = normal_ms * 1.10
    bbb_ms        = normal_ms * 1.20

    if qrs_ms <= normal_ms:
        return "normal", None

    v1_rprime = (
        r_prime_by_lead.get("V1")
        if _lead_has_flags(representative_leads, "V1", "reliable_for_qrs")
        else None
    )
    v1_r = _lp(representative_leads, "V1", "r_amp_mv", required_flags=("reliable_for_qrs",))
    i_r  = _lp(representative_leads, "I",  "r_amp_mv", required_flags=("reliable_for_qrs",))
    i_s  = _lp(representative_leads, "I",  "s_amp_mv", required_flags=("reliable_for_qrs",))
    v6_r = _lp(representative_leads, "V6", "r_amp_mv", required_flags=("reliable_for_qrs",))
    v6_s = _lp(representative_leads, "V6", "s_amp_mv", required_flags=("reliable_for_qrs",))

    # Pediatric RBBB: R' ≥ 0.15 mV (not 0.10 as in adult) and, when the R'
    # duration is measured, R' must also last ≥ 20 ms (Table 4-5).  When the
    # duration is unavailable we fall back to the amplitude-only criterion so a
    # missing measurement never suppresses an otherwise-valid RBBB.
    v1_rprime_dur = (r_prime_dur_by_lead or {}).get("V1")
    rbbb_v1 = (
        v1_rprime is not None
        and v1_rprime >= PEDS_RBBB_R_PRIME_MV
        and (v1_rprime_dur is None or v1_rprime_dur >= PEDS_RBBB_R_PRIME_DUR_MS)
    )
    rbbb_lat = (
        (i_s  is not None and i_s  < -RBBB_LATERAL_S_MIN_MV) or
        (v6_s is not None and v6_s < -RBBB_LATERAL_S_MIN_MV)
    )
    lbbb_v1  = v1_r is None or v1_r <= LBBB_V1_R_MAX_MV
    lbbb_lat = (
        (v6_r is not None and v6_r > 0.30) or
        (i_r  is not None and i_r  > 0.10)
    )

    if qrs_ms > bbb_ms:
        if rbbb_v1 and rbbb_lat:
            return "bbb", "RBBB"
        if lbbb_v1 and lbbb_lat:
            return "bbb", "LBBB"
        return "bbb", "IVCD"

    if qrs_ms > borderline_ms:
        if rbbb_v1:
            return "nonspecific_ivcd", "incomplete_RBBB"
        return "nonspecific_ivcd", "IVCD"

    return "borderline_ivcd", None


def _peds_dextrocardia(
    gf: "GlobalFeatures",
    representative_leads: Dict[str, RepresentativeLeadFeatures],
) -> bool:
    """
    Pediatric dextrocardia criteria (Chapter 4 — differs from adult):
    • Frontal P axis between 90° and 180°
    • Lead I or V6 has a negative P wave
    • Leads I AND V6 have a large S wave (> 0.6 mV)
    • P wave amplitude in Lead III > Lead II
    Returns True if ≥ 3 of 4 criteria are met.
    """
    # 1. P axis 90°–180°
    p_axis_ok = False
    if gf.p_axis_deg is not None:
        p_ax = ((float(gf.p_axis_deg) + 180) % 360) - 180
        p_axis_ok = PEDS_DEXTRO_P_AXIS_LOW <= p_ax <= PEDS_DEXTRO_P_AXIS_HIGH

    # 2. Negative P in I or V6
    p_i  = _lp(representative_leads, "I",  "p_amp_mv", required_flags=("reliable_for_p",))
    p_v6 = _lp(representative_leads, "V6", "p_amp_mv", required_flags=("reliable_for_p",))
    neg_p = (p_i is not None and p_i < 0) or (p_v6 is not None and p_v6 < 0)

    # 3. Large S in both I and V6
    s_i  = _lp(representative_leads, "I",  "s_amp_mv", required_flags=("reliable_for_qrs",))
    s_v6 = _lp(representative_leads, "V6", "s_amp_mv", required_flags=("reliable_for_qrs",))
    large_s = (
        s_i  is not None and abs(s_i)  > PEDS_DEXTRO_S_MV and
        s_v6 is not None and abs(s_v6) > PEDS_DEXTRO_S_MV
    )

    # 4. P amplitude III > II
    p_ii  = _lp(representative_leads, "II",  "p_amp_mv", required_flags=("reliable_for_p",))
    p_iii = _lp(representative_leads, "III", "p_amp_mv", required_flags=("reliable_for_p",))
    p_iii_gt_ii = p_ii is not None and p_iii is not None and p_iii > p_ii

    return sum([p_axis_ok, neg_p, large_s, p_iii_gt_ii]) >= 3


def _peds_lsh(
    representative_leads: Dict[str, RepresentativeLeadFeatures],
) -> Optional[str]:
    """
    Left Septal Hypertrophy (Chapter 4):
    • Prominent R in V1 (≥ PEDS_LSH_R_V1_MV) + Q in V5 AND V6 → "lsh"
    • Moderate R in V1 (≥ PEDS_LSH_CONSIDER_R_V1_MV) + Q in V5 AND V6 → "consider_lsh"
    """
    r_v1 = _lp(representative_leads, "V1", "r_amp_mv", required_flags=("reliable_for_qrs",))
    q_v5 = _lp(representative_leads, "V5", "q_amp_mv", required_flags=("reliable_for_qrs",))
    q_v6 = _lp(representative_leads, "V6", "q_amp_mv", required_flags=("reliable_for_qrs",))
    if r_v1 is None:
        return None
    has_q = (q_v5 is not None and q_v5 < 0) and (q_v6 is not None and q_v6 < 0)
    if not has_q:
        return None
    if r_v1 >= PEDS_LSH_R_V1_MV:
        return "lsh"
    if r_v1 >= PEDS_LSH_CONSIDER_R_V1_MV:
        return "consider_lsh"
    return None


def _peds_classify_qtc(
    qtc_ms: Optional[float],
    age_years: float,
    is_female: Optional[bool],
    st_depression_leads: Optional[Dict[str, float]] = None,
    positive_t_count: int = 0,
    rvh_class: Optional[str] = None,
    lvh_class: Optional[str] = None,
    lsh: Optional[str] = None,
    bvh: bool = False,
    bbb: Optional[str] = None,
    qrs_width_class: Optional[str] = None,
) -> Tuple[str, Optional[str]]:
    """
    Pediatric QTc classification (Chapter 4) with age/sex-dependent thresholds.

    Returns (qtc_class, electrolyte_hint).
    electrolyte_hint: None | "hypercalcemia" | "hypocalcemia" | "hypokalemia"

    Suppression: RVH, LSH, LVH, BVH, VCD (BBB) suppress prolongation statements.
    Electrolyte hints:
      QTc < 310 ms → hypercalcemia
      QTc > 520 ms → hypocalcemia (base)
      QTc > 520 ms + ST dep + positive T in ≥2 leads → hypokalemia (overrides hypocalcemia)
    """
    if qtc_ms is None:
        return "indeterminate", None

    # Determine age/sex-adjusted thresholds
    female = bool(is_female)
    if age_years < 5.0:
        b_prol = PEDS_QTC_BPROL_U5_MS
        prol   = PEDS_QTC_PROL_U5_MS
    elif age_years < 13.0:
        b_prol = PEDS_QTC_BPROL_5TO12_MS
        prol   = PEDS_QTC_PROL_5TO12_MS
    else:
        b_prol = PEDS_QTC_BPROL_GIRL_MS if female else PEDS_QTC_BPROL_BOY_MS
        prol   = PEDS_QTC_PROL_GIRL_MS  if female else PEDS_QTC_PROL_BOY_MS

    # Classification
    if qtc_ms < PEDS_QTC_SHORT_MS:
        qtc_class = "borderline_short"
    elif qtc_ms > QTC_SIG_PROLONGED_MS:
        qtc_class = "significantly_prolonged"
    elif qtc_ms > prol:
        qtc_class = "prolonged"
    elif qtc_ms > b_prol:
        qtc_class = "borderline_prolonged"
    else:
        qtc_class = "normal"

    # Suppression by VCD, RVH, LSH, LVH, BVH.  A merely borderline IVCD (mild)
    # and a weak "consider"-level RVH/LVH are too soft to mask a prolonged QTc.
    vcd_present = bbb is not None and qrs_width_class != "borderline_ivcd"
    wide_qrs_suppressor = (
        vcd_present
        or rvh_class in ("probable", "definitive")
        or lsh is not None
        or bvh
        or lvh_class in ("probable", "definitive")
    )
    if qtc_class in ("borderline_prolonged", "prolonged", "significantly_prolonged") and wide_qrs_suppressor:
        qtc_class = "normal"

    # Electrolyte hints
    hint: Optional[str] = None
    if qtc_ms < QTC_HYPERCALCEMIA_MS:
        hint = "hypercalcemia"
    elif qtc_ms > PEDS_QTC_HYPOCALCEMIA_MS:
        n_dep = sum(1 for v in (st_depression_leads or {}).values() if v <= QTC_HYPOKALEMIA_ST_THR)
        if n_dep >= 2 and positive_t_count >= 2:
            hint = "hypokalemia"
        else:
            hint = "hypocalcemia"

    return qtc_class, hint


# ============================================================================
# Step 1 – Lead reversal flags (already in metadata / quality)
# ============================================================================

def _lead_reversal_flags(
    features: ECGFeatures,
) -> Tuple[Optional[str], bool]:
    """Extract limb and precordial reversal flags."""
    lead_reversal = features.metadata.get("lead_reversal", {})
    if isinstance(lead_reversal.get("limb"), dict):
        limb_reversal = lead_reversal.get("limb", {})
        precordial_rev = bool(lead_reversal.get("precordial", {}).get("suspected"))
    else:
        limb_reversal = lead_reversal
        precordial_rev = False
    limb_msg: Optional[str] = None
    if limb_reversal:
        # Only the boolean swap verdicts belong in a reversal message. The
        # detector also reports the Einthoven residual, which is a calibration
        # measurement and would otherwise read as another swap flag.
        parts = [f"{k}={v}" for k, v in limb_reversal.items() if v is True]
        if parts:
            limb_msg = "; ".join(parts)
    if not precordial_rev:
        precordial_rev = any(
            rep.params.get("probable_precordial_reversal", False)
            for rep in features.representative_leads.values()
        )
    return limb_msg, precordial_rev


# ============================================================================
# Public entry point
# ============================================================================

def _suppress_pacing_unreliable_rhythm_claims(
    interpretation: object,
    pacing_context: Dict[str, object],
) -> None:
    stop_rhythm = pacing_context.get(
        "suppress_further_rhythm_interpretation"
    ) is True
    ventricular_pacing = any(
        pacing_context.get(name) is True
        for name in (
            "ventricular_pacing_present",
            "dual_chamber_pacing_present",
            "wide_qrs_pacing_like_context",
        )
    )
    if not stop_rhythm and not ventricular_pacing:
        return
    # Even intermittent ventricular pacing invalidates V1/QRS-axis RVH
    # morphology. Do this independently of the stronger rhythm-stop gate.
    if hasattr(interpretation, "rvh_suspected"):
        setattr(interpretation, "rvh_suspected", False)
    if hasattr(interpretation, "rvh_class"):
        setattr(interpretation, "rvh_class", None)
    if not stop_rhythm:
        return
    for field_name in (
        "complete_av_block",
        "av_dissociation",
        "probable_af",
        "wpw_pattern",
        "digitalis_effect_suspected",
        "nonspecific_t_abnormality",
    ):
        if hasattr(interpretation, field_name):
            setattr(interpretation, field_name, False)
    if hasattr(interpretation, "avb_grade"):
        setattr(interpretation, "avb_grade", None)
    clear_to_empty_dict = (
        "pathological_q_leads",
        "st_elevation_leads",
        "st_depression_leads",
    )
    for field_name in clear_to_empty_dict:
        if hasattr(interpretation, field_name):
            setattr(interpretation, field_name, {})
    clear_to_empty_list = (
        "q_wave_territories",
        "st_territories_elevated",
        "st_territories_depressed",
        "stemi_suspected_codes",
        "reciprocal_pairs",
        "tall_t_leads",
    )
    for field_name in clear_to_empty_list:
        if hasattr(interpretation, field_name):
            setattr(interpretation, field_name, [])
    if hasattr(interpretation, "reciprocal_change_detected"):
        setattr(interpretation, "reciprocal_change_detected", False)
    if hasattr(interpretation, "t_axis_class"):
        setattr(interpretation, "t_axis_class", "indeterminate")
    if hasattr(interpretation, "qrs_t_angle_deg"):
        setattr(interpretation, "qrs_t_angle_deg", None)


def _apply_rule_summary_to_interpretation(
    interpretation: object,
    rule_summary: Dict[str, object],
    pacing_context: Dict[str, object],
) -> None:
    """Apply rule-summary evidence without downgrading existing positives."""
    if not isinstance(rule_summary, dict):
        return

    preexcitation = rule_summary.get("preexcitation") or {}
    if isinstance(preexcitation, dict) and bool(preexcitation.get("wpw_pattern", False)):
        interpretation.wpw_pattern = True

    pauses = rule_summary.get("pauses") or {}
    if not isinstance(pauses, dict):
        return

    if bool(pauses.get("pauses_detected", False)):
        interpretation.pauses_detected = True
        pause_longest_ms = pauses.get("pause_longest_ms")
        if pause_longest_ms is not None:
            interpretation.pause_longest_ms = pause_longest_ms

    second_degree_avb = pauses.get("second_degree_avb")
    if second_degree_avb:
        interpretation.second_degree_avb = str(second_degree_avb)
        if (
            not bool(pacing_context.get("suppress_further_rhythm_interpretation", False))
            and getattr(interpretation, "pr_class", None) != "normal"
        ):
            interpretation.avb_grade = 2


def _apply_measurement_availability_to_interpretation(
    interpretation: object,
    availability: Dict[str, object],
) -> None:
    """Mask rhythm measurements that are unavailable in the current rhythm context."""
    if not isinstance(availability, dict):
        return
    if not bool(availability.get("pr_available", True)):
        interpretation.pr_class = "indeterminate"
        interpretation.avb_grade = None
    if not bool(availability.get("p_axis_available", True)):
        interpretation.p_axis_normal = None
    if not bool(availability.get("atrial_rhythm_available", True)):
        interpretation.p_morphology_class = None
        interpretation.rae_leads = []
        interpretation.lae_suspected = False
        interpretation.lae_definite = False
        interpretation.ptf_v1_class = None
        reasons = set(availability.get("reasons") or [])
        if (
            getattr(interpretation, "rr_irregularity_class", None) == "irregular"
            and (
                "probable_af" in reasons
                or "atrial_measurements_unavailable" in reasons
            )
        ):
            interpretation.probable_af = True


def interpret(features: ECGFeatures) -> ECGInterpretation:
    """
    Run full clinical interpretation on an ECGFeatures object.

    Returns ECGInterpretation with all derived flags, classes, and
    territory assignments.
    """
    gf  = features.global_features
    rl  = features.representative_leads
    bts = features.beats
    bfs = features.beat_features

    meta: Optional[PatientMeta] = features.metadata.get("patient_meta")
    is_female: Optional[bool] = (
        meta.sex.lower() in ("f", "female") if (meta and meta.sex) else None
    )
    resolved_age = resolve_patient_age(meta)
    patient_age = resolved_age.age_years
    patient_age_days = resolved_age.age_days

    # Pediatric routing: Chapter 4 applies for ages 0 to < 16 years
    is_ped = patient_age is not None and 0 <= patient_age < PEDS_MAX_AGE_YEARS

    # ── Pre-compute aggregated beat-level data ────────────────────────────────
    r_prime_by_lead     = _median_r_prime_per_lead(bfs)
    r_prime_dur_by_lead = _median_r_prime_dur_per_lead(bfs)
    q_dur_by_lead   = _median_q_dur_per_lead(bfs, features.fs)
    p_dur_by_lead   = _median_p_dur_per_lead(bfs, features.fs, rl)
    per_beat        = _aggregate_per_beat(bts, bfs)

    # ── Step 2: Rhythm & rate ─────────────────────────────────────────────────
    hr_class                        = _classify_heart_rate(gf.heart_rate_bpm, patient_age_days)
    rr_cv, rr_irr_class, prob_af   = _rr_irregularity(bts, rl)

    # Advanced rhythm: premature complexes, bigeminy, trigeminy, pauses, AVB
    prem_result   = _detect_premature_complexes(bts, per_beat)
    bigeminy_type, trigeminy_flag = _detect_beat_patterns(prem_result["beat_types"])
    pauses_flag, pause_longest    = _detect_pauses(bts)
    complete_avb, av_diss, sec_deg_avb = _detect_avb_and_dissociation(bts, per_beat, gf)
    pacemaker_artifact            = bool(gf.paced_rhythm)

    # ── Step 3: Axis & intervals ──────────────────────────────────────────────
    if is_ped:
        qrs_axis_class = _peds_classify_qrs_axis(gf.qrs_axis_deg, float(patient_age))
    else:
        qrs_axis_class = _classify_qrs_axis(gf.qrs_axis_deg)

    p_axis_normal: Optional[bool] = None
    if gf.p_axis_deg is not None:
        p_norm = ((float(gf.p_axis_deg) + 180) % 360) - 180
        p_axis_normal = P_AXIS_SINUS_LOW <= p_norm <= P_AXIS_SINUS_HIGH

    t_axis_class, qrs_t_angle = _classify_t_axis(gf.t_axis_deg, gf.qrs_axis_deg)

    delta_any = any(bf.delta_present for bf in bfs)
    pr_class, avb_grade = _classify_pr(
        gf.pr_ms, delta_any, hr_bpm=gf.heart_rate_bpm, age=patient_age
    )
    # Upgrade avb_grade from complete_avb detection
    if complete_avb and avb_grade != 1:
        avb_grade = 3
    elif sec_deg_avb is not None and avb_grade is None and pr_class != "normal":
        avb_grade = 2

    qrs_width_measurement_disagreement = False
    if is_ped:
        qrs_width_class, bbb = _peds_classify_qrs_width(
            gf.qrs_ms, float(patient_age), rl, r_prime_by_lead, r_prime_dur_by_lead
        )
    else:
        qrs_width_class, bbb, qrs_width_measurement_disagreement = _resolve_qrs_width_class(
            gf.qrs_ms, gf.qrs_wide_ms, rl, r_prime_by_lead
        )
    wpw = _detect_wpw(gf, rl, bfs)

    # ── Step 4: Wave morphology ───────────────────────────────────────────────

    # Dextrocardia: use pediatric criteria for pediatric patients
    if is_ped:
        dextrocardia = _peds_dextrocardia(gf, rl)
    else:
        dextrocardia = _detect_dextrocardia(gf, rl)

    # Initialise pediatric-only outputs
    lsh_out: Optional[str] = None
    bvh_out: bool          = False
    pericarditis_out: bool = False
    early_repol_out: bool  = False
    pediatric_hypertrophy_evidence: Dict[str, object] = {}
    mi_evidence: Dict[str, object] = {"available": False, "reason": "morphology_bypassed"}
    mi_statement_candidates: List[Dict[str, object]] = []

    # Lead-reversal flags are needed before ST/RVH analysis so precordial
    # findings can be gated when a precordial reversal is detected.
    limb_rev_msg, precordial_rev = _lead_reversal_flags(features)
    st_exclude_leads: Tuple[str, ...] = tuple(PRECORDIAL_LEADS) if precordial_rev else ()

    if dextrocardia:
        p_class, rae_leads, lae_susp, lae_def, ptf_cls = None, [], False, False, None
        path_q, q_territories      = {}, []
        r_prog_class, r_transition = "indeterminate", None
        tall_t                     = []
        st_ele, st_dep             = {}, {}
        terr_ele, terr_dep         = [], []
        stemi_codes                = []
        reciprocal, recip_pairs    = False, []
        st_rate_rel                = False
        lvh_criteria, lvh_class    = [], None
        lvh_secondary_repol        = False
        low_v_class                = None
        rvh_class_out              = None
        posterior_mi               = False
        copd                       = False
        lv_strain                  = False
    else:
        p_class, rae_leads, lae_susp, lae_def, ptf_cls = _p_wave_morphology(
            rl, p_dur_by_lead, gf.ptf_v1_mv_ms, is_pediatric=is_ped
        )
        path_q, q_territories      = _pathological_q_waves(rl, q_dur_by_lead)
        r_prog_class, r_transition = _r_wave_progression(rl)
        tall_t                     = _tall_t_leads(rl)

        # ── Step 5: Anatomical localisation ──────────────────────────────────
        (
            st_ele, st_dep,
            terr_ele, terr_dep,
            stemi_codes,
            reciprocal, recip_pairs,
            st_rate_rel,
        ) = _st_analysis(
            rl,
            hr_bpm=gf.heart_rate_bpm,
            age=patient_age,
            exclude_leads=st_exclude_leads,
        )

        low_v_class   = _low_voltage(rl)

        # RVH graded scoring (pediatric: bypassed if RBBB; adult: always runs).
        # A precordial reversal corrupts the V1 R-dominance that RVH relies on,
        # so RVH is suppressed when a reversal is detected.
        rae_confirmed = bool(rae_leads)
        if is_ped and bbb == "RBBB":
            rvh_class_out = None    # pediatric RVH bypassed in presence of RBBB
        elif precordial_rev:
            rvh_class_out = None    # V1-based RVH unreliable under precordial reversal
        else:
            rvh_class_out = _rvh_scored(
                rl, r_prime_by_lead, rae_confirmed, qrs_axis_class,
                r_prime_dur_by_lead=r_prime_dur_by_lead,
            )

        # LV strain needed before LVH scoring
        lv_strain = _lv_strain_present(rl)

        # Pediatric LVH bypassed if RBBB or LBBB; adult bypassed only based on axis/age
        peds_bbb_bypass = is_ped and bbb in ("RBBB", "LBBB")
        if peds_bbb_bypass:
            lvh_criteria, lvh_class, lvh_secondary_repol = [], None, False
        else:
            lvh_criteria, lvh_class, lvh_secondary_repol = _lvh_criteria(
                rl, gf.qrs_ms, is_female,
                age=patient_age,
                qrs_axis_class=qrs_axis_class,
                lae_suspected=lae_susp or lae_def,
                lv_strain=lv_strain,
                bbb=bbb,
            )

        posterior_mi = _posterior_mi_r_dominant(rl)
        copd         = _copd_pattern(low_v_class, gf, bool(rae_leads), qrs_axis_class)
        pacing_context_for_mi = (
            features.metadata.get("rhythm_analysis", {}).get("pacing_context", {})
            if isinstance(features.metadata.get("rhythm_analysis", {}), dict)
            else {}
        )
        mi_evidence = build_mi_evidence(
            representative_leads=rl,
            st_elevation_leads=st_ele,
            st_depression_leads=st_dep,
            stemi_codes=stemi_codes,
            posterior_mi_suspected=posterior_mi,
            r_progression_class=r_prog_class,
            is_pediatric=is_ped,
            initial_qrs_axis_deg=estimate_initial_qrs_axis_deg(rl),
        )
        mi_statement_candidates = build_mi_statement_candidates(
            mi_evidence,
            bundle_branch_block=bbb,
            pacing_context=pacing_context_for_mi if isinstance(pacing_context_for_mi, dict) else {},
        )

        # ── Pediatric-specific morphology ─────────────────────────────────────
        if is_ped:
            pediatric_hypertrophy_evidence = build_pediatric_hypertrophy_evidence(
                representative_leads=rl,
                age_years=patient_age,
                sex=meta.sex if meta else None,
                qrs_axis_class=qrs_axis_class,
                bundle_branch_block=bbb,
            )
            rvh_evidence = pediatric_hypertrophy_evidence.get("rvh", {})
            if isinstance(rvh_evidence, dict):
                if rvh_evidence.get("bypassed_by"):
                    rvh_class_out = None
                else:
                    rvh_class = rvh_evidence.get("class")
                    rvh_class_out = str(rvh_class) if rvh_class is not None else None

            lvh_evidence = pediatric_hypertrophy_evidence.get("lvh", {})
            if isinstance(lvh_evidence, dict):
                if lvh_evidence.get("bypassed_by"):
                    lvh_criteria, lvh_class, lvh_secondary_repol = [], None, False
                else:
                    lvh_evidence_class = lvh_evidence.get("class")
                    lvh_class = str(lvh_evidence_class) if lvh_evidence_class is not None else None
                    lvh_criteria = list(dict.fromkeys(
                        [*lvh_criteria, *lvh_evidence.get("criteria", [])]
                    )) if lvh_class is not None else []
                    # Pediatric LVH grading uses percentile-table evidence, not
                    # the adult _lvh_criteria() scoring the strain flag above
                    # was computed against -- re-derive it against whichever
                    # class pediatric evidence actually produced.
                    lvh_secondary_repol = bool(lvh_class is not None and lv_strain)

            lsh_out = _peds_lsh(rl)
            bvh_evidence = pediatric_hypertrophy_evidence.get("bvh", {})
            bvh_out = isinstance(bvh_evidence, dict) and bool(bvh_evidence.get("suspected"))

            # Pericarditis: ST elevation in all three territory groups (ages 5–15)
            if patient_age is not None and PEDS_PERICARDITIS_AGE_LOW <= patient_age <= PEDS_PERICARDITIS_AGE_HIGH:
                all_three = (
                    any(t in terr_ele for t in ("anterior", "anteroseptal")) and
                    "lateral" in terr_ele and
                    "inferior" in terr_ele
                )
                if all_three:
                    pericarditis_out = True

            # Early repolarization: nonspecific ST elevation, no T inversion, ages 13–15
            if (
                patient_age is not None and
                PEDS_EARLY_REPOL_AGE_LOW <= patient_age <= PEDS_EARLY_REPOL_AGE_HIGH and
                bool(st_ele) and
                not stemi_codes and
                not pericarditis_out
            ):
                # Check that no T-wave is inverted in the ST-elevated leads
                any_t_inv = any(
                    _lp(rl, ld, "t_amp_mv", required_flags=("reliable_for_t",)) is not None
                    and (_lp(rl, ld, "t_amp_mv", required_flags=("reliable_for_t",)) or 0) < 0
                    for ld in list(st_ele.keys())
                )
                if not any_t_inv:
                    early_repol_out = True

            # BVH suppresses individual RVH/LVH when confirmed
            if bvh_out:
                rvh_class_out = None
                lvh_class     = None
                lvh_secondary_repol = False

    # ── QTc classification ────────────────────────────────────────────────────
    if is_ped and patient_age is not None:
        # Count leads with positive T for hypokalemia criterion
        pos_t_count = sum(
            1 for ld in STANDARD_12_LEADS
            if (
                _lp(rl, ld, "t_amp_mv", required_flags=("reliable_for_t",)) is not None
                and (_lp(rl, ld, "t_amp_mv", required_flags=("reliable_for_t",)) or 0) > 0
            )
        )
        qtc_class, qtc_elec_hint = _peds_classify_qtc(
            gf.qtc_bazett_ms,
            age_years=float(patient_age),
            is_female=is_female,
            st_depression_leads=st_dep,
            positive_t_count=pos_t_count,
            rvh_class=rvh_class_out,
            lvh_class=lvh_class,
            lsh=lsh_out,
            bvh=bvh_out,
            bbb=bbb,
            qrs_width_class=qrs_width_class,
        )
    else:
        qtc_class, qtc_elec_hint = _classify_qtc(
            gf.qtc_bazett_ms,
            st_depression_leads=st_dep,
            rvh_class=rvh_class_out if not dextrocardia else None,
            lvh_class=lvh_class,
            bbb=bbb,
            qrs_width_class=qrs_width_class,
        )
    if (
        int(features.metadata.get("n_reliable_qt_leads") or 0) == 0
        and qtc_class in (
            "possible_short", "borderline_short",
            "borderline_prolonged", "prolonged", "significantly_prolonged",
        )
    ):
        qtc_class = "indeterminate"
        qtc_elec_hint = None
    if getattr(gf, "qt_reliability", None) not in {"reliable", "rescued"}:
        qtc_class = "indeterminate"
        qtc_elec_hint = None

    # DXL: LBBB causes secondary repolarization changes that make ST elevation/depression,
    # T-wave abnormality, and Q-wave infarct signs unreliable — suppress all of them.
    if bbb == "LBBB":
        st_ele        = {}
        st_dep        = {}
        terr_ele      = []
        terr_dep      = []
        stemi_codes   = []
        reciprocal    = False
        recip_pairs   = []
        tall_t        = []
        path_q        = {}
        q_territories = []
        t_axis_class  = "indeterminate"
        qrs_t_angle   = None
        st_rate_rel   = False

    # Extreme tachycardia critical value: DXL_Threshold_Reference.xlsx flags
    # HR > 220-age bpm as a critical-value alert (age-predicted maximal heart
    # rate used as an implausibility ceiling rather than an exercise target).
    extreme_tachy_critical = bool(
        patient_age is not None
        and gf.heart_rate_bpm is not None
        and gf.heart_rate_bpm > (EXTREME_TACHYCARDIA_AGE_OFFSET_BPM - patient_age)
    )

    # Digitalis effect hint: short QTc together with a repolarization
    # abnormality (ST depression) in the precordial leads -- the classic
    # "scooped" ST + shortened QT pattern of therapeutic digoxin.
    digitalis_effect = bool(
        qtc_class == "short" and any(lead in PRECORDIAL_LEADS for lead in st_dep)
    )

    # Nonspecific T-wave abnormality: QRS-T angle beyond the normal limit.
    nonspecific_t_abnormal = bool(
        qrs_t_angle is not None and qrs_t_angle > QRS_T_ANGLE_ABNORMAL
    )

    interpretation = ECGInterpretation(
        # Rhythm
        rr_cv                    = rr_cv,
        rr_irregularity_class    = rr_irr_class,
        probable_af              = prob_af,
        heart_rate_class         = hr_class,
        # Axis
        qrs_axis_class           = qrs_axis_class,
        p_axis_normal            = p_axis_normal,
        t_axis_class             = t_axis_class,
        qrs_t_angle_deg          = qrs_t_angle,
        # Conduction
        pr_class                 = pr_class,
        avb_grade                = avb_grade,
        qrs_width_class          = qrs_width_class,
        bundle_branch_block      = bbb,
        qrs_width_measurement_disagreement = qrs_width_measurement_disagreement,
        qtc_class                = qtc_class,
        wpw_pattern              = wpw,
        # P wave / atrial
        p_morphology_class       = p_class,
        rae_leads                = rae_leads,
        lae_suspected            = lae_susp,
        lae_definite             = lae_def,
        ptf_v1_class             = ptf_cls,
        # Q waves
        pathological_q_leads     = path_q,
        q_wave_territories       = q_territories,
        # R progression
        r_progression_class      = r_prog_class,
        r_s_transition_lead      = r_transition,
        # ST
        st_elevation_leads       = st_ele,
        st_depression_leads      = st_dep,
        st_territories_elevated  = terr_ele,
        st_territories_depressed = terr_dep,
        stemi_suspected_codes    = stemi_codes,
        reciprocal_change_detected = reciprocal,
        reciprocal_pairs         = recip_pairs,
        # LVH
        lvh_voltage_criteria     = lvh_criteria,
        lvh_class                = lvh_class,
        lvh_secondary_repol_abnormality = lvh_secondary_repol,
        # Low voltage
        low_voltage_class        = low_v_class,
        # RVH
        rvh_suspected            = rvh_class_out is not None,
        # Lead reversal
        limb_reversal_suspected  = limb_rev_msg,
        precordial_reversal_suspected = precordial_rev,
        # Tall T
        tall_t_leads             = tall_t,
        # Advanced rhythm (DXL Chapter 2)
        premature_complexes      = prem_result["premature_complexes"],
        bigeminy                 = bigeminy_type,
        trigeminy                = trigeminy_flag,
        non_sustained_vt         = prem_result["nsVT"],
        pauses_detected          = pauses_flag,
        pause_longest_ms         = pause_longest,
        complete_av_block        = complete_avb,
        av_dissociation          = av_diss,
        second_degree_avb        = sec_deg_avb,
        pacemaker_like_artifact  = pacemaker_artifact,
        # Morphology extensions (DXL Chapter 3)
        dextrocardia_suspected   = dextrocardia,
        rvh_class                = rvh_class_out,
        copd_pattern             = copd,
        qtc_electrolyte_hint     = qtc_elec_hint,
        posterior_mi_suspected   = posterior_mi,
        mi_evidence              = mi_evidence,
        mi_statement_candidates  = mi_statement_candidates,
        st_rate_related          = st_rate_rel,
        extreme_tachycardia_critical = extreme_tachy_critical,
        digitalis_effect_suspected = digitalis_effect,
        nonspecific_t_abnormality = nonspecific_t_abnormal,
        # Pediatric extensions (DXL Chapter 4)
        is_pediatric             = is_ped,
        lsh_suspected            = lsh_out,
        bvh_suspected            = bvh_out,
        pediatric_hypertrophy_evidence = pediatric_hypertrophy_evidence,
        pericarditis_suspected   = pericarditis_out,
        early_repolarization_suspected = early_repol_out,
    )
    rhythm_analysis = features.metadata.get("rhythm_analysis", {})
    rule_summary = (
        rhythm_analysis.get("rule_summary", {})
        if isinstance(rhythm_analysis, dict)
        else {}
    )
    pacing_context = (
        rhythm_analysis.get("pacing_context", {})
        if isinstance(rhythm_analysis, dict)
        else {}
    )
    availability = (
        rhythm_analysis.get("availability", {})
        if isinstance(rhythm_analysis, dict)
        else {}
    )
    _apply_rule_summary_to_interpretation(interpretation, rule_summary, pacing_context)
    _apply_measurement_availability_to_interpretation(interpretation, availability)
    _suppress_pacing_unreliable_rhythm_claims(interpretation, pacing_context)
    return interpretation
