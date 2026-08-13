# Philips Rhythm Spec Coverage Matrix

## Purpose

This note maps the rhythm-analysis requirements summarized in `要点.pdf`
back to the current codebase, with a status for each item:

- `Implemented`: there is a concrete code path that substantially matches the spec intent.
- `Partial`: there is related code, but it is simplified, incomplete, or missing required control flow.
- `Missing`: no corresponding implementation was found in the current repository.

## Short Answer

**2026-07-12 update:** the historical gap statements below are no longer all
current. The extractor now runs pacing detection/control, QRST-template
subtraction, AF/AFL residual analysis, pause/escape/interpolated-candidate
logic, pre-excitation availability gating, and a structured candidate resolver.
The public-guideline `clinical_interpretation` layer consumes only validated
rhythm evidence; the richer DXL-inspired `rhythm_inputs` and
`statement_engine` remain reference-only. This is still not a claim of Philips
or DXL equivalence and it has not been clinically validated.

The current repository does **not** implement the full rhythm-analysis flow
described in `philips_rhythm.pdf` / `要点.pdf`.

It is now better described as:

- a `DXL-inspired` measurement and interpretation pipeline
- with pacing-first measurement policy, structured rhythm candidates, QRST
  subtraction, and AF/AFL signal-analysis branches implemented
- but without the complete proprietary Philips Chapter 2 code matrix and
  clinically validated equivalence

Evidence for that framing:

- The project README explicitly says it is "`DXL` style but not the private DXL implementation":
  `feature_extraction/README.md:3-20`
- The interpretation layer calls itself a research scaffold:
  `feature_extraction/ecgfeat/interpret.py:1-16`
- The phase-1 gap-closure plan says the goal is to get "materially closer" to the Philips/DXL requirement set rather than claim parity:
  `docs/superpowers/plans/2026-06-26-ecg-spec-gap-closure-phase1.md:5-20`
- The README still lists "overlay a rhythm / morphology rule engine" as a next step:
  `feature_extraction/README.md:69-75`

## About `要点.pdf`

`要点.pdf` is a reasonable engineering summary of Chapter 2 of
`philips_rhythm.pdf`. It adds structure, pseudo-code, configuration guidance,
and test suggestions that are **not** present as-is in the Philips PDF, but the
high-level rule breakdown is consistent with the source document.

## Coverage Matrix

| Spec area | Requirement from `要点.pdf` | Current code evidence | Status | Notes |
| --- | --- | --- | --- | --- |
| Top-level orchestration | Pacing-first flow before basic rhythm | `ECGFeatureExtractor` runs spike detection/validation, capture state, measurement-beat selection and pacing policy before final measurements | Implemented | Public final statements still use conservative availability gates. |
| Top-level orchestration | `primary_statement` + `additional_statements` output object | `rhythm_statements.py`, `statement_engine.py`; exported `statement_engine` candidate/final/suppressed lists | Implemented (reference layer) | Unified public final output is separately resolved under `clinical_interpretation`. |
| Top-level orchestration | `stop_further_interpretation` / `bypass_remaining_algorithm` flags | No matching runtime output object found; only plan/spec text mentions them | Missing | The control-plane described in `要点.pdf` is not present as a concrete result type. |
| Top-level orchestration | Evidence-rich statement objects with priority/suppression | `CandidateStatement`, `resolve_statement_candidates`, plus unified `RuleEvaluation` | Implemented | Proprietary-equivalence is not asserted; reference and authoritative outputs are explicitly separated. |
| Paced rhythm | Multilead pacing-spike detection | `detect_pacing_spikes` implements high-pass + multilead vote clustering: `feature_extraction/ecgfeat/quality.py:153-227` | Implemented | This is the clearest paced-rhythm-related feature already in place. |
| Paced rhythm | Default pacing analysis in normal extractor runs | Most runtime entry points instantiate `ECGFeatureExtractor(...)` without `enable_pacing=True`: `batch_extract_ecgfeat.py:366`, `evaluate_ludb.py:581`, `compare_annotations.py:784` | Missing | In practice, pacing is usually off unless a caller explicitly enables it. |
| Paced rhythm | Distinguish continuous vs intermittent pacing and stop non-paced rhythm-pattern analysis | `api.py` only tags `paced_beat_ids` and passes them to grouping/delineation: `feature_extraction/ecgfeat/api.py:114-159` | Partial | Beat tagging exists, but no Philips-style rhythm-control stop logic was found. |
| Paced rhythm | `PACED RHYTHM` vs `PACED COMPLEXES` statements | No paced statement generator found | Missing | Current code stores `paced_rhythm` as a boolean, not a rhythm statement. |
| Paced rhythm | Demand pacing behavior detection | No matching code found | Missing | No demand behavior logic located. |
| Paced rhythm | Pacemaker-like artifact as a separate outcome | `pacemaker_like_artifact` is currently derived from `gf.paced_rhythm`: `feature_extraction/ecgfeat/interpret.py:2092`, `feature_extraction/ecgfeat/interpret.py:2347` | Partial | This does not match the spec intent of distinguishing artifact from true paced rhythm. |
| Paced rhythm | Magnet / failure-to-sense / failure-to-capture logic | No matching code found | Missing | No fixed-rate async spike analysis located. |
| Paced rhythm | Under ventricular pacing, allow AF only and suppress other atrial rhythm labels | No matching code found | Missing | No ventricular-pacing-specific AF branch located. |
| Basic rhythm | Adult rate thresholds: tachy >= 100, brady < 50, complete AVB landmark < 45 | `feature_extraction/ecgfeat/interpret.py:39-46`, `feature_extraction/ecgfeat/interpret.py:464-473` | Implemented | Matches the Chapter 2 adult defaults. |
| Basic rhythm | P-axis sinus origin range (-30 to 120 degrees) | `feature_extraction/ecgfeat/interpret.py:77-78`, `feature_extraction/ecgfeat/interpret.py:2100-2104` | Implemented | P-axis sinus-range logic is present. |
| Basic rhythm | Basic rhythm statements for sinus / atrial / junctional / ventricular rhythms | Only scalar fields like `heart_rate_class` and `p_axis_normal` are computed: `feature_extraction/ecgfeat/models.py:216-235`, `feature_extraction/ecgfeat/interpret.py:2287-2304` | Partial | Measurements exist, but no primary rhythm statement layer was found. |
| Basic rhythm | Complete AV block based on low ventricular rate plus AV asynchrony | `complete_avb` is effectively `hr < 45 bpm`: `feature_extraction/ecgfeat/interpret.py:1574-1607` | Partial | The current code does not require the full asynchrony logic described in the spec. |
| Basic rhythm | AV dissociation | PR range / PR SD heuristic exists: `feature_extraction/ecgfeat/interpret.py:1597-1602` | Partial | Present as a simplified heuristic rather than the fuller mechanism-oriented rule set. |
| Basic rhythm | Atrial fibrillation using RR variability plus atrial-signal analysis after QRST subtraction | `atrial.py:build_qrst_subtracted_residual`, `_classify_af_afl`; validation flag is required by the unified rule | Implemented | An unvalidated subtraction makes the public AF rule `unavailable`. |
| Basic rhythm | Atrial flutter detection | time-domain and spectral residual analysis in `atrial.py`; candidate in `rhythm_statements.py` | Implemented | Remains research-use and requires external clinical validation. |
| Ventricular preexcitation | Delta-wave-based preexcitation support | Beat-level delta flag exists: `feature_extraction/ecgfeat/models.py:122`, `feature_extraction/ecgfeat/delineate.py:1068-1103`, `feature_extraction/ecgfeat/delineate.py:1297`, `feature_extraction/ecgfeat/delineate.py:1690` | Implemented | There is real delta-wave support at the beat level. |
| Ventricular preexcitation | Short PR + wide QRS + delta -> WPW/preexcitation decision | `_detect_wpw`: `feature_extraction/ecgfeat/interpret.py:708-719` | Partial | Present, but simplified relative to the multi-lead thresholding described in `要点.pdf`. |
| Ventricular preexcitation | Different delta-lead count threshold when PR is short | No matching configurable lead-count logic found | Missing | The current code uses `delta_any`, not lead-count thresholds. |
| Ventricular preexcitation | Accessory pathway side classification | No matching code found | Missing | No left/right accessory pathway classification located. |
| Ventricular preexcitation | Bypass remaining algorithm once preexcitation is detected | No bypass control object found | Missing | WPW is a boolean interpretation field, not a hard pipeline branch. |
| Premature complexes | Premature beat threshold: RR shortened by 15 percent | `PVC_RR_SHORTENING_PCT = 15.0`: `feature_extraction/ecgfeat/interpret.py:47-49`; used in `_detect_premature_complexes`: `feature_extraction/ecgfeat/interpret.py:1396-1492` | Implemented | This aligns well with the spec summary. |
| Premature complexes | APC / JPC / VPC classification | `_detect_premature_complexes`: `feature_extraction/ecgfeat/interpret.py:1396-1492` | Partial | Present, but simplified to RR + QRS width + P confidence rather than full morphology/polarity/compensatory-pause logic. |
| Premature complexes | VPC pair / triplet / NSVT | Consecutive V runs are detected: `feature_extraction/ecgfeat/interpret.py:1449-1474`, `feature_extraction/ecgfeat/interpret.py:2340-2341` | Partial | NSVT support exists, but still inside a simplified classifier. |
| Premature complexes | Ventricular / supraventricular bigeminy | Pattern detector exists: `feature_extraction/ecgfeat/interpret.py:1495-1538`, `feature_extraction/ecgfeat/interpret.py:2339` | Implemented | Sequence-pattern logic is present. |
| Premature complexes | Ventricular trigeminy | Pattern detector exists: `feature_extraction/ecgfeat/interpret.py:1495-1538`, `feature_extraction/ecgfeat/interpret.py:2340` | Implemented | This part is already coded. |
| Premature complexes | Interpolated VPC | `rhythm_rules.classify_post_pause_or_interpolated_beats` | Partial | Emits conservative candidates rather than a definitive proprietary label. |
| Premature complexes | Aberrant supraventricular premature complexes | No dedicated aberrancy classifier found | Missing | No rule matching that module was found. |
| Pauses / escapes / AV block | Long pause when RR > 140 percent of background RR | `RR_PAUSE_FACTOR = 1.40`: `feature_extraction/ecgfeat/interpret.py:47-49`; used in `_detect_pauses`: `feature_extraction/ecgfeat/interpret.py:1541-1557` | Implemented | This threshold is present. |
| Pauses / escapes / AV block | Escape-beat origin classification (atrial / junctional / ventricular) | `rhythm_rules.detect_pauses_and_av_block` exports `escape_origin` | Partial | Current output is supraventricular/ventricular and does not fully reproduce all proprietary origin labels. |
| Pauses / escapes / AV block | Second-degree AV block from P count > QRS count | atrial-event counts and pause evidence in `detect_pauses_and_av_block` | Partial | Conservative evidence path exists; full Philips equivalence is not established. |
| Pauses / escapes / AV block | Mobitz I / Wenckebach | `_detect_wenckebach` heuristic exists: `feature_extraction/ecgfeat/interpret.py:1560-1571`, wired in `feature_extraction/ecgfeat/interpret.py:1603-1605` | Partial | Implemented as a PR-trend heuristic, but without the fuller pause/P-count framing in the spec. |
| PR / AV conduction | Age x heart-rate dependent AVB1 threshold table | `_PR_AVB1_TABLE`, `_pr_avb1_threshold`, `_classify_pr`: `feature_extraction/ecgfeat/interpret.py:50-62`, `feature_extraction/ecgfeat/interpret.py:560-589` | Implemented | This is one of the strongest spec matches. |
| PR / AV conduction | Borderline prolonged PR category | Current `pr_class` is only `short / normal / avb1 / indeterminate`: `feature_extraction/ecgfeat/models.py:229`, `feature_extraction/ecgfeat/interpret.py:575-589` | Missing | `要点.pdf` expects a separate borderline PR state. |
| Reporting / payload | Structured evidence object for rhythm statements | top-level `rhythm_inputs`, `statement_engine`, and authoritative `clinical_interpretation` | Implemented | Legacy statement engine is reference-only; unified AF projection requires validated evidence. |
| Reporting / payload | Expose advanced rhythm findings in default app/report flow | `app.py` summary includes `probable_af`, `heart_rate_class`, `pr_class`, `qrs_width_class`, `BBB`, `WPW`: `app.py:442-450`; demo Step 2 only prints rate / RR irregularity / probable AF: `demo_feature_extraction.py:1150-1155` | Partial | Many advanced Chapter 2 fields exist internally but are not fully surfaced. |
| Testing | Broad unit coverage for Chapter 2 rhythm rules | Only light pacing tests and a few interpretation field checks were found: `tests/test_pacing_grouping.py:13-107`, `tests/test_ecg_report_sheet.py:55-61`, `tests/test_medgemma_ecg_core.py:50-62` | Partial | Coverage is not close to the test matrix proposed in `要点.pdf`. |

## What Is Already Strong

These areas are already aligned enough with the spec that they look like solid
foundations rather than missing work:

- adult rate thresholds
- sinus-range P-axis logic
- dynamic PR AVB1 thresholds from the age x HR table
- 15 percent RR premature-beat threshold
- 140 percent RR pause threshold
- bigeminy / trigeminy pattern detection
- beat-level delta-wave support

## Biggest Gaps Relative to `要点.pdf`

If the goal is to make the implementation behave like the summarized Philips
Chapter 2 flow, the highest-impact missing pieces are:

1. Complete and validate the remaining proprietary rhythm code matrix and
   control-flow semantics without conflating them with the public final layer.

2. More complete premature / pause / AV block logic
   - interpolated VPC
   - aberrant supraventricular premature beats
   - escape-beat origin
   - second-degree AV block via P/QRS counting

3. Preexcitation control flow
   - multi-lead delta thresholding
   - accessory pathway side classification
   - hard bypass of downstream rhythm interpretation

## Practical Bottom Line

If someone asks, "Is the current rhythm analysis implemented according to
`philips_rhythm.pdf` / `要点.pdf`?", the most accurate answer is:

> The repository implements a meaningful rhythm-analysis pipeline including
> pacing policy, structured statements, QRST-residual AF/AFL analysis and a
> conservative public-rule projection. It is not a complete or clinically
> validated implementation of the proprietary Philips rhythm specification;
> Philips/DXL-inspired results remain reference-only.
