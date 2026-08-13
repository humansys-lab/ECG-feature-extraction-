<!-- i18n-nav -->
[中文](p_wave_snr_stability_fusion_implementation.md) | [English](p_wave_snr_stability_fusion_implementation.en.md) | [日本語](p_wave_snr_stability_fusion_implementation.ja.md)
<!-- /i18n-nav -->

> Translation note: This English version is provided for convenience. If it differs from the source document, the source document prevails.

<a id="p-波局部-snr边界稳定度与稳健融合改造"></a>
# P-wave local SNR, boundary stability and robust fusion transformation

Generation time: 2026-07-28

<a id="改造范围"></a>
## Transformation scope

This transformation maintains the existing P candidate, raw/corrected/consensus three-layer boundary and downstream output
Compatible, the following deterministic DSP evidence has been added:

1. Generate zero-phase `35 Hz` low-pass detection view of the final P candidate; amplitude, area, and morphology
   Continue to use the original measurement signal.
2. Estimate local noise from the TP quiet window verified by the T offset of the previous beat on a beat-by-beat basis. PR/PQ
   Contains Ta and is not used as a noise window; the pre-P fallback of the first beat is clearly marked as the previous T wave context is unknown.
   Output:
   - `p_local_noise_rms_mv`
   - `p_local_noise_source`
   - `p_local_snr`
   - `p_informative`
   - `p_tp_gap_ms`
   - `p_quiet_window_available`
   - `p_on_t_overlap_risk`
   - `p_ta_overlap_risk`
   - `p_baseline_method`
   - `p_baseline_confidence`
3. Compare tangent with `4%、6%、8%、10%、12%` threshold bounds on low-pass P view,
   Constructed with MAD:
   - `p_onset_sigma_ms`
   - `p_offset_sigma_ms`
   - `p_boundary_stability_method`
4. When at least 4 leads with evidence of local SNR/stability form a valid cluster:
   - onset uses the second earliest boundary;
   -offset uses second night bounds.
5. If the protected early onset causes all offsets to violate P-duration geometric constraints, fall back to
   cluster median to avoid reducing complete boundary coverage.
6. Candidate marker `p_isoelectric_uninformative` lower than `2 sigma`. it does not participate in high confidence
   Sequential statistics, but can still fall back into the old low-confidence cluster fallback when there are not enough high-quality leads.
7. When TP is insufficient at high heart rate, spurious local SNR is not calculated. The code only fits both sides of P candidates
   Continuous Robust Quadratic Trend as diagnostic view, capping bounding confidence at `0.45`, and prohibiting
   second-earliest/second-latest fusion; the trend does not cover the original P bounds.
8. Ta risk acts solely on offset confidence. The existence confidence of P will not be affected by Ta/no TP.
   Clear directly to distinguish "P is seen" from "equipotential boundaries cannot be determined reliably".

<a id="ludb-全量结果"></a>
## LUDB Full results

Run `analyze_ludb_p_boundaries.py` on 200 LUDB records:

| Indicators | Before transformation | After transformation |
|---|---:|---:|
| Complete P boundary coverage | 91.10% | 91.17% |
| Consensus onset bias | +9.31 ms | +4.95 ms |
| Consensus onset SD | 35.45 ms | 34.33 ms |
| Consensus onset MAE | 25.64 ms | 22.03 ms |
| Consensus offset bias | -9.75 ms | -8.65 ms |
| Consensus offset SD | 26.13 ms | 26.84 ms |
| Consensus offset MAE | 17.32 ms | 17.37 ms |
| Consensus P-duration bias | -19.06 ms | -13.60 ms |
| Consensus P-duration SD | 23.07 ms | 22.35 ms |
| Consensus P-duration MAE | 25.40 ms | 21.31 ms |
| Record-level P-duration bias | -20.48 ms | -15.04 ms |
| Record-level P-duration MAE | 23.53 ms | 18.59 ms |

Result directory:

- `ludb_p_boundary_analysis_snr_fusion/`
- `ludb_cse_evaluation_snr_fusion/`

<a id="cse-风格代理评估"></a>
## CSE Style Agent Evaluation

| Indicators | Before transformation | After transformation | Description |
|---|---:|---:|---|
| P duration original bias | -20.48 ms | -15.04 ms | improved 5.44 ms |
| P duration Original MAE | 23.53 ms | 18.59 ms | Improved 4.94 ms |
| P duration trimmed bias | -22.11 ms | -16.06 ms | Still not up to ±10 ms |
| P duration trimmed SD | 13.12 ms | 13.15 ms | Stay within 15 ms limit |
| PR original bias | -5.51 ms | -4.57 ms | Small improvement |
| PR original MAE | 10.59 ms | 10.20 ms | Small improvement |
| PR coverage rate | 92.6% | 92.6% | Not declining, still a follow-up gap |
| QRS/QT | unchanged | unchanged | no joint regression |

This transformation significantly reduces the P-duration shrinkage caused by multi-lead fusion, but the SD/MAE of P offset
There is no synchronization improvement, indicating that the independent offset area/quiet-window method should be added first in the next stage.
Instead of continuing to adjust only the global order statistics.

<a id="验证"></a>
## Verify

- Full test set: `927 passed, 2 skipped, 145 subtests passed`
- LUDB P Boundary: 200/200 Success
- LUDB CSE Style Agent: 200/200 Success

<a id="ta-高心率无-tp-防护追加改造"></a>
## Ta / High heart rate without TP protection additional modification

The following protections have been added against the physiological indiscernibility of Ta and P-on-T:

1. The PR/PQ segment completely exits the P-local noise candidate set. For subsequent shots, only the previous shot T offset is reached.
   Leave at least `16 ms` data after guard between P onset of this beat before marking
   `p_quiet_window_available=True`.
2. When TP is insufficient `p_local_snr=None`, do not forge low-noise evidence; use P sideband on both sides
   Fitting continuous robust quadratic trend, only stability view, baseline confidence is `0.25`.
3. `p_tp_gap_ms <30 ms` mark `p_on_t_overlap_risk`, onset/offset confidence
   Upper limit `0.45`, and disable second-earliest/second-latest extreme value fusion.
4. Only when you have a verified TP baseline, check the PR area behind P for signs of polarity opposite to P and amplitudes of at least
   Ta-like excursion of `max(10 uV, 2 sigma_noise)`; individually lower offset when hitting
   Confidence, not cleared P-presence confidence.
5. After extreme value fusion is disabled, if different lead subsets form invalid global geometry, only the same batch is allowed to be used.
   Effective lead's onset/offset **pairwise median** restores coverage, not extreme value statistics.
6. ST-T subtraction of QRST residual adds beat-to-beat time shifting, robust amplitude/bias adaptation, and cosine seams
   taper and second-order differential high-frequency energy gating; the residual path at the median RR `<520 ms`
   It is directly unavailable to avoid the template from intruding into the P area next time.

Compared with the previous version of SNR fusion’s LUDB 200 full results:

| Indicators | SNR fusion | Ta/TP guard final |
|---|---:|---:|
| Complete P boundary coverage | 91.17% | 93.30% |
| Consensus onset bias / SD | +4.95 / 34.33 ms | +4.64 / 35.80 ms |
| Consensus offset bias / SD | -8.65 / 26.84 ms | -8.69 / 27.58 ms |
| Consensus P-duration bias / SD | -13.60 / 22.35 ms | -13.33 / 23.00 ms |
| Record P-duration bias / MAE | -15.04 / 18.59 ms | -15.03 / 18.66 ms |
| P-duration trimmed SD | 13.15 ms | 12.56 ms |
| PR coverage / MAE | 92.6% / 10.20 ms | 93.1% / 9.85 ms |

Security gating does not address the systematic shrinkage of existing P-duration of approximately `-15 ms`; its gains are primarily
Avoid using the Ta/T-tail as a "quiet baseline" and improve full border coverage. beat-level onset,
The SD/MAE of offset and duration are slightly degraded, indicating the new pairwise median fallback
Expanded coverage of difficult samples; these difficult samples must rely on independent high heart rate/Ta annotation sets for continued verification.

Final result directory:

- `ludb_p_boundary_analysis_ta_tp_guard_final/`
- `ludb_cse_evaluation_ta_tp_guard_final/`
