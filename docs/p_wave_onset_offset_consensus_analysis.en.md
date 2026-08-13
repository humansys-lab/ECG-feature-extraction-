<!-- i18n-nav -->
[中文](p_wave_onset_offset_consensus_analysis.md) | [English](p_wave_onset_offset_consensus_analysis.en.md) | [日本語](p_wave_onset_offset_consensus_analysis.ja.md)
<!-- /i18n-nav -->

> Translation note: This English version is provided for convenience. If it differs from the source document, the source document prevails.

<a id="p-波-onsetoffset-检测问题与共识增强分析"></a>
# P wave onset/offset detection problem and consensus enhancement analysis

This article focuses on the current P-wave related detection, especially the reasons for the large deviations of `P onset` and `P offset`, and how to enhance stability using the method of "each lead, each beat -> representative waveform -> multi-lead consensus".

<a id="1-结论概览"></a>
## 1. Conclusion Overview

The current algorithm already has part of the P-wave consensus mechanism, but it is mainly used for anchoring P-peak candidates and deriving a `pr_consensus_ms`. The true P-wave boundaries, especially for `P offset`, are still primarily determined independently by each lead's local tangent/threshold logic.

<a id="11-已实现的第一阶段修改"></a>
### 1.1 Implemented first phase modifications

This round has completed a version of conservative implementation:

1. Added P boundary consensus bypass field in beat-level `LeadBeatFeatures`, including:
   - `p_onset_consensus_index`
   - `p_offset_consensus_index`
   - `p_dur_consensus_ms`
   - `pr_segment_consensus_ms`
   - `p_onset_consensus_support`
   - `p_offset_consensus_support`
   - `p_onset_consensus_spread_ms`
   - `p_offset_consensus_spread_ms`
   - `p_boundary_consensus_source`

2. `_apply_multilead_consensus(...)` will now perform cluster consensus on reliable P offsets of the same beat:
   - At least 3 reliable leads are required to support the same offset cluster.
   - Does not overwrite original `bf.p.offset`.
   - Only record consensus offset, P duration, and PR segment as bypass fields.

3. `build_representative_lead_features(...)` will aggregate these P consensus boundary fields into `representative_leads.*.params`.

4. The global PR of `compute_global_features(...)` adds conservative consensus gating:
   - raw limb-lead PR must be clearly split.
   - supported `pr_consensus_ms` must be supported by at least 3 P onset consensus.
   - consensus PR must be within the range `120-300 ms`.
   - The consensus PR must be at least 25ms longer than the raw limb median, which is mainly used to correct the short PR caused by the late onset of P.
   - Short consensus will not take over global PR to avoid misuse of 102ms consensus such as record 57.

LUDB spot check：

- Record 39: global PR is adjusted from `136 ms` to `201 ms`. Compared with GT `224 ms`, the error is improved from `-88 ms` to `-23 ms`.
- record 57: `pr_consensus_ms=102 ms` has not been taken over, and the global PR still maintains the raw path to avoid misuse of short consensus.
- record 130: The difference between `pr_consensus_ms=124 ms` and raw median is not enough, so it is not taken over.
- record 5: P offset consensus / P duration consensus bypass fields are generated, but the global PR maintains the original raw path.

<a id="12-已实现的第二阶段修改p-onset-cluster-consensus"></a>
### 1.2 Implemented second phase modification: P onset cluster consensus

Subsequently, the cluster-level consensus enhancement of P onset was completed. The core goal was to solve the problem of "raw per-lead PR being shortened by the fake P close to QRS", while avoiding the unconditional use of the short PR fake consensus for global PR.

Implementation points:

1. `_apply_multilead_consensus(...)` no longer uses a simple P10/median of all P onsets; it will first cluster by onset time in reliable leads of the same beat.
2. P onset cluster needs to satisfy general constraints:
   - Supported by at least 3 reliable leads.
   - The cluster spread does not exceed approximately `45 ms`.
   - cluster PR is located in the physiological range of `120-260 ms`.
   - Record limb/precordial support count and participating lead list.
3. If there is a qualified physiological cluster, `pr_consensus_ms` uses the cluster center; otherwise, it falls back to the original percentile path.
4. `build_representative_lead_features(...)` will aggregate `p_onset_cluster_*` metadata to the representative lead.
5. `compute_global_features(...)`’s PR consensus gating adds a narrower rescue path: only when there is an obvious short PR outlier in the raw PR distribution, and the P onset cluster has enough support, a small spread, and enough limb lead support, the global PR is allowed to be raised from raw median to cluster consensus.

LUDB outlier spot check：

- Record 2: global PR is adjusted from `170.5 ms` to `178.0 ms`. Compared with GT `194.5 ms`, the error is improved from `-24.0 ms` to `-16.5 ms`.
- Record 64: global PR is adjusted from `127.0 ms` to `143.0 ms`. Compared with GT `175.0 ms`, the error is improved from `-48.0 ms` to `-32.0 ms`.
- record 139: global PR is adjusted from `206.0 ms` to `199.0 ms`, relative to GT `162.0 ms`, the error is improved from `+44.0 ms` to `+37.0 ms`.

<a id="13-已实现的第三阶段修改pr-core-rescue-按-beat-投票"></a>
### 1.3 Implemented third phase modification: PR core rescue vote by beat

Further analysis found that `_physiologic_pr_core_from_beats(...)` once directly put all lead-beat PR candidates together to take the upper percentile. This will cause the same type of falsely long PRs for multiple leads on a certain beat to be counted repeatedly, thereby overstretching the global PR.

Currently changed to:

1. Still filter candidates by PR physiological range, P confidence and `p_unreliable` flag.
2. First group according to `beat_id`, and take a PR median in each beat.
3. Then perform the original median / upper-percentile aggregation on the beat-level PR core.

This maintains the ability to use upper core rescue when short PR fake candidates account for the majority, but avoids repeated voting on multiple leads for a single abnormal beat.

LUDB outlier spot check：

- record 182: global PR is adjusted from `202.12 ms` to `136.58 ms`. Compared with GT `134.0 ms`, the error is improved from `+68.12 ms` to `+2.58 ms`.
- Record 139: global PR is further adjusted from `199.0 ms` in the second stage to `173.68 ms`. Compared with GT `162.0 ms`, the error is improved from `+37.0 ms` to `+11.68 ms`.
- On the 12 PR outlier small batch records, the PR MAE dropped from the first-stage result of about `50.47 ms` to the second-stage result of about `47.93 ms` to about `40.40 ms`.

The above modifications do not use record ID, LUDB annotation, file name, diagnostic label or patient metadata for conditional judgment; record is only used as a regression verification sample.

<a id="14-已实现的第四阶段修改强共识回写原始-p-wavebounds"></a>
### 1.4 Implemented fourth phase modification: Strong consensus writes back original P WaveBounds

After referring to the `P wave` technical document in the project root directory, it is further clarified: the goal is not just to make the PR of the final report closer to the reference value, but to make the original P-wave boundary detection itself more reliable. The documentation emphasizes that median/representative P detection should output the true P onset/offset, and report should only consume these measurements.

Therefore, the current `_apply_multilead_consensus(...)` has been expanded from "write-only consensus bypass field" to "conservatively correct raw `WaveBounds` in case of strong consensus":

1. For the obviously outlier raw `p.onset`:
   - A qualifying physiological P onset cluster must exist.
   - The difference between raw onset and cluster center is at least about `20 ms`.
   - Must meet `p_onset < p_peak < p_offset` after correction.
   - P duration, onset-to-peak, and PR segment all need to be within the physiological range.
   - After correction, update raw `pr_ms` and `p_dur_ms` simultaneously, and add `p_onset_corrected_by_consensus` flag.

2. For the obviously outlier raw `p.offset`:
   - There must be a P offset cluster with sufficient support and reasonable spread.
   - The difference between raw offset and consensus offset is at least about `20 ms`.
   - Must meet `p_onset < p_peak < p_offset` after correction.
   - P duration and PR segment must be reasonable.
   - After correction, raw `p_dur_ms` is updated simultaneously, and `p_offset_corrected_by_consensus` flag is added.

The focus of this step is to improve the original beat/lead level boundaries, rather than just fixing the global PR. LUDB spot check：

- Record 2: The raw PRs of some leads such as aVL and V1 are pulled towards the multi-lead consensus; there are still short PR false positives of peaks such as aVR/V3/V5/V6 that are suspected of being wrongly selected and have not been hard repaired.
- Record 5: The raw P offsets of 12 lead-beat rows are corrected by consensus, reducing offset dispersion.
- 12 PR outliers For small batches, the PR MAE is basically the same as the third stage and slightly better: about `40.40 ms -> 40.07 ms`.

This also exposes the next level of the problem: if the P peak of a lead itself selects a false deflection close to QRS, it is not safe enough to just move the onset/offset. The next step that should be done is P peak/candidate level reselection or suppression, preferably combined with raw rhythm residual, I/II + V1-V6 lead selection, baseline-activity threshold, template similarity and PR/PP consistency in the `P wave` document.

<a id="15-已实现的第五阶段修改短-pr-假-p-candidate-抑制"></a>
### 1.5 Implemented fifth stage modification: short PR false P candidate suppression

The fourth stage only shifts onset/offset under the premise that "the same P peak is still credible". For some leads, P peak itself has selected the local alias before QRS. At this time, forcibly moving the onset/offset to the global consensus position will produce unreasonable geometry. Therefore, conservative candidate suppression is currently added:

1. A strong physiological P onset cluster must exist.
2. The raw PR of the current lead is obviously too short, for example, less than about `100 ms`.
3. Cluster PR is at least about `50 ms` longer than raw PR.
4. The P peak of the current lead is too far from the cluster onset, exceeding the acceptable range of a single P wave onset-to-peak.
5. When the above conditions are met, clear the raw P `WaveBounds`, `pr_ms`, P amplitude/area, and P morphology derived fields of the lead/beat, and add:
   - `p_candidate_suppressed_by_consensus`
   - `p_unreliable`

The goal of this step is not to "make up a better-looking P wave", but to stop propagating false P candidates to the representative, P axis, P morphology and report when the peak itself is not credible.

LUDB spot check：

- record 2: short PR false candidates such as aVR/V3/V5/V6 that are close to QRS are significantly reduced; the global PR is adjusted from `175.5 ms` in the fourth stage to `181.0 ms`. Compared with GT `196.5 ms`, the error is improved from `-21.0 ms` to `-15.5 ms`. In per-lead PR, aVR is improved from extremely short `94 ms` to `164 ms`, and V6 becomes N/A to avoid continuing to report false Ps.
- Record 22: global PR is adjusted from `125.0 ms` to `148.0 ms`. Compared with GT `170.0 ms`, the error is improved from `-45.0 ms` to `-22.0 ms`.
- On 12 PR outlier small batches, the PR MAE further dropped from about `40.07 ms` in the fourth stage to about `37.69 ms`.

This is still candidate suppression, not full candidate reselection. The next larger implementation should be based on the `P wave` documentation: reselect P peak from the raw rhythm residual or representative waveform candidate set and score using template similarity, PR/PP consistency, dual-lead support and baseline activity threshold.

<a id="16-已实现的第六阶段修改p-peak-上游融合锚点重选"></a>
### 1.6 Implemented sixth phase modification: P peak upstream fusion anchor point reselection

The fifth stage can prevent obviously erroneous short PR false P candidates from continuing to propagate, but it is still "suppression" in nature. In order to make the original waveform detection itself more accurate, the improvement is currently moved to the P peak selection stage of each lead/beat.

Implementation points:

1. In every lead P search for `delineate_beats(...)`, `_fused_p_peak` is no longer only used to narrow the search window or break ties when the magnitudes are exactly equal.
2. Added `_select_p_peak_with_fused_anchor(...)`:
   - The fused P peak must be within the P search window of the current lead.
   - The fused P peak amplitude must reach the P minimum amplitude threshold for that lead.
   - If the current abs-mode peak is significantly close to QRS, and the distance from fused peak to QRS is within a reasonable P peak range, it is allowed to return to fused peak.
   - If the two amplitudes are close and the fused peak is more consistent with the timing, it is also allowed to return to the fused peak.
3. This logic is used for both the initial P peak search and the constrained re-search after P peak is judged to be outside the physiological window.
4. The conditions only rely on local waveform amplitude, QRS reference point, sampling rate and existing multi-lead fused anchor, without using any record ID or LUDB specific rules.

The difference between this step and the fifth stage is that the fifth stage clears P when the peak is no longer credible; the sixth stage tries to select a more credible P peak before boundary measurement. Therefore, it can reduce subsequent `p_unreliable` and P deletions, rather than simply hiding erroneous candidates.

LUDB outlier spot check:

- Record 2: global PR is adjusted from `181.0 ms` in the fifth stage to `183.0 ms`. Compared with GT `196.5 ms`, the error is improved from `-15.5 ms` to `-13.5 ms`; P missingness is reduced from `37` to `34`.
- record 22: global PR is adjusted from `148.0 ms` to `154.0 ms`. Compared with GT `170.0 ms`, the error is improved from `-22.0 ms` to `-16.0 ms`; P missing is reduced from `29` to `28`.
- record 23: global PR remains unchanged, but P missingness drops from `12` to `11`.
- record 182: global PR maintains `136.58 ms`, which is still very close to GT `134.0 ms`; P deletion drops from `3` to `2`.
- For 12 PR outliers in small batches, the PR MAE dropped slightly from about `37.69 ms` in the fifth stage to about `37.07 ms`.

This is still not a complete raw rhythm residual/template similarity recheck, but has moved the "consensus" forward from the final PR report to the original P peak selection, more directly on the quality of the original `WaveBounds`.

<a id="17-已实现的第七阶段修改p-peak-cluster-支持度融合"></a>
### 1.7 Implemented seventh stage modification: P peak cluster support fusion

The sixth stage solves the problem of "a single lead selects the wrong peak within the search window", but `_fused_p_peak` itself may still be biased by a few high-amplitude leads, or common small fluctuations near the onset of QRS. Beat-level P peak pre-pass fusion is currently further enhanced.

Implementation points:

1. `_fuse_peak_anchor(...)` adds optional per-lead normalization:
   - Each lead candidate is first normalized within this lead to prevent a high-amplitude lead from using its absolute amplitude to overwhelm the low-amplitude P consensus of too many leads.
   - Cluster scoring uses unique lead support; multiple candidates for the same lead will not be voted twice.
2. P peak pre-pass call enables this mode; T peak fusion still maintains the original amplitude/area strategy to avoid disturbing T wave detection.
3. Add P-to-QRS physiological constraints to P peak cluster score:
   - qrs reference is changed to prefer QRS onset representing prior.
   - When there is no QRS onset prior, use `R - 20 ms` as an approximate QRS onset instead of directly using R peak.
   - In this way, short-interval false clusters near QRS onset will not be misjudged as reasonable P peaks because they are still tens of milliseconds away from R peak.
4. Strengthen the penalty for the short QRS-adjacent cluster; if there is no more reasonable alternative, it can still be selected, but as long as there is a P cluster with reasonable timing, the latter will be preferred.
5. Add raw-support guard to global PR consensus:
   - To take over the global PR, a long PR cluster must have at least 2 raw PR values close to the cluster to avoid a single/few abnormal leads from stretching normal PRs into long PRs.
   - When the raw median is pulled up by long PR outliers, and the strong onset cluster has enough raw support, the cluster is allowed to be rescued downward.
   - When raw representative PR is very sparse, give priority to using strong onset clusters rather than letting single-lead raw PR determine global PR.

This step is more upstream than the sixth stage: not only is each lead corrected after a given fused anchor, but in the fused anchor generation stage, "one vote per lead + multi-lead support + QRS onset timing" is used to prevent the anchor itself from being biased.

LUDB full-batch comparison:

- `ludb_full_compare/summary.csv` Reference: PR MAE about `17.02 ms`, PR bias about `-12.02 ms`, `|PR diff| > 20 ms` about `47/151`, `|PR diff| > 40 ms` about `13/151`.
- Current `tmp_p_anchor_final_full/summary.csv`: PR MAE is about `9.78 ms`, PR bias is about `-6.02 ms`, `|PR diff| > 20 ms` is about `18/152`, `|PR diff| > 40 ms` is about `0/152`.
- beat/lead level P missing of any boundary reduced from `6049/25860` to `3005/25860`.
- `p_unreliable` downgraded from `3104` to `2669`.

Typical improvements:

- record 130: global PR is adjusted from `122.5 ms` to `216.0 ms`, relative to GT `207.0 ms`, it is improved from `-84.5 ms` to `+9.0 ms`.
- record 57: global PR is adjusted from `225.0 ms` to `124.0 ms`, relative to GT `139.0 ms`, it is improved from `+86.0 ms` to `-15.0 ms`.
- record 23: global PR is adjusted from `145.0 ms` to `197.0 ms`, relative to GT `210.0 ms`, it is improved from `-65.0 ms` to `-13.0 ms`.
- record 17/39/102 had a regression in the intermediate version due to the wrong reference R peak; after changing to QRS onset reference, it recovered to be close to GT.
- record 11/56/133 was rolled back in the intermediate version due to long PR cluster overtakeover; after adding raw-support guard, record 11 returned from `219 ms` to `149.5 ms`, record 56 returned from `207 ms` to `130 ms`, record 133 returned from `221 ms` Back to `182 ms`.

<a id="18-已实现的第八阶段修改pr-core-双峰保护"></a>
### 1.8 Implemented eighth stage modification: PR core bimodal protection

When further analyzing the remaining outliers, it was found that the representative lead raw PR of some records was obviously abnormal, and `api.py` would enable beat-level `_physiologic_pr_core_from_beats(...)` as protection. However, the old logic fixed `78th percentile` when there are a large number of beat-level PR cores. This is to avoid shortening the PR when false short PR candidates account for the majority; when beat-level PR itself appears with two clusters that are clearly separated, this upper quantile strategy will mistakenly regard the upper cluster as a real PR.

Currently a new conservative bimodal protection is added:

1. Still aggregate by beat first, and each beat only contributes one PR median to avoid repeated voting by multiple leads of the same beat.
2. When there are at least 5 beat-level PR cores and there is a significantly large gap in adjacent sorting values, first determine whether a lower cluster and an upper cluster are formed.
3. This is considered a bimodal split only when the lower cluster has at least 3 beats, the upper cluster has at least 2 beats, and the maximum gap meets both the absolute threshold and the multiple threshold relative to the regular gap.
4. For this bimodal distribution, return the upper quartile of the lower cluster instead of the 78th quartile of the entire beat core.
5. If there is only a single long PR outlier, or the PR distribution changes continuously, keep the original 78th percentile strategy.

This step is not a record-specific rule and does not modify the original P `WaveBounds`. Its function is to prevent the report layer from amplifying the wrong long PR cluster again when the raw P onset has been split.

LUDB full-batch comparison in the seventh stage:

- `tmp_p_anchor_final_full/summary.csv`: PR MAE is about `9.78 ms`, PR bias is about `-6.02 ms`, `|PR diff| > 20 ms` is about `18/152`, `|PR diff| > 40 ms` is about `0/152`.
- Current `tmp_pr_core_split_only_full/summary.csv`: PR MAE is about `9.54 ms`, PR bias is about `-6.30 ms`, `|PR diff| > 20 ms` is `17/152`, `|PR diff| > 40 ms` is `0/152`.
- Beat/lead level P Missing either boundary remains `3005/25860` unchanged.
- `p_unreliable` leaves `2669` unchanged.

Typical changes:

- Record 139: global PR is adjusted from `201.8 ms` to `159.5 ms`. Compared with GT `162.0 ms`, the error is improved from `+39.8 ms` to `-2.5 ms`.
- The rest of the full-batch PR results remain unchanged, indicating that the protection is only triggered when the bimodal condition is truly met.

<a id="19-已实现的第九阶段修改p-candidate-templatepp-context-guard"></a>
### 1.9 Implemented ninth stage modification: P candidate template/PP context guard

Newly added raw beat/lead level P candidate context scoring. It is directly connected after the P detection of each lead/beat and before the multi-lead consensus, so that the subsequent P onset/offset consensus can see the contextual quality of the candidate itself, rather than just patching it on the final global PR report.

New fields include:

1. `p_template_corr`: P fragments in the same group and lead are correlated with the normalized median template. Take the larger value of forward/reverse correlation to avoid false penalties simply because of different P polarities.
2. `p_multilead_support` and `p_multilead_support_score`: Whether the same beat and similar P peak are supported by multiple reliable P leads.
3. `p_pr_consistency_score`: Only enabled when PR distribution is stable and the evidence is sufficient; remain neutral when PR is unstable to avoid false penalties in AV block/AV dissociation scenarios.
4. `p_pp_consistency_score`: Only enabled when the PP rhythm is stable and the evidence is sufficient; remain neutral when the evidence is insufficient or the PP is unstable.
5. `p_candidate_context_score` and `p_context_reason`: Summarize the above evidence to facilitate interpretation of the quality of each P candidate in `*_features.json`.

The current suppression gate is very conservative: candidates will be cleared only when template is weak, multi-lead support is weak, timing is weak, and the total context score is also low. The implementation deliberately avoids immediate suppression in the loop to avoid affecting the support count of subsequent leads after the previous lead is cleared.

LUDB full-batch comparison in the eighth stage:

- `tmp_pr_core_split_only_full/summary.csv`: PR MAE is about `9.54 ms`, PR bias is about `-6.30 ms`, `|PR diff| > 20 ms` is about `17/152`, `|PR diff| > 40 ms` is about `0/152`.
- Current `tmp_p_context_guard_full/summary.csv`: PR MAE is about `9.54 ms`, PR bias is about `-6.30 ms`, `|PR diff| > 20 ms` is about `17/152`, `|PR diff| > 40 ms` is about `0/152`.
- beat/lead level P Missing either boundary remains `3005/25860` unchanged.
- `p_unreliable` leaves `2669` unchanged.
- `p_candidate_context_score` covers `22861/25860` beat/lead rows; in the final full batch, `p_candidate_suppressed_by_context` is `0`, indicating that the current stage mainly provides observable scores without sacrificing P detection integrity.

This step lays the foundation for the subsequent second round of candidate re-selection and raw rhythm residual searches: now you can first observe which candidates have low scores, and then decide whether to use a more reasonable alternative peak within the same P search window.

<a id="110-已实现的第十阶段修改p-candidate-context-re-selection"></a>
### 1.10 Implemented tenth stage modification: P candidate context re-selection

Alternative peak re-selection within the same lead/beat P search window is currently added after the original P detection. This stage does not add missing P, nor does QRST residual search; it only replaces the original `WaveBounds` when the current P candidate has a significantly low score, the alternative full geometry is legal, and the context score is significantly higher.

Implementation points:

1. Save top-k turning-point alternatives in the final P search window for each lead/beat.
2. Alternative: First, return to the local peak value near the raw ECG from the turning point of the smoothed signal to avoid the boundary measurement falling on the smoothed shoulder sample.
3. Remeasure P onset/peak/offset, PR, P duration, P area, P morphology and PTF-V1 for alternative.
4. Use the same set of template similarity, multi-lead support, PR consistency, and PP consistency to rescore.
5. The re-selection gate only allows obviously weak candidates of `old_score < 0.50` to enter replacement; borderline plausible candidates are not replaced even if the alternative is higher, to avoid unnecessary disturbance to report-level PR.

New diagnostic fields include `p_candidate_alternative_count`, `p_reselected_from_peak_index`, `p_reselected_to_peak_index`, `p_reselected_old_context_score`, `p_reselected_new_context_score`, and `p_reselection_reason`. Add the `p_candidate_reselected_by_context` flag when the replacement is successful.

LUDB full-batch comparison of the ninth stage:

- `tmp_p_context_guard_full/summary.csv`: PR MAE `9.5372 ms`, PR bias `-6.3022 ms`, `|PR diff| > 20 ms` is `17/152`, `|PR diff| > 40 ms` is `0/152`.
- Current `tmp_p_reselection_full/summary.csv`: PR MAE `9.5372 ms`, PR bias `-6.3022 ms`, `|PR diff| > 20 ms` is `17/152`, `|PR diff| > 40 ms` is `0/152`.
- beat/lead level P Missing either boundary remains `3005/25860` unchanged.
- `p_unreliable` leaves `2669` unchanged.
- `p_candidate_reselected_by_context` triggers `30` beat/lead rows in the full batch, distributed among `24` records; no record changes in report-level PR, indicating that the current gate is neutral for the final report, but has begun to improve the underlying raw `WaveBounds`.

Remaining questions:

- A few records will still show moderate deviations, for example, the PR of record 133 is still too long. The next step should not continue to rely solely on global PR post-processing, but should expand raw rhythm residual / second-pass consensus P search under the premise of higher confidence, or model P onset / P offset boundary confidence independently before entering.

So the original P-bound/PR aggregation problem can be summarized as:

1. **P peak has started to use fused anchor reselection, cluster support fusion and context scoring, but has not yet fully utilized these scores for alternative candidate reselection. **
2. **Global PR currently mainly uses the limb lead median of raw per-lead PR. In many cases, the calculated `pr_consensus_ms` is not used. **
3. **The boundary confidence of P onset/offset is not modeled separately; `p_confidence` reflects more P wave amplitude and noise, which is not enough to indicate that the boundary is accurate. **
4. **The representative waveform is currently used more as a prior, and the P boundary on the representative waveform has not been fully fed back to the final P onset/offset of each beat/lead. **
5. **P offset has a great influence on morphological characteristics such as P duration, P area, P terminal force, notched/biphasic, etc., and the current deviation will propagate backward. **

"Consensus" can and should be used to enhance P-wave boundary detection, but the current rules of `P10 onset` cannot be simply applied. The P wave amplitude is small, the shape is changeable, and it is easily disturbed by the T wave tail/baseline drift/low amplitude fluctuations of the PR segment. It is recommended to use the hierarchical consensus of "candidate clustering + representative waveform prior + multi-lead support + physiological constraints".

<a id="2-当前-p-波检测流程"></a>
## 2. Current P-wave detection process

<a id="21-代表波形-prior"></a>
### 2.1 Representative waveform prior

The code will first construct the representative beat prior of each group/lead:

- `build_group_priors(...)` will extract the representative waveform of each group and each lead from `representative_beats`.
- `_delineate_rep_prior(...)` will look for the a priori position of P/QRS/T on the representative waveform.
- P wave prior currently uses `_find_peak(...)` to find P peak, and then uses `_find_wave_bounds(..., frac=0.08)` to estimate P onset/offset.

The value of this prior is: it can provide a relatively stable time window for subsequent beat-level detection, reducing the degree of freedom of searching from scratch for each beat.

But there are two limitations here:

1. The P onset/offset representing the waveform prior is still a threshold boundary, not a multi-lead consensus boundary.
2. The current representative waveform is often medoid/single representative beat, not necessarily the median/average template; it has better morphological fidelity, but is not enough for low-amplitude P wave boundary noise reduction.

<a id="22-每个-beat-的多-lead-p-peak-预融合"></a>
### 2.2 Multiple lead P peak pre-fusion for each beat

In the P search window of each beat, the algorithm will first search for P candidates from all leads:

- `_candidate_triplets(...)` generates P peak candidates for each lead.
- `_fuse_peak_anchor(...)` merges multiple lead candidates into one `_fused_p_peak` based on amplitude, area, prior proximity, and reliable lead weight.

This step is the clearest “consensus” point of use in current P-wave detection. Its function is to give all leads a common P peak anchor point to prevent each lead from selecting completely different local fluctuations.

But it also has risks:

- If the fused peak is attracted by a false P, T tail residue or baseline fluctuation close to QRS, the P search windows of all subsequent leads will be biased.
- The current fusion mainly returns a peak index, which lacks sufficient provenance, such as supporting lead number, cluster spread, and competition between candidate clusters.
- P peak consensus cannot directly guarantee that P onset/offset is correct. When P peak is stable, the boundary may still shift significantly due to low-amplitude starting and ending points and baseline instability.

<a id="23-每个-lead-的-p-onsetoffset"></a>
### 2.3 P onset/offset of each lead

On each lead, the main process of P detection is roughly:

1. Narrow the P search window based on `_fused_p_peak` or the representative waveform prior.
2. Use `_find_peak(..., mode="abs")` to find P peak in the window.
3. Use `_select_p_peak_with_fused_anchor(...)` to check whether the abs-mode peak is sucked away by the false peak close to QRS; if the amplitude of the fused P peak reaches the standard and the timing is more reasonable, select the P peak back to the fused anchor.
4. Check whether the distance from P peak to R/QRS is within the physiological range; if constrained re-search is required, also apply the fused anchor selection again.
5. Use the tangent method to estimate P onset:
   - The current call is `_t_onset_tangent(...)`.
6. Use `_p_offset_tangent(...)` to estimate P offset.
7. If necessary, use `_rescue_long_pr_p_onset(...)` to rescue the P onset of the long PR forward.
8. If the geometric relationship is abnormal, `_p_wave_qrs_geometry_mode(...)` may clear P or clear P onset.

This means that the P onset/offset of each lead is now mainly a local geometric result. They are affected by:

- P-wave amplitude is low, and derivatives and tangent points are easily dominated by noise.
- If the baseline is estimated to be a little biased, the onset/offset will move significantly.
- P wave is biphasic, notched, and low, with a single tangent point unstable.
- The offset search end point can be to `qrs_on - 1`. If the PR segment fluctuates at low amplitude or the starting point of QRS is unstable, the offset can easily be dragged too close to QRS.
- The fallback's `_find_wave_bounds(..., frac=0.08)` is sensitive to amplitude and local noise, and different leads will give different lengths of P duration.

<a id="24-当前-multi-lead-consensus"></a>
### 2.4 Current multi-lead consensus

`_apply_multilead_consensus(...)` will do some consensus processing on multiple lead results of the same beat.

For P-wave related fields, the most critical ones at present are:

- Collect reliable lead `bf.p.onset`.
- Compute `global_p_onset` using an earlier quantile.
- Derive `pr_consensus_ms` for each lead.

But there are several important gaps here:

1. **P offset consensus is not calculated. **
2. **P duration consensus is not calculated. **
3. **The consensus P onset/offset is not written back to the final `WaveBounds`. **
4. **`global_p_onset` uses simple quantiles and lacks candidate clustering and cluster support judgment. **
5. **Subsequent global PR mainly does not use `pr_consensus_ms` as the main path. **

Therefore, the current "consensus" strengthening of the P boundary is partial and indirect, rather than a complete closed loop.

<a id="3-ludb-结果中看到的问题"></a>
## 3. Problems seen in the results of LUDB

<a id="31-总体-pr-表现"></a>
### 3.1 Overall PR performance

From `ludb_full_compare/summary.csv`, the PR indicator shows that the P onset related error is not small:

- Number of PR comparable records: 151
- Average deviation: about `-11.77 ms`
- MAE: approx. `16.85 ms`
- median diff: about `-10 ms`
- Absolute error p75: approx. `22.75 ms`
- Maximum absolute error: `88 ms`
- Records with PR error exceeding 20ms: 44/151

This shows that the current P onset has a systematic tendency to be shorter, that is, the algorithm P onset is often detected later, resulting in PR being shorter than the label.

In addition, there are many `p_unreliable` tags in the whole library, indicating that P/QRS geometric relationships and low support scenarios are not uncommon. Typical high percentage records include:

- 104: `126/132` beats marked P as unreliable.
- 45: `104/120` beats marked P as unreliable.
- 111: `91/108` beats marked P as unreliable.
- 90: `101/120` beats marked P as unreliable.

These records suggest that current per-lead P detection often enters the state of "unreliable but still producing partial values" on difficult samples.

<a id="32-记录-5p-peak-尚可p-offset-明显分散"></a>
### 3.2 Record 5: P peak is acceptable, P offset is obviously scattered

The global PR difference of record 5 is not big, the algorithm is about `130 ms`, and the GT is about `138 ms`. But looking at the P boundaries of each lead, the problem is obvious:

- lead I: P onset is about `157 ms` before R, P offset is about `50 ms` before R.
- lead II: P onset is about `162 ms` before R, P offset is about `100 ms` before R.
- lead aVF: P onset is about `158 ms` before R, P offset is about `104 ms` before R.
- lead V1: P onset is about `146 ms` before R, P offset is about `40 ms` before R.
- lead V5: P onset is about `152 ms` before R, P offset is about `42 ms` before R.

Here, P onset is roughly concentrated at `150-160 ms` before R, indicating that the P peak/leading edge anchor point is not completely out of control; but P offset ranges from `40 ms` to `104 ms` before R, resulting in serious inconsistency between P duration and PR segment length.

This is a typical manifestation of the current lack of P offset consensus: each lead determines P offset by itself, and ultimately there is no physiologically reasonable boundary across leads to correct.

<a id="33-记录-39已有-pr-consensus-更接近-gt但全局-pr-没用它"></a>
### 3.3 Record 39: There is already a PR consensus that is closer to GT, but the global PR does not use it

Record 39 is a very critical example:

- Algorithm global PR: about `136 ms`
- GT PR: approx. `224 ms`
- diff: about `-88 ms`

But judging from the per-lead representative results, some leads already have long PRs close to GT:

- aVF: approx. `210 ms`
- V3: approx. `221 ms`
- V4: approx. `222 ms`

The existing `pr_consensus_ms` is approximately `201 ms` on multiple leads, which is obviously closer to GT `224 ms` than the global PR `136 ms`.

This means:

1. The current algorithm has internally captured part of the "long PR consensus" information.
2. But in the end, the global PR of `compute_global_features(...)` mainly still takes the raw limb lead median path.
3. The raw limb median is shortened by short PR false positives such as I/III/aVR, resulting in a significantly smaller global PR.

This type of sample best illustrates: P onset consensus is valuable but currently underutilized for final features.

<a id="34-记录-57130简单-consensus-也可能错"></a>
### 3.4 Record 57/130: Simple consensus can also be wrong

It should be noted that the existing `pr_consensus_ms` cannot be directly and unconditionally replaced by a global PR.

Record 57:

- Algorithm global PR: about `225 ms`
- GT PR: approx. `139 ms`
- Currently `pr_consensus_ms` on multiple leads about `102 ms`

Here both raw PR and consensus PR may deviate from GT, but in different directions. Explain that currently simple onset quantiles/aggregations are not reliable enough.

Record 130:

- Algorithm global PR: about `122.5 ms`
- GT PR: approx. `207 ms`
- Multiple lead raw PRs and `pr_consensus_ms` are on the short side.

This shows that if the fused P peak or onset candidate initially locks into a false deflection close to QRS, the subsequent simple consensus will "consensus" the errors together.

Therefore, P-wave consensus enhancement must perform candidate clustering, support evaluation and physiological constraints. It cannot just calculate a quantile for all onsets.

<a id="4-为什么-p-onset-和-p-offset-容易错"></a>
## 4. Why P onset and P offset are easy to make mistakes

<a id="41-p-波信号本身弱"></a>
### 4.1 The P wave signal itself is weak

P waves are usually small in amplitude, low in slope, and do not start and end with clear peaks. QRS onset usually has a more obvious slope change, T offset can take advantage of longer unimodal/bimodal morphology, and P onset/offset is more easily affected by the following factors:

- baseline wander
- EMG noise
- P wave low and flat
- P wave biphasic or notched
- Small fluctuations on the PR segment
- Slight deviation in QRS onset detection

Therefore, the local tangent method with a single lead and a single beat is naturally not stable enough.

<a id="42-p-onset-被-fused-peak-窗口牵引"></a>
### 4.2 P onset is pulled by the fused peak window

After the current P peak is pre-fused, the P search window for each lead will be narrowed around `_fused_p_peak` or the representative prior. If the fused peak is correct, this will significantly improve stability; but if the fused peak is attracted to the wrong candidate, all leads will look for P waves around the wrong time.

This creates a kind of "wrong synchronization":

- Multiple leads can find a local wavelet.
- These wavelets are close in time and look like there is a consensus.
- But they are actually low-amplitude noise, T/P confusion or PR segment fluctuations before QRS.

Therefore, P peak consensus needs to output cluster support and competition cluster information, and cannot only output a peak index.

<a id="43-p-offset-没有跨-lead-校验"></a>
### 4.3 P offset is not verified across leads

P offset currently mainly uses `_p_offset_tangent(...)`, and `qrs_on - 1` was searched. In cases where the PR segment fluctuates or the onset of QRS is slightly off, the offset may be dragged very close to QRS.

Once the offset is too late, it will result in:

- P duration is too long.
- PR segment is compressed.
- P area is too large.
-P terminal force may be incorrectly expanded.
- P notch/biphasic judgment may be contaminated by PR segment or QRS pre-fluctuation.

Once the offset is too early, it will lead to:

- P duration is too short.
-P terminal component is truncated.
- Loss of the second half of the biphasic P wave.
- P area and PTF are underestimated.

This is the problem with offset scatter in record 5.

<a id="44-p-confidence-不是边界-confidence"></a>
### 4.4 P confidence is not boundary confidence

Currently `p_confidence` is mainly related to P amplitude, noise floor, and detection support. A lead can have high P amplitude and `p_confidence`, but still have the wrong onset/offset.

So it needs to be split:

- `p_presence_confidence`: Whether P wave is present.
- `p_peak_confidence`: Is P peak credible?
- `p_onset_confidence`: Is P onset credible.
- `p_offset_confidence`: Whether P offset is trustworthy.
- `p_boundary_consensus_support`: Whether the boundary is supported by multiple lead/representative waveforms.

Otherwise, the subsequent consensus will mistake "P's existence is very credible" for "P's boundary is very credible".

<a id="5-是否可以用共识加强"></a>
## 5. Can it be strengthened by "consensus"?

Yes, and it is recommended to split the P-wave consensus into 4 layers.

<a id="51-第一层p-peak-候选聚类共识"></a>
### 5.1 The first layer: P peak candidate clustering consensus

Currently `_fuse_peak_anchor(...)` has begun to enhance P peak fusion according to per-lead normalization, unique lead support and P-to-QRS onset timing. But it still only returns one selected peak, and it is recommended to continue to expand the output to candidate clusters:

- The center time of each cluster.
-Support lead number.
- Support lead territory, such as limb, inferior, precordial.
- Time dispersion within the cluster.
- Magnitude/area weighted fraction within cluster.
- Whether it is close to the representative waveform prior.
- Whether the reasonable PR interval is met.

Instead of just returning a fused peak, return:

- `selected_p_peak`
- `selected_cluster_support`
- `selected_cluster_spread_ms`
- `competing_clusters`
- `selection_reason`

In this way, subsequent onset/offset can know whether the current peak anchor point is stable.

<a id="52-第二层p-onset-边界共识"></a>
### 5.2 The second layer: P onset boundary consensus

For each beat, P10 should not be used directly after getting raw P onset for each reliable lead. Suggestions:

1. Collect raw P onset of reliable leads.
2. Eliminate obviously unreasonable candidates:
   - P onset is later than P peak.
   - PR is too short or too long, unless there is already paced/retrograde/AV block etc. context.
   - P duration is outside the relaxed range.
   - onset does not belong to the same P waveform as the currently selected P peak.
3. Cluster onset times.
4. Select the onset cluster that is consistent with the P peak cluster, representative waveform prior, and lead support.
5. Use robust statistic to obtain onset consensus within the cluster, such as median or slightly early percentile.

The key here is: for P onset, there are different costs for early onset and late onset.

- If you want to measure PR, late onset will systematically underestimate PR.
- But if the onset is too early, the T wave tail/noise will be included in P.

Therefore, it is recommended to use the robust early boundary within the cluster instead of the P10 for the overall onset:

- Strong cluster support and small spread: cluster P25 or median can be used.
- Cluster supports general: use median and keep lower confidence.
- There is competition between long PR clusters and short PR clusters: give priority to the representative waveform prior, lead territory support and P peak consistency, do not just choose based on the largest amplitude.

<a id="53-第三层p-offset-边界共识"></a>
### 5.3 The third layer: P offset boundary consensus

This is the layer that is most lacking right now.

It is recommended to calculate for each beat:

- `p_offset_consensus_idx`
- `p_duration_consensus_ms`
- `pr_segment_consensus_ms`
- `p_offset_support_leads`
- `p_offset_spread_ms`

Candidate rules:

1. Each reliable lead gives raw P offset.
2. Delete obvious abnormal offset:
   - offset is earlier than peak.
   - Offset late arrival too close to QRS onset unless explicitly short PR/pre-excitation.
   - P duration exceeds the relaxed upper limit, e.g. exceeds 160-180ms and there is no evidence of strong bimodal/abnormal morphology.
   - P duration is too short, for example less than 40-50ms.
3. Cluster the offset candidates.
4. Prioritize clusters that form a reasonable P duration with onset consensus and P peak consensus.
5. Add PR segment guard:
   - Under normal circumstances `p_offset` should be some distance earlier than `qrs_onset`.
   - If the offset is close to QRS, there must be strong support from multiple leads, otherwise it will be regarded as noise/PR segment fluctuation before QRS.

For the scenario of record 5, offset consensus can prevent the offset of some leads from being dragged to 40ms before R, and can also prevent other leads from being truncated prematurely. The final P duration should fall within the range supported by multiple leads.

<a id="54-第四层代表波形模板共识"></a>
### 5.4 The fourth layer: represents waveform template consensus

It is recommended to keep the current medoid representative beat, but build an additional median/trimmed-mean template specifically for low-magnitude boundaries:

- medoid: retains the true form, suitable for display and local morphological measurement.
- Median template: Reduces random noise and fits P onset/offset boundaries.

The process can be:

1. Press R peak to align beats in the same group.
2. Construct a median template for each lead.
3. Detect P onset/offset on template.
4. Use template P bounds as per-beat/per-lead prior or soft constraint.
5. If the difference between the per-beat raw boundary and the template boundary is too large, but there are no multiple leads to support it, downgrade it or replace it with consensus.

This would particularly benefit P onset/offset because of the low amplitude of the P wave boundary and the large impact of beat-to-beat noise.

<a id="6-建议的改进设计"></a>
## 6. Suggested design improvements

<a id="61-数据结构层面"></a>
### 6.1 Data structure level

It is recommended to add or retain the following provenance fields in beat-level features:

- `p_onset_raw_ms`
- `p_offset_raw_ms`
- `p_onset_consensus_ms`
- `p_offset_consensus_ms`
- `p_duration_consensus_ms`
- `p_peak_consensus_support`
- `p_onset_consensus_support`
- `p_offset_consensus_support`
- `p_boundary_spread_ms`
- `p_boundary_confidence`
- `p_boundary_source`

`p_boundary_source` can be obtained by:

- `raw_local`
- `lead_consensus`
- `template_prior`
- `template_and_lead_consensus`
- `suppressed`

This way subsequent debugging can know where each P boundary comes from.

<a id="62-算法层面"></a>
### 6.2 Algorithm level

It is recommended to modify in the following order:

1. **Enhancement `_fuse_peak_anchor(...)`**
   - Changed from single peak output to cluster-level output.
   - Retain support for lead, cluster spread, and candidate competition information.

2. **Newly added `_cluster_p_boundaries(...)`**
   - Enter multiple lead raw P onset/offsets of the same beat.
   - Output consensus onset, offset, duration, support, spread.

3. **Extension `_apply_multilead_consensus(...)`**
   - Currently only `pr_consensus_ms` is derived.
   - Increase P onset consensus, P offset consensus, P duration consensus.
   - Downgrade or flag unreliable raw boundaries.

4. **Modify global PR aggregation**
   - Currently `_global_pr_median(...)` mainly uses raw limb PR.
   - It is recommended that supported consensus PR be used first when consensus support is strong and raw limb spread is large.
   - But `pr_consensus_ms` cannot be used unconditionally, cluster support and physiological constraints are required.

5. **Represents a new median template for the waveform**
   - medoid remains.
   - median template for P bound prior.

6. **P Morphological features use boundary confidence**
   - P area, P terminal force, notched/biphasic should record whether raw boundary or boundary consensus is used.
   - If boundary confidence is low, morphological conclusions should also be downgraded.

<a id="63-选择-consensus-的建议规则"></a>
### 6.3 Suggested rules for selecting consensus

You can use a conservative version first:

1. If there are less than 3 reliable P leads, the raw boundary will not be forcibly covered, and only the low confidence of the consensus will be recorded.
2. If onset cluster support >= 3 and spread <= 20ms, high-confidence onset consensus can be generated.
3. If offset cluster support >= 3 and spread <= 25ms, and P duration is within a reasonable range, high-confidence offset consensus will be generated.
4. If the difference between raw PR median and consensus PR is > 40ms, check:
   - Which has more lead support.
   - Which is closer to the representative template prior.
   - Which produces more reasonable P duration and PR segment.
5. If there are two P onset clusters, one short PR and one long PR:
   -Select directly whether morning or evening, regardless of time.
   - Check whether the P peak cluster has the same origin.
   - Check if limb/inferior lead is supported.
   - Check if template prior is supported.
   - Check whether the P duration/offset of the cluster is physiologically reasonable.

<a id="7-优先修复点"></a>
## 7. Prioritize repair points

<a id="p0不要只把-prconsensusms-当旁路字段"></a>
### P0: Don’t just use `pr_consensus_ms` as a bypass field

Record 39 shows that the current `pr_consensus_ms` is sometimes significantly better than the global PR, but the global PR does not use it. It is recommended to implement a conservative gating first:

- When the raw limb PR spread is large;
- and consensus PR support is sufficient;
- And the consensus PR is consistent with the representative waveform prior;
- and the consensus PR is within a reasonable physiological range;

Then the global PR is estimated using consensus PR or consensus/raw hybrid.

This step can first improve a batch of cases with late P onset resulting in short PR.

<a id="p1增加-p-offset-consensus"></a>
### P1: Increase P offset consensus

P offset is currently the maximum gap. It is recommended not to rush to cover all lead `WaveBounds`, but to add fields first:

- `p_offset_consensus_ms`
- `p_duration_consensus_ms`
- `p_offset_consensus_support`

Then give priority to consensus P duration with high support among representative/global features.

<a id="p2把-p-boundary-confidence-从-p-presence-confidence-中拆出来"></a>
### P2: Separate P boundary confidence from P presence confidence

If it is not dismantled, the situation of "the existence of P is certain, but the boundary is wrong" will continue to occur. It is recommended to add boundary-level confidence and use it in P area/PTF/notched/biphasic.

<a id="p3代表波形从-medoid-扩展到-median-template"></a>
### P3: Representative waveform extended from medoid to median template

The P wave boundary is a low-amplitude problem, and the median template will be more stable for onset/offset than the single beat medoid. It is recommended to retain medoid as a representative display and add median template as boundary prior.

<a id="8-推荐验证集"></a>
## 8. Recommended verification set

It is recommended to conduct regression verification around the following records:

- **Record 5**: Focus on verifying whether P offset consensus can reduce offset dispersion among leads.
- **Record 39**: Focus on verifying whether the supported consensus PR can correct the short global PR.
- **Log 57**: Prevent naive consensus from pulling PRs to wrong short PRs.
- **Record 120**: Verify whether the algorithm can avoid being dominated by short PR false candidates when there is a long PR supporting lead.
- **Log 130**: When verifying fused peak errors, can cluster/provenance identify low confidence instead of error consensus.

Recommended automated checks:

1. Whether the PR MAE decreases.
2. Whether the PR bias converges from the current approximately `-11.8 ms` to 0.
3. PR > 20ms outlier number is declining.
4. Whether the IQR between leads of P duration decreases.
5. Whether the abnormal close ratio of P offset to QRS onset decreases.
6. Whether the abnormally large value of P area/PTF is reduced.

<a id="9-小结"></a>
## 9. Summary

The problem with current P-wave detection is not just that a certain threshold is too loose or too tight, but that the P boundary has not yet formed a complete closed loop of "multi-lead, multi-beat, representative waveform".

Now you have a good foundation:

- per lead/per beat results are complete;
- Represents the waveform prior to existing;
- P peak already has cross-lead fusion;
- `pr_consensus_ms` has been able to capture better information in some cases.

The next step should be to advance the consensus from P peak to P onset/P offset/P duration, and make the global PR and P morphological features truly use a supported consensus boundary. In this way, late P onset, scattered P offset, unstable P duration, and the resulting PR/PTF/P morphology errors can be systematically improved.

<a id="10-qrst-residual-raw-p-event-反哺更新"></a>
## 10. QRST residual raw P event feedback update

Referring to the raw rhythm P/atrial activity route in `/home/chtmedgemma/projects/ecg_gemma/P wave`, the current implementation adds a more complete but still conservative closed loop:

1. The composite QRST-residual detector in `feature_extraction/ecgfeat/atrial.py` no longer only outputs the peak/sample of the P event, but outputs it simultaneously `onset_sample`, `offset_sample`, `duration_ms`, `amplitude_mv`, `area_mv_ms`, `signed_area_mv_ms`, `template_similarity`, `pp_ms` and other measurement fields.
2. `feature_extraction/ecgfeat/delineate.py` will call the composite raw P detector after scoring the per-lead/per-beat P context to convert the high-confidence raw atrial event into the P alternative corresponding to the beat/lead.
3. The raw event does not directly cover the original P detection; it only enters the reselection gate that has template similarity, PR consistency, PP consistency, and multi-lead support as a candidate.
4. The ordinary alternative still only examines the candidates of `old_score < 0.50`; the raw residual event can examine the borderline candidates of `old_score < 0.65`, but in the end it must still meet the obvious context improvement.
5. The export layer retains the new measurement field of raw P event to facilitate subsequent rhythm analysis, AV block/blocked P inspection, and debugging P detection quality.

LUDB full-batch verification:

- `tmp_p_raw_atrial_context_full/summary.csv` has no change compared to the current `ludb_full_compare/summary.csv`, PR/QRS/QT/QTc/T axis/QT dispersion/missing counts.
- P axis MAE minimal improvement: `17.3203977896` -> `17.3201184719`, changes from two raw-event context reselection of record 100.
- `2203` composite raw atrial events are generated out of the total 200, all with `duration_ms`; the actual lead-beats that trigger raw-event P reselection are `7`.
- The production code scan did not find the record ID, LUDB file name or specific record condition branch; the trigger condition only relies on raw residual event confidence, PR window, template/PP/PR/support context.

Conclusion: This step mainly completes the measurement and feedback channels of raw rhythm P event, which is an infrastructure improvement. It does not significantly change the existing report indicators, but it allows the P wave raw detection to have the QRST residual + contextual second opinion required by the document, providing a safe entrance for subsequent more aggressive P onset/offset raw boundary corrections.
