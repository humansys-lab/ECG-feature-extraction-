<!-- i18n-nav -->
[中文](p_wave_rerun_result_analysis.md) | [English](p_wave_rerun_result_analysis.en.md) | [日本語](p_wave_rerun_result_analysis.ja.md)
<!-- /i18n-nav -->

> Translation note: This English version is provided for convenience. If it differs from the source document, the source document prevails.

<a id="p-波相关重跑结果分析"></a>
# Analysis of P-Wave Related Rerun Results

Analysis Subject: Current rerun results for `ludb_full_compare`.

Result Timeline:

- `ludb_full_compare/summary.csv` Update Time: 2026-07-05 16:55.
- All 200 `*_features.json` have been generated.
- New fields have entered the results, such as `p_offset_consensus_index`, `p_dur_consensus_ms`, `pr_segment_consensus_ms`, `p_boundary_consensus_source`.

<a id="1-总体结论"></a>
## 1. Overall Conclusion

The P-wave related modifications have taken effect, particularly providing significant help for a batch of cases where **raw limb PR was shortened by false P waves near QRS**. However, the overall PR metrics for the entire database have not significantly improved, indicating that the conservative consensus in the first phase only resolved partial scenarios. The remaining outliers are still concentrated in issues such as P onset cluster selection, excessive PR core rescue, and instability in P axis/P morphology.

Key Observations:

1. **PR consensus gating is valuable.**
   The MAE of the current actual global PR is `17.02 ms`; if the raw limb median baseline is back-calculated using current features, the MAE is approximately `26.82 ms`. This means that the global aggregation/rescue path is indeed saving a batch of cases.

2. **However, the current overall database PR metrics still show a slight negative bias.**
   The current PR bias is `-12.02 ms`, and the median diff is `-10 ms`, indicating that the P onset is still generally late.

3. **P offset consensus coverage is good, but it should not directly replace raw morphology yet.**
   Beat-level `p_offset_consensus_index` covers approximately `59.91%` of lead-beat rows; the representative-level P duration consensus median is approximately `80 ms`. However, short/long outliers still exist, for example, the consensus P duration for record 57 can reach `152 ms`.

4. **P axis remains a significant weakness.**
   The P axis MAE is approximately `24.25°`, with 14 records having an absolute error exceeding `90°`. This part is mainly not solvable by PR consensus, but rather issues related to P polarity, P signed area, low-amplitude P waves, and lead support.

5. **There are still many records with high P missing rates.**
   For example, records 104, 45, 111, 90, 34, and 74 all have high algorithm P missing rates. These are more likely issues related to P existence/low-amplitude P/rhythm context, rather than simple onset/offset boundary problems.

<a id="2-全局-pr-统计"></a>
## 2. Global PR Statistics

PR metrics in the current `summary.csv`:

| Metric | Value |
|---|---:|
| Number of comparable records | 151 |
| bias | `-12.02 ms` |
| MAE | `17.02 ms` |
| median diff | `-10.0 ms` |
| abs p75 | `24.25 ms` |
| abs p90 | `39.0 ms` |
| max abs | `86.0 ms` |
| `abs(diff_pr) > 20 ms` | 47 |
| `abs(diff_pr) > 40 ms` | 13 |
| `abs(diff_pr) > 60 ms` | 4 |
| algorithm PR missing | 48 |
| GT PR missing | 25 |

Compared to the old statistics in the document before the first phase of modifications:

- The old PR MAE was approximately `16.85 ms`, and the current `17.02 ms` is roughly flat but slightly worse.
- The old `abs(diff_pr) > 20 ms` was 44 records, and the current count is 47.
- The old max abs was `88 ms`, and the current is `86 ms`.

Therefore, this modification did not result in a global one-size-fits-all improvement; instead, it pulled back some severe negative bias outliers, while other P-related issues remain unresolved.

<a id="3-pr-consensus-的实际效果"></a>
## 3. Actual Effect of PR Consensus

<a id="31-反算-raw-limb-baseline"></a>
### 3.1 Back-calculating Raw Limb Baseline

Back-calculated using current `representative_leads`:

| Path | n | bias | MAE | median | abs p75 | max abs |
|---|---:|---:|---:|---:|---:|---:|
| Current actual global PR | 151 | `-12.02` | `17.02` | `-10.0` | `24.25` | `86.0` |
| raw limb median baseline | 151 | `-22.34` | `26.82` | `-18.0` | `32.5` | `131.0` |
| naive consensus median | 151 | `-19.15` | `21.37` | `-18.0` | `27.0` | `123.0` |

Conclusion:

- Direct raw limb median is significantly worse.
- Blindly using consensus median is also worse.
- The current conservative gating is necessary.

<a id="32-明确使用-pr-consensus-的记录"></a>
### 3.2 Records Explicitly Using PR Consensus

There are currently 12 records where it is clear that the global PR adopted `pr_consensus_ms`:

| record | raw limb median | actual PR | GT PR | raw diff | new diff |
|---:|---:|---:|---:|---:|---:|
| 17 | 115.5 | 199.0 | 207.0 | -91.5 | -8.0 |
| 23 | 106.5 | 145.0 | 210.0 | -103.5 | -65.0 |
| 39 | 136.0 | 201.0 | 224.0 | -88.0 | -23.0 |
| 40 | 129.0 | 164.0 | 191.5 | -62.5 | -27.5 |
| 49 | 98.5 | 136.0 | 174.0 | -75.5 | -38.0 |
| 50 | 92.0 | 122.0 | 139.0 | -47.0 | -17.0 |
| 61 | 82.0 | 146.0 | 186.0 | -104.0 | -40.0 |
| 73 | 80.5 | 126.0 | 170.0 | -89.5 | -44.0 |
| 75 | 76.0 | 136.0 | 156.0 | -80.0 | -20.0 |
| 102 | 98.0 | 162.0 | 201.0 | -103.0 | -39.0 |
| 120 | 159.0 | 186.0 | 220.0 | -61.0 | -34.0 |

Additionally, record 52 adopted consensus, but the GT PR is empty, making evaluation impossible.

This indicates that PR consensus is very effective for cases where "raw PR is lowered by short false P waves," but it is still insufficient for records 23/61/73/102/120, suggesting that the onset consensus is still too late.

<a id="4-当前-pr-outlier"></a>
## 4. Current PR Outliers

Records with the largest current PR errors:

| record | diff_pr | algorithm PR | GT PR | Characteristics |
|---:|---:|---:|---:|---|
| 57 | +86.0 | 225.0 | 139.0 | raw limb PR is long; short consensus=102 was rejected by protection mechanism |
| 130 | -84.5 | 122.5 | 207.0 | both raw and consensus are locked in a short PR cluster near QRS |
| 182 | +68.12 | 202.12 | 134.0 | raw/consensus are both short, but the subsequent PR core rescue is too long |
| 23 | -65.0 | 145.0 | 210.0 | consensus saved, but still short |
| 64 | -48.0 | 127.0 | 175.0 | consensus=136, still short, not triggering enough correction |
| 22 | -46.5 | 125.0 | 171.5 | The onset spread is large, and the raw limb median is short |
| 180 | -46.0 | 124.0 | 170.0 | Both raw and consensus are short |
| 73 | -44.0 | 126.0 | 170.0 | consensus saved, but still short |
| 139 | +44.0 | 206.0 | 162.0 | raw limb PR is too long, consensus=124 is too short |
| 27 | -41.5 | 122.0 | 163.5 | consensus=142 but not taken over |

<a id="5-失败模式分类"></a>
## 5. Failure mode classification

<a id="51-长-pr-没救够"></a>
### 5.1 Long PR is not enough

Representative records: 23, 39, 40, 49, 61, 73, 75, 102, 120, 130.

Performance:

- raw limb median is significantly shorter than GT.
- `pr_consensus_ms` can pull PR in the long direction, but many records are still 30-60ms shorter than GT.
- `p_onset_consensus_spread_ms` is often very large, indicating severe fragmentation of multiple lead onset candidates.

Root cause judgment:

The current P onset consensus is still a relatively coarse aggregate of all onsets, lacking real onset cluster selection. It avoids the latest false P, but does not yet stably select the “earlier and true” long PR cluster when there are multiple P onset clusters.

What to do next:

- Add cluster-level consensus to P onset, instead of just recording the overall onset spread.
- Record support, lead territory, cluster center, and distance from representative prior for each onset cluster.
- Allowing selection of earlier clusters for long PR scenarios, but must have multiple lead and representative waveform support.

<a id="52-短-pr-假-p-仍主导"></a>
### 5.2 Short PR False P is still dominant

Representative records: 130, 64, 22, 180, 27, 9, 29.

Performance:

- `pr_consensus_ms` is also short, indicating that the fused P peak/onset has been locked to the false deflection close to QRS.
- P offset consensus may appear stable, but this is stability based on the wrong P wave.

Root cause judgment:

This is "false consensus": multiple leads all find candidates at small fluctuations close to QRS, with consistent timing, which looks like consensus, but it is not the actual P onset.

What to do next:

- P peak fusion should not only output a peak index, but also need to output candidate clusters.
- When short PR clusters compete with long PR clusters, they cannot be selected based on amplitude/time consistency alone.
- PR segment, P duration, lead territory, and representative template prior should be added for joint scoring.

<a id="53-pr-被救得过长"></a>
### 5.3 PR was saved for too long

Representative records: 57, 182, 139.

record 57:

- raw limb median = 225ms, GT = 139ms.
- `pr_consensus_ms = 102ms`, is rejected by the current gating, which is properly protected.
- But the raw limb PR is still too long, indicating that false early P / T-tail contamination needs to be identified and cannot only prevent short consensus.

record 182:

- raw limb median about 113.5ms.
- consensus median about 109ms.
- Final algorithm PR = 202.12ms.
- This prompts that the subsequent `_physiologic_pr_core_from_beats(...)` or API layer PR rescue will overextend the PR.

record 139:

- raw limb median about 206ms, GT = 162ms.
- consensus median about 124ms.
- raw is too long, consensus is too short, and the real value is in the middle.

What to do next:

- PR core rescue cannot be overwritten just because of current PR <120 或 >240. You need to check whether raw lead median, consensus median, and P onset cluster are supported.
- Added protection for forward outliers: if the raw PR is very long but the consensus/offset/P duration suggests another cluster, the raw limb median cannot be trusted directly.
- A "middle cluster" strategy is required instead of either raw or consensus.

<a id="54-p-offset-consensus-已有覆盖但仍需谨慎"></a>
### 5.4 P offset consensus has been covered, but caution is still required

Currently covered:

- Total number of beat-level rows: 25860.
- `p_onset_consensus_index` Coverage: 23940 rows, approximately `92.58%`.
- `p_offset_consensus_index` Coverage: 15492 rows, about `59.91%`.
- representative-level records with P offset consensus: 183/200.

P duration consensus:

- median: `80 ms`
- p75: `90 ms`
- p90: `106 ms`
- max: `152 ms`

P offset support：

- median support: 7 leads
- p90 support: 11 leads
- offset spread median: 23ms
- offset spread p90: 40ms
- max spread: 68ms

Description:

- P offset consensus coverage is good, and the majority supports enough leads.
- But the offset spread p90 has reached 40ms, indicating that the difficult cases are still scattered.
- The P duration consensus of record 57 is 152ms, which is too long; the P duration of record 24/171 is too short. It cannot be used unconditionally for P morphology/PTF right away.

What to do next:

- Add confidence level to P offset consensus.
- Mark low confidence when `p_dur_consensus_ms <60` or `>140`.
- Before using consensus boundary in morphology/PTF, check P duration, PR segment, and lead territory support.

<a id="55-p-axis-仍是明显问题"></a>
### 5.5 P axis is still an obvious problem

Current P axis statistics:

| Indicators | Values |
|---|---:|
| n | 172 |
| bias | `-10.12°` |
| MAE | `24.25°` |
| median diff | `-1.97°` |
| abs p90 | `58.10°` |
| max abs | `228.83°` |
| abs > 30° | 36 |
| abs > 60° | 17 |
| abs > 90° | 14 |

Maximum outlier:

- record 92: diff `-228.8°`
- record 106: diff `-193.0°`
- record 10: diff `-181.2°`
- record 133:diff `+160.6°`
- record 31: diff `-144.7°`
- record 17: diff `-142.7°`
- record 127: diff `+140.4°`

Root cause judgment:

P axis depends on P signed area / P amplitude / lead polarity. The current first phase mainly improves PR onset/offset and will not automatically repair P axis. P axis outlier is often related to low amplitude P, wrong P polarity, wrong P area integration window, and insufficient lead support.

What to do next:

- P axis should not only use raw P area; high-confidence P boundary or representative/template P area should be used first.
- Do cross-lead consistency checks on limb lead P polarity.
- Check lead reversal, P wave polarity cluster, low amplitude P support for obviously abnormal axes, such as close to 180° or flipped.

<a id="6-高-p-missing-记录"></a>
## 6. High P missing records

Records with high P missing rate:

| record | P missing rate | algorithm P missing | total | PR |
|---:|---:|---:|---:|---|
| 104 | 95.5% | 126 | 132 | None |
| 45 | 86.7% | 104 | 120 | None |
| 111 | 84.3% | 91 | 108 | None |
| 90 | 84.2% | 101 | 120 | None |
| 34 | 80.8% | 97 | 120 | None |
| 74 | 80.0% | 48 | 60 | None |
| 44 | 65.2% | 86 | 132 | None |
| 99 | 63.5% | 99 | 156 | None |
| 51 | 56.7% | 68 | 120 | None |
| 95 | 53.3% | 64 | 120 | None |

Such records should not be resolved by onset/offset tuning first, but should be classified first:

- Is it real no P or AF/flutter-like.
- Whether the P wave is too low in amplitude and is suppressed by `low_p_support`.
- Whether pacing/unknown context suppresses P/PR.
- Whether the representative group selection is not suitable for P-wave measurements.

<a id="7-建议下一轮修改优先级"></a>
## 7. It is recommended to modify the priority in the next round

<a id="p0给-p-onset-做真正-cluster-consensus"></a>
### P0: Make real cluster consensus for P onset

There is currently a P offset cluster, but P onset is still missing selected cluster provenance. The next step should be to add:

- `p_onset_cluster_center_index`
- `p_onset_cluster_support`
- `p_onset_cluster_spread_ms`
- `p_onset_cluster_leads`
- `p_onset_cluster_rank`
- `p_onset_consensus_reason`

The goal is to differentiate between:

- Short PR cluster near QRS.
- Earlier true P onset cluster.
- T-tail/baseline drift cluster.

<a id="p1限制-pr-core-rescue-过度覆盖"></a>
### P1: Limit PR core rescue excessive coverage

Record 182 shows that subsequent PR rescue may drag the originally short raw/consensus PR to 202ms. Suggestions:

- If raw limb median and `pr_consensus_ms` are both within 100-130ms, PR core rescue should not directly cover >180ms unless there is a clear multi-lead long PR onset cluster.
- PR core rescue should record provenance, for example `pr_source=physiologic_pr_core_rescue`.
- Output the debug field for records with a difference >50ms before and after rescue.

<a id="p2对正向-pr-outlier-做-false-early-plong-pr-识别"></a>
### P2: Do false early-P/long-PR identification for forward PR outlier

Record 57/139 explains that raw limb PR can be elongated by early false P. Suggestions:

- If the raw PR is very long but the consensus P duration/PR segment supports a shorter cluster, downgrade the raw long PR first.
- Do not use short consensus directly, but use cluster scoring to select intermediate trusted clusters.

<a id="p3p-offset-consensus-加-confidence不直接进-morphology"></a>
### P3: P offset consensus plus confidence, do not enter morphology directly

It is recommended to add:

- `p_offset_consensus_confidence`
- `p_duration_consensus_reliable`
- `p_boundary_consensus_reject_reason`

Rules:

- support < 4: low confidence.
- offset spread > 35-40ms: low confidence.
- P duration <60ms 或 >140ms: low confidence, unless supported by special forms.

<a id="p4p-axis-单独修"></a>
### P4: P axis is repaired separately

P axis is different from PR and should be in a separate round:

- Use representative/template P signed area.
- Add limb-lead P polarity consistency.
- Reduce the weight of low-volume leads.
- Output a conservative value or unavailable for the suspected lead reversal P axis.

<a id="8-推荐回归记录"></a>
## 8. Recommended regression records

In the next round, it is recommended to fix these records and make small batches:

- PR consensus improvements: 17, 39, 40, 75, 120.
- Long PRs still lacking: 23, 61, 73, 102, 130.
- Positive outlier: 57, 139, 182.
- P offset boundaries: 5, 24, 57, 171.
- P axis outlier: 10, 17, 31, 92, 106, 127, 133.
- High P missing: 34, 45, 74, 90, 104, 111.

Recommended commands:

```bash
.venv_report_regen_20260625/bin/python compare_annotations.py \
  --batch-reports \
  --records 5 17 23 39 40 57 61 73 75 102 120 130 139 171 182 \
  --out-dir tmp_p_wave_next_regression \
  --no-visuals
```

<a id="9-小结"></a>
## 9. Summary

This rerun proves that the direction of the first stage is correct: P boundary consensus and PR consensus gating can save a batch of obvious P onset late/short PR outliers. Record 39 is the most typical example.

However, the current P-wave module has not yet truly completed the closed loop of "each lead, each beat -> representative waveform -> multi-lead cluster consensus". The most worthwhile next steps are **P onset cluster consensus** and **PR rescue provenance/gating** rather than continuing to adjust a single threshold. There is already a basis for P offset consensus, but confidence needs to be added before entering more sensitive features such as morphology/PTF/P axis.
