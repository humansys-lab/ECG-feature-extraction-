#!/usr/bin/env python3
"""Build auditable cross-dataset metrics, cluster bootstrap and report artifacts."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
KEYS = ('dataset','scope','wave','lead','protocol')
NAMES = {'incartdb':'INCART','svdb':'SVDB','pwave':'PWAVE'}
SCOPES = {'detector_full':'全时长 QRS 检测器','pipeline_sampled':'固定片段完整提取','pipeline_full':'全时长完整提取'}


def f1(counts):
    tp,fp,fn=np.asarray(counts).T
    return 2*tp/np.maximum(2*tp+fp+fn,1)


def paired_bootstrap(rows):
    groups=defaultdict(lambda:defaultdict(dict))
    for row in rows:
        key=tuple(row[k] for k in KEYS)
        patient=row['patient']
        previous=groups[key][patient].get(row['method'],np.zeros(3))
        groups[key][patient][row['method']]=previous+np.array([int(row[k]) for k in ('tp','fp','fn')])
    results=[]
    for key,patients in sorted(groups.items()):
        if any(set(v)!={'default','enhanced'} for v in patients.values()):
            raise ValueError(f'unpaired cohort: {key}')
        default=np.array([v['default'] for _,v in sorted(patients.items())])
        enhanced=np.array([v['enhanced'] for _,v in sorted(patients.items())])
        rng=np.random.default_rng(20260922)
        draws=rng.integers(0,len(default),size=(5000,len(default)))
        delta=100*(f1(enhanced[draws].sum(axis=1))-f1(default[draws].sum(axis=1)))
        results.append(dict(zip(KEYS,key), clusters=len(default), bootstrap_repetitions=5000,
                            cluster_unit='patient' if key[0]=='incartdb' else 'source recording',
                            delta_f1_pp=float(100*(f1(enhanced.sum(axis=0))-f1(default.sum(axis=0)))),
                            ci95_pp=[float(v) for v in np.percentile(delta,[2.5,97.5])]))
    return results


def percent(value):
    return '—' if value is None else f'{100*value:.2f}'


def metric_table(metrics):
    lines=['| 数据 / 范围 | 输出 / 容差 | 配置 | TP / FP / FN | Se % | PPV % | F1 % | 峰值 MAE ms |',
           '|---|---|---|---:|---:|---:|---:|---:|']
    for m in metrics:
        mae='—' if m['peak_mae_ms'] is None else f"{m['peak_mae_ms']:.2f}"
        lines.append(f"| {NAMES[m['dataset']]} / {SCOPES[m['scope']]} | {m['lead']} / {m['protocol'].removeprefix('peak_')} | "
                     f"{m['method']} | {m['tp']} / {m['fp']} / {m['fn']} | {percent(m['sensitivity'])} | "
                     f"{percent(m['precision'])} | {percent(m['f1'])} | {mae} |")
    return lines


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'ecgfeat_cross_dataset_20260922')
    args=parser.parse_args(); out=args.out
    metrics=[]; novel=[]; rows=[]; records=[]; failures=[]; executions={}; manifests={}
    for dataset,expected in [('incartdb',300),('svdb',312),('pwave',24)]:
        folder=out/dataset
        summary=json.loads((folder/'summary.json').read_text())
        if summary['completed_tasks']!=expected:
            raise ValueError(f'incomplete cohort: {dataset}')
        metrics.extend(summary['metrics']); novel.extend(summary['excluding_prior_source_records'])
        failures.extend(summary['failures'])
        with (folder/'detection_by_record.csv').open() as handle:
            rows.extend(csv.DictReader(handle))
        records.extend(json.loads((folder/'records.json').read_text()))
        executions[dataset]=json.loads((folder/'execution.json').read_text())
        manifests[dataset]=json.loads((folder/'manifest.json').read_text())
    frozen=json.loads((out/'production_source_sha256.json').read_text())
    current={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted((ROOT/'feature_extraction/ecgfeat').rglob('*.py'))}
    source_unchanged=current==frozen
    evaluated_frozen=all(m['source_sha256']==frozen for m in manifests.values())
    for dataset,manifest in manifests.items():
        expected={(t['record'],t['scope'],t['variant']) for t in manifest['tasks']}
        actual={(r['record'],r['scope'],r['variant']) for r in records if r['dataset']==dataset}
        assert actual==expected, f'task coverage differs from predeclared manifest: {dataset}'
    for record in records:
        intended=90 if record['scope']=='pipeline_sampled' else record['duration_seconds']
        assert abs(record['evaluated_seconds']-intended)<1e-6
        assert abs(sum(t['core_seconds'] for t in record['timings'])-intended)<1e-6
    composition=json.loads((out/'pwave/composition_manifest.json').read_text())
    assert composition['unique_complete_tasks']==24 and composition['unchanged_scoring_rows']==216
    for component in composition['components']:
        manifest=json.loads((out/component['manifest']).read_text())
        assert manifest['source_sha256']==frozen
        for path,digest in component['checkpoints'].items():
            assert hashlib.sha256((out/path).read_bytes()).hexdigest()==digest
    for r in rows:
        assert int(r['tp'])+int(r['fn'])==int(r['gt_count'])
        assert int(r['tp'])+int(r['fp'])==int(r['det_count'])
    paired=defaultdict(dict)
    for r in rows:
        paired[tuple(r[k] for k in ('record',*KEYS))][r['method']]=r
    assert all(set(v)=={'default','enhanced'} and v['default']['gt_count']==v['enhanced']['gt_count'] for v in paired.values())
    uncertainty=paired_bootstrap(rows)
    # Record-level regressions retain both improved and worsened cases.
    changes=[]
    for key,v in paired.items():
        a,b=v['default'],v['enhanced']
        if a['protocol'] != ('peak_75ms' if a['wave']=='QRS' else 'peak_150ms'):
            continue
        if a['lead'] not in {'QRS','P_native'}:
            continue
        ca=[int(a[k]) for k in ('tp','fp','fn')]; cb=[int(b[k]) for k in ('tp','fp','fn')]
        changes.append(dict(record=a['record'],**{k:a[k] for k in KEYS},default_f1=float(f1(ca)),
                            enhanced_f1=float(f1(cb)),delta_f1_pp=float(100*(f1(cb)-f1(ca))),
                            default_counts=ca,enhanced_counts=cb))
    symbol_groups=defaultdict(lambda:np.zeros(3,dtype=int))
    for dataset in NAMES:
        with (out/dataset/'qrs_by_symbol.csv').open() as handle:
            for row in csv.DictReader(handle):
                key=tuple(row[k] for k in ('dataset','scope','method','protocol','symbol'))
                symbol_groups[key]+=np.array([int(row[k]) for k in ('gt_count','tp','fn')])
    symbols=[dict(zip(('dataset','scope','method','protocol','symbol'),key),gt_count=int(values[0]),
                  tp=int(values[1]),fn=int(values[2]),sensitivity=float(values[1]/values[0]))
             for key,values in sorted(symbol_groups.items())]
    cohorts={}
    for dataset in NAMES:
        selected={r['record']:r for r in records if r['dataset']==dataset}
        cohorts[dataset]=dict(records=len(selected),patients=len({r['patient'] for r in selected.values()}) if dataset=='incartdb' else None,
                             duration_hours=sum(r['duration_seconds'] for r in selected.values())/3600,
                             fresh_source_records=[k for k,v in selected.items() if not v['overlap_reasons']],
                             overlapping_source_records={k:v['overlap_reasons'] for k,v in selected.items() if v['overlap_reasons']})
    latency=json.loads((out/'latency.json').read_text())
    assert latency['source_sha256']==frozen, 'latency benchmark source mismatch'
    validation=dict(passed=not failures and evaluated_frozen and source_unchanged, completed_tasks=len(records),
                    failed_tasks=len(failures), failed_windows=sum(len(r['failures']) for r in records),
                    processed_windows=sum(len(r['timings']) for r in records),
                    identical_truth_denominators=True, counts_consistent=True,
                    evaluated_source_matches_freeze=evaluated_frozen, production_source_unchanged=source_unchanged,
                    initial_annotation_validation_aborts=4, repaired_tasks=4,
                    pwave_reused_predictions_rescored_identically=True,
                    pwave_duplicate_reference_entries_removed=78,
                    exact_predeclared_task_coverage=True, exact_predeclared_signal_duration=True,
                    incart_tasks_resumed_after_process_termination=executions['incartdb'].get('resumed_tasks',0))
    analysis=dict(cohorts=cohorts,metrics=metrics,excluding_prior_source_records=novel,cluster_bootstrap=uncertainty,
                  record_changes=sorted(changes,key=lambda r:r['delta_f1_pp']),qrs_sensitivity_by_reference_symbol=symbols,
                  execution=executions,latency=latency,validation=validation)
    (out/'analysis.json').write_text(json.dumps(analysis,indent=2))
    (out/'validation.json').write_text(json.dumps(validation,indent=2))
    primary=[m for m in metrics if (m['wave']=='QRS' and m['protocol']=='peak_75ms') or
             (m['lead']=='P_native' and m['protocol']=='peak_150ms')]
    lines=['# ECG feature extraction：新增数据集冻结评估（2026-09-22）','',
           '本轮只新增评估工具，没有调整生产算法、阈值或默认开关，也没有使用深度学习。'
           '生产代码与实际评估副本逐文件 SHA-256 一致；所有测试对象、窗口与容差在看结果之前固定。','',
           '主要结论：QRS 增强在 INCART 上只提高约 0.03 个 F1 百分点，在 SVDB 上下降约 0.09 个百分点，'
           '按来源分组的统计区间均跨过零，尚不能证明普遍收益。PWAVE 的 P_native F1 小幅提高，'
           '但 ±50 ms 峰位匹配 F1 仍只有 69.12%，门控后 P 波召回率仅 47.94%。'
           '下一轮应优先验证 P 峰定位、QRS 质量筛选后的峰位稳定性，以及 P 波接受门控的召回。','',
           '## 数据与覆盖','',
           '| 数据 | 记录 / 人群 | 原始总时长 | 运行范围 | 没有已知旧来源重叠的记录 |',
           '|---|---|---:|---|---:|']
    for dataset,c in cohorts.items():
        scope='全时长 QRS 检测器；每条首/中/尾各 30 s 完整提取' if dataset!='pwave' else '全时长完整提取，评分 P 和 QRS'
        population=f"{c['records']} 条 / {c['patients']} 位患者" if c['patients'] else f"{c['records']} 条，按来源记录统计"
        lines.append(f"| {NAMES[dataset]} | {population} | {c['duration_hours']:.3f} h | {scope} | {len(c['fresh_source_records'])} |")
    lines+=['',
        '- [INCART 1.0.0](https://physionet.org/content/incartdb/1.0.0/)：12 导联，257 Hz；参考点通常是 QRS 中部，位置未逐一人工校正。峰值 MAE 受这一标注定义影响。',
        '- [SVDB 1.0.0](https://physionet.org/content/svdb/1.0.0/)：78 条半小时记录，实际文件为 2 通道、128 Hz。与 BUT、QTDB 有来源重叠，按整条来源记录排除后另报。',
        '- [PWAVE 1.0.0](https://physionet.org/content/pwave/1.0.0/)：12 条 MIT-BIH 记录，360 Hz；两位专家制作/复核 P 波峰标注。官方说明标注不保证穷尽，未匹配预测不能一律认定为生理上不存在的 P 波。QRS 参考来自原 [MIT-BIH 1.0.0](https://physionet.org/content/mitdb/1.0.0/)。',
        '- PWAVE 只有记录 122 未出现在此前 BUT、QTDB 或 NSTDB 的来源记录中。新增的 P 波标注测试与新患者泛化是不同证据。',
        '- PWAVE 原始 22,108 个 P 标记包含 78 个同位置同符号重复项：119 中 77 个、214 中 1 个；去重后真值为 22,030。初次 4 项任务被严格标注检查拦截，修正评分器后补跑；其他 20 项复用原预测。对最终 24 项用修正后的评分器逐项重算，216 行评分及 336,147 个匹配与保存结果完全一致。没有改动提取算法。审计见 `pwave_duplicate_annotation_audit.json` 与 `pwave/composition_manifest.json`。',
        '- 三组数据均无本轮可用的人工 P/T 起止点真值，不能据此宣称 P 波宽度、T 终点或 QT 测量精度改善。',
        '', '## 配置和评分','',
        '`default`：当前代码，全部 refinement 开关关闭。`enhanced`：开启 `qrs_adaptive_consensus`、'
        '`p_pathology_candidates`、`atrial_event_validation`、`t_sequence_offset_only`。单独 QRS 检测器只应用对应的 QRS 开关。', '',
        'INCART 使用标准 12 导联输入；SVDB/PWAVE 使用 limited 模式与原始通道名称。'
        'WFDB 增益/基线换算后的 mV 直接输入；只规范 AVR/AVL/AVF 名称大小写。'
        '内部统一 500 Hz，INCART 50 Hz 陷波，其余 60 Hz。每 30 s 核心区前后各带 2 s 上下文；'
        '固定抽样窗口分别为首 30 s、中间 30 s、末 30 s，每条只覆盖 90 s，不能写成全时长完整提取。', '',
        '检测点换回原始采样轴，以核心区归属去重；全记录进行一对一匹配，优先最大匹配数、再最小时间误差。'
        '不利用标注选导联、选窗口、移动时间轴或调参。QRS 主容差 75 ms，补充 50/150 ms；'
        'P 主容差 150 ms，补充 50 ms。容差按原始采样率取整。异常窗口不产生预测，真值仍保留为漏检。', '',
        'Se=TP/(TP+FN)，PPV=TP/(TP+FP)，F1=2TP/(2TP+FP+FN)。MAE 只针对已匹配事件，必须结合漏检率看。', '',
        '## 主结果','']
    lines+=metric_table(primary)
    lines+=['','![结果对比](../ecgfeat_cross_dataset_20260922/validation_results.png)',
            '', 'QRS 检测器全时长结果不含后续起搏处理、补检与描记；完整提取结果取最终输出的心搏，二者分开报告。',
            '', '## 更严格定位与宽容差复核','']
    secondary=[m for m in metrics if (m['lead']=='P_native' and m['protocol']=='peak_50ms') or
                (m['scope']=='detector_full' and m['wave']=='QRS' and m['protocol'] in {'peak_50ms','peak_150ms'})]
    lines+=metric_table(secondary)
    lines+=['','## 排除既往来源记录','']
    novel_primary=[m for m in novel if m['dataset']!='incartdb' and
                   ((m['wave']=='QRS' and m['protocol']=='peak_75ms') or(m['lead']=='P_native' and m['protocol']=='peak_150ms'))]
    lines+=metric_table(novel_primary)
    lines+=['','完整排除名单保存在 `analysis.json → cohorts`。这是已知来源去重，无法排除数据发布方未提供的跨库身份关系。',
            '', '## 参考心搏类型的检出率','',
            '按 ±75 ms 与参考点匹配统计 QRS 检出：N=正常心搏，A/S=参考标记的房性/室上性早搏，V=室早。'
            '这里没有评估检测器能否正确分类心搏。其余类型保存在 JSON 和 CSV。','',
            '| 数据 / 参考类型 | 心搏数 | 默认 Se % | 增强 Se % |','|---|---:|---:|---:|']
    for dataset in NAMES:
        scope='pipeline_full' if dataset=='pwave' else 'detector_full'
        for symbol in ('N','A','S','V'):
            pair={s['method']:s for s in symbols if s['dataset']==dataset and s['scope']==scope
                  and s['protocol']=='peak_75ms' and s['symbol']==symbol}
            if set(pair)=={'default','enhanced'}:
                lines.append(f"| {NAMES[dataset]} / {symbol} | {pair['default']['gt_count']} | "
                             f"{percent(pair['default']['sensitivity'])} | {percent(pair['enhanced']['sensitivity'])} |")
    lines+=['','室早的增强匹配召回率在 ±75/150 ms 下分别为：INCART 89.57%/96.85%、'
            'SVDB 87.07%/92.87%、PWAVE 90.28%/92.56%。因此这些错误同时包含峰位偏差和未匹配心搏；'
            '尤其 INCART 的参考是 QRS 中部，不能将全部匹配 FN 等同于没有发现 QRS。',
            '', '## P 波候选、接受结果与房性事件','',
            '`P_native` 是首个实测通道的描记 P 波峰；`P_accepted` 还要求当前 P 波质量评估接受；'
            '`P_atrial` 是房性事件候选。三者目标不同，不能用候选召回率代替最终接受率。','']
    lines+=metric_table([m for m in metrics if m['dataset']=='pwave' and m['lead'] in {'P_accepted','P_atrial'} and m['protocol']=='peak_150ms'])
    lines+=['','PWAVE 103 的全时长增强 P_native F1 在 ±150 ms 时为 98.94%，±50 ms 时为 53.86%。'
            '原始 MLII 波形的事后核查显示，开始几拍的预测峰落在参考 P 波峰后约 75–108 ms 的位置，'
            '这不是统一移动参考轴得到的结果。本轮保持算法不变，该例用于定位下一轮改进问题。','',
            '![PWAVE 103 峰位复核](../ecgfeat_cross_dataset_20260922/pwave_103_peak_audit.png)']
    lines+=['','## 改善是否稳定','',
            '对相同来源成对比较 F1，固定种子 20260922、5000 次簇 bootstrap。INCART 按 32 位患者重采样；'
            'SVDB/PWAVE 按来源记录重采样。区间只描述本数据集内的抽样不确定性，不能消除旧数据来源重叠。','',
            '| 数据 / 范围 / 输出 | ΔF1 百分点 | 95% 区间 | 簇数 |','|---|---:|---:|---:|']
    for u in uncertainty:
        if u['protocol']==('peak_75ms' if u['wave']=='QRS' else 'peak_150ms') and u['lead'] in {'QRS','P_native'}:
            lines.append(f"| {NAMES[u['dataset']]} / {SCOPES[u['scope']]} / {u['lead']} | {u['delta_f1_pp']:+.3f} | "
                         f"[{u['ci95_pp'][0]:+.3f}, {u['ci95_pp'][1]:+.3f}] | {u['clusters']} |")
    lines+=['','## 最明显的退化记录','',
            '| 数据 / 范围 / 记录 | 输出 | 默认 F1 % | 增强 F1 % | Δ 百分点 |','|---|---|---:|---:|---:|']
    for c in sorted(changes,key=lambda r:r['delta_f1_pp'])[:10]:
        lines.append(f"| {NAMES[c['dataset']]} / {SCOPES[c['scope']]} / {c['record']} | {c['lead']} | "
                     f"{percent(c['default_f1'])} | {percent(c['enhanced_f1'])} | {c['delta_f1_pp']:+.3f} |")
    lines+=['','SVDB 862 的退化主要表现为定位偏移：增强配置在 ±75 ms 下为 TP/FP/FN=1928/245/260，'
            '放宽至 ±150 ms 后为 2173/0/15；默认分别为 2164/12/24 和 2176/0/12。'
            '其中 245 个增强点在宽容差下仍能匹配。对 60–90 s 核心窗口的复现与原保存预测逐点一致：'
            'ECG1/ECG2 的 bSQI 分别为 1.0/0.738，增强路径只使用 ECG1，默认使用两通道。'
            '图示和检测窗审计指向质量筛选后能量峰/局部搜索的落点稳定性问题，而总体匹配分数本身不能直接说明真实漏拍数量。'
            '审计保存于 `svdb/862_timing_audit.json`；本轮未据此修改参数。','',
            '![SVDB 862 定位退化复核](../ecgfeat_cross_dataset_20260922/svdb_862_timing_audit.png)',
            '', '逐记录全部结果与参考心搏类型分层均保留。类型分层统计的是已知类型心搏的检出率，不是心律分类准确率。',
            '', '## 运行速度','',
            '完整队列的并行作业耗时保存在各目录 `execution.json`；进程竞争下的时延不能作为纯算法加速比。'
            '下面另用单进程、数值库单线程：每库固定首/末记录前 32 s，预热一次、计时三次、交替配置顺序。'
            '排除文件读取与评分，包含预处理、质量评估、检测、描记及其余完整提取步骤。重复运行的 QRS/P 输出哈希一致。','',
            '| 数据 / 记录 | 配置 | 32 s 信号耗时中位数 | 耗时 / 信号时长 |','|---|---|---:|---:|']
    for t in latency['results']:
        lines.append(f"| {NAMES[t['dataset']]} / {t['record']} | {t['variant']} | {t['median_seconds']:.3f} s | {t['real_time_factor']:.4f} |")
    lines+=['','另在每库首记录对增强配置做 cProfile，以下列出累计耗时较高的内部函数。'
            '插桩会增加耗时；累计时间存在嵌套，不能相加，也不能替代上面的独立计时。','',
            '| 数据 | 函数 | 调用数 | 累计耗时 s |','|---|---|---:|---:|']
    for profile in latency.get('profiles',[]):
        functions=[f for f in profile['functions'] if f['function'] not in {'extract','<listcomp>','<dictcomp>','<genexpr>'}]
        for f in functions[:5]:
            lines.append(f"| {NAMES[profile['dataset']]} | `{f['file']}:{f['function']}` | {f['calls']} | {f['cumulative_seconds']:.3f} |")
    lines+=['','这是固定窗口的离线吞吐测试，带双向滤波和上下文，不能直接解释为在线设备的实时延迟承诺。',
            '', '## 复现与完整性','',
            f"共 {validation['completed_tasks']} 个记录/范围/配置任务、{validation['processed_windows']} 次窗口处理，"
            f"失败任务 {validation['failed_tasks']}，失败窗口 {validation['failed_windows']}。"
            f"生产文件未变化：{source_unchanged}；运行副本匹配冻结哈希：{evaluated_frozen}。",'',
            'INCART 初始进程在 298 项完成后收到 SIGTERM；最后 2 项使用归档的原始评估脚本断点续跑。'
            '续跑前核对了代码、原始数据、标注来源及现有检查点哈希，保持原 fingerprint，未跳过未完成记录。', '',
            '```bash', '.venv/bin/python scripts/download_ecgfeat_cross_dataset.py --workers 8',
            '# source 可指向当前仓库，严格复现本轮请使用冻结副本；使用新的 out，避免覆盖已有结果。',
            '.venv/bin/python scripts/evaluate_ecgfeat_cross_dataset.py \\\n  --datasets incartdb svdb pwave \\\n  --source ecgfeat_cross_dataset_20260922/frozen_source \\\n  --out ecgfeat_cross_dataset_reproduction --workers 12',
            '.venv/bin/python -m pytest tests/test_ecgfeat_cross_dataset_protocol.py -q', '```','',
            '本轮目录：`ecgfeat_cross_dataset_20260922/`。`protocol_predeclared.json` 保存预先固定的范围；'
            '`production_source_sha256.json` 保存算法冻结哈希；每库 `manifest.json` 记录代码、数据及协议哈希；'
            '`checkpoints/` 保存逐记录预测、匹配、排除标注、窗口计时；CSV 可供复核。'
            '`analysis.json` 汇总结果与统计区间，`validation.json` 保存完整性检查。','']
    (ROOT/'docs/ecgfeat_cross_dataset_20260922.md').write_text('\n'.join(lines))
    print(json.dumps(validation,indent=2))
    if not validation['passed']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
