"""Publish completed quality evidence without modifying frozen science/CV."""
import argparse
import json
from pathlib import Path
import statistics
from datetime import datetime
from vlm_probe.evaluate import sha_file,write_json
from rarevlm_paths import PROJECT_ROOT


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);a=p.parse_args()
    cfg=json.loads(a.config.read_text());root=Path(cfg['output_root'])
    result=json.loads((root/'analysis/ANALYSIS.json').read_text())
    review=json.loads((root/'analysis/INDEPENDENT_REVIEW.json').read_text())
    assert result['status']=='completed' and review['status']=='passed'
    diagnostic=json.loads((root/'selection/DIAGNOSTICS.json').read_text())
    runs=[json.loads(p.read_text()) for p in sorted((root/'runs').glob('*/seed*/TRAIN_RESULT.json'))]
    assert len(runs)==12
    launch=json.loads((root/'LAUNCH.json').read_text())
    whole_wall=(datetime.fromisoformat(result['completed_at'])-datetime.fromisoformat(launch['started_at'])).total_seconds()
    runtime={c:{'active_seconds_per_seed':[x['active_seconds'] for x in runs if x['condition']==c],
        'peak_allocated_gib_max':max(x['peak_allocated_gib'] for x in runs if x['condition']==c)} for c in cfg['conditions']}
    for phase in ['training','dev','test']:
        status=root/(phase+'_queue')/'STATUS.json'
        if not status.exists():continue
        s=json.loads(status.read_text())
        assert s['status']=='completed' and s['resource_query_failures']==0
        assert all(x['state']=='completed' and x['contention_observations']==0 for x in s['jobs'].values())
    portable={'schema_version':'rarevlm.retrieval_quality.metrics.v1',**result,'selection_diagnostics':diagnostic,
        'runtime':runtime,'independent_review':review,'protocol':'docs/CLIP_RETRIEVAL_QUALITY_V1_PROTOCOL.md',
        'wall_seconds_since_original_launch':whole_wall,
        'reporting_recovery':json.loads((root/'analysis/REPORTING_RECOVERY.json').read_text()),
        'scope':'train-only caption curation;original and controlled mismatch branches separate;no formal CV update'}
    write_json(PROJECT_ROOT/'results/clip_retrieval_quality_v1_metrics.json',portable)
    lines=['# 图文数据一致性筛选：完成结果','',
        '对应多模态数据管线、图文表示训练与科学评测；独立12次完整训练，原始数据提准与受控错配鲁棒性分别验收。',
        '', '[冻结协议](CLIP_RETRIEVAL_QUALITY_V1_PROTOCOL.md) · [便携指标](../results/clip_retrieval_quality_v1_metrics.json)',
        '', '## 公平比较与工程规模','',
        '完整28000图；每图固定保留4条，112000描述／1120000正描述曝光、10epoch／8750更新／132480参数；Final@10、meanCE+KL、三seed{0,42,200}。随机／分数两组在各分支的初始化、负样本序列、曝光和更新hash严格一致。原始与错配分支采样排除可不同，跨分支不作纯噪声单因素归因。',
        '', '| 划分 | 条件 | 文搜图R1均值±SD | 图搜文R1均值±SD |','| --- | --- | ---: | ---: |']
    for split in ['dev','test']:
        data=result[split]
        if data is None:continue
        for c,s in data['summaries'].items():
            values=[f'{s[m]["mean"]:.3f} ± {s[m]["sample_sd"]:.3f}' for m in ['T2I_R@1','I2T_R@1']]
            lines.append(f'| {split} | {c} | {values[0]} | {values[1]} |')
        values=[f'{data["reference_clean_full5"][m]["mean"]:.3f} ± {data["reference_clean_full5"][m]["sample_sd"]:.3f}' for m in ['T2I_R@1','I2T_R@1']]
        lines.append(f'| {split} | 历史clean-full5，监督量更多 | {values[0]} | {values[1]} |')
    lines += ['', '## 主比较与结论边界','']
    map_lines=[]
    for branch in ['clean','noisy']:
        d=result['dev']['comparisons'][branch];ci=d['intervals']['T2I_R@1'];i=d['intervals']['I2T_R@1']
        lines += [f'- **{branch} dev score4−random4**：T2I {ci["delta_mean_pp"]:+.3f}pp，单侧98.75%下界{ci["one_sided_lower_pp"]:.3f}pp；I2T {i["delta_mean_pp"]:+.3f}pp、下界{i["one_sided_lower_pp"]:.3f}pp；预设全门控{d["gate"]["passed"]}，T2I正向seed{d["positive_seed_directions"]}/3。']
        if result['test'] is not None and branch in result['test']['comparisons']:
            t=result['test']['comparisons'][branch];ci=t['intervals']['T2I_R@1'];i=t['intervals']['I2T_R@1']
            lines += [f'- **{branch} 固定test**：T2I {ci["delta_mean_pp"]:+.3f}pp、单侧98.75%下界{ci["one_sided_lower_pp"]:.3f}pp；I2T {i["delta_mean_pp"]:+.3f}pp、下界{i["one_sided_lower_pp"]:.3f}pp。score4相对历史clean-full5 T2I点差{t["score_minus_clean_full5_point_pp"]["T2I_R@1"]:+.3f}pp，I2T{t["score_minus_clean_full5_point_pp"]["I2T_R@1"]:+.3f}pp。']
            map_lines.append(f'{branch}固定test T2I/I2T相对matched random4 {ci["delta_mean_pp"]:+.3f}/{i["delta_mean_pp"]:+.3f}pp')
        else:
            lines += [f'- **{branch} 未通过dev全门控，没有本轮新test**。不回调保留数、噪声／分数／门槛或checkpoint。']
            map_lines.append(f'{branch} dev T2I{ci["delta_mean_pp"]:+.3f}pp，全门控失败未新增test')
    nr=diagnostic['conditions']['noisy_random4'];ns=diagnostic['conditions']['noisy_score4']
    lines += ['', '区间按图像簇、固定三checkpoint均值计算，T2I每图五caption先聚合；四主端点alpha=.05/4。seed样本SD单列。历史dev/test已曝光、预训练重叠未知，不是盲测或全部历史多重比较校正。',
        '', '## 筛选诊断与代码原理','',
        f'训练集每图置换1/5描述，总28000条来源配对改变。random4保留{nr["retained_reassigned_pairs"]}条（{100*nr["retained_reassigned_pair_fraction"]:.3f}%）；score4保留{ns["retained_reassigned_pairs"]}条（{100*ns["retained_reassigned_pair_fraction"]:.3f}%），排除{100*ns["removed_reassigned_pairs"]/28000:.3f}%已知置换。语义相似置换可能仍合理；这些是受控来源诊断，不能当原始真实错标率。',
        '', '`selection.py:select_slots`只读实际train特征／图SHA／槽序；s=image·text，取最高4条。相同教师用于筛选与KL，分数同源偏差存在。原始数据上的低分描述可能包含真实细节，不能把低分自动叫错标。`loss.py:selected_terms`保持meanCE＋KL，I2T[32,384]、T2I[128,96]；未选描述不进入教师或监督。诊断flags／donor只在选择后检查，不用于修复训练标签。',
        '', '## 工程与学习','',
        f'12次训练active秒：总{sum(x["active_seconds"] for x in runs):.2f}，单次范围{min(x["active_seconds"] for x in runs):.2f}—{max(x["active_seconds"] for x in runs):.2f}；筛选CPU{diagnostic["train_only_selection_seconds"]:.2f}秒；首次启动至完成壁钟{whole_wall:.2f}秒（含训练、两卡队列、评测和JSON修复；不含启动前代码准备）。恢复controller壁钟{result["controller_seconds"]:.2f}秒单列。allocated峰值最高{max(x["peak_allocated_gib"] for x in runs):.3f}GiB。不是完整CLIP预训练速度。',
        f'5项数学／边界合同、四真实训练batch梯度与manifest对齐通过；{len(review["checked_cells"])}份完整图库预测由独立Python标量Recall/nDCG复核，{review["protected_files_unchanged"]}个历史科学文件／正式简历哈希保持。0资源查询失败、0观察到的竞争，不抢占现有任务。',
        '12份dev完成后原统计JSON保存遇到numpy.bool_错误；独立`tools/resume_quality_report.py`只在报告边界转换NumPy scalar，原冻结源码／计算公式／门槛／模型和预测不改、原失败日志保留，0重训练／0重复dev预测。复现或恢复本协议时用该wrapper调用冻结pipeline；helper与原来源锁另行归档。',
        '', f'真实图文／筛选／改善退化学习页：`{root}/learning/学习页.html`。图片为来源原图缩略图，实际CLIP输入仍源协议224预处理；不是attention。个人掌握与语义案例审查范围另记，不自动当本人独立贡献。',
        '', '## JD与项目使用','',
        '字节[Seed官方方向](https://seed.bytedance.com/en/seedearlycareer)明确多模态数据管线与科学评测；阿里[国际站多模态实习原文](https://career.nankai.edu.cn/correcruit/content/id/118629.html)强调图文表示／实验分析。腾讯既有社招数据质量要求仅作技术参考。本实验支持图文数据筛选的对照与诊断能力，不代表工业中文商品、生成式数据合成、CPT/RL或在线服务已完成。',
        '参考[DataComp](https://github.com/mlfoundations/datacomp)固定架构／预算及CLIP score过滤思想，revision与参考文件SHA已归档；没有执行其预训练框架或复现其benchmark。正式CV保持；针对数据质量岗位可先准备该研究原理／负结果／压力测试与强基线边界。']
    (PROJECT_ROOT/'docs/CLIP_RETRIEVAL_QUALITY_V1_RESULTS.md').write_text('\n'.join(lines)+'\n')
    path=PROJECT_ROOT/'docs/JD_EVIDENCE_MAP.md';text=path.read_text()
    start=text.index('## 图文数据质量研究（当前运行）');end=text.index('## 岗位定向选择',start)
    text=text[:start]+'''## 图文数据质量研究（已完成）

独立[一致性筛选结果](CLIP_RETRIEVAL_QUALITY_V1_RESULTS.md)：完整28k图、4条件×3seed共12新训练，原始／20%受控来源置换、score4／random4分别配对。'''+ '；'.join(map_lines)+'''。原始收益与受控错配鲁棒性分开，不把后者包装为干净数据提准。

已知置换保留率random20.04%→score0.393%，这是选择后的来源诊断，不是原始数据错标率。各分支初始化／采样序列／监督量严格一致，强clean-full5历史结果监督量更多并单列。对应多模态数据管线、筛选与科学评测；尚无工业商品或生成式数据合成增益。个人贡献／掌握、岗位资格限制沿用上文，正式CV保持。

'''+text[end:]
    path.write_text(text)
    entry='\n\n本轮图文数据质量研究已完成：'+ '；'.join(map_lines)+'。12次训练，失败分支不新增test；详见[筛选结果](docs/CLIP_RETRIEVAL_QUALITY_V1_RESULTS.md)。原始数据收益与错配压力测试分别报告，正式简历保持。\n'
    path=PROJECT_ROOT/'README.md';text=path.read_text()
    if '本轮图文数据质量研究已完成：' not in text:path.write_text(text+entry)
    path=PROJECT_ROOT/'docs/NEXT_ACTIONS_20261014.md';text=path.read_text()
    if 'clip_retrieval_quality_v1完成更新' not in text:path.write_text(text+'\n\n## clip_retrieval_quality_v1完成更新\n\n'+'；'.join(map_lines)+
        '。12次完整训练、匹配4对4监督与采样；known mismatch诊断不当原始语义错标。失败策略不自动扩增。下一步先用真实图片、select_slots、selected_terms和独立评分解释成功／失败与历史full5差距；不因该实验堆更多项目，不将掌握未确认内容写入正式CV。\n')
    print('QUALITY_REPORT_COMPLETED',len(runs),'trainings',len(review['checked_cells']),'evaluation cells')


if __name__=='__main__':main()
