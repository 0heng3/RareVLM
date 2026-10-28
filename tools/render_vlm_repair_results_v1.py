"""Publish completed repair evidence and fixed-point gradients, without model IO."""

import html
import json
from pathlib import Path


PACKAGE = Path(__file__).resolve().parents[1]
OUTPUTS = PACKAGE.parents[1] / 'outputs'
ROOT = OUTPUTS / 'vlm_gradient_mechanism_v1'
REPORT = PACKAGE / 'docs/VLM_SEMANTIC_AND_TRAINING_REPAIRS_V1.md'
PAGE = PACKAGE / 'docs/learning/数据准入修复_参数梯度诊断.html'


def read(path):
    return json.loads(path.read_text())


def main():
    completion = read(ROOT / 'COMPLETION.json')
    if completion['status'] != 'completed':
        raise ValueError('Only publish complete gradient measurements')
    result, cohort = read(ROOT / 'ANALYSIS.json'), read(ROOT / 'COHORT.json')
    lines = [
        '# 数据准入与训练元数据修复：实际梯度诊断', '',
        '2026-10-19。执行用户授权的下一步修复，历史实验、标签、模型权重与门槛保持。数据和metadata改动均为独立新入口；实际GPU工作仅做固定参数梯度测量，0 optimizer更新、0新生成、未使用确认结果。', '',
        '## 已完成的修复', '',
        '- 样本级否决优先于整体通过率：任一QA审核unknown/不通过，整对不进入新训练池；新池593对/1186QA，排除7对/14QA。T02/T08明确阻断。其余1155QA仍标记仅官方高共识、未经逐图验证，31QA有审核支持；不称全池干净或把不确定等同错标。',
        '- 语义规则作用于整个训练池，保留排除理由、原问题/GT和图像；既有困难原图审核只用于训练准入，不改历史评分。准备工具使用独立训练导出。早期结构查看曾展示旧合并JSON中的少量confirm元数据，未用于选择、评分或模型计算。',
        '- 新加载入口同步LoRA dropout运行值和PEFT保存配置。真实PEFT小模型CPU保存、通用重载均保持0，权重精确不变；大模型梯度入口已实际使用新loader。旧adapter配置文件不改。',
        '- 新增CPU检查分别为数据准入10项、dropout10项、梯度数学17项、首答案拆分2项，均通过；没有重复旧加载/mask/save整套审查。', '',
        '## 实际测量的范围', '',
        '只测initial与普通CE Final200，同一4QA可靠难例批、8QA普通人工批、8QA Text回放；共40个checkpoint×QA前向、16次标量梯度反传，另有gradient checkpointing内部重算。探针为蜡烛条纹no、单猫镜像no、巴士同色yes、蓝色砧板yes。四图均由助手查看原图支持原GT，非专家新标注，也不代表整个训练池。普通人工来源为4yes/4no，未经全量逐图核验。', '',
        '```text',
        'G_CE = .5 g_Text逐题CE + .5 g_普通人工逐题CE',
        'G_answer3 = G_CE + (1/3) g_普通人工首答案',
        '局部未预条件SGD下降方向的一阶探针损失变化 = -eta * dot(g_probe, G)',
        '```', '',
        '探针不加入合成更新。answer3只模拟人工首答案系数三倍、格式系数保持，未进行这种训练；系数不再次归一，梯度范数也不强制相等。正dot意味着该局部方向一阶降低探针损失；负dot意味着一阶增大。不是历史Adam步、有限步保证、全训练因果或能力分数。', '',
        '| checkpoint | 探针 | 与Text余弦 | 与普通人工余弦 | CE合成dot | 首答三倍合成dot |',
        '|---|---|---:|---:|---:|---:|',
    ]
    matrix = ''
    summaries = []
    for role, label in [('initial', '初始'), ('ce', 'CE Final200')]:
        value = result['results'][role]
        standard, weighted = value['standard_ce_direction'], value['answer3_direction']
        for probe, title in [('probe_no_answer', '可靠no难例'), ('probe_yes_answer', '可靠yes难例')]:
            text_cos = standard['pairwise'][probe]['text_ce']['cosine']
            human_cos = standard['pairwise'][probe]['human_ce']['cosine']
            sd = standard['groups'][probe]['dot_with_objective']
            wd = weighted['groups'][probe]['dot_with_objective']
            lines.append(f'| {label} | {title} | {text_cos:+.6f} | {human_cos:+.6f} | {sd:+.6e} | {wd:+.6e} |')
            summaries.append({'role': role, 'probe': probe, 'standard_dot': sd, 'answer3_dot': wd,
                              'standard_first_order_direction': 'loss_down' if sd > 0 else 'loss_up' if sd < 0 else 'stationary',
                              'answer3_first_order_direction': 'loss_down' if wd > 0 else 'loss_up' if wd < 0 else 'stationary'})
            cells = ''
            for scalar in (text_cos, human_cos):
                color = '#e5f2ed' if scalar >= 0 else '#fae9e6'
                cells += f'<td style="background:{color}">{scalar:+.5f}</td>'
            for dot in (sd, wd):
                color = '#e5f2ed' if dot > 0 else '#fae9e6'
                effect = '局部下降' if dot > 0 else '局部上升' if dot < 0 else '一阶静止'
                cells += f'<td style="background:{color}">{dot:+.5e}<br>{effect}</td>'
            matrix += f'<tr><td>{label}</td><td>{title}</td>{cells}</tr>'
        lines.extend(['', f"{label}：LoRA前后哈希一致={value['lora_unchanged']}，实际训练配置dropout={value['dropout_config']}，"
                      f"可训练参数={value['trainable_parameters']}，截至该点worker累计耗时={value['worker_elapsed_seconds']:.2f}秒，"
                      f"峰值allocated={value['peak_allocated_gib']:.3f}GiB。", ''])
    lines.extend(['## 来源分解与方案取舍', '',
                  '| checkpoint | no·(.5人工CE) | no·(.5TextCE) | 人工CE与首答案梯度余弦 |',
                  '|---|---:|---:|---:|'])
    for role, label in [('initial', '初始'), ('ce', 'CE Final200')]:
        standard = result['results'][role]['standard_ce_direction']
        source = standard['groups']['probe_no_answer']['weighted_source_dots']
        alignment = standard['pairwise']['human_ce']['human_answer']['cosine']
        lines.append(f"| {label} | {source['human_ce']:+.6f} | {source['text_ce']:+.6f} | {alignment:.6f} |")
    lines.extend(['',
        '两处固定点的no·普通CE方向均为负，首答案三倍仍未转正：初始从−.8358变−2.1048；Final从−.2379变−.0499。Final正例方向反而从−.1058变−1.1848。因此本轮不支持直接启动统一首答案三倍SFT。',
        '普通人工CE与首答案梯度余弦约.99987/.99990，说明在此微批上加权首答主要放大已有方向，不支持格式梯度主导冲突的叙述。初始人工与Text都对no方向不利；Final人工已经有利但Text抵消，不能单归因于某一个来源。',
        '普通人工题已经4yes/4no，数量均衡不等于困难监督方向充分。后续应针对可信困难正负例设计一个明确的监督/更新因素，并与匹配强CE比较；不能换名重跑此前净0的hard oversampling，也不能仅把human/Text做梯度投影就承诺修复。负相关不等于不可共同优化，不能据局部结果认定LoRA容量不足。', '',
    ])
    lines.extend([
        '## 解释边界与后续决策', '',
        '这些点积能判断已登记少量训练题的局部方向，不能单凭小样本扩大到全部12个no原错或整段Adam训练。来源间负余弦也不自动等于合成更新有害，必须看合成dot。未将首答案损失、曝光份额或logit导数冒充参数梯度。', '',
        '只有依据这些结果提出明确的单变化假设，才另行登记短预算训练；本轮没有自动启动首答案加权SFT、改变采样或解冻视觉，也没有新增模型能力提升。', '',
        '## 代码和证据', '',
        '- 数据：[admission.py](../src/vlm_semantic_admission_v1/admission.py)、[训练准入结果](/opt/data/private/longtail_vlm/rarevlm_pilot/outputs/vlm_semantic_admission_v1/PREPARATION_RESULT.json)。',
        '- 新训练加载入口：[model.py](../src/vlm_runtime_metadata_v1/model.py)。未来新训练使用此入口；历史协议复现保留旧loader。',
        '- 梯度：[components.py](../src/vlm_gradient_mechanism_v1/components.py)、[metrics.py](../src/vlm_gradient_mechanism_v1/metrics.py)、[runner](../tools/run_vlm_gradient_mechanism_v1.py)。',
        '- [事前协议](VLM_GRADIENT_MECHANISM_V1_PROTOCOL.md)、[完整参数梯度分析](/opt/data/private/longtail_vlm/rarevlm_pilot/outputs/vlm_gradient_mechanism_v1/ANALYSIS.json)、[图像与梯度方向讲解](learning/数据准入修复_参数梯度诊断.html)。',
    ])
    REPORT.write_text('\n'.join(lines) + '\n')

    cards, paths = '', set()
    probe_batch = {role: next(b for b in result['results'][role]['actual_batches'] if b['source'] == 'probe')
                   for role in ('initial', 'ce')}
    for index, qid in enumerate(cohort['groups']['probe']):
        row = cohort['qa'][str(qid)]
        path = cohort['images'][row['image_id']]['path']
        paths.add(path)
        before, after = [probe_batch[role]['first_answer_ce'][index] for role in ('initial', 'ce')]
        support = cohort['probe_review'][index]['support']
        cards += (f'<article><img src="{html.escape(path, quote=True)}" alt="原始训练图"><h3>{html.escape(row["question"])}</h3>'
                  f'<p>原GT={row["target"]}；首答案NLL：{before:.4f} → {after:.4f}。</p>'
                  f'<p>{html.escape(support)}</p><small>{html.escape(str(qid))}</small></article>')
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>准入修复与真实参数梯度</title><style>body{{font:16px/1.75 system-ui;background:#f2f5f7;color:#183141;margin:0}}main{{max-width:1100px;margin:auto;padding:24px}}section,article{{background:white;padding:20px;border-radius:12px;margin:20px 0}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}img{{width:100%;height:280px;object-fit:contain;background:#eef2f5}}table{{border-collapse:collapse;display:block;overflow:auto}}td,th{{padding:10px;border-bottom:1px solid #ddd;white-space:nowrap;text-align:left}}pre{{padding:14px;background:#eaf0f4;overflow:auto}}a{{color:#07609b}}@media(max-width:650px){{.grid{{grid-template-columns:1fr}}}}</style><main>
<h1>修复已落地；再检查实际参数更新方向</h1><p>新池593对、1186题；已知不确定整对排除，历史标签和分数保留。新loader同步dropout运行和保存配置。本页所有梯度来自真实模型，不是模拟向量；没有optimizer更新或新能力分数。</p>
<section><h2>读表前先看符号</h2><pre>g_probe = 探针首答案平均NLL对1843200个LoRA参数的梯度
G = 声明的人工 / Text损失合成梯度
假想SGD一步：theta_new = theta - eta * G
探针损失的一阶变化约为 -eta * dot(g_probe, G)</pre><p>dot &gt; 0：局部损失下降；dot &lt; 0：局部损失上升。负余弦只说明两个来源方向相反，不能跳过权重就说整次更新有害。以下是两处固定点、小训练样本的局部方向，不是历史Adam因果，也不是泛化得分。</p></section>
<section><h2>真实来源梯度</h2><table><thead><tr><th>checkpoint</th><th>探针</th><th>与Text余弦</th><th>与普通人工余弦</th><th>普通CE合成dot</th><th>首答三倍合成dot</th></tr></thead><tbody>{matrix}</tbody></table><p>合成方向使用8Text与8普通人工题的.5/.5权重，探针未混入更新。首答三倍只是已登记的梯度方向比较，不是已经训练的模型。颜色有符号和文字说明，矩阵不是attention。</p></section>
<section><h2>四个真实训练探针</h2><p>原图仅按网页宽度显示，不是模型processor输入重建。它们由助手核看支持GT，不是专家全量审核；镜像题区分一只真实猫与反射影像。</p><div class="grid">{cards}</div></section>
<section><h2>代码与限制</h2><p><a href="../VLM_SEMANTIC_AND_TRAINING_REPAIRS_V1.md">完整修复报告</a> · <a href="../VLM_GRADIENT_MECHANISM_V1_PROTOCOL.md">测量前冻结的协议</a> · <a href="{ROOT / 'ANALYSIS.json'}">完整参数梯度数值</a></p><p>尚未验证任何新训练配方能提升模型能力。下一训练应根据结果单独登记，而不是看见某种梯度就直接加时长、加权或解冻全部视觉模块。</p></section></main></html>'''
    missing = [path for path in paths if not Path(path).is_file()]
    if missing:
        raise ValueError(f'Missing original probe images: {missing}')
    PAGE.write_text(page)
    progress = read(ROOT / 'PROGRESS.json')
    progress.update(status='completed', total_checkpoints=2, total_microbatches=10,
                    worker_elapsed_seconds=read(ROOT / 'WORKER_COMPLETE.json')['worker_elapsed_seconds'])
    (ROOT / 'PROGRESS.json').write_text(json.dumps(progress, ensure_ascii=False, indent=2) + '\n')
    published = {'status': 'completed', 'report': str(REPORT), 'learning_page': str(PAGE),
                 'original_probe_images': len(paths), 'missing_images': missing,
                 'source_gradient_direction_summary': summaries, 'new_optimizer_updates': 0}
    (ROOT / 'DELIVERY.json').write_text(json.dumps(published, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(published, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
