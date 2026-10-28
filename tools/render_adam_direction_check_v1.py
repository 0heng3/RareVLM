"""Explain existing Adam direction results with original images and signed bars."""

import html
import json
from pathlib import Path


PACKAGE = Path(__file__).resolve().parents[1]
OUTPUTS = PACKAGE.parents[1] / 'outputs'
ROOT = OUTPUTS / 'vlm_adam_direction_v1'
REPORT = PACKAGE / 'docs/VLM_ADAM_DIRECTION_CHECK_V1.md'
PAGE = PACKAGE / 'docs/learning/Adam真实更新方向检查.html'


def read(path):
    return json.loads(path.read_text())


def main():
    a = read(ROOT / 'ANALYSIS.json')
    c = read(OUTPUTS / 'vlm_gradient_mechanism_v1/COHORT.json')
    measurements = a['measurements']
    choices = [
        ('initial_raw_sgd_at_recorded_lr', '初始点·SGD假想方向'),
        ('actual_first_parameter_displacement', '初始点·历史真实首步位移'),
        ('initial_cpu_shadow_adam_standard', '初始点·CPU影子Adam普通CE'),
        ('initial_cpu_shadow_adam_answer3', '初始点·CPU影子Adam首答三倍'),
        ('ce_raw_sgd_at_recorded_lr', 'Final点·SGD假想方向'),
        ('ce_cpu_shadow_adam_standard', 'Final点·CPU影子Adam普通CE'),
        ('ce_cpu_shadow_adam_answer3', 'Final点·CPU影子Adam首答三倍'),
    ]
    table = ['| 位置与方向 | no首答损失一阶变化 | yes首答损失一阶变化 |', '|---|---:|---:|']
    for key, label in choices:
        v = measurements[key]['probes']
        table.append(f"| {label} | {v['probe_no_answer']['first_order_loss_change']:+.6e} | {v['probe_yes_answer']['first_order_loss_change']:+.6e} |")
    report = [
        '# Adam实际更新方向错误点检查 v1', '', '2026-10-19。只读原checkpoint、保存梯度和优化器状态；0 VLM加载、0 GPU前向、0真实模型参数更新。', '',
        '## 核心发现', '',
        '需要修正的是归因方式：原始梯度在SGD下的符号，不能直接代表Adam实际位移的符号。Final点中，普通CE的SGD方向对no探针的一阶变化为正，但带保存动量的CPU影子Adam为负。不能据此前两个SGD固定点宣称Adam一直损害可靠负例。', '',
        '首步历史真实FP32位移的一阶估计对no不利、对yes有利。首答案三倍在初始点加重no方向的不利，在Final点减弱no改善并增加yes方向的不利；本次局部结果仍未支持直接启动统一首答案三倍SFT。', '',
        '## 先统一符号', '',
        '```text', 'SGD:  delta_theta = -lr * G',
        'Adam: delta_theta 取决于一阶动量、二阶矩、bias correction和epsilon',
        '探针损失一阶变化 ≈ grad(L_probe) dot delta_theta',
        '负值：局部下降；正值：局部上升', '```', '',
        '以下是2个no和2个yes训练探针的均值梯度。它们不加入合成更新，不代表全量原错。Final影子使用保存末步LR作统一比较尺度，原训练已结束，没有实际执行第201步。', '',
        *table, '',
        '## 真实状态提供的核对', '',
        f"- 梯度诊断的普通人工8QA、Text8QA与历史第1步完整批次完全一致：{a['same_first_batch_as_gradient_probe_sources']}。",
        f"- 首步动量反推已应用梯度与此前保存梯度的相对L2误差为{a['first_applied_gradient_vs_saved_probe_objective']['relative_l2_difference']:.9e}，余弦{a['first_applied_gradient_vs_saved_probe_objective']['cosine']:.15f}。未发现此处累积归一或坐标顺序错误。",
        f"- 理想FP64末动量公式与历史首步FP32真实位移的相对L2差为{a['first_moment_increment_vs_observed_displacement_relative_l2']:.9e}；不能把理想公式当逐bit GPU位移。",
        '- 175→200真实参数位移与Final探针梯度的投影另存于ANALYSIS.json。它是端点线性化，不是25步实际损失变化积分。第200步动量方向也不能用theta200梯度冒充theta199时的实际损失变化。', '',
        '## 当前可以排除和不能排除的内容', '',
        '- 本轮没有定位到新的Adam实现故障或首步梯度错配。历史参数、moments、科学源码与梯度文件SHA前后不变。',
        '- 数据存在的问题不因本检查消失；但负提升也不能仅凭原始梯度方向归因数据、回放或优化器。',
        '- 旧梯度工件没有独立保存参数名。本次依照两个未变生产者的named_parameters遍历顺序、checkpoint有序参数及首步动量核对重建坐标表；不是重新运行GPU捕获坐标。后续梯度工件应直接携带参数名和shape。',
        '- 没有重新生成checkpoint1的探针NLL；真实首步参数位移的正投影仍是局部一阶预测，不是已经实测到该步NLL增加。BF16计算、有限步和非线性可能影响实际损失。', '',
        '## 复现与学习', '',
        f"分析记录CPU耗时{a['cpu_seconds']:.3f}秒，3项新优化器数学CPU检查通过；另执行4次内存副本上的影子优化器步骤，真实VLM参数更新为0。", '',
        '- [核算脚本](../tools/check_adam_update_directions_v1.py)、[Adam数学与影子步骤](../src/vlm_gradient_mechanism_v1/adam.py)、[原图与方向图解](learning/Adam真实更新方向检查.html)。',
        '- [完整分析](/opt/data/private/longtail_vlm/rarevlm_pilot/outputs/vlm_adam_direction_v1/ANALYSIS.json)、[重建坐标表](/opt/data/private/longtail_vlm/rarevlm_pilot/outputs/vlm_adam_direction_v1/PARAMETER_ORDER_RECONSTRUCTED.json)。', '',
        '后续算法假设需要检查实际优化器位移下的困难正负例方向，再通过独立生成能力和保持任务验证。当前没有新增训练收益，也不据小探针直接放行长训或选择新算法。',
    ]
    REPORT.write_text('\n'.join(report) + '\n')

    def bars(items):
        values = [(label, p, measurements[key]['probes'][p]['first_order_loss_change'])
                  for key, label in items for p in ('probe_no_answer', 'probe_yes_answer')]
        bound = max(abs(v) for _, _, v in values) * 1.1
        height = 70 + len(values) * 34
        body = [f'<svg viewBox="0 0 960 {height}" role="img" aria-label="探针损失一阶变化条形图">',
                f'<line x1="560" x2="560" y1="20" y2="{height-35}" stroke="#445b6b"/>',
                '<text x="370" y="17">← 局部损失下降</text><text x="590" y="17">局部损失上升 →</text>']
        for i, (label, probe, value) in enumerate(values):
            y = 43 + i * 34
            width = abs(value) / bound * 175
            x = 560 - width if value < 0 else 560
            target = 'no' if probe == 'probe_no_answer' else 'yes'
            color = '#167b65' if value < 0 else '#b34a35'
            body.extend([f'<text x="7" y="{y+11}" font-size="13">{html.escape(label)} · {target}</text>',
                         f'<rect x="{x:.4f}" y="{y}" width="{width:.4f}" height="15" fill="{color}"/>',
                         f'<text x="755" y="{y+12}" font-size="13">{value:+.5e}</text>'])
        body.append('</svg>')
        return ''.join(body)

    cards, paths = '', set()
    nll = a['saved_probe_first_answer_nll']
    for i, qid in enumerate(c['groups']['probe']):
        row = c['qa'][str(qid)]
        path = c['images'][row['image_id']]['path']
        paths.add(path)
        cards += (f'<article><img src="{html.escape(path, quote=True)}" alt="原始训练图"><h3>{html.escape(row["question"])}</h3>'
                  f'<p>原GT={row["target"]}；原有initial→Final首答NLL：{nll["initial"][i]:.4f} → {nll["ce"][i]:.4f}。</p>'
                  '<p>这不是新增checkpoint1评测，图像按网页显示尺寸展示。</p></article>')
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SGD与Adam的更新方向检查</title><style>body{{font:16px/1.75 system-ui;background:#f2f5f7;color:#183141;margin:0}}main{{max-width:1100px;margin:auto;padding:24px}}section,article{{background:white;padding:20px;border-radius:12px;margin:20px 0}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}img{{width:100%;height:280px;object-fit:contain;background:#eef2f5}}svg{{width:100%;background:#f7fafb}}pre{{padding:14px;background:#eaf0f4;overflow:auto}}a{{color:#07609b}}@media(max-width:650px){{.grid{{grid-template-columns:1fr}}}}</style><main>
<h1>原始梯度不能直接替代Adam更新方向</h1><p>本轮复用保存的梯度与第1/175/200步状态，0 VLM加载、0 GPU、0真实参数更新；4次影子步骤仅发生在CPU内存副本。</p>
<section><h2>先看公式</h2><pre>delta_SGD  = -lr * G
delta_Adam = -lr * m_hat / (sqrt(v_hat) + eps)
探针损失的一阶变化 ≈ g_probe dot delta_theta</pre><p>坐标逐项缩放和历史动量能改变方向。以下负值表示局部损失下降，正值表示上升；颜色、正负号及坐标文字共同编码。它们不是已实测的逐步NLL变化，也不是泛化分数。</p></section>
<section><h2>初始点：历史首步位移提供直接核对</h2>{bars(choices[:4])}<p>梯度诊断与历史首步的16QA完全相同；动量反推梯度相对误差约5.36×10⁻⁸。CPU普通CE影子结果与历史真实参数位移的投影近似一致，但不宣称GPU逐bit等价。</p></section>
<section><h2>Final点：Adam动量使no探针的符号翻转</h2>{bars(choices[4:])}<p>原始SGD预测no损失上升，带保存动量的Adam影子步骤预测no损失下降。首答案三倍在此处削弱no改善并增加yes方向的不利。该影子步骤沿用最后LR比较，不是恢复原训练至第201步。</p><p>每个面板使用独立对称比例，不跨面板比较条长；具体数值均直接标出。</p></section>
<section><h2>四个真实训练探针</h2><div class="grid">{cards}</div></section>
<section><h2>结论范围</h2><p>当前需要修正归因方法；没有发现新的Adam实现错误。两组小训练探针、两个固定点不能代表全量错误或历史Adam全程。下一项算法实验仍需独立生成收益和保持证据。</p><p><a href="../VLM_ADAM_DIRECTION_CHECK_V1.md">完整报告与代码</a> · <a href="{ROOT / 'ANALYSIS.json'}">原始核算记录</a></p></section></main></html>'''
    missing = [p for p in paths if not Path(p).is_file()]
    if missing:
        raise ValueError(f'Missing original images: {missing}')
    PAGE.write_text(page)
    delivery = {'status': 'completed', 'report': str(REPORT), 'page': str(PAGE),
                'original_images': len(paths), 'missing_images': missing,
                'trained_model_updates': 0, 'new_gpu_forwards': 0}
    (ROOT / 'DELIVERY.json').write_text(json.dumps(delivery, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(delivery, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
