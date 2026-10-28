"""Render the new scope study from immutable initialization and actual progress."""

import html
import json
from pathlib import Path
import struct


PACKAGE = Path(__file__).resolve().parents[1]
OUTPUTS = PACKAGE.parents[1] / 'outputs'
ROOT = OUTPUTS / 'vlm_attention_scope_v1'
DEST = PACKAGE / 'docs/learning/QV与QKVO_LoRA容量对照.html'


def read(path):
    return json.loads(path.read_text())


def main():
    config = read(PACKAGE / 'configs/vlm_attention_scope_v1.json')
    init = read(ROOT / 'initialization/MANIFEST.json')
    manifest = read(ROOT / 'data/TRAINING_MANIFEST.json')
    lookup = {str(q['question_id']): q for q in manifest['splits']['train_all']}
    with (ROOT / 'initialization/adapter/adapter_model.safetensors').open('rb') as f:
        header = json.loads(f.read(struct.unpack('<Q', f.read(8))[0]))
    shapes = ''
    for projection in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
        key = 'base_model.model.model.layers.0.self_attn.' + projection
        a, b = [header[key + '.lora_' + factor + '.weight']['shape'] for factor in ('A', 'B')]
        shapes += f'<tr><td>{projection}</td><td>{a}</td><td>{b}</td><td>{"保留原SFT值" if projection in ("q_proj", "v_proj") else "新A随机，B严格0"}</td></tr>'
    rows = ''
    for role, label in [('initial', '共同初始'), ('qv', 'Q/V控制'), ('qkvo', 'Q/K/V/O候选')]:
        directory = ROOT / 'initial' if role == 'initial' else ROOT / 'runs' / role
        values = []
        for task in ('human', 'text'):
            p = directory / task / 'COMPLETE.json'
            values.append(f'{read(p)["score_percent"]:.6f}%' if p.exists() else '待完成')
        progress = directory / 'train/PROGRESS.json'
        state = read(progress) if progress.exists() else {}
        steps = f'{state.get("completed_updates", state.get("updates", 0))}/200' if role != 'initial' else '0更新'
        rows += f'<tr><td>{label}</td><td>{steps}</td><td>{values[0]}</td><td>{values[1]}</td></tr>'
    parity = read(ROOT / 'BASELINE_PARITY.json') if (ROOT / 'BASELINE_PARITY.json').exists() else {}
    parity_text = '完整1024QA新图与原基线逐题一致，已放行训练' if parity.get('passed') else '完整新基线仍在核对，尚未放行训练' if not parity else '初始一致性失败，停止训练'
    cards, paths = '', set()
    for qid in ('vqa2_train2014:360575013', 'vqa2_train2014:444344000'):
        q = lookup[qid]
        path = manifest['images'][q['image_id']]['path']
        paths.add(path)
        cards += f'<article><img src="{html.escape(path, quote=True)}" alt="原始训练图"><h3>{html.escape(q["question"])}</h3><p>原GT={q["target"]}。这是已有可靠训练错例，用来理解问题，不据它单独判断新候选成功。</p></article>'
    result_path = ROOT / 'SCREEN_ANALYSIS.json'
    final = read(result_path) if result_path.exists() else {}
    result_note = '完整筛选尚未完成；当前没有新增能力提升结论。' if not final else f'完整筛选联合通过={final["passed"]}；区间、纠正与新错见原始分析。'
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Q/V与Q/K/V/O LoRA容量对照</title><style>body{{font:16px/1.75 system-ui;background:#f2f5f7;color:#183141;margin:0}}main{{max-width:1100px;margin:auto;padding:24px}}section,article{{background:white;padding:20px;border-radius:12px;margin:20px 0}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}img{{width:100%;height:290px;object-fit:contain;background:#eef2f5}}svg{{width:100%;background:#f7fafb}}pre{{padding:16px;background:#eaf0f4;overflow:auto}}table{{width:100%;border-collapse:collapse;display:block;overflow:auto}}td,th{{padding:10px;border-bottom:1px solid #dde3e7;white-space:nowrap;text-align:left}}button{{padding:10px;margin:8px;border:1px solid #8aabb9;border-radius:6px;background:#eaf3f8;cursor:pointer}}a{{color:#07609b}}@media(max-width:650px){{.grid{{grid-template-columns:1fr}}}}</style><main>
<h1>扩大注意力适配范围，是否真能学到更多？</h1><p>本轮只改变可训练的语言注意力投影范围。两组共用同一初始化、593训练对＋1600Text、3200曝光、200更新、普通CE及学习率。视觉tower/merger仍冻结；新增K/O也是共享语言decoder参数，不能称专门视觉模块。</p>
<section><h2>Q、K、V、O分别做什么</h2><p><b>Q</b>生成当前位置的查询，<b>K</b>生成可匹配的键，<b>V</b>承载被汇总的内容，<b>O</b>将注意力汇总映射回隐藏表示。Q/V也能影响视觉信息利用；本轮不预设它们表达能力不足。</p>
<button onclick="scope('qv')">控制：只训练Q/V</button><button onclick="scope('qkvo')">候选：训练Q/K/V/O</button><p id="scope"></p>
<svg viewBox="0 0 1000 280" role="img" aria-label="共享decoder注意力结构和可训练投影示意"><defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10Z" fill="#527080"/></marker></defs>
<g stroke="#527080" stroke-width="2" fill="none" marker-end="url(#arrow)"><path d="M130 140L240 50"/><path d="M130 140L240 140"/><path d="M130 140L240 230"/><path d="M350 50L500 110"/><path d="M350 140L500 140"/><path d="M350 230L500 170"/><path d="M660 140L740 140"/><path d="M850 140L930 140"/></g>
<rect x="10" y="105" width="120" height="70" fill="#e1eaf0"/><text x="70" y="135" text-anchor="middle">图文token</text><text x="70" y="158" text-anchor="middle">隐藏状态h</text>
<rect x="240" y="20" width="110" height="60" fill="#b9e5dc"/><text x="295" y="57" text-anchor="middle">Q LoRA</text>
<rect id="key" x="240" y="110" width="110" height="60"/><text x="295" y="147" text-anchor="middle">K LoRA</text>
<rect x="240" y="200" width="110" height="60" fill="#b9e5dc"/><text x="295" y="237" text-anchor="middle">V LoRA</text>
<rect x="500" y="100" width="160" height="80" fill="#e1eaf0"/><text x="580" y="133" text-anchor="middle">softmax(QKᵀ/√d)</text><text x="580" y="159" text-anchor="middle">汇总V</text>
<rect id="out" x="740" y="110" width="110" height="60"/><text x="795" y="147" text-anchor="middle">O LoRA</text><text x="945" y="147">输出</text></svg>
<p>两组图结构相同，按钮只切换requires_grad示意。绿色表示可训练，灰色为冻结。此图不是模型attention实测。</p></section>
<section><h2>为什么从同一个函数起步</h2><pre>原投影： y = W_frozen h + (alpha/r) B A h
新K/O：  A按固定seed初始化；B = 0
所以初始新增输出 (alpha/r) B A h = 0
控制组：新K/O的A、B一直冻结
候选组：首步B可有非零梯度，A因B=0而初始零梯度是正常现象</pre>
<p>原Q/V的{init['original_tensors']}个FP32张量原值保留；新增{init['added_tensors']}个张量。控制更新{config['expected_trainable_by_role']['qv']:,}参数，候选更新{config['expected_trainable_by_role']['qkvo']:,}参数；额外训练成本另报。</p>
<table><thead><tr><th>实际第0层投影</th><th>A shape</th><th>B shape</th><th>初始化</th></tr></thead><tbody>{shapes}</tbody></table>
<p>张量shape直接来自本轮真实safetensors头，不是教学假设。K/V输出维度小于Q/O与底座的分组查询注意力结构有关。</p></section>
<section><h2>生成时的实际状态快照</h2><p>{parity_text}。静态页面不会自行刷新实验状态。</p><table><thead><tr><th>模型</th><th>更新数</th><th>human QA</th><th>TextVQA</th></tr></thead><tbody>{rows}</tbody></table><p>{result_note}</p></section>
<section><h2>真实训练问题</h2><div class="grid">{cards}</div><p>原图按网页显示尺寸展示，不冒充processor输入。新增容量是否能纠错，必须看完整独立图像筛选、纠正/新错与保持任务。</p></section>
<section><h2>验收与代码</h2><p>候选须对初始与新Q/V控制均QA至少+1pp、成对至少+2pp、净纠正至少6且组件簇区间下界为正，同时保护yes/no及Text切片；失败停止，不扩epoch或按筛选集调参。</p>
<p><a href="../VLM_ATTENTION_SCOPE_V1_PROTOCOL.md">冻结协议</a> · <a href="../../src/vlm_attention_scope_v1/model.py">初始化与冻结代码</a> · <a href="../../src/vlm_attention_scope_v1/train.py">真实训练循环</a> · <a href="{ROOT}">实验输出</a></p></section></main>
<script>function scope(mode){{let full=mode==='qkvo';for(let id of ['key','out'])document.getElementById(id).setAttribute('fill',full?'#b9e5dc':'#e1eaf0');document.getElementById('scope').textContent=full?'候选：Q/K/V/O均可更新，3,686,400参数':'控制：Q/V可更新，K/O保持零输出，1,843,200参数';}}scope('qv');</script></html>'''
    missing = [p for p in paths if not Path(p).is_file()]
    if missing:
        raise ValueError(f'Missing original images: {missing}')
    DEST.write_text(page)
    print(json.dumps({'status': 'rendered_saved_records_only', 'path': str(DEST), 'missing_images': missing,
                      'model_forwards': 0, 'optimizer_updates': 0}, ensure_ascii=False))


if __name__ == '__main__':
    main()
