"""Render real saved images/readouts and an explicitly marked teaching control."""

import html
import json
import math
from pathlib import Path


PILOT = Path('/opt/data/private/longtail_vlm/rarevlm_pilot')
PACKAGE = PILOT / 'packages/RareVLM_minimal_20261001'
DIAG = PILOT / 'outputs/vlm_deep_pipeline_diagnosis_v1'
READOUT = PILOT / 'outputs/vlm_fixed_error_readout_v1'
DEST = PACKAGE / 'docs/learning/数据_目标_训练流程深入诊断.html'


def load(path):
    return json.loads(path.read_text())


def jsonlines(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def main():
    cases = load(DIAG / 'data/VISUAL_REVIEW_COHORT.json')
    answer = load(DIAG / 'answer_signal/ANALYSIS.json')
    qas = {q['question_id']: q for q in jsonlines(DIAG / 'answer_signal/QA_ANSWER_SIGNAL.jsonl')}
    pairs = {(r['role'], r['pair_id']): r for r in jsonlines(READOUT / 'PAIR_DIAGNOSES.jsonl')
             if r['task'] == 'train_selected'}
    cohort = load(READOUT / 'data/COHORT.json')
    manifest = load(Path(cohort['manifests']['train_selected']['path']))
    rows = {q['question_id']: q for q in manifest['splits']['dev']}
    bus = next(c for c in cases if c['question_id'] == 'vqa2_train2014:360575013')
    pair = pairs['ce_rank', bus['pair_id']]
    image_paths = set()

    def image_tag(path, caption):
        image_paths.add(path)
        return f'<figure><a href="{html.escape(path, quote=True)}"><img loading="lazy" src="{html.escape(path, quote=True)}" alt="{html.escape(caption, quote=True)}"></a><figcaption>{html.escape(caption)}</figcaption></figure>'

    pair_images = ''
    for side, target in [('positive', 'yes'), ('negative', 'no')]:
        q = rows[pair[f'{side}_question_id']]
        path = manifest['images'][q['image_id']]['path']
        pair_images += image_tag(path, f"原图 · GT={target} · {q['question_id']}")
    table_rows = ''
    defaults = {}
    for role, label in [('initial', '初始'), ('ce', '普通CE'), ('ce_rank', 'CE＋排序')]:
        r = pairs[role, bus['pair_id']]
        m = r['readouts']['canonical']
        g = r['generation']
        defaults[role] = [m['positive_margin'], m['negative_margin']]
        table_rows += (f"<tr><td>{label}</td><td>{m['positive_margin']:+.3f}</td><td>{m['negative_margin']:+.3f}</td>"
                       f"<td>{m['gap']:.3f}</td><td>{g['positive_prediction']} / {g['negative_prediction']}</td></tr>")

    cards = ''
    for case in cases:
        q = qas[case['question_id']]
        a, b = [q['models'][r]['target_first_token_nll'] for r in ('initial', 'ce_rank')]
        cards += ('<article>' + image_tag(case['image_path'], case['question_id'])
                  + f"<h3>{html.escape(case['question'])}</h3>"
                  + f"<p>GT：<b>{case['target']}</b>（原标注{case['answer_vote_count']}/10票）。"
                  + f"初始 / CE / 排序：<code>{case['initial_prediction']} / {case['ce_prediction']} / {case['rank_prediction']}</code>。</p>"
                  + f"<p>曝光步：{case['exposure_steps']}；最终yes−no margin：<b>{case['rank_canonical_margin']:+.3f}</b>；"
                  + f"GT首答案NLL：{a:.3f} → {b:.3f}。</p>"
                  + '<p class="note">图像观察：' + html.escape(case['visual_review']['observation']) + '</p></article>')

    groups = answer['answer_signal_groups']
    nll_table = ''
    for key, title in [('initial_wrong_target_yes', '初始答错、GT=yes'),
                       ('initial_wrong_target_no', '初始答错、GT=no'),
                       ('initial_wrong_qa_rank_still_wrong', '原错且最终仍错'),
                       ('initial_correct_qa_within_initial_wrong_pairs', '原错对内、初始答对的题')]:
        x = groups[key]
        before, after = [x['by_model'][r]['target_first_token_nll']['mean'] for r in ('initial', 'ce_rank')]
        correct = x['by_model']['ce_rank']['correct']
        nll_table += f'<tr><td>{title}</td><td>{x["qa"]}</td><td>{correct}</td><td>{before:.3f} → {after:.3f}</td></tr>'

    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RareVLM：数据、目标与训练流程深入诊断</title>
<style>
body{font:16px/1.75 system-ui,sans-serif;color:#172b3a;background:#f3f6f8;margin:0}main{max-width:1100px;margin:auto;padding:24px}section,article{background:white;border:1px solid #dae2e8;border-radius:12px;padding:22px;margin:20px 0}h1{font-size:27px}h2{font-size:22px}h3{font-size:18px}p{margin:12px 0}.note{color:#465b6a}.warn{border-left:5px solid #b66013;padding-left:15px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}.cards article{margin:0}.cards{display:grid;grid-template-columns:1fr 1fr;gap:20px}figure{margin:0}img{width:100%;height:285px;object-fit:contain;background:#edf1f4}figcaption{font-size:13px;color:#526473;word-break:break-all}table{border-collapse:collapse;width:100%;display:block;overflow:auto}td,th{padding:9px 13px;border-bottom:1px solid #dde5ea;text-align:left;white-space:nowrap}code,pre{font-family:ui-monospace,monospace;background:#edf2f5}pre{padding:16px;overflow:auto}button{padding:9px;margin:6px;border:1px solid #b7c8d5;background:#eef4f9;border-radius:6px;cursor:pointer}input[type=range]{width:75%;vertical-align:middle}svg{max-width:100%;background:#f6f8fa}.metric{font-size:20px;font-weight:650}.links a{margin-right:15px}a{color:#075d9a}@media(max-width:700px){.grid,.cards{grid-template-columns:1fr}main{padding:12px}section,article{padding:16px}h1{font-size:23px}}
</style><main>
<h1>模型到底哪里没有学会？</h1>
<p>本页连接真实原图、已有预测、首答案概率、训练日志和损失源码。本轮<b>0次训练更新、0次GPU前向</b>；没有读取确认集，没有改写标签或历史分数。</p>
<section><h2>先区分三个量</h2><p><b>相对排序</b>：正图的yes−no分数比负图高。<b>绝对判定</b>：正图分数要大于0，负图小于0。<b>实际生成</b>：模型在整个词表中选出的答案；词形集合概率与最大单token并不相同。</p>
<pre>m = logit(yes) − logit(no)
gap = m正图 − m负图
Lrank = softplus(−gap)
排序正确：gap > 0
两图判对：m正图 > 0 且 m负图 < 0</pre>
<p>最终筛选23个错对中，20个canonical排序已正确。排序可以改善分离，却没有直接限制共同偏移；并不意味着原CE完全没有绝对监督。</p></section>
<section><h2>真实案例：这些巴士是否同色？</h2><p>同一问题：<b>__QUESTION__</b>。两图均保留原人类标签，展示使用原始文件；网页显示尺寸不是训练processor的输入重建。</p>
<div class="grid">__PAIR_IMAGES__</div>
<table><thead><tr><th>模型</th><th>正图margin</th><th>负图margin</th><th>gap</th><th>真实生成 正/负</th></tr></thead><tbody>__PAIR_TABLE__</tbody></table>
<p class="warn">正图里的巴士都有黄色主体和红色条带，原GT=yes有可见依据；模型仍答no。这说明不能把全部残错归因于错标。</p>
<h3>拖动分数，观察排序与判定何时分离</h3>
<p>初始值来自该对的CE＋排序Final实测。操作后的值仅是<b>教学模拟</b>，不会运行模型或改实验。</p>
<p>正图m：<input id="pos" type="range" min="-16" max="16" step=".125"><output id="pv"></output></p>
<p>负图m：<input id="neg" type="range" min="-16" max="16" step=".125"><output id="nv"></output></p>
<button id="shift">两图共同＋1</button><button id="restore">恢复Final实测</button><span id="mode"></span>
<svg viewBox="0 0 700 140" role="img" aria-label="两图margin与零判定边界"><line x1="350" x2="350" y1="5" y2="115" stroke="#485e70" stroke-dasharray="4 4"></line><text x="350" y="135" text-anchor="middle">0：yes/no边界</text><text x="8" y="40">正图</text><text x="8" y="90">负图</text><rect id="bp" y="20" height="28" fill="#167c80"></rect><rect id="bn" y="70" height="28" fill="#7656a6"></rect></svg>
<p id="metrics" class="metric"></p><p id="decision"></p>
<p class="note">条形图只是实测/教学logit margin，不是attention。并列m=0没有严格判对，真实全词表argmax另见上表。</p></section>
<section><h2>用答案token的损失核对“学会”</h2><p>人工题实际监督三个token：<code>yes/no, &lt;|im_end|&gt;, 换行</code>，在线CE是它们的均值。以下只读既有prompt-only全词表概率计算<b>−log p(GT首答案)</b>，没有把三token均值当作答案损失。</p>
<table><thead><tr><th>训练诊断组</th><th>QA数</th><th>排序Final答对</th><th>GT首答案NLL 初始→Final</th></tr></thead><tbody>__NLL_TABLE__</tbody></table>
<p>原错yes题修复11/34，原错no题0/12；35个原错且仍错QA的首答案NLL几乎未变。模型对已有正确答案更自信，不等于困难决策被纠正。各组是错例富集诊断，不代表总体训练准确率。</p></section>
<section><h2>数据与训练：已经查到的事实</h2>
<p><b>数据准入：</b>事前审核unknown的T02/T08仍随14/16整池门槛进入训练，T08还进入最终9个修复pair。审核总体通过没有排除已知不确定样本；历史分数保留。</p>
<p><b>曝光与损失：</b>修复9对、残错36对均2/3只见1次、1/3见2次。原错QA虽仅61/1600人工曝光，却占人工CE值总和45.865%；不能把曝光份额或loss份额当参数梯度份额。</p>
<p><b>更新与框架：</b>已记录梯度不支持长期零梯度；首答mask、因果移位和四微批累积未发现错误。实际只训练36层语言q/v LoRA共1843200参数，视觉与merger冻结；容量是否不足仍需独立对照。</p>
<pre># loss.py:68 — 每题平均
qa_ce = values.sum(1) / counts
# train.py:268 — 每更新16题、4个pair，跨微批累积一次step
objective = qa_ce.sum()/16 + 0.1 * pair_rank.sum()/4</pre>
<p>每人工首答案在CE表达式中的系数为1/48。这不是“只获得1/48的参数梯度”，也不证明提高权重就能改善独立任务。</p></section>
<section><h2>8个固定原图残错</h2><p>从36残错pair内按固定SHA次序选4个假阳性、4个假阴性。以下是助手逐图观察，不是专家新标注，也不能估计全池噪声率。</p><div class="cards">__CASES__</div></section>
<section><h2>从开源实现学什么</h2><p>TRL的有效token平均、LLaMA-Factory的assistant mask与视觉模块开关、Qwen兼容代码的LoRA分支，都应与实际输入及模块清单对应。框架默认值不能替代协议；更换Trainer不会自动修好语义含混的监督。</p>
<p class="links"><a href="../VLM_DEEP_PIPELINE_DIAGNOSIS_V1.md">主诊断报告与源码行号</a><a href="__UPSTREAM__">3仓库固定提交精读</a><a href="../VLM_DEEP_DATA_DIAGNOSIS_V1.md">逐图及数据证据</a></p>
<p>下一轮先统一可靠监督准入，再选择一个监督变化做匹配CE短预算对照。数据、LoRA位置、损失和训练时长不能一起改变后声称定位了原因。DPO需同图同问可靠偏好；GRPO需可靠奖励与有效采样差异。</p></section>
<p class="note">所有数值来自保存工件；未补新模型结果。首答案dense读出与训练tail-head在BF16 GEMM形状上可能不同，本页不宣称逐bit等价或隔离了全部因果。</p>
</main><script>
const D=__DEFAULTS__;let edited=false;const p=document.getElementById('pos'),n=document.getElementById('neg');
function paint(){const a=+p.value,b=+n.value,g=a-b,sp=Math.max(-g,0)+Math.log1p(Math.exp(-Math.abs(g)));
document.getElementById('pv').textContent=a.toFixed(3);document.getElementById('nv').textContent=b.toFixed(3);
for(const [id,v] of [['bp',a],['bn',b]]){let x=350+v*18;const el=document.getElementById(id);el.setAttribute('x',Math.min(350,x));el.setAttribute('width',Math.max(.8,Math.abs(x-350)));}
document.getElementById('metrics').textContent='gap = '+g.toFixed(3)+'；排序loss = '+sp.toFixed(6);
document.getElementById('decision').textContent='相对排序：'+(g>0?'正确':g===0?'并列':'逆序')+'；正图条件判定：'+(a>0?'yes':a<0?'no':'并列')+'；负图条件判定：'+(b>0?'yes':b<0?'no':'并列')+'；两图是否判对：'+(a>0&&b<0?'是':'否');
document.getElementById('mode').textContent=edited?'当前：教学改动值':'当前：Final实测margin';}
function restore(){[p.value,n.value]=D.ce_rank;edited=false;paint();}
p.oninput=n.oninput=()=>{edited=true;paint();};document.getElementById('restore').onclick=restore;
document.getElementById('shift').onclick=()=>{if(+p.value<=15&&+n.value<=15){p.value=+p.value+1;n.value=+n.value+1;edited=true;paint();}};restore();
</script></html>'''
    for key, val in {
        '__QUESTION__': html.escape(pair['question']), '__PAIR_IMAGES__': pair_images,
        '__PAIR_TABLE__': table_rows, '__NLL_TABLE__': nll_table, '__CASES__': cards,
        '__UPSTREAM__': str(DIAG / 'UPSTREAM_COMPARISON.md'),
        '__DEFAULTS__': json.dumps(defaults, allow_nan=False).replace('<', '\\u003c'),
    }.items():
        page = page.replace(key, val)
    missing = [p for p in image_paths if not Path(p).is_file()]
    if missing or '__DEFAULTS__' in page:
        raise ValueError(f'Incomplete local visualization: {missing}')
    DEST.write_text(page)
    result = {'status': 'completed_saved_records_only', 'path': str(DEST),
              'case_images': len(cases), 'unique_original_images': len(image_paths),
              'missing_images': missing, 'models_loaded': 0, 'optimizer_updates': 0,
              'interactive_default_margins': defaults}
    (DIAG / 'LEARNING_PAGE.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
