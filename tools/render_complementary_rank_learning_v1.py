#!/usr/bin/env python3
"""Render a derived Chinese lesson from real complementary-pair study artifacts.

This tool is outside the scientific input lock. It reads saved artifacts only;
it never trains, generates model answers, changes labels, or copies source images.
All browser resources are local and the interactive numbers are teaching values.
"""

import argparse
from datetime import datetime
import hashlib
from html import escape
import json
import os
from pathlib import Path
from urllib.parse import quote
from zoneinfo import ZoneInfo


PRIVATE = Path('/opt/data/private')
PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT.parents[1] / 'outputs/vlm_complementary_rank_v1'
DEFAULT_OUTPUT = PROJECT / 'docs/learning/同问异图_跨图排序SFT.html'
CONFIG = PROJECT / 'configs/vlm_complementary_rank_v1.json'
PAIR_SALT = 'complementary_rank_learning_v1_fixed_train_pairs'
ROLE_NAMES = {'initial': '初始模型', 'ce': '匹配 CE 控制', 'ce_rank': 'CE＋0.1 跨图排序'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def local_url(path, output):
    path = Path(path).resolve(strict=True)
    if not path.is_relative_to(PRIVATE):
        raise ValueError(f'Local resource is outside the persistent private workspace: {path}')
    return escape(quote(os.path.relpath(path, output.parent), safe='/._-'), quote=True)


def fixed_key(pair):
    payload = json.dumps([PAIR_SALT, pair['pair_id']], ensure_ascii=False,
                         separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def predictions(path):
    """Read complete saved generation rows without repairing an active file."""
    if not path.exists():
        return {}
    rows = {}
    for line in path.read_bytes().splitlines(keepends=True):
        if not line.endswith(b'\n'):
            break
        row = json.loads(line)
        rows[row['question_id']] = row['prediction']
    return rows


def pair_cards(manifest, manifest_path, root, output):
    pairs = manifest['complementary_pairs']['train']
    selected = sorted(pairs, key=fixed_key)[:4]
    if len(selected) != 4:
        raise ValueError('Four real training pairs are required; no placeholder images will be made')
    rows = {row['question_id']: row for row in manifest['splits']['train_all']}
    initial_path = root / 'initial/train/human/predictions.jsonl'
    initial = predictions(initial_path)
    cards, displayed = [], []
    for pair in selected:
        yes = rows[pair['positive_question_id']]
        no = rows[pair['negative_question_id']]
        if (yes['target'] != 'yes' or no['target'] != 'no'
                or yes['question'] != no['question'] or yes['image_id'] == no['image_id']):
            raise ValueError('Displayed pair does not match its exact-question/opposite-label contract')
        figures = []
        for row in (yes, no):
            image = manifest['images'][row['image_id']]
            path = Path(image['path'])
            if not path.is_absolute():
                path = manifest_path.parent / path
            path = path.resolve(strict=True)
            if not path.is_file():
                raise FileNotFoundError(path)
            url = local_url(path, output)
            prediction = initial.get(row['question_id'])
            prediction_text = ('未预测／未验证：尚无该题保存的初始模型生成回答。'
                               if prediction is None else
                               '初始模型实际保存生成：<code>' + escape(str(prediction)) + '</code>。')
            figures.append(f'''<figure>
<a href="{url}"><img src="{url}" alt="原始图像：{escape(row['question'], quote=True)}" loading="lazy"></a>
<figcaption>原始人类监督：<strong>{escape(row['target'])}</strong>；
一致票数 {int(row['answer_vote_count'])}/10。<br>
{prediction_text}<br>候选模型：本训练例没有新增生成评测，不能把监督标签当预测。</figcaption>
<details><summary>查看数据身份与原始10人答案</summary>
<p>QA：<code>{escape(str(row['question_id']))}</code><br>
图像：<code>{escape(str(row['image_id']))}</code><br>
原图 SHA：<code>{escape(image['sha256'])}</code><br>
原始答案：<code>{escape(json.dumps(row['answers'], ensure_ascii=False))}</code></p></details>
</figure>''')
        cards.append(f'''<article class="card"><h3>{escape(pair['question'])}</h3>
<p class="muted">训练配对：<code>{escape(pair['pair_id'])}</code></p>
<div class="images">{''.join(figures)}</div></article>''')
        displayed.append(pair['pair_id'])
    baseline_note = ('页面读取初始模型保存回答：<code>' + escape(str(initial_path)) + '</code>。'
                     if initial_path.exists() else
                     '当前未读取到初始模型训练集生成文件；图片下的 yes/no 均为原始人类监督。')
    return ''.join(cards), displayed, baseline_note


def metric(value):
    return f'{float(value):.4f}'


def analysis_section(path, stage):
    title = '独立筛选' if stage == 'screen' else '条件确认'
    if not path.exists():
        return (f'<section><h2>{title}结果</h2><p class="notice">未验证：尚无 '
                f'<code>{escape(path.name)}</code>。训练 loss 或教学数字不代表能力提升。</p></section>')
    data = read(path)
    score_rows = []
    for role in ('initial', 'ce', 'ce_rank'):
        values = data['scores_percent'][role]
        score_rows.append(f'<tr><th>{ROLE_NAMES[role]}</th><td>{metric(values["qa"])}</td>'
                          f'<td>{metric(values["pair"])}</td><td>{metric(values["text"])}</td></tr>')
    comparison_rows = []
    for name, label in [('qa_vs_initial', 'QA：候选−初始'), ('qa_vs_ce', 'QA：候选−CE'),
                        ('pair_vs_initial', '两图全对：候选−初始'), ('pair_vs_ce', '两图全对：候选−CE')]:
        result = data['comparisons'][name]
        low, high = result['image_cluster_95_ci_pp']
        comparison_rows.append(f'<tr><th>{label}</th><td>{metric(result["delta_pp"])}</td>'
                               f'<td>[{metric(low)}, {metric(high)}]</td>'
                               f'<td>{int(result["corrected"])}/{int(result["broken"])}/'
                               f'{int(result["net_corrected"])}</td></tr>')
    slice_rows = []
    for key, result in data['human_slices'].items():
        slice_rows.append(f'<tr><th>人工 {escape(key)}</th><td>{metric(result["delta_pp"])}</td></tr>')
    for key, result in data['text_parent_frequency_slices'].items():
        slice_rows.append(f'<tr><th>Text {escape(key)}</th><td>{metric(result["delta_pp"])}</td></tr>')
    failures = [name for name, passed in data['criteria'].items() if not passed]
    if not data['passed']:
        boundary = '联合门槛未通过。保留局部正负变化；不能宣称已得到本协议要求的能力提升，也不能据此否定所有 SFT。'
    elif stage == 'screen':
        boundary = '筛选通过，仅支持进入预登记的确认阶段；确认结果尚须单独查看。'
    else:
        boundary = '该内部确认协议通过；结论仍限本数据、本续训种子和已声明指标，不外推正式基准或多种子稳定性。'
    failed_html = ('<p>未通过项：' + '、'.join('<code>' + escape(name) + '</code>' for name in failures) + '。</p>'
                   if failures else '<p>本阶段已记录门槛全部通过。</p>')
    return f'''<section><h2>{title}真实结果</h2><p class="notice">{boundary}</p>
<p>以下数字读取 <code>{escape(path.name)}</code>，均是已保存的实际评测。准确率／得分单位为 %，增益单位为百分点。</p>
<table><thead><tr><th>模型</th><th>QA 准确率</th><th>两图全对率</th><th>TextVQA</th></tr></thead>
<tbody>{''.join(score_rows)}</tbody></table>
<table><thead><tr><th>配对比较</th><th>增益</th><th>95% 图像连通分量簇区间</th><th>纠正／新错／净收益</th></tr></thead>
<tbody>{''.join(comparison_rows)}</tbody></table>
<details><summary>正负切片与完整门槛</summary>
<table><thead><tr><th>候选相对初始</th><th>增益（百分点）</th></tr></thead><tbody>{''.join(slice_rows)}</tbody></table>
{failed_html}</details><p class="muted">{escape(data['scope'])}</p></section>'''


def runtime_status(root):
    parts = []
    headroom = root / 'BASELINE_HEADROOM.json'
    if headroom.exists():
        result = read(headroom)
        parts.append('训练前量化空间门槛：' + ('通过' if result['training_permitted'] else '未通过，停止训练') + '。')
    for role in ('ce', 'ce_rank'):
        path = root / f'runs/{role}/train/PROGRESS.json'
        if not path.exists():
            parts.append(f'{ROLE_NAMES[role]}：尚无训练进度记录。')
            continue
        result = read(path)
        count = result.get('completed_updates', result.get('updates'))
        parts.append(f'{ROLE_NAMES[role]}：保存状态 {escape(str(result["status"]))}，'
                     f'{int(count)}/200 更新，累计 active {float(result["active_seconds"])/60:.1f} 分钟。')
    return '<p>' + '<br>'.join(parts) + '</p>'


def data_review_status(root, manifest_path):
    parts = []
    revision1_path = root / 'data_preparation/DATA_REVIEW_V1_RESULT.json'
    if revision1_path.exists():
        result = read(revision1_path)
        totals = {split: sum(row['split'] == split for row in result['rows'])
                  for split in ('train', 'screen')}
        parts.append(f'R1 数据盲审：训练 {int(result["train_passed_pairs"])}/{totals["train"]}、'
                     f'筛选 {int(result["screen_passed_pairs"])}/{totals["screen"]}；'
                     f'保存状态 <code>{escape(result["status"])}</code>。'
                     f'当时模型预测 {int(result["model_predictions"])}、训练更新 '
                     f'{int(result["training_updates"])}；这是监督数据准入失败，不是模型训练失败或能力提升。')
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    # The study supports a revision-local receipt or its registered shared
    # admission receipt. A receipt for a different manifest cannot mark R2 passed.
    review_paths = [manifest_path.parent / 'DATA_REVIEW.json', root / 'data_preparation/DATA_REVIEW.json']
    current = None
    for path in dict.fromkeys(review_paths):
        if path.exists():
            result = read(path)
            if result.get('manifest_sha256') == manifest_sha:
                current = result
                break
    version = 'R2' if manifest_path.parent.name == 'revision2' else '当前版本'
    if current is None:
        parts.append(f'{version} 数据盲审待核：尚无与当前清单 SHA 绑定的准入记录；不能据数据准备完成宣称可训练或已提分。')
    else:
        state = '通过' if current['status'] == 'passed' else '未通过'
        parts.append(f'{version} 数据准入：{state}；保存状态 <code>{escape(current["status"])}</code>。'
                     '数据通过也不等于模型能力验证通过。')
    return '<p class="notice">' + '<br>'.join(parts) + '</p>'


def render(root, output, manifest_path=None):
    root, output = Path(root).resolve(), Path(output).resolve()
    if not root.is_relative_to(PRIVATE) or not output.is_relative_to(PRIVATE):
        raise ValueError('Study input and derived page must remain under /opt/data/private')
    manifest_path = Path(manifest_path or read(CONFIG)['dataset_manifest']).resolve()
    if not manifest_path.is_relative_to(PRIVATE):
        raise ValueError('Dataset manifest must remain under /opt/data/private')
    manifest = read(manifest_path)
    if manifest['schema_version'] != 'rarevlm.vlm_data.v2':
        raise ValueError('Expected the registered complementary-pair manifest schema')
    cards, selected, prediction_note = pair_cards(manifest, manifest_path, root, output)
    counts = {name: len(manifest['complementary_pairs'][name]) for name in ('train', 'screen', 'confirm')}
    loss_url = local_url(PROJECT / 'src/vlm_complementary_rank/loss.py', output)
    train_url = local_url(PROJECT / 'src/vlm_complementary_rank/train.py', output)
    analysis = analysis_section(root / 'SCREEN_ANALYSIS.json', 'screen')
    analysis += analysis_section(root / 'CONFIRM_ANALYSIS.json', 'confirm')
    generated = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>同问异图：跨图排序 SFT</title>
<style>
body{font:16px/1.65 system-ui,sans-serif;max-width:1080px;margin:32px auto;padding:0 18px;color:#17212c;background:#fafbfc}
h1,h2,h3{line-height:1.3}section{margin:30px 0}code,pre{font-family:ui-monospace,monospace;overflow-wrap:anywhere}
pre{white-space:pre-wrap;background:#edf1f5;padding:14px}a{color:#165aac}.muted{color:#526071;font-size:.9rem}
.notice{background:#fff2d2;border-left:4px solid #a46a00;padding:12px}.card{background:white;border:1px solid #ccd5df;padding:16px;margin:16px 0}
.images{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}figure{margin:0}img{display:block;width:100%;height:310px;object-fit:contain;background:#f1f3f5}
figcaption{margin-top:8px}table{border-collapse:collapse;width:100%;margin:14px 0;font-size:.94rem}th,td{border:1px solid #c8d1da;padding:7px;text-align:left}
label{display:inline-block;margin:8px 18px 8px 0}input[type=number]{width:74px}input[type=range]{width:240px}output{font-variant-numeric:tabular-nums}
.math{font-family:ui-monospace,monospace;background:#eaf2fa;padding:14px}.demo{border:1px solid #bbcde0;padding:16px;background:white}
@media(max-width:650px){.images{grid-template-columns:1fr}table{font-size:.8rem}img{height:270px}body{margin:18px auto}}
</style><body>
<h1>同问异图：让同一个问题随图像改变答案</h1>
<p>研究假设：仅提高所有图片的 yes 倾向，可能同时修好一部分题、破坏另一部分题。
本轮用<strong>同一问题、两幅真实图像、相反的人类答案</strong>，检验是否学到额外的图像条件区分。
这是一项待验证的监督目标研究，不预先承诺提分。</p>
<p class="muted">派生学习页生成于 __TIME__（北京时间）；不参与科学输入锁，不改变监督、训练设置或评测。</p>
<section><h2>1. 实际数据与预算</h2>
<p>当前清单：训练 __TRAIN__ 对、筛选 __SCREEN__ 对、确认 __CONFIRM__ 对；来自官方 VQA-v2 train2014 互补配对，
每边保留原问、原图与至少 9/10 人一致的 yes/no 答案。原始人类一致性不等于新增专家复核，也不能证明预训练未见。</p>
<p>计划每组 200 更新、3200 次 QA 曝光：每更新 8 条 Text 回放＋4 对人工问答（8 QA）。
共 1600 次 Text＋1600 次人工 QA，即 800 次配对抽取；训练清单中的 __TRAIN__ 对不是 800 个独立配对。
重复曝光不能增加独立样本数。两组使用相同输入、顺序、初始化与预算。</p>
__DATA_REVIEW_STATUS__
__STATUS__
</section>
<section><h2>2. 先看 4 对真实训练原图</h2>
<p>以下配对按固定盐值的 pair_id 哈希排序取前 4 对，选择不依赖模型分数。
左右均为原始图像，仅在网页等比显示；不是模型处理后的输入图、裁剪图或注意力图。
它们属于训练数据，不能证明独立泛化。__PREDICTION_NOTE__</p>
__CARDS__
</section>
<section><h2>3. 共同偏置为什么不能降低跨图排序损失？</h2>
<p>从首答案位置取原始词表 logits：<code>m(I,q)=z_yes−z_no</code>。
同一问题的 yes 图希望有较大间隔，no 图希望有较小间隔；比较 <code>gap=m(yes图,q)−m(no图,q)</code>。</p>
<div class="demo">
<p class="notice"><strong>下面所有 a、c、b、margin、gap 和 loss 数字均为教学示意，非模型读数、非注意力。</strong></p>
<label>yes 图条件项 a（教学） <input id="a" type="number" value="1" step="0.1"></label>
<label>no 图条件项 c（教学） <input id="c" type="number" value="-1" step="0.1"></label><br>
<label>共同 yes 偏置 b（教学） <input id="b" type="range" min="-5" max="5" value="0" step="0.1"> <output id="bias"></output></label>
<div class="math" id="demo-values" aria-live="polite"></div>
<p>只滑动 b：两图间隔一起平移，gap 与排序 loss 不变；调整 a 或 c 才会改变相对区分。
这只证明<strong>附加项对纯共同偏置不敏感</strong>。模型仍可能学其他捷径，必须看独立生成评测。</p>
</div>
<pre>m_yes图 = a + b
m_no图  = c + b
gap      = (a+b) − (c+b) = a−c
L_rank   = softplus(−gap) = log(1 + exp(−gap))

CE 控制：L_update = Σ16题 CE_i / 16
候选：   L_update = Σ16题 CE_i / 16 + 0.1 × Σ4对 L_rank / 4</pre>
<p>每题 CE 仍包含全部 assistant 目标 token，并先按该题 token 数取均值。
排序项只比较首答案的 yes/no logits；它不替代生成答案准确率，也不是标准 DPO、PPO 或 GRPO。
候选与控制都用同一次 actor forward 计算 CE 与 rank，不增加 reference 模型前向。</p>
</section>
<section><h2>4. 对照真实代码：为什么首目标位置要减 1？</h2>
<p>假设首个 assistant 目标 token 在 <code>labels[t]</code>，因果模型用前一位置的
<code>logits[t−1]</code> 预测它。若误取 <code>logits[t]</code>，模型已经看到了答案 token，读出的其实是下一 token 的分布。</p>
<pre>input位置： …  t−1（assistant前缀末尾）  t（yes）  t+1（结束标记）
预测目标： …       labels[t]          labels[t+1]

result = forward_pair_loss(model, batch, [(0,1), (2,3)],
                           yes_token_id, no_token_id, tail=True)
loss = result['qa_ce'].sum() / 16
loss += lambda_rank * result['pair_rank'].sum() / 4
loss.backward()</pre>
<p><a href="__LOSS_URL__">loss.py</a>：<code>loss_from_logits</code> 显式对齐 dense／tail logits 与原序列，
从 shift 后第一个非 −100 的目标位置读取 logits；<code>forward_pair_loss</code> 只做一次 forward。
本地 tokenizer 解析 yes/no 的实际 ID，不把词表 ID 写死。</p>
<p><a href="__TRAIN_URL__">train.py</a>：<code>train(config, role, directory)</code> 固定配对微批次，
将 CE 按整更新的 16 QA、rank 按整更新的 4 对归一。
每题实际 token、图像网格、监督位置、CE 与 margin 写入 <code>INPUT_TRACES.jsonl</code>；这些训练记录不等于测试提升。</p>
</section>
<section><h2>5. 什么才算支持这个假设？</h2>
<p>筛选集上同时比较候选与初始模型、候选与新 CE 控制：QA 至少＋1 个百分点，两图全对率至少＋2 个百分点，
各自净纠正至少 6 个且连通分量簇区间下界大于 0；同时守住 yes/no 与 Text 切片。
筛选通过才进入固定确认集；失败不自动加 epoch、不挑 checkpoint。</p>
<p>一对只有两图都答对才记 1。若模型无视图像，对同一个问题一律答 yes 或一律答 no，这一对最多对一题，无法记两图全对。
单看 rank loss 下降、margin 变大或训练题答对，均不能替代独立结果。</p>
</section>
__ANALYSIS__
<footer class="muted">页面只展示派生学习内容与已保存证据，不自动修改简历或增加成果声明。</footer>
<script>
function updateDemo(){
  const a=Number(document.getElementById('a').value), c=Number(document.getElementById('c').value),
        b=Number(document.getElementById('b').value);
  const el=document.getElementById('demo-values');
  if(![a,c,b].every(Number.isFinite)){el.textContent='请输入有限的教学数值。';return;}
  const yes=a+b, no=c+b, gap=yes-no, x=-gap,
        loss=Math.max(x,0)+Math.log1p(Math.exp(-Math.abs(x)));
  document.getElementById('bias').textContent=b.toFixed(1);
  el.textContent='教学示意：m_yes图 = '+yes.toFixed(4)+'；m_no图 = '+no.toFixed(4)+
    '；gap = '+gap.toFixed(4)+'；softplus(−gap) = '+loss.toFixed(6);
}
for(const id of ['a','c','b']) document.getElementById(id).addEventListener('input',updateDemo);
updateDemo();
</script></body></html>
'''
    replacements = {'__TIME__': escape(generated), '__TRAIN__': str(counts['train']),
                    '__SCREEN__': str(counts['screen']), '__CONFIRM__': str(counts['confirm']),
                    '__DATA_REVIEW_STATUS__': data_review_status(root, manifest_path),
                    '__STATUS__': runtime_status(root), '__PREDICTION_NOTE__': prediction_note,
                    '__CARDS__': cards, '__LOSS_URL__': loss_url, '__TRAIN_URL__': train_url,
                    '__ANALYSIS__': analysis}
    for token, value in replacements.items():
        page = page.replace(token, value)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding='utf-8')
    return {'status': 'rendered_derived_learning_page', 'output': str(output),
            'manifest': str(manifest_path), 'fixed_training_pair_ids': selected,
            'existing_original_image_links': len(selected) * 2, 'missing_images': 0,
            'new_model_forward': 0, 'new_training': False, 'images_copied': 0,
            'source_selection': 'fixed pair_id hash, independent of predictions'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--manifest', type=Path,
                        help='Explicit dataset manifest; defaults to config.dataset_manifest')
    args = parser.parse_args()
    print(json.dumps(render(args.study_root, args.output, args.manifest), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
