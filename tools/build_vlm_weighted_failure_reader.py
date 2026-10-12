#!/usr/bin/env python3
"""Build a CPU-only reader from frozen weighted-SFT predictions.

This tool never imports a model, invokes a scorer, or recomputes confidence
intervals. It displays the original image bytes and saved prediction fields.
"""
from __future__ import annotations

import argparse
import ast
import base64
import csv
import hashlib
import html
import io
import json
import math
import mimetypes
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


PRIVATE = Path('/opt/data/private').resolve()
REPO = Path(__file__).resolve().parents[1]
OUTPUTS = REPO.parents[1] / 'outputs'
EXPECTED = {'questions': 1617, 'raw_changed': 39, 'score_changed': 21,
            'improved': 8, 'regressed': 13, 'uniform_zero': 157,
            'uniform_partial': 166, 'uniform_full': 1294,
            'uniform_unresolved': 323}
GROUP_LABELS = {'unseen': '未见目标（0次）', 'rare': '罕见（1–5次）',
                'mid': '中频（6–20次）', 'frequent': '高频（>20次）'}
RECORD_FIELDS = ('image_grid_thw', 'visual_tokens', 'counted_image_tokens',
                 'premerge_patch_tokens', 'raw_width', 'raw_height',
                 'resized_width', 'resized_height', 'input_tokens',
                 'padded_input_tokens', 'input_exceeds_training_768',
                 'input_truncated', 'generated_nonpad_tokens',
                 'visual_token_budget', 'numerical_path', 'precision',
                 'quantization', 'input_lock_sha256')
INPUT_FIELDS = ('question_sha256', 'question_prompt_sha256', 'image_sha256',
                'image_grid_thw', 'visual_tokens', 'counted_image_tokens',
                'premerge_patch_tokens', 'raw_width', 'raw_height',
                'resized_width', 'resized_height', 'input_tokens',
                'input_truncated', 'visual_token_budget', 'numerical_path',
                'precision', 'quantization')


def private_path(value):
    path = Path(value).resolve()
    if not path.is_relative_to(PRIVATE):
        raise ValueError(f'path outside persistent workspace: {path}')
    return path


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_predictions(path):
    records = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line]
    indexed = {str(row['question_id']): row for row in records}
    if len(indexed) != len(records):
        raise ValueError(f'duplicate question ID in {path}')
    return indexed


def write_private(path, text):
    with Path(path).open('w', encoding='utf-8', newline='') as handle:
        handle.write(text)
    Path(path).chmod(0o600)


def json_text(value):
    return json.dumps(value, ensure_ascii=False, indent=2) + '\n'


def e(value):
    return html.escape(str(value), quote=True)


def extract_function(path, name):
    text = Path(path).read_text(encoding='utf-8')
    tree = ast.parse(text)
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    snippet = '\n'.join(text.splitlines()[node.lineno - 1:node.end_lineno]) + '\n'
    return {'path': str(path), 'function': name, 'line': node.lineno,
            'source_sha256': sha_file(path), 'snippet_sha256': sha_bytes(snippet.encode()),
            'snippet': snippet}


def text_candidates(before, after):
    """Surface-form hints only; never identify a visual failure mechanism."""
    flags = []
    digits = lambda value: re.findall(r'\d+(?:[.:/]\d+)*', value)
    if digits(before) != digits(after) and (digits(before) or digits(after)):
        flags.append('数字变化候选')
    left, right = before.casefold().split(), after.casefold().split()
    if len(right) > len(left) and all(Counter(right)[word] >= count for word, count in Counter(left).items()):
        flags.append('增词候选')
    if len(left) > len(right) and all(Counter(left)[word] >= count for word, count in Counter(right).items()):
        flags.append('减词候选')
    return flags


def direction(delta):
    if math.isclose(delta, 0, abs_tol=1e-12):
        return 'unchanged'
    return 'improved' if delta > 0 else 'regressed'


def build_case(question, dev_group, uniform, frequency, images, weight_by_target):
    qid = str(question['question_id'])
    if str(uniform['image_id']) != str(question['image_id']) or str(frequency['image_id']) != str(question['image_id']):
        raise ValueError(f'image alignment changed for {qid}')
    for field in INPUT_FIELDS:
        if uniform.get(field) != frequency.get(field):
            raise ValueError(f'paired input mismatch {qid}: {field}')
    image = images[str(question['image_id'])]
    if image['sha256'] != uniform['image_sha256']:
        raise ValueError(f'manifest image hash mismatch: {qid}')
    for saved in (uniform, frequency):
        if not isinstance(saved['prediction'], str):
            raise ValueError(f'non-text prediction: {qid}')
        if not math.isfinite(float(saved['score'])) or not 0 <= float(saved['score']) <= 1:
            raise ValueError(f'invalid saved score: {qid}')
        if saved['input_truncated']:
            raise ValueError(f'unexpected truncation: {qid}')
    answers = question['answers']
    if len(answers) != 10:
        raise ValueError(f'expected ten human references: {qid}')
    delta = float(frequency['score']) - float(uniform['score'])
    return {
        'question_id': qid, 'image_id': str(question['image_id']),
        'question': question['question'], 'references_raw': answers,
        'reference_raw_distribution': [{'answer': answer, 'count': count}
            for answer, count in sorted(Counter(answers).items(), key=lambda pair: (-pair[1], pair[0]))],
        'dev_reference_target': question['target'],
        'normalized_dev_target': dev_group['normalized_target'],
        'training_target_frequency': dev_group['target_frequency'],
        'frequency_bin': dev_group['frequency_bin'],
        'training_target_unique_images': dev_group['training_target_unique_images'],
        'same_target_training_weight': weight_by_target.get(dev_group['normalized_target']),
        'dev_example_training_weight': None,
        'uniform': {'prediction': uniform['prediction'], 'score': uniform['score'],
                    'saved_processing': {k: uniform[k] for k in RECORD_FIELDS}},
        'frequency': {'prediction': frequency['prediction'], 'score': frequency['score'],
                      'saved_processing': {k: frequency[k] for k in RECORD_FIELDS}},
        'saved_score_delta': delta, 'direction': direction(delta),
        'raw_answer_changed': uniform['prediction'] != frequency['prediction'],
        'text_change_candidates': text_candidates(uniform['prediction'], frequency['prediction']),
        'original_image': {'path': image['path'], 'sha256': image['sha256'],
                           'width': image['width'], 'height': image['height']},
        'manual_classification': 'pending', 'manual_evidence': '',
        'manual_classification_scope': 'human review pending; surface-form flags are not error mechanisms',
    }


def render_card(case, image_uri):
    saved = case['uniform']['saved_processing']
    fre = case['frequency']['saved_processing']
    qid = case['question_id']
    names = {'improved': '得分上升', 'regressed': '得分下降', 'unchanged': '文本改变、得分持平'}
    refs = ''.join(f'<li><code>{e(row["answer"])}</code> × {row["count"]}</li>'
                   for row in case['reference_raw_distribution'])
    flags = ' / '.join(case['text_change_candidates']) or '无自动文本候选标签'
    weight = case['same_target_training_weight']
    weight_text = f'{weight:.6f}' if weight is not None else '无：train_all中不存在同target'
    token_rows = ''.join(f'<tr><th>{e(key)}</th><td>{e(saved[key])}</td><td>{e(fre[key])}</td></tr>'
                         for key in ('image_grid_thw', 'visual_tokens', 'input_tokens',
                                     'padded_input_tokens', 'generated_nonpad_tokens',
                                     'resized_width', 'resized_height', 'input_truncated'))
    options = [('pending', '待确认'), ('reading', '文字识别'), ('question', '题意或目标定位'),
               ('number', '数字/空间关系'), ('wording', '输出措辞'), ('reference', '参考答案歧义'),
               ('other', '其他'), ('unclear', '证据不足')]
    option_html = ''.join(f'<option value="{key}">{label}</option>' for key, label in options)
    return f'''<article class="case" id="q{e(qid)}" data-qid="{e(qid)}" data-group="{e(case['frequency_bin'])}" data-direction="{e(case['direction'])}">
<h3>Q{e(qid)} · {e(GROUP_LABELS[case['frequency_bin']])} · <span class="{e(case['direction'])}">{e(names[case['direction']])}</span></h3>
<p class="question">{e(case['question'])}</p>
<div class="comparison"><figure><img loading="lazy" src="{image_uri}" alt="Q{e(qid)}原图预览"><figcaption>原文件字节嵌入；CSS缩放展示。原尺寸{case['original_image']['width']}×{case['original_image']['height']}，这是原图预览。</figcaption></figure>
<div><div class="answer"><strong>普通CE</strong><p>{e(case['uniform']['prediction'])}</p><span>保存得分：{case['uniform']['score']:.3f}</span></div>
<div class="answer"><strong>频次加权</strong><p>{e(case['frequency']['prediction'])}</p><span>保存得分：{case['frequency']['score']:.3f}；差值{case['saved_score_delta']:+.3f}</span></div>
<p>文本候选标签：{e(flags)}。这些标签只描述词面变化，不能说明错误原因。</p>
<details><summary>十名标注者的原始参考分布（共10票）</summary><ul>{refs}</ul><p>dev参考众数target：<code>{e(case['dev_reference_target'])}</code>；频次计数用归一target：<code>{e(case['normalized_dev_target'])}</code>。</p></details>
<p>train_all实际target曝光：<strong>{case['training_target_frequency']}次</strong>，覆盖{case['training_target_unique_images']}张图；同target训练归一权重：{e(weight_text)}。本dev题没有训练权重，参考仅用于生成后评分/诊断分组。</p>
<details><summary>已保存的实际processor/token记录</summary><table><thead><tr><th>记录项</th><th>普通CE</th><th>加权</th></tr></thead><tbody>{token_rows}</tbody></table><p>视觉token预算上限1024，实际值以记录为准；processor记录尺寸不代表本页重建了模型输入。原图预览未裁剪，未生成注意力图。</p></details>
<label>人工分类（此浏览器本地记录）：<select class="manual-select" data-qid="{e(qid)}">{option_html}</select></label>
<label>看图依据：<textarea class="manual-note" data-qid="{e(qid)}" rows="2" placeholder="指出具体位置、字符或参考歧义；证据不足可保留待确认"></textarea></label>
<small>image_id={e(case['image_id'])}；原图SHA256={e(case['original_image']['sha256'])}</small></div></div></article>'''


def make_html(summary, cases, unresolved, snippets, image_uris, audit):
    group_rows = ''.join(f'<tr><td>{e(GROUP_LABELS[group])}</td><td>{row["questions"]}</td><td>{row["raw_changed"]}</td><td>{row["improved"]}</td><td>{row["regressed"]}</td><td>{row["mean_delta_pp"]:+.3f}</td></tr>'
                         for group, row in summary['groups'].items())
    code_blocks = ''.join(f'<details><summary>{e(s["function"])}：{e(s["path"])}:{s["line"]}</summary><pre><code>{e(s["snippet"])}</code></pre><p>文件SHA256：<code>{s["source_sha256"]}</code><br>片段SHA256：<code>{s["snippet_sha256"]}</code></p></details>' for s in snippets)
    cards = ''.join(render_card(case, image_uris[case['image_id']]) for case in cases)
    table_rows = ''.join(f'<tr><td>{e(case["question_id"])}</td><td>{e(GROUP_LABELS[case["frequency_bin"]])}</td><td>{e(case["question"])}</td><td>{e(case["uniform"]["prediction"])}</td><td>{case["uniform"]["score"]:.3f}</td><td>pending</td></tr>' for case in unresolved)
    z = audit['weights']['normalization_constant']
    weight_rows = ''.join(f'<tr><td>{freq}</td><td>{raw:.6f}</td><td>{raw/z:.6f}</td></tr>' for freq, raw in ((1, 2), (5, 4**.25), (20, 1), (100, 1)))
    return '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>频次加权SFT：真实失败对照学习页</title><style>
body{font:16px/1.65 system-ui,sans-serif;max-width:1280px;margin:auto;padding:24px;background:#f6f7fa;color:#172233}h1,h2,h3{line-height:1.3}section,.case{background:white;border:1px solid #d6dde8;border-radius:12px;margin:20px 0;padding:20px}.banner{background:#eaf2ff;border-left:5px solid #2361bc;padding:14px}table{border-collapse:collapse;width:100%;margin:12px 0}th,td{border:1px solid #d6dde8;padding:8px;text-align:left;vertical-align:top}code{overflow-wrap:anywhere}pre{overflow:auto;max-height:560px;background:#111827;color:#edf2ff;padding:16px}.comparison{display:grid;grid-template-columns:minmax(260px,1fr) minmax(350px,1.2fr);gap:20px}figure{margin:0}img{width:100%;max-height:650px;object-fit:contain;background:#eef1f5}figcaption,small{font-size:13px;color:#536379;display:block;overflow-wrap:anywhere}.answer{border:1px solid #bac8df;border-radius:8px;padding:12px;margin-bottom:12px}.answer p{font-size:21px;font-weight:600;margin:6px 0;white-space:pre-wrap;overflow-wrap:anywhere}.question{font-size:19px;font-weight:600}.improved{color:#096d45}.regressed{color:#a42336}.unchanged{color:#5b5770}details{margin:12px 0}summary{cursor:pointer}select,input,textarea{font:inherit;padding:7px;max-width:100%}textarea{display:block;width:95%}label{display:inline-block;margin:8px 12px 8px 0}.filters{position:sticky;top:0;background:#eef3ff;border:1px solid #c7d6ef;padding:12px;z-index:2}.hidden{display:none}.rowlist{max-height:480px;overflow:auto}.steps{display:flex;gap:12px;flex-wrap:wrap}.steps div{flex:1;background:#edf3fb;padding:12px;border-radius:8px;min-width:180px}.source{font-size:13px;overflow-wrap:anywhere}@media(max-width:800px){body{padding:12px}.comparison{grid-template-columns:1fr}.filters{position:static}}
</style></head><body>''' + f'''
<h1>频次加权SFT：从真实预测理解负结果</h1>
<p class="banner">单seed42、内部dev、原权重BF16/1024推理。此页只整理已有预测；新增训练0次、GPU预测0条、重新评分0次、bootstrap 0次。它是学习与人工诊断材料，不是新实验。</p>
<section><h2>先看结果，再提出解释</h2><p>共{summary['questions']}题，普通CE {summary['uniform_mean_score_percent']:.3f}% → 频次加权 {summary['frequency_mean_score_percent']:.3f}%，差值{summary['mean_delta_pp']:+.3f}个百分点。</p>
<p>39题原始回答文本变化；其中8题得分提高、13题下降、18题得分持平。其余1578题回答文本相同。普通CE：157题零分、166题部分分、1294题满分；323题未满分，约19.98%。这些计数来自保存记录，未调用评分函数。</p>
<table><thead><tr><th>诊断分组</th><th>QA</th><th>文本变化</th><th>得分升</th><th>得分降</th><th>组均值变化pp</th></tr></thead><tbody>{group_rows}</tbody></table>
<p>rare的唯一得分下降案例为Q29766：<code>spring</code> → <code>spring water</code>；保存得分1.0→0.6。因此不能把该组−0.117pp解释为所有罕见能力下降。组置信区间以原分析为准，本页不重新计算。</p>
<p>建议先看 <a href="#q29766">29766：措辞与共识</a>、<a href="#q18020">18020</a>、<a href="#q30170">30170</a>、<a href="#q10275">10275</a>，再检查全部39条，防止只选支持某个解释的案例。</p></section>
<section><h2>连接原理、代码与实际权重</h2><div class="steps"><div>① token层：只监督assistant回答<br>prompt标签−100仍提供上下文</div><div>② 每题：回答token CE取平均<br>答案长短不自动改变题目权重</div><div>③ 全局：每题loss乘固定权重<br>按真实QA数聚合，DDP补偿world</div></div>
<p><code>r(f)=clip((20/f)^0.25, 1, 2)</code>；<code>w(f)=r(f)/Z</code>；锁定的全训练集归一常数 <strong>Z={z:.12f}</strong>。权重由32,985条train_all实际target的曝光频次计算，dev与正式集不参与权重计算。f=0没有训练样本，不定义其训练权重。</p>
<table><thead><tr><th>同target训练曝光f</th><th>原始权重r</th><th>归一训练权重w</th></tr></thead><tbody>{weight_rows}</tbody></table>
<p>高频训练样本的归一权重为0.630513，不是1；全训练集平均为1。罕见QA份额为63.741%，加权目标份额为75.730%；提高loss份额不保证生成正确率提高。这里是32,985条训练题的份额，不能把dev组当作训练组。</p>
<p>比较改动：<code>L=Σᵢ wᵢ ℓᵢ / N</code>，ℓᵢ是每题assistant token平均CE。该实现每个rank用<code>world/actual</code>缩放局部和，补偿DDP的rank梯度平均；分母不是当前batch权重之和。零权重补位样本不算真实QA。TextVQA则对十名标注者逐一留出并作共识计分；措辞改变可能改变参考匹配分，但该现象仍需逐题确认。</p>
{code_blocks}
<h3>每个案例都按三个问题看</h3><ol><li>图上有没有足够证据？能指出位置或字符吗？原图可读不自动说明processor后的模型能读。</li><li>两组回答发生了什么词面/得分变化？十人参考是否一致？</li><li>什么证据能区分“读取错误、定位错误、输出措辞、参考歧义”？没有证据时保留待确认，不猜注意力。</li></ol></section>
<h2>39条完整回答变化对照</h2><div class="filters"><label>频次分组 <select id="group"><option value="all">全部</option>{''.join(f'<option value="{key}">{e(value)}</option>' for key,value in GROUP_LABELS.items())}</select></label><label>得分方向 <select id="direction"><option value="all">全部</option><option value="improved">上升</option><option value="regressed">下降</option><option value="unchanged">持平</option></select></label><label>题号/文字 <input id="search" type="search" placeholder="29766 / spring"></label><span id="visible"></span><p>人工选择默认待确认，只存于此浏览器localStorage；不会写回源预测、CSV或任何评分结果。部分浏览器禁用file页localStorage，此时仍可浏览。</p></div>
{cards}
<section><h2>普通CE的323条未满分题：待人工审查清单</h2><p>所有人工机制字段均为pending。CSV包含问题、两回答、参考分布、图像路径与SHA、token记录，适合下一轮按证据人工分类。39条差异只是loss改动对照，不代表全部未解决问题。</p><div class="rowlist"><table><thead><tr><th>QID</th><th>组</th><th>问题</th><th>普通CE回答</th><th>分数</th><th>分类</th></tr></thead><tbody>{table_rows}</tbody></table></div></section>
<section><h2>可复核来源与限制</h2><p class="source">输入来源：{e(summary['source_root'])}<br>源manifest：{e(summary['manifest_path'])}<br>详细SHA与文件清单位于同目录PROVENANCE.json、SUMMARY.json、cases.json。</p><p>预览直接嵌入原图文件字节，并核对manifest与预测的原图SHA。没有裁剪、重采样或重建processor输入，没有注意力/模型热图。每题均保留实际视觉token、输入长度与processor尺寸记录。频次是回答文本监督曝光代理，不是语义类别或模型预训练知识。保存得分不等于图像读取能力的唯一测量。</p><p>下一步先人工确认机制，再用训练数据建立独立可复核诊断；本页不选新权重、不选checkpoint、不启动额外训练，也不改简历。</p></section>
<script>
const cards=[...document.querySelectorAll('.case')];
function filterCases(){{const g=document.getElementById('group').value,d=document.getElementById('direction').value,s=document.getElementById('search').value.toLowerCase();let n=0;for(const c of cards){{const show=(g==='all'||c.dataset.group===g)&&(d==='all'||c.dataset.direction===d)&&c.textContent.toLowerCase().includes(s);c.classList.toggle('hidden',!show);if(show)n++;}}document.getElementById('visible').textContent='显示 '+n+' / '+cards.length+' 题';}}
for(const id of ['group','direction','search'])document.getElementById(id).addEventListener('input',filterCases);filterCases();
for(const input of document.querySelectorAll('.manual-select,.manual-note')){{const key='rarevlm-weighted-failure-v1:'+input.dataset.qid+':'+(input.classList.contains('manual-select')?'class':'note');try{{const v=localStorage.getItem(key);if(v!==null)input.value=v;}}catch(e){{}}input.addEventListener('input',()=>{{try{{localStorage.setItem(key,input.value);}}catch(e){{}}}});}}
</script></body></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', default=str(OUTPUTS / 'vlm_weighted_sft_v1'))
    parser.add_argument('--manifest', default=str(OUTPUTS / 'vlm_deep_v2/prepared/manifest.json'))
    parser.add_argument('--output', default=str(OUTPUTS / 'vlm_weighted_failure_review_v1'))
    args = parser.parse_args()
    source_root, manifest_path, output = map(private_path, (args.source_root, args.manifest, args.output))
    if output == source_root or output.is_relative_to(source_root):
        raise ValueError('new reader cannot overwrite the frozen experiment directory')
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'output is not empty; refusing overwrite: {output}')
    os.umask(0o077)
    manifest = read_json(manifest_path)
    weight_path, audit_path = source_root / 'WEIGHT_LOCK.json', source_root / 'AUDIT.json'
    weight_lock, audit = read_json(weight_path), read_json(audit_path)
    if not audit['complete']:
        raise ValueError('source annotation audit incomplete')
    if audit['weight_lock_file_sha256'] != sha_file(weight_path):
        raise ValueError('source weight file changed')
    if weight_lock['manifest_file_sha256'] != sha_file(manifest_path):
        raise ValueError('source manifest bytes changed')
    if audit['lock_sha256'] != weight_lock['lock_sha256']:
        raise ValueError('audit/weight lock identity mismatch')
    uniform_path = source_root / 'runs/uniform/seed42/dev/predictions.jsonl'
    frequency_path = source_root / 'runs/frequency/seed42/dev/predictions.jsonl'
    uniform, frequency = read_predictions(uniform_path), read_predictions(frequency_path)
    rows = manifest['splits']['dev']
    dev_groups = {str(row['question_id']): row for row in weight_lock['dev_rows']}
    qids = {str(row['question_id']) for row in rows}
    if not len(qids) == len(rows) == EXPECTED['questions'] or any(set(mapping) != qids for mapping in (uniform, frequency, dev_groups)):
        raise ValueError('full dev question coverage is not exactly aligned')
    weight_by_target = {}
    for row in weight_lock['rows']:
        target = row['normalized_target']
        if target in weight_by_target and weight_by_target[target] != row['weight']:
            raise ValueError('inconsistent same-target training weights')
        weight_by_target[target] = row['weight']
    aligned = [build_case(row, dev_groups[str(row['question_id'])], uniform[str(row['question_id'])],
                          frequency[str(row['question_id'])], manifest['images'], weight_by_target)
               for row in rows]
    changed = [case for case in aligned if case['raw_answer_changed']]
    unresolved = [case for case in aligned if float(case['uniform']['score']) < 1]
    counts = {'questions': len(aligned), 'raw_changed': len(changed),
              'score_changed': sum(case['direction'] != 'unchanged' for case in aligned),
              'improved': sum(case['direction'] == 'improved' for case in aligned),
              'regressed': sum(case['direction'] == 'regressed' for case in aligned),
              'uniform_zero': sum(case['uniform']['score'] == 0 for case in aligned),
              'uniform_partial': sum(0 < case['uniform']['score'] < 1 for case in aligned),
              'uniform_full': sum(case['uniform']['score'] == 1 for case in aligned),
              'uniform_unresolved': len(unresolved)}
    if counts != EXPECTED:
        raise ValueError(f'saved-record counts differ from declared reader contract: {counts}')
    if any(case['direction'] != 'unchanged' and not case['raw_answer_changed'] for case in aligned):
        raise ValueError('equal raw answer has differing saved score')
    group_cases = defaultdict(list)
    for case in aligned:
        group_cases[case['frequency_bin']].append(case)
    groups = {}
    for key in GROUP_LABELS:
        current = group_cases[key]
        groups[key] = {'questions': len(current),
                       'raw_changed': sum(case['raw_answer_changed'] for case in current),
                       'improved': sum(case['direction'] == 'improved' for case in current),
                       'regressed': sum(case['direction'] == 'regressed' for case in current),
                       'mean_delta_pp': math.fsum(case['saved_score_delta'] for case in current) / len(current) * 100}
    priority = ['29766', '18020', '30170', '10275']
    changed.sort(key=lambda case: (priority.index(case['question_id']) if case['question_id'] in priority else len(priority), int(case['question_id'])))
    if not set(priority) <= {case['question_id'] for case in changed}:
        raise ValueError('a highlighted saved comparison is absent')
    snippets = [extract_function(REPO / 'vlm_weighted/train.py', 'weighted_objective'),
                extract_function(REPO / 'vlm_deep/common.py', 'answer_loss'),
                extract_function(REPO / 'vlm_deep/common.py', 'consensus')]
    source_paths = [manifest_path, weight_path, audit_path, uniform_path, frequency_path,
                    source_root / 'analysis/ANALYSIS.json', Path(__file__).resolve()]
    source_paths.extend(Path(snippet['path']) for snippet in snippets)
    source_hashes = {str(path): sha_file(path) for path in dict.fromkeys(source_paths)}
    image_uris, image_hashes = {}, {}
    for case in changed:
        image_id = case['image_id']
        if image_id in image_uris:
            continue
        image_path = private_path(manifest_path.parent / case['original_image']['path'])
        data = image_path.read_bytes()
        digest = sha_bytes(data)
        if digest != case['original_image']['sha256']:
            raise ValueError(f'original image bytes changed: {image_id}')
        mime = mimetypes.guess_type(image_path.name)[0]
        if mime not in ('image/jpeg', 'image/png', 'image/webp'):
            raise ValueError(f'unsupported original image MIME: {mime}')
        image_uris[image_id] = f'data:{mime};base64,' + base64.b64encode(data).decode('ascii')
        image_hashes[image_id] = {'path': str(image_path), 'sha256': digest, 'byte_count': len(data)}
    summary = {'schema_version': 'rarevlm.weighted_failure_reader.v1',
               'complete': True, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
               'source_root': str(source_root), 'manifest_path': str(manifest_path),
               **counts, 'raw_unchanged': len(aligned) - len(changed),
               'changed_text_score_unchanged': len(changed) - counts['score_changed'],
               'uniform_mean_score_percent': math.fsum(case['uniform']['score'] for case in aligned) / len(aligned) * 100,
               'frequency_mean_score_percent': math.fsum(case['frequency']['score'] for case in aligned) / len(aligned) * 100,
               'mean_delta_pp': math.fsum(case['saved_score_delta'] for case in aligned) / len(aligned) * 100,
               'groups': groups, 'new_training_runs': 0, 'new_gpu_predictions': 0,
               'scorer_calls': 0, 'bootstrap_calls': 0, 'human_classification_completed': 0,
               'source_hashes': source_hashes, 'displayed_unique_images': len(image_uris),
               'checks': {'full_dev_question_coverage': True, 'paired_recorded_input_alignment': True,
                          'ten_references_each': True, 'saved_score_range_valid': True,
                          'declared_counts_match': True, 'displayed_original_image_sha_verified': True,
                          'weight_and_manifest_file_locks_match': True},
               'limits': ['one fixed seed42 checkpoint pair, internal dev only',
                          'scores copied from frozen saved predictions; no re-scoring',
                          'text-change candidates are not visual error-mechanism labels',
                          'dev frequency is diagnostic; dev examples are not assigned training weights',
                          'original image preview is not a reconstructed model processor input'],
               'artifacts': ['SUMMARY.json', 'PROVENANCE.json', 'cases.json',
                             'unresolved_uniform.csv', '学习页.html', 'README.md']}
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    output.chmod(0o700)
    write_private(output / 'cases.json', json_text(changed))
    fields = ['question_id', 'image_id', 'frequency_bin', 'training_target_frequency',
              'question', 'uniform_prediction', 'uniform_score', 'frequency_prediction',
              'frequency_score', 'saved_score_delta', 'reference_raw_distribution',
              'original_image_path', 'original_image_sha256', 'raw_width', 'raw_height',
              'visual_tokens', 'input_tokens', 'processor_resized_width',
              'processor_resized_height', 'manual_classification', 'manual_evidence',
              'recommended_action']
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for case in unresolved:
        saved = case['uniform']['saved_processing']
        writer.writerow({'question_id': case['question_id'], 'image_id': case['image_id'],
                         'frequency_bin': case['frequency_bin'],
                         'training_target_frequency': case['training_target_frequency'],
                         'question': case['question'],
                         'uniform_prediction': case['uniform']['prediction'],
                         'uniform_score': case['uniform']['score'],
                         'frequency_prediction': case['frequency']['prediction'],
                         'frequency_score': case['frequency']['score'],
                         'saved_score_delta': case['saved_score_delta'],
                         'reference_raw_distribution': json.dumps(case['reference_raw_distribution'], ensure_ascii=False),
                         'original_image_path': str(manifest_path.parent / case['original_image']['path']),
                         'original_image_sha256': case['original_image']['sha256'],
                         'raw_width': saved['raw_width'], 'raw_height': saved['raw_height'],
                         'visual_tokens': saved['visual_tokens'], 'input_tokens': saved['input_tokens'],
                         'processor_resized_width': saved['resized_width'],
                         'processor_resized_height': saved['resized_height'],
                         'manual_classification': 'pending', 'manual_evidence': '',
                         'recommended_action': 'pending'})
    write_private(output / 'unresolved_uniform.csv', stream.getvalue())
    write_private(output / '学习页.html', make_html(summary, changed, unresolved, snippets, image_uris, audit))
    write_private(output / 'README.md', '# 频次加权SFT失败对照学习包\n\n'
                  '打开 `学习页.html`，先看Q29766，再检查全部39条对照。页面无需网络。\n\n'
                  '`cases.json` 包含39条原始回答变化；`unresolved_uniform.csv` 包含普通CE全部323条未满分题。'
                  '人工机制字段均待确认。此工具只汇总保存记录，未训练、推理、重新评分或bootstrap。\n\n'
                  '原图按原文件字节嵌入并核对SHA；模型processor记录另列，不能把预览当模型输入或注意力。'
                  '浏览器人工标注只存在localStorage，不会修改源预测或输出CSV。\n')
    artifact_hashes = {path.name: sha_file(path) for path in output.iterdir() if path.is_file()}
    provenance = {'schema_version': 'rarevlm.weighted_failure_provenance.v1',
                  'source_hashes': source_hashes, 'original_images': image_hashes,
                  'frozen_function_snippets': snippets, 'artifact_hashes': artifact_hashes,
                  'image_transformation': 'none; original bytes embedded; CSS display scaling only',
                  'rendering_safety': 'all dataset text HTML escaped; no dataset text interpolated into JavaScript',
                  'source_mutation': 'none', 'scoring': 'saved values only'}
    write_private(output / 'PROVENANCE.json', json_text(provenance))
    write_private(output / 'SUMMARY.json', json_text(summary))
    print(json.dumps({'complete': True, 'output': str(output), 'counts': counts,
                      'html_bytes': (output / '学习页.html').stat().st_size,
                      'displayed_unique_images': len(image_uris)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
