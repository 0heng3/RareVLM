"""Read saved scores only; display final native-scope results and original images."""
import collections
import datetime
import html
import json
import os
from pathlib import Path
import statistics
import textwrap

from rarevlm_paths import PROJECT_ROOT
from vlm_deep.common import sha_file, write_json

BASE = PROJECT_ROOT.parents[1] / 'outputs'
PARENT = BASE / 'vlm_native_scope_v1'
ROOT = BASE / 'vlm_native_scope_readout_v1'
read = lambda p: json.loads(Path(p).read_text())


def main():
    manifest = read(PARENT / 'data/MANIFEST.json')
    rows = {q['question_id']: q for q in manifest['evaluations']['transfer_dev']}
    paths = {'start': PARENT / 'baseline/transfer_dev/PREDICTIONS.json', **{
        role: PARENT / f'runs/{role}/transfer/transfer_dev/PREDICTIONS.json' for role in ('language', 'visual')}}
    predictions = {role: {q['question_id']: q for q in read(path)} for role, path in paths.items()}
    start = predictions['start']; report = {'status': 'completed', 'observed_at': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        'scope': 'Posthoc descriptive 1617 internal review; original scores/labels retained. No new inference, scoring, bootstrap or training.',
        'files_sha256': {str(p): sha_file(p) for p in [*paths.values(), PARENT / 'data/MANIFEST.json', PARENT / 'TRANSFER_ANALYSIS.json']}, 'roles': {}}
    for role in ('language', 'visual'):
        after = predictions[role]
        if set(after) != set(rows) or set(start) != set(rows):
            raise ValueError('Incomplete saved predictions')
        if any(after[q]['input_sha256'] != start[q]['input_sha256'] for q in rows):
            raise ValueError('Actual prefixes differ')
        changes = [{**rows[q], 'before': start[q]['prediction'], 'after': after[q]['prediction'],
                    'scores': [start[q]['vqa_score'], after[q]['vqa_score']], 'delta': after[q]['vqa_score'] - start[q]['vqa_score']}
                   for q in rows if abs(after[q]['vqa_score'] - start[q]['vqa_score']) > 1e-9]
        strata = {}
        for field in ('stratum', 'frequency_slice'):
            groups = collections.defaultdict(list)
            for q, row in rows.items():
                groups[row[field]].append(q)
            strata[field] = {key: {'n': len(ids), 'delta_pp': 100 * statistics.mean(after[q]['vqa_score'] - start[q]['vqa_score'] for q in ids)} for key, ids in groups.items()}
        report['roles'][role] = {'answer_string_changes': sum(after[q]['prediction'] != start[q]['prediction'] for q in rows),
            'score_gains': sum(q['delta'] > 0 for q in changes), 'score_losses': sum(q['delta'] < 0 for q in changes),
            'score_sum_gained': sum(q['delta'] for q in changes if q['delta'] > 0), 'score_sum_lost': -sum(q['delta'] for q in changes if q['delta'] < 0),
            'score_sum_net': sum(q['delta'] for q in changes), 'strata': strata, 'changes': changes}
    write_json(ROOT / 'TRANSFER_REVIEW.json', report)
    selected = ['23696', '24857', '25452']
    notes = {'23696': '原图前方蓝衣跑者号码235可见；从另一跑者202转为目标号码。单例同时涉及区域选择与读数，不证明独立新能力。',
             '24857': '牌上可见Drive-Thru Service Only；视觉Final漏掉Only。确有文本遗漏，不仅是增加合理别名后的评分差。',
             '25452': '车身可见34 T 532；视觉Final漏掉T。该个案支持字符遗漏，不能据此推断全部退化的内部原因。'}
    cases = {q['question_id']: q for q in report['roles']['visual']['changes']}
    analysis = read(PARENT / 'TRANSFER_ANALYSIS.json')
    e = lambda x: html.escape(str(x))
    intro = '<h1>1617题完整验证：初筛收益未延续</h1><p>共同起点87.198516%，语言86.295609%，视觉87.266543%。视觉相对起点+0.068027pp，95%[−0.453437,+0.599549]；未达+0.5pp与正区间门槛。视觉比退化的语言控制好，不等于提升起点能力。</p><p>原512题初筛视觉+0.78125pp仍保留；完整1617视觉18题升、15题降，净增1.1题等效满分得分。单词+0.374564pp、多词−0.682303pp是事后描述，无新增切片区间。历史内部cohort、单seed，非外部盲测。</p>'
    cards = []
    for qid in selected:
        q = cases[qid]; path = Path(manifest['images'][q['image_id']]['path'])
        cards.append(f'<article><h3>QA {e(qid)}</h3><img src="{e(os.path.relpath(path, ROOT))}"><p>{e(q["question"])}</p><p>起点：{e(q["before"])}（{q["scores"][0]:.1f}）<br>视觉：{e(q["after"])}（{q["scores"][1]:.1f}）<br>原target：{e(q["target"])}</p><p>{e(notes[qid])}</p></article>')
    table = ''.join(f'<tr><td>{e(role)}</td><td>{e(q["question_id"])}</td><td>{e(q["question"])}</td><td>{e(q["before"])}</td><td>{e(q["after"])}</td><td>{q["scores"][0]:.2f}→{q["scores"][1]:.2f}</td></tr>' for role, rec in report['roles'].items() for q in rec['changes'])
    learning = '<h2>为什么下一步先测答案读出？</h2><p>自回归概率 P(y|图,问)=∏ₜP(yₜ|图,问,y₍＜ₜ₎)。SFT提高GT-prefix下的token概率；完整正确句的总概率较高，不保证每一步greedy都选对，也不保证正确EOS位置。DPO通过相对reference的logP(正)−logP(负)差训练偏好；如果错误是局部token选择、而已有静态负例排序已过关，不能仅靠换算法名称预期收益。</p><p>新诊断仅用128条原训练纠错，比较chosen/旧negative/当前实际回答的总分与均值、局部top1、正文及EOS，不将上面验证案例回填训练。<a href="COMPLETION.json">实际诊断结果</a> · <a href="PLAN.json">冻结诊断输入</a> · <a href="TRANSFER_REVIEW.json">全部1617升降</a></p>'
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>RareVLM完整结果与真实案例</title><style>body{font:16px/1.8 system-ui;max-width:1440px;margin:30px auto;padding:20px;background:#f4f6f8;color:#142e40}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}article{background:white;padding:16px}img{width:100%;height:350px;object-fit:contain}table{border-collapse:collapse;font-size:14px}td,th{padding:8px;border:1px solid #cad5dc}details{overflow:auto}@media(max-width:800px){.cards{grid-template-columns:1fr}}</style>' + intro + '<div class="cards">' + ''.join(cards) + '</div><p>图片为原图的等比缩放展示，未裁剪/标注为模型注意力；助手非专家事后复核，不修改原GT或评分。</p>' + learning + '<details><summary>展开完整得分变化（language与visual均列出，忽略≤1e-9浮点差）</summary><table><tr><th>方案</th><th>QA</th><th>问题</th><th>起点</th><th>Final</th><th>原得分</th></tr>' + table + '</table></details></html>'
    (ROOT / '结果与原图.html').write_text(page)
    from PIL import Image, ImageDraw, ImageFont, ImageOps
    fontpath = '/opt/data/private/.tools/fonts/NotoSansCJKsc-Regular.otf'
    font = ImageFont.truetype(fontpath, 21); small = ImageFont.truetype(fontpath, 18); title = ImageFont.truetype(fontpath, 30)
    canvas = Image.new('RGB', (1600, 970), '#f4f6f8'); draw = ImageDraw.Draw(canvas)
    draw.text((28, 20), '1617题：视觉适配 +0.068pp，95%区间跨0；未确认能力提升', fill='#142e40', font=title)
    draw.text((28, 70), '18题增益 / 15题退化；以下为原图缩放展示与真实保存回答，不是模型注意力。', fill='#475569', font=font)
    for column, qid in enumerate(selected):
        q = cases[qid]; x = 20 + 530 * column; y = 128
        draw.rounded_rectangle((x, y, x + 510, 925), radius=10, fill='white')
        draw.text((x + 14, y + 10), f'QA {qid}   {q["scores"][0]:.1f} → {q["scores"][1]:.1f}', fill='#087f5b' if q['delta'] > 0 else '#b02a37', font=font)
        im = ImageOps.contain(Image.open(manifest['images'][q['image_id']]['path']).convert('RGB'), (480, 400))
        canvas.paste(im, (x + (510 - im.width) // 2, y + 55 + (400 - im.height) // 2)); ty = y + 470
        lines = textwrap.wrap(q['question'], width=46)
        lines += ['', '起点：' + q['before'], '视觉：' + q['after'], 'target：' + q['target'], '']
        lines += [notes[qid][i:i + 23] for i in range(0, len(notes[qid]), 23)]
        for line in lines:
            draw.text((x + 14, ty), line, fill='#142e40', font=small); ty += 26
    draw.text((28, 940), '来源：原1617预测；单seed内部事后分析。保留原标注与评分，未进行新增推理或重评分。', fill='#475569', font=small)
    canvas.save(ROOT / '完整验证与真实升降.png')
    print(json.dumps({'report': str(ROOT / 'TRANSFER_REVIEW.json'), 'roles': {r: {k:v for k,v in q.items() if k not in ('changes', 'strata')} for r,q in report['roles'].items()}}, ensure_ascii=False))


if __name__ == '__main__':
    main()
