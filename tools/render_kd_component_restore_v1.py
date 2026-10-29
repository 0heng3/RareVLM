"""Render the completed factorial intervention and its genuine answer changes."""
import html
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps
from rarevlm_paths import PROJECT_ROOT
from vlm_deep.common import write_json

BASE = PROJECT_ROOT.parents[1] / 'outputs'
ROOT = BASE / 'vlm_kd_component_restore_v1'
OUT = ROOT / 'result_review_v1'
FONT = '/opt/data/private/.tools/fonts/NotoSansCJKsc-Regular.otf'
ROLES = ('start', 'language_only', 'visual_only', 'kd')
LABELS = {'start': '00 强SFT起点', 'language_only': '10 仅语言新增更新',
          'visual_only': '01 仅视觉新增更新', 'kd': '11 KD两侧更新'}


def main():
    read = lambda p: json.loads(p.read_text())
    result = read(ROOT / 'ANALYSIS.json'); data = read(ROOT / 'data/MANIFEST.json')
    records = {role: {q['question_id']: q for q in read(ROOT / ('reused' if role in ('start', 'kd') else 'runs') / role / 'transfer_dev/PREDICTIONS.json')} for role in ROLES}
    counts = {}
    for role in ('language_only', 'visual_only'):
        counts[role] = {'score_improved_vs_kd': sum(q['vqa_score'] > records['kd'][qid]['vqa_score'] + 1e-9 for qid, q in records[role].items()),
            'score_regressed_vs_kd': sum(q['vqa_score'] < records['kd'][qid]['vqa_score'] - 1e-9 for qid, q in records[role].items())}
    meta = {q['question_id']: q for q in data['evaluations']['transfer_dev']}
    cases = []
    for qid in ('6107', '23696'):
        q = meta[qid]
        cases.append({'question_id': qid, 'question': q['question'], 'target': q['target'], 'answers': q['answers'],
            'image': data['images'][q['image_id']], 'outputs': {r: records[r][qid] for r in ROLES},
            'selection': 'Posthoc illustrative paired intervention case; no label re-adjudication or representativeness claim.'})
    write_json(OUT / 'CASE_REVIEW.json', {'status': 'completed', 'counts': counts, 'cases': cases, 'new_model_calls': 0})
    write_json(PROJECT_ROOT / 'results/vlm_kd_component_restore_v1_metrics.json', result)
    canvas = Image.new('RGB', (1640, 1450), '#f4f6f5'); draw = ImageDraw.Draw(canvas); fonts = {}
    def text(x, y, value, size=22, color='#172a3d'):
        font = fonts.setdefault(size, ImageFont.truetype(FONT, size)); draw.text((x, y), str(value), font=font, fill=color)
    def wrap(x, y, value, width, size=20, lines=3):
        font = fonts.setdefault(size, ImageFont.truetype(FONT, size)); chunks = []; line = ''
        for char in str(value):
            if draw.textlength(line + char, font=font) > width: chunks.append(line); line = char
            else: line += char
        if line: chunks.append(line)
        for i, line in enumerate(chunks[:lines]): text(x, y + i * (size + 7), line, size)
    text(30, 23, '恢复哪一侧参数能挽回退化？语言 / 视觉 2×2 消融', 34)
    text(30, 81, '3618次新生成，0训练更新；00/11结果复用。两worker exit0、0争用，约23.3分钟完成。', 22)
    text(30, 123, '只恢复指定侧全部A/B因子到强SFT起点；不是关闭整个模块，也不是独立scope重训。', 21, '#607080')
    for i, role in enumerate(ROLES):
        x = 30 + i * 400; m = result['metrics'][role]
        draw.rounded_rectangle((x, 187, x + 380, 465), radius=12, fill='white')
        text(x + 16, 202, LABELS[role], 23)
        for j, (key, label) in enumerate((('transfer_dev', '完整任务1617'), ('full_rare342', '完整rare342'), ('native_hard', '训练拟合128'), ('retention', '保持64'))):
            text(x + 16, 258 + j * 46, label, 20)
            text(x + 215, 258 + j * 46, f'{m[key]:.3f}%', 24)
    restore = result['contrasts']['restore_language_01_minus_11']['transfer_dev']
    text(30, 503, f"恢复语言：01−11 = {restore['delta_pp']:+.3f}pp，五比较99%区间 [+0.118, +2.020]。", 25, '#127467')
    text(30, 550, '恢复视觉：10−11 = +0.557pp，95%为正，但99%区间 [−0.074, +1.218] 跨0。', 23)
    text(30, 594, '交互项 −0.439pp，99% [−1.216, +0.304] 跨0；不能断言负交互已证实。', 22)
    text(30, 639, '所有组合仍低于88.027%的起点；01保持64仍下降1.563pp，不作为新能力候选。', 23, '#b24645')
    for i, case in enumerate(cases):
        x = 30 + i * 800
        draw.rounded_rectangle((x, 710, x + 780, 1320), radius=12, fill='white')
        text(x + 18, 724, ('语言恢复纠正本例' if i == 0 else '视觉恢复纠正本例') + ' · q' + case['question_id'], 25)
        with Image.open(case['image']['path']) as raw: picture = ImageOps.contain(raw.convert('RGB'), (355, 255))
        canvas.paste(picture, (x + 20 + (355 - picture.width) // 2, 785 + (255 - picture.height) // 2))
        wrap(x + 400, 792, case['question'], 350, 20, 5)
        wrap(x + 400, 960, '原target: ' + case['target'], 350, 21, 2)
        for j, role in enumerate(ROLES):
            q = case['outputs'][role]
            text(x + 20, 1070 + j * 45, f"{LABELS[role]}: {q['prediction']}  | VQA={q['vqa_score']:.1f}", 21)
        text(x + 20, 1267, '同一图像/问题/解码；事后示例，未重新裁定GT。', 19, '#607080')
    text(30, 1352, '已验证：在KD视觉状态下，恢复语言改善固定模型输出。未验证：唯一病因、数据/LR因果、新学生提升。', 21)
    text(30, 1394, '区间仅覆盖固定单seed模型的内部图像簇变化；99%是五主比较校正，不是全历史或外部确认。', 20, '#607080')
    canvas.save(OUT / '组件恢复_结果与真实案例.png')
    articles = []
    for case in cases:
        image_url = '../../' + str(Path(case['image']['path']).relative_to(BASE))
        lines = '\n'.join(f"{LABELS[r]}: {case['outputs'][r]['prediction']} | VQA={case['outputs'][r]['vqa_score']:.1f}" for r in ROLES)
        articles.append('<article><h2>q' + case['question_id'] + '</h2><img src="' + html.escape(image_url) + '"><p>' + html.escape(case['question']) + '</p><p>原target: ' + html.escape(case['target']) + '</p><pre>' + html.escape(lines) + '</pre><p>这些例子说明两个组件都可能影响具体回答，不能用单例推出总体因果。</p></article>')
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>组件恢复消融学习</title><style>body{max-width:1400px;margin:30px auto;font:18px/1.7 system-ui;background:#f4f6f5;color:#172a3d}article,section{background:white;padding:24px;margin:22px 0}img{max-width:100%;max-height:600px}pre{white-space:pre-wrap}</style><h1>从共同退化定位到组件条件效应</h1><section><p>LoRA使用W_eff = W_base + scaling·B·A。本实验保持W_base和scaling不变，把一侧的A和B同时换回强SFT起点，另一侧保留KD值。语言侧不是零，视觉侧也不是零：保留的是此前已有能力。</p><p>令四端点分数为S00、S10、S01、S11；恢复语言的条件效应为S01−S11，恢复视觉为S10−S11；交互为S11−S10−S01+S00。参数来自联合训练，混合端点没有重新协同训练，所以不能当成独立scope算法实验或新能力候选。</p><p><a href="../../../packages/RareVLM_minimal_20261001/tools/run_kd_component_restore_v1.py">真实干预代码mixed_factors/worker</a> · <a href="../../../packages/RareVLM_minimal_20261001/docs/VLM_KD_COMPONENT_RESTORE_V1_RESULTS.md">完整结果和边界</a> · <a href="../../../packages/RareVLM_minimal_20261001/docs/VLM_FULL_TEACHER_TRANSFER_V1_PROTOCOL.md">下一步完整教师验证</a></p><pre>' + html.escape(json.dumps(result, ensure_ascii=False, indent=2)) + '</pre></section>' + ''.join(articles) + '</html>'
    (OUT / '组件恢复_公式源码与案例.html').write_text(page)
    print(json.dumps({'status': 'rendered', 'figure': str(OUT / '组件恢复_结果与真实案例.png'), 'counts': counts, 'new_model_calls': 0}, ensure_ascii=False))


if __name__ == '__main__': main()
