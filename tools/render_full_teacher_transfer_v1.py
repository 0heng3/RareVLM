"""Render frozen teacher/student capability evidence, not a student gain claim."""
import html
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps
from rarevlm_paths import PROJECT_ROOT
from vlm_deep.common import write_json

BASE = PROJECT_ROOT.parents[1] / 'outputs'
ROOT = BASE / 'vlm_full_teacher_transfer_v1'
OUT = ROOT / 'result_review_v1'
FONT = '/opt/data/private/.tools/fonts/NotoSansCJKsc-Regular.otf'


def main():
    read = lambda p: json.loads(p.read_text())
    result = read(ROOT / 'ANALYSIS.json'); data = read(ROOT / 'DATA.json')
    a = read(ROOT / 'baseline/transfer_dev/PREDICTIONS.json'); b = read(ROOT / 'teacher/transfer_dev/PREDICTIONS.json')
    if [(q['question_id'], q['input_sha256']) for q in a] != [(q['question_id'], q['input_sha256']) for q in b]:
        raise ValueError('Saved paired input identity differs')
    pairs = list(zip(a, b)); meta = {q['question_id']: q for q in data['evaluations']['transfer_dev']}
    gains = sorted([(s, t) for s, t in pairs if s['vqa_score'] < .1 and t['vqa_score'] > .9], key=lambda pair: int(pair[0]['question_id']))
    losses = sorted([(s, t) for s, t in pairs if s['vqa_score'] > .9 and t['vqa_score'] < .1], key=lambda pair: int(pair[0]['question_id']))
    examples = gains[:2] + losses[:1]
    cases = []
    for student, teacher in examples:
        row = meta[student['question_id']]
        cases.append({'question_id': row['question_id'], 'question': row['question'], 'target': row['target'],
            'answers': row['answers'], 'image': data['images'][row['image_id']], 'student': student, 'teacher': teacher,
            'selection': 'Posthoc first two full corrections and first full regression by numeric qid; not representative or label adjudication.'})
    changes = {'score_improved': sum(t['vqa_score'] > s['vqa_score'] + 1e-9 for s, t in pairs),
        'score_regressed': sum(t['vqa_score'] < s['vqa_score'] - 1e-9 for s, t in pairs),
        'score_unchanged': sum(abs(t['vqa_score'] - s['vqa_score']) <= 1e-9 for s, t in pairs),
        'score_gained': sum(max(0., t['vqa_score'] - s['vqa_score']) for s, t in pairs),
        'score_lost': sum(max(0., s['vqa_score'] - t['vqa_score']) for s, t in pairs)}
    write_json(OUT / 'CASE_REVIEW.json', {'status': 'completed', 'changes': changes, 'cases': cases, 'new_model_calls': 0})
    write_json(PROJECT_ROOT / 'results/vlm_full_teacher_transfer_v1_metrics.json', result)
    canvas = Image.new('RGB', (1660, 1380), '#f4f6f5'); draw = ImageDraw.Draw(canvas); fonts = {}
    def text(x, y, value, size=22, color='#172a3d'):
        font = fonts.setdefault(size, ImageFont.truetype(FONT, size)); draw.text((x, y), str(value), font=font, fill=color)
    def wrap(x, y, value, width, size=18, lines=3):
        font = fonts.setdefault(size, ImageFont.truetype(FONT, size)); parts = []; line = ''
        for char in str(value):
            if draw.textlength(line + char, font=font) > width: parts.append(line); line = char
            else: line += char
        if line: parts.append(line)
        for i, part in enumerate(parts[:lines]): text(x, y + i * (size + 7), part, size)
    text(30, 22, '7B教师在完整任务更强：蒸馏有机会，学生尚未提分', 34)
    text(30, 82, '1681次新生成，四worker exit0、0争用；约21分48秒结束，0更新/0GT前向。', 23)
    text(30, 127, '固定内部任务、相同问题/图像/greedy32/评分；两模型训练状态也不同，不作纯规模因果。', 21, '#607080')
    draw.rounded_rectangle((30, 180, 1630, 465), radius=12, fill='white')
    for x, label in ((50, '指标'), (470, '强3B SFT学生'), (800, '完整7B教师'), (1130, '教师−学生，95%图像簇区间')): text(x, 202, label, 21)
    for i, (key, label) in enumerate((('transfer_dev', '完整1617 VQA%'), ('full_rare342', '完整rare342 VQA%'), ('retention', '保持64 VQA%'))):
        y = 266 + i * 62; c = result['teacher_vs_strong_student'][key]
        text(50, y, label, 23)
        text(470, y, f"{result['metrics']['start'][key]:.3f}%", 25)
        text(800, y, f"{result['metrics']['teacher'][key]:.3f}%", 25, '#127467')
        text(1130, y, f"{c['delta_pp']:+.3f}pp [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]", 21)
    text(30, 503, '完整任务145题涨分、37题降分；长尾27涨8降，保持2涨0降，全部生成正常终止。', 24)
    text(30, 552, '证实的是冻结教师的任务优势；旧KD仍为86.846%，不能把教师93.537%写作学生训练收益。', 22, '#b24645')
    for i, case in enumerate(cases):
        x = 30 + i * 540
        draw.rounded_rectangle((x, 620, x + 520, 1255), radius=12, fill='white')
        text(x + 17, 635, ('教师纠正' if i < 2 else '教师仍会退化') + ' · q' + case['question_id'], 24)
        with Image.open(case['image']['path']) as raw: picture = ImageOps.contain(raw.convert('RGB'), (480, 265))
        canvas.paste(picture, (x + 20 + (480 - picture.width) // 2, 690 + (265 - picture.height) // 2))
        wrap(x + 18, 978, case['question'], 480, 19, 3)
        wrap(x + 18, 1065, '原target: ' + case['target'], 480, 18, 2)
        wrap(x + 18, 1120, f"3B: {case['student']['prediction']}  ({case['student']['vqa_score']:.1f})", 480, 18, 2)
        wrap(x + 18, 1180, f"7B: {case['teacher']['prediction']}  ({case['teacher']['vqa_score']:.1f})", 480, 18, 2)
    text(30, 1290, '真实原图与保存回答；事后例子、未重判GT。教师也有错误，原生任务优势不保证每个软标签都正确。', 21)
    text(30, 1333, '下一实验只使用固定训练题的学生当前生成前缀；完整评测题不回填训练。', 21, '#607080')
    canvas.save(OUT / '完整教师_结果与真实案例.png')
    articles = []
    for case in cases:
        url = '../../' + str(Path(case['image']['path']).relative_to(BASE))
        lines = f"3B学生: {case['student']['prediction']} | VQA={case['student']['vqa_score']}\n7B教师: {case['teacher']['prediction']} | VQA={case['teacher']['vqa_score']}"
        articles.append('<article><h2>q' + case['question_id'] + '</h2><img src="' + html.escape(url) + '"><p>' + html.escape(case['question']) + '</p><p>原target: ' + html.escape(case['target']) + '</p><pre>' + html.escape(lines) + '</pre></article>')
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>教师能力与学生前缀蒸馏</title><style>body{max-width:1400px;margin:30px auto;font:18px/1.7 system-ui;background:#f4f6f5;color:#172a3d}article,section{background:white;padding:24px;margin:22px 0}img{max-width:100%;max-height:600px}pre{white-space:pre-wrap}</style><h1>教师更强，为什么学生可能仍学不到？</h1><section><p>GT-prefix KD在正确答案历史上比较下一token概率；学生自由生成后，会进入自己的历史。若前面已经选错token，后续状态不一定在GT训练轨迹上。GKD用学生生成的历史向教师查询，而不是把错误文字当硬标签。</p><p>本次新目标仍为GT CE（含EOS）＋T² KL(q_teacher||p_student)，T=2。区别是辅助KL的prefix由当前学生生成。固定103训练题中47题/127位置已有不同于GT的前缀，说明两种状态来源确有差异，但尚不能证明这是唯一失败原因或新方法一定成功。</p><p><a href="https://arxiv.org/abs/2306.13649">GKD原论文</a> · <a href="https://github.com/huggingface/trl/blob/main/trl/experimental/gkd/gkd_trainer.py">官方参考源码</a> · <a href="../../../packages/RareVLM_minimal_20261001/src/vlm_onpolicy_kd_v1/objective.py">实际原始token拼接与KL</a> · <a href="../../../packages/RareVLM_minimal_20261001/tools/run_onpolicy_teacher_kd_v1.py">复用旧trainer的新目标绑定</a> · <a href="../../../packages/RareVLM_minimal_20261001/docs/VLM_ONPOLICY_KD_V1_PROTOCOL.md">新固定协议</a></p><p>本项目使用部署greedy前缀、原共享实际词表和逐QA平均，未直接套官方文本Trainer；不是PPO/GRPO，也不冒称论文随机采样版本的完整复现。真实提升须看新学生Final32对强起点的完整评测及保持门槛。</p><pre>' + html.escape(json.dumps(result, ensure_ascii=False, indent=2)) + '</pre></section>' + ''.join(articles) + '</html>'
    (OUT / '教师能力与学生前缀蒸馏_公式源码案例.html').write_text(page)
    print(json.dumps({'status': 'rendered', 'figure': str(OUT / '完整教师_结果与真实案例.png'), 'changes': changes, 'cases': [q['question_id'] for q in cases], 'new_model_calls': 0}, ensure_ascii=False))


if __name__ == '__main__': main()
