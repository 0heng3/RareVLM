"""Describe completed KD outputs and render genuine cases; no new model calls."""
import html
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps
from rarevlm_paths import PROJECT_ROOT
from vlm_deep.common import sha_file, write_json

BASE = PROJECT_ROOT.parents[1] / 'outputs'
ROOT = BASE / 'vlm_full_teacher_kd_v1'
OUT = ROOT / 'result_review_v1'
FONT = '/opt/data/private/.tools/fonts/NotoSansCJKsc-Regular.otf'
ROLES = ('start', 'hard', 'uniform', 'kd')
LABELS = {'start': '强SFT起点', 'hard': '硬GT控制', 'uniform': '均匀尾控制', 'kd': '7B教师KD'}


def main():
    read = lambda p: json.loads(p.read_text())
    analysis = read(ROOT / 'ANALYSIS.json'); data = read(ROOT / 'data/MANIFEST.json')
    paths = {'start': ROOT / 'baseline/transfer_dev/PREDICTIONS.json'}
    paths.update({r: ROOT / f'runs/{r}/transfer/transfer_dev/PREDICTIONS.json' for r in ROLES[1:]})
    rows = {r: read(p) for r, p in paths.items()}
    identity = lambda entries: [(q['question_id'], q['image_id'], q['input_sha256']) for q in entries]
    if any(identity(v) != identity(rows['start']) or len(v) != 1617 for v in rows.values()):
        raise ValueError('Whole saved endpoint identities differ')
    records = {r: {q['question_id']: q for q in v} for r, v in rows.items()}
    bad = {}; good = {}; statistics = {}
    normalize = lambda s: ' '.join(s.casefold().split())
    for role in ROLES[1:]:
        bad[role] = {qid for qid, q in records[role].items() if q['vqa_score'] < records['start'][qid]['vqa_score'] - 1e-9}
        good[role] = {qid for qid, q in records[role].items() if q['vqa_score'] > records['start'][qid]['vqa_score'] + 1e-9}
        statistics[role] = {'qa_improved': len(good[role]), 'qa_regressed': len(bad[role]),
            'qa_score_unchanged': 1617 - len(good[role]) - len(bad[role]),
            'sum_score_gained': sum(records[role][q]['vqa_score'] - records['start'][q]['vqa_score'] for q in good[role]),
            'sum_score_lost': sum(records['start'][q]['vqa_score'] - records[role][q]['vqa_score'] for q in bad[role]),
            'regressed_with_longer_decoded_text': sum(len(records[role][q]['prediction']) > len(records['start'][q]['prediction']) for q in bad[role]),
            'regressed_with_previous_answer_as_substring': sum(normalize(records['start'][q]['prediction']) in normalize(records[role][q]['prediction']) for q in bad[role]),
            'unterminated': sum(not q['terminated'] for q in rows[role])}
    common = set.intersection(*bad.values()); others = bad['hard'] | bad['uniform']
    summary = {'status': 'completed', 'per_role_vs_start': statistics,
        'all_three_regression_qa': len(common), 'all_three_regression_ids': sorted(common, key=int),
        'kd_regression_also_at_least_one_control': len(bad['kd'] & others),
        'kd_regression_only': len(bad['kd'] - others),
        'kd_regression_only_ids': sorted(bad['kd'] - others, key=int),
        'all_three_gain_qa': len(set.intersection(*good.values())),
        'common_regressions_kd_score_lost': sum(records['start'][q]['vqa_score'] - records['kd'][q]['vqa_score'] for q in common),
        'prediction_sha256': {r: sha_file(p) for r, p in paths.items()},
        'optimizer_updates': 0, 'new_model_calls': 0,
        'scope': 'Posthoc descriptive overlap; score changes are not binary correct counts. Substring/length counts are output descriptions, not causal error labels.'}
    write_json(OUT / 'ANALYSIS.json', summary)
    meta = {q['question_id']: q for q in data['evaluations']['transfer_dev']}
    case_ids = ('6107', '23696', '5796')
    cases = [{'question_id': qid, 'question': meta[qid]['question'], 'target': meta[qid]['target'],
        'image': data['images'][meta[qid]['image_id']], 'answers': meta[qid]['answers'],
        'outputs': {r: records[r][qid] for r in ROLES},
        'selection': 'Posthoc illustrative case; not a blinded or representative sample, no GT adjudication.'} for qid in case_ids]
    write_json(OUT / 'SELECTED_CASES.json', cases)
    result = {**analysis, 'posthoc_saved_prediction_review': summary}
    write_json(PROJECT_ROOT / 'results/vlm_full_teacher_kd_v1_metrics.json', result)
    canvas = Image.new('RGB', (1660, 1530), '#f4f6f5'); draw = ImageDraw.Draw(canvas)
    fonts = {}
    def text(x, y, value, size=22, color='#172a3d'):
        if size not in fonts: fonts[size] = ImageFont.truetype(FONT, size)
        draw.text((x, y), str(value), font=fonts[size], fill=color)
    def wrap(x, y, value, width, size=19, max_lines=3):
        font = fonts.setdefault(size, ImageFont.truetype(FONT, size)); line = ''; lines = []
        for char in str(value):
            if char == '\n' or draw.textlength(line + char, font=font) > width:
                lines.append(line); line = '' if char == '\n' else char
            else: line += char
        if line: lines.append(line)
        for i, line in enumerate(lines[:max_lines]): text(x, y + i * (size + 7), line, size)
    text(30, 22, '7B教师有信号，3B学生仍退化：完整结果与真实回答', 34)
    text(30, 79, '全部训练/评测正常结束；三臂共96更新、7731新生成。以下读取固定Final32。', 23)
    text(30, 120, '1617题为历史使用的内部开发集；95%图像簇区间不包含训练seed或历史选择不确定性。', 21, '#607080')
    draw.rounded_rectangle((30, 175, 1630, 548), radius=12, fill='white')
    for x, title in [(50, '模型'), (340, '训练拟合128'), (615, '完整任务1617'), (900, '完整rare342'), (1190, '相对起点完整任务差值')]: text(x, 195, title, 21)
    for i, role in enumerate(ROLES):
        y = 252 + i * 61; m = analysis['metrics'][role]
        text(50, y, LABELS[role], 22)
        for x, key in ((340, 'native_hard'), (615, 'transfer_dev'), (900, 'full_rare342')): text(x, y, f'{m[key]:.3f}%', 23)
        if role != 'start':
            value = analysis['vs_start'][role]['transfer_dev']; text(1190, y, f"{value['delta_pp']:+.3f}pp", 22, '#b24645')
            text(1360, y, f"[{value['ci95'][0]:+.3f}, {value['ci95'][1]:+.3f}]", 18, '#607080')
        else: text(1190, y, '共同强起点', 21)
    text(50, 501, 'KD−硬GT +0.173pp，KD−均匀尾 +0.136pp，两者95%区间均跨0；不是蒸馏优势。', 21)
    text(30, 578, '逐题分数变化：涨分数 / 降分数（VQA部分分，不等同于答对数）', 26)
    for i, role in enumerate(ROLES[1:]):
        x = 30 + i * 540; s = statistics[role]
        draw.rounded_rectangle((x, 630, x + 520, 752), radius=12, fill='white')
        text(x + 18, 643, LABELS[role], 22); text(x + 18, 685, f"{s['qa_improved']}涨 / {s['qa_regressed']}降", 25)
        text(x + 240, 691, f"得{s['sum_score_gained']:.1f} / 失{s['sum_score_lost']:.1f}分", 20)
    text(30, 778, 'KD的45条退化：39条也在控制组下降，6条仅KD下降；16条三组共同下降。', 24)
    text(30, 823, '共同改动更值得定位；尚不能据此确认语言、视觉、数据或学习率是唯一原因。', 22, '#607080')
    for i, case in enumerate(cases):
        x = 30 + i * 540
        draw.rounded_rectangle((x, 890, x + 520, 1455), radius=12, fill='white')
        text(x + 17, 902, ('共同退化' if i < 2 else '共同改善') + ' · q' + case['question_id'], 23)
        with Image.open(case['image']['path']) as raw: picture = ImageOps.contain(raw.convert('RGB'), (480, 200))
        canvas.paste(picture, (x + 20 + (480 - picture.width) // 2, 950 + (200 - picture.height) // 2))
        wrap(x + 17, 1165, case['question'], 480, 18, 2)
        wrap(x + 17, 1220, '原target: ' + case['target'], 480, 17, 2)
        for j, role in enumerate(ROLES):
            q = case['outputs'][role]
            wrap(x + 17, 1282 + j * 37, f"{LABELS[role]}: {q['prediction']} ({q['vqa_score']:.1f})", 480, 15, 2)
    text(30, 1480, '原图等比展示；案例为事后说明，不改GT、不伪造注意力、不把个例当总体归因。', 20, '#607080')
    canvas.save(OUT / '真实结果与回答变化.png')
    formula = 'L = CE(GT,含EOS) + T² KL(q_teacher(T) || p_student(T)), T=2。hard用onehot；uniform保留教师GT概率、均分其余质量。'
    cards = []
    for case in cases:
        relative = str(Path(case['image']['path']).relative_to(BASE))
        image_url = '../../' + relative
        outputs = '\n'.join(f"{LABELS[r]}: {case['outputs'][r]['prediction']} | VQA={case['outputs'][r]['vqa_score']:.1f}" for r in ROLES)
        cards.append('<article><h2>q' + case['question_id'] + '</h2><img src="' + html.escape(image_url) + '"><p>' + html.escape(case['question']) + '</p><p>原target: ' + html.escape(case['target']) + '</p><pre>' + html.escape(outputs) + '</pre><p>事后示例，未重新裁定参考答案；完整原参考保存在SELECTED_CASES.json。</p></article>')
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>KD结果与源码学习</title><style>body{max-width:1400px;margin:30px auto;font:18px/1.7 system-ui;background:#f4f6f5;color:#172a3d}article,section{background:white;padding:24px;margin:22px 0}img{max-width:100%;max-height:600px}pre{white-space:pre-wrap}</style><h1>为什么教师会答，学生仍未泛化？</h1><section><p>' + html.escape(formula) + '</p><p>KL在正确答案前缀上约束每个正文token的概率分布；自由生成使用自身生成前缀。训练拟合提高不保证其他图像上的完整回答改善。共同变化包含训练范围、学习率和额外困难题目标，不能用三个共同退化的对照直接拆出唯一因素。</p><p>真实源码：<a href="../../../packages/RareVLM_minimal_20261001/src/vlm_full_teacher_kd_v1/loss.py">辅助目标</a> · <a href="../../../packages/RareVLM_minimal_20261001/tools/run_full_teacher_kd_v1.py">实际训练</a> · <a href="../../../packages/RareVLM_minimal_20261001/docs/VLM_KD_COMPONENT_RESTORE_V1_PROTOCOL.md">下一步：零训练组件恢复</a>。原三臂阴性关闭，不扫温度、权重、scope或LR。</p><pre>' + html.escape(json.dumps(summary, ensure_ascii=False, indent=2)) + '</pre></section>' + ''.join(cards) + '</html>'
    (OUT / '蒸馏结果_原理与代码.html').write_text(page)
    print(json.dumps({'status': 'completed', 'review': summary, 'figure': str(OUT / '真实结果与回答变化.png')}, ensure_ascii=False))


if __name__ == '__main__': main()
