"""Read-only completed student-prefix KD diagnosis; never alter original scores."""
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps
from rarevlm_paths import PROJECT_ROOT
from vlm_deep.common import write_json

BASE = PROJECT_ROOT.parents[1] / 'outputs'
ROOT = BASE / 'vlm_onpolicy_teacher_kd_v1'
OUT = ROOT / 'result_review_v1'
FONT = '/opt/data/private/.tools/fonts/NotoSansCJKsc-Regular.otf'


def main():
    read = lambda p: json.loads(p.read_text())
    result = read(ROOT / 'ANALYSIS.json'); data = read(ROOT / 'data/MANIFEST.json')
    trajectories = [q for p in sorted((ROOT / 'runs/op_kd/train/rollouts').glob('*.json')) for q in read(p)['rollouts']]
    wrong = [q for q in trajectories if q['body_ids'] != q['gt_body_ids']]
    trajectory = {'exposures': len(trajectories), 'body_tokens': sum(q['body_tokens'] for q in trajectories),
        'off_gt_prefix_positions': sum(q['off_GT_prefix_positions'] for q in trajectories),
        'exposures_with_off_gt_prefix': sum(q['off_GT_prefix_positions'] > 0 for q in trajectories),
        'distinct_qa_with_off_gt_prefix': len({q['question_id'] for q in trajectories if q['off_GT_prefix_positions'] > 0}),
        'raw_body_different': len(wrong), 'different_body_without_off_gt_prefix': sum(q['off_GT_prefix_positions'] == 0 for q in wrong),
        'strict_gt_prefix_early_end': sum(len(q['body_ids']) < len(q['gt_body_ids']) and q['body_ids'] == q['gt_body_ids'][:len(q['body_ids'])] for q in trajectories),
        'first_body_token_differs': sum(q['body_ids'][0] != q['gt_body_ids'][0] for q in wrong)}
    paths = {'start': ROOT / 'baseline/transfer_dev/PREDICTIONS.json',
        'kd': BASE / 'vlm_full_teacher_kd_v1/runs/kd/transfer/transfer_dev/PREDICTIONS.json',
        'op_kd': ROOT / 'runs/op_kd/transfer/transfer_dev/PREDICTIONS.json'}
    rows = {r: read(p) for r, p in paths.items()}
    identity = lambda v: [(q['question_id'], q['image_id'], q['input_sha256']) for q in v]
    if any(identity(v) != identity(rows['start']) or len(v) != 1617 for v in rows.values()):
        raise ValueError('Complete paired prediction identity differs')
    records = {r: {q['question_id']: q for q in v} for r, v in rows.items()}; bad = {}; counts = {}
    for role in ('kd', 'op_kd'):
        bad[role] = {q for q in records['start'] if records[role][q]['vqa_score'] < records['start'][q]['vqa_score'] - 1e-9}
        good = {q for q in records['start'] if records[role][q]['vqa_score'] > records['start'][q]['vqa_score'] + 1e-9}
        counts[role] = {'score_improved': len(good), 'score_regressed': len(bad[role]),
            'score_gained': sum(records[role][q]['vqa_score'] - records['start'][q]['vqa_score'] for q in good),
            'score_lost': sum(records['start'][q]['vqa_score'] - records[role][q]['vqa_score'] for q in bad[role])}
    shared = bad['kd'] & bad['op_kd']
    diagnosis = {'status': 'completed', 'trajectory': trajectory, 'full_task_changes': counts,
        'shared_regression_qa': len(shared), 'new_only_regression_qa': len(bad['op_kd'] - bad['kd']),
        'new_model_calls': 0, 'new_updates': 0,
        'scope': 'Posthoc saved-trajectory and prediction description; overlap is not unique causal attribution.'}
    write_json(OUT / 'DIAGNOSIS.json', diagnosis)
    write_json(PROJECT_ROOT / 'results/vlm_onpolicy_teacher_kd_v1_metrics.json', {**result, 'posthoc_diagnosis': diagnosis})
    meta = {q['question_id']: q for q in data['evaluations']['transfer_dev']}
    worst = sorted(shared, key=lambda q: (records['op_kd'][q]['vqa_score'] - records['start'][q]['vqa_score'], int(q)))[0]
    cases = [{'question_id': q, 'question': meta[q]['question'], 'target': meta[q]['target'],
        'image': data['images'][meta[q]['image_id']], 'outputs': {r: records[r][q] for r in records},
        'selection': 'Posthoc illustration: prior regression repaired, highest shared score loss, prior common gain; no GT re-adjudication.'} for q in ('6107', worst, '5796')]
    write_json(OUT / 'CASES.json', cases)
    canvas = Image.new('RGB', (1640, 1370), '#f4f6f5'); draw = ImageDraw.Draw(canvas); fonts = {}
    def text(x, y, value, size=21, color='#172a3d'):
        font = fonts.setdefault(size, ImageFont.truetype(FONT, size)); draw.text((x, y), str(value), font=font, fill=color)
    def wrap(x, y, value, width, size=18, lines=3):
        font = fonts.setdefault(size, ImageFont.truetype(FONT, size)); chunks = []; line = ''
        for char in str(value):
            if draw.textlength(line + char, font=font) > width: chunks.append(line); line = char
            else: line += char
        if line: chunks.append(line)
        for i, line in enumerate(chunks[:lines]): text(x, y + i * (size + 6), line, size)
    labels = {'start': '强SFT起点', 'kd': '旧GT前缀KD', 'op_kd': '新学生前缀KD'}
    text(30, 22, '学生前缀已实际改变，完整任务仍未提升', 34)
    text(30, 78, '32更新、206训练生成、2577评测全部正常结束；固定配置停止，不继续扫参。', 23)
    for x, label in ((40, '模型'), (440, '完整1617'), (800, '完整rare342'), (1200, '保持64')): text(x, 155, label, 23)
    for i, role in enumerate(labels):
        y = 218 + i * 65; text(40, y, labels[role], 23)
        for x, key in ((440, 'transfer_dev'), (800, 'full_rare342'), (1200, 'retention')): text(x, y, f"{result['metrics'][role][key]:.3f}%", 25)
    text(30, 435, '新−起点 −0.996pp，95% [−1.713, −0.289]；新−旧KD +0.186pp，区间跨0。', 24, '#b24645')
    text(30, 486, '实际不同前缀位置251/710（35.4%），覆盖82次曝光/48题；新机制不是空操作。', 23)
    text(30, 533, '完整任务21题涨、38题降；30条退化与旧KD重叠，8条仅新模型退化。', 23)
    for i, case in enumerate(cases):
        x = 30 + i * 535
        draw.rounded_rectangle((x, 610, x + 515, 1265), radius=12, fill='white')
        text(x + 16, 627, ['修复旧KD错误', '两种KD仍共同退化', '两种KD共同改善'][i] + ' · q' + case['question_id'], 21)
        with Image.open(case['image']['path']) as raw: picture = ImageOps.contain(raw.convert('RGB'), (475, 245))
        canvas.paste(picture, (x + 20 + (475 - picture.width) // 2, 680 + (245 - picture.height) // 2))
        wrap(x + 16, 945, case['question'], 475, 18, 3)
        wrap(x + 16, 1030, '原target: ' + case['target'], 475, 18, 2)
        for j, role in enumerate(labels):
            q = case['outputs'][role]; wrap(x + 16, 1090 + j * 53, f"{labels[role]}: {q['prediction']} ({q['vqa_score']:.1f})", 475, 16, 2)
    text(30, 1300, '真实原图与原评分；事后案例未重新判定GT。固定单seed内部结果，不代表外部或种子稳定性。', 20)
    canvas.save(OUT / '学生前缀KD_结果与真实案例.png')
    print(json.dumps(diagnosis, ensure_ascii=False))


if __name__ == '__main__': main()
