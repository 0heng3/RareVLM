"""Render the completed source-holdout evidence, without rescoring or model calls."""
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps
from rarevlm_paths import PROJECT_ROOT

ROOT = PROJECT_ROOT.parents[1] / 'outputs/vlm_stvqa_holdout_transfer_v1'
FONT = '/opt/data/private/.tools/fonts/NotoSansCJKsc-Regular.otf'


def main():
    read = lambda path: json.loads(path.read_text())
    result = read(ROOT / 'ANALYSIS.json'); data = read(ROOT / 'DATA.json')
    queue = read(ROOT / 'queue/STATUS.json')
    roles = ['start', 'old_ce', 'old_dpo', 'new_ce', 'new_dpo']
    labels = ['共同起点', '旧裁词CE', '旧裁词DPO', '新整图SFT', '新整图DPO']
    records = {'start': read(ROOT / 'baseline/PREDICTIONS.json')}
    records.update({role: read(ROOT / 'results' / role / 'PREDICTIONS.json') for role in roles[1:]})
    q = data['rows'][0]
    cases = {role: next(row for row in rows if row['question_id'] == q['question_id']) for role, rows in records.items()}
    canvas = Image.new('RGB', (1600, 1330), '#f4f6f5'); draw = ImageDraw.Draw(canvas)
    def text(x, y, value, size=23, color='#172a3d'):
        draw.text((x, y), str(value), font=ImageFont.truetype(FONT, size), fill=color)
    text(30, 22, '256图来源保持：NLL变好，生成准确率仍未证实提升', 34)
    text(30, 82, '1024新生成 + 1024 GT前向；8个worker exit0、0争用、0更新。', 22)
    text(30, 120, '这是固定单seed模型的内部来源验证；不是ANLS、原生VQA或独立外部确认。', 21)
    for left, title in [(30, '自由生成：exact正确数 / 256'), (815, 'GT前缀：平均正文NLL，越低越好')]:
        draw.rounded_rectangle((left, 175, left + 755, 620), radius=12, fill='white')
        text(left + 20, 192, title, 27)
        for i, role in enumerate(roles):
            m = result['summaries'][role]; y = 260 + i * 63
            text(left + 20, y, labels[i], 21)
            value = m['exact_percent'] / 100 if left == 30 else m['mean_body_nll'] / .5
            color = '#127467' if role == 'new_ce' else '#316f9e' if role == 'new_dpo' else '#bdcdd5'
            draw.rectangle((left + 180, y + 5, left + 180 + 385 * value, y + 30), fill=color)
            shown = f"{m['literal_correct']}/256  {m['exact_percent']:.3f}%" if left == 30 else f"{m['mean_body_nll']:.6f}"
            text(left + 575, y, shown, 19)
        text(left + 20, 582, '横条从0开始；所有角色0长度删失。', 18, '#607080')
    text(35, 650, '新SFT−起点 +0.781pp；四比较校正98.75%区间 [−1.172, +2.734]。', 23)
    text(35, 692, '新DPO−新SFT −0.391pp；校正区间 [−2.344, +1.172]，未证实来源内损伤。', 23)
    draw.rounded_rectangle((30, 755, 1570, 1250), radius=12, fill='white')
    text(50, 772, '预测前固定第一例：GT概率有所提高，实际答案仍错', 28)
    with Image.open(data['images'][q['image_id']]['path']) as raw:
        image = ImageOps.contain(raw.convert('RGB'), (520, 390))
    canvas.paste(image, (50, 830))
    text(610, 835, 'What is the white text on the dog\'s collar?', 22)
    text(610, 880, '原参考：woof；实际输出始终为hoof。', 23)
    for i, role in enumerate(('start', 'new_ce', 'new_dpo')):
        row = cases[role]
        text(610, 935 + i * 48, f"{labels[roles.index(role)]}：{row['prediction']}  bodyNLL={row['body_nll']:.4f}", 23)
    text(610, 1097, 'NLL使用GT前缀；自由生成按自己的已生成前缀继续。', 21, '#607080')
    text(610, 1135, 'GT平均概率提高，不保证每步argmax或完整答案正确。', 21, '#607080')
    text(610, 1173, '原图等比显示；无注意力重建、无新人工标注。', 21, '#607080')
    text(30, 1282, '原ANALYSIS/预测/评分/门槛未改；只读取完整结果，0新模型调用。', 20)
    canvas.save(ROOT / '来源保持_结果与NLL解释.png')
    (PROJECT_ROOT / 'results/vlm_stvqa_holdout_transfer_v1_metrics.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print({'status': 'rendered', 'worker_failures': sum(j['exit_code'] != 0 for j in queue['jobs'].values()),
           'overview': str(ROOT / '来源保持_结果与NLL解释.png'), 'new_model_calls': 0})


if __name__ == '__main__':
    main()
