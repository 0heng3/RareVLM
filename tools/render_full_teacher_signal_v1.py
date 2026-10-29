"""Display saved teacher/student answers; this figure is not a student gain claim."""
import html
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps
from rarevlm_paths import PROJECT_ROOT

ROOT = PROJECT_ROOT.parents[1] / 'outputs/vlm_full_teacher_signal_v1'
FONT = '/opt/data/private/.tools/fonts/NotoSansCJKsc-Regular.otf'


def main():
    read = lambda p: json.loads(p.read_text())
    result = read(ROOT / 'COMPLETION.json'); data = read(ROOT / 'DATA.json'); details = read(ROOT / 'DIAGNOSTICS.json')
    meta = {q['question_id']: q for q in data['rows']}
    selected = [(q, '教师纠正学生残错') for q in details if q['novel_teacher_correction']][:2]
    selected += [(q, '教师也有退化：此题不参与KD') for q in details if q['student']['literal_correct'] and not q['teacher']['literal_correct']][:1]
    image = Image.new('RGB', (1600, 1310), '#f4f6f5'); draw = ImageDraw.Draw(image)
    def text(x, y, value, size=23, color='#172a3d'):
        draw.text((x, y), str(value), font=ImageFont.truetype(FONT, size), fill=color)
    def wrap(x, y, value, width=460, size=21, maximum=3):
        font = ImageFont.truetype(FONT, size); lines = []; line = ''
        for char in value:
            if draw.textlength(line + char, font=font) > width:
                lines.append(line); line = char
            else:
                line += char
        if line: lines.append(line)
        for i, line in enumerate(lines[:maximum]): text(x, y + i * (size + 9), line, size)
    text(30, 20, '更强整图教师的监督准入：已通过，学生提分仍待验证', 33)
    text(30, 80, '同一128条已见训练题；两个不同冻结模型；本图来自0更新诊断，不是学生训练后提升。', 22)
    cards = [('7B教师', '105 / 128'), ('较强3B SFT', '36 / 128'), ('新增纠正 / 退化', '71 / 2'), ('有结构信息', '73 题'), ('最终辅助mask', '103 题')]
    for i, (label, value) in enumerate(cards):
        x = 25 + i * 315; draw.rounded_rectangle((x, 140, x + 300, 255), radius=12, fill='white')
        text(x + 17, 152, label, 20); text(x + 17, 196, value, 30, '#127467')
    articles = []
    for i, (record, title) in enumerate(selected):
        q = meta[record['question_id']]; x = 25 + i * 525; y = 290
        draw.rounded_rectangle((x, y, x + 505, 1145), radius=12, fill='white')
        wrap(x + 17, y + 14, title, size=23, maximum=2)
        path = Path(data['images'][q['image_id']]['path'])
        with Image.open(path) as raw: picture = ImageOps.contain(raw.convert('RGB'), (470, 315))
        image.paste(picture, (x + 17 + (470 - picture.width) // 2, y + 85 + (315 - picture.height) // 2))
        wrap(x + 17, y + 424, q['question'], maximum=3)
        wrap(x + 17, y + 535, '原参考：' + q['target'], maximum=3)
        wrap(x + 17, y + 645, '3B学生：' + record['student']['prediction'], maximum=2)
        wrap(x + 17, y + 720, '7B教师：' + record['teacher']['prediction'], maximum=2)
        text(x + 17, y + 805, '训练题 q' + q['question_id'], 17, '#607080')
        articles.append('<article><h2>' + html.escape(title) + '</h2><p>' + html.escape(q['question']) + '</p><pre>' + html.escape(json.dumps({'target': q['target'], 'student': record['student'], 'teacher': record['teacher'], 'T2': record['temperatures']['2']}, ensure_ascii=False, indent=2)) + '</pre></article>')
    text(30, 1172, '新增纠正均正文NLL下降0.82594；共享151665实际token；填充概率质量最大0.1973%。', 21)
    text(30, 1215, '原图等比展示；无GT裁图、无注意力图。教师正确与非GT结构都须检验，学生能力以完整评测为准。', 20)
    text(30, 1260, '三组新学生：原CE + 额外硬GT / 同GT质量均匀尾 / 真实教师T²KL；同起点、同mask、同预算。', 20)
    image.save(ROOT / '教师监督_真实结果.png')
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>完整教师监督与三组蒸馏</title><style>body{max-width:1450px;margin:30px auto;font:18px/1.7 system-ui;background:#f4f6f5;color:#172a3d}img{max-width:100%}article,section{background:white;padding:20px;margin:20px 0}pre{white-space:pre-wrap}</style><h1>7B整图教师：监督信号已通过，学生能力尚待完整验证</h1><section><p>教师105/128、学生36/128都是已见训练池的严格匹配计数；不能写成学生从36提升到105。教师不同模型与预训练能力同时变化，不作纯参数量因果解释。103个自然回答正确且正文/EOS置信合格的缓存用于下一训练。</p><pre>原CE：每QA有效assistant token均值，含真实EOS\n辅助：T² KL(q || softmax(z_student/T))，T=2，正文不含EOS\nhard: q=onehot(GT)\nuniform: q[GT]=p_teacher[GT]，其他实际token均分剩余概率\nkd: q=p_teacher；实际token映射一致后各自归一化</pre><p>均匀尾保留教师的GT概率和非GT总量，只去掉具体非GT结构，因此KD若超过它，才支持这些结构带来额外收益。KL数值受目标熵影响，不能按哪组loss更低选模型。</p><p><a href="../../packages/RareVLM_minimal_20261001/src/vlm_full_teacher_kd_v1/loss.py">真实损失代码</a> · <a href="../../packages/RareVLM_minimal_20261001/docs/VLM_FULL_TEACHER_KD_V1_PROTOCOL.md">三组固定协议与完整门槛</a> · <a href="../rarevlm_monitor/状态看板.html">实时训练看板</a></p></section><img src="教师监督_真实结果.png">' + ''.join(articles) + '</html>'
    (ROOT / '教师监督_源码原理与真实案例.html').write_text(page)
    (PROJECT_ROOT / 'results/vlm_full_teacher_signal_v1_metrics.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print({'status': 'rendered', 'cases': len(selected), 'new_model_calls': 0})


if __name__ == '__main__':
    main()
