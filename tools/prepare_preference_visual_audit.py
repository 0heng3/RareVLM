"""Render the predeclared SHA sample; never select based on observed answers."""
import hashlib
import json
from pathlib import Path
import textwrap
from PIL import Image, ImageDraw, ImageFont

ROOT = Path('/opt/data/private/longtail_vlm/rarevlm_pilot/outputs')
source = ROOT / 'vlm_preference_data_v1'
out = ROOT / 'vlm_preference_review_v1'
plan = json.loads((out / 'AUDIT_PLAN.json').read_text())
raw = (source / 'analysis/PREFERENCES.json').read_bytes()
assert hashlib.sha256(raw).hexdigest() == plan['source_preferences_sha256']
pairs = {x['question_id']: x for x in json.loads(raw)['conservative_candidates']}
data = json.loads((source / 'DATA_ROSTER.json').read_text())
rows = {x['question_id']: x for x in data['questions']}
records = []
for index, qid in enumerate(plan['selected_question_ids']):
    pair = pairs[qid]
    im = data['images'][pair['image_id']]
    records.append({'index': index, 'pair': pair, 'answers': rows[qid]['answers'],
                    'image_path': str(Path(data['image_root']) / im['path'])})
font = ImageFont.truetype('/opt/data/private/.tools/fonts/NotoSansCJKsc-Regular.otf', 18)
for page in range(8):
    sheet = Image.new('RGB', (1600, 1600), 'white')
    draw = ImageDraw.Draw(sheet)
    for local, rec in enumerate(records[page * 8:page * 8 + 8]):
        x = rec['pair']; left = local % 2 * 800; top = local // 2 * 400
        img = Image.open(rec['image_path']).convert('RGB')
        img.thumbnail((395, 365)); sheet.paste(img, (left, top + 30))
        txt = (f"#{rec['index']} QID {x['question_id']}\nQ: {x['question']}\n"
               f"C: {x['chosen']}\nR: {x['rejected']}\n"
               f"Votes: {x['quality_screen']['chosen_normalized_votes']}/10")
        yy = top + 8
        for line in txt.splitlines():
            for wrapped in textwrap.wrap(line, 34):
                draw.text((left + 402, yy), wrapped, font=font, fill='black'); yy += 24
        draw.line((left, top + 399, left + 799, top + 399), fill='gray')
    sheet.save(out / f'page{page}.jpg', quality=94)
(out / 'SAMPLE.json').write_text(json.dumps(records, ensure_ascii=False, indent=2) + '\n')
for r in records:
    x = r['pair']
    print(r['index'], x['question_id'], x['question'], 'C=', repr(x['chosen']), 'R=', repr(x['rejected']))
