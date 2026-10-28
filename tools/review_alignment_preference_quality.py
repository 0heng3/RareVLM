"""Independent train-only quality sensitivity;retain frozen85 candidates/results."""
from collections import Counter
import json
from pathlib import Path
from vlm_alignment_signal.common import write_json
from vlm_preference_data.quality import quality_check
from vlm_probe.evaluate import now,sha_file

root=Path('/opt/data/private/longtail_vlm/rarevlm_pilot/outputs/vlm_alignment_signal_v1')
roster=json.loads((root/'DATA_ROSTER.json').read_text());rows={r['question_id']:r for r in roster['questions']}
source=root/'analysis/CANDIDATE_PREFERENCES.json';pairs=json.loads(source.read_text())['pairs'];checks=[];reasons=Counter()
for p in pairs:
    check=quality_check(rows[p['question_id']],p['chosen'],p['rejected'])
    checks.append({**p,'quality_screen':check});reasons.update(check['reasons'])
passed=sum(x['quality_screen']['candidate_admitted'] for x in checks)
write_json(root/'learning_review/QUALITY_SENSITIVITY.json',{'created_at':now(),'original_candidates_sha256':sha_file(source),
    'original_candidates_unchanged':True,'raw_score_pairs':len(pairs),'conservative_candidates':passed,
    'fraction_of512_questions':passed/512,'exclusion_reasons_multilabel':dict(reasons),'checks':checks,
    'source_sha256':sha_file(Path('src/vlm_preference_data/quality.py')),
    'scope':'new train-only conservative candidate screen;not rerun or changed official scoring/gate;not all verified human preference',
    'optimizer_updates':0})
print('QUALITY_SCREEN',len(pairs),'raw ->',passed,'conservative;reasons',dict(reasons))
