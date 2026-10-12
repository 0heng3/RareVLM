"""Check completed deep runs, exact training coverage and stored benchmark scores."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from .common import content_sha,read_manifest,sha_file,write_json
from .evaluate import aggregate,score_prediction


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--asset-root',type=Path,required=True)
    p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path(__file__).resolve().parents[1]/'configs/vlm_deep_v2.json')
    args=p.parse_args();c=json.loads(args.config.read_text());m=read_manifest(args.asset_root/'prepared/manifest.json',True)
    for split in ['train_all','dev','final']:
        pixels={m['images'][r['image_id']]['pixel_sha256'] for r in m['splits'][split]}
        for other in ['train_all','dev','final']:
            if other==split:continue
            assert not pixels&{m['images'][r['image_id']]['pixel_sha256'] for r in m['splits'][other]}
    train_count=0;eval_count=0;predictions=0
    for file in args.run_root.rglob('TRAIN_RESULT.json'):
        t=json.loads(file.read_text());assert t['config_sha256']==content_sha(c)
        assert t['manifest_sha256']==m['manifest_sha256']
        assert t['question_count']==t['actual_questions_seen']==len(m['splits'][t['split']])
        assert t['completed_updates']==math.ceil(t['question_count']/c['effective_batch'])
        assert t['trainable_parameters']==c['methods'][t['method']]['expected_trainable']
        assert t['frozen_parameter_sha256_before']==t['frozen_parameter_sha256_after']
        assert t['skipped_updates']==0 and t['epochs_completed']==1
        for name,digest in t['adapter_files_sha256'].items():assert sha_file(file.parent/'adapter'/name)==digest
        train_count+=1
    for file in args.run_root.rglob('EVALUATION.json'):
        e=json.loads(file.read_text());assert e['config_sha256']==content_sha(c)
        if e['task']=='textvqa':
            references=m['splits'][e['split']];expected_sha=m['manifest_sha256']
        else:
            transfer=json.loads((args.asset_root/'transfer'/e['task']/'manifest.json').read_text())
            references=transfer['records'];expected_sha=transfer['manifest_sha256']
        assert e['manifest_sha256']==expected_sha
        assert sha_file(file.parent/'predictions.json')==e['predictions_sha256']
        rows=json.loads((file.parent/'predictions.json').read_text())
        lookup={r['question_id']:r for r in references}
        assert len(rows)==len(lookup)==e['n'] and {r['question_id'] for r in rows}==set(lookup)
        for r in rows:
            reference=lookup[r['question_id']]
            assert r['image_id']==reference['image_id']
            assert math.isclose(r['score'],score_prediction(r['prediction'],reference,e['task']),abs_tol=1e-12)
        score,_=aggregate(rows,e['task']);assert math.isclose(score,e['score_percent'],abs_tol=1e-10)
        eval_count+=1;predictions+=len(rows)
    expected=3*len(c['training_seeds'])+2*len(c['training_seeds'])
    assert train_count==expected,f'{train_count}/{expected} training runs complete'
    # Four numerical paths + three baseline tasks + dev/final/transfer for every trained checkpoint.
    assert eval_count==4+3+4*expected,f'{eval_count} evaluations are incomplete'
    result={'passed':True,'completed_training_runs':train_count,'completed_evaluations':eval_count,
        'stored_predictions_rescored':predictions,'official_image_artifacts_verified':len(m['images']),
        'full_textvqa_validation_questions':len(m['splits']['final']),
        'split_pixels_disjoint':True,'frozen_weights_and_parameter_budgets_verified':True}
    write_json(args.run_root/'VERIFICATION.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
