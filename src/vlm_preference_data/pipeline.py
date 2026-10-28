"""Full10000-image sampling and conservative quality review,zero training."""
import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from rarevlm_paths import PROJECT_ROOT,SOURCE_ROOT
from research_runtime.queue import run_queue
from vlm_deep.common import read_manifest,NORMALIZER,consensus
from vlm_probe.evaluate import lock_file,now,sha_file
from vlm_alignment_signal.common import write_json
from .quality import quality_check

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);a=p.parse_args()
    cfg=json.loads(a.config.read_text());root=Path(cfg['output_root']);root.mkdir(parents=True,exist_ok=True)
    owner=(root/'STUDY.lock').open('a+');fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB);started=time.monotonic()
    parent=Path(cfg['parent_audit_root']);parent_result=json.loads((parent/'analysis/ANALYSIS.json').read_text())
    if not parent_result['gates']['dpo_supply_passed']:raise ValueError('Parent supply gate failed')
    m=read_manifest(Path(cfg['manifest_path']));images=m['images']
    excluded={r['image_id'] for s in ['dev','final'] for r in m['splits'][s]}
    badpixels={images[i]['pixel_sha256'] for i in excluded};badbytes={images[i]['sha256'] for i in excluded}
    def key(r):return hashlib.sha256(f'rarevlm-alignment-v1:{cfg["selection_seed"]}:{r["question_id"]}:{images[r["image_id"]]["pixel_sha256"]}'.encode()).digest()
    rows=[];seen=set();frequency=Counter(NORMALIZER(r['target']) for r in m['splits']['train_all'])
    for r in sorted(m['splits']['train_all'],key=key):
        im=images[r['image_id']]
        if r['source_split']!='train' or r['image_id'] in excluded or im['pixel_sha256'] in badpixels or im['sha256'] in badbytes or im['pixel_sha256'] in seen:continue
        seen.add(im['pixel_sha256']);rows.append({**r,'train_target_frequency':frequency[NORMALIZER(r['target'])]})
        if len(rows)==cfg['questions']:break
    if len(rows)!=10000:raise ValueError('Complete10000 distinct train images unavailable')
    for r in rows:
        im=images[r['image_id']]
        if sha_file(Path(cfg['manifest_path']).parent/im['path'])!=im['sha256']:raise ValueError('Selected training image changed')
    previous=json.loads((parent/'DATA_ROSTER.json').read_text())
    if [r['question_id'] for r in rows[:512]]!=[r['question_id'] for r in previous['questions']]:raise ValueError('Parent roster prefix differs')
    roster={'questions':rows,'images':{r['image_id']:images[r['image_id']] for r in rows},
        'image_root':str(Path(cfg['manifest_path']).parent),'source_manifest_sha256':m['manifest_sha256'],
        'parent_overlap_questions':512,'selection':'same fixed hash;one question per distinct pixel hash;not score-selected'}
    lock_file(root/'DATA_ROSTER.json',roster)
    parent_bind=json.loads((parent/'DATA_BINDINGS.json').read_text())
    if sha_file(PROJECT_ROOT/cfg['reference_config'])!=parent_bind['reference_config_sha256']:raise ValueError('Frozen reference config changed')
    if sha_file(Path(cfg['manifest_path']))!=parent_bind['manifest_sha256']:raise ValueError('Frozen data manifest changed')
    parent_lock=json.loads((parent/'STUDY_INPUT_LOCK.json').read_text())
    for name,digest in parent_lock['sources'].items():
        if sha_file(SOURCE_ROOT/name)!=digest:raise ValueError('Frozen parent source changed:'+name)
    protected=json.loads((parent/'FROZEN_GUARD.json').read_text())['protected_sha256']
    for file,digest in protected.items():
        path=Path(file)
        if path.parent==Path(cfg['model_root']) and sha_file(path)!=digest:raise ValueError('Initial base weight/config changed')
    for file,digest in parent_bind['adapter_files_sha256'].items():
        if sha_file(Path(file))!=digest:raise ValueError('Initial adapter changed')
    lock_file(root/'DATA_BINDINGS.json',{'parent_bindings_sha256':sha_file(parent/'DATA_BINDINGS.json'),
        'parent_analysis_sha256':sha_file(parent/'analysis/ANALYSIS.json'),'manifest_sha256':sha_file(Path(cfg['manifest_path'])),
        'roster_sha256':sha_file(root/'DATA_ROSTER.json'),'adapter_files_sha256':parent_bind['adapter_files_sha256']})
    names=[str(p.relative_to(SOURCE_ROOT)) for folder in ['vlm_preference_data','vlm_alignment_signal'] for p in sorted((SOURCE_ROOT/folder).glob('*.py'))]
    names+=['vlm_deep/common.py','vlm_deep/model.py','vlm_sft/common.py','vlm_probe/evaluate.py','vlm_precision/evaluate.py',
        'vlm_joint/evaluate.py','research_runtime/queue.py','vlm_deep/third_party/vqa_eval_metric.py']
    identity={'config_sha256':sha_file(a.config),'sources':{n:sha_file(SOURCE_ROOT/n) for n in names},
        'data_bindings_sha256':sha_file(root/'DATA_BINDINGS.json'),
        'protocol_sha256':sha_file(PROJECT_ROOT/'docs/VLM_PREFERENCE_DATA_V1_PROTOCOL.md'),
        'quality_contract_sha256':sha_file(root/'CPU_CONTRACTS.json')}
    lock_file(root/'STUDY_INPUT_LOCK.json',identity)
    def status(phase,**kw):write_json(root/'STATUS.json',{'status':phase,'updated_at':now(),'controller_pid':os.getpid(),
        'source_lock_sha256':sha_file(root/'STUDY_INPUT_LOCK.json'),'existing_tasks_untouched':True,'optimizer_updates':0,**kw})
    if (root/'COMPLETION.json').exists():status('completed');return
    try:
        status('sampling_full_data');jobs=[{'key':f'role{r}','command':[sys.executable,'-m','vlm_alignment_signal.generate','--config',str(a.config),'--role',str(r)],
            'completion_marker':str(root/'roles'/f'role{r}'/'COMPLETE.json')} for r in cfg['roles']]
        run_queue(root/'sampling_queue',jobs,identity);status('quality_screening')
        by_id={r['question_id']:r for r in rows};raw=[];accepted=[];reasons=Counter();count=0
        for role in cfg['roles']:
            directory=root/'roles'/f'role{role}';done=json.loads((directory/'COMPLETE.json').read_text())
            if done['predictions_sha256']!=sha_file(directory/'predictions.jsonl'):raise ValueError('Saved generation changed')
            for line in (directory/'predictions.jsonl').read_text().splitlines():
                x=json.loads(line);row=by_id[x['question_id']];count+=1
                if [consensus(s['prediction'],row['answers']) for s in x['samples']]!=x['rewards']:raise ValueError('Complete-data rescore mismatch')
                if x['preference'] is None:continue
                pair=x['preference'];c=x['samples'][pair['chosen_index']]['prediction'];r=x['samples'][pair['rejected_index']]['prediction']
                q=quality_check(row,c,r);record={'question_id':row['question_id'],'image_id':row['image_id'],'question':row['question'],
                    'chosen':c,'rejected':r,'quality_screen':q,'label_source':'program ten-reference consensus,not direct human preference',
                    'source_prediction_lock_sha256':x['input_lock_sha256']}
                raw.append(record);reasons.update(q['reasons'])
                if q['candidate_admitted']:accepted.append(record)
        if count!=10000:raise ValueError('Incomplete full data')
        write_json(root/'analysis/PREFERENCES.json',{'raw_candidates':raw,'conservative_candidates':accepted,
            'training_allowed':False,'semantic_review':'pending;rules do not guarantee semantics'})
        result={'status':'completed','completed_at':now(),'questions':count,'generated_responses':count*5,
            'raw_pairs':len(raw),'conservative_pairs':len(accepted),'exclusion_reasons_multilabel':dict(reasons),
            'optimizer_updates':0,'trained_adapters':0,'training_launched':False,'controller_seconds':time.monotonic()-started,
            'parent_overlap_questions':512,'limits':cfg['limits']}
        write_json(root/'analysis/ANALYSIS.json',result);write_json(root/'COMPLETION.json',{'status':'completed',
            'analysis_sha256':sha_file(root/'analysis/ANALYSIS.json'),'completed_at':now(),'optimizer_updates':0})
        status('completed',raw_pairs=len(raw),conservative_pairs=len(accepted));print('FULL_PREFERENCE_DATA_COMPLETE',result,flush=True)
    except BaseException as e:status('failed',reason=repr(e));raise
    finally:owner.close()

if __name__=='__main__':main()
