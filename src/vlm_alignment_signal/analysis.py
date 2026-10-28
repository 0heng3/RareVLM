import json
import math
from pathlib import Path
import numpy as np
from vlm_deep.common import consensus,NORMALIZER
from vlm_probe.evaluate import now,sha_file
from .common import write_json,group_advantages,preference
from .prepare import verify_guard

def wilson(k,n):
    z=1.959963984540054;p=k/n;den=1+z*z/n
    mid=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return {'count':k,'denominator':n,'fraction':p,'descriptive_wilson95':[mid-half,mid+half],
        'scope':'one event per distinct audit image;descriptive binomial approximation,not market or generalization CI'}

def analyze(config_path):
    cfg=json.loads(config_path.read_text());root=Path(cfg['output_root']);roster=json.loads((root/'DATA_ROSTER.json').read_text())
    by_qid={r['question_id']:r for r in roster['questions']};items=[];contracts=[];times=[]
    for role in cfg['roles']:
        d=root/'roles'/f'role{role}';r=json.loads((d/'COMPLETE.json').read_text())
        if r['predictions_sha256']!=sha_file(d/'predictions.jsonl'):raise ValueError('Saved generation changed')
        items += [json.loads(line) for line in (d/'predictions.jsonl').read_text().splitlines()]
        contracts+=r['dpo_gradient_contracts'];times.append(r['sampling_active_seconds'])
    if len(items)!=512 or len({x['question_id'] for x in items})!=512 or len({x['image_id'] for x in items})!=512:raise ValueError('Incomplete/distinct-image audit')
    for x in items:
        row=by_qid[x['question_id']];rewards=[consensus(s['prediction'],row['answers']) for s in x['samples']]
        if rewards!=x['rewards'] or consensus(x['greedy']['prediction'],row['answers'])!=x['greedy_score']:raise ValueError('Independent rescore differs')
        if not np.allclose(group_advantages(rewards),x['advantages'],rtol=0,atol=1e-12):raise ValueError('Advantage differs')
        pair=preference([s['prediction'] for s in x['samples']],rewards,NORMALIZER,
            cfg['preference_min_chosen_score'],cfg['preference_max_rejected_score'])
        if pair!=x['preference']:raise ValueError('Preference eligibility differs')
        if x['input']['input_references_used'] or x['input']['input_truncated']:raise ValueError('Input reference/truncation violation')
    n=len(items);pairs=sum(x['preference'] is not None for x in items)
    varied=sum(max(x['rewards'])>min(x['rewards']) for x in items)
    summary={'greedy_train_consensus_percent':100*float(np.mean([x['greedy_score'] for x in items])),
        'sample_train_consensus_percent':100*float(np.mean([r for x in items for r in x['rewards']])),
        'dpo_pair_supply':wilson(pairs,n),'grpo_nonzero_reward_groups':wilson(varied,n),
        'constant_reward_groups':n-varied,
        'all_four_zero_reward_groups':sum(max(x['rewards'])==0 for x in items),
        'all_four_full_score_groups':sum(min(x['rewards'])==1 for x in items),
        'mean_unique_normalized_samples':float(np.mean([x['unique_normalized_samples'] for x in items])),
        'empty_sample_answers':sum(not s['prediction'].strip() for x in items for s in x['samples'])}
    slices={}
    for name,lo,hi in [('rare_1_5',1,5),('mid_6_20',6,20),('frequent_21plus',21,10**9)]:
        selected=[x for x in items if lo<=x['train_target_frequency']<=hi]
        slices[name]={'questions':len(selected),'dpo_pairs':sum(x['preference'] is not None for x in selected),
            'informative_groups':sum(max(x['rewards'])>min(x['rewards']) for x in selected),
            'greedy_consensus_percent':100*float(np.mean([x['greedy_score'] for x in selected])) if selected else None}
    pairs_file=[]
    for x in items:
        if x['preference'] is None:continue
        c=x['preference']['chosen_index'];r=x['preference']['rejected_index']
        pairs_file.append({'question_id':x['question_id'],'image_id':x['image_id'],'question':x['question'],
            'chosen':x['samples'][c]['prediction'],'rejected':x['samples'][r]['prediction'],
            'chosen_score':x['rewards'][c],'rejected_score':x['rewards'][r],
            'label_source':'rule-based ten-human-reference consensus;not direct human preference',
            'semantic_review':'pending','audit_only_not_a_training_manifest':True})
    write_json(root/'analysis/CANDIDATE_PREFERENCES.json',{'pairs':pairs_file,'training_allowed':False,
        'reason':'audit corpus only;semantic checks/full data protocol required'})
    result={'status':'completed','completed_at':now(),'questions':n,'generated_responses':n*5,
        'summary':summary,'train_target_frequency_slices':slices,'gpu_dpo_gradient_contracts':contracts,
        'gates':{'dpo_supply_passed':pairs/n>=cfg['signal_gate']['dpo_pair_fraction_min'],
            'grpo_supply_passed':varied/n>=cfg['signal_gate']['grpo_nonzero_reward_fraction_min'],
            'training_quality_passed':False,'training_launched':False,
            'scope':'supply screening;zero parameter updates,no dev/test improvements'},
        'sampling_active_seconds_per_role':times,
        'max_sampling_allocated_gib':max(x['resource']['peak_allocated_gib'] for x in items),
        'protected_files_unchanged':verify_guard(root),'independently_rescored_responses':n*5,
        'limits':cfg['limits']}
    write_json(root/'analysis/ANALYSIS.json',result);return result
