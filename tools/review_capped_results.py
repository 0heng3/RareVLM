"""Independent lineage/quota review after the frozen capped study completes."""
import json
from pathlib import Path
import numpy as np
from rarevlm_paths import PROJECT_ROOT,SOURCE_ROOT
from vlm_probe.evaluate import now,sha_file,write_json


def main():
    config=PROJECT_ROOT/'configs/clip_retrieval_capped_v1.json';cfg=json.loads(config.read_text());root=Path(cfg['output_root'])
    complete=json.loads((root/'COMPLETION.json').read_text());result=json.loads((root/'analysis/ANALYSIS.json').read_text())
    if complete['status']!='completed' or complete['analysis_sha256']!=sha_file(root/'analysis/ANALYSIS.json'):raise ValueError('Incomplete study')
    lock=json.loads((root/'STUDY_INPUT_LOCK.json').read_text())
    if sha_file(config)!=lock['config_sha256']:raise ValueError('Config changed')
    if sha_file(PROJECT_ROOT/'docs/CLIP_RETRIEVAL_CAPPED_V1_PROTOCOL.md')!=lock['protocol_sha256']:raise ValueError('Protocol changed')
    for name,digest in lock['sources'].items():
        if sha_file(SOURCE_ROOT/name)!=digest:raise ValueError('Frozen source changed')
    guard=json.loads((root/'FROZEN_GUARD.json').read_text())
    if any(sha_file(Path(p))!=digest for p,digest in guard['protected_sha256'].items()):raise ValueError('Protected historical/resume file changed')
    records=[]
    for seed in cfg['training_seeds']:
        b=root/'runs/mixed_distill'/f'seed{seed}';c=root/'runs/capped_distill'/f'seed{seed}'
        counts_b=np.load(b/'negative_occurrences.npy');counts_c=np.load(c/'negative_occurrences.npy')
        old=Path(cfg['feature_root'])/'runs/mixed_hard_negatives'/f'seed{seed}'/'negative_occurrences.npy'
        h=np.load(c/'epoch_hard_counts.npy');r=np.load(c/'epoch_random_counts.npy')
        if h.shape!=(10,28000) or h.max()>8 or not np.array_equal((h+r).sum(0),counts_c):raise ValueError('Capped accounting failed')
        if not (h.sum(1)+r.sum(1)==875*64).all():raise ValueError('Incomplete per-epoch budget')
        first_b=json.loads((b/'FIRST_GPU_POOL.json').read_text());first_c=json.loads((c/'FIRST_GPU_POOL.json').read_text())
        if any(first_b[k]!=first_c[k] for k in ['anchor_ids','hard_ids','random_ids']):raise ValueError('Inactive quota changes first pool')
        records.append({'seed':seed,'uncapped_negative_counts_exactly_match_historical_ce_mining':bool(np.array_equal(counts_b,np.load(old))),
            'historical_mining_counts_sha256':sha_file(old),'first_gpu_pool_B_C_exact':True,
            'quota_max':int(h.max()),'hard_slots_per_epoch':h.sum(1).tolist(),'random_slots_per_epoch':r.sum(1).tolist(),
            'negative_exposure_max_B':int(counts_b.max()),'negative_exposure_max_C':int(counts_c.max())})
    observations={}
    for phase in ['training','dev']+(['test'] if result['test'] else []):
        o=json.loads((root/(phase+'_queue')/'STATUS.json').read_text())
        observations[phase]={'status':o['status'],'query_failures':o['resource_query_failures'],
            'contention_observations':sum(j['contention_observations'] for j in o['jobs'].values()),
            'jobs':{key:{k:j.get(k) for k in ['pid','gpu_uuid','admitted_at','ended_at','state']} for key,j in o['jobs'].items()}}
    for split in ['dev']+(['test'] if result['test'] else []):
        if len(result[split]['independent_top10_checks'])!=9:raise ValueError('Missing rescore contracts')
    report={'status':'passed','reviewed_at':now(),'review_source_sha256':sha_file(Path(__file__)),
        'analysis_sha256':sha_file(root/'analysis/ANALYSIS.json'),'scientific_source_config_protocol':'unchanged',
        'historical_and_formal_resume_guard':'unchanged','protected_files':len(guard['protected_sha256']),
        'exposure_contracts':records,'resource_observations':observations,
        'limits':['old CE mining reproduction is an observed diagnostic,not a new independent baseline',
                  'same cap does not measure semantic false negatives;time not hardware-exclusive or online']}
    write_json(root/'analysis/EXPOSURE_REVIEW.json',report)
    print('REVIEW_PASSED',{'old_mining_exact':[r['uncapped_negative_counts_exactly_match_historical_ce_mining'] for r in records],
          'quota_max':[r['quota_max'] for r in records],'resource_observations':{k:{f:v[f] for f in ['query_failures','contention_observations']} for k,v in observations.items()}})


if __name__=='__main__':main()
