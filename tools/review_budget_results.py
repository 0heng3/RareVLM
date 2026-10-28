"""Independent scalar ranking rescore plus source, pair and timing checks."""
import json
import math
from pathlib import Path
import numpy as np
from rarevlm_paths import PROJECT_ROOT,SOURCE_ROOT
from vlm_probe.evaluate import sha_file,now
from clip_retrieval_rerank.common import write_json


def main():
    config=PROJECT_ROOT/'configs/clip_retrieval_budget_v1.json';cfg=json.loads(config.read_text());root=Path(cfg['output_root'])
    done=json.loads((root/'COMPLETION.json').read_text());result=json.loads((root/'analysis/ANALYSIS.json').read_text())
    if done['status']!='completed' or done['analysis_sha256']!=sha_file(root/'analysis/ANALYSIS.json'):raise ValueError('Incomplete study')
    lock=json.loads((root/'STUDY_INPUT_LOCK.json').read_text())
    if sha_file(config)!=lock['config_sha256'] or sha_file(PROJECT_ROOT/'docs/CLIP_RETRIEVAL_BUDGET_V1_PROTOCOL.md')!=lock['protocol_sha256']:raise ValueError('Config/protocol changed')
    if any(sha_file(SOURCE_ROOT/n)!=digest for n,digest in lock['sources'].items()):raise ValueError('Frozen source changed')
    if any(sha_file(Path(p))!=digest for p,digest in lock['installed_transformers_sources'].items()):raise ValueError('Installed matcher source changed')
    guard=json.loads((root/'FROZEN_GUARD.json').read_text())['files']
    if any(sha_file(Path(p))!=digest for p,digest in guard.items()):raise ValueError('Historical/resume guard failed')
    if result['dev']['status']=='calibration_rejected_before_gpu':
        write_json(root/'analysis/INDEPENDENT_REVIEW.json',{'status':'passed','reviewed_at':now(),'review_source_sha256':sha_file(Path(__file__)),
            'analysis_sha256':sha_file(root/'analysis/ANALYSIS.json'),'prediction_cells':0,'protected_files':len(guard),
            'scope':'calibration rejection and source integrity only;no actual speed/quality prediction claim'})
        print('CALIBRATION_ONLY_REVIEW_PASSED');return
    records=[]
    for split in ['dev']+(['test'] if result['test'] else []):
        for condition in ['full','adaptive']:
            for seed in cfg['training_seeds']:
                d=root/'evaluation'/f'{condition}_seed{seed}'/split;marker=json.loads((d/'EVALUATION.json').read_text())
                if sha_file(d/'per_query.npz')!=marker['per_query_sha256']:raise ValueError('Saved prediction changed')
                with np.load(d/'per_query.npz') as z:
                    for direction in ['T2I','I2T']:
                        rankings=z[direction.lower()+'_top50'];expected=[]
                        ideal=sum(1/math.log2(j+2) for j in range(5 if direction=='I2T' else 1))
                        recall={k:[] for k in [1,5,10,50]}
                        for query,row in enumerate(rankings):
                            target=query if direction=='I2T' else query//5
                            relevant=[int(candidate)//5==target if direction=='I2T' else int(candidate)==target for candidate in row]
                            for k in recall:recall[k].append(any(relevant[:k]))
                            expected.append(sum((1/math.log2(j+2)) for j in range(10) if relevant[j])/ideal)
                        for k,value in recall.items():
                            if not np.array_equal(value,z[f'{direction}_R@{k}']):raise ValueError('Independent Recall mismatch')
                        if not np.allclose(expected,z[direction+'_nDCG@10'],atol=1e-12,rtol=0):raise ValueError('Independent nDCG mismatch')
                records.append({'split':split,'condition':condition,'seed':seed,'independent_full_top50_rescore':'passed'})
        plan=json.loads((root/'routes'/split/'PLAN.json').read_text())
        counts={variant:0 for variant in ['full','adaptive']}
        for role in range(cfg['pair_scoring_roles']):
            d=root/'matcher'/split/f'role{role}';worker=json.loads((d/'MATCH_COMPLETE.json').read_text())
            for variant in counts:
                counts[variant]+=worker['conditions'][variant]['pairs']
                chunks=d/variant/'chunks';record=worker['conditions'][variant]
                observed=0
                for name,digest in record['chunk_files'].items():
                    path=chunks/name;meta=json.loads(path.with_suffix('.json').read_text())
                    if sha_file(path)!=digest or meta['pairs']!=meta['end']-meta['begin']:raise ValueError('Actual quality chunk provenance changed')
                    observed+=meta['pairs']
                if observed!=record['pairs']:raise ValueError('Incomplete actual quality forwards')
            timing=json.loads((d/'TIMING.json').read_text())
            if timing['pair_batch_size']!=32 or len(timing['samples'])!=3:raise ValueError('Timing contract differs')
        if counts!={'full':plan['full_unique_pairs'],'adaptive':plan['adaptive_unique_pairs']}:raise ValueError('Actual unique pair budgets differ')
    write_json(root/'analysis/INDEPENDENT_REVIEW.json',{'status':'passed','reviewed_at':now(),'review_source_sha256':sha_file(Path(__file__)),
        'analysis_sha256':sha_file(root/'analysis/ANALYSIS.json'),'prediction_checks':records,'protected_files':len(guard),
        'scientific_sources_config_protocol':'unchanged','old_scientific_and_formal_resume':'unchanged',
        'actual_selected_pair_forward_and_timing_contract':'passed','limits':['cached-gallery offlinescope','historical data exposure','no new training']})
    print('REVIEW_PASSED',{'prediction_cells':len(records),'protected_files':len(guard)})


if __name__=='__main__':main()
