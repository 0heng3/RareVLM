"""Independently rescore saved Top10 with Python scalars; no neural prediction."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from vlm_probe.evaluate import now,sha_file,write_json
from clip_retrieval_quality.prepare import verify_guard


def independent_vectors(ranks,direction):
    output={f'{direction}_R@{k}':[] for k in [1,5,10]};output[f'{direction}_nDCG@10']=[]
    discount=[1/math.log2(j+2) for j in range(10)]
    ideal=sum(discount[:5]) if direction=='I2T' else 1.
    for q,order in enumerate(ranks.tolist()):
        relevant=[int(idx)//5==q if direction=='I2T' else int(idx)==q//5 for idx in order]
        for k in [1,5,10]:output[f'{direction}_R@{k}'].append(float(any(relevant[:k])))
        output[f'{direction}_nDCG@10'].append(sum(float(r)*d for r,d in zip(relevant,discount))/ideal)
    return {k:np.asarray(v,dtype=np.float64) for k,v in output.items()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);a=p.parse_args()
    cfg=json.loads(a.config.read_text());root=Path(cfg['output_root']);analysis=json.loads((root/'analysis/ANALYSIS.json').read_text())
    checked=[]
    for split in ['dev','test']:
        result=analysis[split]
        if result is None:continue
        for condition in result['summaries']:
            for seed in cfg['training_seeds']:
                d=root/'evaluation'/f'{condition}_seed{seed}'/split
                cell=json.loads((d/'EVALUATION.json').read_text())
                if cell['per_query_sha256']!=sha_file(d/'per_query.npz'):raise ValueError('Saved predictions changed')
                with np.load(d/'per_query.npz') as saved:
                    for direction in ['T2I','I2T']:
                        shape=(5000,10) if direction=='T2I' else (1000,10)
                        assert saved[direction.lower()+'_top10'].shape==shape
                        for metric,vec in independent_vectors(saved[direction.lower()+'_top10'],direction).items():
                            assert np.allclose(vec,saved[metric],rtol=0,atol=1e-6),metric
                            assert abs(100*float(vec.mean())-cell['metrics_percent'][metric])<2e-5,metric
                checked.append({'split':split,'condition':condition,'seed':seed,'six_recalls_and_two_ndcg_verified':True})
    guard=verify_guard(root);lock=json.loads((root/'STUDY_INPUT_LOCK.json').read_text())
    from rarevlm_paths import SOURCE_ROOT
    for name,digest in lock['sources'].items():assert sha_file(SOURCE_ROOT/name)==digest,name
    failed_branches=[b for b,c in analysis['dev']['comparisons'].items() if not c['gate']['passed']]
    for b in failed_branches:
        assert not any((root/'evaluation'/f'{b}_{m}4_seed{s}/test').exists() for m in ['random','score'] for s in cfg['training_seeds'])
    write_json(root/'analysis/INDEPENDENT_REVIEW.json',{'status':'passed','completed_at':now(),
        'checked_cells':checked,'protected_files_unchanged':guard,'source_hashes_unchanged':True,
        'failed_branches_no_new_test':failed_branches,'new_neural_predictions':0,
        'scope':'Python scalar Recall/NDCG from actual saved Top10;original branch gates and analysis preserved'})
    print('INDEPENDENT_REVIEW_PASSED',len(checked),'cells,',guard,'protected files')


if __name__=='__main__':main()
