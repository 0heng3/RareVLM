"""Verify actual training and paired generated predictions against locked images."""
import argparse
import json
import math
from pathlib import Path
from .common import consensus_score,content_sha,read_manifest,sha_file,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--asset-root',type=Path,required=True);p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path(__file__).resolve().parents[1]/'configs/vlm_sft_textvqa.json')
    args=p.parse_args();config=json.loads(args.config.read_text())
    m=read_manifest(args.asset_root/'prepared/manifest.json',verify_images=True)
    train=json.loads((args.run_root/'train/TRAIN_RESULT.json').read_text())
    assert train['manifest_sha256']==m['manifest_sha256'] and train['config_sha256']==content_sha(config)
    assert train['epochs_completed']==config['epochs'] and train['world_size']==config['expected_world_size']
    assert train['optimizer_steps_per_rank']==config['training_images']//(config['expected_world_size']*config['gradient_accumulation'])
    assert train['frozen_parameter_sha256_before']==train['frozen_parameter_sha256_after'] and train['frozen_parameters_unchanged']
    assert train['history'][-1]['skipped_updates_across_ranks']==0
    assert all('lora_' in name and 'visual' not in name for name in train['trainable_names'])
    for name,digest in train['adapter_files_sha256'].items():assert sha_file(args.run_root/'train/adapter'/name)==digest
    lookup={r['image_id']:r for r in m['splits']['final']}
    for method in ['base','adapted']:
        d=args.run_root/method;result=json.loads((d/'EVALUATION.json').read_text())
        assert sha_file(d/'predictions.json')==result['predictions_sha256']
        rows=json.loads((d/'predictions.json').read_text())
        assert len(rows)==len(lookup) and {r['image_id'] for r in rows}==set(lookup)
        assert result['manifest_sha256']==m['manifest_sha256'] and result['config_sha256']==content_sha(config)
        scores=[]
        for r in rows:
            reference=lookup[r['image_id']]
            assert r['question_id']==reference['question_id']
            score=consensus_score(r['prediction'],reference['answers'])
            assert math.isclose(score,r['consensus_score'],abs_tol=1e-12);scores.append(score)
        assert math.isclose(sum(scores)/len(scores)*100,result['vqa_consensus_percent'],abs_tol=1e-10)
    record={'passed':True,'image_hashes_verified':sum(len(v) for v in m['splits'].values()),
            'paired_final_predictions_verified':2*len(lookup),'completed_optimizer_steps':train['optimizer_steps_per_rank'],
            'frozen_base_weights_verified':True,'language_lora_only':True,
            'manifest_sha256':m['manifest_sha256'],'source_data_filters_recorded':True}
    write_json(args.run_root/'VERIFICATION.json',record);print(json.dumps(record),flush=True)


if __name__=='__main__':main()
