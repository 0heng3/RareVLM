"""Compare fixed-input generative runs and export only compact public metrics."""
import argparse
import json
from pathlib import Path
from .common import sha_file,write_json


def main():
    import numpy as np
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--asset-root',type=Path,required=True);p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--public-output',type=Path)
    args=p.parse_args()
    base_dir=args.run_root/'base';adapted_dir=args.run_root/'adapted'
    base=json.loads((base_dir/'EVALUATION.json').read_text());adapted=json.loads((adapted_dir/'EVALUATION.json').read_text())
    train=json.loads((args.run_root/'train/TRAIN_RESULT.json').read_text())
    manifest=json.loads((args.asset_root/'prepared/manifest.json').read_text())
    image_transports={}
    for records in manifest['splits'].values():
        for row in records:
            name=row.get('image_transport','pinned Parquet embedded image')
            image_transports[name]=image_transports.get(name,0)+1
    keys=('manifest_sha256','config_sha256','model_provenance_sha256','n','generation','metric_scope')
    if any(base[k]!=adapted[k] for k in keys):raise ValueError('base and LoRA runs do not share final inputs and settings')
    for directory,result in [(base_dir,base),(adapted_dir,adapted)]:
        if sha_file(directory/'predictions.json')!=result['predictions_sha256']:raise ValueError('predictions changed')
    pbase=json.loads((base_dir/'predictions.json').read_text());plora=json.loads((adapted_dir/'predictions.json').read_text())
    if [r['image_id'] for r in pbase]!=[r['image_id'] for r in plora]:raise ValueError('unpaired final image order')
    delta=np.array([b['consensus_score']-a['consensus_score'] for a,b in zip(pbase,plora)])
    rng=np.random.default_rng(20261009)
    samples=np.array([delta[rng.integers(0,len(delta),len(delta))].mean()*100 for _ in range(2000)])
    summary={'scope':'single-seed, one-epoch engineering pilot on an image-disjoint TextVQA subset; compact declared normalization; not official full benchmark',
             'dataset':'TextVQA','model':'Qwen2.5-VL-3B-Instruct',
             'data_images':{'train':train['training_images'],'internal_validation':128,'final':base['n']},
             'train_seed':0,'epochs':train['epochs_completed'],'world_size':train['world_size'],
             'data_transport':manifest.get('transport','pinned Parquet'),
             'source_pool_rows':manifest.get('source_pool_rows'),
             'excluded_unusable_answer_rows':sum(len(p.get('excluded_unusable_answers',[])) for p in manifest.get('source_metadata',[])),
             'image_transport_counts':image_transports,
             'trainable_parameters':train['trainable_parameters'],'total_parameters':train['total_parameters'],
             'training_precision':train['training_precision'],'quantization':train.get('quantization'),
             'trainable_percent':100*train['trainable_parameters']/train['total_parameters'],
             'optimizer_steps':train['optimizer_steps_per_rank'],'effective_batch':train['effective_batch_size'],
             'frozen_weights_unchanged':train['frozen_parameters_unchanged'],
             'baseline_vqa_consensus_percent':base['vqa_consensus_percent'],
             'lora_vqa_consensus_percent':adapted['vqa_consensus_percent'],
             'baseline_any_answer_match_percent':base['normalized_any_answer_match_percent'],
             'lora_any_answer_match_percent':adapted['normalized_any_answer_match_percent'],
             'paired_delta_pp':float(delta.mean()*100),
             'conditional_paired_image_bootstrap_95_pp':np.quantile(samples,[.025,.975]).tolist(),
             'bootstrap_scope':'same fixed 256 final images; not training-seed or dataset-generalization uncertainty',
             'images_score_improved':int((delta>0).sum()),'images_score_degraded':int((delta<0).sum()),
             'images_score_equal':int((delta==0).sum()),
             'train_duration_seconds':train['duration_seconds'],'train_peak_gpu_allocated_gib':train['peak_gpu_allocated_gib'],
             'base_inference':base['timing'],'lora_inference':adapted['timing'],
             'manifest_sha256':base['manifest_sha256'],'config_sha256':base['config_sha256'],
             'model_provenance_sha256':base['model_provenance_sha256'],
             'execution_source_sha256':{name:train['source_sha256'][name] for name in ('train.py','model.py','common.py')},
             'report_source_sha256':sha_file(Path(__file__)),
             'runtime_environment':json.loads((args.asset_root/'ENVIRONMENT.json').read_text()) if (args.asset_root/'ENVIRONMENT.json').exists() else None,
             'result_files_sha256':{'TRAIN_RESULT.json':sha_file(args.run_root/'train/TRAIN_RESULT.json'),
                 'base/EVALUATION.json':sha_file(base_dir/'EVALUATION.json'),
                 'adapted/EVALUATION.json':sha_file(adapted_dir/'EVALUATION.json')},
             'model_and_dataset_weights_bundled':False}
    write_json(args.run_root/'SUMMARY.json',summary)
    if args.public_output:write_json(args.public_output,summary)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
