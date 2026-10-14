"""CPU verification and paired-image analysis for native-image OCR augmentation."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import csv
import json
import math
from pathlib import Path
from rarevlm_paths import SOURCE_ROOT
import statistics
import sys
sys.dont_write_bytecode=True
from vlm_deep.common import NORMALIZER,training_target
from vlm_probe.evaluate import content_sha,lock_file,now,private_path,read_manifest,sha_file,write_json
from .evaluate import (CONDITIONS,DEFAULT_CONFIG,REPO,adapter_binding,artifact_paths,
    dev_data,read_json,source_hashes,validate_config,validate_prediction)

LABELS=('unseen_0','rare_1_5','medium_6_20','frequent_gt20')
EXPECTED_COUNTS=dict(zip(LABELS,(687,342,171,417)))
COMPARISONS=('baseline','empty_template','shuffled_ocr')


def bucket(f): return LABELS[0] if f==0 else LABELS[1] if f<=5 else LABELS[2] if f<=20 else LABELS[3]


def prepare_slices(asset_root,output_root):
    path=asset_root/'prepared/manifest.json';manifest=read_manifest(path)
    train=manifest['splits']['train_all'];dev=manifest['splits']['dev']
    if len(train)!=32985 or len(dev)!=1617: raise ValueError('frozen training/dev count differs')
    frequency=Counter()
    for row in train:
        if row['target']!=training_target(row['answers']): raise ValueError('actual training target differs')
        frequency[NORMALIZER(row['target'])]+=1
    features=[]
    for row in dev:
        if row['target']!=training_target(row['answers']): raise ValueError('actual dev modal target differs')
        target=NORMALIZER(row['target']);f=frequency[target]
        features.append({'question_id':row['question_id'],'image_id':row['image_id'],
            'sft_target_normalized':target,'train_all_sft_target_qa_frequency':f,
            'sft_target_answer_frequency':bucket(f)})
    counts=dict(Counter(r['sft_target_answer_frequency'] for r in features))
    if counts!=EXPECTED_COUNTS: raise ValueError('frozen actual-target dev bins differ')
    lock={'schema_version':'rarevlm.ocr_context.slices.v1','manifest_sha256':manifest['manifest_sha256'],
        'manifest_file_sha256':sha_file(path),'normalizer_source_sha256':sha_file(SOURCE_ROOT/'vlm_deep/third_party/vqa_eval_metric.py'),
        'training_target_source_sha256':sha_file(SOURCE_ROOT/'vlm_deep/common.py'),'features':features,
        'slice_counts':counts,'train_all_questions':32985,'dev_questions':1617,'dev_images':1024,
        'frequency_sha256':content_sha(dict(sorted(frequency.items()))),
        'definition':'verified actual train_all raw-modal target,normalized and counted once per QA;dev target lookup',
        'limitations':'answer-text supervision proxy,not semantic class or pretraining frequency;slice differences confounded'}
    lock['slice_lock_sha256']=content_sha(lock);lock_file(output_root/'SLICE_LOCK.json',lock)
    return lock


def readiness(asset_root,output_root,config_path):
    config=validate_config(read_json(config_path));slices=prepare_slices(asset_root,output_root)
    ready={'schema_version':'rarevlm.ocr_context.analysis_ready.v1','config_sha256':sha_file(config_path),
        'analysis_source_sha256':sha_file(Path(__file__)),'slice_lock_sha256':slices['slice_lock_sha256'],
        'bootstrap_replicates':2000,'bootstrap_seed':20261013,'comparisons':list(COMPARISONS),
        'expected_predictions':6468,'scope':'CPU analysis prepared;no generation result or advancement implied'}
    write_json(output_root/'ANALYSIS_READY.json',ready);return ready


def read_cell(root,condition,rows,images,config_path,config):
    directory=root/'runs'/condition/'dev';path=directory/'predictions.jsonl'
    evaluation=read_json(directory/'EVALUATION.json')
    if not evaluation: return {'complete':False,'condition':condition,'path':str(directory)}
    lock=read_json(directory/'INPUT_LOCK.json')
    if not lock or lock.get('condition')!=condition or lock.get('ocr_config_sha256')!=sha_file(config_path) or lock.get('source_sha256')!=source_hashes():
        raise ValueError('cell source/config/condition lock differs')
    for key,value in {'batch_size':4,'seed':42,'split':'dev','numerical_path':'original_bf16',
        'precision':'bf16','quantization':'original','vision_precision':'bf16','visual_token_budget':1024,
        'max_new_tokens':32,'do_sample':False,'attention_implementation':'sdpa',
        'input_truncation':False,'tf32_matmul':False,'tf32_cudnn':False}.items():
        if lock.get(key)!=value: raise ValueError(f'cell fixed contract differs:{key}')
    training_path=Path(lock['training_binding']['path'])
    if sha_file(training_path)!=lock['training_binding']['sha256']: raise ValueError('fixed training result changed')
    adapter=training_path.parent/'adapter'
    files={p.name:sha_file(p) for p in sorted(adapter.iterdir()) if p.is_file()}
    if files!=lock['adapter_files_sha256']: raise ValueError('fixed adapter changed')
    paths=artifact_paths(config,root)
    for key,value in lock['prepared_binding'].items():
        if Path(value['path'])!=paths[key] or sha_file(paths[key])!=value['sha256']: raise ValueError('prepared input/detector changed')
    data=path.read_bytes()
    if not data.endswith(b'\n'): raise ValueError('completed prediction cache has interrupted tail')
    records=[json.loads(line) for line in data.splitlines()]
    if len(records)!=len(rows): raise ValueError('completed prediction count differs')
    for index,(record,row) in enumerate(zip(records,rows)): validate_prediction(record,row,images,lock,index,len(rows))
    score=100*statistics.mean(r['score'] for r in records)
    if (evaluation.get('status')!='completed' or evaluation['predictions_sha256']!=sha_file(path)
        or evaluation['input_lock_sha256']!=content_sha(lock) or not math.isclose(score,evaluation['score_percent'],abs_tol=1e-10,rel_tol=0)):
        raise ValueError('completed score/cache hash differs')
    return {'complete':True,'condition':condition,'score_percent':score,'rows':records,'lock':lock,
        'evaluation':evaluation,'provenance':{'prediction_file':str(path),'prediction_sha256':sha_file(path),
        'evaluation_sha256':sha_file(directory/'EVALUATION.json'),'input_lock_sha256':content_sha(lock)}}


def paired_interval(before,after,features,label,seed,reference):
    import numpy as np
    clusters=defaultdict(lambda:[0.,0]);pairs=[]
    for left,right,feature in zip(before,after,features):
        if (left['question_id'],left['image_id'])!=(right['question_id'],right['image_id']) or left['question_id']!=feature['question_id']:
            raise ValueError('unpaired question/image')
        if label!='overall' and feature['sft_target_answer_frequency']!=label: continue
        delta=right['score']-left['score'];clusters[left['image_id']][0]+=delta;clusters[left['image_id']][1]+=1
        pairs.append((left['score'],right['score']))
    if not pairs: raise ValueError('declared slice empty')
    matrix=np.asarray(list(clusters.values()),dtype=float);rng=np.random.default_rng(seed);values=[]
    for _ in range(2000):
        sample=matrix[rng.integers(0,len(matrix),size=len(matrix))].sum(axis=0);values.append(100*sample[0]/sample[1])
    return {'comparison':f'predicted_ocr_minus_{reference}','reference_condition':reference,'candidate_condition':'predicted_ocr',
        'slice':label,'questions':len(pairs),'image_clusters':len(matrix),
        'reference_score_percent':100*statistics.mean(p[0] for p in pairs),
        'candidate_score_percent':100*statistics.mean(p[1] for p in pairs),
        'delta_pp':100*statistics.mean(p[1]-p[0] for p in pairs),
        'image_cluster_bootstrap_95_pp':np.quantile(values,[.025,.975]).tolist(),
        'bootstrap_replicates':2000,'bootstrap_seed':seed,
        'inference_role':'single predeclared primary contrast' if reference=='baseline' and label=='overall' else 'secondary/slice exploratory unadjusted',
        'uncertainty':'fixed existing checkpoint on historically explored internal dev;resample image clusters retaining same-image qualifying QA;secondary intervals unadjusted;not seed stability/equivalence'}


def gate(contrasts,config):
    expected={'overall_min_delta_vs_baseline_pp':.5,'primary_ci_lower_min_pp':0.,
        'rare_min_delta_vs_baseline_pp':0.,'frequent_min_delta_vs_baseline_pp':-.5}
    declared=config.get('dev_gate',{})
    for key,value in expected.items():
        if declared.get(key)!=value: raise ValueError(f'frozen advancement threshold differs:{key}')
    lookup={(r['reference_condition'],r['slice']):r for r in contrasts}
    overall=lookup[('baseline','overall')]
    checks={'overall_delta_ge_0_5_pp':overall['delta_pp']>=.5,
        'primary_image_cluster_ci_lower_gt_zero':overall['image_cluster_bootstrap_95_pp'][0]>0.,
        'rare_point_delta_ge_zero':lookup[('baseline','rare_1_5')]['delta_pp']>=0.,
        'frequent_point_delta_ge_minus_0_5_pp':lookup[('baseline','frequent_gt20')]['delta_pp']>=-.5}
    return {'passed':all(checks.values()),'checks':checks,'thresholds':expected,
        'primary_comparison':'predicted_ocr minus native baseline overall',
        'decision':'review complete dev evidence before separately declared replication/formal confirmation' if all(checks.values()) else 'retain complete result;no automatic tuning or expansion',
        'automatic_new_training':False,'automatic_formal_evaluation':False,
        'interpretation':'quality advancement screen for one checkpoint;rare/frequent point guards are not noninferiority tests;dev passed is not formal-set or seed-stability confirmation'}


def preparation_cost(paths,config):
    context=read_json(paths['context_lock']); full=read_json(paths['input_contract'])
    ready=read_json(paths['ocr_manifest'])
    ocr_status=read_json(Path(ready['raw_path']).parent/'OCR_STATUS.json') or {}
    observations={k:v for k,v in ready.items() if any(t in k.lower() for t in ('time','timing','seconds','resource','recognition','detection','model','source','package'))}
    detector=read_json(private_path(config['detector_predictions_path']))
    image_ids=set(context['images'])
    detector_seconds=sum(float(detector['images'][iid]['detection_seconds']) for iid in image_ids)
    recognition_seconds=float(ready['recognition_predict_api_seconds_saved'])
    return {'unique_images':1024,'questions':1617,
        'ocr_ready_artifact':{'path':str(paths['ocr_manifest']),'sha256':sha_file(paths['ocr_manifest'])},
        'historical_cached_detector_predict_api_seconds_once_per_unique_image':detector_seconds,
        'historical_cached_detector_seconds_per_unique_image':detector_seconds/1024,
        'new_recognition_predict_api_seconds_once_per_unique_image':recognition_seconds,
        'new_recognition_seconds_per_unique_image':recognition_seconds/1024,
        'ocr_predict_component_seconds_amortized_per_dev_qa':(detector_seconds+recognition_seconds)/1617,
        'offline_ocr_observations':observations,
        'recognizer_last_invocation_observations':ocr_status,
        'context_build_observations':{k:v for k,v in context.items() if any(t in k.lower() for t in ('seconds','timing','wall'))},
        'full_cpu_input_validation_observations':{k:v for k,v in full.items() if any(t in k.lower() for t in ('seconds','timing','wall'))},
        'scope':'detector historical cache and new recognizer once per unique image;reuse across QA/arms;amortized component sums exclude decode/crop/model setup and are not fresh request latency;offline preparation separate from cached VLM phases;no online speedup claim'}


def analyze(asset_root,output_root,config_path):
    config=validate_config(read_json(config_path));slices=prepare_slices(asset_root,output_root)
    rows,images,_,_=dev_data(asset_root)
    cells={c:read_cell(output_root,c,rows,images,config_path,config) for c in CONDITIONS}
    result={'schema_version':'rarevlm.ocr_context.analysis.v1','updated_at':now(),
        'complete':all(c['complete'] for c in cells.values()),'completed_cells':sum(c['complete'] for c in cells.values()),
        'expected_cells':4,'expected_predictions':6468,'config_sha256':sha_file(config_path),
        'analysis_source_sha256':sha_file(Path(__file__)),'slice_lock_sha256':slices['slice_lock_sha256'],
        'scope':'historically explored internal dev;single fixed existing checkpoint;unchanged native image with four textual context conditions;no official evaluation'}
    if not result['complete']:
        result.update(status='waiting_predictions',pending=[c['condition'] for c in cells.values() if not c['complete']])
        write_json(output_root/'analysis/ANALYSIS.json',result);return result
    baseline=cells['baseline']['lock']
    for cell in cells.values():
        for key in ('ocr_config_sha256','reference_config_sha256','source_sha256','packages','model_revision',
            'model_provenance_sha256','processor_and_metadata_sha256','actual_parameter_dtypes','manifest_sha256',
            'adapter_files_sha256','training_binding','prepared_binding','batch_size','question_order_sha256'):
            if cell['lock'].get(key)!=baseline.get(key): raise ValueError(f'common four-condition contract differs:{key}')
    # Actual source RGB and float32 processor image tensors, not merely caps,
    # must match per question across all arms. Text/input lengths may differ.
    for index in range(len(rows)):
        candidate=cells['predicted_ocr']['rows'][index]
        for condition in ('baseline','empty_template','shuffled_ocr'):
            control=cells[condition]['rows'][index]
            for key in ('question_id','image_id','question_sha256','question_prompt_sha256','image_grid_thw',
                'visual_tokens','per_image_visual_tokens','counted_image_tokens','image_slots','input_truncated',
                'rgb_frame_sha256','pixel_values_sha256','pixel_values_shape','batch_start','batch_size'):
                if candidate.get(key) is None or candidate.get(key)!=control.get(key):
                    raise ValueError(f'four-arm native-image contract differs:{key}')
    seed=config.get('bootstrap_seed',config.get('evaluation',{}).get('bootstrap_seed',20261013))
    replicates=config.get('bootstrap_replicates',config.get('evaluation',{}).get('bootstrap_replicates',2000))
    if seed!=20261013 or replicates!=2000: raise ValueError('frozen exploratory bootstrap differs')
    contrasts=[paired_interval(cells[reference]['rows'],cells['predicted_ocr']['rows'],slices['features'],label,
        seed+index*5+offset,reference) for index,reference in enumerate(COMPARISONS)
        for offset,label in enumerate(('overall',*LABELS))]
    summaries={}
    for condition,cell in cells.items():
        slice_scores={label:{'questions':sum(f['sft_target_answer_frequency']==label for f in slices['features']),
            'score_percent':100*statistics.mean(r['score'] for r,f in zip(cell['rows'],slices['features']) if f['sft_target_answer_frequency']==label)} for label in LABELS}
        summaries[condition]={'score_percent':cell['score_percent'],'slices':slice_scores,
            'resource':cell['evaluation']['resource'],'hardware':cell['lock']['hardware'],'provenance':cell['provenance']}
    prepared_paths=artifact_paths(config,output_root)
    primary=next(c for c in contrasts if c['reference_condition']=='baseline' and c['slice']=='overall')
    result.update(status='complete',validated_predictions=6468,condition_summary=summaries,contrasts=contrasts,
        primary_contrast=primary,gate=gate(contrasts,config),preparation_cost=preparation_cost(prepared_paths,config),
        limitations=['actual native RGB/pixel/grid/visual tokens fixed across four conditions;total text length and compute can differ',
            'one predeclared overall primary contrast;remaining14 pointwise exploratory intervals not multiplicity-adjusted',
            'historically explored internal dev;public pretraining overlap unknown',
            'single fixed existing checkpoint;no new training or seed stability claim',
            'OCR predictions are auxiliary image-derived text;no answer/target/question-based OCR selection',
            'cached OCR VLM phase cost excludes fresh detector/recognizer preparation;no online speedup claim',
            'rare/frequent gate guards use point estimates;not equivalence or noninferiority evidence'])
    directory=output_root/'analysis';directory.mkdir(parents=True,exist_ok=True)
    write_json(directory/'ANALYSIS.json',result)
    with (directory/'paired_contrasts.csv').open('w',newline='') as stream:
        fields=['comparison','slice','questions','image_clusters','reference_score_percent','candidate_score_percent','delta_pp','ci_low_pp','ci_high_pp']
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for contrast in contrasts:
            writer.writerow({**{k:contrast[k] for k in fields if k in contrast},
                'ci_low_pp':contrast['image_cluster_bootstrap_95_pp'][0],'ci_high_pp':contrast['image_cluster_bootstrap_95_pp'][1]})
    findings=['# Native-image OCR context results','',
        f'All4 cells complete;6,468 predictions independently validated. Quality gate passed: {result["gate"]["passed"]}.','',
        '| condition | dev score (%) | generation QA/s | cached-context VLM phases QA/s |','|---|---:|---:|---:|']
    for c,s in summaries.items():
        r=s['resource'];findings.append(f'| {c} | {s["score_percent"]:.6f} | {r["generation_questions_per_second"]:.6f} | {r["context_preprocess_h2d_generation_questions_per_second"]:.6f} |')
    findings+=['',f'Primary predicted OCR minus baseline: {primary["delta_pp"]:+.6f} pp, paired image-cluster95% interval {primary["image_cluster_bootstrap_95_pp"]}.',
        '','Secondary comparisons: OCR minus empty template and shuffled image OCR. Frequency slices use actual train_all SFT target exposure.',
        '','One predeclared primary contrast on historically explored dev. Other intervals are exploratory and unadjusted. No seed-stability, equivalence, noninferiority, or online latency claim.',
        '','Actual native image pixels and visual tokens match across all4 arms;total prompt lengths and compute need not match.',
        '','OCR is offline preparation reported separately from cached-context VLM phases.',
        '','No automatic additional training, official benchmark evaluation, or resume update.']
    (directory/'findings.md').write_text('\n'.join(findings)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset-root',type=Path,required=True);parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=DEFAULT_CONFIG);parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    args.asset_root=private_path(args.asset_root);args.output_root=private_path(args.output_root);args.config=private_path(args.config)
    result=readiness(args.asset_root,args.output_root,args.config) if args.prepare_only else analyze(args.asset_root,args.output_root,args.config)
    print(json.dumps({k:result.get(k) for k in ('complete','completed_cells','validated_predictions','gate','analysis_source_sha256')},ensure_ascii=False),flush=True)


if __name__=='__main__': main()
