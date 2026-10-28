"""Fixed-training 64-QA coordinate check; zero optimizer updates, no heldout data."""
import argparse
import json
import math
from pathlib import Path
import sys
import time

from rarevlm_paths import PROJECT_ROOT, SOURCE_ROOT
from vlm_deep.common import NORMALIZER, sha_file, write_json
from .data import pixel_box, question

ROOT=PROJECT_ROOT.parents[1]/'outputs/vlm_region_coordinates_v1'
PARENT=PROJECT_ROOT.parents[1]/'outputs/vlm_region_ocr_sft_v1'
MODES=('normalized_dimensions','pixel_dimensions')


def read(path):return json.loads(Path(path).read_text())


def prepare():
    from vlm_region_ocr_sft_v1.study import verify
    verify()
    if (ROOT/'PLAN.json').exists():raise FileExistsError('Already registered; inspect existing status')
    source=read(PARENT/'data/REGION_TRAIN_PROBE.json')
    cached=read(PARENT/'initial/region_train_probe/PREDICTIONS.json')
    if [q['question_id'] for q in source['qa']]!=[q['question_id'] for q in cached]:raise ValueError('Original fixed cohort differs')
    data={'images':source['images'],'variants':{mode:[] for mode in MODES}}
    for q,old in zip(source['qa'],cached):
        image=source['images'][q['image_id']];width,height=old['resized_width'],old['resized_height']
        box=pixel_box(q['box_xyxy_original_pixels'],image['width'],image['height'],width,height)
        for mode in MODES:
            coordinates=q['box_xyxy_normalized_1000'] if mode=='normalized_dimensions' else box
            data['variants'][mode].append({**q,'question':question(coordinates,width,height,mode),
                'coordinate_frame':mode,'actual_processed_dimensions':[width,height],
                'expected_grid_thw':old['image_grid_thw'],'box_xyxy_processed_pixels':box})
    write_json(ROOT/'DATA.json',data)
    paths=[ROOT/'DATA.json',ROOT/'upstream/COORDINATE_EVIDENCE.json',
        PARENT/'initial/region_train_probe/PREDICTIONS.json',PARENT/'PRELAUNCH_INPUT_LOCK.json',
        PROJECT_ROOT/'docs/VLM_REGION_COORDINATES_V1_PROTOCOL.md',PROJECT_ROOT/'tests/test_vlm_region_coordinates_v1.py']
    paths+=list((SOURCE_ROOT/'vlm_region_coordinates_v1').glob('*.py'))
    write_json(ROOT/'PLAN.json',{'status':'prepared','qa_per_mode':64,'training_updates':0,
        'mode_order':list(MODES),'cohort':'Exact fixed original training probe, no new samples or heldout reading',
        'model':'Original SFT42, same numerical policy; no new model or optimizer',
        'generation':'raw greedy, max32, batch4, original full images, no crop or OCR answer input',
        'primary_metric':'casefold exact without VQA article/number/punctuation removal',
        'gate_for_considering_training':{'net_correct_gain_vs_both_normalized_controls':8,'paired_image_95_ci_lower_gt':0},
        'gate_scope':'Exploratory input evidence only; no heldout model-capability claim or automatic long training',
        'maximum_worker_seconds':300,'frozen_files':{str(p):sha_file(p) for p in paths}})


def worker(mode):
    import torch
    from vlm_region_ocr_sft_v1.study import actor, read as parent_read, CONFIG
    from vlm_attention_scope_repair_v1.generation import raw_greedy_config
    from vlm_probe.evaluate import evaluate_batch
    from vlm_human_sft.evaluate import lora_hash
    plan=read(ROOT/'PLAN.json')
    for path,value in plan['frozen_files'].items():
        if sha_file(path)!=value:raise ValueError('Frozen probe changed: '+path)
    data=read(ROOT/'DATA.json');rows=data['variants'][mode];directory=ROOT/mode;directory.mkdir(exist_ok=True)
    model,processor,_=actor(parent_read(CONFIG)['initial_adapter']);before=lora_hash(model)
    model.eval();model.config.use_cache=True;model.generation_config=raw_greedy_config(model.generation_config,32)
    predictions=[];begun=time.monotonic()
    with torch.inference_mode():
        for start in range(0,64,4):
            chunk=rows[start:start+4]
            answers,_=evaluate_batch(model,processor,chunk,data['images'],PARENT/'data',next(model.parameters()).device,
                1024,processor.tokenizer.convert_tokens_to_ids('<|image_pad|>'),32)
            for q,a in zip(chunk,answers):
                if a['image_grid_thw']!=q['expected_grid_thw'] or [a['resized_width'],a['resized_height']]!=q['actual_processed_dimensions']:
                    raise ValueError('Actual processor frame differs from the coordinates supplied to model')
                predictions.append({'question_id':q['question_id'],'image_id':q['image_id'],'target':q['target'],**a,
                    'raw_exact':a['prediction']==q['target'],'casefold_exact':a['prediction'].casefold()==q['target'].casefold(),
                    'normalized_correct':bool(a['prediction'].strip()) and NORMALIZER(a['prediction'])==NORMALIZER(q['target'])})
            write_json(directory/'PREDICTIONS.json',predictions)
            write_json(directory/'PROGRESS.json',{'status':'running','completed_qa':len(predictions),'expected_qa':64,'active_seconds':time.monotonic()-begun})
            if time.monotonic()-begun>plan['maximum_worker_seconds']:raise TimeoutError('Fixed coordinate probe budget reached')
    if lora_hash(model)!=before:raise ValueError('Read-only probe changed adapter')
    result={'status':'completed','qa':64,'training_updates':0,'active_seconds':time.monotonic()-begun,
        'lora_sha256':before,'adapter_unchanged':True,**{k:sum(q[k] for q in predictions) for k in ('raw_exact','casefold_exact','normalized_correct')}}
    write_json(directory/'COMPLETE.json',result);write_json(directory/'PROGRESS.json',result);print(json.dumps(result),flush=True)


def analyze():
    from vlm_sft_resolution.experiment import paired_interval
    records={mode:read(ROOT/mode/'PREDICTIONS.json') for mode in MODES}
    original=read(PARENT/'initial/region_train_probe/PREDICTIONS.json')
    records['original_normalized']=[{**q,'casefold_exact':q['prediction'].casefold()==q['target'].casefold()} for q in original]
    candidate=records['pixel_dimensions'];comparisons={};gates={}
    for before in ('original_normalized','normalized_dimensions'):
        rows=records[before]
        if [q['question_id'] for q in rows]!=[q['question_id'] for q in candidate]:raise ValueError('Unpaired coordinate cohort')
        corrected=sum(not a['casefold_exact'] and b['casefold_exact'] for a,b in zip(rows,candidate))
        broken=sum(a['casefold_exact'] and not b['casefold_exact'] for a,b in zip(rows,candidate))
        transformed=lambda values:[{'question_id':q['question_id'],'image_id':q['image_id'],'score':float(q['casefold_exact'])} for q in values]
        interval=paired_interval(transformed(rows),transformed(candidate),6000,20261020)
        comparisons[before]={'corrected':corrected,'broken':broken,'net_correct':corrected-broken,**interval}
        gates[before]=corrected-broken>=8 and interval['image_cluster_95_ci_pp'][0]>0
    result={'status':'completed','training_updates':0,'models_unchanged':True,
        'scores':{mode:{key:sum(q[key] for q in rows) for key in ('raw_exact','casefold_exact','normalized_correct')} for mode,rows in records.items()},
        'comparisons':comparisons,'input_evidence_gate_passed':all(gates.values()),
        'scope':'Fixed training-only input-policy diagnosis, not model capability improvement',
        'next':'Consider a separately registered pixel-coordinate training study only if input evidence gate passes; otherwise stop coordinate-only branch'}
    write_json(ROOT/'RESULTS.json',result);write_json(ROOT/'COMPLETION.json',result)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare',action='store_true');parser.add_argument('--mode',choices=MODES);args=parser.parse_args()
    if args.prepare:prepare();return
    if args.mode:worker(args.mode);return
    from research_runtime.queue import run_queue
    run_queue(ROOT/'queue',[{'key':mode,'command':[sys.executable,'-u','-m',__package__+'.study','--mode',mode],
        'completion_marker':str(ROOT/mode/'COMPLETE.json')} for mode in MODES],{'plan_sha256':sha_file(ROOT/'PLAN.json')})
    analyze()


if __name__=='__main__':main()
