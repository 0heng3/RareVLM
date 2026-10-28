"""Same-image coordinate switch check, using fixed official region pairs and zero updates."""
import argparse
import json
from pathlib import Path
import re
import sys
import time

from rarevlm_paths import PROJECT_ROOT,SOURCE_ROOT
from vlm_deep.common import encode,content_sha,sha_file,write_json

ROOT=PROJECT_ROOT.parents[1]/'outputs/vlm_region_binding_probe_v1'
POOL=PROJECT_ROOT.parents[1]/'outputs/vlm_region_binding_data_v1'
PIXEL=PROJECT_ROOT.parents[1]/'outputs/vlm_region_pixel_sft_v1'
ROLES=('initial','pixel')


def read(path):return json.loads(Path(path).read_text())


def prepare():
    from vlm_region_ocr_sft_v1.study import CONFIG
    from vlm_probe.evaluate import processor_for
    if (ROOT/'PLAN.json').exists():raise FileExistsError('Already prepared')
    pool=read(POOL/'PAIR_CANDIDATES.json');pairs=pool['pairs'][:64]
    if len(pairs)!=64:raise ValueError('Need the fixed first64 hash-ordered eligible pairs')
    c=read(CONFIG);base=read(c['baseline_config']);processor=processor_for(Path(base['model_root']),read(base['reference_config']),1024)
    rows=[];images={p['source_image_id']:pool['images'][p['source_image_id']] for p in pairs}
    for pair in pairs:
        a,b=pair['a'],pair['b']
        skeleton=lambda q:re.sub(r'\[[0-9, ]+\]','[BOX]',q['question'])
        if skeleton(a)!=skeleton(b) or a['prompt_suffix']!=b['prompt_suffix'] or a['image_id']!=b['image_id']:
            raise ValueError('Only numeric region coordinates may change within an image pair')
        for name,q in [('a',a),('b',b)]:
            item=encode(processor,q,Path(images[q['image_id']]['path']),False)
            grid=item['image_grid_thw'][0].tolist()
            if [grid[2]*14,grid[1]*14]!=q['actual_processed_dimensions']:raise ValueError('Actual pixel coordinate frame differs')
            rows.append({**q,'pair_id':pair['pair_id'],'pair_side':name,
                'expected_grid_thw':grid,'expected_input_tokens':int(item['input_ids'].shape[1]),
                'prefix_sha256':content_sha(item['input_ids'].tolist())})
    write_json(ROOT/'DATA.json',{'rows':rows,'images':images})
    final=read(PIXEL/'runs/pixel/train/TRAIN_COMPLETE.json')
    adapters={'initial':c['initial_adapter'],'pixel':final['adapter']}
    files=[Path(__file__).resolve(),ROOT/'DATA.json',POOL/'PAIR_CANDIDATES.json',
        PROJECT_ROOT/'docs/VLM_REGION_BINDING_PROBE_V1_PROTOCOL.md',
        SOURCE_ROOT/'vlm_region_ocr_sft_v1/study.py',SOURCE_ROOT/'vlm_probe/evaluate.py',
        SOURCE_ROOT/'vlm_attention_scope_repair_v1/generation.py']
    files += [Path(path)/name for path in adapters.values() for name in ('adapter_config.json','adapter_model.safetensors')]
    write_json(ROOT/'PLAN.json',{'training_updates':0,'pairs':64,'qa_per_model':128,
        'selection':'First64 of predeclared hash-ordered full training candidate pool; no error-based selection',
        'scope':'Training-image coordinate-switch diagnosis, not heldout or resume-ready capability',
        'adapters':adapters,'lora':{'initial':final['initial_lora_sha256'],'pixel':final['final_lora_sha256']},
        'generation':'Same FP32 adapter/head, BF16 base, raw greedy/max32/batch4/budget1024',
        'primary':'Both official words correct on the same image; changed answers alone do not imply correctness',
        'active_cap_seconds':240,'files':{str(p):sha_file(p) for p in files},
        'images':{im['path']:im['sha256'] for im in images.values()}})


def worker(role):
    import torch
    from vlm_region_ocr_sft_v1.study import actor
    from vlm_attention_scope_repair_v1.generation import raw_greedy_config
    from vlm_probe.evaluate import evaluate_batch
    from vlm_human_sft.evaluate import lora_hash
    plan=read(ROOT/'PLAN.json')
    for path,value in {**plan['files'],**plan['images']}.items():
        if sha_file(path)!=value:raise ValueError('Frozen coordinate-switch input changed')
    model,processor,_=actor(plan['adapters'][role]);before=lora_hash(model)
    if before!=plan['lora'][role]:raise ValueError('Checkpoint identity differs')
    model.eval();model.config.use_cache=True;model.generation_config=raw_greedy_config(model.generation_config,32)
    data=read(ROOT/'DATA.json');rows=data['rows'];directory=ROOT/role;directory.mkdir(exist_ok=True)
    path=directory/'PREDICTIONS.json';predictions=read(path) if path.exists() else []
    if len(predictions)%4 or [q['question_id'] for q in predictions]!=[q['question_id'] for q in rows[:len(predictions)]]:raise ValueError('Cache grouping differs')
    begun=time.monotonic()
    with torch.inference_mode():
        for start in range(len(predictions),128,4):
            chunk=rows[start:start+4]
            answers,_=evaluate_batch(model,processor,chunk,data['images'],ROOT,next(model.parameters()).device,
                1024,processor.tokenizer.convert_tokens_to_ids('<|image_pad|>'),32)
            for q,a in zip(chunk,answers):
                if a['image_grid_thw']!=q['expected_grid_thw'] or a['input_tokens']!=q['expected_input_tokens']:
                    raise ValueError('Actual inference prefix differs')
                predictions.append({'question_id':q['question_id'],'pair_id':q['pair_id'],'side':q['pair_side'],
                    'image_id':q['image_id'],'target':q['target'],**a,
                    'raw_exact':a['prediction']==q['target'],
                    'casefold_exact':a['prediction'].casefold()==q['target'].casefold()})
            write_json(path,predictions);write_json(directory/'PROGRESS.json',{'status':'running','completed_qa':len(predictions),'expected_qa':128,'active_seconds':time.monotonic()-begun})
            if time.monotonic()-begun>plan['active_cap_seconds']:raise TimeoutError('Read-only coordinate-switch budget reached')
    if lora_hash(model)!=before:raise ValueError('Read-only check changed adapter')
    paired=[]
    for start in range(0,128,2):
        a,b=predictions[start:start+2]
        if a['pair_id']!=b['pair_id'] or a['side']!='a' or b['side']!='b':raise ValueError('Region-pair order differs')
        paired.append({'pair_id':a['pair_id'],'both_correct':a['casefold_exact'] and b['casefold_exact'],
            'same_answer':a['prediction'].casefold()==b['prediction'].casefold(),
            'a_correct':a['casefold_exact'],'b_correct':b['casefold_exact'],
            'a_answer_is_b_target':a['prediction'].casefold()==b['target'].casefold(),
            'b_answer_is_a_target':b['prediction'].casefold()==a['target'].casefold()})
    write_json(directory/'PAIRS.json',paired)
    result={'status':'completed','training_updates':0,'adapter_unchanged':True,'lora_sha256':before,
        'qa_correct':sum(q['casefold_exact'] for q in predictions),'qa':128,'pairs':64,
        **{key:sum(q[key] for q in paired) for key in ('both_correct','same_answer','a_correct','b_correct','a_answer_is_b_target','b_answer_is_a_target')},
        'active_seconds':time.monotonic()-begun}
    write_json(directory/'COMPLETE.json',result);write_json(directory/'PROGRESS.json',result)


def analyze():
    from vlm_sft_resolution.experiment import paired_interval
    counts={role:read(ROOT/role/'COMPLETE.json') for role in ROLES}
    values={role:read(ROOT/role/'PAIRS.json') for role in ROLES}
    transform=lambda role:[{'question_id':q['pair_id'],'image_id':q['pair_id'],'score':float(q['both_correct'])} for q in values[role]]
    result={'status':'completed','training_updates':0,'scope':read(ROOT/'PLAN.json')['scope'],'models':counts,
        'pair_exact_weight_effect':paired_interval(transform('initial'),transform('pixel'),6000,20261020),
        'next':'Use this baseline for a separately registered same-image multi-region training study; no automatic longer training'}
    write_json(ROOT/'COMPLETION.json',result)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare',action='store_true');parser.add_argument('--role',choices=ROLES);args=parser.parse_args()
    if args.prepare:prepare();return
    if args.role:worker(args.role);return
    from research_runtime.queue import run_queue
    run_queue(ROOT/'queue',[{'key':role,'command':[sys.executable,'-u',str(Path(__file__).resolve()),'--role',role],
        'completion_marker':str(ROOT/role/'COMPLETE.json')} for role in ROLES],{'plan_sha256':sha_file(ROOT/'PLAN.json')})
    analyze()


if __name__=='__main__':main()
