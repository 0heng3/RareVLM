"""Zero-update transfer diagnosis: fixed new training cohort, full view versus oracle ROI."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time

from rarevlm_paths import PROJECT_ROOT,SOURCE_ROOT
from vlm_deep.common import NORMALIZER,content_sha,encode,sha_file,write_json

ROOT=PROJECT_ROOT.parents[1]/'outputs/vlm_pixel_transfer_diagnosis_v1'
PARENT=PROJECT_ROOT.parents[1]/'outputs/vlm_region_pixel_sft_v1'
COORD=PROJECT_ROOT.parents[1]/'outputs/vlm_region_coordinates_v1'
ROLES=('initial','pixel')
VIEWS=('full','oracle_crop')


def read(path):return json.loads(Path(path).read_text())


def bounded(prediction,target):
    return bool(re.search(r'(?<!\w)'+re.escape(target.casefold())+r'(?!\w)',prediction.casefold()))


def prepare():
    from PIL import Image
    from vlm_region_coordinates_v1.data import question
    from vlm_region_pixel_sft_v1.study import verify
    from vlm_probe.evaluate import processor_for
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
    verify()
    if (ROOT/'PLAN.json').exists():raise FileExistsError('Already registered; inspect existing results')
    manifest=read(PARENT/'data/PIXEL_TRAIN.json')
    excluded={q['question_id'] for q in read(COORD/'DATA.json')['variants']['pixel_dimensions']}
    candidates=[q for q in manifest['splits']['train_all'] if q['study_source']=='region_aux' and q['source_question_id'] not in excluded]
    key=lambda q:hashlib.sha256(('pixel-transfer-v1:20261020:'+q['source_question_id']).encode()).hexdigest()
    selected=sorted(candidates,key=key)[:256]
    if len(selected)!=256 or len({q['image_id'] for q in selected})!=256:raise ValueError('Need 256 fixed unique new training images')
    c=read(PROJECT_ROOT/'configs/vlm_region_pixel_sft_v1.json');base=read(c['baseline_config']);ref=read(base['reference_config'])
    processor=processor_for(Path(base['model_root']),ref,1024)
    ROOT.mkdir(parents=True,exist_ok=True);(ROOT/'crops').mkdir(exist_ok=True)
    images={};rows={view:[] for view in VIEWS};begun=time.monotonic()
    for number,q in enumerate(selected,1):
        im=manifest['images'][q['image_id']];path=Path(im['path'])
        if sha_file(path)!=im['sha256']:raise ValueError('Full image changed')
        with Image.open(path) as raw:
            image=raw.convert('RGB')
            if image.size!=(im['width'],im['height']):raise ValueError('Image dimensions changed')
            x0,y0,x1,y1=q['box_xyxy_original_pixels']
            box=[max(0,math.floor(x0)-2),max(0,math.floor(y0)-2),min(im['width'],math.ceil(x1)+2),min(im['height'],math.ceil(y1)+2)]
            crop=image.crop(box);native_crop_size=list(crop.size)
            scale=max(1.0,28/min(crop.size))
            if scale>1:crop=crop.resize((math.ceil(crop.width*scale),math.ceil(crop.height*scale)),Image.Resampling.LANCZOS)
            crop_path=ROOT/'crops'/(q['image_id']+'.png');crop.save(crop_path)
            crop_width,crop_height=crop.size
        images[q['image_id']]=im
        crop_id=q['image_id']+':oracle_crop'
        images[crop_id]={'path':str(crop_path),'sha256':sha_file(crop_path),'width':crop_width,'height':crop_height,
            'source_image_id':q['image_id'],'source_image_sha256':im['sha256'],'source_crop_xyxy_pixels':box,
            'native_crop_size':native_crop_size,'saved_crop_size':[crop_width,crop_height],
            'preprocessor_legal_min_side':28,'legal_pre_resize_scale_xy':[crop_width/native_crop_size[0],crop_height/native_crop_size[1]]}
        h,w=smart_resize(crop_height,crop_width,factor=28,min_pixels=ref['min_pixels'],max_pixels=1024*28*28)
        full={**q,'source_image_id':q['image_id'],'origin_question_id':q['source_question_id']}
        cropped={'question_id':q['question_id']+':oracle_crop','image_id':crop_id,
            'source_image_id':q['image_id'],'origin_question_id':q['source_question_id'],
            'target':q['target'],'prompt_suffix':q['prompt_suffix'],'annotation_id':q['annotation_id'],
            'source_annotation_xyxy_pixels':q['box_xyxy_original_pixels'],
            'box_xyxy_original_pixels':[0,0,crop_width,crop_height],'box_xyxy_processed_pixels':[0,0,w,h],
            'coordinate_frame':'pixel_dimensions',
            'question':question([0,0,w,h],w,h,'pixel_dimensions'),'actual_processed_dimensions':[w,h]}
        for view,row in [('full',full),('oracle_crop',cropped)]:
            item=encode(processor,row,Path(images[row['image_id']]['path']),False)
            grid=item['image_grid_thw'][0].tolist();actual=[grid[2]*14,grid[1]*14]
            if actual!=row['actual_processed_dimensions']:raise ValueError('Actual processor dimensions differ from the supplied region frame')
            row={**row,'expected_grid_thw':grid,'expected_input_tokens':int(item['input_ids'].shape[1]),
                'expected_visual_tokens':int(item['image_grid_thw'].prod())//4,
                'cpu_prefix_sha256':content_sha(item['input_ids'].tolist())}
            if row['expected_visual_tokens']>1024 or row['expected_input_tokens']>2048:raise ValueError('Diagnostic input exceeds budget')
            rows[view].append(row)
        if number%16==0:write_json(ROOT/'DATA_PROGRESS.json',{'status':'preparing','completed_images':number,'expected_images':256,'seconds':time.monotonic()-begun})
    write_json(ROOT/'DATA.json',{'images':images,'views':rows})
    files=[Path(__file__).resolve(),ROOT/'DATA.json',PROJECT_ROOT/'docs/VLM_PIXEL_TRANSFER_DIAGNOSIS_V1_PROTOCOL.md',
        PARENT/'COMPLETION.json',PARENT/'STUDY_INPUT_LOCK.json',COORD/'COMPLETION.json',
        SOURCE_ROOT/'vlm_region_ocr_sft_v1/study.py',SOURCE_ROOT/'vlm_probe/evaluate.py',
        SOURCE_ROOT/'vlm_attention_scope_repair_v1/generation.py']
    adapters={'initial':c['initial_adapter'],'pixel':str(PARENT/'runs/pixel/train/adapter')}
    files += [Path(p)/name for p in adapters.values() for name in ('adapter_model.safetensors','adapter_config.json')]
    final=read(PARENT/'runs/pixel/train/TRAIN_COMPLETE.json')
    write_json(ROOT/'PLAN.json',{'qa_per_view':256,'views':list(VIEWS),'roles':list(ROLES),'training_updates':0,
        'selection':'SHA256(pixel-transfer-v1:20261020:source_question_id), exclude previous fixed64, no score inputs',
        'scope':'All 256 images are training images; no heldout improvement or unseen-data claim',
        'crop_policy':'Official word xyxy, outward rounding +2 original pixels, clamped; full crop is the declared box',
        'crop_limit':'Oracle region, isolation and effective resolution change together; not deployable or a pure localization causal effect',
        'primary_metric':'casefold exact; raw exact and bounded-target presence separate; no score replacement',
        'adapters':adapters,'expected_lora':{'initial':final['initial_lora_sha256'],'pixel':final['final_lora_sha256']},
        'numerics':'Same BF16 base/FP32 adapter and frozen head; raw greedy/max32/batch4/budget1024',
        'preparation_revision':2,'worker_active_cap_seconds':600,'bootstrap_replicates':6000,'bootstrap_seed':20261020,
        'automatic_training':False,'source_cpu_seconds':time.monotonic()-begun,
        'files':{str(p):sha_file(p) for p in files},
        'image_files':{str(Path(im['path'])):im['sha256'] for im in images.values()}})
    write_json(ROOT/'DATA_PROGRESS.json',{'status':'completed','images':256,'registered_inference_encodings':512,'seconds':time.monotonic()-begun})


def worker(role):
    import torch
    from vlm_region_ocr_sft_v1.study import actor
    from vlm_attention_scope_repair_v1.generation import raw_greedy_config
    from vlm_probe.evaluate import evaluate_batch
    from vlm_human_sft.evaluate import lora_hash
    plan=read(ROOT/'PLAN.json')
    for path,value in {**plan['files'],**plan['image_files']}.items():
        if sha_file(path)!=value:raise ValueError('Frozen diagnostic input changed: '+path)
    model,processor,_=actor(plan['adapters'][role]);before=lora_hash(model)
    if before!=plan['expected_lora'][role]:raise ValueError('Unexpected checkpoint')
    model.eval();model.config.use_cache=True;model.generation_config=raw_greedy_config(model.generation_config,32)
    data=read(ROOT/'DATA.json');begun=time.monotonic();resources={}
    with torch.inference_mode():
        for view in VIEWS:
            rows=data['views'][view];directory=ROOT/role/view;directory.mkdir(parents=True,exist_ok=True)
            path=directory/'PREDICTIONS.json';predictions=read(path) if path.exists() else []
            if len(predictions)%4 or [q['question_id'] for q in predictions]!=[q['question_id'] for q in rows[:len(predictions)]]:raise ValueError('Cache grouping differs')
            resources[view]=[]
            for start in range(len(predictions),256,4):
                chunk=rows[start:start+4]
                answers,resource=evaluate_batch(model,processor,chunk,data['images'],ROOT,
                    next(model.parameters()).device,1024,processor.tokenizer.convert_tokens_to_ids('<|image_pad|>'),32)
                resources[view].append(resource)
                for q,a in zip(chunk,answers):
                    if a['image_grid_thw']!=q['expected_grid_thw'] or a['input_tokens']!=q['expected_input_tokens'] or a['visual_tokens']!=q['expected_visual_tokens']:
                        raise ValueError('Actual inference input differs from the registered CPU encoding')
                    predictions.append({'question_id':q['question_id'],'origin_question_id':q['origin_question_id'],
                        'source_image_id':q['source_image_id'],'image_id':q['image_id'],'target':q['target'],**a,
                        'raw_exact':a['prediction']==q['target'],'casefold_exact':a['prediction'].casefold()==q['target'].casefold(),
                        'bounded_target_present':bounded(a['prediction'],q['target'])})
                write_json(path,predictions)
                write_json(ROOT/role/'PROGRESS.json',{'status':'running','view':view,'completed_qa':len(predictions),'expected_qa':256,'active_seconds':time.monotonic()-begun})
                if time.monotonic()-begun>plan['worker_active_cap_seconds']:raise TimeoutError('Diagnostic budget reached')
            write_json(directory/'NEW_BATCH_RESOURCES.json',resources[view])
    if lora_hash(model)!=before:raise ValueError('Read-only diagnosis changed adapter')
    write_json(ROOT/role/'COMPLETE.json',{'status':'completed','training_updates':0,'active_seconds':time.monotonic()-begun,
        'adapter_unchanged':True,'lora_sha256':before,'predictions_per_view':256})
    write_json(ROOT/role/'PROGRESS.json',read(ROOT/role/'COMPLETE.json'))


def analyze():
    from vlm_sft_resolution.experiment import paired_interval
    plan=read(ROOT/'PLAN.json');records={role:{view:read(ROOT/role/view/'PREDICTIONS.json') for view in VIEWS} for role in ROLES}
    def compare(before,after):
        if [(q['origin_question_id'],q['source_image_id']) for q in before]!=[(q['origin_question_id'],q['source_image_id']) for q in after]:raise ValueError('Unpaired comparison')
        transform=lambda rows:[{'question_id':q['origin_question_id'],'image_id':q['source_image_id'],'score':float(q['casefold_exact'])} for q in rows]
        fixed=[(a,b) for a,b in zip(before,after) if not a['casefold_exact'] and b['casefold_exact']]
        broken=[(a,b) for a,b in zip(before,after) if a['casefold_exact'] and not b['casefold_exact']]
        return {**paired_interval(transform(before),transform(after),plan['bootstrap_replicates'],plan['bootstrap_seed']),
            'corrected':len(fixed),'broken':len(broken),
            'corrected_target_already_present_before':sum(a['bounded_target_present'] for a,b in fixed),
            'corrected_without_prior_bounded_target':sum(not a['bounded_target_present'] for a,b in fixed),
            'broken_target_still_present_after':sum(b['bounded_target_present'] for a,b in broken)}
    result={'status':'completed','training_updates':0,'scope':plan['scope'],'crop_limit':plan['crop_limit'],
        'counts':{role:{view:{key:sum(q[key] for q in rows) for key in ('raw_exact','casefold_exact','bounded_target_present')} for view,rows in views.items()} for role,views in records.items()},
        'weight_effect':{view:compare(records['initial'][view],records['pixel'][view]) for view in VIEWS},
        'oracle_crop_effect':{role:compare(records[role]['full'],records[role]['oracle_crop']) for role in ROLES},
        'next':'Use source-grounded error types to choose a new hypothesis; no automatic optimizer or longer training'}
    write_json(ROOT/'RESULTS.json',result);write_json(ROOT/'COMPLETION.json',result)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare',action='store_true');parser.add_argument('--role',choices=ROLES);args=parser.parse_args()
    if args.prepare:prepare();return
    if args.role:worker(args.role);return
    from research_runtime.queue import run_queue
    run_queue(ROOT/'queue',[{'key':role,'command':[sys.executable,'-u',str(Path(__file__).resolve()),'--role',role],
        'completion_marker':str(ROOT/role/'COMPLETE.json')} for role in ROLES],{'plan_sha256':sha_file(ROOT/'PLAN.json')})
    analyze()


if __name__=='__main__':main()
