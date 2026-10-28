"""New native-QA/OCR-evidence cohort; fixed-model headroom, zero training."""
import argparse
from collections import Counter
import html
import json
import os
from pathlib import Path
import re
import sys
import time
import unicodedata
from urllib.parse import quote
from rarevlm_paths import PROJECT_ROOT,SOURCE_ROOT
from vlm_deep.common import content_sha,sha_file,write_json
from vlm_grpo_iteration_v1.common import actor,read

BASE=PROJECT_ROOT.parents[1]/'outputs'
ROOT=BASE/'vlm_native_ocr_correction_v1'
DATA=ROOT/'revision2/DATA.json'
PROTOCOL=PROJECT_ROOT/'docs/VLM_NATIVE_OCR_PROBE_V1_PROTOCOL.md'
RESULT=PROJECT_ROOT/'docs/VLM_NATIVE_OCR_PROBE_V1_RESULTS.md'
PAGE=PROJECT_ROOT/'docs/learning/原生文字问答_双源监督与纠错空间.html'
MODELS=('initial','qv_hard','mlp_kl')


def literal(value):
    value=unicodedata.normalize('NFKC',str(value)).casefold().strip()
    return re.sub(r'^[^a-z0-9]+|[^a-z0-9]+$','',value)


def prepare():
    import torch
    from vlm_deep.common import encode
    from vlm_probe.evaluate import inference_row,processor_for
    from vlm_region_ocr_sft_v1.data import bounds,excludes_other_word_centers
    if (ROOT/'PLAN.json').exists():raise FileExistsError('Already frozen')
    torch.set_num_threads(2);started=time.monotonic();old=read(ROOT/'CANDIDATE_ROWS.json');inventory=read(ROOT/'INVENTORY.json')
    parent=read(BASE/'vlm_reference_replay_v1/PLAN.json')
    annotation_path=BASE/'vlm_region_ocr_sft_v1/source_data/TextOCR_0.1_train.json';annotations=read(annotation_path)
    rows=[];removed=[]
    for q in old['rows']:
        target=literal(q['target']);transcription=literal(q['ocr_evidence']['original_transcription'])
        if len(q['target'].split())!=1 or not re.fullmatch(r'[a-z0-9]{3,24}',target) or not re.search(r'[a-z]',target) or target!=transcription:
            removed.append({'question_id':q['question_id'],'reason':'literal_target_not_exactly_evidenced','target':q['target'],'official_ocr':q['ocr_evidence']['original_transcription']});continue
        im=old['images'][q['image_id']];others=[]
        for aid in annotations['imgToAnns'][q['image_id']]:
            if aid==q['ocr_annotation_id']:continue
            try:others.append(bounds(annotations['anns'][aid]['points'],im['width'],im['height']))
            except (ValueError,KeyError,TypeError):continue
        if not excludes_other_word_centers(q['ocr_evidence']['box_xyxy_original_pixels'],others):
            removed.append({'question_id':q['question_id'],'reason':'another_word_center_inside_evidence_box'});continue
        rows.append({**q,'stratum':'singleword','evaluation_scored':True})
    if len(rows)<1024:raise ValueError('Insufficient fixed native cohort; do not lower source filters')
    images={q['image_id']:old['images'][q['image_id']] for q in rows}
    processor=processor_for(Path(parent['model_root']),read(parent['reference_config']),1024)
    for index,q in enumerate(rows):
        sample=encode(processor,inference_row(q),Path(images[q['image_id']]['path']),False)
        grid=sample['image_grid_thw'][0].tolist();box=q['ocr_evidence']['box_xyxy_original_pixels'];im=images[q['image_id']]
        effective=[(box[2]-box[0])*grid[2]*14/im['width'],(box[3]-box[1])*grid[1]*14/im['height']]
        if max(abs(a-b) for a,b in zip(effective,q['ocr_evidence']['effective_region_pixels']))>1e-6:raise ValueError('Actual processor resize differs from source visibility proof')
        q.update(expected_prefix_tokens=int(sample['input_ids'].shape[1]),expected_grid=grid,inference_input_sha256=content_sha(sample['input_ids'].tolist()),actual_evidence_region_pixels=effective)
        if (index+1)%64==0:write_json(ROOT/'CPU_PROGRESS.json',{'status':'encoding_revision2','completed':index+1,'expected':len(rows),'active_seconds':time.monotonic()-started})
    write_json(DATA,{'rows':rows,'images':images})
    receipt={'status':'passed','source_v1_qa':len(old['rows']),'revision2_qa':len(rows),'removed':removed,
             'frequency_counts':dict(Counter(q['frequency_slice'] for q in rows)),'native_questions_targets_answers_unchanged':True,
             'bbox_used_only_for_selection_and_display':True,'actual_input_full_original_image':True,'actual_resize_matches':True,
             'new_model_calls':0,'training_updates':0,'active_seconds':time.monotonic()-started,'source_inventory_sha256':sha_file(ROOT/'INVENTORY.json'),
             'source_annotations_sha256':inventory['annotation_sha256'],'data_sha256':sha_file(DATA)}
    write_json(ROOT/'revision2/PREPARE_RECEIPT.json',receipt)
    old_hard=read(BASE/'vlm_reference_replay_v1/runs/hard_replay/train/TRAIN_COMPLETE.json')
    mlp=read(BASE/'vlm_mlp_reference_v1/runs/soft_kl/train/TRAIN_COMPLETE.json')
    models={'initial':{'adapter':parent['initial_adapter'],'lora_sha256':parent['initial_lora_sha256']},
            'qv_hard':{'adapter':old_hard['adapter'],'lora_sha256':old_hard['final_lora_sha256']},
            'mlp_kl':{'adapter':mlp['adapter'],'lora_sha256':mlp['final_lora_sha256']}}
    paths=[Path(__file__).resolve(),PROJECT_ROOT/'tools/inventory_native_ocr_v1.py',PROTOCOL,ROOT/'INVENTORY.json',ROOT/'CANDIDATE_ROWS.json',DATA,ROOT/'revision2/PREPARE_RECEIPT.json',BASE/'vlm_reference_replay_v1/PLAN.json',Path(parent['reference_config']),Path(parent['model_root'])/'config.json',BASE/'vlm_reference_replay_v1/runs/hard_replay/train/TRAIN_COMPLETE.json',BASE/'vlm_mlp_reference_v1/runs/soft_kl/train/TRAIN_COMPLETE.json']
    paths += [SOURCE_ROOT/p for p in ('vlm_grpo_iteration_v1/common.py','vlm_fit_positive_control_v1/study.py','vlm_fp32_readout_v1/model.py','vlm_fp32_adapter/model.py','vlm_runtime_metadata_v1/model.py','vlm_deep/common.py','vlm_deep/evaluate.py','vlm_sft/common.py','vlm_probe/evaluate.py','vlm_sampled_signal_v1/trajectory.py','vlm_attention_scope_repair_v1/generation.py','vlm_human_sft/evaluate.py','research_runtime/queue.py')]
    for m in models.values():paths += [Path(m['adapter'])/n for n in ('adapter_config.json','adapter_model.safetensors')]
    plan={'status':'prepared','models':models,'files':{str(p):sha_file(p) for p in paths},'images':{v['path']:v['sha256'] for v in images.values()},
          'base_weight_files':parent['base_weight_files'],'packages':parent['packages'],'qa_per_model':len(rows),'new_generations':3*len(rows),
          'formal_optimizer_updates':0,'new_adapters':0,'automatic_training':False,'minimum_future_error_pool':128,'minimum_future_unique_targets':96,
          'generation_cap_seconds_per_shard':900,'original_sft_exposure':'all cohort images were in original train_all; excluded from current extra-training/held cohorts',
          'interpretation':'New diagnostic cohort; never replace prior failed acceptance gates.'}
    write_json(ROOT/'PLAN.json',plan);render()
    RESULT.write_text(f'# 原生文字问答：准备完成\n\n原1458条中严格证据筛选后{len(rows)}条，三模型共{3*len(rows)}次生成，0训练/新adapter。等待共享GPU准入。\n\n[协议](VLM_NATIVE_OCR_PROBE_V1_PROTOCOL.md) · [原图与证据](learning/原生文字问答_双源监督与纠错空间.html)\n')
    print('PREPARED_NATIVE_OCR',{'qa':len(rows),'removed':len(removed),'generations':3*len(rows),'cpu_seconds':time.monotonic()-started},flush=True)


def verify():
    plan=read(ROOT/'PLAN.json')
    for path,digest in {**plan['files'],**plan['images']}.items():
        if sha_file(path)!=digest:raise ValueError('Frozen diagnostic input changed: '+path)
    for path,metadata in plan['base_weight_files'].items():
        stat=Path(path).stat()
        if metadata!={'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns}:raise ValueError('Base storage changed')
    return plan


def batches(data,shard):
    return [data['rows'][i:i+4] for j,i in enumerate(range(0,len(data['rows']),4)) if j%2==shard]


def worker(role,shard):
    import torch
    from vlm_fit_positive_control_v1.study import generated
    from vlm_human_sft.evaluate import lora_hash
    plan=verify();data=read(DATA);cfg=plan['models'][role];model,processor,_=actor(adapter=cfg['adapter'])
    model.eval();model.config.use_cache=True
    if lora_hash(model)!=cfg['lora_sha256']:raise ValueError('Fixed model identity mismatch')
    groups=batches(data,shard);flat=[q for g in groups for q in g];directory=ROOT/'runs'/role/f'shard_{shard}';path=directory/'PREDICTIONS.json';preds=read(path) if path.exists() else []
    if len(preds)>len(flat) or (len(preds)!=len(flat) and len(preds)%4) or [q['question_id'] for q in preds]!=[q['question_id'] for q in flat[:len(preds)]]:raise ValueError('Whole-batch resume differs')
    prior=read(directory/'PROGRESS.json').get('active_seconds',0.) if (directory/'PROGRESS.json').exists() else 0.;begin=time.monotonic();torch.cuda.reset_peak_memory_stats()
    for chunk in groups[(len(preds)+3)//4:]:
        answers=generated(model,processor,chunk,data)
        for q,a in zip(chunk,answers):
            if a['input_sha256']!=q['inference_input_sha256']:raise ValueError('Actual prompt bytes differ')
            preds.append({**a,'frequency_slice':q['frequency_slice'],'literal_exact':literal(a['prediction'])==literal(q['target'])})
        temporary=path.with_suffix('.tmp');write_json(temporary,preds);temporary.replace(path)
        active=prior+time.monotonic()-begin;peak=torch.cuda.max_memory_allocated()/2**30
        write_json(directory/'PROGRESS.json',{'status':'fixed_model_generation','model':role,'shard':shard,'completed_qa':len(preds),'expected_qa':len(flat),'active_seconds':active,'peak_allocated_gib':peak,'optimizer_updates':0})
        if active>900 or peak>65:raise RuntimeError('Diagnostic budget exceeded')
    if lora_hash(model)!=cfg['lora_sha256']:raise ValueError('Read-only inference changed weights')
    write_json(directory/'COMPLETE.json',{'status':'completed','qa':len(preds),'optimizer_updates':0,'active_seconds':prior+time.monotonic()-begin})


def analyze():
    from vlm_grpo_iteration_v1.study import paired
    from vlm_deep.common import NORMALIZER
    data=read(DATA);records={};scores={}
    for role in MODELS:
        lookup={q['question_id']:q for shard in (0,1) for q in read(ROOT/'runs'/role/f'shard_{shard}/PREDICTIONS.json')}
        rows=[lookup[q['question_id']] for q in data['rows']];records[role]=rows;write_json(ROOT/'runs'/role/'PREDICTIONS.json',rows)
        scores[role]={'vqa_percent':100*sum(q['vqa_score'] for q in rows)/len(rows),'full':sum(q['vqa_score']>=.999999 for q in rows),
                      'zero_or_low':sum(q['vqa_score']<=.3+1e-8 for q in rows),'literal_exact_percent':100*sum(q['literal_exact'] for q in rows)/len(rows)}
    comparisons={r:paired(records['initial'],records[r]) for r in MODELS if r!='initial'}
    raw={q['question_id']:q for q in data['rows']};errors=[];targets=Counter()
    for p in sorted(records['initial'],key=lambda q:content_sha(['native_ocr_error_v1',q['question_id']])):
        q=raw[p['question_id']];target=NORMALIZER(q['target']);pred=NORMALIZER(p['prediction'])
        if p['vqa_score']>.3+1e-8 or not p['terminated'] or target in pred.split() or targets[target]>=2:continue
        targets[target]+=1;errors.append({**q,'initial_prediction':p})
    readiness=len(errors)>=128 and len(targets)>=96
    write_json(ROOT/'ERROR_CANDIDATES.json',{'rows':errors,'images':data['images'],'source':'original-model low-score natural-EOS errors with official double-source evidence','not_expert_relabeling':True})
    result={'status':'completed','scores':scores,'comparisons':comparisons,'candidate_errors':len(errors),'unique_error_targets':len(targets),
            'training_pool_count_gate_passed':readiness,'automatic_training':False,'new_optimizer_updates':0,'new_adapters':0,'goal_complete':False,
            'meaning':'Data headroom and fixed-model transfer diagnostic; no new training gain and no rescue of prior failed gates.',
            'next':'Review new error supervision before any separate registered training; do not fill from dev or lower count gates.'}
    write_json(ROOT/'ANALYSIS.json',result)
    table='|固定模型|VQA%|满分题|字面exact%|\n|---|---:|---:|---:|\n'
    for role,s in scores.items():table+=f'|{role}|{s["vqa_percent"]:.6f}|{s["full"]}/{len(data["rows"])}|{s["literal_exact_percent"]:.6f}|\n'
    RESULT.write_text('# 原生文字问答：完整诊断\n\n'+table+f'\n独立文字证据下的低分候选{len(errors)}条、不同target{len(targets)}，数量准入={readiness}。0新训练/新adapter。原SFT已见这些图，但当前额外微调未用；新诊断不能回写旧失败通过。\n\n[协议](VLM_NATIVE_OCR_PROBE_V1_PROTOCOL.md) · [原图、实际回答与证据](learning/原生文字问答_双源监督与纠错空间.html)\n')
    render(result);write_json(ROOT/'COMPLETION.json',result);return result


def render(result=None):
    data=read(DATA);relative=lambda p:quote(os.path.relpath(p,PAGE.parent),safe='/._-');cards=[]
    cases=sorted(data['rows'],key=lambda q:content_sha(['native_ocr_preview_v1',q['question_id']]))[:8]
    for case in cases:
        im=data['images'][case['image_id']];x0,y0,x1,y1=case['ocr_evidence']['box_xyxy_original_pixels'];body=''
        for role in MODELS:
            path=ROOT/'runs'/role/'PREDICTIONS.json'
            if path.exists():
                q=next(q for q in read(path) if q['question_id']==case['question_id'])
                body+='<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in (role,q['prediction'],q['vqa_score']))+'</tr>'
        overlay=f'left:{100*x0/im["width"]:.4f}%;top:{100*y0/im["height"]:.4f}%;width:{100*(x1-x0)/im["width"]:.4f}%;height:{100*(y1-y0)/im["height"]:.4f}%'
        cards.append(f'<article><h2>q{case["question_id"]}</h2><figure><img loading="lazy" src="{relative(im["path"])}" alt="原图，红框为官方OCR展示证据"><span class="box" style="{overlay}"></span></figure><p>{html.escape(case["question"])}</p><p>原目标：{html.escape(case["target"])}；官方OCR：{html.escape(case["ocr_evidence"]["original_transcription"])}</p><p>模型实际输入中文字区域大小：{html.escape(str(case["actual_evidence_region_pixels"]))}像素；红框仅页面展示，模型未接收框或OCR文字。</p><table><tr><th>固定模型</th><th>真实回答</th><th>VQA</th></tr>{body}</table></article>')
    status='准备/运行中，0训练，无新能力提升。' if result is None else html.escape(str(result['scores']))+'；0新训练，不能当新增模型提分。'
    PAGE.write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>原生文字监督</title><style>body{max-width:1100px;margin:28px auto;padding:0 18px;font:17px/1.7 system-ui;background:#f8fafc;color:#1e293b}article{padding:18px;margin:18px 0;background:white;border:1px solid #cbd5e1}figure{position:relative;margin:0;width:100%}img{display:block;width:100%;height:auto}.box{position:absolute;border:2px solid #ef4444;box-sizing:border-box;pointer-events:none}td,th{border:1px solid #cbd5e1;padding:6px}table{border-collapse:collapse}</style><h1>原问题、原图与独立文字证据</h1><p>'+status+'</p><p>8/10原问答共识 + 独立TextOCR字面一致 + 唯一目标标注 + 实际resize可见尺寸。仅用于筛选，不改变模型提示或输入像素。</p><p>VQA官方计分与字面证据核验分开；冠词删除不能用来证明整段文字可见。revision2排除了不完整证据及框内其他词中心，原标签与旧盘点保留。</p><p>所有图都在原SFT训练白名单内、当前额外训练和held图像/像素外。数据筛选未使用新模型输出，例子在新预测前固定；高共识+OCR不等于专家全量重标注。</p><p><a href="'+relative(PROTOCOL)+'">预登记</a> · <a href="'+relative(Path(__file__).resolve())+'">实际选择和生成代码</a></p>'+''.join(cards)+'</html>')


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--worker');args=p.parse_args()
    if args.prepare:prepare();return
    if args.worker:
        role,index=args.worker.rsplit(':',1)
        if role not in MODELS:raise ValueError('Unknown model')
        if index not in ('0','1'):raise ValueError('Unknown whole-batch shard')
        worker(role,int(index));return
    from research_runtime.queue import run_queue
    verify();jobs=[{'key':f'{role}_{i}','command':[sys.executable,'-u',str(Path(__file__).resolve()),'--worker',f'{role}:{i}'],
                   'completion_marker':str(ROOT/'runs'/role/f'shard_{i}/COMPLETE.json')} for role in MODELS for i in (0,1)]
    run_queue(ROOT/'queue',jobs,{'plan_sha256':sha_file(ROOT/'PLAN.json')});result=analyze();print('NATIVE_OCR_COMPLETE',result,flush=True)


if __name__=='__main__':main()
