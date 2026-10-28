"""Annotation-assisted training-teacher feasibility; never a benchmark method."""
import argparse
import json
from pathlib import Path
import sys
import time

from rarevlm_paths import PROJECT_ROOT
from vlm_deep.common import NORMALIZER,content_sha,encode,sha_file,write_json,collate
from vlm_grpo_iteration_v1.common import read,chosen_sample

BASE=PROJECT_ROOT.parents[1]/'outputs'
ROOT=BASE/'vlm_crop_teacher_signal_v1'
SOURCE=BASE/'vlm_review_resolution_repair_v1'
PARENT=BASE/'vlm_visual_preference_v1'
PROTOCOL=PROJECT_ROOT/'docs/VLM_CROP_TEACHER_SIGNAL_V1_PROTOCOL.md'
ROLES=('start','full_dpo')


def prepare():
    import torch
    from PIL import Image
    from vlm_probe.evaluate import processor_for,inference_row
    if (ROOT/'PLAN.json').exists():raise FileExistsError('Teacher study already frozen')
    torch.set_num_threads(2);c=read(PARENT/'CONFIG.json');base=read(c['baseline_config'])
    processor=processor_for(Path(base['model_root']),read(base['reference_config']),1024)
    review=read(SOURCE/'REVIEW_CORRECTIONS.json');oldplan=read(SOURCE/'PLAN.json')
    originals={e['question']['question_id']:e for e in oldplan['entries']}
    selected=[e for e in review['entries'] if e['new_supported_by_visible_evidence']]
    if len(selected)!=7:raise ValueError('Require fixed7 reviewed candidates, not an expanded pool')
    olddata=read(BASE/'vlm_late_rank_matched_dpo_v1/data/MANIFEST.json')
    blocked={q['image_id'] for q in olddata['rows']}
    blocked.update(q['image_id'] for rows in olddata['evaluations'].values() for q in rows)
    images={};pairs=[];paths=[Path(__file__),PROTOCOL,SOURCE/'PLAN.json',SOURCE/'REVIEW_CORRECTIONS.json',PARENT/'CONFIG.json',PARENT/'STUDY_INPUT_LOCK.json',Path(c['baseline_config']),Path(base['reference_config']),BASE/'vlm_native_scope_v1/CONFIG.json']
    for record in selected:
        qid=record['question_id'];old=originals[qid];q=old['question'];source=Path(record['raw_image']);box=record['display_crop_box_xyxy']
        if q['image_id'] in blocked or sha_file(source)!=record['raw_sha256']:raise ValueError('Training-only isolation/image identity failed')
        raw=Image.open(source).convert('RGB');x0,y0,x1,y1=box
        if not (0<=x0<x1<=raw.width and 0<=y0<y1<=raw.height):raise ValueError('Invalid fixed teacher box')
        dest=ROOT/'teacher_inputs'/f'{qid}.png';dest.parent.mkdir(parents=True,exist_ok=True);crop=raw.crop(box);crop.save(dest)
        if Image.open(dest).convert('RGB').tobytes()!=crop.tobytes():raise ValueError('Native pixel crop changed')
        pair={'source_question_id':qid,'source_image_id':q['image_id'],'teacher_box_xyxy':box,'conditions':{}}
        for condition,path in [('full',source),('crop',dest)]:
            image_id=q['image_id']+':'+condition;row={**q,'question_id':qid+':'+condition,'image_id':image_id}
            images[image_id]={'path':str(path),'sha256':sha_file(path)}
            prefix=encode(processor,inference_row(row),path,False);sample=chosen_sample(processor,row,images)
            if sample['input_ids'].shape[1]>2048:raise ValueError('Training prefix exceeds fixed budget')
            row.update(expected_prefix_tokens=int(prefix['input_ids'].shape[1]),expected_grid=prefix['image_grid_thw'][0].tolist(),
                       inference_input_sha256=content_sha(prefix['input_ids'].tolist()),
                       chosen_input_sha256=content_sha(sample['input_ids'].tolist()),chosen_labels_sha256=content_sha(sample['labels'].tolist()))
            pair['conditions'][condition]=row;paths.append(path)
        pairs.append(pair)
    write_json(ROOT/'DATA.json',{'pairs':pairs,'images':images,'source_sft_images_seen':True,'training_teacher_oracle_not_benchmark':True})
    models={'start':{'adapter':c['initial_adapter'],'lora_sha256':read(PARENT/'STUDY_INPUT_LOCK.json')['initial_lora_sha256']}}
    final=read(PARENT/'runs/ce_dpo/train/TRAIN_COMPLETE.json');models['full_dpo']={'adapter':final['adapter'],'lora_sha256':final['final_lora_sha256']}
    for m in models.values():paths += [Path(m['adapter'])/n for n in ('adapter_config.json','adapter_model.safetensors')]
    paths += [ROOT/'DATA.json']+[PROJECT_ROOT/'src'/n for n in ('vlm_native_scope_v1/model.py','vlm_batch_stable_v1/model.py','vlm_fit_positive_control_v1/study.py','vlm_grpo_iteration_v1/common.py','research_runtime/queue.py')]
    write_json(ROOT/'PLAN.json',{'status':'frozen','models':models,'pairs':7,'new_generations':28,'teacher_forced_sequences':28,
               'optimizer_updates':0,'activity_cap_seconds':300,'automatic_training':False,'primary_teacher':'start',
               'files':{str(f):sha_file(f) for f in paths},'scope':'Annotation-assisted training candidate admission only; not model ability evaluation.'})


def verify():
    for f,digest in read(ROOT/'PLAN.json')['files'].items():
        if sha_file(f)!=digest:raise ValueError('Frozen teacher input changed: '+f)


def worker(role):
    import torch
    from vlm_native_scope_v1.model import load
    from vlm_human_sft.evaluate import lora_hash
    from vlm_fit_positive_control_v1.study import generated,token_statistics
    from vlm_sft.common import to_device
    verify();plan=read(ROOT/'PLAN.json');data=read(ROOT/'DATA.json');started=time.monotonic()
    model,processor,_=load(plan['models'][role]['adapter'],'visual',False);model.eval();model.requires_grad_(False);before=lora_hash(model)
    if before!=plan['models'][role]['lora_sha256']:raise ValueError('Wrong frozen teacher checkpoint')
    directory=ROOT/'runs'/role;result=[];torch.cuda.reset_peak_memory_stats()
    eos=processor.tokenizer.convert_tokens_to_ids('<|im_end|>')
    with torch.inference_mode():
        for index,pair in enumerate(data['pairs']):
            dest=directory/'pairs'/f'{index:02d}.json'
            if dest.exists():
                record=read(dest)
                if record['source_question_id']!=pair['source_question_id'] or record['lora_sha256']!=before:raise ValueError('Teacher resume identity differs')
                result.append(record);continue
            records={}
            for condition in (('full','crop') if index%2==0 else ('crop','full')):
                q=pair['conditions'][condition];a=generated(model,processor,[q],data)[0]
                if a['input_sha256']!=q['inference_input_sha256']:raise ValueError('Actual teacher prompt differs')
                sample=chosen_sample(processor,q,data['images'])
                if content_sha(sample['input_ids'].tolist())!=q['chosen_input_sha256'] or content_sha(sample['labels'].tolist())!=q['chosen_labels_sha256']:raise ValueError('Teacher GT-prefix binding differs')
                batch=to_device(collate([sample],processor.tokenizer.pad_token_id),next(model.parameters()).device)
                keep=int((batch['labels']!=-100).sum(1).max())+1
                hook=model.get_output_embeddings().register_forward_pre_hook(lambda m,args:(args[0][:,-keep:,:],))
                try:out=model(**{k:v for k,v in batch.items() if k!='labels'},use_cache=False)
                finally:hook.remove()
                stats=token_statistics(out.logits,batch['labels'],eos)[0]
                records[condition]={'prediction':a,'strict_target_match':NORMALIZER(a['prediction'])==NORMALIZER(q['target']),
                                    'teacher_prefix':stats,'actual_grid':q['expected_grid']}
                del out,batch,sample
            record={'source_question_id':pair['source_question_id'],'lora_sha256':before,'conditions':records,'teacher_box_xyxy':pair['teacher_box_xyxy']}
            write_json(dest,record);result.append(record)
            write_json(directory/'PROGRESS.json',{'status':'teacher_diagnostic','completed_pairs':len(result),'expected_pairs':7,
                       'new_generations':2*len(result),'teacher_forced_sequences':2*len(result),'optimizer_updates':0,
                       'active_seconds':time.monotonic()-started,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30})
            if time.monotonic()-started>300 or torch.cuda.max_memory_allocated()/2**30>65:raise RuntimeError('Teacher diagnostic budget exceeded')
    if len(result)!=7 or lora_hash(model)!=before:raise ValueError('Teacher count/weights changed')
    write_json(directory/'ROWS.json',result);write_json(directory/'COMPLETE.json',{'status':'completed','pairs':7,'new_generations':14,'teacher_forced_sequences':14,'optimizer_updates':0,'model_hash_unchanged':True,'active_seconds':time.monotonic()-started})


def analyze():
    verify();summaries={}
    for role in ROLES:
        rows=read(ROOT/'runs'/role/'ROWS.json');summary={}
        for condition in ('full','crop'):
            a=[q['conditions'][condition] for q in rows]
            summary[condition]={'strict_correct':sum(q['strict_target_match'] for q in a),
                                'body_all_argmax':sum(q['teacher_prefix']['content_target_argmax'] for q in a),
                                'mean_body_nll':sum(q['teacher_prefix']['content_ce'] for q in a)/7,
                                'mean_eos_nll':sum(q['teacher_prefix']['eos_ce'] for q in a)/7}
        summary['wrong_full_to_correct_crop']=sum(not q['conditions']['full']['strict_target_match'] and q['conditions']['crop']['strict_target_match'] for q in rows)
        summary['body_full_wrong_to_crop_argmax']=sum(not q['conditions']['full']['teacher_prefix']['content_target_argmax'] and q['conditions']['crop']['teacher_prefix']['content_target_argmax'] for q in rows)
        summary['mean_body_nll_drop']=summary['full']['mean_body_nll']-summary['crop']['mean_body_nll'];summaries[role]=summary
    a=summaries['start'];primary=a['crop']['strict_correct']>=6 and a['wrong_full_to_correct_crop']>=4 and a['mean_body_nll_drop']>=.2
    body=a['crop']['body_all_argmax']>=6 and a['body_full_wrong_to_crop_argmax']>=4 and a['mean_body_nll_drop']>=.2
    result={'status':'completed','phase':'training_teacher_signal_supported' if primary else 'full_teacher_signal_insufficient',
            'summaries':summaries,'primary_complete_teacher_signal':primary,'secondary_body_teacher_signal':body,
            'primary_teacher':'start','optimizer_updates':0,'new_generations':28,'teacher_forced_sequences':28,
            'automatic_training':False,'goal_complete':False,'scope':'Fixed7 annotation-assisted train-candidate oracle; no inference method, benchmark gain or population causal claim.'}
    write_json(ROOT/'ANALYSIS.json',result);write_json(ROOT/'COMPLETION.json',result)
    write_json(ROOT/'DECISION.json',{'status':'completed','decision':'preregister_sufficient_data_student_teacher_training_before_any_updates' if primary else 'no_full_output_teacher_training_admitted','new_training_admitted':False})
    render(result)


def render(result):
    import html,os
    e=lambda s:html.escape(str(s));data=read(ROOT/'DATA.json');records={r:{q['source_question_id']:q for q in read(ROOT/'runs'/r/'ROWS.json')} for r in ROLES};cards=[]
    for pair in data['pairs']:
        full=pair['conditions']['full'];crop=pair['conditions']['crop'];lines=[]
        for role in ROLES:
            a=records[role][pair['source_question_id']]['conditions'];lines.append(e(role)+'：整图 '+e(a['full']['prediction']['prediction'])+' → 局部 '+e(a['crop']['prediction']['prediction'])+f'；正文NLL {a["full"]["teacher_prefix"]["content_ce"]:.3f} → {a["crop"]["teacher_prefix"]["content_ce"]:.3f}')
        cards.append('<article><h3>QA '+pair['source_question_id']+'</h3><img src="'+e(os.path.relpath(data['images'][full['image_id']]['path'],ROOT))+'"><img class="crop" src="'+e(os.path.relpath(data['images'][crop['image_id']]['path'],ROOT))+'"><p>'+e(full['question'])+'</p><p>'+'<br>'.join(lines)+'</p></article>')
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>训练候选局部教师诊断</title><style>body{font:16px/1.8 system-ui;max-width:1400px;margin:25px auto;padding:20px;background:#f4f6f8;color:#173044}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}article{background:white;padding:16px}img{width:100%;height:260px;object-fit:contain}.crop{height:180px}pre{white-space:pre-wrap;background:#e8eff3;padding:18px}@media(max-width:800px){.cards{grid-template-columns:1fr}}</style><h1>局部教师可用性：不是模型能力提升</h1><p>只用7条训练候选，标注辅助区域是真实教师诊断输入；与先前仅显示的放大crop不同。目标位置已知，非部署方法，非benchmark分数。0训练/28生成/28答案前缀前向。</p><pre>'+e(json.dumps(result['summaries'],ensure_ascii=False,indent=2))+'</pre><p>主教师门槛通过：'+str(result['primary_complete_teacher_signal'])+'。正文辅助信号：'+str(result['secondary_body_teacher_signal'])+'。均不自动放行训练，仍须足量可靠数据和另行固定对照。</p><pre>待检验的未来训练：CE(正确答案 | 整图) + lambda KL(冻结教师[局部,GT前缀] || 学生[整图,GT前缀])\n本轮未实现或运行该训练，推理不应依赖答案位置。crop同时改变尺度与上下文，不能作纯定位因果。</pre><div class="cards">'+''.join(cards)+'</div><p><a href="COMPLETION.json">完成数据</a> · <a href="'+e(os.path.relpath(PROTOCOL,ROOT))+'">事前协议</a> · <a href="'+e(os.path.relpath(Path(__file__),ROOT))+'">真实代码</a> · <a href="../rarevlm_monitor/状态看板.html">观察台</a></p></html>'
    (ROOT/'局部教师可用性与真实输入.html').write_text(page)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--worker',choices=ROLES);args=parser.parse_args()
    if args.prepare:prepare();return
    if args.worker:worker(args.worker);return
    from research_runtime.queue import run_queue
    verify();entry=str(Path(__file__).resolve());run_queue(ROOT/'queue',[{'key':r,'command':[sys.executable,'-u',entry,'--worker',r],'completion_marker':str(ROOT/'runs'/r/'COMPLETE.json')} for r in ROLES],{'plan_sha256':sha_file(ROOT/'PLAN.json')});analyze()


if __name__=='__main__':main()
