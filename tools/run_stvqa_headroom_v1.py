"""Fixed train slots and source-holdout probe; not mining or training."""
import argparse
from pathlib import Path
import sys,time
from rarevlm_paths import PROJECT_ROOT
from vlm_deep.common import encode,collate,content_sha,sha_file,write_json
from vlm_grpo_iteration_v1.common import read,chosen_sample
from prepare_stvqa_source_subset_v1 import literal

BASE=PROJECT_ROOT.parents[1]/'outputs';SOURCE=BASE/'vlm_stvqa_source_v1';ROOT=BASE/'vlm_stvqa_headroom_v1'
PROTOCOL=PROJECT_ROOT/'docs/VLM_STVQA_HEADROOM_V1_PROTOCOL.md'


def prepare():
    import torch
    from vlm_probe.evaluate import processor_for,inference_row
    if (ROOT/'PLAN.json').exists():raise FileExistsError('Headroom plan frozen')
    torch.set_num_threads(2);original=read(SOURCE/'DATA.json');inventory=read(SOURCE/'SOURCE_INVENTORY.json')
    if not inventory['source_data_count_gate']:raise ValueError('Insufficient source data')
    c=read(BASE/'vlm_visual_preference_v1/CONFIG.json');base=read(c['baseline_config'])
    processor=processor_for(Path(base['model_root']),read(base['reference_config']),1024)
    train=set(original['training_slot_ids']);rows=[]
    for q in original['rows']:
        if q['cohort']!='source_holdout' and q['question_id'] not in train:continue
        q={**q,'probe_cohort':'train_slots' if q['question_id'] in train else 'source_holdout'}
        path=Path(original['images'][q['image_id']]['path']);prefix=encode(processor,inference_row(q),path,False)
        sample=chosen_sample(processor,q,original['images'])
        if sample['input_ids'].shape[1]>2048:raise ValueError('Source input exceeds budget')
        q.update(expected_prefix_tokens=int(prefix['input_ids'].shape[1]),expected_grid=prefix['image_grid_thw'][0].tolist(),
                 inference_input_sha256=content_sha(prefix['input_ids'].tolist()),chosen_input_sha256=content_sha(sample['input_ids'].tolist()),
                 chosen_labels_sha256=content_sha(sample['labels'].tolist()))
        rows.append(q)
        if len(rows)%32==0:write_json(ROOT/'CPU_PROGRESS.json',{'completed':len(rows),'expected':384,'model_calls':0})
    if len(rows)!=384 or sum(q['probe_cohort']=='train_slots' for q in rows)!=128:raise ValueError('Wrong fixed cohort counts')
    groups=[]
    for cohort in ('train_slots','source_holdout'):
        chunk=[q['question_id'] for q in rows if q['probe_cohort']==cohort]
        groups.extend(chunk[i:i+4] for i in range(0,len(chunk),4))
    images={q['image_id']:original['images'][q['image_id']] for q in rows}
    write_json(ROOT/'DATA.json',{'rows':rows,'images':images,'groups':groups})
    parent=read(BASE/'vlm_crop_teacher_signal_v1/PLAN.json')
    paths=[Path(__file__),PROTOCOL,ROOT/'DATA.json',SOURCE/'SOURCE_INVENTORY.json',SOURCE/'SOURCE_SUBSET_LOCK.json',
           PROJECT_ROOT/'tools/prepare_stvqa_source_subset_v1.py',PROJECT_ROOT/'src/vlm_deep/common.py']
    paths += [Path(m['path']) for m in images.values()]
    write_json(ROOT/'PLAN.json',{'status':'frozen','model':parent['models']['start'],'qa':384,'generations':384,'teacher_forced_sequences':384,
        'optimizer_updates':0,'automatic_training':False,'files':{**parent['files'],**{str(p):sha_file(p) for p in paths}}})
    print({'status':'prepared','qa':384,'training':False},flush=True)


def verify():
    for p,digest in read(ROOT/'PLAN.json')['files'].items():
        if sha_file(p)!=digest:raise ValueError('Frozen source probe changed: '+p)


def worker(shard):
    import torch
    from vlm_native_scope_v1.model import load
    from vlm_human_sft.evaluate import lora_hash
    from vlm_fit_positive_control_v1.study import generated,token_statistics
    from vlm_sft.common import to_device
    verify();plan=read(ROOT/'PLAN.json');data=read(ROOT/'DATA.json');lookup={q['question_id']:q for q in data['rows']};started=time.monotonic()
    model,processor,_=load(plan['model']['adapter'],'visual',False);model.requires_grad_(False);before=lora_hash(model)
    if before!=plan['model']['lora_sha256']:raise ValueError('Wrong initial model')
    groups=data['groups'][shard::2];result=[];directory=ROOT/f'runs/shard_{shard}';eos=processor.tokenizer.convert_tokens_to_ids('<|im_end|>')
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for index,ids in enumerate(groups):
            rows=[lookup[i] for i in ids];predictions=generated(model,processor,rows,data)
            samples=[chosen_sample(processor,q,data['images']) for q in rows]
            for q,s in zip(rows,samples):
                if content_sha(s['input_ids'].tolist())!=q['chosen_input_sha256'] or content_sha(s['labels'].tolist())!=q['chosen_labels_sha256']:raise ValueError('Actual GT input changed')
            batch=to_device(collate(samples,processor.tokenizer.pad_token_id),next(model.parameters()).device)
            keep=int(batch['labels'].ne(-100).sum(1).max())+1;hook=model.get_output_embeddings().register_forward_pre_hook(lambda m,args:(args[0][:,-keep:,:],))
            try:out=model(**{k:v for k,v in batch.items() if k!='labels'},use_cache=False)
            finally:hook.remove()
            stats=token_statistics(out.logits,batch['labels'],eos)
            for q,p,s in zip(rows,predictions,stats):
                if p['input_sha256']!=q['inference_input_sha256']:raise ValueError('Actual source inference input changed')
                p.pop('vqa_score',None)
                result.append({**p,'probe_cohort':q['probe_cohort'],'source_literal_exact':any(literal(p['prediction'])==literal(a) for a in q['answers']),
                               'body_nll':s['content_ce'],'eos_nll':s['eos_ce'],'body_all_argmax':s['content_target_argmax']})
            write_json(directory/'PREDICTIONS.json',result);write_json(directory/'PROGRESS.json',{'completed':len(result),'expected':sum(len(g) for g in groups),
                'new_generations':len(result),'teacher_forced_sequences':len(result),'optimizer_updates':0,'active_seconds':time.monotonic()-started,
                'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30})
            del out,batch,samples
            if time.monotonic()-started>900 or torch.cuda.max_memory_allocated()/2**30>65:raise RuntimeError('Source probe budget exceeded')
    if lora_hash(model)!=before:raise ValueError('Frozen model changed')
    write_json(directory/'COMPLETE.json',{'status':'completed','qa':len(result),'active_seconds':time.monotonic()-started,'optimizer_updates':0,'model_unchanged':True})


def analyze():
    verify();rows=[q for s in (0,1) for q in read(ROOT/f'runs/shard_{s}/PREDICTIONS.json')];summaries={}
    for cohort in ('train_slots','source_holdout'):
        selected=[q for q in rows if q['probe_cohort']==cohort];n=len(selected)
        summaries[cohort]={'qa':n,'literal_correct':sum(q['source_literal_exact'] for q in selected),
            'exact_percent':100*sum(q['source_literal_exact'] for q in selected)/n,
            'mean_body_nll':sum(q['body_nll'] for q in selected)/n,'mean_eos_nll':sum(q['eos_nll'] for q in selected)/n,
            'length_censored':sum(not q['terminated'] for q in selected)}
    a=summaries['train_slots'];signal=a['mean_body_nll']>=.2 and a['qa']-a['literal_correct']>=16
    result={'status':'completed','phase':'new_source_supervision_signal' if signal else 'fixed_source_signal_insufficient','summaries':summaries,
        'supervision_signal':signal,'new_generations':384,'teacher_forced_sequences':384,'optimizer_updates':0,'new_training_admitted':False,
        'goal_complete':False,'scope':'Fixed train slots plus internal source holdout; not official ANLS or model capability improvement'}
    write_json(ROOT/'COMPLETION.json',result);write_json(ROOT/'DECISION.json',{'status':'completed','new_training_admitted':False,
        'decision':'preregister_matched_source_replacement_sft_dpo' if signal else 'stop_fixed_source_no_mining_or_slot_reselection'});print(result,flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--worker',type=int,choices=(0,1));args=parser.parse_args()
    if args.prepare:prepare();return
    if args.worker is not None:worker(args.worker);return
    from research_runtime.queue import run_queue
    verify();run_queue(ROOT/'queue',[{'key':f'shard_{s}','command':[sys.executable,'-u',str(Path(__file__).resolve()),'--worker',str(s)],'completion_marker':str(ROOT/f'runs/shard_{s}/COMPLETE.json')} for s in (0,1)],{'plan_sha256':sha_file(ROOT/'PLAN.json')});analyze()


if __name__=='__main__':main()
