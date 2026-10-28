"""One observed sign-flip case: exact-prefix / batch / precision paths, zero updates."""
import argparse
from pathlib import Path
import sys
import time
from rarevlm_paths import PROJECT_ROOT, SOURCE_ROOT
from vlm_deep.common import sha_file, write_json
from vlm_region_ocr_sft_v1.study import actor, read

ROOT = PROJECT_ROOT.parents[1] / 'outputs/vlm_error_numeric_path_v1'
TRACE = ROOT.parent / 'vlm_error_decision_trace_v1'
OLD = ROOT.parent / 'vlm_error_supervision_sft_v1'
QID = '20675'

def prepare():
    if (ROOT / 'PLAN.json').exists():
        raise FileExistsError('Already frozen; inspect current queue')
    plan = read(TRACE / 'PLAN.json')
    critical = next(q for q in read(TRACE / 'initial/GENERATION.json') if q['question_id']==QID)['critical']
    micro = next(m for m in read(TRACE / 'DATA.json')['micros'] if QID in m['ids'])
    paths = [Path(__file__).resolve(), PROJECT_ROOT / 'docs/VLM_ERROR_NUMERIC_PATH_V1_PROTOCOL.md',
        TRACE / 'PLAN.json', TRACE / 'DATA.json', TRACE / 'initial/GENERATION.json', TRACE / 'dpo_sft/TEACHER.json',
        OLD / 'data/CPU_INPUTS.json', OLD / 'reference/CACHE.json']
    paths += [Path(plan['roles']['dpo_sft']['adapter']) / f for f in ('adapter_model.safetensors', 'adapter_config.json')]
    paths += [SOURCE_ROOT / f for f in ('vlm_region_ocr_sft_v1/study.py','vlm_region_preference_probe_v1/common.py',
        'vlm_fp32_readout_v1/model.py','vlm_fp32_adapter/model.py','vlm_runtime_metadata_v1/model.py',
        'vlm_attention_scope_repair_v1/generation.py','research_runtime/queue.py')]
    write_json(ROOT / 'PLAN.json', {'status':'prepared','question_id':QID,'role':'dpo_sft','micro':micro,
        'critical_index':critical['label_index'],'correct_id':critical['correct_id'],'wrong_id':critical['wrong_id'],
        'adapter':plan['roles']['dpo_sft']['adapter'],'lora_sha256':plan['roles']['dpo_sft']['lora_sha256'],
        'files':{str(f):sha_file(f) for f in paths},'training_updates':0})
    print('Prepared fixed posthoc numerical path case', flush=True)

def run():
    import torch
    from vlm_error_supervision_sft_v1.common import make_batch
    from vlm_sft.common import to_device
    from vlm_human_sft.evaluate import lora_hash
    from vlm_attention_scope_repair_v1.generation import raw_greedy_config
    plan = read(ROOT / 'PLAN.json')
    for path,digest in plan['files'].items():
        if sha_file(path)!=digest:
            raise ValueError('Frozen numerical-path input changed: '+path)
    model,processor,_=actor(plan['adapter'])
    if lora_hash(model)!=plan['lora_sha256']:
        raise ValueError('Wrong numerical-path model')
    model.eval()
    model.generation_config=raw_greedy_config(model.generation_config,1)
    data=read(TRACE/'DATA.json');lookup={q['question_id']:q for q in data['rows']}
    rows=[lookup[q] for q in plan['micro']['ids']]
    full,binding,_,_=make_batch(processor,data,rows,read(OLD/'data/CPU_INPUTS.json'))
    if binding!=read(OLD/'reference/CACHE.json')['batches'][plan['micro']['key']]['binding']:
        raise ValueError('Actual training microbatch differs')
    focal=plan['micro']['ids'].index(QID)
    first=int((full['labels'][focal]!=-100).nonzero()[0])
    cut=first+plan['critical_index']
    correct,wrong=plan['correct_id'],plan['wrong_id']
    if int(full['labels'][focal,cut])!=correct:
        raise ValueError('Focal label shifted')
    keep=int((full['labels']!=-100).sum(1).max())+1
    full={k:v for k,v in full.items() if k!='labels'}
    prefix={k:(v[:,:cut] if k in ('input_ids','attention_mask') else v) for k,v in full.items()}
    patches=int(prefix['image_grid_thw'][:4].prod(-1).sum())
    prefix4={k:(v[:4] if k in ('input_ids','attention_mask','image_grid_thw') else v[:patches]) for k,v in prefix.items()}
    device=next(model.parameters()).device
    full=to_device(full,device);prefix=to_device(prefix,device);prefix4=to_device(prefix4,device)
    core=model.get_base_model()
    positions_full=core.get_rope_index(input_ids=full['input_ids'],image_grid_thw=full['image_grid_thw'],attention_mask=full['attention_mask'])[0]
    positions_prefix=core.get_rope_index(input_ids=prefix['input_ids'],image_grid_thw=prefix['image_grid_thw'],attention_mask=prefix['attention_mask'])[0]
    positions_prefix4=core.get_rope_index(input_ids=prefix4['input_ids'],image_grid_thw=prefix4['image_grid_thw'],attention_mask=prefix4['attention_mask'])[0]
    if not torch.equal(positions_full[:,:,:cut],positions_prefix) or not torch.equal(positions_prefix[:,:4],positions_prefix4):
        raise ValueError('RoPE prefix positions differ; cannot isolate numerical paths')
    results={}
    vectors={}
    started=time.monotonic()
    with torch.inference_mode():
        for precision in ('original_bf16','upcast_fp32'):
            if precision=='upcast_fp32':
                model.float()
                model.eval()
            for name,batch,col,tail,generate in (
                    ('full8',full,cut-1,keep,False),
                    ('prefix8',prefix,-1,1,False),
                    ('prefix4',prefix4,-1,1,False),
                    ('prefix4_generate1',prefix4,-1,1,True)):
                head=model.get_output_embeddings();capture={}
                def hook(module,args):
                    h=args[0][focal,col].detach().float()
                    capture['hidden']=h.cpu().clone()
                    capture['hidden_dtype']=str(args[0].dtype)
                    capture['dot_margin']=float(torch.dot(h,head.weight[correct].float()-head.weight[wrong].float()))
                    return (args[0][:,-tail:,:],)
                handle=head.register_forward_pre_hook(hook)
                try:
                    if generate:
                        out=model.generate(**batch,max_new_tokens=1,do_sample=False,use_cache=True,
                            return_dict_in_generate=True,output_scores=True)
                        logits=out.scores[0][focal]
                        actual_id=int(out.sequences[focal,-1])
                    else:
                        out=model(**batch,use_cache=False)
                        pos=(col if col>=0 else batch['input_ids'].shape[1]-1)-(batch['input_ids'].shape[1]-out.logits.shape[1])
                        logits=out.logits[focal,pos]
                        actual_id=int(logits.argmax())
                finally:
                    handle.remove()
                label=precision+':'+name
                vectors[label]=capture.pop('hidden')
                results[label]={**capture,'actual_head_margin':float(logits[correct]-logits[wrong]),
                    'argmax_id':actual_id,'argmax_text':processor.tokenizer.decode([actual_id]),
                    'input_shape':list(batch['input_ids'].shape)}
                write_json(ROOT/'PROGRESS.json',{'status':'reading_paths','conditions_completed':len(results),'expected_conditions':8,
                    'active_seconds':time.monotonic()-started,'training_updates':0})
                del out,logits
                if time.monotonic()-started>300:
                    raise TimeoutError('Fixed numerical path cap reached')
    if lora_hash(model)!=plan['lora_sha256']:
        raise ValueError('LoRA changed during numerical path isolation')
    for precision in ('original_bf16','upcast_fp32'):
        reference=vectors[precision+':prefix4']
        for name in ('full8','prefix8','prefix4_generate1'):
            v=vectors[precision+':'+name];diff=v-reference
            results[precision+':'+name]['hidden_max_abs_delta_vs_prefix4']=float(diff.abs().max())
            results[precision+':'+name]['hidden_l2_delta_vs_prefix4']=float(diff.norm())
    result={'status':'completed','question_id':QID,'training_updates':0,'same_rope_prefix_positions':True,
        'lora_unchanged':True,'base_upcast_only_in_memory_no_adapter_export':True,'conditions':results,
        'active_seconds':time.monotonic()-started,
        'scope':'One posthoc case; FP32 also changes possible SDPA kernels. Not training improvement or a universal cause.'}
    write_json(ROOT/'COMPLETION.json',result)
    print(result,flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare',action='store_true');p.add_argument('--worker',action='store_true')
    a=p.parse_args()
    if a.prepare:prepare()
    elif a.worker:run()
    else:
        from research_runtime.queue import run_queue
        run_queue(ROOT/'queue',[{'key':'isolate','command':[sys.executable,'-u',str(Path(__file__).resolve()),'--worker'],
            'completion_marker':str(ROOT/'COMPLETION.json')}],{'plan_sha256':sha_file(ROOT/'PLAN.json')})

if __name__=='__main__':main()

