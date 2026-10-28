"""Use the frozen probe loop with label-free, task-independent generation."""
import argparse
from pathlib import Path
import sys
from rarevlm_paths import PROJECT_ROOT
from vlm_deep.common import encode,collate,content_sha,sha_file,write_json
from vlm_grpo_iteration_v1.common import read
import run_stvqa_headroom_v1 as study

ROOT=study.ROOT


def source_generated(model,processor,rows,data):
    import torch
    from vlm_probe.evaluate import inference_row
    from vlm_sft.common import to_device
    from vlm_sampled_signal_v1.trajectory import stop_at_eos
    from vlm_attention_scope_repair_v1.generation import raw_greedy_config
    model.generation_config=raw_greedy_config(model.generation_config,32)
    items=[encode(processor,inference_row(q),Path(data['images'][q['image_id']]['path']),False) for q in rows]
    batch=to_device(collate(items,processor.tokenizer.pad_token_id),next(model.parameters()).device)
    with torch.inference_mode():output=model.generate(**batch,max_new_tokens=32,do_sample=False,use_cache=True)
    result=[]
    for index,(q,item) in enumerate(zip(rows,items)):
        ids,terminated=stop_at_eos(output[index,batch['input_ids'].shape[1]:].tolist(),model.generation_config.eos_token_id,32)
        prediction=processor.tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False).strip()
        if int(item['input_ids'].shape[1])!=q['expected_prefix_tokens'] or item['image_grid_thw'][0].tolist()!=q['expected_grid']:
            raise ValueError('Source inference frame differs')
        result.append({'question_id':q['question_id'],'image_id':q['image_id'],'prediction':prediction,
            'completion_ids':ids,'terminated':terminated,'input_sha256':content_sha(item['input_ids'].tolist()),
            'visual_tokens':int(item['image_grid_thw'].prod(-1).sum())//4})
    return result


def verify():
    study.verify()
    lock=read(ROOT/'SOURCE_INFERENCE_ADAPTER_PLAN.json')
    if sha_file(ROOT/'PLAN.json')!=lock['original_probe_plan_sha256']:raise ValueError('Probe plan changed')
    for p,digest in lock['files'].items():
        if sha_file(p)!=digest:raise ValueError('Frozen source inference adapter changed')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');parser.add_argument('--worker',type=int,choices=(0,1));args=parser.parse_args()
    if args.freeze:
        if (ROOT/'SOURCE_INFERENCE_ADAPTER_PLAN.json').exists():raise FileExistsError('Source adapter frozen')
        study.verify();paths=[Path(__file__),PROJECT_ROOT/'src/vlm_probe/evaluate.py',PROJECT_ROOT/'src/vlm_attention_scope_repair_v1/generation.py',PROJECT_ROOT/'src/vlm_sampled_signal_v1/trajectory.py']
        write_json(ROOT/'SOURCE_INFERENCE_ADAPTER_PLAN.json',{'status':'frozen','files':{str(p):sha_file(p) for p in paths},
            'original_probe_plan_sha256':sha_file(ROOT/'PLAN.json'),'change':'Raw label-free inference with no task scoring; references only used afterward for source exact',
            'fake_answers_added':0,'preexisting_probe_gpu_calls':0,'original_probe_source_preserved':True})
        print({'status':'source_adapter_frozen','ten_vote_scorer_used':False});return
    verify()
    if args.worker is not None:
        import vlm_fit_positive_control_v1.study as generation_provider
        generation_provider.generated=source_generated
        study.worker(args.worker);return
    from research_runtime.queue import run_queue
    run_queue(ROOT/'queue',[{'key':f'shard_{s}','command':[sys.executable,'-u',str(Path(__file__).resolve()),'--worker',str(s)],
        'completion_marker':str(ROOT/f'runs/shard_{s}/COMPLETE.json')} for s in (0,1)],{'plan_sha256':sha_file(ROOT/'PLAN.json'),
        'source_adapter_sha256':sha_file(ROOT/'SOURCE_INFERENCE_ADAPTER_PLAN.json')})
    study.analyze()


if __name__=='__main__':main()
