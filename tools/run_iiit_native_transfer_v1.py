"""Fresh1617 VQA evaluation of four already-frozen IIIT configurations."""
import argparse
from pathlib import Path
import sys
import time

from rarevlm_paths import PROJECT_ROOT
from vlm_deep.common import content_sha,sha_file,write_json
from vlm_grpo_iteration_v1.common import actor,read
from vlm_fit_positive_control_v1.study import generated

BASE=PROJECT_ROOT.parents[1]/'outputs'
OLD=BASE/'vlm_iiit_scope_v1'
ROOT=BASE/'vlm_iiit_native_transfer_v1'
PROTOCOL=PROJECT_ROOT/'docs/VLM_IIIT_NATIVE_TRANSFER_V1_PROTOCOL.md'


def prepare():
    import torch
    from vlm_probe.evaluate import processor_for,inference_row
    from vlm_deep.common import encode
    if (ROOT/'PLAN.json').exists():raise FileExistsError('Transfer study already frozen')
    torch.set_num_threads(2);data=read(BASE/'vlm_literal_case_sft_v1/DATA.json');rows=data['evaluations']['transfer_dev'];images=data['images']
    init=read(OLD/'INITIALIZATION.json');reference=read(BASE/'vlm_reference_replay_v1/PLAN.json');proc=processor_for(Path(init['model_root']),read(reference['reference_config']),1024)
    original=read(BASE/'vlm_deep_v2/prepared/manifest.json');initial_ids={q['image_id'] for q in original['splits']['train_all']}
    current=read(OLD/'DATA.json');training_ids={q['image_id'] for q in current['training_rows']};eval_ids={q['image_id'] for q in rows}
    if training_ids&eval_ids:raise ValueError('Current training image ID overlap')
    for index,q in enumerate(rows):
        item=encode(proc,inference_row(q),Path(images[q['image_id']]['path']),False)
        if int(item['input_ids'].shape[1])!=q['expected_prefix_tokens'] or item['image_grid_thw'][0].tolist()!=q['expected_grid']:raise ValueError('Legacy transfer frame changed')
        q['inference_input_sha256']=content_sha(item['input_ids'].tolist())
        if (index+1)%256==0:write_json(ROOT/'CPU_PROGRESS.json',{'completed':index+1,'expected':1617})
    write_json(ROOT/'DATA.json',{'images':images,'evaluations':{'transfer_dev':rows}})
    parent=read(OLD/'PLAN.json');models={'initial':{'adapter':init['adapter'],'lora_sha256':init['composite_lora_sha256']}}
    for role in parent['roles']:
        result=read(OLD/'runs'/role/'train/TRAIN_COMPLETE.json');models[role]={'adapter':result['adapter'],'lora_sha256':result['final_lora_sha256']}
    files=dict(parent['files'])
    for path in [Path(__file__),PROTOCOL,ROOT/'DATA.json',OLD/'INITIALIZATION.json',BASE/'vlm_literal_case_sft_v1/DATA.json']:
        files[str(path)]=sha_file(path)
    plan={'status':'prepared','files':files,'models':models,'packages':parent['packages'],'base_weight_files':parent['base_weight_files'],
          'new_image_files':{images[i]['path']:images[i]['sha256'] for i in eval_ids},'qa':1617,'total_generations':6468,'new_optimizer_updates':0,
          'original_sft_train_all_image_id_overlap':len(initial_ids&eval_ids),'current_training_image_id_overlap':0}
    write_json(ROOT/'PLAN.json',plan);print('TRANSFER_PREPARED',{'qa':1617,'models':4,'original_sft_image_id_overlap':plan['original_sft_train_all_image_id_overlap']},flush=True)


def verify():
    from vlm_object_train.train import package_lock
    plan=read(ROOT/'PLAN.json')
    for path,digest in {**plan['files'],**plan['new_image_files']}.items():
        if sha_file(path)!=digest:raise ValueError('Frozen transfer input/source changed')
    if package_lock()!=plan['packages']:raise ValueError('Runtime versions changed')
    return plan


def worker(role):
    import torch
    from vlm_human_sft.evaluate import lora_hash
    from vlm_object_train.train import lock_json
    plan=verify();model,processor,_=actor(adapter=plan['models'][role]['adapter']);before=lora_hash(model)
    if before!=plan['models'][role]['lora_sha256']:raise ValueError('Wrong frozen model')
    data=read(ROOT/'DATA.json');rows=data['evaluations']['transfer_dev'];directory=ROOT/'runs'/role
    lock_json(directory/'INPUT_LOCK.json',{'plan_sha256':sha_file(ROOT/'PLAN.json'),'model_lora_sha256':before})
    path=directory/'PREDICTIONS.json';preds=read(path) if path.exists() else[]
    if len(preds)>1617 or (len(preds)%4 and len(preds)!=1617) or [q['question_id'] for q in preds]!=[q['question_id'] for q in rows[:len(preds)]]:raise ValueError('Resume differs')
    start=time.monotonic()
    for begin in range(len(preds),1617,4):
        chunk=rows[begin:begin+4];answers=generated(model,processor,chunk,data)
        if any(a['input_sha256']!=q['inference_input_sha256'] for q,a in zip(chunk,answers)):raise ValueError('Actual transfer prefix differs')
        preds.extend(answers);temp=path.with_suffix('.tmp');write_json(temp,preds);temp.replace(path)
        write_json(directory/'PROGRESS.json',{'status':'evaluating','completed':len(preds),'expected':1617,'active_seconds':time.monotonic()-start})
        if time.monotonic()-start>1800 or torch.cuda.max_memory_allocated()/2**30>65:raise RuntimeError('Transfer budget cap')
    if lora_hash(model)!=before:raise ValueError('Read-only model changed')
    write_json(directory/'COMPLETE.json',{'status':'completed','qa':1617,'optimizer_updates':0})


def analyze():
    from vlm_grpo_iteration_v1.study import paired
    plan=read(ROOT/'PLAN.json');records={r:read(ROOT/'runs'/r/'PREDICTIONS.json') for r in plan['models']};meta={q['question_id']:q for q in read(ROOT/'DATA.json')['evaluations']['transfer_dev']}
    metrics={r:100*sum(q['vqa_score'] for q in rows)/len(rows) for r,rows in records.items()};comparisons={};slices={}
    for role in plan['models']:
        if role=='initial':continue
        comparisons[role]=paired(records['initial'],records[role]);slices[role]={}
        for group in ['singleword','multiword','zero','rare_1_5']:
            ids={qid for qid,q in meta.items() if (q['stratum']==group if group in ['singleword','multiword'] else q['answer_frequency']==0 if group=='zero' else 1<=q['answer_frequency']<=5)}
            slices[role][group]=paired([q for q in records['initial'] if q['question_id'] in ids],[q for q in records[role] if q['question_id'] in ids])
    result={'status':'completed','metrics':metrics,'vs_initial':comparisons,'slices':slices,'new_generations':6468,'optimizer_updates':0,'goal_complete':False,'scope':'Historically used internal VQA1617;four fixed single-checkpoint comparisons,no new blind/seed-stable evidence.'}
    write_json(ROOT/'COMPLETION.json',result);print('NATIVE_TRANSFER_COMPLETE',result,flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--worker');args=parser.parse_args()
    if args.prepare:prepare();return
    if args.worker:worker(args.worker);return
    from research_runtime.queue import run_queue
    verify();identity={'plan_sha256':sha_file(ROOT/'PLAN.json')};entry=str(Path(__file__).resolve());order=['initial','sparse_language','dense_language','dense_visual']
    run_queue(ROOT/'queue',[{'key':r,'command':[sys.executable,'-u',entry,'--worker',r],'completion_marker':str(ROOT/'runs'/r/'COMPLETE.json')} for r in order],identity);analyze()


if __name__=='__main__':main()
