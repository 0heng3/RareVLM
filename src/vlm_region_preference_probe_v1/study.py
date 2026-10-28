"""Short reciprocal grounding-preference positive control; no main-screen tuning."""
import argparse
from pathlib import Path
import json,sys,time
import torch
from rarevlm_paths import PROJECT_ROOT,SOURCE_ROOT
from vlm_deep.common import sha_file,write_json,content_sha
from vlm_sft.common import to_device
from vlm_human_sft.evaluate import lora_hash
from vlm_region_ocr_sft_v1.study import actor
from vlm_alignment_signal.loss import sequence_logps
from vlm_object_train.train import package_lock
from .common import ROOT,CONFIG,read,make_batch,evaluate

ROLES=('sft','dpo_sft')


def prepare():
    data=read(ROOT/'DATA_COMPLETE.json')
    if data['reciprocal_pairs']!=14 or data['actual_supervised_cpu_encodings']!=312:
        raise ValueError('Reviewed14-pair/312-encoding cohort required')
    if (ROOT/'PRELAUNCH_INPUT_LOCK.json').exists():raise FileExistsError('Already frozen; inspect existing queue')
    c=read(CONFIG);base=read(c['baseline_config'])
    parent=PROJECT_ROOT.parents[1]/'outputs/vlm_region_ocr_sft_v1'
    parent_lock=read(parent/'PRELAUNCH_INPUT_LOCK.json')
    if package_lock()!=parent_lock['packages']:raise ValueError('Package versions differ from initial baseline')
    for path,value in parent_lock['base_weight_files'].items():
        st=Path(path).stat()
        if {'bytes':st.st_size,'mtime_ns':st.st_mtime_ns}!=value:raise ValueError('Base weights changed')
    files=[CONFIG,PROJECT_ROOT/'docs/VLM_REGION_PREFERENCE_PROBE_V1_PROTOCOL.md',
        PROJECT_ROOT/'tools/prepare_region_preference_probe_v1.py',ROOT/'DATA_COMPLETE.json',ROOT/'DATA_REVIEW.json',
        ROOT/'CANDIDATES_WITH_REVIEW.json',Path(c['baseline_config']),Path(base['reference_config'])]
    files+=list((ROOT/'data').glob('*'))+list((ROOT/'review').glob('*'))+list((SOURCE_ROOT/'vlm_region_preference_probe_v1').glob('*.py'))
    files += [Path(c['initial_adapter'])/name for name in ('adapter_config.json','adapter_model.safetensors')]
    files += [SOURCE_ROOT/name for name in ('vlm_alignment_signal/loss.py','vlm_fp32_readout_v1/model.py',
        'vlm_fp32_adapter/model.py','vlm_runtime_metadata_v1/model.py','vlm_probe/evaluate.py',
        'vlm_attention_scope_repair_v1/generation.py','vlm_deep/common.py','vlm_deep/evaluate.py',
        'vlm_sft/common.py','vlm_object_train/train.py','research_runtime/queue.py','vlm_region_ocr_sft_v1/study.py')]
    manifest=read(ROOT/'data/MANIFEST.json')
    write_json(ROOT/'PRELAUNCH_INPUT_LOCK.json',{'files':{str(p):sha_file(p) for p in sorted(set(files))},
        'images':{im['path']:im['sha256'] for im in manifest['images'].values()},
        'packages':parent_lock['packages'],'base_weight_files':parent_lock['base_weight_files'],
        'initial_lora_sha256':read(parent/'STUDY_INPUT_LOCK.json')['initial_lora_sha256'],
        'scope':'Small training-only positive control, no formal benchmark or resume gain'})


def verify(training=False):
    lock=read(ROOT/'PRELAUNCH_INPUT_LOCK.json')
    for path,value in {**lock['files'],**lock['images']}.items():
        if sha_file(path)!=value:raise ValueError('Frozen short-probe source/data changed: '+path)
    if package_lock()!=lock['packages']:raise ValueError('Package versions changed')
    for path,value in lock['base_weight_files'].items():
        st=Path(path).stat()
        if {'bytes':st.st_size,'mtime_ns':st.st_mtime_ns}!=value:raise ValueError('Base storage changed')
    for path,value in lock['base_weight_files'].items():
        st=Path(path).stat()
        if {'bytes':st.st_size,'mtime_ns':st.st_mtime_ns}!=value:raise ValueError('Base storage changed')
    if training:
        study=read(ROOT/'STUDY_INPUT_LOCK.json')
        for path,value in study['new_baseline_and_reference_files'].items():
            if sha_file(path)!=value:raise ValueError('Initial/reference cache changed')


def baseline():
    verify();c=read(CONFIG);model,processor,_=actor(c['initial_adapter']);before=lora_hash(model)
    if before!=read(ROOT/'PRELAUNCH_INPUT_LOCK.json')['initial_lora_sha256']:raise ValueError('Initial checkpoint differs')
    metrics=evaluate(model,processor,ROOT/'baseline')
    if lora_hash(model)!=before:raise ValueError('Read-only baseline changed adapter')
    write_json(ROOT/'baseline/COMPLETE.json',{'status':'completed','training_updates':0,'lora_sha256':before,'metrics':metrics})


def reference():
    verify();c=read(CONFIG);model,processor,_=actor(c['initial_adapter']);before=lora_hash(model)
    if before!=read(ROOT/'PRELAUNCH_INPUT_LOCK.json')['initial_lora_sha256']:raise ValueError('Reference checkpoint differs')
    model.eval();model.config.use_cache=False
    data=read(ROOT/'data/MANIFEST.json');lookup={q['question_id']:q for q in data['rows']}
    cpu=read(ROOT/'data/CPU_INPUTS.json');updates=read(ROOT/'data/SCHEDULE.json')['updates']
    directory=ROOT/'reference';directory.mkdir(exist_ok=True)
    cache={'reference_lora_sha256':before,'batches':{}};started=time.monotonic()
    with torch.inference_mode():
        for step,update in enumerate(updates):
            for micro,ids in enumerate(update['micros']):
                rows=[lookup[q] for q in ids]
                if rows[0]['study_source']!='preference':continue
                batch,key,_,_=make_batch(processor,data,rows,cpu)
                gpu=to_device(batch,next(model.parameters()).device)
                values,counts=sequence_logps(model,gpu)
                if len(values)!=8 or values.dtype!=torch.float32 or not bool(torch.isfinite(values).all()):
                    raise ValueError('Invalid SFT reference sequence log-probabilities')
                cache['batches'][f'{step+1}:{micro}']={'binding':key,
                    'chosen_logps':values[:4].cpu().tolist(),'rejected_logps':values[4:].cpu().tolist(),
                    'assistant_counts':counts.cpu().tolist()}
                del gpu,values,counts,batch
                write_json(directory/'CACHE.json',cache)
                write_json(directory/'PROGRESS.json',{'status':'caching_reference','completed_batches':len(cache['batches']),
                    'expected_batches':64,'active_seconds':time.monotonic()-started})
                if time.monotonic()-started>c['maximum_reference_seconds']:raise TimeoutError('Reference-cache budget exceeded')
    if len(cache['batches'])!=64 or lora_hash(model)!=before:raise ValueError('Reference coverage or identity differs')
    write_json(directory/'COMPLETE.json',{'status':'completed','training_updates':0,'lora_sha256':before,
        'reference_batches':64,'reference_sequences':512,'active_seconds':time.monotonic()-started,'cache_sha256':sha_file(directory/'CACHE.json')})


def condition(role):
    from .train import train
    verify(True);model,processor,result=train(role)
    before=lora_hash(model)
    if before!=result['final_lora_sha256']:raise ValueError('Final identity differs')
    metrics=evaluate(model,processor,ROOT/'runs'/role)
    if lora_hash(model)!=before:raise ValueError('Evaluation changed Final')
    write_json(ROOT/'runs'/role/'COMPLETE.json',{'status':'completed','training_updates':32,
        'metrics':metrics,'final_lora_sha256':before,'training_active_seconds':result['active_seconds']})


def analyze():
    from vlm_sft_resolution.experiment import paired_interval
    metrics={'initial':read(ROOT/'baseline/COMPLETE.json')['metrics']}
    records={}
    for role in ('initial',)+ROLES:
        directory=ROOT/'baseline' if role=='initial' else ROOT/'runs'/role
        records[role]={name:read(directory/name/'PREDICTIONS.json') for name in ('hard','retention')}
        if role!='initial':metrics[role]=read(directory/'COMPLETE.json')['metrics']
    comparisons={};learning={}
    for role in ROLES:
        before,after=records['initial']['hard'],records[role]['hard']
        if [q['question_id'] for q in before]!=[q['question_id'] for q in after]:raise ValueError('Hard cohort unpaired')
        transform=lambda rows:[{'question_id':q['question_id'],'image_id':q['image_id'],'score':float(q['casefold_exact'])} for q in rows]
        fixed=sum(not a['casefold_exact'] and b['casefold_exact'] for a,b in zip(before,after))
        broken=sum(a['casefold_exact'] and not b['casefold_exact'] for a,b in zip(before,after))
        initial_anchor,anchor=records['initial']['retention'],records[role]['retention']
        if [q['question_id'] for q in initial_anchor]!=[q['question_id'] for q in anchor]:raise ValueError('Retention cohort unpaired')
        score_delta=sum(b['vqa_score']-a['vqa_score'] for a,b in zip(initial_anchor,anchor))
        new_anchor_errors=sum(b['vqa_score']<a['vqa_score'] for a,b in zip(initial_anchor,anchor))
        pair_delta=metrics[role]['hard']['both_correct']-metrics['initial']['hard']['both_correct']
        comparisons[role]={'hard_corrected':fixed,'hard_broken':broken,'hard_net':fixed-broken,
            'pair_net':pair_delta,'retention_score_points_delta':score_delta,'retention_new_lower_score_qa':new_anchor_errors,
            'training_image_cluster_95_interval':paired_interval(transform(before),transform(after),6000,20261020)}
        learning[role]=fixed-broken>=4 and pair_delta>=2 and score_delta>=-1.0 and new_anchor_errors<=2
    advantage=(learning['dpo_sft'] and metrics['dpo_sft']['hard']['casefold_correct']>=metrics['sft']['hard']['casefold_correct']+2
        and metrics['dpo_sft']['hard']['both_correct']>=metrics['sft']['hard']['both_correct']+2
        and metrics['dpo_sft']['retention']['score_points']>=metrics['sft']['retention']['score_points']-1.)
    result={'status':'completed','scope':'Training-only32-step learning control; no main benchmark, independent generalization, or resume gain',
        'metrics':metrics,'comparisons':comparisons,'learning_positive_control_passed':learning,
        'dpo_advantage_for_further_research':advantage,
        'next':'No automatic longer training; choose next objective from actual generation and retention, not DPO loss'}
    write_json(ROOT/'ANALYSIS.json',result);write_json(ROOT/'COMPLETION.json',result)
    lines=['# 区域偏好32步正控结果','',result['scope'],'','|模型|区域QA正确|同图pair全对|保持题VQA分|','|---|---:|---:|---:|']
    lines += [f'|{role}|{v["hard"]["casefold_correct"]}/{v["hard"]["qa"]}|{v["hard"]["both_correct"]}/{v["hard"]["pairs"]}|{v["retention"]["score_percent"]:.6f}|' for role,v in metrics.items()]
    lines += ['',f'训练内学习正控：{learning}；DPO相对同数据SFT优势准入：{advantage}。','完整纠错、区间、保持与边界见对应输出。']
    (PROJECT_ROOT/'docs/VLM_REGION_PREFERENCE_PROBE_V1_RESULTS.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--baseline',action='store_true');parser.add_argument('--reference',action='store_true')
    parser.add_argument('--role',choices=ROLES);args=parser.parse_args()
    if args.prepare:prepare();return
    if args.baseline:baseline();return
    if args.reference:reference();return
    if args.role:
        try:condition(args.role)
        except Exception as exc:
            write_json(ROOT/'runs'/args.role/'ERROR.json',{'status':'failed','error':repr(exc),'other_tasks_untouched':True})
            raise
        return
    verify()
    from research_runtime.queue import run_queue
    run_queue(ROOT/'initial_queue',[{'key':name,'command':[sys.executable,'-u','-m',__package__+'.study','--'+name],
        'completion_marker':str(ROOT/name/'COMPLETE.json')} for name in ('baseline','reference')],
        {'prelaunch_sha256':sha_file(ROOT/'PRELAUNCH_INPUT_LOCK.json')})
    baseline_hash=read(ROOT/'baseline/COMPLETE.json')['lora_sha256'];reference_hash=read(ROOT/'reference/COMPLETE.json')['lora_sha256']
    if baseline_hash!=reference_hash:raise ValueError('Initial and cached reference checkpoints differ')
    paths=[ROOT/'reference/CACHE.json',ROOT/'reference/COMPLETE.json',ROOT/'baseline/COMPLETE.json',
        ROOT/'baseline/hard/PREDICTIONS.json',ROOT/'baseline/retention/PREDICTIONS.json']
    value={'initial_lora_sha256':baseline_hash,'prelaunch_sha256':sha_file(ROOT/'PRELAUNCH_INPUT_LOCK.json'),
        'new_baseline_and_reference_files':{str(p):sha_file(p) for p in paths}}
    if (ROOT/'STUDY_INPUT_LOCK.json').exists() and read(ROOT/'STUDY_INPUT_LOCK.json')!=value:raise ValueError('Study baseline/reference lock differs')
    write_json(ROOT/'STUDY_INPUT_LOCK.json',value)
    run_queue(ROOT/'training_queue',[{'key':role,'command':[sys.executable,'-u','-m',__package__+'.study','--role',role],
        'completion_marker':str(ROOT/'runs'/role/'COMPLETE.json')} for role in ROLES],
        {'study_sha256':sha_file(ROOT/'STUDY_INPUT_LOCK.json')})
    analyze()


if __name__=='__main__':main()
