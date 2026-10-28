"""Larger prospective native-error pool; exact action IDs and whole-batch sharding."""
import argparse
from collections import Counter
import html
import os
from pathlib import Path
import sys
import time
from urllib.parse import quote
from rarevlm_paths import PROJECT_ROOT,SOURCE_ROOT
from vlm_deep.common import NORMALIZER,content_sha,encode,read_manifest,sha_file,write_json
from vlm_error_coverage_v1.data import assigned_batches
from vlm_grpo_iteration_v1.common import actor,read

ROOT=PROJECT_ROOT.parents[1]/'outputs/vlm_error_coverage_mining_v1'
OLD=ROOT.parent/'vlm_grpo_iteration_v1'
CURVE=ROOT.parent/'vlm_grpo_learning_curve_v1'
PROTOCOL=PROJECT_ROOT/'docs/VLM_ERROR_COVERAGE_MINING_V1_PROTOCOL.md'
RESULT=PROJECT_ROOT/'docs/VLM_ERROR_COVERAGE_MINING_V1_RESULTS.md'


def prepare():
    import torch
    from vlm_probe.evaluate import inference_row,processor_for
    torch.set_num_threads(2)
    if (ROOT/'PLAN.json').exists():raise FileExistsError('Already frozen; inspect existing workflow')
    old=read(OLD/'PLAN.json');source_path=ROOT.parent/'vlm_deep_v2/prepared/manifest.json'
    source=read_manifest(source_path)
    excluded={q['image_id'] for fold in ('dev','final') for q in source['splits'][fold]}
    exclusion_paths=[Path(f) for f in old['files'] if '/outputs/' in f and f.endswith('.json')
        and Path(f).name in ('DATA.json','MANIFEST.json','EVAL.json','CONTROL_TRAIN.json')]
    for path in exclusion_paths:
        d=read(path);excluded.update(d.get('images',{}))
        for key in ('rows','hard_eval','retention_eval','validation_eval'):
            excluded.update(q['image_id'] for q in d.get(key,[]))
    pixel=lambda im:im.get('pixel_sha256',im['sha256'])
    excluded_pixels={pixel(source['images'][iid]) for iid in excluded if iid in source['images']}
    frequency=Counter(NORMALIZER(q['target']) for q in source['splits']['train_all'])
    eligible=[q for q in source['splits']['train_all'] if q['image_id'] not in excluded and pixel(source['images'][q['image_id']]) not in excluded_pixels
        and Counter(q['answers'])[q['target']]>=8 and 1<=len(q['target'].split())<=6 and 1<=frequency[NORMALIZER(q['target'])]<=20]
    eligible.sort(key=lambda q:content_sha(['error_coverage_v1:selection',q['image_id'],q['question_id']]))
    selected=[];seen=set()
    for fold,name,count in [('validation','multiword',256),('training','multiword',256),
                            ('validation','singleword',256),('training','singleword',2816)]:
            bucket=[]
            for q in eligible:
                px=pixel(source['images'][q['image_id']])
                if px in seen or (len(q['target'].split())>1)!=(name=='multiword'):continue
                bucket.append({**q,'fold':fold,'stratum':name,'answer_frequency':frequency[NORMALIZER(q['target'])],
                    'frequency_slice':'rare_1_5' if frequency[NORMALIZER(q['target'])]<=5 else 'freq_6_20'})
                seen.add(px)
                if len(bucket)==count:break
            if len(bucket)!=count:raise ValueError('Insufficient independent '+fold+'/'+name+';no fallback')
            selected.extend(bucket)
    selected.sort(key=lambda q:(q['fold'],content_sha(['error_coverage_v1:execution',q['question_id']])))
    shards=[assigned_batches(selected,i) for i in (0,1)]
    if any(sum(map(len,s))!=1792 for s in shards):raise ValueError('Unbalanced fixed whole-batch sharding')
    processor=processor_for(Path(old['model_root']),read(old['reference_config']),1024)
    images={};started=time.monotonic()
    for index,q in enumerate(selected):
        im=source['images'][q['image_id']];path=(source_path.parent/im['path']).resolve()
        sample=encode(processor,inference_row(q),path,False)
        q.update(expected_prefix_tokens=int(sample['input_ids'].shape[1]),expected_grid=sample['image_grid_thw'][0].tolist(),inference_input_sha256=content_sha(sample['input_ids'].tolist()))
        images[q['image_id']]={**im,'path':str(path)}
        if (index+1)%64==0:write_json(ROOT/'CPU_PROGRESS.json',{'status':'encoding','completed_qa':index+1,'expected_qa':3584,'seconds':time.monotonic()-started})
    data={'rows':selected,'images':images};write_json(ROOT/'DATA.json',data)
    paths=[Path(__file__).resolve(),PROTOCOL,ROOT/'DATA.json',OLD/'PLAN.json',source_path,
        PROJECT_ROOT/'tests/test_vlm_error_coverage_v1.py',SOURCE_ROOT/'vlm_error_coverage_v1/data.py',SOURCE_ROOT/'vlm_error_coverage_v1/__init__.py']+exclusion_paths
    paths+=[SOURCE_ROOT/f for f in ('vlm_grpo_iteration_v1/common.py','vlm_fp32_adapter/model.py','vlm_fp32_readout_v1/model.py','vlm_runtime_metadata_v1/model.py',
        'vlm_sampled_signal_v1/trajectory.py','vlm_human_sft/evaluate.py','vlm_deep/common.py','vlm_deep/evaluate.py','vlm_sft/common.py',
        'vlm_probe/evaluate.py','vlm_attention_scope_repair_v1/generation.py','research_runtime/queue.py')]
    paths+=[Path(old['adapter'])/name for name in ('adapter_config.json','adapter_model.safetensors')]+[Path(old['reference_config'])]
    plan={k:old[k] for k in ('adapter','lora_sha256','model_root','reference_config','packages','base_weight_files')}
    plan.update(status='prepared',training_candidates=3072,validation_qa=512,unique_images=3584,worker_qa=1792,worker_active_cap_seconds=1500,
        training_updates=0,automatic_training=False,minimum_reviewed_errors=128,review_cap=256,files={str(p):sha_file(p) for p in paths},
        images={v['path']:v['sha256'] for v in images.values()},excluded_images=len(excluded),
        strata_counts=dict(Counter(q['fold']+':'+q['stratum'] for q in selected)),frequency_counts=dict(Counter(q['fold']+':'+q['frequency_slice'] for q in selected)),
        scope='Fixed new native-question train/validation pool within original SFT whitelist; not initial-unseen.')
    write_json(ROOT/'PLAN.json',plan)
    RESULT.write_text('# 扩大原生错误覆盖：准备完成，0训练\n\n3584唯一图/QA：3072训练候选（2816单词/256多词）、512验证（各256）。频次1–20，按1–5/6–20分别报告；旧图/诊断同pixel排除。实际CPU前缀编码完成，完整四题批次分片。\n\n先等只读纠错曲线结束再共享空闲准入，预计GPU阶段约15–25分钟/两卡并行，排队另计。自动筛选不等于可信错误，需128独立图可见审核及类别/target多样性才另立训练，0自动训练。\n\n[固定协议](VLM_ERROR_COVERAGE_MINING_V1_PROTOCOL.md)\n')
    print('PREPARED_NEW_COVERAGE',{'qa':3584,'unique_images':3584,'cpu_seconds':time.monotonic()-started,'strata':plan['strata_counts']},flush=True)


def verify():
    from vlm_object_train.train import package_lock
    plan=read(ROOT/'PLAN.json')
    for path,digest in {**plan['files'],**plan['images']}.items():
        if sha_file(path)!=digest:raise ValueError('Frozen coverage input changed: '+path)
    if package_lock()!=plan['packages']:raise ValueError('Package versions changed')
    for path,value in plan['base_weight_files'].items():
        st=Path(path).stat()
        if value!={'bytes':st.st_size,'mtime_ns':st.st_mtime_ns}:raise ValueError('Base storage changed')
    return plan


def worker(index):
    import torch
    from vlm_probe.evaluate import inference_row
    from vlm_deep.common import collate
    from vlm_sft.common import to_device
    from vlm_deep.evaluate import score_prediction
    from vlm_human_sft.evaluate import lora_hash
    from vlm_sampled_signal_v1.trajectory import stop_at_eos
    from vlm_attention_scope_repair_v1.generation import raw_greedy_config
    plan=verify();model,processor,_=actor()
    if lora_hash(model)!=plan['lora_sha256']:raise ValueError('Miner is not original SFT')
    model.eval();model.generation_config=raw_greedy_config(model.generation_config,32)
    eos=model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else list(eos)
    data=read(ROOT/'DATA.json');batches=assigned_batches(data['rows'],index)
    directory=ROOT/f'shard_{index}';path=directory/'PREDICTIONS.json';rows=read(path) if path.exists() else []
    flat=[q for g in batches for q in g]
    if len(rows)%4 or [q['question_id'] for q in rows]!=[q['question_id'] for q in flat[:len(rows)]]:raise ValueError('Mining resume batch changed')
    begun=time.monotonic();prior=read(directory/'PROGRESS.json').get('active_seconds',0.) if (directory/'PROGRESS.json').exists() else 0.
    torch.cuda.reset_peak_memory_stats()
    handle=model.get_output_embeddings().register_forward_pre_hook(lambda m,args:(args[0][:,-1:,:],))
    try:
        with torch.inference_mode():
            for chunk in batches[len(rows)//4:]:
                samples=[encode(processor,inference_row(q),Path(data['images'][q['image_id']]['path']),False) for q in chunk]
                for q,sample in zip(chunk,samples):
                    if content_sha(sample['input_ids'].tolist())!=q['inference_input_sha256']:raise ValueError('Actual miner prompt changed')
                batch=to_device(collate(samples,processor.tokenizer.pad_token_id),next(model.parameters()).device)
                output=model.generate(**batch,max_new_tokens=32,do_sample=False,use_cache=True)
                for i,q in enumerate(chunk):
                    ids,ended=stop_at_eos(output[i,batch['input_ids'].shape[1]:].tolist(),eos,32)
                    prediction=processor.tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False).strip()
                    rows.append({'question_id':q['question_id'],'image_id':q['image_id'],'fold':q['fold'],'stratum':q['stratum'],'frequency_slice':q['frequency_slice'],
                        'target':q['target'],'prediction':prediction,'completion_ids':ids,'terminated':ended,'vqa_score':score_prediction(prediction,q,'textvqa')})
                write_json(path,rows)
                active=prior+time.monotonic()-begun;peak=torch.cuda.max_memory_allocated()/2**30
                write_json(directory/'PROGRESS.json',{'status':'generating','completed_qa':len(rows),'expected_qa':1792,'active_seconds':active,'peak_allocated_gib':peak,'training_updates':0})
                if active>1500 or peak>65:raise RuntimeError('Registered mining budget exceeded')
                del batch,output,samples
    finally:handle.remove()
    if lora_hash(model)!=plan['lora_sha256']:raise ValueError('Read-only mining changed adapter')
    write_json(directory/'COMPLETE.json',{'status':'completed','qa':len(rows),'training_updates':0,'lora_unchanged':True,'active_seconds':prior+time.monotonic()-begun})


def relation(target,prediction):
    t,p=NORMALIZER(target),NORMALIZER(prediction)
    if not p:return 'empty'
    if t.startswith(p+' '):return 'missing_words'
    if p.startswith(t+' '):return 'extra_context'
    if len(t.split())==len(p.split())==1:
        previous=list(range(len(p)+1))
        for i,a in enumerate(t,1):
            current=[i]
            for j,b in enumerate(p,1):current.append(min(previous[j]+1,current[-1]+1,previous[j-1]+(a!=b)))
            previous=current
        if previous[-1]/max(len(t),len(p))<=.4:return 'possible_spelling'
    return 'different_text_or_target'


def summarize(rows):
    return {'qa':len(rows),'vqa_percent':100*sum(q['vqa_score'] for q in rows)/len(rows),'full':sum(q['vqa_score']>=.999999 for q in rows),'length_censored':sum(not q['terminated'] for q in rows)}


def analyze():
    from vlm_deep.common import consensus
    data=read(ROOT/'DATA.json');predictions={q['question_id']:q for i in (0,1) for q in read(ROOT/f'shard_{i}/PREDICTIONS.json')}
    if len(predictions)!=3584:raise ValueError('Incomplete coverage predictions')
    ordered=[predictions[q['question_id']] for q in data['rows']]
    candidates=[]
    for q in data['rows']:
        r=predictions[q['question_id']]
        if q['fold']!='training' or not r['terminated'] or r['vqa_score']>.3+1e-12 or consensus(q['target'],q['answers'])-r['vqa_score']<.7-1e-12 or NORMALIZER(r['prediction'])==NORMALIZER(q['target']):continue
        candidates.append({**q,**r,'error_type_candidate':relation(q['target'],r['prediction']),
            'review_sha256':content_sha(['error_coverage_v1:review',q['question_id']]),'semantic_support_status':'pending_original_image_review'})
    candidates.sort(key=lambda q:q['review_sha256'])
    write_json(ROOT/'PREDICTIONS.json',ordered);write_json(ROOT/'TRAINING_CANDIDATES.json',candidates)
    groups={fold:summarize([q for q in ordered if q['fold']==fold]) for fold in ('training','validation')}
    slices={fold+':'+key:summarize([q for q in ordered if q['fold']==fold and (q['stratum']==key or q['frequency_slice']==key)]) for fold in ('training','validation') for key in ('singleword','multiword','rare_1_5','freq_6_20')}
    result={'status':'completed','qa':3584,'training_updates':0,'metrics':groups,'slices':slices,'algorithmic_candidates':len(candidates),
        'error_type_candidates':dict(Counter(q['error_type_candidate'] for q in candidates)),'visibly_reviewed_errors':0,'training_allowed':False,
        'next':'Inspect fixed first256 original-image candidates;need128 diverse supported unique training errors. No automatic training.'}
    page=PROJECT_ROOT/'docs/learning/扩大错误覆盖_原图候选审核.html';link=lambda path:quote(os.path.relpath(path,page.parent),safe='/._-')
    cards=[]
    for q in candidates[:256]:
        cards.append(f'<article><h3>q{q["question_id"]} · {html.escape(q["error_type_candidate"])}</h3><img loading="lazy" src="{link(data["images"][q["image_id"]]["path"])}" alt="训练候选原图"><p>{html.escape(q["question"])}</p><p>原GT：{html.escape(q["target"])}；实际原始SFT：{html.escape(q["prediction"])}；原分{q["vqa_score"]:.1f}</p><p>审核pending，低分不自动当作语义错；需看目标、别名、完整文本和边缘截断。</p></article>')
    page.write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>扩大原生错误覆盖</title><style>body{max-width:1080px;margin:30px auto;padding:0 18px;font:17px/1.7 system-ui;background:#f8fafc}article{padding:16px;background:white;border:1px solid #cbd5e1;margin:18px 0}img{width:100%;max-height:540px;object-fit:contain}</style><h1>扩大原生错误覆盖：固定训练候选原图</h1><p>3584真实原始SFT生成，0模型更新。以下只展示训练候选，验证错误不展示/回填。分类是算法代理，尚未完成语义审核，不称新增能力。</p>'+''.join(cards)+'</html>')
    RESULT.write_text('# 扩大原生错误覆盖：生成已完成，待语义审核\n\n'+str(groups)+'\n\n算法候选'+str(len(candidates))+'，类型'+str(result['error_type_candidates'])+'；0已审核/0更新，training_allowed=false。至少128不同图像的可信错误及类别/target多样性才考虑后续独立强SFT对照，不凑负例/改GT。\n\n[固定协议](VLM_ERROR_COVERAGE_MINING_V1_PROTOCOL.md) · [固定原图候选（待审核）](learning/扩大错误覆盖_原图候选审核.html)\n')
    write_json(ROOT/'COMPLETION.json',result);print(result,flush=True)


def controller():
    from research_runtime.queue import run_queue
    while not (CURVE/'COMPLETION.json').exists():
        p=CURVE/'queue/STATUS.json'
        if p.exists() and read(p)['status']=='failed':raise RuntimeError('Curve failed;inspect before subsequent GPU work')
        write_json(ROOT/'WORKFLOW_STATUS.json',{'status':'waiting_learning_curve','controller_pid':os.getpid(),'training_updates':0});time.sleep(15)
    write_json(ROOT/'WORKFLOW_STATUS.json',{'status':'waiting_shared_gpu','controller_pid':os.getpid(),'training_updates':0})
    run_queue(ROOT/'queue',[{'key':f'shard_{i}','command':[sys.executable,'-u',str(Path(__file__).resolve()),'--shard',str(i)],
        'completion_marker':str(ROOT/f'shard_{i}/COMPLETE.json')} for i in (0,1)],{'plan_sha256':sha_file(ROOT/'PLAN.json')})
    analyze();write_json(ROOT/'WORKFLOW_STATUS.json',{'status':'completed','training_updates':0})


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--shard',type=int,choices=[0,1]);a=p.parse_args()
    if a.prepare:prepare()
    elif a.shard is not None:worker(a.shard)
    else:controller()


if __name__=='__main__':main()
