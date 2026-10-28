import argparse
from collections import Counter
import html
import os
from pathlib import Path
import sys
import time
from urllib.parse import quote
from rarevlm_paths import PROJECT_ROOT,SOURCE_ROOT
from vlm_deep.common import content_sha,sha_file,write_json
from .common import ROOT,MINING,ROLES,actor,chosen_sample,read
from .data import cell,order,schedule

PROTOCOL=PROJECT_ROOT/'docs/VLM_COVERAGE_SFT_V1_PROTOCOL.md'
RESULT=PROJECT_ROOT/'docs/VLM_COVERAGE_SFT_V1_RESULTS.md'
PAGE=PROJECT_ROOT/'docs/learning/扩大错误覆盖SFT_128与重复32.html'
OLD=ROOT.parent/'vlm_grpo_iteration_v1'


def render(result=None):
    data=read(ROOT/'DATA.json');subset=set(read(ROOT/'SCHEDULE.json')['repeat_subset'])
    relative=lambda p:quote(os.path.relpath(p,PAGE.parent),safe='/._-')
    cards=[]
    for q in data['hard_rows']:
        cards.append(f'<article><h3>q{q["question_id"]} · {html.escape(q["review_error_type"])} · 重复32子集={q["question_id"] in subset}</h3><img loading="lazy" src="{relative(data["images"][q["image_id"]]["path"])}" alt="原始训练图"><p>{html.escape(q["question"])}</p><p>原GT：{html.escape(q["target"])}；原始模型：{html.escape(q["prediction"])}</p><p>{html.escape(q["visible_evidence"])}</p></article>')
    status='准备/运行中，尚无本轮模型提升结论。' if result is None else html.escape(str(result['scores_percent']))+'；联合通过='+str(result['passed'])
    PAGE.write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>覆盖128与重复32</title><style>body{max-width:1100px;margin:30px auto;padding:0 18px;font:17px/1.7 system-ui;background:#f8fafc;color:#1e293b}article{padding:16px;margin:18px 0;background:white;border:1px solid #cbd5e1}img{width:100%;max-height:480px;object-fit:contain}pre{background:#e2e8f0;padding:16px;white-space:pre-wrap}.note{background:#fff7ed;padding:16px}</style><h1>扩大不同错误覆盖：128题与32题重复SFT</h1><p class="note">'+status+'</p><pre>两组相同：64次更新，512困难曝光＋512完全相同回放\n候选：128个不同可信错误，各4次\n控制：32个分层子集，按同分层slot循环，总512次\n每个困难slot：错误类型＋频次桶＋答案长短一致\nL_step = Σ16题 [−平均assistant内容与EOS的log概率] / 16</pre><p>控制并非每题严格16次，小层曝光按候选逐slot匹配。两组困难图片/具体难度/视觉token仍有差异，不能称等计算或抽象样本数唯一因果。控制和候选都用原始SFT起点，普通CE，无新增RL损失。</p><p>先前17题的GRPO没有超过强SFT。新128题来自固定前184个候选的原图复核（131支持、53排除/不确定），125种不同目标，每target最多2。这里只展示入选训练题，不把53/184外推全池噪声率。</p><p>新512验证及原64保持固定，初始预测同batch4原样复用；训练诊断保留矿池完整四题上下文，仅128题评分。Final64事前固定，不按曲线挑点。每步分源CE/梯度/耗时有追加日志，loss下降或训练满分不是泛化提升。</p>'+f'<p><a href="{relative(PROTOCOL)}">预登记</a> · <a href="{relative(RESULT)}">结果</a> · <a href="{relative(SOURCE_ROOT/"vlm_coverage_sft_v1/train.py")}">真实训练循环</a></p>'+''.join(cards)+'</html>')


def prepare():
    import torch
    from vlm_probe.evaluate import processor_for
    torch.set_num_threads(2)
    if (ROOT/'PLAN.json').exists():raise FileExistsError('Already frozen; inspect existing workflow')
    admission=read(MINING/'TRAINING_ADMISSION.json')
    if admission['status']!='passed' or not admission['training_allowed']:raise ValueError('Visible semantic admission not passed')
    approved=read(MINING/'REVIEWED_TRAINING.json');mine=read(MINING/'DATA.json');preds=read(MINING/'PREDICTIONS.json')
    hard=[{**q,'training_source':'hard'} for q in approved['rows']];hard_ids={q['question_id'] for q in hard}
    if len(hard)!=128 or len({q['image_id'] for q in hard})!=128:raise ValueError('Need128 unique approved errors')
    predictions={q['question_id']:q for q in preds}
    replay=[q for q in mine['rows'] if q['fold']=='training' and q['question_id'] not in hard_ids and predictions[q['question_id']]['vqa_score']>=.999999]
    replay.sort(key=lambda q:order('coverage_sft_v1:replay',q['question_id']))
    replay=[{**q,'training_source':'replay'} for q in replay[:512]]
    if len(replay)!=512:raise ValueError('Insufficient original-correct replay')
    sched=schedule(hard,replay);lookup={q['question_id']:q for q in hard+replay}
    for a,b in zip(sched['roles']['repeat32'],sched['roles']['coverage128']):
        for micro in (0,2):
            if a['micros'][micro]!=b['micros'][micro]:raise ValueError('Matched replay order differs')
        for micro in (1,3):
            if any(cell(lookup[x])!=cell(lookup[y]) for x,y in zip(a['micros'][micro],b['micros'][micro])):raise ValueError('Matched hard strata differ')
    validation=[{**q,'evaluation_scored':True} for q in mine['rows'] if q['fold']=='validation']
    context=[]
    for start in range(0,len(mine['rows']),4):
        chunk=mine['rows'][start:start+4]
        if any(q['question_id'] in hard_ids for q in chunk):context.extend({**q,'evaluation_scored':q['question_id'] in hard_ids} for q in chunk)
    old=read(OLD/'DATA.json');retention=[{**q,'evaluation_scored':True} for q in old['retention_eval']]
    images_all={**mine['images'],**old['images']};used={q['image_id'] for q in hard+replay+validation+context+retention}
    images={iid:images_all[iid] for iid in used}
    pixel=lambda q:images[q['image_id']].get('pixel_sha256',images[q['image_id']]['sha256'])
    sets=[{pixel(q) for q in rows} for rows in (hard,replay,validation,retention)]
    if any(sets[i]&sets[j] for i in range(4) for j in range(i+1,4)):raise ValueError('Training/replay/validation/retention pixel overlap')
    data={'training_rows':hard+replay,'hard_rows':hard,'images':images,'evaluations':{'training_context':context,'validation':validation,'retention':retention}}
    write_json(ROOT/'DATA.json',data);write_json(ROOT/'SCHEDULE.json',sched)
    for name,rows in [('training_context',context),('validation',validation)]:
        write_json(ROOT/'initial'/name/'PREDICTIONS.json',[{**predictions[q['question_id']],'evaluation_scored':q['evaluation_scored']} for q in rows])
    initial_ret=read(OLD/'initial/retention/PREDICTIONS.json')
    if [q['question_id'] for q in initial_ret]!=[q['question_id'] for q in retention]:raise ValueError('Retention reuse grouping differs')
    write_json(ROOT/'initial/retention/PREDICTIONS.json',[{**q,'evaluation_scored':True} for q in initial_ret])
    base=read(MINING/'PLAN.json');processor=processor_for(Path(base['model_root']),read(base['reference_config']),1024)
    cpu={};begun=time.monotonic()
    for index,q in enumerate(hard+replay):
        item=chosen_sample(processor,q,images)
        cpu[q['question_id']]={'input_ids':content_sha(item['input_ids'].tolist()),'labels':content_sha(item['labels'].tolist()),
            'visual_tokens':int(item['image_grid_thw'].prod(-1).sum())//4}
        if (index+1)%64==0:write_json(ROOT/'CPU_PROGRESS.json',{'status':'encoding_training','completed':index+1,'expected':640,'seconds':time.monotonic()-begun})
    write_json(ROOT/'CPU_INPUTS.json',cpu)
    write_json(ROOT/'BASELINE_REUSE.json',{'initial_adapter':base['adapter'],'lora_sha256':base['lora_sha256'],'batch_size':4,
        'original_whole_mining_batches_preserved':True,'new_baseline_model_calls':0,'validation_source':str(MINING/'PREDICTIONS.json'),
        'retention_source':str(OLD/'initial/retention/PREDICTIONS.json'),'training_context_qa':len(context),'scored_training_qa':128})
    paths=[PROTOCOL,Path(__file__),PROJECT_ROOT/'tools/run_coverage_sft_v1.py',PROJECT_ROOT/'tests/test_vlm_coverage_sft_v1.py',
        ROOT/'DATA.json',ROOT/'SCHEDULE.json',ROOT/'CPU_INPUTS.json',ROOT/'BASELINE_REUSE.json',MINING/'PLAN.json',MINING/'DATA.json',
        MINING/'PREDICTIONS.json',MINING/'TRAINING_ADMISSION.json',MINING/'VISIBLE_REVIEW.json',MINING/'REVIEWED_TRAINING.json',OLD/'PLAN.json',OLD/'DATA.json',OLD/'initial/retention/PREDICTIONS.json']
    paths+=list((SOURCE_ROOT/'vlm_coverage_sft_v1').glob('*.py'))
    paths+=[ROOT/'initial'/name/'PREDICTIONS.json' for name in data['evaluations']]
    paths+=[SOURCE_ROOT/f for f in ('vlm_grpo_iteration_v1/common.py','vlm_grpo_iteration_v1/study.py','vlm_sampled_signal_v1/trajectory.py',
        'vlm_fp32_readout_v1/model.py','vlm_fp32_adapter/model.py','vlm_runtime_metadata_v1/model.py','vlm_deep/common.py','vlm_sft/common.py',
        'vlm_object_train/train.py','vlm_probe/evaluate.py','vlm_human_sft/evaluate.py','vlm_deep/evaluate.py','vlm_attention_scope_repair_v1/generation.py','research_runtime/queue.py')]
    paths+=[Path(base['adapter'])/n for n in ('adapter_config.json','adapter_model.safetensors')]+[Path(base['reference_config'])]
    plan={k:base[k] for k in ('adapter','lora_sha256','model_root','reference_config','packages','base_weight_files')}
    plan.update(status='prepared',updates_per_role=64,roles=list(ROLES),unique_hard_per_role={'repeat32':32,'coverage128':128},
        chosen_exposures_per_role=1024,hard_exposures_per_role=512,replay_exposures_per_role=512,
        generation_qa_per_role=sum(len(rows) for rows in data['evaluations'].values()),training_context_qa=len(context),scored_training_qa=128,
        files={str(path):sha_file(path) for path in paths},images={v['path']:v['sha256'] for v in images.values()},automatic_more_training=False,automatic_main_evaluation=False)
    write_json(ROOT/'PLAN.json',plan)
    RESULT.write_text('# 匹配错误覆盖SFT：准备完成\n\n覆盖128与重复32各64步/1024相同预算，完整回放与困难slot分层匹配。640真实训练CPU编码完成；初始验证512和原64保持及训练原批上下文直接同条件复用。尚无新更新或能力结论。\n\n[预登记](VLM_COVERAGE_SFT_V1_PROTOCOL.md) · [128原图及对照子集](learning/扩大错误覆盖SFT_128与重复32.html)\n')
    render();print('PREPARED_MATCHED_COVERAGE',{'hard':128,'targets':admission['unique_normalized_targets'],'repeat':32,'replay':512,'strata':len(sched['strata']),'eval_qa_per_role':plan['generation_qa_per_role']},flush=True)


def verify():
    from vlm_object_train.train import package_lock
    plan=read(ROOT/'PLAN.json')
    for path,digest in {**plan['files'],**plan['images']}.items():
        if sha_file(path)!=digest:raise ValueError('Frozen coverage SFT input changed: '+path)
    if package_lock()!=plan['packages']:raise ValueError('Package versions changed')
    for path,value in plan['base_weight_files'].items():
        st=Path(path).stat()
        if value!={'bytes':st.st_size,'mtime_ns':st.st_mtime_ns}:raise ValueError('Base storage changed')
    return plan


def evaluate(model,processor,role):
    import torch
    from vlm_probe.evaluate import evaluate_batch
    from vlm_deep.evaluate import score_prediction
    from vlm_attention_scope_repair_v1.generation import raw_greedy_config
    data=read(ROOT/'DATA.json');directory=ROOT/'runs'/role;model.eval();model.config.use_cache=True
    model.generation_config=raw_greedy_config(model.generation_config,32);begun=time.monotonic()
    with torch.inference_mode():
        for name,rows in data['evaluations'].items():
            path=directory/name/'PREDICTIONS.json';preds=read(path) if path.exists() else []
            if len(preds)%4 or [q['question_id'] for q in preds]!=[q['question_id'] for q in rows[:len(preds)]]:raise ValueError('Evaluation batch grouping differs')
            for start in range(len(preds),len(rows),4):
                chunk=rows[start:start+4]
                answers,_=evaluate_batch(model,processor,chunk,data['images'],ROOT,next(model.parameters()).device,1024,processor.tokenizer.convert_tokens_to_ids('<|image_pad|>'),32)
                for q,a in zip(chunk,answers):
                    if a['input_tokens']!=q['expected_prefix_tokens'] or a['image_grid_thw']!=q['expected_grid']:raise ValueError('Actual inference frame differs')
                    preds.append({'question_id':q['question_id'],'image_id':q['image_id'],'target':q['target'],'stratum':q['stratum'],
                        'frequency_slice':q.get('frequency_slice'),'evaluation_scored':q['evaluation_scored'],**a,'vqa_score':score_prediction(a['prediction'],q,'textvqa')})
                write_json(path,preds);write_json(directory/'EVAL_PROGRESS.json',{'status':'evaluating','cohort':name,'completed_qa':len(preds),'expected_qa':len(rows),'active_seconds':time.monotonic()-begun})
                if time.monotonic()-begun>1200:raise TimeoutError('1200sec fixed evaluation budget')


def condition(role):
    from .train import train
    from vlm_human_sft.evaluate import lora_hash
    verify();model,processor,result=train(role)
    if lora_hash(model)!=result['final_lora_sha256']:raise ValueError('Final identity differs')
    evaluate(model,processor,role)
    write_json(ROOT/'runs'/role/'COMPLETE.json',{'status':'completed','train':result})


def scored(rows):return [q for q in rows if q['evaluation_scored']]
def percent(rows):return 100*sum(q['vqa_score'] for q in rows)/len(rows)


def analyze():
    from vlm_grpo_iteration_v1.study import paired
    records={r:{n:scored(read((ROOT/'initial' if r=='initial' else ROOT/'runs'/r)/n/'PREDICTIONS.json')) for n in ('training_context','validation','retention')} for r in ('initial',)+ROLES}
    scores={r:{n:percent(rows) for n,rows in parts.items()} for r,parts in records.items()}
    comparisons={r:paired(records[r]['validation'],records['coverage128']['validation']) for r in ('initial','repeat32')}
    gates={r+':overall':value['delta_pp']>=.5 and value['ci95'][0]>0 for r,value in comparisons.items()};slices={}
    for r in ('initial','repeat32'):
        for key,field in [('singleword','stratum'),('multiword','stratum'),('rare_1_5','frequency_slice'),('freq_6_20','frequency_slice')]:
            a=[q for q in records[r]['validation'] if q[field]==key];c=[q for q in records['coverage128']['validation'] if q[field]==key]
            value=paired(a,c);slices[r+':'+key]=value
            gates[r+':'+key]=value['delta_pp']>=(.5 if r=='initial' and key=='rare_1_5' else -.5)
        gates[r+':retention']=scores['coverage128']['retention']-scores[r]['retention']>=-.5
    full={r:sum(q['vqa_score']>=.999999 for q in parts['training_context']) for r,parts in records.items()}
    gates['training_net32']=full['coverage128']-full['initial']>=32
    changes=[]
    for a,c in zip(records['initial']['validation'],records['coverage128']['validation']):
        if a['vqa_score']!=c['vqa_score']:changes.append({'question_id':a['question_id'],'target':a['target'],'before':a['prediction'],'after':c['prediction'],'score_delta':c['vqa_score']-a['vqa_score']})
    result={'status':'completed','passed':all(gates.values()),'scores_percent':scores,'comparisons':comparisons,'slices':slices,'gates':gates,'training_full':full,'validation_changes':changes,
        'next':'Preregister independent confirmation/replication;no automatic main evaluation.' if all(gates.values()) else 'Stop this configuration;no steps/seeds/main/CV expansion.'}
    write_json(ROOT/'ANALYSIS.json',result)
    table='|模型|训练128满分|新512 VQA|保持64 VQA|\n|---|---:|---:|---:|\n'+''.join(f'|{r}|{full[r]}|{s["validation"]:.6f}|{s["retention"]:.6f}|\n' for r,s in scores.items())
    RESULT.write_text('# 匹配错误覆盖SFT完成\n\n'+table+'\n候选对初始与新控制：'+str(comparisons)+'\n\n联合门槛通过='+str(result['passed'])+'；'+result['next']+'\n\n两组各64更新/1024曝光，数据身份与难度/视觉token可不同，单seed42内部筛选，非等计算/盲测或三seed主benchmark。初始同条件预测复用，实际生成背景题不计128训练纠错。原分数与门槛不更改，不以语义审核改写成功。\n\n[预登记](VLM_COVERAGE_SFT_V1_PROTOCOL.md) · [128原图及子集](learning/扩大错误覆盖SFT_128与重复32.html)\n')
    render(result);write_json(ROOT/'COMPLETION.json',result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--role',choices=ROLES);a=p.parse_args()
    if a.prepare:prepare()
    elif a.role:condition(a.role)
    else:
        from research_runtime.queue import run_queue
        run_queue(ROOT/'queue',[{'key':r,'command':[sys.executable,'-u','-m','vlm_coverage_sft_v1.study','--role',r],
            'completion_marker':str(ROOT/'runs'/r/'COMPLETE.json')} for r in ROLES],{'plan_sha256':sha_file(ROOT/'PLAN.json')})
        print(analyze(),flush=True)


if __name__=='__main__':main()
