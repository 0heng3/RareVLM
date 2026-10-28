"""Fixed32 pilot: worst-assistant-token CE vs a saved matched mean-CE control."""
import argparse
from collections import Counter
import html
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time
from urllib.parse import quote

from rarevlm_paths import PROJECT_ROOT, SOURCE_ROOT
from vlm_deep.common import content_sha, sha_file, write_json
from vlm_grpo_iteration_v1.common import actor, chosen_sample, read
from run_literal_tile_v1 import canonical
import run_literal_case_sft_v1 as core

BASE = PROJECT_ROOT.parents[1] / 'outputs'
ROOT = BASE / 'vlm_literal_token_focus_v1'
OLD = BASE / 'vlm_literal_case_sft_v1'
PROTOCOL = PROJECT_ROOT / 'docs/VLM_LITERAL_TOKEN_FOCUS_V1_PROTOCOL.md'
RESULT = PROJECT_ROOT / 'docs/VLM_LITERAL_TOKEN_FOCUS_V1_RESULTS.md'
PAGE = PROJECT_ROOT / 'docs/learning/最难词元损失_平均概率与自由生成.html'
NAMES = ('hard','clean_literal','literal_validation','seen_validation','retention')
core.ROOT = ROOT


def training_row(q):
    return {**q,'target':canonical(q['target'])} if q['study_source']=='preference' else q


def reductions(logits, labels):
    import torch
    import torch.nn.functional as F
    targets = labels[:, -logits.shape[1]:][:, 1:]
    valid = targets.ne(-100); counts = valid.sum(1)
    if not torch.equal(counts, labels.ne(-100).sum(1)) or not bool(counts.gt(0).all()):
        raise ValueError('Loss readout lost assistant tokens')
    nll = F.cross_entropy(logits[:, :-1].float().transpose(1,2), targets, reduction='none', ignore_index=-100)
    return nll.sum(1)/counts, nll.masked_fill(~valid,-float('inf')).max(1).values, counts


def forward_logits(model, batch):
    from vlm_sampled_signal_v1.trajectory import readout_tail_length
    keep = readout_tail_length(batch['labels'])
    handle = model.get_output_embeddings().register_forward_pre_hook(lambda module,args:(args[0][:,-keep:,:],))
    try:
        return model(**{k:v for k,v in batch.items() if k!='labels'},use_cache=False).logits
    finally:
        handle.remove()


def samples_for(processor, rows, data, cpu):
    samples=[]
    for q in rows:
        item=chosen_sample(processor,training_row(q),data['images'])
        if any(content_sha(item[k].tolist())!=cpu[q['question_id']][k] for k in ('input_ids','labels')):
            raise ValueError('Actual canonical input/labels changed')
        samples.append(item)
    return samples


def prepare():
    import torch
    from vlm_sampled_signal_v1.trajectory import completion_logps
    from vlm_object_train.train import learning_rate
    if (ROOT/'PLAN.json').exists(): raise FileExistsError('Study already frozen')
    torch.set_num_threads(2); torch.manual_seed(42)
    labels=torch.tensor([[-100,-100,1,2,5],[-100,-100,-100,3,5]])
    logits=torch.randn(2,4,6,requires_grad=True)
    means,maxima,counts=reductions(logits,labels); old,old_counts=completion_logps(logits,labels)
    assert torch.equal(counts,old_counts) and torch.allclose(means,-old/counts,atol=1e-6,rtol=1e-6)
    assert bool((maxima>=means).all())
    maxima.sum().backward(); active=logits.grad.abs().sum(-1).gt(0)
    assert active.sum(1).tolist()==[1,1] and not bool(active[:,-1].any()) and not bool(active[1,0])
    cpu_check={'status':'passed','mean_matches_established_sequence_formula':True,'max_ge_mean':True,'one_hard_position_per_qa_receives_max_gradient':True,'masked_and_unused_positions_have_no_gradient':True}
    checkpoints=list((OLD/'runs/canonical_case/train').glob('checkpoint_000032_*/adapter'))
    if len(checkpoints)!=1: raise ValueError('Need one closed canonical checkpoint32')
    checkpoint=checkpoints[0]; lock=read(OLD/'runs/canonical_case/probes/step32/INPUT_LOCK.json')
    data=read(OLD/'DATA.json'); cpu=read(OLD/'CPU_INPUTS.json')['canonical_case']; schedule={'updates':read(OLD/'SCHEDULE.json')['updates'][:32]}
    exposure=Counter(qid for u in schedule['updates'] for m in u['micros'] for qid in m)
    if any(exposure[q['question_id']]!=(2 if q['study_source']=='preference' else 1) for q in data['rows']):
        raise ValueError('32-step exposure differs')
    state=torch.load(checkpoint.parent/'state.pt',map_location='cpu',weights_only=False)
    if state['completed_updates']!=32 or state['accumulated_questions']!=0 or state['stats']['sources']!={'common':256,'preference':256}:
        raise ValueError('Saved control is not matched closed step32')
    if state['optimizer']['param_groups'][0]['lr']!=learning_rate(31,128,{'lr':2e-4,'warmup_ratio':8/128}):
        raise ValueError('Saved control learning-rate horizon differs')
    write_json(ROOT/'DATA.json',data);write_json(ROOT/'CPU_INPUTS.json',cpu);write_json(ROOT/'SCHEDULE.json',schedule);write_json(ROOT/'LOSS_CHECK.json',cpu_check)
    for name in NAMES:
        write_json(ROOT/'baseline'/name/'PREDICTIONS.json',read(OLD/'baseline'/name/'PREDICTIONS.json'))
    write_json(ROOT/'runs/mean32/hard/PREDICTIONS.json',read(OLD/'runs/canonical_case/probes/step32/hard/PREDICTIONS.json'))
    plan=read(OLD/'PLAN.json'); files=dict(plan['files'])
    paths=[Path(__file__),PROTOCOL,SOURCE_ROOT/'vlm_fit_positive_control_v1/study.py',ROOT/'DATA.json',ROOT/'CPU_INPUTS.json',ROOT/'SCHEDULE.json',ROOT/'LOSS_CHECK.json',
           checkpoint/'adapter_model.safetensors',checkpoint/'adapter_config.json',checkpoint.parent/'state.pt',OLD/'runs/canonical_case/probes/step32/INPUT_LOCK.json',
           OLD/'runs/canonical_case/train/TRAIN_COMPLETE.json',OLD/'runs/canonical_case/hard/PREDICTIONS.json',ROOT/'runs/mean32/hard/PREDICTIONS.json']
    paths+=list((ROOT/'baseline').glob('*/PREDICTIONS.json'))
    for path in paths: files[str(path)]=sha_file(path)
    previews=sorted(data['evaluations']['hard'],key=lambda q:content_sha(['token_focus:preview',q['question_id']]))[:8]
    plan.update(files=files,roles=['mean32','max32'],updates=32,lr_horizon=128,warmup_updates=8,lr_schedule='linear_decay_after_linear_warmup_not_cosine',
                control_adapter=str(checkpoint),control_lora_sha256=lock['model_lora_sha256'],control_optimizer_updates_reused=32,
                new_optimizer_updates=32,new_generations={'mean32':960,'max32':1088},diagnostic_teacher_forced_qa=128,
                preview_questions=[q['question_id'] for q in previews])
    write_json(ROOT/'PLAN.json',plan);render();print('PREPARED',{'loss_check':cpu_check,'reused_checkpoint32':str(checkpoint),'new_updates':32,'new_generations':2048},flush=True)


def diagnostic():
    import torch
    from vlm_deep.common import collate
    from vlm_sft.common import to_device
    from vlm_human_sft.evaluate import lora_hash
    from vlm_object_train.train import lock_json
    from vlm_fit_positive_control_v1.study import token_statistics
    core.verify(); data=read(ROOT/'DATA.json');cpu=read(ROOT/'CPU_INPUTS.json')
    control=read(OLD/'runs/canonical_case/train/TRAIN_COMPLETE.json');model,processor,_=actor(adapter=control['adapter'])
    if lora_hash(model)!=control['final_lora_sha256']: raise ValueError('Wrong fixed128 model')
    directory=ROOT/'diagnostic';lock_json(directory/'INPUT_LOCK.json',{'model_lora_sha256':lora_hash(model),'plan_sha256':sha_file(ROOT/'PLAN.json')})
    path=directory/'TOKENS.json';records=read(path) if path.exists() else [];rows=data['evaluations']['hard'];start=time.monotonic()
    if len(records)>128 or len(records)%4 or [q['question_id'] for q in records]!=[q['question_id'] for q in rows[:len(records)]]:
        raise ValueError('Diagnostic resume differs')
    eos=processor.tokenizer.convert_tokens_to_ids('<|im_end|>');model.eval();model.config.use_cache=False
    with torch.inference_mode():
        for begin in range(len(records),128,4):
            chunk=rows[begin:begin+4];batch=to_device(collate(samples_for(processor,chunk,data,cpu),processor.tokenizer.pad_token_id),next(model.parameters()).device)
            logits=forward_logits(model,batch);values=token_statistics(logits,batch['labels'],eos)
            for q,value in zip(chunk,values):
                value.update(question_id=q['question_id'],target=q['target'],training_target=training_row(q)['target'],
                             token_fragments=[processor.tokenizer.decode([i],skip_special_tokens=False,clean_up_tokenization_spaces=False) for i in value['target_ids']])
                records.append(value)
            write_json(path,records);write_json(directory/'PROGRESS.json',{'status':'evaluating','completed_qa':len(records),'expected_qa':128,'active_seconds':time.monotonic()-start})
            if time.monotonic()-start>300 or torch.cuda.max_memory_allocated()/2**30>65: raise RuntimeError('Diagnostic cap')
            del batch,logits
    if lora_hash(model)!=control['final_lora_sha256']: raise ValueError('Read-only model changed')
    old={q['question_id']:q for q in read(OLD/'runs/canonical_case/hard/PREDICTIONS.json')};wrong=[q for q in records if not old[q['question_id']]['literal_exact']]
    low=[q for q in wrong if q['all_ce']<=1 and min(q['target_minus_best_rival'][:-1])<0]
    median_ratio=statistics.median(max(-v for v in q['target_logps'])/max(q['all_ce'],1e-12) for q in low) if low else 0
    eos_median=statistics.median(q['eos_ce'] for q in wrong)
    result={'status':'completed','qa':128,'wrong_generated_words':len(wrong),'low_mean_ce_but_wrong_content_argmax':len(low),
            'low_group_median_worst_over_mean':median_ratio,'wrong_group_median_eos_ce':eos_median,
            'training_admitted':len(low)>=16 and median_ratio>=1.5 and eos_median<.1,'optimizer_updates':0,
            'scope':'Teacher-forced training words only; no generalization evidence or uniquely identified root cause.'}
    write_json(directory/'COMPLETE.json',result);print('TOKEN_DIAGNOSTIC',result,flush=True)


def train():
    import torch
    import numpy as np
    from vlm_deep.common import collate
    from vlm_sft.common import to_device
    from vlm_sampled_signal_v1.trajectory import completion_logps
    from vlm_human_sft.evaluate import lora_hash
    from vlm_object_train.train import learning_rate,lock_json,save_checkpoint,resume_checkpoint
    plan=core.verify();data=read(ROOT/'DATA.json');cpu=read(ROOT/'CPU_INPUTS.json');schedule=read(ROOT/'SCHEDULE.json')['updates'];rows={q['question_id']:q for q in data['rows']}
    directory=ROOT/'runs/max32/train';directory.mkdir(parents=True,exist_ok=True);closed=directory/'TRAIN_COMPLETE.json'
    if closed.exists():
        result=read(closed);model,processor,_=actor(adapter=result['adapter']);return model,processor,result
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    model,processor,_=actor(training=True)
    if lora_hash(model)!=plan['initial_lora_sha256']: raise ValueError('Fresh original SFT required')
    parameters=[p for p in model.parameters() if p.requires_grad]
    if sum(p.numel() for p in parameters)!=1843200 or any(p.dtype!=torch.float32 for p in parameters): raise ValueError('Wrong FP32 trainables')
    lineage={'role':'max32','plan_sha256':sha_file(ROOT/'PLAN.json'),'schedule_sha256':sha_file(ROOT/'SCHEDULE.json'),'gpu_uuid':os.environ['CUDA_VISIBLE_DEVICES']}
    lock_json(directory/'INPUT_LOCK.json',lineage);lineage_sha=content_sha(lineage);optimizer=torch.optim.AdamW(parameters,lr=2e-4,weight_decay=0.)
    stats={'active_seconds':0.,'trace_bytes':0,'sources':{'common':0,'preference':0},'assistant_tokens':0,'per_question':{}};completed=0
    if (directory/'LATEST.json').exists(): completed,stats=resume_checkpoint(directory,model,optimizer,lineage_sha,lineage['schedule_sha256'],32)
    path=directory/'INPUT_TRACES.jsonl';path.touch(exist_ok=True);prior=stats['active_seconds'];started=time.monotonic();durations=[]
    with path.open('r+b') as trace:
        if path.stat().st_size<stats['trace_bytes']: raise ValueError('Saved trace truncated')
        trace.truncate(stats['trace_bytes']);trace.seek(0,os.SEEK_END)
        def checkpoint(step):
            stats.update(active_seconds=prior+time.monotonic()-started,trace_bytes=trace.tell());trace.flush();os.fsync(trace.fileno());save_checkpoint(directory,model,optimizer,step,stats,lineage_sha,lineage['schedule_sha256'])
        for step in range(completed,32):
            if prior+time.monotonic()-started>900: checkpoint(step);raise TimeoutError('Training cap900sec')
            begin=time.monotonic();model.train();model.get_base_model().visual.eval();model.config.use_cache=False;optimizer.zero_grad(set_to_none=True)
            lr=learning_rate(step,128,{'lr':2e-4,'warmup_ratio':8/128})
            for group in optimizer.param_groups: group['lr']=lr
            totals={'common_ce':0.,'hard_ce':0.,'hard_optimized':0.};first_errors=[]
            for micro,ids in enumerate(schedule[step]['micros']):
                chunk=[rows[qid] for qid in ids];samples=samples_for(processor,chunk,data,cpu)
                batch=to_device(collate(samples,processor.tokenizer.pad_token_id),next(model.parameters()).device)
                logits=forward_logits(model,batch);means,maxima,counts=reductions(logits,batch['labels'])
                if not bool(torch.isfinite(means).all()) or not bool(torch.isfinite(maxima).all()): raise FloatingPointError('Nonfinite token CE')
                if step==0:
                    old,old_counts=completion_logps(logits,batch['labels']);error=float((means+old/old_counts).abs().max());first_errors.append(error)
                    if error>1e-6 or not torch.equal(counts,old_counts): raise ValueError('Actual mean differs from established formula')
                    del old,old_counts
                source=chunk[0]['study_source']
                if any(q['study_source']!=source for q in chunk): raise ValueError('Unexpected mixed-source micro')
                optimized=maxima if source=='preference' else means
                (optimized.sum()/16).backward()
                for q,mean,value,count in zip(chunk,means.detach().cpu().tolist(),optimized.detach().cpu().tolist(),counts.tolist()):
                    totals['hard_ce' if source=='preference' else 'common_ce']+=mean
                    if source=='preference': totals['hard_optimized']+=value
                    stats['sources'][source]+=1;stats['assistant_tokens']+=int(count);stats['per_question'][q['question_id']]=stats['per_question'].get(q['question_id'],0)+1
                    trace.write((json.dumps({'update':step+1,'micro':micro,'question_id':q['question_id'],'source':source,'chosen_ce':mean,'optimized_loss':value,'assistant_tokens':int(count),'input_sha256':cpu[q['question_id']]['input_ids'],'labels_sha256':cpu[q['question_id']]['labels']})+'\n').encode())
                del logits,batch,means,maxima,counts,optimized,samples
            norm=float(torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True))
            if norm<=0: raise ValueError('No effective learning gradient')
            optimizer.step();torch.cuda.synchronize();durations.append(time.monotonic()-begin)
            if step==0 or (step+1)%8==0: checkpoint(step+1)
            progress={'status':'training','completed_updates':step+1,'planned_updates':32,'lr':lr,'lr_horizon':128,'gradient_norm':norm,
                      'hard_mean_ce':totals['hard_ce']/8,'hard_max_ce':totals['hard_optimized']/8,'native_mean_ce':totals['common_ce']/8,
                      'recent_seconds_per_update':statistics.mean(durations[-8:]),'active_seconds':prior+time.monotonic()-started,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30}
            write_json(directory/'PROGRESS.json',progress)
            if step==0: write_json(directory/'FIRST_UPDATE.json',{**progress,'all_actual_cpu_labels_matched':True,'mean_formula_max_errors':first_errors})
            if progress['peak_allocated_gib']>65: checkpoint(step+1);raise RuntimeError('Memory cap65GiB')
    if stats['sources']!={'common':256,'preference':256}: raise ValueError('32-step source budget differs')
    if any(stats['per_question'][q['question_id']]!=(2 if q['study_source']=='preference' else 1) for q in data['rows']): raise ValueError('Per-question budget differs')
    destination=directory/'adapter';model.save_pretrained(destination,safe_serialization=True)
    result={'status':'completed','updates':32,'adapter':str(destination),'final_lora_sha256':lora_hash(model),'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,**stats}
    write_json(closed,result);write_json(directory/'PROGRESS.json',result);return model,processor,result


def condition(role):
    from vlm_human_sft.evaluate import lora_hash
    plan=core.verify()
    if role=='mean32':
        model,processor,_=actor(adapter=plan['control_adapter']);expected=plan['control_lora_sha256'];names=NAMES[1:]
    elif role=='max32':
        model,processor,result=train();expected=result['final_lora_sha256'];names=NAMES
    else: raise ValueError('Unknown condition')
    if lora_hash(model)!=expected: raise ValueError('Wrong condition weights')
    core.evaluate(model,processor,ROOT/'runs'/role,names)
    if lora_hash(model)!=expected: raise ValueError('Read-only evaluation changed weights')
    write_json(ROOT/'runs'/role/'COMPLETE.json',{'status':'completed','role':role,'updates':32,'optimizer_updates_new':32 if role=='max32' else 0,'final_lora_sha256':expected})


def analyze():
    from vlm_grpo_iteration_v1.study import paired
    literal={'hard','clean_literal','literal_validation'};records={role:{n:read(path/n/'PREDICTIONS.json') for n in NAMES} for role,path in [('initial',ROOT/'baseline'),('mean32',ROOT/'runs/mean32'),('max32',ROOT/'runs/max32')]}
    metrics={role:{n:{'qa':len(rows),'full':sum(float(q['literal_exact']) if n in literal else q['vqa_score']>=.999999 for q in rows),'score_percent':100*sum(float(q['literal_exact']) if n in literal else q['vqa_score'] for q in rows)/len(rows)} for n,rows in group.items()} for role,group in records.items()}
    def pair(a,b,n):
        normalize=lambda xs:[{**q,'vqa_score':float(q['literal_exact']) if n in literal else q['vqa_score']} for q in xs]
        return paired(normalize(records[a][n]),normalize(records[b][n]))
    initial={n:pair('initial','max32',n) for n in NAMES};matched={n:pair('mean32','max32',n) for n in NAMES};seen={q['question_id']:q for q in read(ROOT/'DATA.json')['evaluations']['seen_validation']};slices={}
    for group in ('singleword','multiword','rare_1_5'):
        ids={qid for qid,q in seen.items() if (q['stratum']==group if group!='rare_1_5' else q.get('answer_frequency',100)<=5)}
        slices[group]=paired([q for q in records['initial']['seen_validation'] if q['question_id'] in ids],[q for q in records['max32']['seen_validation'] if q['question_id'] in ids])
    protection=initial['literal_validation']['delta_pp']>=-.5 and initial['retention']['delta_pp']>=-.5 and all(x['delta_pp']>=-.5 for x in slices.values())
    pilot={'fit_gain_at_least16':metrics['max32']['hard']['full']-metrics['mean32']['hard']['full']>=16,
           'word_preservation':initial['clean_literal']['delta_pp']>=0 and matched['clean_literal']['delta_pp']>=0,
           'native_preservation':initial['seen_validation']['delta_pp']>=0 and matched['seen_validation']['delta_pp']>=0,'protections':protection}
    ability={'fitting_learnable':metrics['max32']['hard']['full']>=96,
             'new_word_gain':initial['clean_literal']['delta_pp']>=3 and initial['clean_literal']['ci95'][0]>0,
             'native_gain':initial['seen_validation']['delta_pp']>=.5 and initial['seen_validation']['ci95'][0]>0,'protections':protection}
    result={'status':'completed','metrics':metrics,'vs_initial':initial,'vs_mean32':matched,'native_slices':slices,'pilot_checks':pilot,'pilot_forward_signal':all(pilot.values()),
            'ability_checks':ability,'ability_screen_passed':all(ability.values()),'optimizer_updates_new':32,'optimizer_updates_control_reused':32,'new_generations':2048,
            'automatic_long_training':False,'goal_complete':False,'scope':'Single-seed internal32-update algorithm pilot; conditional image-cluster CI, not blind/seed-stable ability confirmation.'}
    write_json(ROOT/'ANALYSIS.json',result);render(result);return result


def render(result=None):
    data=read(ROOT/'DATA.json');plan=read(ROOT/'PLAN.json');meta={q['question_id']:q for q in data['evaluations']['hard']};link=lambda p:quote(os.path.relpath(p,PAGE.parent),safe='/._-')
    state='已准备固定32步先导；尚无新能力结果。';diagnostic=ROOT/'diagnostic/COMPLETE.json';tokens={}
    if diagnostic.exists():
        d=read(diagnostic);state=f'固定模型{d["wrong_generated_words"]}词生成错误；其中{d["low_mean_ce_but_wrong_content_argmax"]}词平均CE<=1但正文GT未胜出；EOS CE中位数{d["wrong_group_median_eos_ce"]:.6f}。训练准入={d["training_admitted"]}。'
        tokens={q['question_id']:q for q in read(ROOT/'diagnostic/TOKENS.json')}
    table=''
    if result:
        table='|模型|训练词/128|新词验证%|旧词验证%|native512%|保持64%|\n|---|---:|---:|---:|---:|---:|\n'
        for role,m in result['metrics'].items(): table+='|'+role+'|'+str(m['hard']['full'])+'|'+ '|'.join(f'{m[n]["score_percent"]:.6f}' for n in NAMES[1:])+'|\n'
        state+=f' 32步推进信号={result["pilot_forward_signal"]}；原能力门槛={result["ability_screen_passed"]}；不自动长训。'
    RESULT.write_text('# 最难词元损失：32步先导\n\n'+state+'\n\n'+table+'\n[协议](VLM_LITERAL_TOKEN_FOCUS_V1_PROTOCOL.md) · [图像、词元概率与源码](learning/最难词元损失_平均概率与自由生成.html)\n')
    cards=[]
    for qid in plan['preview_questions']:
        q=meta[qid];detail='';outputs=[]
        if qid in tokens:
            t=tokens[qid];detail='<table><tr><th>目标片段</th><th>NLL</th><th>GT rank</th><th>GT−最佳竞争项</th></tr>'+''.join('<tr><td>'+html.escape(f)+'</td><td>'+f'{-lp:.4f}'+'</td><td>'+str(rank)+'</td><td>'+f'{margin:.4f}'+'</td></tr>' for f,lp,rank,margin in zip(t['token_fragments'],t['target_logps'],t['target_ranks'],t['target_minus_best_rival']))+'</table><p>以上是旧Final128在正确前缀下的诊断，非新32步自由生成或视觉attention。</p>'
        for role in ('mean32','max32'):
            path=ROOT/'runs'/role/'hard/PREDICTIONS.json'
            if path.exists():
                a=next((x for x in read(path) if x['question_id']==qid),None)
                if a: outputs.append(role+'：'+html.escape(a['prediction'])+' / '+str(a['literal_exact']))
        cards.append('<article><h2>'+html.escape(q['target'])+'</h2><img src="'+link(data['images'][q['image_id']]['path'])+'" alt="实际训练词图"><p>初始错误 '+html.escape(q['rejected_target'])+'；canonical训练 '+html.escape(training_row(q)['target'])+'。</p>'+detail+'<p>'+'；'.join(outputs)+'</p></article>')
    PAGE.write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>最难词元与平均损失</title><style>body{max-width:1060px;margin:30px auto;padding:0 18px;font:17px/1.7 system-ui}article{padding:16px;border:1px solid #cbd5e1;margin:18px 0}img{width:480px;max-width:100%}table{border-collapse:collapse}td,th{border:1px solid #cbd5e1;padding:6px}pre{white-space:pre-wrap;background:#f1f5f9;padding:16px}</style><h1>平均loss下降，为什么一个词仍然拼错？</h1><p>'+html.escape(state)+'</p><p>teacher forcing把正确前缀交给模型，平均CE又把不同位置平均。greedy每一步选当前最高概率项，一个关键位置落后就可能产生整个错误词。新实验只改变纠错词内部如何聚合loss：让当前最难位置直接获得梯度，回放仍用平均值。最难位置变化可能振荡，效果必须由完整对照验证。</p><pre>mean32: CE_q = mean_t NLL(q,t)\nmax32:  H_q  = max_t  NLL(q,t)  # 仅hard词，含实际EOS\nL_step = [sum(native8 CE_q) + sum(hard8 H_q)] / 16\n# prompt/padding不参与；native和mean32都保持均值\n# 同512QA/32更新；两条件都是旧128步LR日程的前32步</pre><p>原32步控制直接复用已保存FP32 checkpoint，避免重训；新max从原SFT fresh开始。128词均为已审核训练样本，下面8词预先按SHA固定。图片是模型实际ROI输入，页面缩放不增加细节。所有原图曾进原SFT，训练拟合和词元概率不等于泛化。</p><p><a href="'+link(PROTOCOL)+'">协议与门槛</a> · <a href="'+link(Path(__file__).resolve())+'">实际loss与训练代码</a> · <a href="'+link(SOURCE_ROOT/'vlm_fit_positive_control_v1/study.py')+'">复用的词元诊断</a></p>'+''.join(cards)+'<pre>'+html.escape(table)+'</pre></html>')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--worker');args=parser.parse_args()
    if args.prepare: prepare();return
    if args.worker:
        if args.worker=='diagnostic': diagnostic()
        else: condition(args.worker)
        return
    from research_runtime.queue import run_queue
    core.verify();identity={'plan_sha256':sha_file(ROOT/'PLAN.json')};entry=str(Path(__file__).resolve())
    job=lambda key,marker:{'key':key,'command':[sys.executable,'-u',entry,'--worker',key],'completion_marker':str(marker)}
    run_queue(ROOT/'diagnostic_queue',[job('diagnostic',ROOT/'diagnostic/COMPLETE.json')],identity);d=read(ROOT/'diagnostic/COMPLETE.json');render()
    if not d['training_admitted']:
        write_json(ROOT/'COMPLETION.json',{'status':'completed','outcome':'token_hypothesis_not_supported_no_training','diagnostic':d,'optimizer_updates_new':0,'goal_complete':False});print('STOP_NO_TOKEN_SIGNAL',d,flush=True);return
    run_queue(ROOT/'pilot_queue',[job(r,ROOT/'runs'/r/'COMPLETE.json') for r in ('mean32','max32')],identity)
    result=analyze();write_json(ROOT/'COMPLETION.json',result);print('TOKEN_FOCUS_COMPLETE',result,flush=True)


if __name__=='__main__': main()
