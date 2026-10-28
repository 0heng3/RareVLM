import argparse
from collections import Counter
import html
import json
import os
from pathlib import Path
import sys
import time
from urllib.parse import quote
from rarevlm_paths import PROJECT_ROOT, SOURCE_ROOT
from vlm_deep.common import NORMALIZER, content_sha, encode, read_manifest, sha_file, write_json
from vlm_sampled_signal_v1.trajectory import group_signal
from .common import ROOT, OLD, SAMPLED, actor, chosen_sample, read, rollout_batch

PROTOCOL=PROJECT_ROOT/'docs/VLM_GRPO_ITERATION_V1_PROTOCOL.md'
RESULT=PROJECT_ROOT/'docs/VLM_GRPO_ITERATION_V1_RESULTS.md'
PAGE=PROJECT_ROOT/'docs/learning/单轮GRPO与SFT_实际轨迹和独立验证.html'
EXCLUSIONS=(
    'vlm_region_ocr_sft_v1/data/CONTROL_TRAIN.json',
    'vlm_scope_balanced_sft_v2/data/empirical_mix/MANIFEST.json',
    'vlm_scope_balanced_sft_v2/data/EVAL.json',
    'vlm_scope_transfer_diagnosis_v1/DATA.json',
    'vlm_scope_preference_mining_v1/DATA.json',
    'vlm_region_preference_probe_v1/data/MANIFEST.json',
    'vlm_readout_transfer_v1/DATA.json',
    'vlm_pixel_transfer_diagnosis_v1/DATA.json',
    'vlm_error_supervision_mining_v1/DATA.json',
    'vlm_native_question_binding_v1/DATA.json',
    'vlm_error_supervision_sft_v1/data/MANIFEST.json')


def render(result=None):
    data=read(ROOT/'DATA.json');lookup={q['question_id']:q for q in data['rows']}
    relative=lambda p:quote(os.path.relpath(p,PAGE.parent),safe='/._-')
    cards=[]
    for g in read(ROOT/'REVIEWED_ROLLOUTS.json'):
        q=lookup[g['question_id']]
        rows=''.join(f'<tr><td>{html.escape(s["prediction"])}</td><td>{s["reward"]:.1f}</td><td>{a:.3f}</td><td>{html.escape(str(s["completion_ids"]))}</td></tr>' for s,a in zip(g['samples'],g['advantages']))
        cards.append(f'<article><h3>q{q["question_id"]}：{html.escape(q["target"])}</h3><img loading="lazy" src="{relative(data["images"][q["image_id"]]["path"])}" alt="原始训练图"><p>{html.escape(q["question"])}</p><p>{html.escape(g["visual_note"])}</p><table><tr><th>实际采样</th><th>原VQA分</th><th>优势A</th><th>原始动作ID</th></tr>{rows}</table></article>')
    status='已登记并准备；尚无本轮训练/泛化结果。' if result is None else html.escape(json.dumps(result['scores_percent'],ensure_ascii=False))+f'；联合通过={result["passed"]}'
    PAGE.write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>单轮GRPO对照</title><style>body{max-width:1120px;margin:30px auto;padding:0 18px;font:17px/1.7 system-ui;background:#f8fafc;color:#1e293b}article{background:white;margin:18px 0;padding:16px;border:1px solid #cbd5e1}img{width:100%;max-height:450px;object-fit:contain}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #cbd5e1;padding:6px;overflow-wrap:anywhere}pre{background:#e2e8f0;padding:16px;white-space:pre-wrap}.note{background:#fff7ed;padding:16px}</style><h1>从固定负例DPO到实际采样GRPO：单轮强SFT对照</h1>'
        '<p class="note">'+status+'</p><p>原32题训练纠错没有形成可靠泛化优势。新问题是：在相同真实SFT监督上，实际采样的高低奖励能否额外改变正确决策？</p>'
        '<pre>两组相同：32错题×8 + 256保持回放 → 32次AdamW更新\n控制：L_CE\n候选：L_CE + L_GRPO\nA=(原VQA奖励−组均值)/(组内样本std+1e−4)\nρ=exp(当前动作logp−实际生成时动作logp)\nL_GRPO=−mean_8 mean_token min(ρA, clip(ρ,.8,1.2)A)</pre>'
        '<p>例如A为正时，增加该回答的概率；概率比超过1.2后，裁剪阻止该方向继续获益。A为负时抑制该回答；下降到0.8以下后也限制继续抑制。前向loss值接近0不代表梯度为0。EOS包含在实际动作中，句后padding和模板换行不参与。</p>'
        '<p>仅一轮初始策略rollout，每条只训练一次；后续小批次通过固定behavior概率比修正策略漂移，无critic、KL系数0。新增136条策略前后向成本另计，不称等计算优势。奖励为答案完整性/共识代理，部分品牌简称有信息，不能全部称幻觉或语义错误。</p>'
        '<p>新512验证图固定排除旧研究图；两组均须统一greedy评测。候选对初始和新CE各至少+.5pp且区间为正，并保持单词、多词和64题；训练纠错、采样best-of8和A值都不是能力提升。</p>'
        f'<p><a href="{relative(PROTOCOL)}">完整预登记</a> · <a href="{relative(RESULT)}">结果</a> · <a href="{relative(SOURCE_ROOT/"vlm_grpo_iteration_v1/loss.py")}">真实策略损失源码</a> · <a href="{relative(SOURCE_ROOT/"vlm_grpo_iteration_v1/train.py")}">真实训练循环</a></p><h2>17组原图、实际动作和优势</h2>'+''.join(cards)+'</html>')


def prepare():
    from vlm_probe.evaluate import inference_row, processor_for
    from .common import chosen_sample
    import torch
    torch.set_num_threads(2)
    if (ROOT/'PLAN.json').exists():raise FileExistsError('Already frozen; inspect current workflow')
    sampled=read(SAMPLED/'PLAN.json');source_path=ROOT.parent/'vlm_deep_v2/prepared/manifest.json'
    original=read_manifest(source_path);data=read(OLD/'data/MANIFEST.json')
    excluded={q['image_id'] for split in ('dev','final') for q in original['splits'][split]}
    for relative in EXCLUSIONS:
        d=read(ROOT.parent/relative);excluded.update(d.get('images',{}))
        for key in ('hard_eval','retention_eval','validation_eval','rows'):
            excluded.update(q['image_id'] for q in d.get(key,[]))
    pixel=lambda im:im.get('pixel_sha256',im['sha256'])
    excluded_pixels={pixel(original['images'][iid]) for iid in excluded if iid in original['images']}
    frequency=Counter(NORMALIZER(q['target']) for q in original['splits']['train_all'])
    eligible=[q for q in original['splits']['train_all'] if q['image_id'] not in excluded and pixel(original['images'][q['image_id']]) not in excluded_pixels
        and Counter(q['answers'])[q['target']]>=8 and 1<=frequency[NORMALIZER(q['target'])]<=5 and 1<=len(q['target'].split())<=6]
    eligible.sort(key=lambda q:content_sha(['vlm_grpo_iteration_v1:validation',q['image_id'],q['question_id']]))
    selected=[];seen=set()
    for name,multi in [('multiword',True),('singleword',False)]:
        bucket=[]
        for q in eligible:
            px=pixel(original['images'][q['image_id']])
            if (len(q['target'].split())>1)!=multi or px in seen:continue
            bucket.append({**q,'stratum':name,'answer_frequency':frequency[NORMALIZER(q['target'])]});seen.add(px)
            if len(bucket)==256:break
        if len(bucket)!=256:raise ValueError('Insufficient new independent validation; no fallback')
        selected.extend(bucket)
    selected.sort(key=lambda q:content_sha(['vlm_grpo_iteration_v1:execution',q['question_id']]))
    processor=processor_for(Path(sampled['model_root']),read(sampled['reference_config']),1024)
    for index,q in enumerate(selected):
        im=original['images'][q['image_id']];path=(source_path.parent/im['path']).resolve()
        item=encode(processor,inference_row(q),path,False)
        q.update(expected_prefix_tokens=int(item['input_ids'].shape[1]),expected_grid=item['image_grid_thw'][0].tolist(),inference_input_sha256=content_sha(item['input_ids'].tolist()))
        data['images'][q['image_id']]={**im,'path':str(path)}
        if (index+1)%64==0:write_json(ROOT/'CPU_PROGRESS.json',{'status':'encoding_new_validation','completed':index+1,'expected':512})
    data['validation_eval']=selected
    for q in data['retention_eval']:
        item=encode(processor,inference_row(q),Path(data['images'][q['image_id']]['path']),False)
        q.update(expected_prefix_tokens=int(item['input_ids'].shape[1]),expected_grid=item['image_grid_thw'][0].tolist())
        if q['expected_prefix_tokens']!=q['expected_input_token_count']:
            raise ValueError('Reused retention frame differs from its original input')
    cpu={}
    for q in data['rows']:
        item=chosen_sample(processor,q,data['images'])
        cpu[q['question_id']]={key:content_sha(item[key].tolist()) for key in ('input_ids','labels')}
    notes={
        '11950':'袋上Kingsford，kingford缺s。','14627':'金色瓶黑字NAIL COLOUR，其他采样选择了其他字段或误识别。',
        '33655':'币面环形INDIAN TREATY，旁边说明Peaceful Co-existence不属于目标。','6306':'第一个盒顶OCCY CRYSTINE，Haywood属于第二盒。',
        '2295':'书脊STUDIES FROM LIFE；from life不完整，stories误读。','28846':'船名MILLENNIUM DIAMOND；citycruises为另一个可见品牌，millenium为错拼且原得.3。',
        '2994':'Cardinal Number，单复数不同。','20675':'胸前THE POINTE，最后e可见。','23803':'右侧PALY支持P的展开；pal[y]括号转写不确定，本轮优势0。',
        '25032':'电话最后三位555；9555多一位。','13863':'完整产品名HTC One；HTC为部分品牌信息，原VQA为0，不称语义全错。',
        '9964':'第二行红字FGHIJ，缺J/多I/换字等与完整印刷行不同。','18071':'Xbox360型号文字/机型可见；Xbox为部分信息，保留原.3。',
        '29702':'袋上FOOD MARKET；food是部分文字，保留原完整名称奖励边界。','24291':'左侧衣服HOOTERS，其余采样为错词/其他实体/无关文本。',
        '27824':'右箭头VICTORIA CHELSEA；Vauxhall指向上方。','22712':'广告PEORIA DIAMOND CLUB；其余是完整名称的片段。'}
    groups=[]
    for g in read(SAMPLED/'GROUPS.json'):
        if not g['signal']['clear_reward_pair']:continue
        supported=[s['prediction']!='pal[y]' for s in g['samples']]
        signal=group_signal([s['reward'] for s in g['samples']],supported)
        if not signal['clear_reward_pair']:raise ValueError('Visible review removed a required clear pair')
        groups.append({**g,'reviewed_samples':supported,'advantages':[v if v is not None else 0. for v in signal['advantages']],
            'visual_note':notes[g['question_id']],'reviewer':'assistant inspection of original images and all distinct sampled strings; not user/expert relabeling'})
    if len(groups)!=17:raise ValueError('Expected17 verified groups')
    write_json(ROOT/'DATA.json',data);write_json(ROOT/'CPU_INPUTS.json',cpu)
    write_json(ROOT/'REVIEWED_ROLLOUTS.json',groups)
    write_json(ROOT/'SCHEDULE.json',read(OLD/'data/SCHEDULE.json'))
    for name in ('hard','retention'):
        write_json(ROOT/'initial'/name/'PREDICTIONS.json',read(OLD/'baseline'/name/'PREDICTIONS.json'))
    write_json(ROOT/'BASELINE_REUSE.json',{'cohorts':['hard32','retention64'],'source':str(OLD/'baseline'),
        'same_model_generation_inputs_order':True,'fresh_baseline_generation_qa':512,'scope':'Historical paired baseline reuse, not independent repeats.'})
    paths=[PROTOCOL,Path(__file__),PROJECT_ROOT/'tools/run_grpo_iteration_v1.py',PROJECT_ROOT/'tests/test_vlm_grpo_iteration_v1.py',
        ROOT/'DATA.json',ROOT/'CPU_INPUTS.json',ROOT/'REVIEWED_ROLLOUTS.json',ROOT/'SCHEDULE.json',ROOT/'BASELINE_REUSE.json',
        SAMPLED/'PLAN.json',SAMPLED/'GROUPS.json',SAMPLED/'COMPLETION.json',source_path]
    paths+=list((SOURCE_ROOT/'vlm_grpo_iteration_v1').glob('*.py'))+[ROOT.parent/f for f in EXCLUSIONS]
    paths+=[ROOT/'initial'/name/'PREDICTIONS.json' for name in ('hard','retention')]
    paths+=[SOURCE_ROOT/f for f in ('vlm_sampled_signal_v1/trajectory.py','vlm_fp32_readout_v1/model.py','vlm_fp32_adapter/model.py',
        'vlm_runtime_metadata_v1/model.py','vlm_deep/common.py','vlm_deep/evaluate.py','vlm_sft/common.py','vlm_probe/evaluate.py',
        'vlm_human_sft/evaluate.py','vlm_object_train/train.py','vlm_attention_scope_repair_v1/generation.py','research_runtime/queue.py')]
    upstream=ROOT.parent/'vlm_frontier_bottleneck_review_20261019/upstream/huggingface__trl'
    paths+=[upstream/'COMMIT.json',upstream/'source/trl/trainer/grpo_trainer.py',Path(sampled['reference_config'])]
    paths+=[Path(sampled['adapter'])/f for f in ('adapter_config.json','adapter_model.safetensors')]
    plan={k:sampled[k] for k in ('adapter','lora_sha256','model_root','reference_config','packages','base_weight_files')}
    plan.update(status='prepared',updates_per_role=32,roles=['ce','ce_grpo'],verified_rollout_groups=17,
        additional_policy_sequences=136,new_validation=512,independent_pixels=512,excluded_images=len(excluded),
        files={str(p):sha_file(p) for p in paths},images={im['path']:im['sha256'] for im in data['images'].values()},
        automatic_main_evaluation=False,scope='Single-rollout exploration on initial-SFT training whitelist; not blind benchmark.')
    write_json(ROOT/'PLAN.json',plan)
    RESULT.write_text('# 单轮GRPO对照：准备完成\n\n32步CE与CE+GRPO同起点对照，17组136条原始动作额外策略梯度；新512唯一图/QA验证，64保持。首先测实际行为/训练数值路径，过固定准入后自动两臂训练。尚无更新或能力结论。\n\n[固定协议](VLM_GRPO_ITERATION_V1_PROTOCOL.md) · [原图/真实奖励/优势学习页](learning/单轮GRPO与SFT_实际轨迹和独立验证.html)\n')
    render();print('PREPARED_GRPO_ITERATION',{'new_validation':512,'chosen_cpu_inputs':len(cpu),'reviewed_groups':17},flush=True)


def verify():
    from vlm_object_train.train import package_lock
    plan=read(ROOT/'PLAN.json')
    for path,digest in {**plan['files'],**plan['images']}.items():
        if sha_file(path)!=digest:raise ValueError('Frozen new study input changed: '+path)
    if package_lock()!=plan['packages']:raise ValueError('Package versions changed')
    for path,value in plan['base_weight_files'].items():
        st=Path(path).stat()
        if value!={'bytes':st.st_size,'mtime_ns':st.st_mtime_ns}:raise ValueError('Base storage changed')
    return plan


def reference():
    import torch
    from vlm_sft.common import to_device
    from .loss import token_logps,behavior_logps,clipped_group_loss
    verify();model,processor,_=actor(training=True)
    data=read(ROOT/'DATA.json');lookup={q['question_id']:q for q in data['rows']}
    groups=read(ROOT/'REVIEWED_ROLLOUTS.json');deltas=[];results=[];gradient_norm=0.;started=time.monotonic()
    for index,g in enumerate(groups):
        batch=to_device(rollout_batch(processor,g,lookup[g['question_id']],data['images']),next(model.parameters()).device)
        current,valid=token_logps(model,batch);old=behavior_logps(g['samples'],current,valid)
        values=(current-old)[valid].detach().cpu();deltas.extend(values.tolist())
        if index==0:
            loss,_=clipped_group_loss(current,old,valid,torch.tensor(g['advantages'],device=current.device),.2)
            loss.backward()
            gradient_norm=float(torch.sqrt(sum(p.grad.detach().square().sum() for p in model.parameters() if p.grad is not None)))
            model.zero_grad(set_to_none=True);del loss
        results.append({'question_id':g['question_id'],'action_count':len(values),'max_abs_log_ratio':float(values.abs().max())})
        write_json(ROOT/'REFERENCE_PROGRESS.json',{'status':'checking_behavior_paths','completed_groups':index+1,'expected_groups':17,'seconds':time.monotonic()-started})
        del batch,current,valid,old,values
    x=torch.tensor(deltas);ratio=x.exp();inside=float(((ratio>=.8)&(ratio<=1.2)).double().mean());maximum=float(x.abs().max())
    passed=bool(torch.isfinite(x).all()) and inside>=.99 and maximum<=1 and gradient_norm>0
    result={'status':'completed','passed':passed,'groups':17,'sequences':136,'actions':len(deltas),
        'fraction_initial_ratios_inside_clip':inside,'required_fraction':.99,'max_abs_log_ratio':maximum,
        'first_group_policy_only_gradient_norm':gradient_norm,'training_updates':0,'rows':results,'active_seconds':time.monotonic()-started}
    write_json(ROOT/'REFERENCE_COMPLETE.json',result);print(result,flush=True)


def evaluate(model,processor,directory,names):
    import torch
    from vlm_probe.evaluate import evaluate_batch
    from vlm_deep.evaluate import score_prediction
    from vlm_error_supervision_sft_v1.common import summarize
    from vlm_attention_scope_repair_v1.generation import raw_greedy_config
    data=read(ROOT/'DATA.json');model.eval();model.config.use_cache=True
    model.generation_config=raw_greedy_config(model.generation_config,32)
    started=time.monotonic();summary={}
    with torch.inference_mode():
        for name in names:
            rows=data[name+'_eval'];path=directory/name/'PREDICTIONS.json';predictions=read(path) if path.exists() else []
            if len(predictions)%4 or [q['question_id'] for q in predictions]!=[q['question_id'] for q in rows[:len(predictions)]]:raise ValueError('Evaluation grouping changed')
            for start in range(len(predictions),len(rows),4):
                chunk=rows[start:start+4]
                answers,_=evaluate_batch(model,processor,chunk,data['images'],ROOT,next(model.parameters()).device,1024,processor.tokenizer.convert_tokens_to_ids('<|image_pad|>'),32)
                for q,a in zip(chunk,answers):
                    if a['input_tokens']!=q['expected_prefix_tokens'] or a['image_grid_thw']!=q['expected_grid']:raise ValueError('Evaluation frame changed')
                    predictions.append({'question_id':q['question_id'],'image_id':q['image_id'],'target':q['target'],
                        'stratum':q['stratum'],**a,'vqa_score':score_prediction(a['prediction'],q,'textvqa')})
                write_json(path,predictions);write_json(directory/'EVAL_PROGRESS.json',{'status':'evaluating','cohort':name,'completed_qa':len(predictions),'expected_qa':len(rows),'seconds':time.monotonic()-started})
                if time.monotonic()-started>900:raise TimeoutError('Fixed evaluation budget exceeded')
            summary[name]=summarize(predictions)
    return summary


def condition(role):
    verify()
    if role=='initial':
        model,processor,_=actor();evaluate(model,processor,ROOT/'initial',['validation'])
        write_json(ROOT/'INITIAL_COMPLETE.json',{'status':'completed','new_generation_qa':512,'reused_qa':96})
    else:
        from .train import train
        model,processor,result=train(role)
        metrics=evaluate(model,processor,ROOT/'runs'/role,['hard','validation','retention'])
        write_json(ROOT/'runs'/role/'COMPLETE.json',{'status':'completed','train':result,'metrics':metrics})


def paired(before,after):
    import numpy as np
    if [(q['question_id'],q['image_id']) for q in before]!=[(q['question_id'],q['image_id']) for q in after]:raise ValueError('Paired evaluation mismatch')
    values={}
    for a,b in zip(before,after):
        v=values.setdefault(a['image_id'],[0.,0]);v[0]+=b['vqa_score']-a['vqa_score'];v[1]+=1
    v=np.array(list(values.values()),dtype=np.float64);rng=np.random.default_rng(20261021)
    idx=rng.integers(0,len(v),size=(6000,len(v)));samples=100*v[idx,0].sum(1)/v[idx,1].sum(1)
    return {'delta_pp':float(100*v[:,0].sum()/v[:,1].sum()),'ci95':np.quantile(samples,[.025,.975]).tolist(),'image_clusters':len(v),'scope':'Fixed single-checkpoint conditional image-cluster interval.'}


def analyze():
    from vlm_error_supervision_sft_v1.common import summarize
    records={r:{n:read((ROOT/'initial' if r=='initial' else ROOT/'runs'/r)/n/'PREDICTIONS.json') for n in ('hard','validation','retention')} for r in ('initial','ce','ce_grpo')}
    scores={r:{n:summarize(qs)['vqa_percent'] for n,qs in v.items()} for r,v in records.items()}
    comparisons={r:paired(records[r]['validation'],records['ce_grpo']['validation']) for r in ('initial','ce')}
    gates={r+'_overall':d['delta_pp']>=.5 and d['ci95'][0]>0 for r,d in comparisons.items()}
    slices={}
    for role in ('initial','ce'):
        for name in ('singleword','multiword'):
            a=[q for q in records[role]['validation'] if q['stratum']==name];b=[q for q in records['ce_grpo']['validation'] if q['stratum']==name]
            slices[role+':'+name]=paired(a,b);gates[role+':'+name]=slices[role+':'+name]['delta_pp']>=-.5
        gates[role+':retention']=scores['ce_grpo']['retention']-scores[role]['retention']>=-.5
    full={r:sum(q['vqa_score']>=.999999 for q in v['hard']) for r,v in records.items()}
    gates['training_net8']=full['ce_grpo']-full['initial']>=8
    changes=[]
    for a,b in zip(records['initial']['validation'],records['ce_grpo']['validation']):
        if a['vqa_score']!=b['vqa_score']:changes.append({'question_id':a['question_id'],'target':a['target'],'before':a['prediction'],'after':b['prediction'],'score_delta':b['vqa_score']-a['vqa_score']})
    result={'status':'completed','passed':all(gates.values()),'scores_percent':scores,'comparisons':comparisons,'slices':slices,'gates':gates,'training_full':full,'validation_changes':changes,
        'next':'Preregister independent confirmation before more training/main benchmark.' if all(gates.values()) else 'Stop this configuration; no extra steps/seeds/main evaluation or CV claim.'}
    write_json(ROOT/'ANALYSIS.json',result)
    table='|模型|训练32 VQA|新512 VQA|保持64 VQA|\n|---|---:|---:|---:|\n'+''.join(f'|{r}|{s["hard"]:.6f}|{s["validation"]:.6f}|{s["retention"]:.6f}|\n' for r,s in scores.items())
    RESULT.write_text('# 单轮GRPO对照完成\n\n'+table+'\n候选对初始/新CE：'+json.dumps(comparisons,ensure_ascii=False)+'\n\n联合通过：'+str(result['passed'])+'。'+result['next']+'\n\n本轮每臂32步，单seed42；初始新512实际生成、96历史预测同条件复用，两候选各608实际生成。候选额外136策略前后向，不能声称等计算。训练拟合/奖励不代替泛化，公开训练白名单及历史曝光披露保持。\n\n[固定协议](VLM_GRPO_ITERATION_V1_PROTOCOL.md) · [原图/实际动作/优势学习页](learning/单轮GRPO与SFT_实际轨迹和独立验证.html)\n')
    render(result);write_json(ROOT/'COMPLETION.json',result);return result


def controller():
    from research_runtime.queue import run_queue
    def job(role):return {'key':role,'command':[sys.executable,'-u','-m','vlm_grpo_iteration_v1.study','--role',role],
        'completion_marker':str(ROOT/('REFERENCE_COMPLETE.json' if role=='reference' else 'INITIAL_COMPLETE.json' if role=='initial' else 'runs/'+role+'/COMPLETE.json'))}
    identity={'plan_sha256':sha_file(ROOT/'PLAN.json')}
    run_queue(ROOT/'startup_queue',[job('reference'),job('initial')],identity)
    if not read(ROOT/'REFERENCE_COMPLETE.json')['passed']:
        RESULT.write_text('# GRPO训练未准入\n\n初始behavior/训练路径概率比未过预登记数值门槛，0优化器更新。原始采样信号通过不等于训练数值路径可直接混用。详见输出REFERENCE_COMPLETE.json；不替换behavior概率或放宽门槛启动。\n')
        write_json(ROOT/'COMPLETION.json',{'status':'completed','passed':False,'training_updates':0,'reason':'behavior_training_numerical_gate_failed'})
        return
    run_queue(ROOT/'training_queue',[job('ce'),job('ce_grpo')],identity)
    print(analyze(),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--role',choices=['reference','initial','ce','ce_grpo']);a=p.parse_args()
    if a.prepare:prepare()
    elif a.role=='reference':reference()
    elif a.role:condition(a.role)
    else:controller()


if __name__=='__main__':main()
