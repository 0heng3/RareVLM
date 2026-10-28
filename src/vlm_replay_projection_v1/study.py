import argparse
import html
import json
import os
from pathlib import Path
import sys
from urllib.parse import quote
from rarevlm_paths import PROJECT_ROOT, SOURCE_ROOT
from vlm_deep.common import sha_file, write_json
from .common import ROOT, PARENT, PROTOCOL, RESULT, PAGE, read


def prepare():
    if (ROOT/'PLAN.json').exists():raise FileExistsError('Already frozen')
    from vlm_reference_replay_v1.execution_v2 import verify_execution
    verify_execution()
    parent=read(PARENT/'PLAN.json');previous=read(PARENT/'ANALYSIS.json')
    if previous['status']!='completed' or previous['passed']:raise ValueError('Expected completed negative parent')
    paths=list((SOURCE_ROOT/'vlm_replay_projection_v1').glob('*.py'))
    paths += [PROTOCOL, PROJECT_ROOT/'tools/run_replay_projection_v1.py', PROJECT_ROOT/'tests/test_vlm_replay_projection_v1.py',
              SOURCE_ROOT/'vlm_adam_conflict_v1/projection.py', SOURCE_ROOT/'research_runtime/queue.py',
              ROOT/'CPU_CHECK.json',PARENT/'PLAN.json',PARENT/'ANALYSIS.json',PARENT/'DATA.json',PARENT/'CPU_INPUTS.json',PARENT/'SCHEDULE.json',PARENT/'FIRST_UPDATE_REFERENCE.json']
    paths += list((PARENT/'runs').glob('*/*/PREDICTIONS.json'))
    plan={'status':'prepared','initial_adapter':parent['initial_adapter'],'initial_lora_sha256':parent['initial_lora_sha256'],
          'parent_plan_sha256':sha_file(PARENT/'PLAN.json'),'files':{str(p):sha_file(p) for p in paths},
          'updates_per_arm':64,'new_formal_updates_if_admitted':128,'new_generated_qa_if_admitted':5322,
          'diagnostic_checkpoints':[16,32,48],'diagnostic_discarded_optimizer_updates':3,
          'mechanism_gate':{'minimum_conflicting_points':2,'replay_displacement_cosine_strict_gt':.01,'hard_displacement_cosine_strict_lt':0.},
          'candidate':'projected','fresh_control':'ce','reference_models':['initial','plain','hard_replay'],
          'training_budget_seconds':1800,'evaluation_budget_seconds':2400,'diagnostic_budget_seconds':180,
          'minimum_gain_pp':.5,'point_slice_guard_pp':-.5,'minimum_training_full_gain':32,
          'seeds':[42],'exploratory':True,'automatic_more_training':False,'automatic_formal_eval':False}
    write_json(ROOT/'PLAN.json',plan);render()
    RESULT.write_text('# 回放约束Adam实验\n\n已准备，训练侧机制门槛尚待检验，无新模型能力提升。\n\n[协议](VLM_REPLAY_PROJECTION_V1_PROTOCOL.md) · [原图与代码](learning/回放约束Adam_纠错与保持.html)\n')
    print('PREPARED_REPLAY_PROJECTION',{'new_updates_if_admitted':128,'new_generations_if_admitted':5322},flush=True)


def verify():
    plan=read(ROOT/'PLAN.json')
    for path,digest in plan['files'].items():
        if sha_file(path)!=digest:raise ValueError('Frozen source/input changed: '+path)
    return plan


def analyze():
    from vlm_grpo_iteration_v1.study import paired
    records={}
    for role in ('initial','plain','hard_replay','ce','projected'):
        base=ROOT if role in ('ce','projected') else PARENT
        records[role]={name:[q for q in read(base/'runs'/role/name/'PREDICTIONS.json') if q['evaluation_scored']]
                       for name in ('training_context','seen_validation','retention','transfer_dev')}
    percent=lambda rows:100*sum(q['vqa_score'] for q in rows)/len(rows)
    scores={role:{name:percent(rows) for name,rows in groups.items()} for role,groups in records.items()}
    comparisons={};gates={};slices={};changes={}
    for cohort in ('seen_validation','transfer_dev'):
        for reference in ('initial','plain','hard_replay','ce'):
            old=records[reference][cohort];new=records['projected'][cohort];key=cohort+':'+reference
            value=paired(old,new);comparisons[key]=value;gates[key+':overall']=value['delta_pp']>=.5 and value['ci95'][0]>0
            a={q['question_id']:q for q in old}
            changes[key]=[{ 'question_id':q['question_id'],'reference':a[q['question_id']]['prediction'],'candidate':q['prediction'],
                            'reference_score':a[q['question_id']]['vqa_score'],'candidate_score':q['vqa_score']}
                          for q in new if q['vqa_score']!=a[q['question_id']]['vqa_score']]
            for field,labels in [('stratum',('singleword','multiword')),('frequency_slice',sorted({q['frequency_slice'] for q in old}))]:
                for label in labels:
                    delta=paired([q for q in old if q[field]==label],[q for q in new if q[field]==label]);slices[key+':'+label]=delta
                    gates[key+':'+label]=delta['delta_pp']>=(.5 if reference=='initial' and label=='rare_1_5' else -.5)
    for ref in ('initial','plain','hard_replay','ce'):
        gates[ref+':retention']=scores['projected']['retention']-scores[ref]['retention']>=-.5
    full={r:sum(q['vqa_score']>=.999999 for q in v['training_context']) for r,v in records.items()}
    gates['training_net32']=full['projected']-full['initial']>=32
    protected={q['question_id'] for q in records['initial']['seen_validation'] if q['stratum']=='multiword' and q['vqa_score']>=.999999}
    keep=paired([q for q in records['initial']['seen_validation'] if q['question_id'] in protected],
                [q for q in records['projected']['seen_validation'] if q['question_id'] in protected])
    gates['seen_initial_full_multiword_retention']=keep['delta_pp']>=-.5
    result={'status':'completed','passed':all(gates.values()),'goal_complete':False,'scores_percent':scores,'training_full':full,
            'comparisons':comparisons,'slices':slices,'gates':gates,'previously_full_multiword':keep,
            'diagnostic':read(ROOT/'DIAGNOSTIC.json'),'all_score_changes':changes,
            'scope':'Single-seed exploratory, previously observed internal cohorts; no blind or seed-stability claim.',
            'next':'No automatic extension, formal benchmark, CV change, or checkpoint selection.'}
    write_json(ROOT/'ANALYSIS.json',result)
    table='|组|训练128满分|seen512%|SFT未用图像1617%|保持64%|\n|---|---:|---:|---:|---:|\n'
    for role,v in scores.items():table+=f'|{role}|{full[role]}|{v["seen_validation"]:.6f}|{v["transfer_dev"]:.6f}|{v["retention"]:.6f}|\n'
    RESULT.write_text('# 回放约束Adam完整结果\n\n'+table+'\n联合门槛='+str(result['passed'])+'；非盲测，不称三seed稳定或完整A-GEM复现。全部逐题升降见outputs/vlm_replay_projection_v1/ANALYSIS.json。\n\n[协议](VLM_REPLAY_PROJECTION_V1_PROTOCOL.md) · [原图、几何与实际回答](learning/回放约束Adam_纠错与保持.html)\n')
    render(result);write_json(ROOT/'COMPLETION.json',result);return result


def render(result=None):
    data=read(PARENT/'DATA.json');prior=read(PARENT.parent/'vlm_merger_transfer_v1/POSTRUN_TRANSFER_REVIEW.json')
    relative=lambda p:quote(os.path.relpath(p,PAGE.parent),safe='/._-')
    cases=[{'question_id':q['question_id'],'question':q['question'],'target':q['target'],'image':data['images'][q['image_id']]['path'],'cohort':'training_context'} for q in data['hard_rows'][:4]]
    cases += [{**q,'cohort':'transfer_dev'} for q in prior['examples'][:4]]
    cards=[]
    for case in cases:
        rows=''
        for role in ('initial','plain','hard_replay','ce','projected'):
            base=ROOT if role in ('ce','projected') else PARENT;path=base/'runs'/role/case['cohort']/'PREDICTIONS.json'
            if path.exists():
                values={q['question_id']:q for q in read(path)};q=values.get(case['question_id'])
                if q:rows+='<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in (role,q['prediction'],q['vqa_score']))+'</tr>'
        cards.append(f'<article><h2>{case["cohort"]} q{case["question_id"]}</h2><img loading="lazy" src="{relative(case["image"])}" alt="原始数据图像，非模型注意力"><p>{html.escape(case["question"])}</p><p>原目标：{html.escape(case["target"])}</p><table><tr><th>模型</th><th>保存的真实回答</th><th>VQA得分</th></tr>{rows}</table></article>')
    diagnostic=read(ROOT/'DIAGNOSTIC.json') if (ROOT/'DIAGNOSTIC.json').exists() else None
    status='准备/运行中，无新独立提分结论。' if result is None else '完整评测已结束；联合通过='+str(result['passed'])+'。'
    text='''L = (Σ8 hard CE + Σ8 replay CE)/16，逐QA对内容+真实EOS取token平均。
δAdam = AdamW依据总梯度及动量提出的真实参数位移。
寻找最接近δAdam的δ：min ½||δ−δAdam||²，约束 g_hard·δ≤0 且 g_replay·δ≤0。
CE(θ+δ)≈CE(θ)+g·δ：正投影意味着局部预测loss增加，负投影意味着减少。
已有方向满足约束就不修正；零边界使约束总有可行解，但可能牺牲纠错步长。
约束作用于Adam之后的位移，不把原始梯度投影误称实际Adam保证。
保留原Adam一二阶矩；不根据修正后位移伪造moments。断点存实际参数+原moments。
这只有当前训练批次的一阶保证；BF16有限步误差与未见图像能力必须由真实评测检验。
同一微批只反传一次，两新臂都分源累计，共同记录几何；候选额外应用修正。
参照GEM/A-GEM约束思路，属于本地Adam位移适配，非原创/完整论文复现/RL。
8图在新结果前固定：前4道困难训练题+上轮前4讲解病例，不能据此选模型或改GT。'''
    PAGE.write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>回放约束Adam</title><style>body{max-width:1100px;margin:28px auto;padding:0 18px;font:17px/1.7 system-ui;color:#1e293b;background:#f8fafc}article{padding:18px;margin:18px 0;background:white;border:1px solid #cbd5e1}img{width:100%;max-height:420px;object-fit:contain}pre{white-space:pre-wrap;background:#e2e8f0;padding:16px}table{border-collapse:collapse}td,th{border:1px solid #cbd5e1;padding:6px}</style><h1>纠错与保持：不让固定回放权重替代方向判断</h1><p>'+status+'</p><pre>'+html.escape(text)+'</pre><p><a href="https://arxiv.org/abs/1812.00420">A-GEM论文</a> · <a href="https://github.com/facebookresearch/agem">官方实现</a> · <a href="'+relative(PROTOCOL)+'">预登记</a> · <a href="'+relative(SOURCE_ROOT/'vlm_replay_projection_v1/train.py')+'">真实训练代码</a></p><h2>三个训练检查点的实际结果</h2><pre>'+html.escape(json.dumps(diagnostic,ensure_ascii=False,indent=2))+'</pre>'+''.join(cards)+'</html>')


def condition(role):
    from .train import train
    from .evaluate import evaluate
    directory=ROOT/'runs'/role;directory.mkdir(parents=True,exist_ok=True)
    write_json(directory/'PHASE.json',{'phase':'training'})
    model,processor,result=train(role)
    write_json(directory/'PHASE.json',{'phase':'evaluation','completed_updates':64})
    evaluate(model,processor,role)
    write_json(directory/'COMPLETE.json',{'status':'completed','model':result})
    write_json(directory/'PHASE.json',{'phase':'completed'})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--role',choices=['diagnostic','ce','projected']);args=parser.parse_args()
    if args.prepare:prepare();return
    verify()
    if args.role=='diagnostic':
        from .diagnose import run
        print('DIAGNOSTIC',run(),flush=True);return
    if args.role:condition(args.role);return
    from research_runtime.queue import run_queue
    identity={'plan_sha256':sha_file(ROOT/'PLAN.json')}
    run_queue(ROOT/'diagnostic_queue',[{'key':'diagnostic','command':[sys.executable,'-u','-m','vlm_replay_projection_v1.study','--role','diagnostic'],'completion_marker':str(ROOT/'DIAGNOSTIC.json')}],identity)
    diagnostic=read(ROOT/'DIAGNOSTIC.json');render()
    if not diagnostic['training_allowed']:
        result={'status':'completed','passed':False,'goal_complete':False,'outcome':'mechanism_gate_not_met_no_training', 'diagnostic':diagnostic}
        write_json(ROOT/'COMPLETION.json',result)
        RESULT.write_text('# 回放约束Adam：机制未准入\n\n三个训练检查点未满足事前冲突标准，未启动两组正式训练；没有新模型能力结论。\n\n[原图、公式与诊断](learning/回放约束Adam_纠错与保持.html)\n')
        print('STOPPED_NO_MECHANISM',flush=True);return
    jobs=[{'key':role,'command':[sys.executable,'-u','-m','vlm_replay_projection_v1.study','--role',role], 'completion_marker':str(ROOT/'runs'/role/'COMPLETE.json')} for role in ('ce','projected')]
    run_queue(ROOT/'training_queue',jobs,{**identity,'diagnostic_sha256':sha_file(ROOT/'DIAGNOSTIC.json')})
    print('COMPLETED_REPLAY_PROJECTION',analyze()['passed'],flush=True)


if __name__=='__main__':main()
