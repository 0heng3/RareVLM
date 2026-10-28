import argparse
import html
import os
from pathlib import Path
import sys
from urllib.parse import quote
from rarevlm_paths import SOURCE_ROOT
from vlm_deep.common import sha_file,write_json
from .common import ROOT,PARENT,PROTOCOL,RESULT,PAGE,ROLES,read,verify


def render(result=None):
    data=read(PARENT/'DATA.json');prior=read(PARENT.parent/'vlm_merger_transfer_v1/POSTRUN_TRANSFER_REVIEW.json')
    relative=lambda p:quote(os.path.relpath(p,PAGE.parent),safe='/._-')
    cases=[{'question_id':q['question_id'],'question':q['question'],'target':q['target'],'image':data['images'][q['image_id']]['path'],'cohort':'training_context'} for q in data['hard_rows'][:4]]
    cases += [{**q,'cohort':'transfer_dev'} for q in prior['examples'][:4]]
    cards=[]
    for case in cases:
        rows=''
        for role in ('initial','plain','hard_replay')+ROLES:
            path=(ROOT if role in ROLES else PARENT)/'runs'/role/case['cohort']/'PREDICTIONS.json'
            if path.exists():
                q=next((q for q in read(path) if q['question_id']==case['question_id']),None)
                if q:rows+='<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in (role,q['prediction'],q['vqa_score']))+'</tr>'
        cards.append(f'<article><h2>{case["cohort"]} q{case["question_id"]}</h2><img loading="lazy" src="{relative(case["image"])}" alt="原始数据图片"><p>{html.escape(case["question"])}</p><p>原目标：{html.escape(case["target"])}</p><table><tr><th>模型</th><th>实际保存回答</th><th>VQA</th></tr>{rows}</table></article>')
    admission=read(ROOT/'ADMISSION.json') if (ROOT/'ADMISSION.json').exists() else None
    if result is None:status='准备/运行中，尚无新能力提升。'
    elif result.get('outcome')=='insufficient_donors_no_training':status='固定候选池未达到数据数量准入，正式训练未启动，无新能力提升。'
    else:status='完整评测已结束，联合通过='+str(result['passed'])+'；单seed内部研究。'
    principle='''两臂每步8hard+16replay，64更新，512hard+1024replay曝光。
repeat512：原512回放各两次；unique1024：原512+新512各一次。
新增512的(freq,单/多词)与原512完全匹配；每个原slot对应同分层新slot。
回放多词占480/1024=46.875%，不是扩大多词占比；困难题仍128各四次。
L = .5·mean_hard CE + (.5+β)·mean_replay CE，β沿用训练侧校准28.97766398。
普通扩大batch若用全QA均值，会把hard/replay权重改成1/3与2/3；本实验显式按源归一，仍保持原1/2与1/2及β。
原4微批→6微批，每微批4；两组都计算24QA/步。对照重复回放的理论loss/梯度总量与旧加强CE一致。
候选独立QA1152，对照640；两组1536曝光，但像素/实际token长度不同，不称等FLOPs。
学习的假设：有限回放的梯度估计未覆盖足够情况；增加独立正确示例可能改善保持与迁移，必须看实际得分。
新题只来自原SFT train_all，排当前训练/保持/诊断及dev/final同图像与像素。
>=8/10官方共识+原模型自然EOS满分才作新回放，不把它称专家逐图重标注。
多词固定候选池一次评分，不够就停，不按dev补题；新单词复用既有训练池满分预测。
图片在新训练前固定，来自原4训练例+前轮4讲解例；不能按这些例子挑checkpoint或改评分。'''
    PAGE.write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>回放多样性</title><style>body{max-width:1100px;margin:28px auto;padding:0 18px;font:17px/1.7 system-ui;color:#1e293b;background:#f8fafc}article{background:white;border:1px solid #cbd5e1;padding:18px;margin:18px 0}img{width:100%;max-height:420px;object-fit:contain}pre{white-space:pre-wrap;background:#e2e8f0;padding:16px}td,th{border:1px solid #cbd5e1;padding:6px}table{border-collapse:collapse}</style><h1>同样回放1024次：重复旧题还是覆盖新题</h1><p>'+status+'</p><pre>'+html.escape(principle)+'</pre><p><a href="'+relative(PROTOCOL)+'">预登记协议</a> · <a href="'+relative(RESULT)+'">完整结果</a> · <a href="'+relative(SOURCE_ROOT/'vlm_replay_diversity_v1/train.py')+'">训练源码</a></p><h2>训练数据准入</h2><pre>'+html.escape(str(admission))+'</pre>'+''.join(cards)+'</html>')


def condition(role):
    from .train import train
    from .evaluate import evaluate
    directory=ROOT/'runs'/role;directory.mkdir(parents=True,exist_ok=True)
    write_json(directory/'PHASE.json',{'phase':'training'})
    model,processor,result=train(role)
    write_json(directory/'PHASE.json',{'phase':'evaluation','completed_updates':64})
    evaluate(model,processor,role)
    write_json(directory/'COMPLETE.json',{'status':'completed','train':result})
    write_json(directory/'PHASE.json',{'phase':'completed'})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--worker',choices=['mining_0','mining_1']+list(ROLES));args=parser.parse_args()
    if args.prepare:
        from .data import prepare
        prepare();render();RESULT.write_text('# 回放多样性：已准备\n\n等待固定新多词池的原SFT评分，达分层数量后接续两组64步。尚无新能力提升。\n\n[协议](VLM_REPLAY_DIVERSITY_V1_PROTOCOL.md) · [8原图与公式](learning/回放多样性_重复512与覆盖1024.html)\n');return
    verify()
    if args.worker:
        if args.worker.startswith('mining_'):
            from .mining import worker
            worker(int(args.worker[-1]))
        else:condition(args.worker)
        return
    from research_runtime.queue import run_queue
    identity={'plan_sha256':sha_file(ROOT/'PLAN.json')}
    jobs=[{'key':f'mining_{i}','command':[sys.executable,'-u','-m','vlm_replay_diversity_v1.study','--worker',f'mining_{i}'],'completion_marker':str(ROOT/f'mining_{i}/COMPLETE.json')} for i in (0,1)]
    write_json(ROOT/'WORKFLOW_STATUS.json',{'phase':'mining_training_donors','status':'running'})
    run_queue(ROOT/'mining_queue',jobs,identity)
    from .data import admit
    write_json(ROOT/'WORKFLOW_STATUS.json',{'phase':'training_data_admission','status':'running'})
    admission=admit()
    if not admission['training_allowed']:
        result={'status':'completed','passed':False,'goal_complete':False,'outcome':'insufficient_donors_no_training','admission':admission}
        RESULT.write_text('# 回放多样性：未达到数据准入\n\n固定池正确多词样本不足，未启动正式训练，不追补或降低门槛。详情outputs/vlm_replay_diversity_v1/ADMISSION.json。\n')
        render(result);write_json(ROOT/'COMPLETION.json',result);write_json(ROOT/'WORKFLOW_STATUS.json',{'phase':'completed_no_training','status':'completed'});return
    render();write_json(ROOT/'WORKFLOW_STATUS.json',{'phase':'matched_training_and_evaluation','status':'running'})
    jobs=[{'key':role,'command':[sys.executable,'-u','-m','vlm_replay_diversity_v1.study','--worker',role],'completion_marker':str(ROOT/'runs'/role/'COMPLETE.json')} for role in ROLES]
    run_queue(ROOT/'training_queue',jobs,{**identity,'admission_sha256':sha_file(ROOT/'ADMISSION.json')})
    from .analysis import analyze
    result=analyze();render(result);write_json(ROOT/'COMPLETION.json',result);write_json(ROOT/'WORKFLOW_STATUS.json',{'phase':'completed','status':'completed','passed':result['passed']})
    print('COMPLETED_REPLAY_DIVERSITY',result['passed'],flush=True)


if __name__=='__main__':main()
