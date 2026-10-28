import argparse
from collections import defaultdict
import html
import os
from pathlib import Path
import sys
from urllib.parse import quote
from rarevlm_paths import PROJECT_ROOT,SOURCE_ROOT
from vlm_deep.common import sha_file,write_json
from .common import ROOT,PARENT,DIVERSITY,PROTOCOL,RESULT,PAGE,TOTAL_PARAMETERS,read


def prepare():
    import torch
    from vlm_reference_replay_v1.execution_v2 import verify_execution
    from .initialization import build
    if (ROOT/'PLAN.json').exists():raise FileExistsError('Already frozen')
    torch.set_num_threads(2);verify_execution();prior=read(PARENT/'PLAN.json')
    initializer=build(prior['initial_adapter'],prior['model_root'])
    data=read(PARENT/'DATA.json');cases=[]
    for typ in ('visual_time','numeric_text','spelling','wrong_target'):
        rows=[q for q in data['hard_rows'] if q['review_error_type']==typ]
        cases+=rows[:2]
    if len(cases)!=8:raise ValueError('Need two predetermined training cases per category')
    write_json(ROOT/'LEARNING_CASES.json',cases)
    paths=list((SOURCE_ROOT/'vlm_language_mlp_v1').glob('*.py'))+[PROTOCOL,PROJECT_ROOT/'tools/run_language_mlp_v1.py',ROOT/'CPU_INITIALIZER_REVIEW.json',ROOT/'LEARNING_CASES.json',PARENT/'PLAN.json',PARENT/'DATA.json',PARENT/'SCHEDULE.json',PARENT/'CPU_INPUTS.json',PARENT/'FIRST_UPDATE_REFERENCE.json',DIVERSITY/'ANALYSIS.json']
    paths += [ROOT/'initial_adapter'/n for n in ('adapter_config.json','adapter_model.safetensors')]
    paths += list((PARENT/'runs').glob('*/*/PREDICTIONS.json'))+list((DIVERSITY/'runs/unique1024').glob('*/PREDICTIONS.json'))
    plan={'status':'prepared','files':{**prior['files'],**{str(p):sha_file(p) for p in paths}},'images':prior['images'],
          'initial_lora_sha256':initializer['lora_sha256'],'original_qv_sha256':prior['initial_lora_sha256'],'trainable_parameters':TOTAL_PARAMETERS,
          'old_qv_parameters':1843200,'new_mlp_parameters':2820096,'rank_new_mlp':2,'alpha_new_mlp':4,'new_formal_updates':64,'new_generations':2661,
          'control_reused':True,'matched_control':'plain','control_location':str(PARENT/'runs/plain'),'training_seconds_cap':1800,'evaluation_seconds_cap':2400,
          'automatic_more_training':False,'automatic_formal_evaluation':False}
    write_json(ROOT/'PLAN.json',plan);render()
    RESULT.write_text('# 语言MLP适配：准备完成\n\n真实架构/PEFT混合rank注入、原144 q/v值、360因子/4663296参数及保存恢复通过；还没有真实GPU梯度或能力收益。\n\n[协议](VLM_LANGUAGE_MLP_V1_PROTOCOL.md) · [原图与MLP原理](learning/语言MLP适配_容量与迁移.html)\n')
    print('PREPARED_LANGUAGE_MLP',initializer,flush=True)


def verify():
    plan=read(ROOT/'PLAN.json')
    for path,digest in {**plan['files'],**plan['images']}.items():
        if sha_file(path)!=digest:raise ValueError('Frozen source/input changed: '+path)
    return plan


def render(result=None):
    data=read(PARENT/'DATA.json');cases=read(ROOT/'LEARNING_CASES.json');relative=lambda p:quote(os.path.relpath(p,PAGE.parent),safe='/._-');cards=[]
    for case in cases:
        rows=''
        for role,base in [('initial',PARENT),('plain',PARENT),('hard_replay',PARENT),('unique1024',DIVERSITY),('candidate',ROOT)]:
            path=base/'runs'/role/'training_context/PREDICTIONS.json'
            if path.exists():
                q=next((q for q in read(path) if q['question_id']==case['question_id']),None)
                if q:rows+='<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in (role,q['prediction'],q['vqa_score']))+'</tr>'
        cards.append(f'<article><h2>训练题 q{case["question_id"]} · {case["review_error_type"]}</h2><img loading="lazy" src="{relative(data["images"][case["image_id"]]["path"])}" alt="原图，非放大后的模型输入"><p>{html.escape(case["question"])}</p><p>原目标：{html.escape(case["target"])}</p><p>既有逐图支持记录：{html.escape(str(case["visible_evidence"]))}</p><table><tr><th>模型</th><th>保存的真实回答</th><th>VQA</th></tr>{rows}</table><p>这是训练题，用于解释可学习性，不当作独立能力提升。</p></article>')
    status='准备/运行中，尚无新能力提升。' if result is None else '完整评测结束，联合通过='+str(result['passed'])+'。'
    first=ROOT/'runs/candidate/train/FIRST_SCOPE_CHECK.json';scope=read(first) if first.exists() else None
    scope_text='GPU尚待执行。' if scope is None else str({k:scope[k] for k in ['status','initial_ce_max_difference','trainable_parameters']})+'；108个新B梯度已记录。'
    text='''Attention负责跨位置的信息混合，MLP负责逐位置的非线性变换；不能据模块名称断言某类错误必然由它负责。
MLP(h) = Wdown [SiLU(Wgate h) ⊙ (Wup h)]，h最后维度2048，中间维度11008。
LoRA: W' = W + (α/r) B A；新增MLP r=2、α=4，原q/v r=8、α=16，scale均为2。
gate/up: A[2,2048], B[11008,2]；down: A[2,11008], B[2048,2]。
36层×3矩阵×2×(2048+11008) = 2820096个新增参数；加原q/v1843200，共4663296。
新B严格0，旧q/v逐值保留；开始时新分支输出0，第一步A梯度0而B可非零，第二步后A才可能学习。
CPU真实架构与PEFT注入/保存恢复检查后，还必须核对实际GPU首16输入/CE一致与108个B梯度。
同128困难题+512回放、64步、lr2e−4、普通CE，0新增QA或重编码；复用完全匹配的已有q/v普通CE控制。
这是容量/适配范围变化，同时增加参数与计算，不是等参数位置因果消融或原创算法。
假设是更大的语言适配空间可能改善纠错；训练题拟合/梯度非零不证明独立图像泛化。
两个域的整体、rare与保持等原门槛保留；Final64，不以训练题/最佳中间点挽救失败。
下方8原图和病例在新训练前固定，来自训练集；独立1617图像迁移集另行完整评测。'''
    PAGE.write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>语言MLP适配</title><style>body{max-width:1100px;margin:28px auto;padding:0 18px;font:17px/1.7 system-ui;background:#f8fafc;color:#1e293b}article{background:white;border:1px solid #cbd5e1;padding:18px;margin:18px 0}img{width:100%;max-height:420px;object-fit:contain}pre{white-space:pre-wrap;padding:16px;background:#e2e8f0}td,th{border:1px solid #cbd5e1;padding:6px}table{border-collapse:collapse}</style><h1>从注意力q/v到语言MLP：容量是否限制纠错</h1><p>'+status+'</p><pre>'+html.escape(text)+'</pre><p>'+html.escape(scope_text)+'</p><p><a href="'+relative(PROTOCOL)+'">预登记</a> · <a href="'+relative(SOURCE_ROOT/'vlm_language_mlp_v1/initialization.py')+'">真实初始化代码</a> · <a href="'+relative(SOURCE_ROOT/'vlm_language_mlp_v1/train.py')+'">训练代码</a> · <a href="https://huggingface.co/docs/peft/v0.15.0/package_reference/lora">PEFT混合rank与零B文档</a></p>'+''.join(cards)+'</html>')


def condition():
    from .train import train
    from .evaluate import evaluate
    directory=ROOT/'runs/candidate';directory.mkdir(parents=True,exist_ok=True)
    write_json(directory/'PHASE.json',{'phase':'training'})
    model,processor,result=train();write_json(directory/'PHASE.json',{'phase':'evaluation','completed_updates':64})
    evaluate(model,processor);write_json(directory/'COMPLETE.json',{'status':'completed','train':result});write_json(directory/'PHASE.json',{'phase':'completed'})


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--worker',action='store_true');args=p.parse_args()
    if args.prepare:prepare();return
    verify()
    if args.worker:condition();return
    from research_runtime.queue import run_queue
    run_queue(ROOT/'queue',[{'key':'candidate','command':[sys.executable,'-u','-m','vlm_language_mlp_v1.study','--worker'],'completion_marker':str(ROOT/'runs/candidate/COMPLETE.json')}],{'plan_sha256':sha_file(ROOT/'PLAN.json')})
    from .analysis import analyze
    result=analyze();render(result);write_json(ROOT/'COMPLETION.json',result);print('COMPLETED_LANGUAGE_MLP',result['passed'],flush=True)


if __name__=='__main__':main()
