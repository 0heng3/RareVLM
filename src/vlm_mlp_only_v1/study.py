import argparse
import html
import os
from pathlib import Path
import sys
from urllib.parse import quote
from rarevlm_paths import PROJECT_ROOT,SOURCE_ROOT
from vlm_deep.common import sha_file,write_json
from .common import ROOT,PARENT,DIVERSITY,JOINT,PROTOCOL,RESULT,PAGE,TOTAL_PARAMETERS,read


def prepare():
    import torch
    from safetensors.torch import load_file
    if (ROOT/'PLAN.json').exists():raise FileExistsError('Already frozen')
    parent=read(JOINT/'PLAN.json');result=read(JOINT/'COMPLETION.json')
    if result['status']!='completed' or result['passed']:raise ValueError('Expected completed negative joint experiment')
    for path,digest in parent['files'].items():
        if sha_file(path)!=digest:raise ValueError('Frozen parent source/input changed: '+path)
    initializer=JOINT/'initial_adapter';state=load_file(str(initializer/'adapter_model.safetensors'))
    original=load_file(str(Path(read(PARENT/'PLAN.json')['initial_adapter'])/'adapter_model.safetensors'))
    qv={k:v for k,v in state.items() if '.mlp.' not in k};mlp={k:v for k,v in state.items() if '.mlp.' in k}
    if set(qv)!=set(original) or any(not torch.equal(qv[k],v) for k,v in original.items()):raise ValueError('Original q/v values changed')
    if len(qv)!=144 or len(mlp)!=216 or sum(v.numel() for v in mlp.values())!=TOTAL_PARAMETERS:raise ValueError('Wrong scope inventory')
    if sum('.lora_B.' in k for k in mlp)!=108 or any(bool(v.ne(0).any()) for k,v in mlp.items() if '.lora_B.' in k):raise ValueError('New MLP must start from zero-B initializer')
    receipt={'status':'passed','initializer_reused':str(initializer),'original_qv_values_exact':True,'frozen_qv_factors':144,'frozen_qv_parameters':1843200,
             'trainable_mlp_factors':216,'trainable_mlp_parameters':TOTAL_PARAMETERS,'stored_adapter_parameters':4663296,
             'new_B_zero':True,'new_model_or_gpu_calls':0,'note':'Static initializer scope check; actual requires_grad/optimizer and gradients checked by the worker.'}
    write_json(ROOT/'CPU_SCOPE_REVIEW.json',receipt)
    paths=list((SOURCE_ROOT/'vlm_mlp_only_v1').glob('*.py'))+[PROTOCOL,PROJECT_ROOT/'tools/run_mlp_only_v1.py',ROOT/'CPU_SCOPE_REVIEW.json',JOINT/'PLAN.json',JOINT/'COMPLETION.json',JOINT/'LEARNING_CASES.json']
    paths+=list((JOINT/'runs/candidate').glob('*/PREDICTIONS.json'))
    paths+=[initializer/n for n in ('adapter_config.json','adapter_model.safetensors')]
    plan={'status':'prepared','files':{**parent['files'],**{str(p):sha_file(p) for p in paths}},'images':parent['images'],
          'initial_lora_sha256':parent['initial_lora_sha256'],'initial_adapter':str(initializer),'original_qv_sha256':parent['original_qv_sha256'],
          'trainable_parameters':TOTAL_PARAMETERS,'stored_adapter_parameters':4663296,'optimized_factors':216,'frozen_factors':144,
          'new_formal_updates':64,'new_generations':2661,'matched_control':str(JOINT/'runs/candidate'),'new_initializer_builds':0,'new_input_encodes':0,
          'automatic_more_training':False,'automatic_formal_evaluation':False}
    write_json(ROOT/'PLAN.json',plan);render()
    RESULT.write_text('# 仅训练MLP：准备完成\n\n原始q/v值与MLP零B初始化复用检查通过；仅216因子/2820096参数将更新，结构/适配存储4663296不变。等待共享GPU准入及实际首批检查，无新能力提升。\n\n[协议](VLM_MLP_ONLY_V1_PROTOCOL.md) · [原图与冻结原理](learning/冻结注意力适配_只训练MLP.html)\n')
    print('PREPARED_MLP_ONLY',receipt,flush=True)


def verify():
    plan=read(ROOT/'PLAN.json')
    for path,digest in {**plan['files'],**plan['images']}.items():
        if sha_file(path)!=digest:raise ValueError('Frozen source/input changed: '+path)


def render(result=None):
    data=read(PARENT/'DATA.json');relative=lambda p:quote(os.path.relpath(p,PAGE.parent),safe='/._-');cards=[]
    for case in read(JOINT/'LEARNING_CASES.json'):
        rows=''
        for label,base,role in [('initial',PARENT,'initial'),('qv_CE',PARENT,'plain'),('joint_qv_mlp',JOINT,'candidate'),('mlp_only',ROOT,'candidate')]:
            path=base/'runs'/role/'training_context/PREDICTIONS.json'
            if path.exists():
                q=next((q for q in read(path) if q['question_id']==case['question_id']),None)
                if q:rows+='<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in (label,q['prediction'],q['vqa_score']))+'</tr>'
        cards.append(f'<article><h2>训练题 q{case["question_id"]} · {case["review_error_type"]}</h2><img loading="lazy" src="{relative(data["images"][case["image_id"]]["path"])}" alt="原始数据图像"><p>{html.escape(case["question"])}</p><p>原目标：{html.escape(case["target"])}</p><table><tr><th>模型</th><th>真实回答</th><th>VQA</th></tr>{rows}</table><p>训练题用于解释拟合，独立图像指标另列。</p></article>')
    status='准备/排队/运行中，尚无新提分。' if result is None else '完整评测已结束；联合通过='+str(result['passed'])+'。'
    text='''上一联合训练同128困难题满分88，对q/v控制迁移−.754484pp；增加容量改善拟合却没有保持迁移。
本轮只改更新范围：同一原始SFT+零B的MLP起点，已有q/v144因子固定，新MLP216因子学习。
图结构/存储仍为4663296个LoRA参数，optimizer只接收2820096个MLP参数。
冻结用requires_grad=False及从optimizer参数列表排除；不会断开MLP到前层隐藏状态的梯度。
Q = H Wq，V = H Wv；即使Wq/Wv固定，MLP改变H仍可能改变后续注意力与答案，因此冻结不保证不遗忘。
不能把Wq/Wv固定描述为注意力输出固定，也不能冻结整个decoder或用no_grad切断MLP训练。
L=(Σ8hard CE+Σ8replay CE)/16，128困难各4次+512回放各1次，64步/lr2e−4/同原顺序。
从原始共同初始化fresh Adam训练，不从上轮负Final继续，也不据dev调冻结层或学习率。
同输入首16 CE、108新B梯度及首A0必须通过；每步q/v梯度范数应0，训练后q/v哈希必须保持5d8。
存360个适配因子，但断点中的可训练参数/Adam仅216；恢复使用相同训练范围。
所有原整体/rare/保持/净32门槛保留，新增联合训练作为直接参考，不只和负控制比较。
下方沿用先前固定8个训练病例，不挑新成功样本；仅看训练题不能证明泛化。'''
    PAGE.write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>仅训练MLP</title><style>body{max-width:1100px;margin:28px auto;padding:0 18px;font:17px/1.7 system-ui;color:#1e293b;background:#f8fafc}article{background:white;border:1px solid #cbd5e1;padding:18px;margin:18px 0}img{width:100%;max-height:420px;object-fit:contain}pre{white-space:pre-wrap;background:#e2e8f0;padding:16px}table{border-collapse:collapse}td,th{border:1px solid #cbd5e1;padding:6px}</style><h1>固定已有q/v，让新增MLP学习纠错</h1><p>'+status+'</p><pre>'+html.escape(text)+'</pre><p><a href="'+relative(PROTOCOL)+'">预登记</a> · <a href="'+relative(RESULT)+'">结果</a> · <a href="'+relative(SOURCE_ROOT/'vlm_mlp_only_v1/common.py')+'">实际冻结代码</a> · <a href="'+relative(SOURCE_ROOT/'vlm_mlp_only_v1/train.py')+'">训练循环</a></p>'+''.join(cards)+'</html>')


def condition():
    from .train import train
    from .evaluate import evaluate
    directory=ROOT/'runs/candidate';directory.mkdir(parents=True,exist_ok=True);write_json(directory/'PHASE.json',{'phase':'training'})
    model,processor,result=train();write_json(directory/'PHASE.json',{'phase':'evaluation','completed_updates':64})
    evaluate(model,processor);write_json(directory/'COMPLETE.json',{'status':'completed','train':result});write_json(directory/'PHASE.json',{'phase':'completed'})


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--worker',action='store_true');args=p.parse_args()
    if args.prepare:prepare();return
    verify()
    if args.worker:condition();return
    from research_runtime.queue import run_queue
    run_queue(ROOT/'queue',[{'key':'candidate','command':[sys.executable,'-u','-m','vlm_mlp_only_v1.study','--worker'],'completion_marker':str(ROOT/'runs/candidate/COMPLETE.json')}],{'plan_sha256':sha_file(ROOT/'PLAN.json')})
    from .analysis import analyze
    result=analyze();render(result);write_json(ROOT/'COMPLETION.json',result);print('COMPLETED_MLP_ONLY',result['passed'],flush=True)


if __name__=='__main__':main()
