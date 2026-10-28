import argparse
import html
import os
from pathlib import Path
import shutil
import sys
from urllib.parse import quote
from rarevlm_paths import PROJECT_ROOT,SOURCE_ROOT
from vlm_deep.common import sha_file,write_json
from .common import ROOT,PARENT,SCOPE,JOINT,DIVERSITY,PROTOCOL,RESULT,PAGE,read


def prepare():
    if (ROOT/'PLAN.json').exists():raise FileExistsError('Already frozen')
    parent=read(SCOPE/'PLAN.json');completed=read(SCOPE/'COMPLETION.json')
    if completed['status']!='completed' or completed['passed']:raise ValueError('Expected completed exploratory MLP-only result')
    if not read(SCOPE/'runs/candidate/train/FROZEN_QV_CHECK.json')['qv_values_unchanged']:raise ValueError('Parent q/v was not preserved')
    points=[]
    for step in (16,32,48):
        dirs=list((SCOPE/'runs/candidate/train').glob(f'checkpoint_{step:06d}_*'))
        if len(dirs)!=1:raise ValueError('Need exactly one fixed checkpoint for step '+str(step))
        points.append({'step':step,'adapter':str(dirs[0]/'adapter')})
    for name in ('DATA.json','CPU_INPUTS.json','SCHEDULE.json','FIRST_UPDATE_REFERENCE.json'):
        destination=ROOT/name;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(PARENT/name,destination)
    cache=read(PARENT/'CACHE_COMPLETE.json')
    reuse={'status':'prepared','original_cache_complete_sha256':sha_file(PARENT/'CACHE_COMPLETE.json'),'original_teacher_qv_sha256':read(PARENT/'PLAN.json')['initial_lora_sha256'],
           'actor_initial_sha256':parent['initial_lora_sha256'],'teacher_packets':len(cache['files']),'reference_qa':512,'reference_bytes':cache['bytes'],
           'original_producer_uuid':read(PARENT/'CACHE_INPUT_LOCK.json')['producer_uuid'],'new_teacher_forwards':0,'new_teacher_generations':0,
           'gpu_admission':'Same current numerical policy, exact cached frames, initial replay8 absKL<=1e-5; no forced original producer UUID.',
           'rationale':'q/v original values and zero-B MLP give same intended initial function; actual compatibility still must pass on GPU.'}
    write_json(ROOT/'CACHE_REUSE.json',reuse)
    paths=list((SOURCE_ROOT/'vlm_mlp_reference_v1').glob('*.py'))+[PROTOCOL,PROJECT_ROOT/'tools/run_mlp_reference_v1.py',SCOPE/'PLAN.json',SCOPE/'COMPLETION.json',SCOPE/'runs/candidate/train/FROZEN_QV_CHECK.json',PARENT/'CACHE_COMPLETE.json',PARENT/'CACHE_INPUT_LOCK.json',ROOT/'CACHE_REUSE.json']
    paths += [ROOT/n for n in ('DATA.json','CPU_INPUTS.json','SCHEDULE.json','FIRST_UPDATE_REFERENCE.json')]
    paths += list((SCOPE/'runs/candidate').glob('*/PREDICTIONS.json'))
    for point in points:paths += [Path(point['adapter'])/name for name in ('adapter_config.json','adapter_model.safetensors')]
    plan={'status':'prepared','files':{**parent['files'],**cache['files'],**{str(p):sha_file(p) for p in paths}},'images':parent['images'],
          'initial_adapter':str(JOINT/'initial_adapter'),'initial_lora_sha256':parent['initial_lora_sha256'],'original_qv_sha256':parent['original_qv_sha256'],
          'calibration_checkpoints':points,'trainable_parameters':2820096,'stored_adapter_parameters':4663296,'roles':['soft_kl','hard_replay'],
          'new_formal_updates_if_admitted':128,'new_generations_if_admitted':5322,'new_teacher_cache_bytes':0,'new_teacher_generations':0,
          'calibration_forward_qa':56,'calibration_optimizer_updates':0,'automatic_more_training':False,'automatic_formal_evaluation':False}
    write_json(ROOT/'PLAN.json',plan);render()
    RESULT.write_text('# MLP函数保持：准备完成\n\n已绑定原640训练QA、216可训练MLP因子、旧512QA教师缓存；无新教师生成或输入编码。等待GPU初始KL兼容与训练梯度校准，尚无新提分。\n\n[协议](VLM_MLP_REFERENCE_V1_PROTOCOL.md) · [原图和损失](learning/MLP输出分布保持_KL与硬回放.html)\n')
    print('PREPARED_MLP_REFERENCE',{'cache_packets':len(cache['files']),'new_cache_bytes':0,'conditional_updates':128,'conditional_generations':5322},flush=True)


def verify():
    plan=read(ROOT/'PLAN.json')
    for path,digest in {**plan['files'],**plan['images']}.items():
        if sha_file(path)!=digest:raise ValueError('Frozen input/source changed: '+path)


def render(result=None):
    data=read(ROOT/'DATA.json');relative=lambda p:quote(os.path.relpath(p,PAGE.parent),safe='/._-');cards=[]
    for case in read(JOINT/'LEARNING_CASES.json'):
        rows=''
        for label,base,role in [('initial',PARENT,'initial'),('qv_CE',PARENT,'plain'),('MLP_CE',SCOPE,'candidate'),('MLP_hard',ROOT,'hard_replay'),('MLP_KL',ROOT,'soft_kl')]:
            path=base/'runs'/role/'training_context/PREDICTIONS.json'
            if path.exists():
                q=next((q for q in read(path) if q['question_id']==case['question_id']),None)
                if q:rows+='<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in (label,q['prediction'],q['vqa_score']))+'</tr>'
        cards.append(f'<article><h2>训练题 q{case["question_id"]} · {case["review_error_type"]}</h2><img loading="lazy" src="{relative(data["images"][case["image_id"]]["path"])}" alt="原始训练图片"><p>{html.escape(case["question"])}</p><p>原目标：{html.escape(case["target"])}</p><table><tr><th>模型</th><th>保存的实际回答</th><th>VQA</th></tr>{rows}</table><p>固定训练病例，不替代独立评测。</p></article>')
    status='准备/运行中，尚无新能力提升。' if result is None else ('训练侧校准判为冗余，未开始正式训练。' if result.get('outcome') else '完整评测结束；KL联合通过='+str(result['passed'])+'。')
    cal=read(ROOT/'CALIBRATION.json') if (ROOT/'CALIBRATION.json').exists() else None
    text='''固定原q/v，只更新2820096个MLP参数；图结构/存储4663296不变。
Lbase = (Σ8hard CE + Σ8replay CE)/16。
soft_KL: Lbase + β·mean_replay KL(p_initial || p_actor)。
hard_CE: Lbase + 同β·mean_replay CE_GT。
KL逐有效assistant token对151936完整词表求和，再token平均，再QA平均，T=1。
正确回放题保留原模型的不确定分布；128纠错题没有KL，避免将原错答案蒸馏回来。
教师来自原SFT，只在GT前缀；不等于约束所有自由生成历史或保证视觉/语义保持。
不重建1.186GB教师缓存，旧输入/labels/grid/pixels exact绑定；实际初始8replay KL必须近0。
新β只用MLP-only训练checkpoint16/32/48及next block梯度范数比中位数，clip[.1,100]；不用dev分。
相同β不保证两种目标梯度或Adam步长相等，硬CE初始可以非零，KL应近0。
若三点KL与CE几乎相同则不训练；否则fresh原始初始化/Adam跑两臂各64步。
同640QA/1024曝光/原生成与评分，记录软目标额外计算；初始和所有强参考预测复用。
seen512与SFT图像隔离1617分别全门槛验收，不能只对上轮负模型宣布成功。
下方8张训练图沿用旧固定病例，训练满分和独立图像提分分开。'''
    PAGE.write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MLP函数保持</title><style>body{max-width:1100px;margin:28px auto;padding:0 18px;font:17px/1.7 system-ui;background:#f8fafc;color:#1e293b}article{background:white;border:1px solid #cbd5e1;padding:18px;margin:18px 0}img{width:100%;max-height:420px;object-fit:contain}pre{white-space:pre-wrap;background:#e2e8f0;padding:16px}td,th{border:1px solid #cbd5e1;padding:6px}table{border-collapse:collapse}</style><h1>固定权重后，继续约束输出分布</h1><p>'+status+'</p><pre>'+html.escape(text)+'</pre><p><a href="'+relative(PROTOCOL)+'">预登记</a> · <a href="'+relative(SOURCE_ROOT/'vlm_mlp_reference_v1/reference.py')+'">真实校准代码</a> · <a href="'+relative(SOURCE_ROOT/'vlm_mlp_reference_v1/train.py')+'">训练循环</a></p><h2>实际训练侧校准</h2><pre>'+html.escape(str(cal))+'</pre>'+''.join(cards)+'</html>')


def condition(role):
    from .train import train
    from .evaluate import evaluate
    directory=ROOT/'runs'/role;directory.mkdir(parents=True,exist_ok=True);write_json(directory/'PHASE.json',{'phase':'training'})
    model,processor,result=train(role);write_json(directory/'PHASE.json',{'phase':'evaluation','completed_updates':64})
    evaluate(model,processor,role);write_json(directory/'COMPLETE.json',{'status':'completed','train':result});write_json(directory/'PHASE.json',{'phase':'completed'})


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--role',choices=['calibration','soft_kl','hard_replay']);args=p.parse_args()
    if args.prepare:prepare();return
    verify()
    if args.role=='calibration':
        from .reference import calibrate
        calibrate();return
    if args.role:condition(args.role);return
    from research_runtime.queue import run_queue
    identity={'plan_sha256':sha_file(ROOT/'PLAN.json')}
    run_queue(ROOT/'calibration_queue',[{'key':'calibration','command':[sys.executable,'-u','-m','vlm_mlp_reference_v1.study','--role','calibration'],'completion_marker':str(ROOT/'CALIBRATION.json')}],identity)
    cal=read(ROOT/'CALIBRATION.json');render()
    if not cal['training_allowed']:
        result={'status':'completed','passed':False,'goal_complete':False,'outcome':'redundant_no_training','calibration':cal}
        render(result);write_json(ROOT/'COMPLETION.json',result);RESULT.write_text('# MLP函数保持：梯度校准未准入\n\nKL与硬CE按事前标准基本冗余，未开始正式训练。\n');return
    jobs=[{'key':role,'command':[sys.executable,'-u','-m','vlm_mlp_reference_v1.study','--role',role],'completion_marker':str(ROOT/'runs'/role/'COMPLETE.json')} for role in ('soft_kl','hard_replay')]
    run_queue(ROOT/'training_queue',jobs,{**identity,'calibration_sha256':sha_file(ROOT/'CALIBRATION.json')})
    from .analysis import analyze
    result=analyze();render(result);write_json(ROOT/'COMPLETION.json',result);print('COMPLETED_MLP_REFERENCE',result['passed'],flush=True)


if __name__=='__main__':main()
