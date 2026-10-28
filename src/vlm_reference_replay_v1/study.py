import argparse
import ast
import html
import os
from pathlib import Path
import shutil
import sys
import time
import torch
from urllib.parse import quote
from rarevlm_paths import PROJECT_ROOT,SOURCE_ROOT
from vlm_deep.common import content_sha,sha_file,write_json
from vlm_human_sft.evaluate import lora_hash
from .common import ROOT,TRANSFER,DATA_SOURCE,ROLES,read,actor

PROTOCOL=PROJECT_ROOT/'docs/VLM_REFERENCE_REPLAY_V1_PROTOCOL.md'
RESULT=PROJECT_ROOT/'docs/VLM_REFERENCE_REPLAY_V1_RESULTS.md'
PAGE=PROJECT_ROOT/'docs/learning/教师KL保留与加强CE回放.html'


def prepare():
    if (ROOT/'PLAN.json').exists():raise FileExistsError('Already frozen')
    prior=read(TRANSFER/'PLAN.json');source=read(DATA_SOURCE/'DATA.json');transfer=read(TRANSFER/'DATA.json')
    training=source['training_rows']['expanded240']
    if len(training)!=640 or any(q['training_weight']!=1 for q in training):raise ValueError('Need fixed unweighted expanded240 data')
    schedule=read(DATA_SOURCE/'SCHEDULE.json')['roles']['expanded240'];lookup={q['question_id']:q for q in training}
    for block in schedule:
        for micro,ids in enumerate(block['micros']):
            if len(ids)!=4 or any(lookup[qid]['training_source']!=('replay' if micro in (0,2) else 'hard') for qid in ids):raise ValueError('Source/micro schedule differs')
    images={**source['images'],**transfer['images']};used={q['image_id'] for q in training}|{q['image_id'] for rows in transfer['evaluations'].values() for q in rows}
    images={iid:images[iid] for iid in used}
    pixel=lambda q:images[q['image_id']].get('pixel_sha256',images[q['image_id']]['sha256'])
    trained_pixels={pixel(q) for q in training}
    for name in ('seen_validation','transfer_dev','retention'):
        if trained_pixels&{pixel(q) for q in transfer['evaluations'][name]}:raise ValueError('New training/evaluation pixel overlap')
    required={'question_id','image_id','question','target','answers','stratum','evaluation_scored','expected_prefix_tokens','expected_grid'}
    if any(not required.issubset(q) for rows in transfer['evaluations'].values() for q in rows):raise ValueError('Missing generation metadata')
    write_json(ROOT/'DATA.json',{'training_rows':training,'hard_rows':source['hard_rows'],'images':images,'evaluations':transfer['evaluations']})
    write_json(ROOT/'SCHEDULE.json',schedule);shutil.copyfile(DATA_SOURCE/'CPU_INPUTS.json',ROOT/'CPU_INPUTS.json')
    for cohort in transfer['evaluations']:
        path=ROOT/'runs/initial'/cohort;path.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(TRANSFER/'runs/initial'/cohort/'PREDICTIONS.json',path/'PREDICTIONS.json')
    for name,old in [('training_context','training_context'),('seen_validation','validation'),('retention','retention')]:
        path=ROOT/'runs/plain'/name;path.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(DATA_SOURCE/'runs/expanded240'/old/'PREDICTIONS.json',path/'PREDICTIONS.json')
    first=[read_line(line) for line in (DATA_SOURCE/'runs/expanded240/train/INPUT_TRACES.jsonl').read_text().splitlines()]
    first=[q for q in first if q['update']==1]
    if len(first)!=16:raise ValueError('Missing original first16')
    write_json(ROOT/'FIRST_UPDATE_REFERENCE.json',first)
    points=[]
    for step in (16,32,48):
        dirs=sorted((DATA_SOURCE/'runs/expanded240/train').glob(f'checkpoint_{step:06d}_*'))
        if len(dirs)!=1:raise ValueError('Need one fixed control checkpoint per step')
        state=torch.load(dirs[0]/'state.pt',map_location='cpu',weights_only=False)
        points.append({'step':step,'adapter':str(dirs[0]/'adapter'),'state_path':str(dirs[0]/'state.pt'),
                       'trainable_sha256':state['trainable_tensor_sha256']})
        del state
    plain=read(DATA_SOURCE/'runs/expanded240/train/TRAIN_COMPLETE.json')
    paths=[PROTOCOL,Path(__file__),PROJECT_ROOT/'tools/run_reference_replay_v1.py',PROJECT_ROOT/'tests/test_vlm_reference_replay_v1.py',
           TRANSFER/'PLAN.json',TRANSFER/'EXECUTION_V2_LOCK.json',TRANSFER/'COMPLETION.json',TRANSFER/'POSTRUN_TRANSFER_REVIEW.json',
           DATA_SOURCE/'PLAN.json',DATA_SOURCE/'COMPLETION.json',DATA_SOURCE/'runs/expanded240/train/TRAIN_COMPLETE.json']
    paths+=list((SOURCE_ROOT/'vlm_reference_replay_v1').glob('*.py'))
    paths+=[ROOT/name for name in ('DATA.json','SCHEDULE.json','CPU_INPUTS.json','FIRST_UPDATE_REFERENCE.json')]
    paths+=list((ROOT/'runs/initial').glob('*/PREDICTIONS.json'))+list((ROOT/'runs/plain').glob('*/PREDICTIONS.json'))
    for item in points:paths+=[Path(item['state_path'])]+[Path(item['adapter'])/name for name in ('adapter_config.json','adapter_model.safetensors')]
    paths+=[Path(plain['adapter'])/name for name in ('adapter_config.json','adapter_model.safetensors')]
    plan={key:prior[key] for key in ('packages','base_weight_files','model_root','reference_config')}
    initial=prior['models']['initial']
    plan.update(status='prepared',initial_adapter=initial['adapter'],initial_lora_sha256=initial['lora_sha256'],
                plain_adapter=plain['adapter'],plain_lora_sha256=plain['final_lora_sha256'],calibration_checkpoints=points,
                files={**prior['files'],**read(DATA_SOURCE/'PLAN.json')['files'],**{str(path):sha_file(path) for path in paths}},
                images={v['path']:v['sha256'] for v in images.values()},roles=list(ROLES),new_formal_updates=128,new_generations=6939,
                automatic_more_training=False,automatic_main_evaluation=False)
    write_json(ROOT/'PLAN.json',plan)
    RESULT.write_text('# 教师KL与加强CE：准备完成\n\n同expanded240原数据/初始qv/64更新，原SFT未用图像dev1617与seen512分别评估；教师缓存和训练梯度校准尚未执行，无新提分结论。\n')
    render();print('PREPARED_REFERENCE_REPLAY',{'training_unique':640,'reused_training_cpu':830,'new_input_encodes':0,'formal_updates_if_admitted':128,'new_generations':6939},flush=True)


def read_line(line):
    import json
    return json.loads(line)


def verify():
    from vlm_object_train.train import package_lock
    plan=read(ROOT/'PLAN.json')
    for path,digest in {**plan['files'],**plan['images']}.items():
        if sha_file(path)!=digest:raise ValueError('Frozen input/source changed: '+path)
    if package_lock()!=plan['packages']:raise ValueError('Package versions changed')
    for path,value in plan['base_weight_files'].items():
        stat=Path(path).stat()
        if value!={'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns}:raise ValueError('Base storage changed')
    return plan


def condition(role):
    from .evaluate import evaluate
    plan=verify();directory=ROOT/'runs'/role;directory.mkdir(parents=True,exist_ok=True)
    if role=='plain':
        write_json(directory/'PHASE.json',{'phase':'fixed_plain_transfer_evaluation'})
        model,processor,_=actor(adapter=plan['plain_adapter'])
        if lora_hash(model)!=plan['plain_lora_sha256']:raise ValueError('Fixed plain control changed')
        result={'status':'fixed_reference','new_optimizer_updates':0,'lora_sha256':plan['plain_lora_sha256']}
    else:
        model=None;processor=None
        if role=='soft_kl':
            from .reference import build_cache,calibrate
            write_json(directory/'PHASE.json',{'phase':'teacher_cache','formal_updates':0});build_cache()
            write_json(directory/'PHASE.json',{'phase':'train_only_gradient_calibration','formal_updates':0})
            model,processor,cal=calibrate()
        else:
            started=time.monotonic()
            while not (ROOT/'CALIBRATION.json').exists():
                write_json(directory/'PHASE.json',{'phase':'waiting_train_only_calibration','formal_updates':0})
                if time.monotonic()-started>900:raise TimeoutError('Calibration not ready after900sec')
                time.sleep(15)
            cal=read(ROOT/'CALIBRATION.json')
        if not cal['training_allowed']:
            write_json(directory/'COMPLETE.json',{'status':'completed','outcome':'skipped_redundant_regularizer','optimizer_updates':0});return
        from .train import train
        write_json(directory/'PHASE.json',{'phase':'training','beta':cal['beta']})
        model,processor,result=train(role,model,processor)
        if lora_hash(model)!=result['final_lora_sha256']:raise ValueError('Final model identity differs')
    write_json(directory/'PHASE.json',{'phase':'evaluation'})
    evaluate(model,processor,role)
    write_json(directory/'COMPLETE.json',{'status':'completed','model':result})
    write_json(directory/'PHASE.json',{'phase':'completed'})


def analyze():
    from vlm_grpo_iteration_v1.study import paired
    cal=read(ROOT/'CALIBRATION.json')
    if not cal['training_allowed']:
        result={'status':'completed_diagnostic_no_training','calibration':cal,'goal_complete':False}
        write_json(ROOT/'COMPLETION.json',result);return result
    records={role:{name:[q for q in read(ROOT/'runs'/role/name/'PREDICTIONS.json') if q['evaluation_scored']]
                   for name in ('training_context','seen_validation','retention','transfer_dev')} for role in ('initial',)+ROLES}
    percent=lambda rows:100*sum(q['vqa_score'] for q in rows)/len(rows)
    scores={role:{name:percent(rows) for name,rows in groups.items()} for role,groups in records.items()}
    comparisons={};gates={};slices={}
    for cohort in ('seen_validation','transfer_dev'):
        for reference in ('initial','plain','hard_replay'):
            value=paired(records[reference][cohort],records['soft_kl'][cohort]);key=cohort+':'+reference
            comparisons[key]=value;gates[key+':overall']=value['delta_pp']>=.5 and value['ci95'][0]>0
            for field,values in [('stratum',('singleword','multiword')),('frequency_slice',sorted({q['frequency_slice'] for q in records['initial'][cohort]}))]:
                for label in values:
                    delta=paired([q for q in records[reference][cohort] if q[field]==label],[q for q in records['soft_kl'][cohort] if q[field]==label])
                    slices[key+':'+label]=delta;gates[key+':'+label]=delta['delta_pp']>=(.5 if reference=='initial' and label=='rare_1_5' else -.5)
    for reference in ('initial','plain','hard_replay'):gates[reference+':retention']=scores['soft_kl']['retention']-scores[reference]['retention']>=-.5
    full={role:sum(q['vqa_score']>=.999999 for q in groups['training_context']) for role,groups in records.items()};gates['training_net32']=full['soft_kl']-full['initial']>=32
    protected={q['question_id'] for q in records['initial']['seen_validation'] if q['stratum']=='multiword' and q['vqa_score']>=.999999}
    keep=paired([q for q in records['initial']['seen_validation'] if q['question_id'] in protected],[q for q in records['soft_kl']['seen_validation'] if q['question_id'] in protected]);gates['seen_initial_full_multiword_retention']=keep['delta_pp']>=-.5
    result={'status':'completed','passed':all(gates.values()),'scores_percent':scores,'comparisons':comparisons,'slices':slices,'gates':gates,
            'training_full':full,'calibration':cal,'previously_full_multiword':keep,'goal_complete':False,
            'next':'Joint pass permits separate confirmation only;no automatic extra steps/seeds/formal/CV.'}
    write_json(ROOT/'ANALYSIS.json',result)
    table='|组|训练128满分|seen512%|SFT未用图像dev1617%|保持64%|\n|---|---:|---:|---:|---:|\n'
    for role,v in scores.items():table+=f'|{role}|{full[role]}|{v["seen_validation"]:.6f}|{v["transfer_dev"]:.6f}|{v["retention"]:.6f}|\n'
    RESULT.write_text('# 教师KL与加强CE完整结果\n\n'+table+'\nβ='+str(cal['beta'])+'；联合门槛='+str(result['passed'])+'。单seed内部观察，不称盲测/三seed稳定。\n\n[协议](VLM_REFERENCE_REPLAY_V1_PROTOCOL.md) · [原图与损失原理](learning/教师KL保留与加强CE回放.html)\n')
    render(result);write_json(ROOT/'COMPLETION.json',result);return result


def render(result=None):
    prior=read(TRANSFER/'POSTRUN_TRANSFER_REVIEW.json');data=read(ROOT/'DATA.json');relative=lambda path:quote(os.path.relpath(path,PAGE.parent),safe='/._-')
    saved={role:{q['question_id']:q for q in read(ROOT/'runs'/role/'transfer_dev/PREDICTIONS.json')} for role in ('initial',)+ROLES if (ROOT/'runs'/role/'transfer_dev/PREDICTIONS.json').exists()}
    cards=[]
    for case in prior['examples']:
        body=''
        for role,values in saved.items():
            if case['question_id'] in values:
                q=values[case['question_id']];body+='<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in (role,q['prediction'],q['vqa_score']))+'</tr>'
        cards.append(f'<article><h2>dev q{case["question_id"]}</h2><img loading="lazy" src="{relative(case["image"])}" alt="前轮多词降分原图"><p>{html.escape(case["question"])}</p><p>原目标：{html.escape(case["target"])}；旧merger回答：{html.escape(case["candidate"])}</p><p>这是事后讲解病例，不改标签，不据这些问题选β或加入训练。</p><table><tr><th>组</th><th>本轮实际生成</th><th>VQA</th></tr>{body}</table></article>')
    status='准备/运行中，尚无新提分结果。' if result is None else html.escape(str(result['scores_percent']))+'；联合通过='+str(result['passed'])
    PAGE.write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>教师KL与加强CE</title><style>body{max-width:1100px;margin:30px auto;padding:0 18px;font:17px/1.7 system-ui;background:#f8fafc;color:#1e293b}article{background:white;padding:18px;margin:18px 0;border:1px solid #cbd5e1}img{width:100%;max-height:440px;object-fit:contain}pre{white-space:pre-wrap;background:#e2e8f0;padding:18px}td,th{border:1px solid #cbd5e1;padding:6px}table{border-collapse:collapse}</style><h1>纠错与保留：软分布还是加强硬标签</h1><p>'+status+'</p><pre>两新模型同数据/起点/64步/β，qv1843200参数。\nL_base = (Σ8hard CE + Σ8replay CE) / 16\nsoft_KL = L_base + β · mean_replay KL(p_初始 || p_当前)\nhard_replay = L_base + β · mean_replay CE_GT\nKL方向是初始教师→当前学生，完整词表151936/T=1。\nβ仅由训练checkpoint16/32/48的梯度范数比中位数校准，截在.1–100。\n缓存128批/512QA完整GT前缀分布，输入/labels/grid/pixels字节必须匹配。\n相同β不保证梯度/Adam步长或算力相同，强CE可初始非0、KL近0合法。\n训练前恢复原SFT，不从校准状态继续；无新奖励/策略梯度。\nseen512与SFT未用图像dev1617分列，均须超过初始、普通CE、加强CE。\n以下保存原图与实际预测，分数变化不自动等于语义优劣。</pre>'+f'<p><a href="{relative(PROTOCOL)}">预登记</a> · <a href="{relative(RESULT)}">结果</a> · <a href="{relative(SOURCE_ROOT/"vlm_reference_replay_v1/loss.py")}">KL代码</a> · <a href="{relative(SOURCE_ROOT/"vlm_reference_replay_v1/reference.py")}">梯度校准</a></p>'+''.join(cards)+'</html>')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--role',choices=ROLES);args=parser.parse_args()
    if args.prepare:prepare()
    elif args.role:condition(args.role)
    else:
        from research_runtime.queue import run_queue
        verify();run_queue(ROOT/'queue',[{'key':role,'command':[sys.executable,'-u','-m','vlm_reference_replay_v1.study','--role',role],
                  'completion_marker':str(ROOT/'runs'/role/'COMPLETE.json')} for role in ROLES],{'plan_sha256':sha_file(ROOT/'PLAN.json')})
        result=analyze();print('COMPLETED_REFERENCE_REPLAY',result.get('passed',result['status']),flush=True)


if __name__=='__main__':main()
