"""Data-supervision experiment: fixed baseline -> two matched arms -> screen/confirm."""

import argparse
from collections import Counter
import html
import json
import os
from pathlib import Path
import sys
import time
from urllib.parse import quote

from rarevlm_paths import PROJECT_ROOT,SOURCE_ROOT
from vlm_deep.common import NORMALIZER,content_sha,read_manifest,sha_file,write_json

ROOT=PROJECT_ROOT.parents[1]/'outputs/vlm_region_ocr_sft_v1'
CONFIG=PROJECT_ROOT/'configs/vlm_region_ocr_sft_v1.json'
ROLES=('control','region')


def read(path):return json.loads(Path(path).read_text())


def freeze(path,value):
    path=Path(path)
    if path.exists() and read(path)!=value:raise ValueError('Frozen study input differs: '+str(path))
    write_json(path,value)


def jsonl(path):return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def render_learning(result=None):
    page=PROJECT_ROOT/'docs/learning/区域文字监督SFT_原图与对照.html'
    preview=read(ROOT/'data/VISUAL_PREVIEW.json');m=read(ROOT/'data/REGION_TRAIN.json')
    link=lambda path:quote(os.path.relpath(path,page.parent),safe='/._-')
    cards=[]
    for v in preview:
        q=v['region'];im=m['images'][v['image_id']];x0,y0,x1,y1=q['box_xyxy_original_pixels']
        cards.append(f'<article><h3>官方转写：{html.escape(q["target"])}</h3>'
            f'<svg viewBox="0 0 {im["width"]} {im["height"]}" role="img" aria-label="训练区域标注展示">'
            f'<image href="{link(im["path"])}" width="{im["width"]}" height="{im["height"]}"/>'
            f'<rect x="{x0}" y="{y0}" width="{x1-x0}" height="{y1-y0}" fill="none" stroke="#ef4444" stroke-width="3"/></svg>'
            f'<p>{html.escape(q["question"])}</p><p>共同QA：{html.escape(v["common"]["question"])}</p>'
            f'<p>控制辅助QA：{html.escape(v["control"]["question"])}</p></article>')
    scores='' if result is None else '<h2>完成结果：'+result['stage']+'</h2><table><tr><th>模型</th><th>TextVQA</th><th>Human</th></tr>'+''.join(
        f'<tr><th>{role}</th><td>{v["text"]:.6f}</td><td>{v["human"]:.6f}</td></tr>' for role,v in result['scores_percent'].items())+'</table><p>联合门槛通过：'+str(result['passed'])+'；单seed探索，完整升降与区间见结果JSON。</p>'
    text='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>区域文字监督SFT</title><style>body{max-width:1120px;margin:30px auto;padding:0 20px;font:17px/1.7 system-ui;color:#1e293b;background:#f8fafc}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:18px}article{background:white;padding:16px;border:1px solid #cbd5e1;border-radius:8px}svg{width:100%;max-height:650px;background:#f1f5f9}h3{font-size:18px}table{border-collapse:collapse;background:white;width:100%}td,th{border:1px solid #cbd5e1;padding:10px;text-align:left}pre{background:#0f172a;color:#e2e8f0;padding:18px;white-space:pre-wrap}.note{padding:15px;background:#fff7ed}a{color:#0369a1}</style><h1>区域文字证据监督：同图SFT对照</h1><p>研究假设：新增明确区域与官方转写监督，能改善原图TextVQA及罕见监督答案，并保留人工yes/no能力。已复用1600张已有训练图，两组各3200唯一QA，所有4800唯一输入经CPU真实processor编码。</p><table><tr><th>部分</th><th>控制</th><th>候选</th></tr><tr><td>1600共同QA</td><td>原图＋原TextVQA问题</td><td>完全相同</td></tr><tr><td>1600辅助QA</td><td>同图另一条原问题</td><td>同图＋区域坐标→官方文字</td></tr><tr><td>参数/优化</td><td>q/v LoRA1843200；普通CE/AdamW</td><td>相同</td></tr></table><pre>L_update = Σ16个QA的平均assistant-token CE / 16
序列：图像＋问题（含区域坐标）→ assistant真实文字＋结束符
训练图片没有绘入红框，原图评测不使用GT文字框。</pre><p class="note">下方红框只按官方points在学习页叠加，帮助定位；不是模型attention、不是裁剪后的训练输入。</p>'''+scores+'<h2>固定训练预览：原图、区域和同图控制</h2><div class="cards">'+''.join(cards)+'</div><p>新初始、控制与候选统一采用BF16底座、FP32 LoRA/冻结输出头/CE、显式raw greedy。训练200更新/3200曝光，45分钟/臂为上限，不是ETA。初始64条区域训练题的规范化错误至少4题才放行；该训练信号检查不是泛化收益。</p><p><a href="'+link(PROJECT_ROOT/'docs/VLM_REGION_OCR_SFT_V1_PROTOCOL.md')+'">完整协议</a> · <a href="'+link(SOURCE_ROOT/'vlm_region_ocr_sft_v1/train.py')+'">真实训练循环</a> · <a href="'+link(ROOT/'DATA_COMPLETE.json')+'">数据与输入记录</a></p></html>'
    page.write_text(text)


def prepare():
    from vlm_object_train.train import package_lock
    c=read(CONFIG);data=read(ROOT/'DATA_COMPLETE.json');review=read(ROOT/'VISUAL_REVIEW.json')
    if data['cpu_unique_qa']!=4800 or not all(q['supported'] for q in review['rows']):
        raise ValueError('New input/visual review not ready')
    role_rows={role:{q['question_id']:q for q in read_manifest(ROOT/f'data/{role.upper()}_TRAIN.json')['splits']['train_all']} for role in ROLES}
    schedules=read(ROOT/'data/SCHEDULE.json')['roles']
    image_orders={role:[[role_rows[role][q]['image_id'] for q in micro] for u in schedules[role] for micro in u['micros']] for role in ROLES}
    if image_orders['control']!=image_orders['region']:raise ValueError('Matched image order differs between arms')
    paths=[CONFIG,PROJECT_ROOT/'docs/VLM_REGION_OCR_SFT_V1_PROTOCOL.md',
        PROJECT_ROOT/'tools/prepare_region_ocr_sft_v1.py',PROJECT_ROOT/'tests/test_vlm_region_ocr_sft_v1.py',
        ROOT/'DATA_POLICY.json',ROOT/'DATA_COMPLETE.json',ROOT/'VISUAL_REVIEW.json']
    paths+=list((ROOT/'data').glob('*.json'))+[ROOT/'data/CPU_INPUTS.jsonl']
    paths+=list((SOURCE_ROOT/'vlm_region_ocr_sft_v1').glob('*.py'))
    paths+=[SOURCE_ROOT/name for name in ('vlm_fp32_readout_v1/model.py','vlm_runtime_metadata_v1/model.py',
        'vlm_fp32_adapter/model.py','vlm_fp32_adapter/loss.py','vlm_deep/common.py','vlm_sft/common.py',
        'vlm_object_train/train.py','vlm_probe/evaluate.py','vlm_human_sft/evaluate.py',
        'vlm_attention_scope_repair_v1/generation.py','vlm_sft_resolution/experiment.py','research_runtime/queue.py')]
    base=read(c['baseline_config']);paths+=[Path(c['baseline_config']),Path(c['source_manifest']),Path(base['reference_config'])]
    paths+=[Path(c['initial_adapter'])/n for n in ('adapter_model.safetensors','adapter_config.json')]
    freeze(ROOT/'PRELAUNCH_INPUT_LOCK.json',{'files':{str(p):sha_file(p) for p in sorted(set(paths))},
        'annotation_sha256':data['annotation_sha256'],'annotation_bytes':data['annotation_bytes'],
        'packages':package_lock(),'base_weight_files':{str(p):{'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns}
            for p in sorted(Path(base['model_root']).glob('*.safetensors'))},
        'training_updates_per_role':200,'old_scientific_settings_modified':False})
    render_learning()
    print('PREPARED_REGION_OCR_STUDY',flush=True)


def verify():
    from vlm_object_train.train import package_lock
    lock=read(ROOT/'PRELAUNCH_INPUT_LOCK.json')
    for path,expected in lock['files'].items():
        if sha_file(path)!=expected:raise ValueError('Registered source/data changed: '+path)
    if package_lock()!=lock['packages']:raise ValueError('Registered package versions changed')
    for path,expected in lock['base_weight_files'].items():
        s=Path(path).stat()
        if {'bytes':s.st_size,'mtime_ns':s.st_mtime_ns}!=expected:raise ValueError('Base storage changed')


def actor(adapter):
    import torch
    from vlm_fp32_readout_v1.model import load
    from vlm_probe.evaluate import processor_for
    c=read(CONFIG);base=read(c['baseline_config'])
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model,binding=load(base['model_root'],adapter,torch.device('cuda:0'),training=False)
    return model,processor_for(Path(base['model_root']),read(base['reference_config']),1024),binding


def eval_config(stage):
    c=read(CONFIG);base=read(c['baseline_config'])
    if stage=='confirm' and not read(ROOT/'SCREEN_ANALYSIS.json')['passed']:raise ValueError('Confirmation not admitted')
    return {**base,'initial_adapter':c['initial_adapter'],
        'text_manifest':str(ROOT/f'data/{stage.upper()}_TEXT.json'),
        'human_manifest':str(ROOT/f'data/{stage.upper()}_HUMAN.json'),
        'evaluation_budget':1024,'evaluation_batch_size':4,'max_new_tokens':32,
        'expected_eval_questions':{'human':512,'text':512 if stage=='screen' else read(ROOT/'CONFIRM_DATA.json')['text_questions']}}


def region_probe(model,processor):
    import torch
    from vlm_probe.evaluate import evaluate_batch
    data=read(ROOT/'data/REGION_TRAIN_PROBE.json');rows=data['qa'];directory=ROOT/'initial/region_train_probe'
    directory.mkdir(parents=True,exist_ok=True);path=directory/'PREDICTIONS.json'
    predictions=read(path) if path.exists() else []
    if len(predictions)%4 or [v['question_id'] for v in predictions]!=[q['question_id'] for q in rows[:len(predictions)]]:
        raise ValueError('Training probe cache grouping differs')
    model.eval();model.config.use_cache=True
    from vlm_attention_scope_repair_v1.generation import raw_greedy_config
    model.generation_config=raw_greedy_config(model.generation_config,32)
    with torch.inference_mode():
        for start in range(len(predictions),len(rows),4):
            chunk=rows[start:start+4]
            answers,_=evaluate_batch(model,processor,chunk,data['images'],ROOT/'data',next(model.parameters()).device,
                1024,processor.tokenizer.convert_tokens_to_ids('<|image_pad|>'),32)
            predictions.extend({'question_id':q['question_id'],'image_id':q['image_id'],'target':q['target'],
                **a,'raw_exact':a['prediction']==q['target'],
                'normalized_correct':bool(a['prediction'].strip()) and NORMALIZER(a['prediction'])==NORMALIZER(q['target'])}
                for q,a in zip(chunk,answers))
            write_json(path,predictions)
    errors=sum(not q['normalized_correct'] for q in predictions)
    result={'status':'completed','qa':64,'normalized_errors':errors,'raw_exact_correct':sum(q['raw_exact'] for q in predictions),
        'headroom_passed':errors>=read(CONFIG)['minimum_normalized_probe_errors'],
        'scope':'Fixed training-only signal check; no heldout gain or outcome-based resampling','training_updates':0}
    write_json(directory/'COMPLETE.json',result);return result


def baseline(task):
    from vlm_attention_scope_repair_v1.generation import evaluate
    from vlm_human_sft.evaluate import lora_hash
    verify();c=read(CONFIG);model,processor,binding=actor(c['initial_adapter'])
    freeze(ROOT/f'initial/{task}/MODEL_IDENTITY.json',{'lora_sha256':lora_hash(model),'binding':binding})
    evaluate(model,processor,eval_config('screen'),task,ROOT/'initial'/task)
    if task=='text':region_probe(model,processor)
    write_json(ROOT/f'initial/{task}/BASELINE_COMPLETE.json',{'status':'completed','task':task})


def condition(role):
    from .train import train
    from vlm_attention_scope_repair_v1.generation import evaluate
    from vlm_human_sft.evaluate import lora_hash
    verify();c=read(CONFIG)
    model,processor,result=train(c,role,ROOT/'STUDY_INPUT_LOCK.json')
    if lora_hash(model)!=result['final_lora_sha256']:raise ValueError('Trained model identity differs')
    evaluations={task:evaluate(model,processor,eval_config('screen'),task,ROOT/'runs'/role/task) for task in ('text','human')}
    if lora_hash(model)!=result['final_lora_sha256']:raise ValueError('Read-only evaluation changed adapter')
    write_json(ROOT/'runs'/role/'COMPLETE.json',{'status':'completed','train':result,'evaluations':evaluations})


def analyze(stage):
    from vlm_sft_resolution.experiment import paired_interval
    c=read(CONFIG);ec=eval_config(stage)
    gold={task:read_manifest(ec[task+'_manifest'])['splits']['dev'] for task in ('text','human')}
    original=read_manifest(c['source_manifest']);frequency=Counter(NORMALIZER(q['target']) for q in original['splits']['train_all'])
    components={}
    for pair in read(ROOT/f'data/{stage.upper()}_PAIRS.json'):
        for key in ('positive_question_id','negative_question_id'):components[pair[key]]=pair['component_id']
    records={}
    for role in ('initial',)+ROLES:
        base=ROOT/'confirmation'/role if stage=='confirm' else ROOT/'initial' if role=='initial' else ROOT/'runs'/role
        records[role]={task:jsonl(base/task/'predictions.jsonl') for task in ('text','human')}
        for task in gold:
            if [(q['question_id'],q['image_id']) for q in gold[task]]!=[(q['question_id'],q['image_id']) for q in records[role][task]]:
                raise ValueError('Incomplete or unpaired generated evaluation')
    def compare(before,task,mask):
        def transform(role):
            return [{'question_id':q['question_id'],'image_id':components[q['question_id']] if task=='human' else q['image_id'],
                     'score':q['primary_score']} for q,keep in zip(records[role][task],mask) if keep]
        left,right=transform(before),transform('region')
        interval=paired_interval(left,right,c['bootstrap_replicates'],c['bootstrap_seed'])
        return {**interval,'corrected_qa':sum(b['score']>a['score'] for a,b in zip(left,right)),
            'broken_qa':sum(b['score']<a['score'] for a,b in zip(left,right)),
            'net_score_points':sum(b['score']-a['score'] for a,b in zip(left,right))}
    results={};criteria={}
    for before in ('initial','control'):
        comparisons={}
        for name,select in [('overall',lambda n:True),('rare_1_5',lambda n:1<=n<=5),('frequent_gt20',lambda n:n>20)]:
            value=compare(before,'text',[select(frequency[NORMALIZER(q['target'])]) for q in gold['text']]);comparisons['text_'+name]=value
            if name in ('overall','rare_1_5'):criteria[before+'_text_'+name+'_gain']=value['delta_pp']>=c['minimum_gain_pp']
            else:criteria[before+'_text_frequent_protection']=value['delta_pp']>=-c['protection_margin_pp']
            if name=='overall':criteria[before+'_text_overall_positive_ci']=value['image_cluster_95_ci_pp'][0]>0
        for label in ('overall','yes','no'):
            value=compare(before,'human',[label=='overall' or q['target']==label for q in gold['human']]);comparisons['human_'+label]=value
            criteria[before+'_human_'+label+'_protection']=value['delta_pp']>=-c['protection_margin_pp']
        results[before]=comparisons
    scores={role:{task:100*sum(q['primary_score'] for q in rows)/len(rows) for task,rows in tasks.items()} for role,tasks in records.items()}
    changed={before:{task:[{'question_id':a['question_id'],'image_id':a['image_id'],
        'before':a['prediction'],'after':b['prediction'],'before_score':a['primary_score'],'after_score':b['primary_score']}
        for a,b in zip(records[before][task],records['region'][task]) if a['prediction']!=b['prediction']]
        for task in ('text','human')} for before in ('initial','control')}
    result={'status':'completed','stage':stage,'passed':all(criteria.values()),'scores_percent':scores,
        'comparisons':results,'criteria':criteria,'scope':'Single seed42, historically exposed internal data; no formal multi-seed claim',
        'training_updates_per_role':200,'original_yes_no_branch_gates_unchanged':True}
    write_json(ROOT/f'{stage.upper()}_CHANGED_CASES.json',changed);write_json(ROOT/f'{stage.upper()}_ANALYSIS.json',result)
    render_learning(result)
    lines=['# 区域文字监督SFT结果','',f'阶段：{stage}；联合通过：{result["passed"]}。','',
        '|模型|TextVQA|Human|','|---|---:|---:|']
    lines += [f'|{role}|{v["text"]:.6f}|{v["human"]:.6f}|' for role,v in scores.items()]
    lines += ['',result['scope'],'','两组新增监督、输入与计算成本、罕见切片和完整升降见outputs/vlm_region_ocr_sft_v1。']
    (PROJECT_ROOT/'docs/VLM_REGION_OCR_SFT_V1_RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result),flush=True);return result


def prepare_confirmation():
    c=read(CONFIG);original=read_manifest(c['source_manifest']);screen=read_manifest(ROOT/'data/SCREEN_TEXT.json')
    screen_images={q['image_id'] for q in screen['splits']['dev']}
    remaining=[q for q in original['splits']['dev'] if q['image_id'] not in screen_images]
    images={q['image_id']:{**original['images'][q['image_id']],
        'path':str((Path(c['source_manifest']).parent/original['images'][q['image_id']]['path']).resolve())} for q in remaining}
    manifest={'schema_version':original['schema_version'],'purpose':'Non-screen internal dev; historically exposed, no new blind-test claim',
        'images':images,'splits':{'train_all':[],'train_10k':[],'dev':remaining,'final':[]}}
    manifest['manifest_sha256']=content_sha(manifest);freeze(ROOT/'data/CONFIRM_TEXT.json',manifest)
    old=PROJECT_ROOT.parents[1]/'outputs/vlm_complementary_rank_v1'
    freeze(ROOT/'data/CONFIRM_HUMAN.json',read(old/'data/CONFIRM_EVAL.json'))
    source=read(read(PROJECT_ROOT/'configs/vlm_complementary_rank_v1.json')['dataset_manifest'])
    freeze(ROOT/'data/CONFIRM_PAIRS.json',source['complementary_pairs']['confirm'])
    freeze(ROOT/'CONFIRM_DATA.json',{'text_questions':len(remaining),'note':'Remaining internal dev, not never-exposed benchmark'})


def confirm(role):
    from vlm_attention_scope_repair_v1.generation import evaluate
    verify();c=read(CONFIG);adapter=c['initial_adapter'] if role=='initial' else str(ROOT/'runs'/role/'train/adapter')
    model,processor,_=actor(adapter)
    for task in ('text','human'):evaluate(model,processor,eval_config('confirm'),task,ROOT/'confirmation'/role/task)
    write_json(ROOT/'confirmation'/role/'COMPLETE.json',{'status':'completed','training_updates':0})


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--baseline',choices=('text','human'));parser.add_argument('--role',choices=ROLES)
    parser.add_argument('--confirm',choices=('initial',)+ROLES);args=parser.parse_args()
    if args.prepare:prepare();return
    if args.baseline:baseline(args.baseline);return
    if args.role:
        try:condition(args.role)
        except Exception as exc:
            write_json(ROOT/'runs'/args.role/'ERROR.json',{'status':'failed','error':repr(exc),'existing_tasks_not_terminated':True})
            progress=ROOT/'runs'/args.role/'train/PROGRESS.json'
            write_json(progress,{**(read(progress) if progress.exists() else {}),'status':'failed','error':repr(exc)})
            raise
        return
    if args.confirm:confirm(args.confirm);return
    verify()
    from research_runtime.queue import run_queue
    run_queue(ROOT/'baseline_queue',[{'key':task,'command':[sys.executable,'-u','-m',__package__+'.study','--baseline',task],
        'completion_marker':str(ROOT/'initial'/task/'BASELINE_COMPLETE.json')} for task in ('text','human')],
        {'prelaunch_sha256':sha_file(ROOT/'PRELAUNCH_INPUT_LOCK.json')})
    headroom=read(ROOT/'initial/region_train_probe/COMPLETE.json')
    if not headroom['headroom_passed']:
        write_json(ROOT/'COMPLETION.json',{'status':'stopped_no_observed_corrective_signal','headroom':headroom,'training_updates':0});return
    identities={read(ROOT/'initial'/task/'MODEL_IDENTITY.json')['lora_sha256'] for task in ('text','human')}
    if len(identities)!=1:raise ValueError('Baseline checkpoints differ')
    freeze(ROOT/'STUDY_INPUT_LOCK.json',{'prelaunch_sha256':sha_file(ROOT/'PRELAUNCH_INPUT_LOCK.json'),
        'initial_lora_sha256':identities.pop(),'headroom_sha256':sha_file(ROOT/'initial/region_train_probe/COMPLETE.json')})
    run_queue(ROOT/'training_queue',[{'key':role,'command':[sys.executable,'-u','-m',__package__+'.study','--role',role],
        'completion_marker':str(ROOT/'runs'/role/'COMPLETE.json')} for role in ROLES],
        {'study_sha256':sha_file(ROOT/'STUDY_INPUT_LOCK.json')})
    screen=analyze('screen')
    if not screen['passed']:
        write_json(ROOT/'COMPLETION.json',{**screen,'next':'stop, no automatic more training or CV claim'});return
    prepare_confirmation()
    run_queue(ROOT/'confirmation_queue',[{'key':role,'command':[sys.executable,'-u','-m',__package__+'.study','--confirm',role],
        'completion_marker':str(ROOT/'confirmation'/role/'COMPLETE.json')} for role in ('initial',)+ROLES],
        {'screen_sha256':sha_file(ROOT/'SCREEN_ANALYSIS.json')})
    result=analyze('confirm');write_json(ROOT/'COMPLETION.json',{**result,'next':'no automatic more training; multi-seed evidence still required'})


if __name__=='__main__':main()
