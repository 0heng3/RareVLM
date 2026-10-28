import argparse
from collections import Counter
import html
import os
from pathlib import Path
import shutil
import sys
import time
from urllib.parse import quote
from rarevlm_paths import PROJECT_ROOT, SOURCE_ROOT
from vlm_deep.common import NORMALIZER, content_sha, sha_file, write_json
from .common import ROOT, OLD, ROLES, read

PROTOCOL=PROJECT_ROOT/'docs/VLM_MERGER_TRANSFER_V1_PROTOCOL.md'
RESULT=PROJECT_ROOT/'docs/VLM_MERGER_TRANSFER_V1_RESULTS.md'
PAGE=PROJECT_ROOT/'docs/learning/视觉连接适配与跨图迁移.html'
ASSET=ROOT.parent/'vlm_deep_v2/prepared'
SFT=ROOT.parent/'vlm_weighted_sft_v1'


def bucket(value):
    return 'unseen_0' if value==0 else 'rare_1_5' if value<=5 else 'medium_6_20' if value<=20 else 'frequent_gt20'


def prepare():
    import torch
    from vlm_deep.common import encode
    from vlm_probe.evaluate import inference_row,processor_for
    torch.set_num_threads(2)
    if (ROOT/'PLAN.json').exists():raise FileExistsError('Already frozen')
    prior=read(OLD/'PLAN.json');old=read(OLD/'DATA.json');manifest=read(ASSET/'manifest.json')
    trained=read(SFT/'runs/uniform/seed42/train/TRAIN_RESULT.json')
    if trained['manifest_sha256']!=manifest['manifest_sha256'] or trained['question_count']!=32985 or trained['split']!='train_all':
        raise ValueError('Original SFT lineage/manifest differs')
    train_ids={q['image_id'] for q in manifest['splits']['train_all']};dev_ids={q['image_id'] for q in manifest['splits']['dev']}
    train_pixels={manifest['images'][iid]['pixel_sha256'] for iid in train_ids};dev_pixels={manifest['images'][iid]['pixel_sha256'] for iid in dev_ids}
    if train_ids&dev_ids or train_pixels&dev_pixels or len(dev_ids)!=1024:raise ValueError('Original SFT/dev image leakage')
    if any(q['image_id'] not in train_ids for q in old['training_rows']):raise ValueError('New training not from original train_all')
    separation={'status':'passed','train_qa':32985,'train_images':len(train_ids),'dev_qa':1617,'dev_images':len(dev_ids),
                'image_id_overlap':0,'pixel_overlap':0,'original_sft_manifest_sha256':manifest['manifest_sha256'],
                'sft_training_result_sha256':sha_file(SFT/'runs/uniform/seed42/train/TRAIN_RESULT.json'),
                'pretraining_overlap':'unknown','dev_previously_observed':True,'new_training_qa':640}
    write_json(ROOT/'SFT_IMAGE_SEPARATION.json',separation)
    frequency=Counter(NORMALIZER(q['target']) for q in manifest['splits']['train_all'])
    rows=[{**q,'stratum':'multiword' if len(q['target'].split())>1 else 'singleword','answer_frequency':frequency[NORMALIZER(q['target'])],
           'frequency_slice':bucket(frequency[NORMALIZER(q['target'])]),'evaluation_scored':True} for q in manifest['splits']['dev']]
    counts=Counter(q['frequency_slice'] for q in rows)
    if dict(counts)!={'unseen_0':687,'rare_1_5':342,'medium_6_20':171,'frequent_gt20':417}:raise ValueError('Actual full-SFT target frequency bins differ')
    frames_path=SFT/'runs/uniform/seed42/dev/predictions.jsonl';frame_lock_path=frames_path.parent/'INPUT_LOCK.json'
    frame_lock=read(frame_lock_path);frames={q['question_id']:q for q in (read_line(line) for line in frames_path.read_text().splitlines())}
    source_lock=frame_lock.get('source_sha256',{})
    reuse=(frame_lock['manifest_sha256']==manifest['manifest_sha256'] and frame_lock['reference_config_sha256']==sha_file(prior['reference_config'])
           and all(source_lock.get(name)==sha_file(SOURCE_ROOT/name) for name in ('vlm_deep/common.py','vlm_sft/common.py')))
    images={**old['images'],**{iid:{**manifest['images'][iid],'path':str(ASSET/manifest['images'][iid]['path'])} for iid in dev_ids}}
    processor=processor_for(Path(prior['model_root']),read(prior['reference_config']),1024)
    new_encodes=0
    if reuse:
        for q in rows:
            f=frames.get(q['question_id'])
            if not f or f['image_id']!=q['image_id'] or f['image_sha256']!=images[q['image_id']]['sha256'] or f['question_sha256']!=content_sha(q['question']):
                raise ValueError('Stored input frame question/image differs')
            q['expected_prefix_tokens']=f['input_tokens'];q['expected_grid']=f['image_grid_thw']
        probes=sorted(rows,key=lambda q:content_sha(['merger_transfer_frame8',q['question_id']]))[:8]
        for q in probes:
            item=encode(processor,inference_row(q),Path(images[q['image_id']]['path']),False);new_encodes+=1
            if item['input_ids'].shape[1]!=q['expected_prefix_tokens'] or item['image_grid_thw'][0].tolist()!=q['expected_grid']:
                reuse=False;break
    if not reuse:
        for q in rows:
            item=encode(processor,inference_row(q),Path(images[q['image_id']]['path']),False);new_encodes+=1
            q['expected_prefix_tokens']=item['input_ids'].shape[1];q['expected_grid']=item['image_grid_thw'][0].tolist()
    write_json(ROOT/'INPUT_FRAME_REUSE.json',{'status':'passed','reused_frame_metadata':1617 if reuse else 0,
               'new_cpu_encodes':new_encodes,'model_predictions_reused_for_transfer':0,'frequency_counts':dict(counts),
               'scope':'Historical records supply input tokens/grids only;all three transfer model predictions are fresh.'})
    evaluations={'training_context':old['evaluations']['training_context'],'seen_validation':old['evaluations']['validation'],
                 'retention':old['evaluations']['retention'],'transfer_dev':rows}
    write_json(ROOT/'DATA.json',{'training_rows':old['training_rows'],'hard_rows':old['hard_rows'],'images':images,'evaluations':evaluations})
    write_json(ROOT/'SCHEDULE.json',{'roles':{'candidate':read(OLD/'SCHEDULE.json')['roles']['coverage128']}})
    shutil.copyfile(OLD/'CPU_INPUTS.json',ROOT/'CPU_INPUTS.json')
    for role,source in [('initial',OLD/'initial'),('qv_control',OLD/'runs/coverage128')]:
        for dest,origin in [('training_context','training_context'),('seen_validation','validation'),('retention','retention')]:
            directory=ROOT/'runs'/role/dest;directory.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source/origin/'PREDICTIONS.json',directory/'PREDICTIONS.json')
    first=[read_line(line) for line in (OLD/'runs/coverage128/train/INPUT_TRACES.jsonl').read_text().splitlines()]
    first=[q for q in first if q['update']==1]
    if len(first)!=16:raise ValueError('Missing true old first-update16 reference')
    write_json(ROOT/'FIRST_UPDATE_REFERENCE.json',first)
    initializer=read(ROOT/'CPU_INITIALIZER_REVIEW.json')
    if initializer['status']!='passed' or not initializer['original_language_values_exact'] or not initializer['real_peft_save_load_exact']:
        raise ValueError('Composite CPU initialization not passed')
    control=read(OLD/'runs/coverage128/train/TRAIN_COMPLETE.json')
    models={'initial':{'adapter':prior['adapter'],'lora_sha256':prior['lora_sha256']},
            'qv_control':{'adapter':control['adapter'],'lora_sha256':control['final_lora_sha256']},
            'candidate':{'adapter':str(ROOT/'initial_adapter'),'initial_lora_sha256':initializer['lora_sha256']}}
    paths=[PROTOCOL,PROJECT_ROOT/'tools/run_merger_transfer_v1.py',ASSET/'manifest.json',SFT/'runs/uniform/seed42/train/TRAIN_RESULT.json',
           frames_path,frame_lock_path,OLD/'PLAN.json',OLD/'COMPLETION.json',OLD/'runs/coverage128/train/TRAIN_COMPLETE.json']
    paths+=list((SOURCE_ROOT/'vlm_merger_transfer_v1').glob('*.py'))
    paths+=[ROOT/name for name in ('DATA.json','SCHEDULE.json','CPU_INPUTS.json','SFT_IMAGE_SEPARATION.json','INPUT_FRAME_REUSE.json',
                                  'CPU_INITIALIZER_REVIEW.json','FIRST_UPDATE_REFERENCE.json')]
    paths+=[ROOT/'initial_adapter'/name for name in ('adapter_config.json','adapter_model.safetensors')]
    paths+=list((ROOT/'cpu_roundtrip').glob('adapter*'))
    paths+=list((ROOT/'runs/initial').glob('*/PREDICTIONS.json'))+list((ROOT/'runs/qv_control').glob('*/PREDICTIONS.json'))
    paths+=[SOURCE_ROOT/'vlm_fit_positive_control_v1/study.py']
    for cfg in models.values():paths+=[Path(cfg['adapter'])/name for name in ('adapter_config.json','adapter_model.safetensors')]
    plan={k:prior[k] for k in ('model_root','reference_config','packages','base_weight_files')}
    plan.update(status='prepared',roles=list(ROLES),models=models,lora_sha256=initializer['lora_sha256'],trainable_parameters=1982464,
                files={**prior['files'],**{str(path):sha_file(path) for path in paths}},images={v['path']:v['sha256'] for v in images.values()},
                new_generations=5895,new_training_updates=64,automatic_more_training=False,automatic_main_evaluation=False)
    write_json(ROOT/'PLAN.json',plan)
    RESULT.write_text('# 视觉连接适配与跨图迁移：已准备\n\n原SFT32985与dev1617图片/像素交集0；真实148因子/1982464参数CPU初始化与保存恢复通过。640训练输入复用、迁移输入元数据复用检查通过；迁移三模型分数均新生成。尚无新训练或提分结论。\n')
    render();print('PREPARED_MERGER_TRANSFER',{'transfer_qa':1617,'train_dev_pixels_overlap':0,'new_input_encodes':new_encodes,
           'reused_input_frames':1617 if reuse else 0,'new_generations':5895,'new_trainable':139264},flush=True)


def read_line(line):
    import json
    return json.loads(line)


def verify():
    from vlm_object_train.train import package_lock
    plan=read(ROOT/'PLAN.json')
    for path,digest in {**plan['files'],**plan['images']}.items():
        if sha_file(path)!=digest:raise ValueError('Frozen study input/source changed: '+path)
    if package_lock()!=plan['packages']:raise ValueError('Package versions changed')
    for path,value in plan['base_weight_files'].items():
        stat=Path(path).stat()
        if value!={'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns}:raise ValueError('Base storage changed')
    return plan


def evaluate(model,processor,role):
    import torch
    from vlm_fit_positive_control_v1.study import generated
    data=read(ROOT/'DATA.json');directory=ROOT/'runs'/role;model.eval();model.config.use_cache=True
    prior_seconds=read(directory/'EVAL_PROGRESS.json').get('active_seconds_total',0.) if (directory/'EVAL_PROGRESS.json').exists() else 0.
    begin=time.monotonic();torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for name,rows in data['evaluations'].items():
            path=directory/name/'PREDICTIONS.json';preds=read(path) if path.exists() else []
            if len(preds)>len(rows) or (len(preds)!=len(rows) and len(preds)%4) or [q['question_id'] for q in preds]!=[q['question_id'] for q in rows[:len(preds)]]:
                raise ValueError('Evaluation grouping/partial final batch differs')
            for start in range(len(preds),len(rows),4):
                chunk=rows[start:start+4];answers=generated(model,processor,chunk,data)
                for q,answer in zip(chunk,answers):
                    preds.append({**answer,'stratum':q['stratum'],'frequency_slice':q['frequency_slice'],
                                  'evaluation_scored':q['evaluation_scored'],'expected_grid_checked':q['expected_grid']})
                temporary=path.with_suffix('.tmp');write_json(temporary,preds);temporary.replace(path)
                peak=torch.cuda.max_memory_allocated()/2**30
                write_json(directory/'EVAL_PROGRESS.json',{'status':'evaluating','cohort':name,'completed_qa':len(preds),'expected_qa':len(rows),
                           'active_seconds_this_attempt':time.monotonic()-begin,'active_seconds_total':prior_seconds+time.monotonic()-begin,'peak_allocated_gib':peak})
                if prior_seconds+time.monotonic()-begin>2400:raise TimeoutError('2400sec role evaluation budget')
                if peak>65:raise RuntimeError('65GiB evaluation allocation budget exceeded')


def condition(role):
    from .common import actor
    from vlm_human_sft.evaluate import lora_hash
    plan=verify()
    if role=='candidate':
        from .train import train
        model,processor,result=train(role)
        if lora_hash(model)!=result['final_lora_sha256']:raise ValueError('Candidate final adapter differs')
    else:
        cfg=plan['models'][role];model,processor,_=actor(adapter=cfg['adapter'])
        if lora_hash(model)!=cfg['lora_sha256']:raise ValueError('Reference fixed adapter differs')
        result={'status':'fixed_reference','optimizer_updates':0,'lora_sha256':cfg['lora_sha256']}
    evaluate(model,processor,role)
    write_json(ROOT/'runs'/role/'COMPLETE.json',{'status':'completed','model':result})
    if role=='candidate':write_json(ROOT/'runs'/role/'train/PROGRESS.json',{'status':'completed','completed_updates':64,'role':role})


def analyze():
    from vlm_grpo_iteration_v1.study import paired
    records={role:{cohort:[q for q in read(ROOT/'runs'/role/cohort/'PREDICTIONS.json') if q['evaluation_scored']]
                  for cohort in ('training_context','seen_validation','retention','transfer_dev')} for role in ROLES}
    percent=lambda rows:100*sum(q['vqa_score'] for q in rows)/len(rows)
    scores={role:{name:percent(rows) for name,rows in groups.items()} for role,groups in records.items()}
    gates={};comparisons={};slices={}
    for cohort in ('seen_validation','transfer_dev'):
        for reference in ('initial','qv_control'):
            value=paired(records[reference][cohort],records['candidate'][cohort]);comparisons[cohort+':'+reference]=value
            gates[cohort+':'+reference+':overall']=value['delta_pp']>=.5 and value['ci95'][0]>0
            fields=[('singleword','stratum'),('multiword','stratum')]+[(x,'frequency_slice') for x in sorted({q['frequency_slice'] for q in records['initial'][cohort]})]
            for key,field in fields:
                value=paired([q for q in records[reference][cohort] if q[field]==key],[q for q in records['candidate'][cohort] if q[field]==key])
                name=cohort+':'+reference+':'+key;slices[name]=value
                gates[name]=value['delta_pp']>=(.5 if reference=='initial' and key=='rare_1_5' else -.5)
    for reference in ('initial','qv_control'):
        gates[reference+':retention']=scores['candidate']['retention']-scores[reference]['retention']>=-.5
    full={role:sum(q['vqa_score']>=.999999 for q in values['training_context']) for role,values in records.items()}
    gates['training_net32']=full['candidate']-full['initial']>=32
    protected={q['question_id'] for q in records['initial']['seen_validation'] if q['stratum']=='multiword' and q['vqa_score']>=.999999}
    keep=paired([q for q in records['initial']['seen_validation'] if q['question_id'] in protected],
                [q for q in records['candidate']['seen_validation'] if q['question_id'] in protected])
    gates['seen_initial_full_multiword_retention']=keep['delta_pp']>=-.5
    result={'status':'completed','passed':all(gates.values()),'scores_percent':scores,'comparisons':comparisons,'slices':slices,
            'gates':gates,'training_full':full,'previously_full_multiword':keep,'goal_complete':False,
            'next':'Separate preregistered confirmation only after joint pass;no automatic steps/seeds/formal/CV.'}
    write_json(ROOT/'ANALYSIS.json',result)
    table='|模型|训练128满分|原SFT已用域512 VQA%|SFT未用图像dev1617 VQA%|保持64%|\n|---|---:|---:|---:|---:|\n'
    for role,value in scores.items():table+=f'|{role}|{full[role]}|{value["seen_validation"]:.6f}|{value["transfer_dev"]:.6f}|{value["retention"]:.6f}|\n'
    RESULT.write_text('# 视觉连接适配与跨图迁移完成\n\n'+table+'\n联合门槛='+str(result['passed'])+'；配对区间/频次保护见ANALYSIS。内部已观察dev，不称预训练未见/盲测/三seed稳定。\n\n[预登记](VLM_MERGER_TRANSFER_V1_PROTOCOL.md) · [真实模块、原图与预测](learning/视觉连接适配与跨图迁移.html)\n')
    render(result);write_json(ROOT/'COMPLETION.json',result);return result


def render(result=None):
    data=read(ROOT/'DATA.json');relative=lambda path:quote(os.path.relpath(path,PAGE.parent),safe='/._-')
    chosen=sorted(data['evaluations']['transfer_dev'],key=lambda q:content_sha(['merger_transfer_page',q['question_id']]))[:12]
    saved={role:{q['question_id']:q for q in read(ROOT/'runs'/role/'transfer_dev/PREDICTIONS.json')}
           for role in ROLES if (ROOT/'runs'/role/'transfer_dev/PREDICTIONS.json').exists()};cards=[]
    for q in chosen:
        rows=''
        for role,predictions in saved.items():
            if q['question_id'] in predictions:
                answer=predictions[q['question_id']];rows+='<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in [role,answer['prediction'],answer['vqa_score']])+'</tr>'
        cards.append(f'<article><h2>dev q{q["question_id"]} · {q["frequency_slice"]}</h2><img loading="lazy" src="{relative(data["images"][q["image_id"]]["path"])}" alt="原始SFT未用图像的内部dev原图"><p>{html.escape(q["question"])}</p><p>原目标：{html.escape(q["target"])}</p><table><tr><th>模型</th><th>本轮实际生成</th><th>VQA</th></tr>{rows}</table></article>')
    status='准备/运行中，尚无本轮能力提升结论。' if result is None else html.escape(str(result['scores_percent']))+'；联合通过='+str(result['passed'])
    PAGE.write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>视觉连接适配与跨图迁移</title><style>body{max-width:1100px;margin:30px auto;padding:0 18px;font:17px/1.7 system-ui;background:#f8fafc;color:#1e293b}article{background:white;margin:18px 0;padding:18px;border:1px solid #cbd5e1}img{width:100%;max-height:450px;object-fit:contain}td,th{border:1px solid #cbd5e1;padding:7px}table{border-collapse:collapse}pre{white-space:pre-wrap;background:#e2e8f0;padding:18px}</style><h1>视觉连接模块能否帮助跨图迁移</h1><p>'+status+'</p><pre>原图 → 冻结视觉编码器 → merger两层线性 → 语言模型 → 答案\n                              ↑ 新rank8 LoRA         ↑ 原q/v LoRA\nW有效 = W冻结 + (alpha/r) B A，r=8，alpha=16\n新B=0，所以训练前新增分支输出0，原语言FP32因子逐项不变。\nq/v对照1843200参数；新候选1982464，增加139264（7.56%），非等参数。\n同128困难+512回放、64更新、lr2e−4，不改数据或采样。\n三组：初始／旧qv Final64／新qv+merger Final64。\n旧512：原SFT训练白名单；新1617：原SFT未使用图像(1024图)。\n两个范围分列，均须对初始/控制+.5pp且正95%图像簇下界。\ndev已被历史研究观察，预训练重叠未知，不称盲测。\nCPU真实架构148因子保存/FP32恢复和首步16QA/NLL/新B梯度为启动前检查。</pre>'+f'<p><a href="{relative(PROTOCOL)}">预登记</a> · <a href="{relative(RESULT)}">结果</a> · <a href="{relative(SOURCE_ROOT/"vlm_merger_transfer_v1/train.py")}">训练代码</a> · <a href="{relative(SOURCE_ROOT/"vlm_merger_transfer_v1/initialization.py")}">初始化代码</a></p>'+''.join(cards)+'</html>')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--role',choices=ROLES);args=parser.parse_args()
    if args.prepare:prepare()
    elif args.role:condition(args.role)
    else:
        from research_runtime.queue import run_queue
        verify();run_queue(ROOT/'queue',[{'key':role,'command':[sys.executable,'-u','-m','vlm_merger_transfer_v1.study','--role',role],
                 'completion_marker':str(ROOT/'runs'/role/'COMPLETE.json')} for role in ROLES],{'plan_sha256':sha_file(ROOT/'PLAN.json')})
        print('COMPLETED_MERGER_TRANSFER',analyze()['passed'],flush=True)


if __name__=='__main__':main()
