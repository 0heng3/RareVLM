"""One matched pure-SFT intervention: raw vs metric-canonical word targets."""
import argparse
from collections import Counter
import hashlib
import html
import math
import os
from pathlib import Path
import random
import sys
import time
from urllib.parse import quote

from rarevlm_paths import PROJECT_ROOT, SOURCE_ROOT
from vlm_deep.common import content_sha, sha_file, write_json
from vlm_grpo_iteration_v1.common import actor, chosen_sample, read
from run_literal_tile_v1 import canonical, generate

BASE = PROJECT_ROOT.parents[1]/'outputs'
ROOT = BASE/'vlm_literal_case_sft_v1'
OLD = BASE/'vlm_literal_preference_v1'
PROTOCOL = PROJECT_ROOT/'docs/VLM_LITERAL_CASE_SFT_V1_PROTOCOL.md'
RESULT = PROJECT_ROOT/'docs/VLM_LITERAL_CASE_SFT_V1_RESULTS.md'
PAGE = PROJECT_ROOT/'docs/learning/大小写捷径_统一监督与SFT可学性.html'
ROLES = ('raw_case', 'canonical_case')


def audit_data():
    from PIL import Image, ImageOps, ImageDraw, ImageFont
    if (ROOT/'AUDIT_CANDIDATES.json').exists():
        raise FileExistsError('Blind audit candidates already frozen')
    source = read(BASE/'vlm_literal_roi_v1/CANDIDATE_ROWS.json')
    previous = read(BASE/'vlm_literal_roi_v1/DATA.json')
    used = {q['source_image_id'] for q in previous['rows']}
    rows = [q for q in source['rows'] if q['source_image_id'] not in used and q['sft_answer_word_frequency']==0
            and len(q['target'])>=5 and q['crop_bounds_original_pixels'][3]-q['crop_bounds_original_pixels'][1]>=24
            and q['crop_bounds_original_pixels'][2]-q['crop_bounds_original_pixels'][0]>=60]
    rows.sort(key=lambda q:content_sha(['case_sft_v1:blind_audit',q['source_image_id'],q['annotation_id']]))
    if len(rows)<192:
        raise ValueError('Insufficient fixed new audit candidates')
    records=[]
    for index,q in enumerate(rows[:192]):
        im=source['source_images'][q['source_image_id']]
        if sha_file(im['path'])!=im['sha256']:
            raise ValueError('Original source image changed')
        path=ROOT/'audit/images'/(q['annotation_id']+'.png');path.parent.mkdir(parents=True,exist_ok=True)
        with Image.open(im['path']) as original:
            image=original.convert('RGB').crop(q['crop_bounds_original_pixels']);original_size=list(image.size)
            if min(image.size)<28:
                scale=28/min(image.size);image=image.resize((math.ceil(image.width*scale),math.ceil(image.height*scale)),Image.Resampling.BICUBIC)
            image.save(path)
        records.append({**q,'ordinal':index,'question_id':'roi:'+q['annotation_id'],'image_id':'roi:'+q['source_image_id'],
                        'source_image_path':im['path'],'source_image_sha256':im['sha256'],
                        'source_pixel_sha256':im.get('pixel_sha256',im['sha256']),
                        'image':{'path':str(path),'sha256':sha_file(path),'pixel_sha256':hashlib.sha256(image.tobytes()).hexdigest(),
                                 'width':image.width,'height':image.height},'original_crop_size':original_size,
                        'factor28_compatible_size':list(image.size)})
    try:font=ImageFont.load_default(size=17)
    except TypeError:font=ImageFont.load_default()
    boards=[]
    for number in range(12):
        canvas=Image.new('RGB',(1040,960),'#eef2f6');draw=ImageDraw.Draw(canvas)
        for j,q in enumerate(records[number*16:number*16+16]):
            x=(j%2)*520;y=(j//2)*120;draw.text((x+8,y+4),f'#{q["ordinal"]} official: {q["target"]}',font=font,fill='black')
            image=ImageOps.contain(Image.open(q['image']['path']).convert('RGB'),(500,80),Image.Resampling.BICUBIC)
            canvas.paste(image,(x+8+(500-image.width)//2,y+32+(80-image.height)//2))
        path=ROOT/'audit'/f'board_{number:02d}.png';canvas.save(path);boards.append(str(path))
    write_json(ROOT/'AUDIT_CANDIDATES.json',{'records':records,'boards':boards,'model_outputs_used':False,
               'scope':'New held-word quality audit; labels/image only, no model prediction or outcome selection',
               'protocol_sha256':sha_file(PROTOCOL)})
    print('BLIND_AUDIT_READY',{'candidates':192,'boards':12},flush=True)


def training_row(q,role):
    return {**q,'target':canonical(q['target'])} if role=='canonical_case' and q['study_source']=='preference' else q


def prepare():
    import torch
    from vlm_probe.evaluate import processor_for,inference_row
    from vlm_deep.common import encode
    if (ROOT/'PLAN.json').exists():raise FileExistsError('Study frozen')
    torch.set_num_threads(2);start=time.monotonic();audit=read(ROOT/'AUDIT_CANDIDATES.json');review=read(ROOT/'AUDIT_REVIEW.json')
    if review['status']!='passed' or len(review['selected_ordinals'])!=128 or review['model_predictions_or_scores_used']:
        raise ValueError('Need128 independent image-only reviewed validation words')
    data=read(OLD/'data/MANIFEST.json');parent=read(BASE/'vlm_reference_replay_v1/PLAN.json')
    processor=processor_for(Path(parent['model_root']),read(parent['reference_config']),1024)
    clean=[]
    for index in review['selected_ordinals']:
        q=dict(audit['records'][index]);im=q.pop('image');q.update(question='Transcribe the complete word in the image. Return only that word.',fold='clean_validation')
        if sha_file(im['path'])!=im['sha256']:raise ValueError('New audit image changed')
        data['images'][q['image_id']]=im
        item=encode(processor,inference_row(q),Path(im['path']),False)
        q.update(expected_prefix_tokens=int(item['input_ids'].shape[1]),expected_grid=item['image_grid_thw'][0].tolist(),
                 inference_input_sha256=content_sha(item['input_ids'].tolist()));clean.append(q)
    data['evaluations']['clean_literal']=clean
    current=[q for q in data['rows'] if q['study_source']=='preference']
    old_validation=data['evaluations']['literal_validation']
    for key in ('source_image_id','source_pixel_sha256'):
        if {q[key] for q in clean}&{q[key] for q in current+old_validation}:raise ValueError('Source image/pixel overlap')
    if {canonical(q['target']) for q in clean}&{canonical(q['target']) for q in current+old_validation}:raise ValueError('Word overlap')
    cpu={};counts={}
    for role in ROLES:
        cpu[role]={};counts[role]={'common':0,'preference':0}
        for q in data['rows']:
            item=chosen_sample(processor,training_row(q,role),data['images'])
            if item['input_ids'].shape[1]>2048:raise ValueError('Training input exceeds2048')
            ids=item['labels'][item['labels'].ne(-100)].tolist();counts[role][q['study_source']]+=len(ids)
            cpu[role][q['question_id']]={k:content_sha(item[k].tolist()) for k in ('input_ids','labels')}
            cpu[role][q['question_id']]['assistant_ids']=ids
    old_cpu=read(OLD/'data/CPU_INPUTS.json')
    for q in data['rows']:
        for k in ('input_ids','labels'):
            if cpu['raw_case'][q['question_id']][k]!=old_cpu[q['question_id']+':chosen'][k+'_sha256']:
                raise ValueError('Raw chosen input differs from established source')
        if q['study_source']=='common' and cpu['raw_case'][q['question_id']]!=cpu['canonical_case'][q['question_id']]:
            raise ValueError('Replay labels must be unchanged')
    schedule={'updates':read(OLD/'data/SCHEDULE.json')['updates']*4}
    exposures=Counter(qid for u in schedule['updates'] for micro in u['micros'] for qid in micro)
    if any(exposures[q['question_id']]!=(8 if q['study_source']=='preference' else 4) for q in data['rows']):raise ValueError('Eight/four exposures differ')
    write_json(ROOT/'DATA.json',data);write_json(ROOT/'CPU_INPUTS.json',cpu);write_json(ROOT/'SCHEDULE.json',schedule)
    for name in ('hard','literal_validation','seen_validation','retention'):
        write_json(ROOT/'baseline'/name/'PREDICTIONS.json',read(OLD/'baseline'/name/'PREDICTIONS.json'))
    source=read(OLD/'STUDY_INPUT_LOCK.json');files={**source['files']}
    for path in (Path(__file__),PROTOCOL,ROOT/'AUDIT_CANDIDATES.json',ROOT/'AUDIT_REVIEW.json',ROOT/'DATA.json',ROOT/'CPU_INPUTS.json',ROOT/'SCHEDULE.json'):
        files[str(path)]=sha_file(path)
    images={data['images'][q['image_id']]['path']:data['images'][q['image_id']]['sha256'] for q in clean}
    plan={'status':'prepared','files':files,'new_image_files':images,'packages':source['packages'],'base_weight_files':source['base_weight_files'],
          'initial_lora_sha256':source['initial_lora_sha256'],'updates':128,'warmup_updates':8,'lr':2e-4,'roles':ROLES,
          'new_initial_generations':128,'generation_per_role':1216,'canonical_training_only':True,'new_teacher_sequences':0,
          'assistant_tokens_per_unique_training_catalog':counts,'active_prepare_seconds':time.monotonic()-start,
          'preview_questions':[q['question_id'] for q in data['evaluations']['hard'][:8]]}
    write_json(ROOT/'PLAN.json',plan);RESULT.write_text('# 大小写统一SFT\n\n已准备同128词/256回放、两臂各128步，原始target与评测不改；尚无新结果。\n');render();print('PREPARED',{'updates':128,'unique_training':384,'clean_validation':128,'tokens':counts,'active_seconds':plan['active_prepare_seconds']},flush=True)


def verify():
    from vlm_object_train.train import package_lock
    plan=read(ROOT/'PLAN.json')
    for path,digest in {**plan['files'],**plan['new_image_files']}.items():
        if sha_file(path)!=digest:raise ValueError('Frozen file changed: '+path)
    if package_lock()!=plan['packages']:raise ValueError('Package versions changed')
    for path,expected in plan['base_weight_files'].items():
        st=Path(path).stat()
        if expected!={'bytes':st.st_size,'mtime_ns':st.st_mtime_ns}:raise ValueError('Base storage changed')
    return plan


def evaluate(model,processor,directory,names):
    import torch
    from vlm_fit_positive_control_v1.study import generated
    from vlm_human_sft.evaluate import lora_hash
    from vlm_object_train.train import lock_json
    data=read(ROOT/'DATA.json');model.eval();model.config.use_cache=True
    lock_json(directory/'INPUT_LOCK.json',{'model_lora_sha256':lora_hash(model),'plan_sha256':sha_file(ROOT/'PLAN.json')})
    started=time.monotonic()
    for name in names:
        rows=data['evaluations'][name];path=directory/name/'PREDICTIONS.json';preds=read(path) if path.exists() else []
        if (len(preds)>len(rows) or (len(preds)%4 and len(preds)!=len(rows)) or [q['question_id'] for q in preds]!=[q['question_id'] for q in rows[:len(preds)]]):raise ValueError('Whole-batch evaluation resume changed')
        for begin in range(len(preds),len(rows),4):
            chunk=rows[begin:begin+4]
            answers=generate(model,processor,chunk,data) if name in ('hard','clean_literal','literal_validation') else generated(model,processor,chunk,data)
            preds.extend(answers);tmp=path.with_suffix('.tmp');write_json(tmp,preds);tmp.replace(path)
            write_json(directory/'EVAL_PROGRESS.json',{'status':'evaluating','cohort':name,'completed_qa':len(preds),'expected_qa':len(rows),'active_seconds':time.monotonic()-started})
            if time.monotonic()-started>1500:raise TimeoutError('Evaluation cap1500sec')
            if torch.cuda.max_memory_allocated()/2**30>65:raise RuntimeError('Evaluation memory cap65GiB')


def baseline():
    from vlm_human_sft.evaluate import lora_hash
    plan=verify();model,processor,_=actor()
    if lora_hash(model)!=plan['initial_lora_sha256']:raise ValueError('Wrong initial model')
    evaluate(model,processor,ROOT/'baseline',('clean_literal',))
    if lora_hash(model)!=plan['initial_lora_sha256']:raise ValueError('Read-only baseline changed')
    write_json(ROOT/'baseline/COMPLETE.json',{'status':'completed','new_generations':128,'optimizer_updates':0})


def train(role):
    import torch
    import numpy as np
    from vlm_deep.common import collate
    from vlm_sft.common import to_device
    from vlm_sampled_signal_v1.trajectory import sequence_logps
    from vlm_human_sft.evaluate import lora_hash
    from vlm_object_train.train import learning_rate,lock_json,save_checkpoint,resume_checkpoint
    plan=verify();data=read(ROOT/'DATA.json');cpu=read(ROOT/'CPU_INPUTS.json')[role];schedule=read(ROOT/'SCHEDULE.json')['updates'];rows={q['question_id']:q for q in data['rows']}
    directory=ROOT/'runs'/role/'train';directory.mkdir(parents=True,exist_ok=True);closed=directory/'TRAIN_COMPLETE.json'
    if closed.exists():
        result=read(closed);model,processor,_=actor(adapter=result['adapter']);return model,processor,result
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    model,processor,binding=actor(training=True)
    if lora_hash(model)!=plan['initial_lora_sha256']:raise ValueError('Fresh original SFT required')
    parameters=[p for p in model.parameters() if p.requires_grad]
    if sum(p.numel() for p in parameters)!=1843200 or any(p.dtype!=torch.float32 for p in parameters):raise ValueError('Wrong FP32 q/v trainables')
    lineage={'role':role,'plan_sha256':sha_file(ROOT/'PLAN.json'),'schedule_sha256':sha_file(ROOT/'SCHEDULE.json'),'gpu_uuid':os.environ['CUDA_VISIBLE_DEVICES']}
    lock_json(directory/'INPUT_LOCK.json',lineage);lineage_sha=content_sha(lineage);optimizer=torch.optim.AdamW(parameters,lr=2e-4,weight_decay=0.)
    stats={'active_seconds':0.,'trace_bytes':0,'sources':{'common':0,'preference':0},'assistant_tokens':0,'per_question':{}};completed=0
    if (directory/'LATEST.json').exists():completed,stats=resume_checkpoint(directory,model,optimizer,lineage_sha,lineage['schedule_sha256'],128)
    tracepath=directory/'INPUT_TRACES.jsonl';tracepath.touch(exist_ok=True);prior=stats['active_seconds'];started=time.monotonic();durations=[]
    torch.cuda.reset_peak_memory_stats()
    with tracepath.open('r+b') as trace:
        if tracepath.stat().st_size<stats['trace_bytes']:raise ValueError('Saved trace truncated')
        trace.truncate(stats['trace_bytes']);trace.seek(0,os.SEEK_END)
        def checkpoint(step):
            stats.update(active_seconds=prior+time.monotonic()-started,trace_bytes=trace.tell());trace.flush();os.fsync(trace.fileno())
            save_checkpoint(directory,model,optimizer,step,stats,lineage_sha,lineage['schedule_sha256'])
        for step in range(completed,128):
            if prior+time.monotonic()-started>=1800:checkpoint(step);raise TimeoutError('Training cap1800sec')
            begin=time.monotonic();model.train();model.get_base_model().visual.eval();model.config.use_cache=False;optimizer.zero_grad(set_to_none=True)
            lr=learning_rate(step,128,{'lr':2e-4,'warmup_ratio':8/128})
            for group in optimizer.param_groups:group['lr']=lr
            totals={'common':0.,'preference':0.}
            for micro,ids in enumerate(schedule[step]['micros']):
                chunk=[rows[qid] for qid in ids];samples=[]
                for q in chunk:
                    item=chosen_sample(processor,training_row(q,role),data['images'])
                    if any(content_sha(item[k].tolist())!=cpu[q['question_id']][k] for k in ('input_ids','labels')):raise ValueError('Actual training labels/input differ')
                    samples.append(item)
                batch=to_device(collate(samples,processor.tokenizer.pad_token_id),next(model.parameters()).device);logps,counts=sequence_logps(model,batch);ce=-logps/counts
                if not bool(torch.isfinite(ce).all()):raise FloatingPointError('Nonfinite CE')
                (ce.sum()/16).backward();values=ce.detach().cpu().tolist()
                for q,value,count in zip(chunk,values,counts.tolist()):
                    source=q['study_source'];totals[source]+=value;stats['sources'][source]+=1;stats['assistant_tokens']+=int(count)
                    stats['per_question'][q['question_id']]=stats['per_question'].get(q['question_id'],0)+1
                    trace.write((__import__('json').dumps({'update':step+1,'micro':micro,'question_id':q['question_id'],'source':source,
                                'official_target':q['target'],'training_target':training_row(q,role)['target'],'chosen_ce':value,
                                'assistant_tokens':int(count),'input_sha256':cpu[q['question_id']]['input_ids'],'labels_sha256':cpu[q['question_id']]['labels']})+'\n').encode())
                del batch,logps,counts,ce,samples
            norm=float(torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True))
            if norm<=0:raise ValueError('No learning gradient')
            optimizer.step();torch.cuda.synchronize();elapsed=time.monotonic()-begin;durations.append(elapsed)
            stats['active_seconds']=prior+time.monotonic()-started;peak=torch.cuda.max_memory_allocated()/2**30
            if step+1==32:
                evaluate(model,processor,ROOT/'runs'/role/'probes/step32',('hard',))
            if step==0 or (step+1)%8==0:checkpoint(step+1)
            progress={'status':'training','role':role,'completed_updates':step+1,'planned_updates':128,'mean_common_ce':totals['common']/8,
                      'mean_hard_ce':totals['preference']/8,'gradient_norm':norm,'lr':lr,'recent_seconds_per_update':sum(durations[-8:])/len(durations[-8:]),
                      'active_seconds':prior+time.monotonic()-started,'peak_allocated_gib':peak,'sources':stats['sources']}
            write_json(directory/'PROGRESS.json',progress)
            if step==0:write_json(directory/'FIRST_UPDATE.json',{**progress,'all_actual_cpu_labels_matched':True,'canonical_training_only':role=='canonical_case'})
            if peak>65:checkpoint(step+1);raise RuntimeError('Memory cap65GiB')
    if stats['sources']!={'common':1024,'preference':1024}:raise ValueError('Source exposures differ')
    if any(stats['per_question'][q['question_id']]!=(8 if q['study_source']=='preference' else 4) for q in data['rows']):raise ValueError('Per-question exposures differ')
    destination=directory/'adapter';model.save_pretrained(destination,safe_serialization=True)
    result={'status':'completed','role':role,'updates':128,'adapter':str(destination),'final_lora_sha256':lora_hash(model),
            'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,**stats};write_json(closed,result);write_json(directory/'PROGRESS.json',result);return model,processor,result


def condition(role,transfer=False):
    from vlm_human_sft.evaluate import lora_hash
    verify()
    if transfer:
        result=read(ROOT/'runs'/role/'train/TRAIN_COMPLETE.json');model,processor,_=actor(adapter=result['adapter']);names=('transfer_dev',)
    else:
        model,processor,result=train(role);names=('hard','clean_literal','literal_validation','seen_validation','retention')
    evaluate(model,processor,ROOT/'runs'/role,names)
    if lora_hash(model)!=result['final_lora_sha256']:raise ValueError('Final weights changed during evaluation')
    write_json(ROOT/'runs'/role/('TRANSFER_COMPLETE.json' if transfer else 'COMPLETE.json'),{'status':'completed','updates':128,'final_lora_sha256':result['final_lora_sha256']})


def analyze(transfer=False):
    from vlm_grpo_iteration_v1.study import paired
    data=read(ROOT/'DATA.json');literal={'hard','clean_literal','literal_validation'};names=('hard','clean_literal','literal_validation','seen_validation','retention');records={};metrics={}
    for role in ('initial',)+ROLES:
        path=ROOT/'baseline' if role=='initial' else ROOT/'runs'/role;records[role]={n:read(path/n/'PREDICTIONS.json') for n in names};metrics[role]={}
        for name,rows in records[role].items():
            values=[float(q['literal_exact']) if name in literal else q['vqa_score'] for q in rows]
            metrics[role][name]={'qa':len(rows),'score_percent':100*sum(values)/len(values),'full':sum(x>=.999999 for x in values)}
    def pair(a,b,name):
        normalized=lambda rows:[{**q,'vqa_score':float(q['literal_exact']) if name in literal else q['vqa_score']} for q in rows]
        return paired(normalized(records[a][name]),normalized(records[b][name]))
    comparisons={};checks={};recase={};hardmeta={q['question_id']:q for q in data['evaluations']['hard']};seenmeta={q['question_id']:q for q in data['evaluations']['seen_validation']}
    for role in ROLES:
        changes={n:pair('initial',role,n) for n in names};protections={}
        for group in ('singleword','multiword','rare_1_5'):
            ids={qid for qid,q in seenmeta.items() if (q['stratum']==group if group!='rare_1_5' else q.get('answer_frequency',100)<=5)}
            protections[group]=paired([q for q in records['initial']['seen_validation'] if q['question_id'] in ids],[q for q in records[role]['seen_validation'] if q['question_id'] in ids])
        checks[role]={'fitting_learnable':metrics[role]['hard']['full']>=96,
                      'new_word_gain':changes['clean_literal']['delta_pp']>=3 and changes['clean_literal']['ci95'][0]>0,
                      'native_gain':changes['seen_validation']['delta_pp']>=.5 and changes['seen_validation']['ci95'][0]>0,
                      'old_word_protection':changes['literal_validation']['delta_pp']>=-.5,
                      'retention':changes['retention']['delta_pp']>=-.5,'native_slices':all(v['delta_pp']>=-.5 for v in protections.values())}
        comparisons[role]={'changes':changes,'native_slices':protections}
        recase[role]=sum(not q['literal_exact'] and q['prediction']!=hardmeta[q['question_id']]['rejected_target'] and canonical(q['prediction'])==canonical(hardmeta[q['question_id']]['rejected_target']) for q in records[role]['hard'])
    passed={r:all(c.values()) for r,c in checks.items()};extra={n:pair('raw_case','canonical_case',n) for n in ('hard','clean_literal','seen_validation')}
    result={'status':'completed','metrics':metrics,'comparisons':comparisons,'checks':checks,'screen_passed':passed,
            'canonical_vs_raw':extra,'case_repair_advantage':passed['canonical_case'] and all(extra[n]['delta_pp']>=.5 and extra[n]['ci95'][0]>0 for n in ('clean_literal','seen_validation')),
            'wrong_words_changed_case_only':recase,'optimizer_updates':256,'new_adapters':2,'goal_complete':False,
            'scope':'Single-seed target-canonicalization contrast under common128-step budget; new quality-audited internal given-word regions, original-SFT-seen images; not seed-stable or whole-image localization evidence.'}
    if transfer:
        before=read(BASE/'vlm_reference_replay_v1/runs/initial/transfer_dev/PREDICTIONS.json');strong=read(BASE/'vlm_reference_replay_v1/runs/hard_replay/transfer_dev/PREDICTIONS.json');finals={r:read(ROOT/'runs'/r/'transfer_dev/PREDICTIONS.json') for r in ROLES}
        result['transfer']={r:{'vs_initial':paired(before,finals[r]),'vs_strong_qv_replay':paired(strong,finals[r])} for r in ROLES};result['transfer_canonical_vs_raw']=paired(finals['raw_case'],finals['canonical_case'])
    write_json(ROOT/('ANALYSIS.json' if transfer else 'SCREEN.json'),result)
    table='|模型|128训练满分|新128词exact%|旧256词exact%|512VQA%|64保持%|\n|---|---:|---:|---:|---:|---:|\n'
    for role,m in metrics.items():table+=f'|{role}|{m["hard"]["full"]}|{m["clean_literal"]["score_percent"]:.6f}|{m["literal_validation"]["score_percent"]:.6f}|{m["seen_validation"]["score_percent"]:.6f}|{m["retention"]["score_percent"]:.6f}|\n'
    RESULT.write_text('# 大小写统一SFT：完整结果\n\n'+table+'\n联合通过='+str(passed)+'；大小写修复额外优势='+str(result['case_repair_advantage'])+'。\n\n'+result['scope']+'\n\n[协议](VLM_LITERAL_CASE_SFT_V1_PROTOCOL.md) · [图解与代码](learning/大小写捷径_统一监督与SFT可学性.html)\n');render(result);return result


def render(result=None):
    data=read(ROOT/'DATA.json');cpu=read(ROOT/'CPU_INPUTS.json');rel=lambda p:quote(os.path.relpath(p,PAGE.parent),safe='/._-');cards=[]
    for q in data['evaluations']['hard'][:8]:
        output=''
        for role in ROLES:
            path=ROOT/'runs'/role/'hard/PREDICTIONS.json'
            if path.exists():
                a=next(x for x in read(path) if x['question_id']==q['question_id']);output+='<p>'+role+' Final：'+html.escape(a['prediction'])+' / '+str(a['literal_exact'])+'</p>'
        cards.append('<article><h2>'+html.escape(q['target'])+'</h2><img src="'+rel(data['images'][q['image_id']]['path'])+'" alt="实际词区域输入"><p>原标签 '+html.escape(q['target'])+' → 统一训练标签 '+html.escape(canonical(q['target']))+'；原错误 '+html.escape(q['rejected_target'])+'</p><p>raw assistant IDs '+str(cpu['raw_case'][q['question_id']]['assistant_ids'])+'；canonical IDs '+str(cpu['canonical_case'][q['question_id']]['assistant_ids'])+'</p>'+output+'</article>')
    state='准备/运行中，尚无新能力结果。' if result is None else html.escape(str(result['metrics']))
    PAGE.write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>大小写与SFT可学性</title><style>body{max-width:1100px;margin:30px auto;padding:0 18px;font:17px/1.7 system-ui}article{border:1px solid #cbd5e1;padding:16px;margin:18px 0}img{max-width:100%;width:480px}pre{white-space:pre-wrap;background:#f1f5f9;padding:16px}</style><h1>先消除格式捷径，再检查是否学会词形</h1><p>'+state+'</p><p>旧128题有112题GT/负例大小写风格不同；SFT48/DPO44题只把原错误词换大小写。例：MIRAGE的burger变BURGER，仍不是MIRAGE。新实验只用SFT，以相同128更新预算对比原始/评分一致标签；原图/提示/评分GT和native回放均不改。</p><pre>raw:       training_target = official_target\ncanonical: training_target = NFKC(official_target).casefold().strip()\n# 只对128词纠错题生效，256回放原标签不改\nL = mean_QA mean_assistant_token(-log p(target_token))\n# EOS参与，图像/用户问题/左padding不参与loss\nFinal128与step32训练诊断分开；不是按验证挑checkpoint。</pre><p>128困难题各8次、256回放各4次。标签会改变BPE长度，真实token与成本单列。8个例子在训练前固定；图片是实际given-ROI输入，网页缩放不代表模型处理分辨率。原图曾进原SFT，不称原SFT未见图或无框定位。</p><p>新128词验证从另192题只看图像/官方标签审核，不看模型预测；旧256完整保留。拟合>=96/128仅证明可学性，仍需新词验证与整图VQA/保持共同改善，大小写减少或loss下降不算成功。</p><p><a href="'+rel(PROTOCOL)+'">协议</a> · <a href="'+rel(Path(__file__).resolve())+'">实际实现</a> · <a href="'+rel(ROOT/'AUDIT_REVIEW.json')+'">验证集审核</a></p>'+''.join(cards)+'</html>')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--audit-data',action='store_true');parser.add_argument('--prepare',action='store_true');parser.add_argument('--worker');args=parser.parse_args()
    if args.audit_data:audit_data();return
    if args.prepare:prepare();return
    if args.worker:
        if args.worker=='baseline':baseline()
        elif args.worker.startswith('transfer:'):condition(args.worker.split(':')[1],True)
        elif args.worker in ROLES:condition(args.worker)
        else:raise ValueError('Unknown role')
        return
    from research_runtime.queue import run_queue
    verify();entry=Path(__file__).resolve();identity={'plan_sha256':sha_file(ROOT/'PLAN.json')}
    job=lambda key,arg,marker:{'key':key,'command':[sys.executable,'-u',str(entry),'--worker',arg],'completion_marker':str(marker)}
    run_queue(ROOT/'baseline_queue',[job('baseline','baseline',ROOT/'baseline/COMPLETE.json')],identity)
    run_queue(ROOT/'training_queue',[job(r,r,ROOT/'runs'/r/'COMPLETE.json') for r in ROLES],identity)
    result=analyze()
    if any(result['screen_passed'].values()):
        run_queue(ROOT/'transfer_queue',[job('transfer_'+r,'transfer:'+r,ROOT/'runs'/r/'TRANSFER_COMPLETE.json') for r in ROLES],identity);result=analyze(True)
    write_json(ROOT/'COMPLETION.json',result);print('CASE_SFT_COMPLETE',result,flush=True)


if __name__=='__main__':main()
