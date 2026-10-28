import argparse
from collections import defaultdict
import hashlib
import html
import math
import os
from pathlib import Path
import statistics
import sys
import time
from urllib.parse import quote
from rarevlm_paths import PROJECT_ROOT, SOURCE_ROOT
from vlm_deep.common import content_sha, sha_file, write_json
from vlm_grpo_iteration_v1.common import actor, chosen_sample, chosen_batch, read

ROOT = PROJECT_ROOT.parents[1] / 'outputs/vlm_image_condition_probe_v1'
OLD = ROOT.parent / 'vlm_length_replay_v1'
MINING = ROOT.parent / 'vlm_error_coverage_mining_v1'
ROLES = ('initial', 'expanded240')
PROTOCOL = PROJECT_ROOT / 'docs/VLM_IMAGE_CONDITION_PROBE_V1_PROTOCOL.md'
RESULT = PROJECT_ROOT / 'docs/VLM_IMAGE_CONDITION_PROBE_V1_RESULTS.md'
PAGE = PROJECT_ROOT / 'docs/learning/同网格换图诊断_答案是否依赖原图.html'


def order(salt, value):
    return hashlib.sha256((salt + ':' + value).encode()).hexdigest()


def supplied(row, condition, data):
    iid = row['image_id'] if condition == 'proper' else row['donor_image_id']
    return {**row, 'image_id': iid}


def prepare():
    import torch
    from vlm_probe.evaluate import processor_for
    torch.set_num_threads(2)
    if (ROOT / 'PLAN.json').exists():
        raise FileExistsError('Already frozen')
    if read(OLD / 'COMPLETION.json')['status'] != 'completed':
        raise ValueError('Previous study incomplete')
    previous = read(OLD / 'PLAN.json'); old = read(OLD / 'DATA.json'); mine = read(MINING / 'DATA.json')
    initial = {q['question_id']: q for q in read(OLD / 'initial/validation/PREDICTIONS.json')}
    grids = defaultdict(list)
    for q in mine['rows']:
        if q['fold'] == 'training':
            grids[tuple(q['expected_grid'])].append(q)
    eligible = lambda q: any(x['image_id'] != q['image_id'] for x in grids[tuple(q['expected_grid'])])
    hard = [{**q, 'probe_cohort': 'hard_training'} for q in old['hard_rows']]
    if len(hard) != 128 or not all(eligible(q) for q in hard):
        raise ValueError('Need128 eligible training questions')
    validation = []
    for stratum in ('singleword', 'multiword'):
        available = [q for q in old['evaluations']['validation'] if q['stratum'] == stratum and eligible(q)]
        errors = [q for q in available if initial[q['question_id']]['vqa_score'] < .999999]
        correct = sorted([q for q in available if initial[q['question_id']]['vqa_score'] >= .999999], key=lambda q: order('image_condition_v1:valfill', q['question_id']))
        if len(errors) > 64 or len(available) < 64:
            raise ValueError('Declared128 balanced diagnosis unavailable')
        validation += [{**q, 'probe_cohort': 'validation_diagnostic', 'mined_initial_score': initial[q['question_id']]['vqa_score']} for q in errors + correct[:64-len(errors)]]
    if sum(q['mined_initial_score'] < .999999 for q in validation) != 38:
        raise ValueError('All38 initial nonfull validation questions must be retained')
    rows = sorted(hard, key=lambda q: order('image_condition_v1:batch', q['question_id'])) + sorted(validation, key=lambda q: order('image_condition_v1:batch', q['question_id']))
    for q in rows:
        pool = [x for x in grids[tuple(q['expected_grid'])] if x['image_id'] != q['image_id']]
        donor = min(pool, key=lambda x: order('image_condition_v1:donor:' + q['question_id'], x['image_id']))
        q['donor_image_id'] = donor['image_id']; q['donor_question_id'] = donor['question_id']
        # Descriptive only; this is never used to select or reject a donor.
        q['donor_same_normalized_target'] = ' '.join(donor['target'].lower().split()) == ' '.join(q['target'].lower().split())
    merged = {**mine['images'], **old['images']}; used = {q[k] for q in rows for k in ('image_id', 'donor_image_id')}
    images = {iid: merged[iid] for iid in used}
    if len({images[q['image_id']].get('pixel_sha256', images[q['image_id']]['sha256']) for q in rows}) != 256:
        raise ValueError('Recipients must be unique image clusters')
    page_ids = []
    for cohort in ('hard_training', 'validation_diagnostic'):
        page_ids += [q['question_id'] for q in sorted([q for q in rows if q['probe_cohort'] == cohort], key=lambda q: order('image_condition_v1:page', q['question_id']))[:8]]
    data = {'rows': rows, 'images': images, 'page_ids': page_ids}
    write_json(ROOT / 'DATA.json', data)
    processor = processor_for(Path(previous['model_root']), read(previous['reference_config']), 1024)
    existing = read(OLD / 'CPU_INPUTS.json'); cpu = {}; encoded = 0; start = time.monotonic()
    for i, q in enumerate(rows):
        if q['probe_cohort'] == 'hard_training':
            expected = existing[q['question_id']]
        else:
            item = chosen_sample(processor, q, images); encoded += 1
            expected = {'input_ids': content_sha(item['input_ids'].tolist()), 'labels': content_sha(item['labels'].tolist())}
        item = chosen_sample(processor, supplied(q, 'substituted', data), images); encoded += 1
        if content_sha(item['input_ids'].tolist()) != expected['input_ids'] or content_sha(item['labels'].tolist()) != expected['labels']:
            raise ValueError('Donor changed prompt/answer token identity')
        if item['image_grid_thw'][0].tolist() != q['expected_grid'] or item['pixel_values'].shape[0] != math.prod(q['expected_grid']):
            raise ValueError('Donor changed image grid/tensor frame')
        proper, donor = images[q['image_id']], images[q['donor_image_id']]
        if proper.get('pixel_sha256', proper['sha256']) == donor.get('pixel_sha256', donor['sha256']):
            raise ValueError('Substitution did not change image content')
        cpu[q['question_id']] = {**expected, 'expected_grid': q['expected_grid'], 'patch_feature_dimension': item['pixel_values'].shape[1]}
        if (i+1) % 32 == 0:
            write_json(ROOT / 'CPU_PROGRESS.json', {'completed': i+1, 'expected': 256, 'new_encodes': encoded, 'seconds': time.monotonic()-start})
    if encoded != 384:
        raise ValueError('Expected128 reused and384 new CPU encodes')
    write_json(ROOT / 'CPU_INPUTS.json', cpu)
    final = read(OLD / 'runs/expanded240/train/TRAIN_COMPLETE.json')
    models = {'initial': {'adapter': previous['adapter'], 'lora_sha256': previous['lora_sha256']},
              'expanded240': {'adapter': final['adapter'], 'lora_sha256': final['final_lora_sha256']}}
    receipt = {'status': 'passed', 'recipient_questions': 256, 'hard_training': 128, 'validation_diagnostic': 128,
               'old_initial_nonfull_retained': 38, 'new_cpu_encodes': encoded, 'reused_cpu_encodes': 128,
               'same_target_donor_coincidences': sum(q['donor_same_normalized_target'] for q in rows),
               'same_token_ids_and_labels': True, 'same_grids': True, 'zero_optimizer_updates': True,
               'cpu_seconds': time.monotonic()-start}
    write_json(ROOT / 'CPU_RECEIPT.json', receipt)
    paths = [PROTOCOL, Path(__file__), Path(__file__).parent/'__init__.py', PROJECT_ROOT/'tools/run_image_condition_probe_v1.py',
             ROOT/'DATA.json', ROOT/'CPU_INPUTS.json', ROOT/'CPU_RECEIPT.json', OLD/'PLAN.json', OLD/'COMPLETION.json',
             OLD/'POSTRUN_TRANSFER_REVIEW.json', OLD/'runs/expanded240/train/TRAIN_COMPLETE.json',
             SOURCE_ROOT/'vlm_fit_positive_control_v1/study.py']
    for cfg in models.values():
        paths += [Path(cfg['adapter'])/name for name in ('adapter_config.json', 'adapter_model.safetensors')]
    plan = {k: previous[k] for k in ('packages', 'base_weight_files')}
    plan.update(status='prepared', models=models, roles=list(ROLES), generation_qa_per_role=512, teacher_qa_per_role=512,
                optimizer_updates=0, files={**previous['files'], **{str(path): sha_file(path) for path in paths}}, images={v['path']:v['sha256'] for v in images.values()})
    write_json(ROOT/'PLAN.json', plan)
    RESULT.write_text('# 图像条件探针准备完成\n\n256recipient/两条件/两个固定模型，0更新；384新CPU编码+128复用已通过，问题/答案token与图像网格匹配。尚无图像依赖结论，不替代旧512性能门槛。\n')
    render(); print('PREPARED_IMAGE_CONDITION', receipt, flush=True)


def verify():
    from vlm_object_train.train import package_lock
    plan = read(ROOT/'PLAN.json')
    for path, digest in {**plan['files'], **plan['images']}.items():
        if sha_file(path) != digest:
            raise ValueError('Frozen input/source changed: ' + path)
    if package_lock() != plan['packages']:
        raise ValueError('Package versions changed')
    for path, value in plan['base_weight_files'].items():
        stat = Path(path).stat()
        if value != {'bytes':stat.st_size, 'mtime_ns':stat.st_mtime_ns}:
            raise ValueError('Base storage changed')
    return plan


def teacher(model, processor, rows, data, cpu):
    import torch
    from vlm_sft.common import to_device
    from vlm_sampled_signal_v1.trajectory import readout_tail_length
    from vlm_fit_positive_control_v1.study import token_statistics
    batch = to_device(chosen_batch(processor, rows, data['images'], cpu), next(model.parameters()).device)
    keep = readout_tail_length(batch['labels'])
    handle = model.get_output_embeddings().register_forward_pre_hook(lambda module,args:(args[0][:,-keep:,:],))
    try:
        output = model(**{k:v for k,v in batch.items() if k != 'labels'}, use_cache=False)
    finally:
        handle.remove()
    stats = token_statistics(output.logits, batch['labels'], processor.tokenizer.convert_tokens_to_ids('<|im_end|>'))
    if not all(math.isfinite(q['content_ce']) and math.isfinite(q['eos_ce']) for q in stats):
        raise FloatingPointError('Nonfinite diagnostic CE')
    return stats


def run_role(role):
    import torch
    from vlm_fit_positive_control_v1.study import generated
    from vlm_human_sft.evaluate import lora_hash
    from vlm_object_train.train import lock_json
    plan=verify(); data=read(ROOT/'DATA.json'); cpu=read(ROOT/'CPU_INPUTS.json'); cfg=plan['models'][role]
    directory=ROOT/'runs'/role; directory.mkdir(parents=True,exist_ok=True)
    binding={'plan_sha256':sha_file(ROOT/'PLAN.json'),'role':role,'gpu_uuid':os.environ['CUDA_VISIBLE_DEVICES'],'lora_sha256':cfg['lora_sha256']}
    lock_json(directory/'INPUT_LOCK.json',binding)
    model,processor,_=actor(adapter=cfg['adapter'])
    if lora_hash(model)!=cfg['lora_sha256']:
        raise ValueError('Loaded policy differs')
    model.eval();model.config.use_cache=True;torch.cuda.reset_peak_memory_stats()
    previous=read(directory/'PROGRESS.json').get('active_seconds',0.) if (directory/'PROGRESS.json').exists() else 0.
    start=time.monotonic()
    with torch.inference_mode():
        for index in range(0,256,4):
            chunk=data['rows'][index:index+4];path=directory/'batches'/f'{index//4:04d}.json'
            if path.exists():
                stored=read(path)
                if stored['binding']!=binding or stored['question_ids']!=[q['question_id'] for q in chunk]:
                    raise ValueError('Saved batch lineage differs')
                continue
            if previous+time.monotonic()-start>900:
                raise TimeoutError('900sec diagnostic role budget')
            records=[]
            for condition in ('proper','substituted'):
                actual=[supplied(q,condition,data) for q in chunk]
                stats=teacher(model,processor,actual,data,cpu)
                predictions=generated(model,processor,actual,data)
                for q,pred,values in zip(chunk,predictions,stats):
                    records.append({**pred,'recipient_image_id':q['image_id'],'supplied_image_id':pred['image_id'],
                                    'probe_cohort':q['probe_cohort'],'condition':condition,'teacher':values})
            if any(records[i]['input_sha256']!=records[i+4]['input_sha256'] for i in range(4)):
                raise ValueError('Actual generation prompt changed across image conditions')
            peak=torch.cuda.max_memory_allocated()/2**30
            if peak>65:
                raise RuntimeError('65GiB diagnostic allocation budget exceeded')
            temporary=path.with_suffix('.tmp')
            write_json(temporary,{'binding':binding,'question_ids':[q['question_id'] for q in chunk],'records':records});temporary.replace(path)
            write_json(directory/'PROGRESS.json',{'status':'probing','completed_recipients':index+4,'expected_recipients':256,
                       'saved_generations':2*(index+4),'saved_teacher_qa':2*(index+4),'optimizer_updates':0,
                       'active_seconds':previous+time.monotonic()-start,'peak_allocated_gib':peak})
    if lora_hash(model)!=cfg['lora_sha256']:
        raise ValueError('Fixed policy weights changed')
    write_json(directory/'COMPLETE.json',{'status':'completed','optimizer_updates':0,'generations':512,'teacher_qa':512,
               'lora_sha256_before_after':cfg['lora_sha256'],'active_seconds':previous+time.monotonic()-start,'gpu_uuid':binding['gpu_uuid']})


def records(role):
    return [row for path in sorted((ROOT/'runs'/role/'batches').glob('*.json')) for row in read(path)['records']]


def interval(values):
    import numpy as np
    a=np.asarray(values,dtype=float);rng=np.random.default_rng(20261021)
    samples=a[rng.integers(0,len(a),size=(6000,len(a)))].mean(1)
    return {'mean_nats_per_content_token':float(a.mean()),'ci95':np.quantile(samples,[.025,.975]).tolist(),
            'recipient_image_clusters':len(a),'scope':'Fixed models;one recipient per image;conditional diagnostic interval'}


def analyze():
    all_records={role:records(role) for role in ROLES};summary={}
    lookup={role:{(q['question_id'],q['condition']):q for q in rows} for role,rows in all_records.items()}
    for cohort in ('hard_training','validation_diagnostic'):
        ids=[q['question_id'] for q in read(ROOT/'DATA.json')['rows'] if q['probe_cohort']==cohort]
        models={};gaps={}
        for role in ROLES:
            proper=[lookup[role][(qid,'proper')] for qid in ids];wrong=[lookup[role][(qid,'substituted')] for qid in ids]
            full=[i for i,q in enumerate(proper) if q['vqa_score']>=.999999]
            retained=sum(wrong[i]['vqa_score']>=.999999 for i in full)
            gaps[role]=[b['teacher']['content_ce']-a['teacher']['content_ce'] for a,b in zip(proper,wrong)]
            models[role]={'proper_full':len(full),'substituted_full':sum(q['vqa_score']>=.999999 for q in wrong),
                          'proper_vqa_percent':100*statistics.mean(q['vqa_score'] for q in proper),
                          'substituted_vqa_percent_against_recipient_target':100*statistics.mean(q['vqa_score'] for q in wrong),
                          'retained_full_after_substitution':retained,'lost_full_fraction_after_substitution':(len(full)-retained)/len(full) if full else None,
                          'proper_content_ce':statistics.mean(q['teacher']['content_ce'] for q in proper),
                          'substituted_content_ce':statistics.mean(q['teacher']['content_ce'] for q in wrong),
                          'gap':interval(gaps[role])}
        change=interval([b-a for a,b in zip(gaps['initial'],gaps['expanded240'])]);trained=models['expanded240']
        label='mixed_or_inconclusive'
        if trained['proper_full']>=32 and trained['lost_full_fraction_after_substitution']>=.75 and change['mean_nats_per_content_token']>=.1 and change['ci95'][0]>0:
            label='image_conditioned_training_corrections'
        elif trained['proper_full']>=32 and trained['retained_full_after_substitution']/trained['proper_full']>=.5:
            label='substantial_image_invariant_component'
        summary[cohort]={'models':models,'change_in_gap_final_minus_initial':change,'operational_label':label}
    result={'status':'completed','optimizer_updates':0,'summary':summary,'goal_complete':False,
            'scope':'Image substitution diagnostic;not full512/new generalization/architecture cause proof',
            'next':'Use observed image dependence and remaining transfer errors to preregister training;no automatic main/CV.'}
    write_json(ROOT/'ANALYSIS.json',result)
    table='|cohort/model|proper满分|换图满分|proper正文CE|换图正文CE|\n|---|---:|---:|---:|---:|\n'
    for cohort,value in summary.items():
        for role,v in value['models'].items():
            table+=f'|{cohort}/{role}|{v["proper_full"]}|{v["substituted_full"]}|{v["proper_content_ce"]:.6f}|{v["substituted_content_ce"]:.6f}|\n'
    RESULT.write_text('# 同网格图像条件诊断完成\n\n'+table+'\n完整G变化/区间/条件标签见ANALYSIS；诊断不替代原512负结果、不证明跨图泛化或唯一失败原因。0optimizer/1024生成/1024teacher QA。\n\n[协议](VLM_IMAGE_CONDITION_PROBE_V1_PROTOCOL.md) · [双原图与真实答案](learning/同网格换图诊断_答案是否依赖原图.html)\n')
    render(result);write_json(ROOT/'COMPLETION.json',result);return result


def render(result=None):
    data=read(ROOT/'DATA.json');chosen=set(data['page_ids']);relative=lambda path:quote(os.path.relpath(path,PAGE.parent),safe='/._-')
    saved={role:{(q['question_id'],q['condition']):q for q in records(role)} for role in ROLES};cards=[]
    for q in data['rows']:
        if q['question_id'] not in chosen:continue
        body=''
        for role in ROLES:
            for condition in ('proper','substituted'):
                row=saved[role].get((q['question_id'],condition))
                if row:
                    body+='<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in [role,condition,row['prediction'],row['vqa_score'],round(row['teacher']['content_ce'],4),round(row['teacher']['eos_ce'],4)])+'</tr>'
        cards.append(f'<article><h2>q{q["question_id"]} · {q["probe_cohort"]}</h2><div class="pair"><figure><img loading="lazy" src="{relative(data["images"][q["image_id"]]["path"])}" alt="recipient原始正确图片"><figcaption>proper：问题原图</figcaption></figure><figure><img loading="lazy" src="{relative(data["images"][q["donor_image_id"]]["path"])}" alt="按SHA选择的同网格另一原图"><figcaption>substituted：另一张原图；不保证其中没有目标</figcaption></figure></div><p>{html.escape(q["question"])}</p><p>recipient原目标：{html.escape(q["target"])}，网格{q["expected_grid"]}</p><table><tr><th>模型</th><th>图像条件</th><th>真实生成</th><th>对recipient VQA</th><th>正文CE</th><th>EOS CE</th></tr>{body}</table></article>')
    status='准备/运行中，0更新，无新模型能力结论。' if result is None else html.escape(str(result['summary']))
    PAGE.write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>同网格换图诊断</title><style>body{max-width:1150px;margin:30px auto;padding:0 18px;font:17px/1.65 system-ui;background:#f8fafc;color:#1e293b}article{background:white;padding:16px;margin:20px 0;border:1px solid #cbd5e1}.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}figure{margin:0}img{width:100%;height:340px;object-fit:contain}table{border-collapse:collapse;font-size:14px}td,th{padding:6px;border:1px solid #cbd5e1}pre{white-space:pre-wrap;background:#e2e8f0;padding:18px}.status{overflow-wrap:anywhere}</style><h1>答案是否依赖原图：同网格图像条件干预</h1><p class="status">'+status+'</p><pre>相同问题/正文与EOS token/图像网格/四题batch，仅改变图像内容。\nG = CE_content(替换图) − CE_content(原图)，单位nats/token\nG正：原答案在正确图像下更容易被模型预测。\n对比Final−Initial的G，而非只看训练loss下降。\n生成是实际greedy IDs；CE使用正确答案前缀，二者不冒充同一轨迹。\n128训练困难＋128错误富集验证（含原38未满分）；不代表总体512准确率。\n同网格另一图不自动是负例；正确图依赖也可能是图像记忆。\n0参数更新，不可能凭本探针宣称新模型提分。</pre>'+f'<p><a href="{relative(PROTOCOL)}">预登记</a> · <a href="{relative(RESULT)}">结果</a> · <a href="{relative(Path(__file__))}">实际代码</a></p>'+''.join(cards)+'</html>')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--role',choices=ROLES);args=parser.parse_args()
    if args.prepare:prepare()
    elif args.role:run_role(args.role)
    else:
        from research_runtime.queue import run_queue
        verify();run_queue(ROOT/'queue',[{'key':role,'command':[sys.executable,'-u','-m','vlm_image_condition_probe_v1.study','--role',role],
                    'completion_marker':str(ROOT/'runs'/role/'COMPLETE.json')} for role in ROLES],{'plan_sha256':sha_file(ROOT/'PLAN.json')})
        print('COMPLETED_IMAGE_DIAGNOSTIC',analyze()['summary'],flush=True)


if __name__=='__main__':main()
