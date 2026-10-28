from collections import Counter
from pathlib import Path
import math
import time
from vlm_deep.common import NORMALIZER,content_sha,encode,read_manifest,sha_file,write_json
from .common import ROOT,PARENT,MINING,PROTOCOL,ROLES,read,cell,order,chosen_sample,schedules


def prepare():
    import torch
    from rarevlm_paths import PROJECT_ROOT,SOURCE_ROOT
    from vlm_probe.evaluate import processor_for,inference_row
    from vlm_reference_replay_v1.execution_v2 import verify_execution
    if (ROOT/'PLAN.json').exists():raise FileExistsError('Already frozen')
    verify_execution();torch.set_num_threads(2)
    old=read(PARENT/'DATA.json');prior=read(PARENT/'PLAN.json');mine=read(MINING/'DATA.json')
    predictions={q['question_id']:q for q in read(MINING/'PREDICTIONS.json')}
    source_path=PARENT.parent/'vlm_deep_v2/prepared/manifest.json';source=read_manifest(source_path)
    old_replay=[q for q in old['training_rows'] if q['training_source']=='replay'];counts=Counter(cell(q) for q in old_replay)
    excluded={q['image_id'] for q in old['training_rows']}
    for rows in old['evaluations'].values():excluded.update(q['image_id'] for q in rows)
    for split in ('dev','final'):excluded.update(q['image_id'] for q in source['splits'][split])
    images={**source['images'],**old['images']}
    pixel=lambda iid:images[iid].get('pixel_sha256',images[iid]['sha256'])
    seen={pixel(i) for i in excluded if i in images};frequency=Counter(NORMALIZER(q['target']) for q in source['splits']['train_all'])
    eligible=[]
    for q in source['splits']['train_all']:
        f=frequency[NORMALIZER(q['target'])]
        if q['image_id'] in excluded or pixel(q['image_id']) in seen or Counter(q['answers'])[q['target']]<8 or not 1<=f<=20 or not 2<=len(q['target'].split())<=6:continue
        eligible.append({**q,'fold':'training','stratum':'multiword','answer_frequency':f,'frequency_slice':'rare_1_5' if f<=5 else 'freq_6_20'})
    pool=[];inventory={}
    for c,n in sorted(counts.items()):
        if c[1]!='multiword':continue
        candidates=[];local=set()
        for q in sorted([q for q in eligible if cell(q)==c],key=order):
            px=pixel(q['image_id'])
            if px in seen or px in local:continue
            local.add(px);candidates.append(q)
        cap=min(2*n,len(candidates));cap-=cap%4
        if cap<n:raise ValueError('Insufficient fixed multiword candidates: '+str((c,n,len(candidates))))
        picked=candidates[:cap];pool+=picked;seen.update(pixel(q['image_id']) for q in picked)
        inventory[str(c)]={'required':n,'eligible_unique':len(candidates),'fixed_candidates':cap}
    singles=[]
    for c,n in sorted(counts.items()):
        if c[1]!='singleword':continue
        eligible_single=[q for q in mine['rows'] if q['fold']=='training' and cell(q)==c and q['image_id'] not in excluded
                         and pixel(q['image_id']) not in seen and predictions[q['question_id']]['vqa_score']>=.999999 and predictions[q['question_id']]['terminated']]
        picked=[]
        for q in sorted(eligible_single,key=order):
            if pixel(q['image_id']) in seen:continue
            picked.append(q);seen.add(pixel(q['image_id']))
            if len(picked)==n:break
        if len(picked)!=n:raise ValueError('Insufficient reused correct singleword donors: '+str(c))
        singles+=picked
    images_used={q['image_id'] for q in pool+singles}
    resolved={i:{**source['images'][i],'path':str((source_path.parent/source['images'][i]['path']).resolve())} for i in images_used}
    processor=processor_for(Path(prior['model_root']),read(prior['reference_config']),1024)
    cpu=read(PARENT/'CPU_INPUTS.json');started=time.monotonic()
    for index,q in enumerate(pool+singles):
        if q in pool:
            prefix=encode(processor,inference_row(q),Path(resolved[q['image_id']]['path']),False)
            q.update(expected_prefix_tokens=int(prefix['input_ids'].shape[1]),expected_grid=prefix['image_grid_thw'][0].tolist(),inference_input_sha256=content_sha(prefix['input_ids'].tolist()))
        if q['question_id'] not in cpu:
            sample=chosen_sample(processor,q,resolved)
            cpu[q['question_id']]={'input_ids':content_sha(sample['input_ids'].tolist()),'labels':content_sha(sample['labels'].tolist()),'visual_tokens':int(sample['image_grid_thw'].prod(-1).sum())//4}
        if (index+1)%32==0:write_json(ROOT/'CPU_PROGRESS.json',{'status':'preparing_training_only_pool','completed':index+1,'expected':len(pool+singles),'active_seconds':time.monotonic()-started})
    write_json(ROOT/'POOL.json',{'rows':pool,'singleword_donors':singles,'images':resolved,'cell_inventory':inventory})
    write_json(ROOT/'CPU_INPUTS.json',cpu)
    files=list((SOURCE_ROOT/'vlm_replay_diversity_v1').glob('*.py'))
    files += [PROTOCOL,PROJECT_ROOT/'tools/run_replay_diversity_v1.py',PROJECT_ROOT/'tests/test_vlm_replay_diversity_v1.py',ROOT/'CPU_CHECK.json',ROOT/'POOL.json',ROOT/'CPU_INPUTS.json',source_path,
              PARENT/'PLAN.json',PARENT/'DATA.json',PARENT/'SCHEDULE.json',PARENT/'FIRST_UPDATE_REFERENCE.json',PARENT/'CALIBRATION.json',MINING/'DATA.json',MINING/'PREDICTIONS.json']
    files += list((PARENT/'runs').glob('*/*/PREDICTIONS.json'))
    plan={'status':'prepared','files':{**prior['files'],**{str(p):sha_file(p) for p in files}},'parent_plan_sha256':sha_file(PARENT/'PLAN.json'),
          'initial_adapter':prior['initial_adapter'],'initial_lora_sha256':prior['initial_lora_sha256'],
          'beta':read(PARENT/'CALIBRATION.json')['beta'],'initial_new_mining_qa':len(pool),'reused_correct_singleword':len(singles),
          'required_new_rows':512,'updates_per_role':64,'exposures_per_role':1536,'hard_exposures':512,'replay_exposures':1024,
          'roles':list(ROLES),'candidate':'unique1024','formal_updates_if_admitted':128,'new_eval_generations_if_admitted':5322,
          'automatic_more_training':False,'automatic_formal_eval':False,'images':{v['path']:v['sha256'] for v in resolved.values()},
          'train_budget_seconds':1800,'eval_budget_seconds':2400,'mining_budget_seconds':600,
          'cell_inventory':inventory,'preparation_active_seconds':time.monotonic()-started}
    write_json(ROOT/'PLAN.json',plan)
    print('PREPARED_REPLAY_DIVERSITY',{'pool':len(pool),'reused_singleword':len(singles),'cells':inventory,'cpu_seconds':time.monotonic()-started},flush=True)


def admit():
    old=read(PARENT/'DATA.json');pool=read(ROOT/'POOL.json');preds={}
    for shard in (0,1):preds.update({q['question_id']:q for q in read(ROOT/f'mining_{shard}/PREDICTIONS.json')})
    replay=[q for q in old['training_rows'] if q['training_source']=='replay'];counts=Counter(cell(q) for q in replay)
    new=list(pool['singleword_donors']);details=[];passed=True
    for c,n in sorted(counts.items()):
        if c[1]!='multiword':continue
        good=[q for q in pool['rows'] if cell(q)==c and preds[q['question_id']]['vqa_score']>=.999999 and preds[q['question_id']]['terminated']]
        details.append({'cell':list(c),'required':n,'correct_with_natural_eos':len(good)})
        if len(good)<n:passed=False
        new+=sorted(good,key=order)[:n]
    receipt={'status':'completed','training_allowed':passed,'cells':details,'new_selected':len(new),'new_optimizer_updates':0,
             'meaning':'Official >=8/10 consensus plus initial greedy full score in fixed mining grouping; not exhaustive visual relabeling.'}
    if not passed:write_json(ROOT/'ADMISSION.json',receipt);return receipt
    if len(new)!=512 or Counter(cell(q) for q in new)!=counts:raise ValueError('Donor quota mismatch')
    new=[{**q,'training_source':'replay','training_weight':1.} for q in new]
    data={**old,'training_rows':{'repeat512':old['training_rows'],'unique1024':old['training_rows']+new},'images':{**old['images'],**pool['images']}}
    schedule,mapping=schedules(read(PARENT/'SCHEDULE.json'),replay,new)
    for role,blocks in schedule.items():
        exposures=Counter(q for b in blocks for m in b['micros'] for q in m)
        if len(blocks)!=64 or any(len(b['micros'])!=6 or any(len(m)!=4 for m in b['micros']) for b in blocks):raise ValueError('Expected64x24')
        for q in data['training_rows'][role]:
            expected=4 if q['training_source']=='hard' else (2 if role=='repeat512' else 1)
            if exposures[q['question_id']]!=expected:raise ValueError('Per-question exposure mismatch')
        pixels=lambda rows:{data['images'][q['image_id']].get('pixel_sha256',data['images'][q['image_id']]['sha256']) for q in rows}
        held=set().union(*(pixels(rows) for name,rows in data['evaluations'].items() if name!='training_context'))
        if pixels(data['training_rows'][role]) & held:raise ValueError('Training/evaluation pixel overlap')
    write_json(ROOT/'DATA.json',data);write_json(ROOT/'SCHEDULE.json',schedule);write_json(ROOT/'DONOR_MAPPING.json',mapping)
    receipt['files']={str(ROOT/name):sha_file(ROOT/name) for name in ['DATA.json','SCHEDULE.json','DONOR_MAPPING.json']}
    receipt['plan_sha256']=sha_file(ROOT/'PLAN.json');write_json(ROOT/'ADMISSION.json',receipt);return receipt
