from vlm_deep.common import write_json
from vlm_grpo_iteration_v1.study import paired
from .common import ROOT,PARENT,RESULT,ROLES,read


def analyze():
    records={}
    for role in ('initial','plain','hard_replay')+ROLES:
        base=ROOT if role in ROLES else PARENT
        records[role]={n:[q for q in read(base/'runs'/role/n/'PREDICTIONS.json') if q['evaluation_scored']] for n in ('training_context','seen_validation','retention','transfer_dev')}
    percent=lambda rows:100*sum(q['vqa_score'] for q in rows)/len(rows)
    scores={r:{n:percent(rows) for n,rows in groups.items()} for r,groups in records.items()}
    comparisons={};slices={};gates={};changes={}
    for cohort in ('seen_validation','transfer_dev'):
        for ref in ('initial','plain','hard_replay','repeat512'):
            key=cohort+':'+ref;old=records[ref][cohort];new=records['unique1024'][cohort];value=paired(old,new)
            comparisons[key]=value;gates[key+':overall']=value['delta_pp']>=.5 and value['ci95'][0]>0
            lookup={q['question_id']:q for q in old}
            changes[key]=[{'question_id':q['question_id'],'reference':lookup[q['question_id']]['prediction'],'candidate':q['prediction'],
                           'reference_score':lookup[q['question_id']]['vqa_score'],'candidate_score':q['vqa_score']} for q in new if q['vqa_score']!=lookup[q['question_id']]['vqa_score']]
            for field,labels in [('stratum',('singleword','multiword')),('frequency_slice',sorted({q['frequency_slice'] for q in old}))]:
                for label in labels:
                    v=paired([q for q in old if q[field]==label],[q for q in new if q[field]==label]);slices[key+':'+label]=v
                    gates[key+':'+label]=v['delta_pp']>=(.5 if ref=='initial' and label=='rare_1_5' else -.5)
    for ref in ('initial','plain','hard_replay','repeat512'):gates[ref+':retention']=scores['unique1024']['retention']-scores[ref]['retention']>=-.5
    full={r:sum(q['vqa_score']>=.999999 for q in groups['training_context']) for r,groups in records.items()}
    gates['training_net32']=full['unique1024']-full['initial']>=32
    protected={q['question_id'] for q in records['initial']['seen_validation'] if q['stratum']=='multiword' and q['vqa_score']>=.999999}
    keep=paired([q for q in records['initial']['seen_validation'] if q['question_id'] in protected],[q for q in records['unique1024']['seen_validation'] if q['question_id'] in protected])
    gates['seen_initial_full_multiword_retention']=keep['delta_pp']>=-.5
    result={'status':'completed','passed':all(gates.values()),'goal_complete':False,'scores_percent':scores,'training_full':full,
            'comparisons':comparisons,'slices':slices,'gates':gates,'all_score_changes':changes,'previously_full_multiword':keep,
            'admission':read(ROOT/'ADMISSION.json'),'scope':'Fixed single-seed internal exploration; not blind or seed-stable.',
            'next':'No automatic steps/seeds/formal benchmark/CV change.'}
    write_json(ROOT/'ANALYSIS.json',result)
    table='|组|训练128满分|seen512%|SFT未用图像1617%|保持64%|\n|---|---:|---:|---:|---:|\n'
    for role,v in scores.items():table+=f'|{role}|{full[role]}|{v["seen_validation"]:.6f}|{v["transfer_dev"]:.6f}|{v["retention"]:.6f}|\n'
    RESULT.write_text('# 回放多样性完整结果\n\n'+table+'\n联合门槛='+str(result['passed'])+'。两臂同1536曝光/64步/源权重，但不同独立样本与token成本，不能称严格等FLOPs。全部逐题升降见outputs/vlm_replay_diversity_v1/ANALYSIS.json。\n\n[协议](VLM_REPLAY_DIVERSITY_V1_PROTOCOL.md) · [原图与原理](learning/回放多样性_重复512与覆盖1024.html)\n')
    return result
