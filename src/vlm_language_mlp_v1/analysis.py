from collections import defaultdict
from vlm_deep.common import write_json
from vlm_grpo_iteration_v1.study import paired
from .common import ROOT,PARENT,DIVERSITY,RESULT,read


def analyze():
    locations={'initial':PARENT,'plain':PARENT,'hard_replay':PARENT,'unique1024':DIVERSITY,'candidate':ROOT}
    records={r:{n:[q for q in read(base/'runs'/r/n/'PREDICTIONS.json') if q['evaluation_scored']] for n in ('training_context','seen_validation','retention','transfer_dev')} for r,base in locations.items()}
    percent=lambda rows:100*sum(q['vqa_score'] for q in rows)/len(rows)
    scores={r:{n:percent(rows) for n,rows in groups.items()} for r,groups in records.items()}
    comparisons={};slices={};gates={};changes={}
    for cohort in ('seen_validation','transfer_dev'):
        for ref in ('initial','plain','hard_replay','unique1024'):
            old=records[ref][cohort];new=records['candidate'][cohort];key=cohort+':'+ref;v=paired(old,new)
            comparisons[key]=v;gates[key+':overall']=v['delta_pp']>=.5 and v['ci95'][0]>0
            lookup={q['question_id']:q for q in old}
            changes[key]=[{'question_id':q['question_id'],'reference':lookup[q['question_id']]['prediction'],'candidate':q['prediction'],
                           'reference_score':lookup[q['question_id']]['vqa_score'],'candidate_score':q['vqa_score']} for q in new if q['vqa_score']!=lookup[q['question_id']]['vqa_score']]
            for field,labels in [('stratum',('singleword','multiword')),('frequency_slice',sorted({q['frequency_slice'] for q in old}))]:
                for label in labels:
                    delta=paired([q for q in old if q[field]==label],[q for q in new if q[field]==label]);slices[key+':'+label]=delta
                    gates[key+':'+label]=delta['delta_pp']>=(.5 if ref=='initial' and label=='rare_1_5' else -.5)
    for ref in ('initial','plain','hard_replay','unique1024'):gates[ref+':retention']=scores['candidate']['retention']-scores[ref]['retention']>=-.5
    full={r:sum(q['vqa_score']>=.999999 for q in groups['training_context']) for r,groups in records.items()};gates['training_net32']=full['candidate']-full['initial']>=32
    protected={q['question_id'] for q in records['initial']['seen_validation'] if q['stratum']=='multiword' and q['vqa_score']>=.999999}
    keep=paired([q for q in records['initial']['seen_validation'] if q['question_id'] in protected],[q for q in records['candidate']['seen_validation'] if q['question_id'] in protected])
    gates['seen_initial_full_multiword_retention']=keep['delta_pp']>=-.5
    hard={q['question_id']:q for q in read(PARENT/'DATA.json')['hard_rows']};classes={}
    for role,groups in records.items():
        c=defaultdict(lambda:{'qa':0,'full':0,'score_sum':0.})
        for q in groups['training_context']:
            typ=hard[q['question_id']]['review_error_type'];c[typ]['qa']+=1;c[typ]['full']+=q['vqa_score']>=.999999;c[typ]['score_sum']+=q['vqa_score']
        classes[role]=dict(c)
    result={'status':'completed','passed':all(gates.values()),'goal_complete':False,'scores_percent':scores,'training_full':full,'training_error_classes':classes,
            'comparisons':comparisons,'slices':slices,'gates':gates,'all_score_changes':changes,'previously_full_multiword':keep,
            'scope':'One exploratory seed and historically observed internal cohorts; additional trainable capacity is not matched parameter/FLOP budget.',
            'next':'No automatic extra steps, seeds, official benchmark, or CV update.'}
    write_json(ROOT/'ANALYSIS.json',result)
    table='|组|训练128满分|seen512%|SFT图像隔离1617%|保持64%|\n|---|---:|---:|---:|---:|\n'
    for role,v in scores.items():table+=f'|{role}|{full[role]}|{v["seen_validation"]:.6f}|{v["transfer_dev"]:.6f}|{v["retention"]:.6f}|\n'
    RESULT.write_text('# 语言MLP适配完整结果\n\n'+table+'\n联合门槛='+str(result['passed'])+'。新候选4663296参数、q/v控制1843200；额外参数/计算不可忽略。训练错误分型与全部升降见outputs/vlm_language_mlp_v1/ANALYSIS.json。\n\n[预登记](VLM_LANGUAGE_MLP_V1_PROTOCOL.md) · [8张训练原图与MLP公式](learning/语言MLP适配_容量与迁移.html)\n')
    return result
