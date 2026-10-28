from collections import defaultdict
from vlm_deep.common import write_json
from vlm_grpo_iteration_v1.study import paired
from .common import ROOT,PARENT,SCOPE,JOINT,DIVERSITY,RESULT,read


def analyze():
    locations={'initial':(PARENT,'initial'),'qv_ce':(PARENT,'plain'),'qv_hard':(PARENT,'hard_replay'),
               'unique1024':(DIVERSITY,'unique1024'),'joint':(JOINT,'candidate'),'mlp_ce':(SCOPE,'candidate'),
               'mlp_hard':(ROOT,'hard_replay'),'mlp_kl':(ROOT,'soft_kl')}
    records={label:{n:[q for q in read(base/'runs'/role/n/'PREDICTIONS.json') if q['evaluation_scored']] for n in ('training_context','seen_validation','retention','transfer_dev')} for label,(base,role) in locations.items()}
    percent=lambda rows:100*sum(q['vqa_score'] for q in rows)/len(rows)
    scores={r:{n:percent(rows) for n,rows in groups.items()} for r,groups in records.items()}
    comparisons={};slices={};gates={};changes={};references=tuple(r for r in locations if r!='mlp_kl')
    for cohort in ('seen_validation','transfer_dev'):
        for ref in references:
            old=records[ref][cohort];new=records['mlp_kl'][cohort];key=cohort+':'+ref;v=paired(old,new)
            comparisons[key]=v;gates[key+':overall']=v['delta_pp']>=.5 and v['ci95'][0]>0
            lookup={q['question_id']:q for q in old}
            changes[key]=[{'question_id':q['question_id'],'reference':lookup[q['question_id']]['prediction'],'candidate':q['prediction'],
                           'reference_score':lookup[q['question_id']]['vqa_score'],'candidate_score':q['vqa_score']} for q in new if abs(q['vqa_score']-lookup[q['question_id']]['vqa_score'])>1e-8]
            for field,labels in [('stratum',('singleword','multiword')),('frequency_slice',sorted({q['frequency_slice'] for q in old}))]:
                for label in labels:
                    delta=paired([q for q in old if q[field]==label],[q for q in new if q[field]==label]);slices[key+':'+label]=delta
                    gates[key+':'+label]=delta['delta_pp']>=(.5 if ref=='initial' and label=='rare_1_5' else -.5)
    for ref in references:gates[ref+':retention']=scores['mlp_kl']['retention']-scores[ref]['retention']>=-.5
    full={r:sum(q['vqa_score']>=.999999 for q in groups['training_context']) for r,groups in records.items()};gates['training_net32']=full['mlp_kl']-full['initial']>=32
    protected={q['question_id'] for q in records['initial']['seen_validation'] if q['stratum']=='multiword' and q['vqa_score']>=.999999}
    keep=paired([q for q in records['initial']['seen_validation'] if q['question_id'] in protected],[q for q in records['mlp_kl']['seen_validation'] if q['question_id'] in protected]);gates['seen_initial_full_multiword_retention']=keep['delta_pp']>=-.5
    secondary={c+':'+r:paired(records[r][c],records['mlp_hard'][c]) for c in ('seen_validation','transfer_dev') for r in ('initial','mlp_ce','qv_hard')}
    hard={q['question_id']:q for q in read(ROOT/'DATA.json')['hard_rows']};classes={}
    for role,groups in records.items():
        counts=defaultdict(lambda:{'qa':0,'full':0,'score_sum':0.})
        for q in groups['training_context']:
            row=counts[hard[q['question_id']]['review_error_type']];row['qa']+=1;row['full']+=q['vqa_score']>=.999999;row['score_sum']+=q['vqa_score']
        classes[role]=dict(counts)
    result={'status':'completed','passed':all(gates.values()),'goal_complete':False,'scores_percent':scores,'training_full':full,'training_error_classes':classes,
            'comparisons':comparisons,'secondary_hard_comparisons':secondary,'slices':slices,'gates':gates,'all_score_changes':changes,
            'previously_full_multiword':keep,'calibration':read(ROOT/'CALIBRATION.json'),
            'scope':'Single-seed exploratory, repeatedly observed internal cohorts; KL compute differs from hard CE.',
            'next':'No automatic extra steps, seeds, formal benchmark, or CV change.'}
    write_json(ROOT/'ANALYSIS.json',result)
    table='|组|训练128满分|seen512%|SFT图像隔离1617%|保持64%|\n|---|---:|---:|---:|---:|\n'
    for role,v in scores.items():table+=f'|{role}|{full[role]}|{v["seen_validation"]:.6f}|{v["transfer_dev"]:.6f}|{v["retention"]:.6f}|\n'
    RESULT.write_text('# MLP函数保持：完整结果\n\n'+table+'\nKL候选联合门槛='+str(result['passed'])+'；hard对照独立报告，不将控制替换为候选挽救原实验。全部升降与区间见outputs/vlm_mlp_reference_v1/ANALYSIS.json。\n\n[协议](VLM_MLP_REFERENCE_V1_PROTOCOL.md) · [原图与KL原理](learning/MLP输出分布保持_KL与硬回放.html)\n')
    return result
