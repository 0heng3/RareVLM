#!/usr/bin/env python3
"""Join fixed training errors to immutable exposure traces; CPU, standard library only."""
from __future__ import annotations
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

PRIVATE = Path('/opt/data/private')
BASE = PRIVATE / 'longtail_vlm/rarevlm_pilot'
SRC = BASE / 'outputs/vlm_complementary_rank_v1'
FIXED = BASE / 'outputs/vlm_fixed_error_readout_v1'
OUT = BASE / 'outputs/vlm_deep_pipeline_diagnosis_v1/data'
PACKAGE = BASE / 'packages/RareVLM_minimal_20261001'

def read(path):
    return json.loads(Path(path).read_text())

def rows(path):
    with Path(path).open() as handle:
        for line_number, line in enumerate(handle, 1):
            yield line_number, json.loads(line)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')

def desc(values):
    values = list(values)
    return {'n': len(values), 'mean': statistics.mean(values) if values else None,
            'median': statistics.median(values) if values else None,
            'min': min(values) if values else None, 'max': max(values) if values else None}

def hist(values):
    return dict(sorted(Counter(values).items()))

def learning_rate(update, total, config):
    step = update - 1
    warm = max(math.ceil(total * config['warmup_ratio']), 1)
    factor = (step + 1) / warm if step < warm else (total - step) / max(total - warm, 1)
    return config['lr'] * factor

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train = read(SRC / 'data/TRAIN_EVAL.json')
    qa = {str(x['question_id']): x for x in train['splits']['dev']}
    initial = {str(x['question_id']): (number, x) for number, x in rows(SRC / 'initial/train/human/predictions.jsonl')}
    cohort = read(FIXED / 'data/COHORT.json')
    selected_pairs = {x['pair_id']: x for x in cohort['pairs']['train_selected']}
    pair_diagnoses = defaultdict(dict)
    pair_lines = defaultdict(dict)
    for number, row in rows(FIXED / 'PAIR_DIAGNOSES.jsonl'):
        if row['task'] == 'train_selected':
            pair_diagnoses[row['pair_id']][row['role']] = row
            pair_lines[row['pair_id']][row['role']] = number
    config_path = PACKAGE / 'configs/vlm_complementary_rank_v1.json'
    config = read(config_path)
    schedule = read(SRC / 'data/SCHEDULE.json')
    total = len(schedule['updates'])
    exposures = defaultdict(lambda: defaultdict(list))
    provenance = [SRC/'data/TRAIN_EVAL.json', SRC/'initial/train/human/predictions.jsonl', FIXED/'data/COHORT.json',
                  FIXED/'PAIR_DIAGNOSES.jsonl', config_path, SRC/'data/SCHEDULE.json']
    omit = {'input_ids', 'labels', 'attention_mask'}
    for role in ['ce', 'ce_rank']:
        path = SRC / 'runs' / role / 'train/INPUT_TRACES.jsonl'
        provenance.append(path)
        for number, row in rows(path):
            scheduled = schedule['updates'][row['update']-1]['micros'][row['microbatch_index']]
            assert scheduled['question_ids'][row['microbatch_question_position']] == row['question_id']
            assert scheduled['source'] == row['source']
            if row['source'] == 'human':
                assert row['target'] == qa[row['question_id']]['target']
                assert row['image_sha256'] == train['images'][row['image_id']]['sha256']
            item = {k: v for k, v in row.items() if k not in omit}
            item.update({'source_path': str(path), 'source_line': number,
                         'lr_at_update': learning_rate(row['update'], total, config)})
            item['initial_actual_generation_wrong'] = bool(row['source'] == 'human' and initial[str(row['question_id'])][1]['primary_score'] != 1.)
            exposures[role][str(row['question_id'])].append(item)
    readouts = {}
    predictions = {}
    for role in ['initial', 'ce', 'ce_rank']:
        path = FIXED / 'runs' / role / 'train_selected/readout.jsonl'
        provenance.append(path)
        readouts[role] = {str(x['question_id']): dict(x, source_path=str(path), source_line=n) for n, x in rows(path)}
        if role == 'initial':
            predictions[role] = {k: dict(v, source_path=str(SRC/'initial/train/human/predictions.jsonl'), source_line=n) for k, (n,v) in initial.items()}
        else:
            path = FIXED / 'runs' / role / 'train_generation/predictions.jsonl'
            provenance.append(path)
            predictions[role] = {str(x['question_id']): dict(x, source_path=str(path), source_line=n) for n, x in rows(path)}
    joined = []
    for qid in readouts['initial']:
        row = qa[qid]
        pairid = row['complementary_pair_id']
        pair = selected_pairs[pairid]
        group = 'repaired' if pair_diagnoses[pairid]['ce_rank']['generation']['both_correct'] else 'residual'
        item = {'question_id': qid, 'pair_id': pairid, 'anchor_group': pair['anchor_group'],
                'original_wrong_pair_final_group': group if pair['anchor_group'] == 'initial_wrong' else None,
                'question': row['question'], 'target': row['target'], 'answer_vote_count': row['answer_vote_count'],
                'answers': row['answers'], 'question_type': row.get('question_type'), 'question_tags': pair.get('question_tags',[]),
                'image_id': row['image_id'], 'image': train['images'][row['image_id']],
                'initial_wrong_qa': initial[qid][1]['primary_score'] != 1.,
                'predictions': {role: predictions[role][qid] for role in predictions},
                'readouts': {role: readouts[role][qid] for role in readouts},
                'training_exposures': {role: exposures[role][qid] for role in ['ce','ce_rank']},
                'pair_diagnosis_source_lines': pair_lines[pairid]}
        joined.append(item)
    with (OUT/'TRAIN_QA_JOIN.jsonl').open('w') as handle:
        for item in joined:
            handle.write(json.dumps(item, ensure_ascii=False, allow_nan=False)+'\n')
    wrongpairs = [x for x in selected_pairs.values() if x['anchor_group']=='initial_wrong']
    report = {'scope': 'Fixed Final200 training diagnosis; no GPU, no model updates, no confirmation evaluation.',
              'source_sha256': {str(p): sha(p) for p in provenance},
              'counts': {'full_training_qa': len(qa), 'full_training_pairs': len(qa)//2,
                         'initial_wrong_qa': sum(v[1]['primary_score'] != 1 for v in initial.values()),
                         'initial_wrong_pairs': len(wrongpairs), 'joined_diagnostic_qa': len(joined)},
              'groups': {}, 'all_training': {}}
    pairs_out = []
    for pair in wrongpairs:
        pid=pair['pair_id']; qids=[pair['positive_question_id'],pair['negative_question_id']]
        qrows=[next(x for x in joined if x['question_id']==qid) for qid in qids]
        rankexp=exposures['ce_rank'][qids[0]]
        assert [x['update'] for x in rankexp]==[x['update'] for x in exposures['ce_rank'][qids[1]]]
        pairs_out.append({'pair_id':pid,'question':pair['question'],'question_tags':pair.get('question_tags',[]),
                          'ce_rank_repaired':pair_diagnoses[pid]['ce_rank']['generation']['both_correct'],
                          'ce_repaired':pair_diagnoses[pid]['ce']['generation']['both_correct'],
                          'pair_exposures':len(rankexp),'updates':[x['update'] for x in rankexp],
                          'lrs':[x['lr_at_update'] for x in rankexp],
                          'initial_wrong_question_ids':[x['question_id'] for x in qrows if x['initial_wrong_qa']],
                          'rank_final_wrong_question_ids':[x['question_id'] for x in qrows if x['predictions']['ce_rank']['primary_score']!=1.],
                          'generation': {r:pair_diagnoses[pid][r]['generation'] for r in ['initial','ce','ce_rank']},
                          'canonical': {r:pair_diagnoses[pid][r]['readouts']['canonical'] for r in ['initial','ce','ce_rank']},
                          'exposure_trace_lines':{r:[exposures[r][qid][i]['source_line'] for i in range(len(rankexp)) for qid in qids] for r in ['ce','ce_rank']}})
    write('TRAIN_PAIR_JOIN.json', pairs_out)
    for group in ['repaired','residual']:
        ps=[x for x in pairs_out if x['ce_rank_repaired'] == (group=='repaired')]
        qs=[x for x in joined if x['original_wrong_pair_final_group']==group]
        initialwrong=[x for x in qs if x['initial_wrong_qa']]
        ev=[e for x in initialwrong for e in x['training_exposures']['ce_rank']]
        wrongnow=[x for x in qs if x['predictions']['ce_rank']['primary_score'] != 1.]
        report['groups'][group]={'pairs':len(ps),'pair_exposures':sum(x['pair_exposures'] for x in ps),
          'pair_exposure_histogram':hist(x['pair_exposures'] for x in ps),'initial_wrong_qa':len(initialwrong),
          'initial_wrong_qa_exposures':len(ev),'initial_wrong_qa_exposure_histogram':hist(len(x['training_exposures']['ce_rank']) for x in initialwrong),
          'initial_wrong_qa_exposure_updates':desc(x['update'] for x in ev),'initial_wrong_qa_exposure_lr':desc(x['lr_at_update'] for x in ev),
          'exposure_lt_half_peak_lr':sum(x['lr_at_update']<config['lr']/2 for x in ev),
          'initial_wrong_qa_online_mean_three_token_ce':desc(x['qa_ce'] for x in ev),
          'initial_wrong_qa_first_online_ce':desc(x['training_exposures']['ce_rank'][0]['qa_ce'] for x in initialwrong),
          'initial_wrong_qa_last_online_ce':desc(x['training_exposures']['ce_rank'][-1]['qa_ce'] for x in initialwrong),
          'qa_with_two_exposures_online_ce_delta':desc(x['training_exposures']['ce_rank'][-1]['qa_ce']-x['training_exposures']['ce_rank'][0]['qa_ce'] for x in initialwrong if len(x['training_exposures']['ce_rank'])==2),
          'final_wrong_qa':len(wrongnow),'final_wrong_target_histogram':hist(x['target'] for x in wrongnow),
          'official_question_type':hist(x['question_type'] for x in wrongnow),
          'question_tags':hist(t for x in ps for t in x['question_tags']),
          'vote_count':hist(x['answer_vote_count'] for x in qs),
          'final_wrong_canonical_abs_margin':desc(abs(x['readouts']['ce_rank']['canonical_margin']) for x in wrongnow),
          'final_wrong_conditional_chosen_probability':desc(max(x['readouts']['ce_rank']['canonical_conditional_yes_probability'],1-x['readouts']['ce_rank']['canonical_conditional_yes_probability']) for x in wrongnow)}
    for role in ['ce','ce_rank']:
        all_ev=[e for v in exposures[role].values() for e in v]
        human=[e for e in all_ev if e['source']=='human']
        wrong=[e for e in human if e['initial_actual_generation_wrong']]
        allce=sum(e['qa_ce'] for e in all_ev)
        report['all_training'][role]={'qa_exposures':len(all_ev),'human_qa_exposures':len(human),
            'human_unique_qa':len({e['question_id'] for e in human}), 'human_exposure_histogram':hist(len(v) for q,v in exposures[role].items() if q in qa),
            'initial_wrong_human_exposures':len(wrong),'initial_wrong_percent_of_human_exposures':100*len(wrong)/len(human),
            'initial_wrong_percent_of_all_exposures':100*len(wrong)/len(all_ev),
            'initial_wrong_qa_ce_sum':sum(e['qa_ce'] for e in wrong),
            'initial_wrong_qa_ce_share_of_total_ce_percent':100*sum(e['qa_ce'] for e in wrong)/allce,
            'initial_wrong_qa_ce_share_of_human_ce_percent':100*sum(e['qa_ce'] for e in wrong)/sum(e['qa_ce'] for e in human),
            'human_qa_ce_sum':sum(e['qa_ce'] for e in human),'all_qa_ce_sum':allce,
            'truncated_exposures':sum(e['truncated'] for e in all_ev),
            'human_assistant_token_count':hist(e['assistant_tokens'] for e in human)}
    # Fixed diagnostic selection: four final false-positive and four final false-negative QA
    # from the 36 unrepaired original-error pairs, salted SHA order within each direction.
    visual=[]
    observations = {
        'vqa2_train2014:509235000': ('insufficient_fine_detail', '原图中领带与领口遮挡最上方纽扣；no标签未发现明确反证，但不能从此图可靠判断扣眼是否打开。'),
        'vqa2_train2014:444344000': ('label_visually_supported', '桌上红色蜡烛/烛杯可见纹理和亮点，未见成组条纹；原no标签有直接图像支持，rank却回答yes。'),
        'vqa2_train2014:71754001': ('label_plausible_detail_limited', '暗光玻璃容器顶部可见疑似合上的玻璃盖及金属扣；原no合理，但反光与开合结构限制独立判断把握。'),
        'vqa2_train2014:184218002': ('question_semantics_require_context', '女子明显向侧方伸展接球；right to her可指直接到她面前，也易被误读为右侧，球路和意图不是单帧可完整验证的状态。'),
        'vqa2_train2014:297559000': ('label_supported_by_unpacking_evidence', 'Wii控制器与线缆已取出散放，支持包装已经拆开的yes语义；箱子本体开口不直接可见，不能把open一词一律当物理开口检测。'),
        'vqa2_train2014:472676002': ('small_overlapping_people_count_uncertain', '人物位于海滩底部且高度较小、身体与腿部重叠；可明确见一名主体，无法据全图可靠排除紧贴重叠的第二人，不认定原yes错标。'),
        'vqa2_train2014:360575013': ('label_visually_supported', '前景和后方多辆巴士均为黄色主体配红色条带/车顶，原same color=yes有直接图像支持，rank仍回答no。'),
        'vqa2_train2014:543166001': ('subjective_category_and_proportion', '停车计时器/汽车/建筑混合灰银白色；predominance需要约定面积比例及silver与gray/white边界，无法按明确客观规则判定yes。'),
    }
    for target in ['no','yes']:
        candidates=[x for x in joined if x['original_wrong_pair_final_group']=='residual' and x['target']==target and x['predictions']['ce_rank']['primary_score']!=1.]
        candidates.sort(key=lambda x:hashlib.sha256(('deep_data_review_v1|'+x['question_id']).encode()).hexdigest())
        for x in candidates[:4]:
            visual.append({'selection_rule':'salted SHA order; 4 final false positives and 4 final false negatives within 36 residual training pairs',
                           **{k:x[k] for k in ['question_id','pair_id','question','target','answer_vote_count','answers','image_id','question_type','question_tags']},
                           'image_path':x['image']['path'],'image_sha256':x['image']['sha256'],
                           'initial_prediction':x['predictions']['initial']['prediction'],
                           'ce_prediction':x['predictions']['ce']['prediction'],'rank_prediction':x['predictions']['ce_rank']['prediction'],
                           'rank_canonical_margin':x['readouts']['ce_rank']['canonical_margin'],
                           'exposure_steps':[e['update'] for e in x['training_exposures']['ce_rank']],
                           'visual_review': {'status':'assistant_original_image_review_completed',
                             'method':'Each original image opened with view_image; no crop, resize, model inference, or label change.',
                             'classification':observations[x['question_id']][0],
                             'observation':observations[x['question_id']][1],
                             'original_label_unchanged':True}})
    write('VISUAL_REVIEW_COHORT.json',visual)
    review_path=SRC/'data_preparation/revision2/DATA_REVIEW.json'
    prior_review=read(review_path)
    known_uncertain=[]
    for row in prior_review['rows']:
        if row['split']!='train' or row['pair_passed']:
            continue
        qrows=[x for x in qa.values() if x['complementary_pair_id']==row['pair_id']]
        qids=[str(x['question_id']) for x in qrows]
        case={'prior_blind_review':row,'retained_in_actual_training':bool(qrows),
              'training_qa':qrows,
              'initial_generation':{qid:initial[qid][1] for qid in qids},
              'actual_rank_training_exposures':{qid:exposures['ce_rank'][qid] for qid in qids},
              'rank_fixed_diagnosis':pair_diagnoses.get(row['pair_id'],{}).get('ce_rank'),
              'source_path':str(review_path),'source_sha256':sha(review_path)}
        known_uncertain.append(case)
    write('KNOWN_PRETRAIN_REVIEW_UNCERTAIN.json',known_uncertain)
    report['known_prior_blind_review_uncertainty']={'train_review_passed':prior_review['train_passed_pairs'],
       'train_review_total':16,'required_train_passed':prior_review['required']['train'],
       'known_uncertain_pairs_retained':sum(x['retained_in_actual_training'] for x in known_uncertain),
       'repaired_original_wrong_pairs_previously_marked_uncertain':sum(bool(x['rank_fixed_diagnosis'] and x['rank_fixed_diagnosis']['anchor_group']=='initial_wrong' and x['rank_fixed_diagnosis']['generation']['both_correct']) for x in known_uncertain),
       'interpretation':'The admission gate did not promise every pair was reviewed clean. Known uncertain samples remained; this is not an estimate of a population noise rate, and historical metrics are unchanged.'}
    report['limitations']=['Frozen prompt-only readouts and teacher-forced online training traces are different measurements.',
      'qa_ce averages all assistant tokens; human labels contain answer, im_end and newline. It is not first-answer-token NLL.',
      'Exposure LR and CE mass are descriptive; Adam, clipping and shared parameters prevent identifying causal gradient contribution.',
      'Selected 340-QA cohort is error-enriched, not overall training performance.',
      'Original official human vote agreement does not remove semantic ambiguity. Eight selected images do not estimate full-data noise.',
      'No confirmation evaluation, threshold fitting, new GPU forward, training, or relabeling performed.']
    write('DATA_DIAGNOSIS.json',report)
    print(json.dumps({'counts':report['counts'],'groups':report['groups'],'all_training':report['all_training'],'visual':[{'question_id':x['question_id'],'question':x['question'],'target':x['target'],'image_path':x['image_path']} for x in visual]},ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
