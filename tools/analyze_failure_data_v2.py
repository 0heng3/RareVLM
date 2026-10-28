"""Existing-data coverage analysis; no model, re-scoring or label changes."""
from collections import Counter, defaultdict
import gc
import hashlib
import json
from pathlib import Path
import re
import runpy
import time

PROJECT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT.parents[1] / 'outputs'
OUT = OUTPUTS / 'vlm_failure_factor_review_v2'


def read(p):
    return json.loads(Path(p).read_text())


def write(p, value):
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    started = time.monotonic()
    OUT.mkdir(exist_ok=True, parents=True)
    normalizer = runpy.run_path(str(PROJECT / 'src/vlm_deep/third_party/vqa_eval_metric.py'))['EvalAIAnswerProcessor']()
    m = read(OUTPUTS / 'vlm_deep_v2/prepared/manifest.json')
    train = m['splits']['train_all']; dev = m['splits']['dev']
    freq = Counter(normalizer(r['target']) for r in train)
    baseline_file = OUTPUTS / 'vlm_weighted_sft_v1/runs/uniform/seed42/dev/predictions.jsonl'
    pred = [json.loads(line) for line in baseline_file.read_text().splitlines()]
    grouped = defaultdict(list)
    for q, p in zip(dev, pred):
        assert str(q['question_id']) == str(p['question_id'])
        f = freq[normalizer(q['target'])]
        name = 'unseen0' if f == 0 else 'rare1_5' if f <= 5 else 'mid6_20' if f <= 20 else 'frequent21plus'
        grouped[name].append(p['score'])
    total_missing = sum(1-p['score'] for p in pred)
    text = {'train_qa': len(train), 'train_unique_target_strings': len(freq),
        'top_training_answers': freq.most_common(15),
        'annotation_feedback_target_count': freq['answering does not require reading text in image'],
        'unanswerable_target_count': freq['unanswerable'],
        'dev_loss_by_target_exposure': {name: {'qa': len(xs), 'score_percent': 100*sum(xs)/len(xs),
            'missing_score': sum(1-x for x in xs), 'share_of_missing_score': sum(1-x for x in xs)/total_missing}
            for name, xs in grouped.items()},
        'scope': 'downstream target-string exposure, not visual classes or pretraining exposure; meta answers are not automatically wrong labels'}
    write(OUT/'TEXT_DATA.json', text)
    obj = read(OUTPUTS/'vlm_object_presence_v2/TRAIN_MANIFEST.json')
    classes = set(obj['category_scope'])
    positive = [r for r in obj['rows'] if r['target'] == 'yes']
    object_summary = {'train_qa': len(obj['rows']), 'images': len(obj['images']), 'class_count': len(classes),
        'categories': sorted(classes), 'class_counts': dict(Counter(r['category_name'] for r in obj['rows'])),
        'positive_qa': len(positive), 'small_only_positive_qa': sum(all(b['coco_area_bin']=='small' for b in r['bbox_evidence']) for r in positive),
        'any_small_positive_qa': sum(any(b['coco_area_bin']=='small' for b in r['bbox_evidence']) for r in positive),
        'positive_instance_area_bins': dict(Counter(b['coco_area_bin'] for r in positive for b in r['bbox_evidence']))}
    pope = read(OUTPUTS/'vlm_posttrain_pope_v1/data/manifest.json')
    questions = {r['question_id']: r for r in pope['records']}
    observations = []
    pfile = OUTPUTS/'vlm_public_posttrain_diagnostic_v1/runs/initial/seed42/pope/predictions.jsonl'
    for line in pfile.open():
        p = json.loads(line); q = questions[p['question_id']]
        match = re.fullmatch(r'Is there an? (.+) in the image\?', q['question'])
        if not match: raise ValueError('unrecognized POPE object question')
        observations.append({'qid': q['question_id'], 'image_id': int(q['image_id'].split('_')[-1]),
            'object': match.group(1), 'label': q['label'], 'predicted': p['official_prediction'],
            'fn': q['label']=='yes' and p['official_prediction']=='no',
            'raw_width': p['raw_width'], 'raw_height': p['raw_height']})
    fn = [r for r in observations if r['fn']]
    positive_obs = [r for r in observations if r['label']=='yes']
    coverage = {'qa_observations': len(observations), 'positive_qa_observations': len(positive_obs), 'fn_qa_observations': len(fn),
        'fn_by_object': dict(Counter(r['object'] for r in fn).most_common()),
        'fn_in_training_classes': sum(r['object'] in classes for r in fn),
        'fn_outside_training_classes': sum(r['object'] not in classes for r in fn),
        'scope': 'POPE variants repeat image-object pairs; descriptive observation counts, not independent samples'}
    write(OUT/'POPE_COVERAGE.json', coverage)
    print('CLASS_COVERAGE', json.dumps(coverage), flush=True)
    # Existing official annotations, already acquired and verified by the old data study.
    # Read once for a new size-coverage query; no rehash or relabeling.
    wanted = {r['image_id'] for r in positive_obs}
    image_meta = {}; boxes = defaultdict(list); category_name = {}
    lock = read(OUTPUTS/'vlm_object_presence_v1/SOURCE_LOCK.json')
    for item in lock['annotation_lock']['json_files']:
        data = read(item['path'])
        category_name.update({c['id']:c['name'] for c in data['categories']})
        image_meta.update({x['id']:x for x in data['images'] if x['id'] in wanted})
        for a in data['annotations']:
            if a['image_id'] in wanted:
                boxes[(a['image_id'],category_name[a['category_id']])].append(a)
        del data;gc.collect()
    def bucket(row):
        if row['object'] not in classes: return 'outside_18_classes'
        annotations = boxes.get((row['image_id'], row['object']), [])
        image = image_meta.get(row['image_id'])
        if not annotations or image is None: return 'missing_annotation_alignment'
        for b in annotations:
            width = b['bbox'][2] * row['raw_width'] / image['width']
            height = b['bbox'][3] * row['raw_height'] / image['height']
            if not b.get('iscrowd',0) and min(width,height)>=24 and b['area']/(image['width']*image['height'])>=.005:
                return 'inside_18_and_size_eligible'
        return 'inside_18_but_size_or_crowd_excluded'
    counts = defaultdict(Counter); unique = defaultdict(list)
    for row in positive_obs:
        row['bucket'] = bucket(row)
        counts[row['bucket']]['positive_observations'] += 1
        counts[row['bucket']]['fn_observations'] += int(row['fn'])
        unique[(row['image_id'],row['object'])].append(row)
    unique_counts = defaultdict(Counter)
    for rows in unique.values():
        assert len({r['bucket'] for r in rows})==1
        name=rows[0]['bucket'];unique_counts[name]['positive_image_object_pairs'] += 1
        unique_counts[name]['fn_in_any_repeated_observation'] += int(any(r['fn'] for r in rows))
        unique_counts[name]['fn_in_all_repeated_observations'] += int(all(r['fn'] for r in rows))
    coverage.update(size_buckets={k:dict(v) for k,v in counts.items()},unique_pair_buckets={k:dict(v) for k,v in unique_counts.items()},
        size_rule='same noncrowd instance; bbox short side>=24 in actual raw image and COCO segmentation area/image area>=.005',
        matching_positive_images=len(image_meta),object_training=object_summary)
    write(OUT/'POPE_COVERAGE.json',coverage)
    # A finite, train-only optimization positive control; never a resume result.
    excluded={'unanswerable','answering does not require reading text in image'}
    valid={str(q['question_id']):q for q in train if normalizer(q['target']) not in excluded
           and normalizer(q['target']) and sum(normalizer(a)==normalizer(q['target']) for a in q['answers'])>=8}
    prior=read(OUTPUTS/'vlm_failure_code_review_v1/FIT.json')['fixed_sample_loss']['initial']['train_all']['records']
    hard=sorted([r for r in prior if str(r['question_id']) in valid],key=lambda r:(-r['ce'],str(r['question_id'])))[:16]
    fit=[valid[str(r['question_id'])] for r in hard];used={r['image_id'] for r in fit}
    def key(q,salt):return hashlib.sha256((salt+'|'+str(q['question_id'])).encode()).hexdigest()
    easy=sorted([q for q in valid.values() if q['image_id'] not in used],key=lambda q:key(q,'factor_v2_control'))[:16]
    fit+=easy;used|={q['image_id'] for q in easy}
    probe=sorted([q for q in valid.values() if q['image_id'] not in used],key=lambda q:key(q,'factor_v2_probe'))[:32]
    assert len(fit)==len(probe)==32 and not ({q['image_id'] for q in fit}&{q['image_id'] for q in probe})
    plan={'status':'registered_before_optimizer_updates','fit_question_ids':[q['question_id'] for q in fit],
        'probe_question_ids':[q['question_id'] for q in probe], 'selection':'16 harder high-consensus train QA by existing train NLL,16 input-hash high-consensus QA;32 image-disjoint train-domain probe QA',
        'answer_vote_minimum':8,'fit_qa':32,'probe_qa':32,'updates':32,'effective_batch':16,'microbatch':4,
        'seed':42,'fixed_learning_rate':True,'initial_checkpoint':'existing uniform SFT42 raw FP32 LoRA; same across all arms',
        'arms':{'amp_lr5e6':{'global_bf16_autocast':True,'lr':5e-6},
                'fp32_lora_lr5e6':{'global_bf16_autocast':False,'lr':5e-6},
                'fp32_lora_lr1e4':{'global_bf16_autocast':False,'lr':1e-4}},
        'evaluation':'same BF16 base plus raw FP32 LoRA/no autocast in every arm; official QA consensus and FP32 assistant CE',
        'positive_learning_check':'fit CE falls at least50% and fit consensus point does not decrease; diagnostic only',
        'scope':'real-data memorization/optimization diagnostic, not full experiment or new ability gain; probe was part of prior SFT train_all, not a new independent benchmark'}
    write(OUT/'CONTROL_PLAN.json',plan)
    write(OUT/'DATA_ANALYSIS.json',{'status':'completed','text':text,'object_and_pope':coverage,'seconds':time.monotonic()-started,'new_gpu_operations':0,'new_optimizer_updates':0})
    print('COMPLETE',json.dumps({'size_buckets':coverage['size_buckets'],'unique_pair_buckets':coverage['unique_pair_buckets'],'control_fit_qa':32,'control_probe_qa':32,'seconds':time.monotonic()-started}),flush=True)


if __name__=='__main__':main()
