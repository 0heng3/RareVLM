"""Audit actual answer-granularity exposure and inventory genuine same-image replay candidates; no GPU."""
from pathlib import Path
from collections import Counter, defaultdict
import hashlib
import json
from rarevlm_paths import PROJECT_ROOT
from vlm_deep.common import read_manifest, encode, sha_file, write_json

BASE = PROJECT_ROOT.parents[1] / 'outputs'
ROOT = BASE / 'vlm_answer_granularity_audit_v1'
read = lambda path: json.loads(Path(path).read_text())


def words(row):
    return len(row['target'].strip().split())


def describe(rows):
    counts = Counter(words(row) for row in rows)
    return {'qa_exposures': len(rows), 'whitespace_word_count': dict(sorted(counts.items())),
            'multiword_qa': sum(n for k, n in counts.items() if k >= 2),
            'multiword_percent': 100 * sum(n for k, n in counts.items() if k >= 2) / len(rows)}


def main():
    if (ROOT / 'REPORT.json').exists():
        raise FileExistsError('Audit already completed; inspect saved evidence')
    config = read(PROJECT_ROOT / 'configs/vlm_region_preference_probe_v1.json')
    original = read_manifest(config['source_manifest'])
    current = read(BASE / 'vlm_region_preference_probe_v1/data/MANIFEST.json')
    schedule = read(BASE / 'vlm_region_preference_probe_v1/data/SCHEDULE.json')
    lookup = {q['question_id']: q for q in current['rows']}
    exposed = [lookup[qid] for update in schedule['updates'] for micro in update['micros'] for qid in micro]
    assert len(exposed) == 512
    changed = read(BASE / 'vlm_plain_sft_main_v1/CHANGED_CASES.json')['text']
    truncations = [q for q in changed if q['after'].strip() and q['before'].casefold().startswith(q['after'].casefold() + ' ')]
    training = original['splits']['train_all']
    pool = read(BASE / 'vlm_region_binding_data_v1/PAIR_CANDIDATES.json')
    allowed = set(pool['images']) - {q['image_id'] for q in current['hard_eval']}
    by_image = defaultdict(list)
    for q in training:
        if q['image_id'] in allowed and 1 <= words(q) <= 6 and any(c.isalnum() for c in q['target']):
            if Counter(q['answers'])[q['target']] >= 8:
                by_image[q['image_id']].append(q)
    pairs = []
    key = lambda q: hashlib.sha256(q['question_id'].encode()).hexdigest()
    for iid, rows in by_image.items():
        short = sorted((q for q in rows if words(q) == 1), key=key)
        long = sorted((q for q in rows if words(q) >= 2), key=key)
        if short and long:
            a, b = short[0], long[0]
            if a['question'] == b['question']:
                continue
            pairs.append({'image_id': iid, 'short_question': a, 'long_question': b,
                          'short_votes': Counter(a['answers'])[a['target']], 'long_votes': Counter(b['answers'])[b['target']]})
    pairs.sort(key=lambda q: hashlib.sha256(('scope_replay_v1:' + q['image_id']).encode()).hexdigest())
    selected_images = {q['image_id']: {**original['images'][q['image_id']],
        'path': str((Path(config['source_manifest']).parent / original['images'][q['image_id']]['path']).resolve())} for q in pairs}
    # Only a real eight-example CPU encoding spot check, not full candidate admission.
    from vlm_probe.evaluate import processor_for
    base = read(config['baseline_config'])
    processor = processor_for(Path(base['model_root']), read(base['reference_config']), 1024)
    checked = []
    for pair in pairs[:4]:
        for side in ('short_question', 'long_question'):
            q = pair[side]
            item = encode(processor, q, Path(selected_images[q['image_id']]['path']), True)
            n = int((item['labels'] != -100).sum())
            assert n > 0 and item['input_ids'].shape[1] <= 2048
            checked.append({'qid': q['question_id'], 'target': q['target'], 'assistant_tokens': n,
                            'input_tokens': int(item['input_ids'].shape[1])})
    report = {'status': 'completed', 'scope': 'CPU diagnosis and candidate inventory only; no training admission or new model gain',
              'word_definition': 'Whitespace word count is an answer-granularity proxy, not tokenizer length or visual category count',
              'source_train': describe(training), 'current_exposures': describe(exposed),
              'current_replay': describe([q for q in exposed if q['study_source'] == 'common']),
              'current_region': describe([q for q in exposed if q['study_source'] == 'preference']),
              'main_literal_prefix_truncations': truncations,
              'prefix_truncation_net_score_points': sum(q['after_score'] - q['before_score'] for q in truncations),
              'damaging_prefix_truncation_score_points': sum(q['after_score'] - q['before_score'] for q in truncations if q['after_score'] < q['before_score']),
              'same_image_short_long_candidate_pairs': len(pairs), 'cpu_spot_check': checked,
              'candidate_boundary': 'Two genuine different VQA questions/targets on same training image; >=8/10 literal votes, no invented truncation/padding, no eval image recycling. Not outcome-selected or expert audited.',
              'causal_boundary': 'Mixture difference plus observed truncations supports a scope-transfer hypothesis, not a unique causal diagnosis.',
              'new_optimizer_updates': 0, 'source_sha256': sha_file(Path(__file__))}
    write_json(ROOT / 'TRAIN_PAIR_CANDIDATES.json', {'status': 'inventory_only_not_admitted_for_training', 'pairs': pairs, 'images': selected_images})
    write_json(ROOT / 'REPORT.json', report)
    print(json.dumps({k: v for k, v in report.items() if k not in ('main_literal_prefix_truncations', 'cpu_spot_check')}, ensure_ascii=False, indent=2))
    print('REAL_CPU_EXAMPLES', json.dumps(checked, ensure_ascii=False))


if __name__ == '__main__':
    main()
