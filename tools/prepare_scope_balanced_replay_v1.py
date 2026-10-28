"""Prepare image-matched answer-scope replay views only; no optimizer or main-screen reuse for labels."""
from pathlib import Path
import json
from rarevlm_paths import PROJECT_ROOT
from vlm_deep.common import encode, content_sha, write_json
from vlm_probe.evaluate import processor_for

BASE = PROJECT_ROOT.parents[1] / 'outputs'
ROOT = BASE / 'vlm_scope_balanced_replay_v1'
read = lambda path: json.loads(Path(path).read_text())
MARKER = 'answering does not require reading text in the image'


def main():
    if (ROOT / 'DATA_COMPLETE.json').exists():
        raise FileExistsError('Already prepared; do not overwrite existing views')
    candidates = read(BASE / 'vlm_answer_granularity_audit_v1/TRAIN_PAIR_CANDIDATES.json')
    pairs = candidates['pairs'][:256]
    if len(pairs) != 256:
        raise ValueError('Need256 actual same-image candidates, no fallback')
    c = read(PROJECT_ROOT / 'configs/vlm_region_preference_probe_v1.json')
    source = read(c['source_manifest'])
    training = source['splits']['train_all']
    nonmarker = [q for q in training if q['target'] != MARKER]
    ratio = sum(len(q['target'].split()) >= 2 for q in nonmarker) / len(nonmarker)
    counts = {'empirical_mix': 67, 'scope_balanced': round(512 * ratio)}
    assert counts['scope_balanced'] == 145
    views = {}
    unique = {}
    for role, count in counts.items():
        rows = []
        for i, pair in enumerate(pairs):
            q = dict(pair['long_question'] if i < count else pair['short_question'])
            assert q['image_id'] == pair['image_id'] and q['target'] != MARKER
            q['study_source'] = 'common'
            rows.append(q)
            unique[q['question_id']] = q
        views[role] = rows
    assert [q['image_id'] for q in views['empirical_mix']] == [q['image_id'] for q in views['scope_balanced']]
    initial = read(BASE / 'vlm_region_preference_probe_v1/data/MANIFEST.json')
    hard = [q for q in initial['rows'] if q['study_source'] == 'preference']
    assert not {q['image_id'] for q in hard} & {q['image_id'] for q in views['empirical_mix']}
    base = read(c['baseline_config'])
    processor = processor_for(Path(base['model_root']), read(base['reference_config']), 1024)
    cpu = {}
    for qid, q in unique.items():
        item = encode(processor, q, Path(candidates['images'][q['image_id']]['path']), True)
        if item['input_ids'].shape[1] > 2048 or int((item['labels'] != -100).sum()) <= 0:
            raise ValueError('New replay outside original training shape budget')
        cpu[qid] = {'input_ids_sha256': content_sha(item['input_ids'].tolist()),
                    'labels_sha256': content_sha(item['labels'].tolist()),
                    'input_tokens': int(item['input_ids'].shape[1]), 'assistant_tokens': int((item['labels'] != -100).sum())}
    images = {q['image_id']: candidates['images'][q['image_id']] for q in views['empirical_mix']}
    write_json(ROOT / 'MATCHED_REPLAY_VIEWS.json', {'status': 'prepared_data_only_not_trained', 'views': views,
               'images': images, 'shared_region_rows': hard, 'initial_adapter': c['initial_adapter']})
    write_json(ROOT / 'REPLAY_CPU_INPUTS.json', cpu)
    result = {'status': 'prepared_data_only_not_trained', 'paired_replay_images': 256,
              'actual_unique_replay_cpu_encodings': len(cpu), 'new_optimizer_updates': 0,
              'identified_source_marker_count': len(training) - len(nonmarker),
              'source_multiword_percent_excluding_identified_marker': ratio * 100,
              'replay_multiword_counts': counts, 'whole512_multiword_percent': {k: 100 * v / 512 for k, v in counts.items()},
              'question_target_swaps_same_image': sum(a['question_id'] != b['question_id'] for a, b in zip(*views.values())),
              'boundary': 'Only actual training VQA questions and >=8/10 original answers. Both question and target differ on swapped slots; not a pure causal length manipulation. Same images/QA exposures, not equal assistant token compute. No expert audit, model run, benchmark or training-admission claim.'}
    write_json(ROOT / 'DATA_COMPLETE.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
