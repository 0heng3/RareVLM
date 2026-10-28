"""Apply newly inspected training evidence; isolate inherited decoding processors.

No optimizer, training, heldout confirmation, historical relabeling or old-cache
overwrite. The optional GPU worker compares two generation policies on a fixed
16-QA TRAINING cohort and three immutable checkpoints, one token per answer.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

PACKAGE = Path(__file__).resolve().parents[1]
OUTPUTS = PACKAGE.parents[1] / 'outputs'
PARENT = OUTPUTS / 'vlm_attention_scope_v1'
ROOT = OUTPUTS / 'vlm_attention_scope_repair_v1'

# Every row below was inspected as an original image in the failure diagnosis;
# candle/mirror evidence also has earlier saved visual review provenance.
# Uncertainty means insufficient support, not proven annotation error.
REVIEW = {
    '238618005': ('question_semantics_require_context',
        'Close-up of a cow through rails; the opening state of a distinct gate is not clearly visible.'),
    '483069000': ('insufficient_fine_detail',
        'Red material is visible near the dog neck; the neck is partly occluded. A confident no-collar label is not established.'),
    '373254015': ('label_visually_supported',
        'Visible adult male at the left wears a dark cap; the foreground child hat has a dark upper surface. No visible adult man wears a white hat.'),
    '310783011': ('insufficient_fine_detail',
        'The woman lower body is obscured by a large foreground pink display. Skirt versus trousers cannot be reliably read.'),
    '196971004': ('label_visually_supported',
        'One orange cat and its mirror reflection, not two distinct cats; supported by the earlier original-image review.'),
    '487941010': ('label_visually_supported',
        'The man stands on two separate water skis, not a single board.'),
    '89939000': ('question_semantics_require_context',
        'Background has large graphics/graffiti; whether these count as ads is not established by legible business content.'),
    '253166001': ('question_semantics_require_context',
        'One player and court staff are visible. Whether opponent means visible opponent or an opponent in the match is unspecified.'),
    '444344000': ('label_visually_supported',
        'The red candle cups show no grouped stripes; supported by the earlier original-image review.'),
    '486361008': ('insufficient_fine_detail',
        'Curtain, TV and furniture are visible; the presence and exact location of a window relative to the cat are not established.'),
}


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, value):
    from vlm_deep.common import write_json
    write_json(Path(p), value)


def prepare():
    from prepare_semantic_admission_v1 import write_once
    from vlm_deep.common import content_sha
    from vlm_attention_scope_v1.study import verify as verify_parent
    from vlm_attention_scope_repair_v1.admission import apply_visual_reviews
    verify_parent()
    if (ROOT / 'INPUT_LOCK.json').exists():
        verify(); return
    v1 = OUTPUTS / 'vlm_semantic_admission_v1'
    receipt = read(v1 / 'PREPARATION_RESULT.json')
    source_path = Path(receipt['inputs']['source_train_only_export']['path'])
    prior_path = Path(receipt['inputs']['pretraining_review']['path'])
    for key, path in [('source_train_only_export', source_path), ('pretraining_review', prior_path)]:
        if sha(path) != receipt['inputs'][key]['sha256']:
            raise ValueError('Original admission source changed')
    source = read(source_path)
    source_rows = source['splits']['dev'] or source['splits']['train_all']
    rows = {q['question_id']: q for q in source_rows}
    reviews = []
    for suffix, (classification, evidence) in REVIEW.items():
        row = rows['vqa2_train2014:' + suffix]
        im = source['images'][row['image_id']]
        if sha(im['path']) != im['sha256']:
            raise ValueError('Reviewed original image bytes changed')
        reviews.append({'question_id': row['question_id'], 'pair_id': row['complementary_pair_id'],
            'source_split': 'train', 'target': row['target'], 'question': row['question'],
            'question_sha256': content_sha(row['question']), 'image_sha256': im['sha256'],
            'image_path': im['path'], 'classification': classification, 'evidence': evidence,
            'reviewer': 'assistant original-image inspection; not independent expert relabeling',
            'selection': 'All ten historical no training errors; diagnostic selection, not a noise-rate sample'})
    previous = v1 / 'ADDITIONAL_TRAINING_REVIEW.json'
    view, decisions, qa, summary = apply_visual_reviews(source, read(prior_path), read(previous), reviews)
    view['purpose'] = 'New post-scope train-only view with evidence-bound visual uncertainty vetoes'
    view['semantic_admission']['post_scope_review_version'] = 1
    view['manifest_sha256'] = content_sha(view)
    old = read(OUTPUTS / 'vlm_semantic_admission_v2/TRAINING_MANIFEST.json')
    old_ids = {q['question_id'] for q in old['splits']['train_all']}
    new_ids = {q['question_id'] for q in view['splits']['train_all']}
    if new_ids - old_ids:
        raise ValueError('This repair may not reintroduce excluded examples')
    summary.update(status='completed', new_reviewed_qa=len(reviews),
        review_classifications=dict(Counter(r['classification'] for r in reviews)),
        newly_excluded_qa=len(old_ids - new_ids), newly_excluded_pairs=len(old_ids - new_ids)//2,
        old_labels_and_heldout_unchanged=True, noise_rate_estimated=False,
        training_automatically_allowed=False, human_capability_gain_claim=False)
    for name, obj in [('VISUAL_TRAIN_REVIEW.json', reviews), ('TRAINING_MANIFEST.json', view),
                      ('PAIR_ADMISSION.json', decisions), ('QA_ADMISSION.json', qa),
                      ('PREPARATION_RESULT.json', summary)]:
        write_once(ROOT / name, obj)
    # Reuse a training-only diagnostic catalog. All historical no errors and six
    # hash-selected correct anchors. No heldout inputs or scores are read here.
    probe_path = PARENT / 'failure_analysis_v1/fixed_train_probe_v1/TRAIN_EVAL.json'
    probe = read(probe_path)
    neg = [q for q in probe['splits']['dev']
           if q['probe_group'] == 'historically_wrong' and q['target'] == 'no']
    anchors = []
    for label in ('yes', 'no'):
        candidates = [q for q in probe['splits']['dev']
                      if q['probe_group'] == 'correct_anchor' and q['target'] == label]
        candidates.sort(key=lambda q: hashlib.sha256(('post_scope_decode_v1' + q['question_id']).encode()).hexdigest())
        anchors.extend(candidates[:3])
    selected = neg + anchors
    if len(neg) != 10 or len(selected) != 16:
        raise ValueError('Fixed train-policy diagnostic changed')
    write_once(ROOT / 'POLICY_PROBE.json', {'qa': selected,
        'images': {q['image_id']: probe['images'][q['image_id']] for q in selected}})
    protocol = read(PARENT / 'failure_analysis_v1/fixed_train_probe_v1/PROTOCOL.json')
    protocol.update(generation_comparison=['inherited_greedy_repetition_1.05', 'explicit_raw_greedy_repetition_1'],
        qa_per_model=16, generation_tokens=1, maximum_worker_seconds=300,
        scope='Train-only first-token processor diagnosis, not a capability evaluation',
        training_updates=0, confirmation_used=False,
        cohort_path=str(ROOT / 'POLICY_PROBE.json'))
    write_once(ROOT / 'PROTOCOL.json', protocol)
    files = [Path(__file__), source_path, prior_path, previous, probe_path,
        PACKAGE / 'src/vlm_attention_scope_repair_v1/admission.py',
        PACKAGE / 'src/vlm_attention_scope_repair_v1/generation.py',
        PACKAGE / 'tests/test_vlm_attention_scope_repair_v1.py'] + list(ROOT.glob('*.json'))
    for model in protocol['models'].values():
        files.extend(Path(model['adapter']) / n for n in ('adapter_config.json', 'adapter_model.safetensors'))
    write_once(ROOT / 'INPUT_LOCK.json', {'files': {str(p): sha(p) for p in files},
        'training_updates': 0, 'parent_scientific_files_immutable': True})
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def verify():
    from vlm_attention_scope_v1.study import verify as verify_parent
    verify_parent()
    for path, expected in read(ROOT / 'INPUT_LOCK.json')['files'].items():
        if sha(path) != expected:
            raise ValueError('Repair input changed: ' + path)


def worker():
    import torch
    from vlm_deep.common import collate, encode, to_device
    from vlm_runtime_metadata_v1.model import load
    from vlm_fp32_adapter.model import restore_fp32_adapter
    from vlm_probe.evaluate import processor_for, inference_row
    from vlm_human_sft.evaluate import lora_hash, binary_answer
    from vlm_attention_scope_repair_v1.generation import raw_greedy_config

    verify()
    started = time.monotonic()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    p = read(ROOT / 'PROTOCOL.json'); c = read(p['baseline_config']); data = read(p['cohort_path'])
    model, _ = load(c['model_root'], p['models']['initial']['adapter'], torch.device('cuda:0'), training=False)
    processor = processor_for(Path(c['model_root']), read(c['reference_config']), 1024)
    records = []
    for role in ('initial', 'qv', 'qkvo'):
        if role != 'initial':
            restore_fp32_adapter(model, p['models'][role]['adapter'])
        if lora_hash(model) != p['models'][role]['lora_sha256']:
            raise ValueError('Diagnostic loaded a different checkpoint')
        for start in range(0, 16, 4):
            if time.monotonic() - started > p['maximum_worker_seconds']:
                raise TimeoutError('Bounded diagnostic exceeded its budget')
            chunk = data['qa'][start:start+4]
            encoded = [encode(processor, inference_row(q), Path(data['images'][q['image_id']]['path']), False)
                       for q in chunk]
            batch = to_device(collate(encoded, processor.tokenizer.pad_token_id), torch.device('cuda:0'))
            values = {}
            for policy_name, policy in [('inherited', model.generation_config),
                                         ('raw', raw_greedy_config(model.generation_config, 1))]:
                with torch.inference_mode(), torch.autocast('cuda', enabled=False):
                    output = model.generate(**batch, generation_config=policy, do_sample=False,
                        max_new_tokens=1, use_cache=True, return_dict_in_generate=True,
                        output_logits=True, output_scores=True)
                raw = output.logits[0].float(); scores = output.scores[0].float()
                values[policy_name] = [{'token_id': int(output.sequences[i, -1]),
                    'token_text': processor.tokenizer.decode(output.sequences[i, -1:]),
                    'raw_argmax_token_id': int(raw[i].argmax()),
                    'processed_argmax_token_id': int(scores[i].argmax()),
                    'processor_changed_logits': int((raw[i] != scores[i]).sum()),
                    'max_processor_logit_change': float((raw[i] - scores[i]).abs().max())}
                    for i in range(len(chunk))]
            records.extend({'role': role, 'question_id': q['question_id'], 'target': q['target'],
                'probe_group': q['probe_group'],
                **{name: {**values[name][i],
                    'binary_correct': binary_answer(values[name][i]['token_text']) == q['target']}
                   for name in ('inherited', 'raw')}} for i, q in enumerate(chunk))
            write(ROOT / 'PROGRESS.json', {'status': 'diagnosing_decoding', 'role': role,
                'completed_qa_this_role': start+4, 'elapsed_seconds': time.monotonic()-started,
                'training_updates': 0})
        if lora_hash(model) != p['models'][role]['lora_sha256']:
            raise ValueError('Read-only diagnostic changed adapter weights')
    summary = {}
    for role in ('initial', 'qv', 'qkvo'):
        group = [r for r in records if r['role'] == role]
        summary[role] = {'qa': len(group),
            'policy_token_changes': sum(r['inherited']['token_id'] != r['raw']['token_id'] for r in group),
            'inherited_raw_argmax_disagreements': sum(r['inherited']['token_id'] != r['inherited']['raw_argmax_token_id'] for r in group),
            'raw_policy_raw_argmax_disagreements': sum(r['raw']['token_id'] != r['raw']['raw_argmax_token_id'] for r in group),
            'hard_no_correct': {name: sum(r[name]['binary_correct'] for r in group
                if r['probe_group'] == 'historically_wrong') for name in ('inherited', 'raw')}}
    verify()
    write(ROOT / 'DECODE_DIAGNOSTIC.json', {'status': 'completed', 'results': summary,
        'records': records, 'worker_seconds': time.monotonic()-started, 'training_updates': 0,
        'scope': '48 checkpoint-times-QA evaluated with two first-token policies, not 96 independent examples',
        'new_capability_gain_claim': False})
    write(ROOT / 'WORKER_COMPLETE.json', {'status': 'completed', 'training_updates': 0})
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare(); return
    if args.worker:
        worker(); return
    verify()
    from research_runtime.queue import run_queue
    run_queue(ROOT / 'queue', [{'key': 'decode_policy',
        'command': [sys.executable, '-u', str(Path(__file__).resolve()), '--worker'],
        'completion_marker': str(ROOT / 'WORKER_COMPLETE.json')}],
        {'input_lock_sha256': sha(ROOT / 'INPUT_LOCK.json'), 'training_updates': 0})
    write(ROOT / 'COMPLETION.json', {'status': 'completed', 'training_updates': 0,
        'training_data_repair': str(ROOT / 'PREPARATION_RESULT.json'),
        'decoding_diagnostic': str(ROOT / 'DECODE_DIAGNOSTIC.json'),
        'new_capability_gain_claim': False, 'automatic_training_started': False})


if __name__ == '__main__':
    main()
