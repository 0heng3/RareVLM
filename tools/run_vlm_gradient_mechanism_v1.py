"""Measure source gradients at two immutable checkpoints; never instantiate an optimizer."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


PACKAGE = Path(__file__).resolve().parents[1]
OUTPUTS = PACKAGE.parents[1] / 'outputs'
ROOT = OUTPUTS / 'vlm_gradient_mechanism_v1'
DATA = OUTPUTS / 'vlm_semantic_admission_v1'
OLD = OUTPUTS / 'vlm_complementary_rank_v1'
FIXED = OUTPUTS / 'vlm_fixed_error_readout_v1'
PROBES = ['vqa2_train2014:444344000', 'vqa2_train2014:196971004',
          'vqa2_train2014:360575013', 'vqa2_train2014:178520002']


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(payload)
    tmp.replace(path)


def prepare():
    if (ROOT / 'INPUT_LOCK.json').exists():
        verify()
        print('Already prepared; unchanged inputs')
        return
    manifest_path = DATA / 'TRAINING_MANIFEST.json'
    data = read(manifest_path)
    if not data['training_allowed'] or data['splits']['dev'] or data['splits']['final']:
        raise ValueError('Need a train-only admitted manifest')
    human = {q['question_id']: q for q in data['splits']['train_all']}
    if any(q not in human for q in PROBES):
        raise ValueError('A reviewed training probe was not admitted')
    if [human[q]['target'] for q in PROBES] != ['no', 'no', 'yes', 'yes']:
        raise ValueError('Probe targets changed')
    schedule_path = OLD / 'data/SCHEDULE.json'
    schedule = read(schedule_path)
    chosen_human, chosen_text = [], []
    for update in schedule['updates']:
        for micro in update['micros']:
            qids = micro['question_ids']
            if micro['source'] == 'human' and len(chosen_human) < 8:
                if all(q in human and q not in PROBES for q in qids):
                    chosen_human.extend(qids)
            if micro['source'] == 'text' and len(chosen_text) < 8:
                chosen_text.extend(qids)
        if len(chosen_human) == len(chosen_text) == 8:
            break
    if len(chosen_human) != 8 or len(chosen_text) != 8:
        raise ValueError('Could not form fixed source microbatches')
    replay_path = OUTPUTS / 'vlm_hard_visual_sft_v1/data/UNIFORM_TRAIN.json'
    replay = read(replay_path)
    replay_rows = {q['ancestor_question_id']: {**q, 'question_id': q['ancestor_question_id']}
                   for q in replay['splits']['train_all'] if q['study_source'] == 'text'}
    selected = {q: human[q] for q in PROBES + chosen_human}
    selected.update({q: replay_rows[q] for q in chosen_text})
    images = {row['image_id']: data['images'][row['image_id']] for q, row in selected.items() if q in human}
    for q in chosen_text:
        row = selected[q]
        image = dict(replay['images'][row['image_id']])
        image['path'] = str((replay_path.parent / image['path']).resolve(strict=True))
        images[row['image_id']] = image
    for image in images.values():
        if not Path(image['path']).is_absolute() or not Path(image['path']).is_file():
            raise ValueError('Need existing original image paths')
    frozen = read(FIXED / 'data/COHORT.json')
    cohort = {
        'scope': 'train-only local gradient probe, not training or capability evaluation',
        'groups': {'probe': PROBES, 'ordinary_human': chosen_human, 'text': chosen_text},
        'qa': selected, 'images': images,
        'probe_review': [
            {'question_id': PROBES[0], 'support': 'Original red candle cups show no grouped stripes; original no supported.',
             'method': 'assistant original-image inspection in preceding diagnosis'},
            {'question_id': PROBES[1], 'support': 'One orange cat and its mirror reflection; not two distinct cats.',
             'method': 'assistant original-image inspection before GPU selection freeze'},
            {'question_id': PROBES[2], 'support': 'Visible buses share yellow bodies and red stripes/roofs.',
             'method': 'assistant original-image inspection in preceding diagnosis'},
            {'question_id': PROBES[3], 'support': 'Blue board and handle beneath carrots/wooden board are visible.',
             'method': 'assistant original-image inspection before GPU selection freeze'},
        ],
        'models': {role: frozen['models'][role] for role in ('initial', 'ce')},
        'model_root': frozen['model_root'], 'reference_config': frozen['reference_config'],
        'protocol': str(PACKAGE / 'docs/VLM_GRADIENT_MECHANISM_V1_PROTOCOL.md'),
        'numerics': {'base': 'bfloat16', 'adapter_master': 'float32', 'loss': 'float32',
                     'global_autocast': False, 'dropout': 0, 'tf32': False,
                     'gradient_checkpointing_use_reentrant': False, 'head': 'historical_tail'},
        'standard_weights': {'probe_no_answer': 0, 'probe_yes_answer': 0,
                             'human_ce': .5, 'human_answer': 0, 'text_ce': .5},
        'answer3_weights': {'probe_no_answer': 0, 'probe_yes_answer': 0,
                            'human_ce': .5, 'human_answer': 1 / 3, 'text_ce': .5},
        'maximum_worker_seconds': 600, 'optimizer_updates': 0,
    }
    write(ROOT / 'COHORT.json', cohort)
    sources = [manifest_path, DATA / 'ADMISSION_POLICY.json', DATA / 'PREPARATION_RESULT.json',
               schedule_path, replay_path, FIXED / 'data/COHORT.json',
               Path(cohort['reference_config']['path']), Path(cohort['protocol']), Path(__file__),
               PACKAGE / 'src/vlm_gradient_mechanism_v1/components.py',
               PACKAGE / 'src/vlm_gradient_mechanism_v1/metrics.py',
               PACKAGE / 'src/vlm_runtime_metadata_v1/model.py',
               PACKAGE / 'src/vlm_fp32_adapter/model.py',
               PACKAGE / 'src/vlm_deep/common.py', PACKAGE / 'src/vlm_sft/common.py',
               PACKAGE / 'src/vlm_probe/evaluate.py', ROOT / 'COHORT.json']
    for model in cohort['models'].values():
        sources.extend([Path(model['adapter_weights']['path']), Path(model['adapter_config']['path'])])
    write(ROOT / 'INPUT_LOCK.json', {'files': [{'path': str(p), 'sha256': sha(p)} for p in sources],
                                   'scope': 'fixed train-only source-gradient diagnostic; zero optimizer updates'})
    print(json.dumps({'status': 'prepared', 'group_qa': {k: len(v) for k, v in cohort['groups'].items()},
                      'models': list(cohort['models']), 'optimizer_updates': 0}))


def verify():
    for record in read(ROOT / 'INPUT_LOCK.json')['files']:
        if sha(record['path']) != record['sha256']:
            raise ValueError('Frozen diagnostic input changed: ' + record['path'])


def worker():
    import torch
    from vlm_deep.common import collate, encode, to_device
    from vlm_fp32_adapter.model import restore_fp32_adapter
    from vlm_runtime_metadata_v1.model import load
    from vlm_human_sft.evaluate import lora_hash
    from vlm_probe.evaluate import processor_for
    from vlm_gradient_mechanism_v1.components import forward_components
    from vlm_gradient_mechanism_v1.metrics import capture_gradients, compose_gradients, summarize_gradients

    verify()
    cohort = read(ROOT / 'COHORT.json')
    pending = [r for r in cohort['models'] if not (ROOT / 'runs' / r / 'COMPLETE.json').exists()]
    if not pending:
        write(ROOT / 'WORKER_COMPLETE.json', {'status': 'completed', 'optimizer_updates': 0})
        return
    started = time.monotonic()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda:0')
    first = cohort['models'][pending[0]]
    model, binding = load(cohort['model_root'], first['adapter'], device, training=True,
                          dropout=0., gradient_checkpointing=True)
    processor = processor_for(Path(cohort['model_root']), read(cohort['reference_config']['path']), 1024)
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    params = [p for _, p in named]
    if sum(p.numel() for p in params) != 1843200:
        raise ValueError('Unexpected trainable parameter budget')
    records = []
    for role in pending:
        entry = cohort['models'][role]
        if role != pending[0]:
            restore_fp32_adapter(model, entry['adapter'])
        if lora_hash(model) != entry['lora_sha256']:
            raise ValueError('Checkpoint values differ')
        model.train()
        model.get_base_model().visual.eval()
        snapshots, batches = {}, []
        for group in ('probe', 'ordinary_human', 'text'):
            qids = cohort['groups'][group]
            for start in range(0, len(qids), 4):
                if time.monotonic() - started > cohort['maximum_worker_seconds']:
                    raise TimeoutError('Diagnostic budget reached before next microbatch')
                ids = qids[start:start + 4]
                selected = [cohort['qa'][str(q)] for q in ids]
                encoded = [encode(processor, q, Path(cohort['images'][q['image_id']]['path']), True)
                           for q in selected]
                for sample in encoded:
                    if int(sample['input_ids'].shape[-1]) > 2048 or int(sample['image_grid_thw'].prod()) // 4 > 1024:
                        raise ValueError('No input truncation allowed')
                batch = to_device(collate(encoded, processor.tokenizer.pad_token_id), device)
                with torch.autocast('cuda', enabled=False):
                    out = forward_components(model, batch)
                    if group == 'probe':
                        losses = {'probe_no_answer': out['first_answer_ce'][:2].mean(),
                                  'probe_yes_answer': out['first_answer_ce'][2:].mean()}
                    elif group == 'ordinary_human':
                        if not bool(out['counts'].eq(3).all()):
                            raise ValueError('Expected three-token binary human targets')
                        losses = {'human_ce': out['qa_ce'].mean(), 'human_answer': out['first_answer_ce'].mean()}
                    else:
                        losses = {'text_ce': out['qa_ce'].mean()}
                piece = {}
                for index, (name, value) in enumerate(losses.items()):
                    gradients = torch.autograd.grad(value, params, retain_graph=index < len(losses) - 1)
                    piece[name] = capture_gradients(named, gradients)
                    del gradients
                    if piece[name].missing_gradients:
                        raise ValueError('Expected connected gradients for every trainable adapter tensor')
                    snapshots.setdefault(name, []).append((len(ids) / len(qids), piece[name]))
                record = {'role': role, 'source': group, 'question_ids': ids,
                          'qa_ce': out['qa_ce'].detach().cpu().tolist(),
                          'first_answer_ce': out['first_answer_ce'].detach().cpu().tolist(),
                          'assistant_counts': out['counts'].tolist(),
                          'sequence_lengths': [int(x['input_ids'].shape[-1]) for x in encoded],
                          'visual_tokens': [int(x['image_grid_thw'].prod()) // 4 for x in encoded],
                          'assistant_target_ids': [x['labels'][x['labels'].ne(-100)].tolist() for x in encoded]}
                batches.append(record)
                progress = {'status': 'measuring_gradients', 'role': role, 'source': group,
                            'completed_microbatches_this_role': len(batches), 'expected_microbatches_this_role': 5,
                            'worker_elapsed_seconds': time.monotonic() - started, 'optimizer_updates': 0}
                write(ROOT / 'PROGRESS.json', progress)
                print(json.dumps(progress), flush=True)
                del losses, out, batch, encoded, piece, value
        groups = {name: compose_gradients({str(i): item for i, (_, item) in enumerate(parts)},
                   {str(i): w for i, (w, _) in enumerate(parts)}) for name, parts in snapshots.items()}
        result = {'role': role, 'standard_ce_direction': summarize_gradients(groups, cohort['standard_weights']),
                  'answer3_direction': summarize_gradients(groups, cohort['answer3_weights']),
                  'actual_batches': batches, 'lora_unchanged': lora_hash(model) == entry['lora_sha256'],
                  'trainable_parameters': sum(p.numel() for p in params),
                  'dropout_config': {k: float(v.lora_dropout) for k, v in model.peft_config.items()},
                  'worker_elapsed_seconds': time.monotonic() - started, 'optimizer_updates': 0,
                  'peak_allocated_gib': torch.cuda.max_memory_allocated() / 1024 ** 3}
        if not result['lora_unchanged'] or any(p.grad is not None for p in params):
            raise ValueError('Read-only autograd diagnostic unexpectedly changed weights/assigned parameter.grad')
        directory = ROOT / 'runs' / role
        write(directory / 'ANALYSIS.json', result)
        torch.save({name: item.values for name, item in groups.items()}, directory / 'gradient_vectors.pt')
        write(directory / 'COMPLETE.json', {'status': 'completed', 'role': role, 'optimizer_updates': 0,
                                          'lora_unchanged': True, 'analysis_sha256': sha(directory / 'ANALYSIS.json')})
        records.append(result)
        del groups, snapshots
    verify()
    write(ROOT / 'RUNTIME_BINDING.json', binding)
    write(ROOT / 'WORKER_COMPLETE.json', {'status': 'completed', 'worker_elapsed_seconds': time.monotonic() - started,
                                        'optimizer_updates': 0, 'new_generation': 0})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare()
        return
    if args.worker:
        worker()
        return
    verify()
    from research_runtime.queue import run_queue
    run_queue(ROOT / 'queue', [{'key': 'two_checkpoints',
        'command': [sys.executable, '-u', str(Path(__file__).resolve()), '--worker'],
        'completion_marker': str(ROOT / 'WORKER_COMPLETE.json')}],
        {'input_lock_sha256': sha(ROOT / 'INPUT_LOCK.json'), 'optimizer_updates': 0})
    results = {r: read(ROOT / 'runs' / r / 'ANALYSIS.json') for r in ('initial', 'ce')}
    write(ROOT / 'ANALYSIS.json', {'status': 'completed', 'results': results,
          'interpretation': 'Fixed-point, small-cohort local SGD directions, not historical Adam causality or capability gains',
          'optimizer_updates': 0, 'new_generation': 0, 'next_training_automatic': False})
    write(ROOT / 'COMPLETION.json', {'status': 'completed', 'optimizer_updates': 0, 'new_generation': 0,
                                    'analysis_path': str(ROOT / 'ANALYSIS.json'), 'new_training_gain_claim': False})
    print('COMPLETED fixed-checkpoint source-gradient diagnostic', flush=True)


if __name__ == '__main__':
    main()
