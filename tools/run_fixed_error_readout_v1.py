"""Fixed initial/Final inference diagnostics; no optimizer or confirmation data."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

from rarevlm_paths import PROJECT_ROOT, SOURCE_ROOT
from vlm_deep.common import content_sha, read_manifest, sha_file, write_json

ROOT = PROJECT_ROOT.parents[1]/'outputs/vlm_fixed_error_readout_v1'
ROLES = ('initial', 'ce', 'ce_rank')
TASKS = ('train_selected', 'screen')


def read(path):
    return json.loads(Path(path).read_text())


def prepare():
    from vlm_human_sft.evaluate import source_files
    cohort = read(ROOT/'data/COHORT.json')
    paths = [Path(__file__), PROJECT_ROOT/'tools/prepare_fixed_error_readout_v1.py',
        PROJECT_ROOT/'docs/VLM_FIXED_ERROR_READOUT_V1_PROTOCOL.md',
        PROJECT_ROOT/'tests/test_vlm_fixed_error_readout.py', ROOT/'data/COHORT.json']
    paths += list((SOURCE_ROOT/'vlm_fixed_error_readout').glob('*.py')) + source_files()
    paths += [Path(cohort['manifests'][task]['path']) for task in TASKS]
    paths += [Path(cohort['baseline_config']['path']), Path(cohort['reference_config']['path'])]
    paths += list(Path(cohort['model_root']).glob('*.json'))
    for role in ROLES:
        for field in ('adapter_weights', 'adapter_config', 'source_receipt'):
            paths.append(Path(cohort['models'][role][field]['path']))
        for source in cohort['generation_sources'][role].values():
            if source.get('reused'):
                paths.append(Path(source['path']))
    identity = {'files': {str(path): sha_file(path) for path in sorted(set(paths))},
        'base_weight_files': {str(path): {'bytes': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns}
            for path in sorted(Path(cohort['model_root']).glob('*.safetensors'))},
        'optimizer_updates': 0, 'canonical_boundary': 0., 'near_zero_marker_only': .125,
        'head_strategy': 'dense full prompt head; extract last logits afterward',
        'new_readout_qa': 2556, 'new_generation_qa': 680, 'no_threshold_fitting': True,
        'confirmation_access': False, 'readout_math_cpu_checks': 21}
    from vlm_object_train.train import lock_json, package_lock
    identity['packages'] = package_lock()
    lock_json(ROOT/'INPUT_LOCK.json', identity)
    print(json.dumps({k:v for k,v in identity.items() if k not in ('files','packages')}), flush=True)


def verify():
    binding = read(ROOT/'INPUT_LOCK.json')
    for path, expected in binding['files'].items():
        if sha_file(path) != expected:
            raise ValueError(f'Diagnostic inputs changed: {path}')
    for path, expected in binding['base_weight_files'].items():
        stat = Path(path).stat()
        if {'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns} != expected:
            raise ValueError('Frozen base model storage changed')


def token_aliases(tokenizer):
    result = {}; skipped = {}
    for word in ('yes', 'no'):
        values = [word, word.capitalize(), word.upper()]
        values += [' '+value for value in list(values)]
        single = {}
        for value in values:
            ids = tokenizer.encode(value, add_special_tokens=False)
            if len(ids) == 1:
                single[value] = ids[0]
            else:
                skipped[value] = ids
        result[word] = {'ids': sorted(set(single.values())), 'spellings': single}
        canonical = tokenizer.encode(word, add_special_tokens=False)
        if len(canonical) != 1:
            raise ValueError('Canonical binary answers must each be one token')
        result[word]['canonical'] = canonical[0]
    if set(result['yes']['ids']) & set(result['no']['ids']):
        raise ValueError('Alias sets overlap')
    return {'groups': result, 'skipped_multitoken_spellings': skipped}


def cached_rows(path, original):
    if not path.exists():
        return []
    lines = path.read_bytes().splitlines(keepends=True); rows = []; sizes = []
    for index, line in enumerate(lines):
        try:
            value = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            if index != len(lines)-1:
                raise
            break
        rows.append(value); sizes.append(len(line))
    keep = len(rows)//4*4
    rows = rows[:keep]
    if [(r['question_id'], r['image_id']) for r in rows] != [(r['question_id'], r['image_id']) for r in original[:keep]]:
        raise ValueError('Cached prompt readout identities differ')
    with path.open('r+b') as stream:
        stream.truncate(sum(sizes[:keep]))
    return rows


def readout_task(model, processor, role, task, cohort, aliases):
    import torch
    from vlm_deep.common import collate, encode, to_device
    from vlm_probe.evaluate import inference_row, describe_input
    from vlm_human_sft.evaluate import binary_answer
    from vlm_object_train.train import lock_json
    from vlm_fixed_error_readout.readout import forward_first_readout
    manifest_path = Path(cohort['manifests'][task]['path'])
    manifest = read_manifest(manifest_path); original = manifest['splits']['dev']
    directory = ROOT/f'runs/{role}/{task}'; directory.mkdir(parents=True, exist_ok=True)
    lock_json(directory/'INPUT_LOCK.json', {'study_input_sha256': sha_file(ROOT/'INPUT_LOCK.json'),
        'role': role, 'task': task, 'aliases': aliases, 'manifest_sha256': sha_file(manifest_path),
        'expected_lora_sha256': cohort['models'][role]['lora_sha256'],
        'gpu_uuid': os.environ['CUDA_VISIBLE_DEVICES'], 'head_strategy': 'dense',
        'prompt_only': True, 'optimizer_updates': 0})
    path = directory/'readout.jsonl'; previous = cached_rows(path, original)
    if (directory/'COMPLETE.json').exists():
        if len(previous) != len(original):
            raise ValueError('Completed readout population is incomplete')
        return read(directory/'COMPLETE.json')
    model.eval(); device = next(model.parameters()).device
    image_token = processor.tokenizer.convert_tokens_to_ids('<|image_pad|>')
    yes, no = aliases['groups']['yes'], aliases['groups']['no']
    begun = time.monotonic(); completed = len(previous); start_count = completed
    with path.open('a') as stream, torch.inference_mode():
        for start in range(completed, len(original), 4):
            chunk = original[start:start+4]
            encoded = [encode(processor, inference_row(row), manifest_path.parent/manifest['images'][row['image_id']]['path'], False)
                       for row in chunk]
            details = [describe_input(sample, row, manifest['images'], processor, image_token, 1024)
                       for sample, row in zip(encoded, chunk)]
            batch = to_device(collate(encoded, processor.tokenizer.pad_token_id), device)
            with torch.autocast('cuda', enabled=False):
                values = forward_first_readout(model, batch, yes['ids'], no['ids'],
                    yes['canonical'], no['canonical'], slice_head=False)
            torch.cuda.synchronize(device)
            for row, value, evidence in zip(chunk, values, details):
                decoded = processor.tokenizer.decode([value['top1_token_id']], skip_special_tokens=False,
                    clean_up_tokenization_spaces=False)
                record = {'question_id': row['question_id'], 'image_id': row['image_id'],
                    'batch_start': start, 'image_sha256': manifest['images'][row['image_id']]['sha256'],
                    'question_sha256': content_sha(row['question']), **value, **evidence,
                    'top1_decoded': decoded, 'top1_binary': binary_answer(decoded),
                    'head_strategy': 'dense_prompt_last_position', 'teacher_forcing': False}
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+'\n')
            stream.flush(); os.fsync(stream.fileno()); completed += len(chunk)
            if completed % 32 == 0 or completed == len(original):
                elapsed = time.monotonic()-begun
                progress = {'status': 'reading_fixed_logits', 'role': role, 'task': task,
                    'completed_questions': completed, 'expected_questions': len(original),
                    'new_elapsed_seconds': elapsed, 'remaining_seconds_estimate':
                    elapsed/max(completed-start_count, 1)*(len(original)-completed), 'optimizer_updates': 0}
                write_json(directory/'PROGRESS.json', progress); print(json.dumps(progress), flush=True)
            del batch, encoded, values
    result = {'status': 'completed', 'role': role, 'task': task, 'qa': completed,
              'new_elapsed_seconds': time.monotonic()-begun, 'readout_sha256': sha_file(path),
              'optimizer_updates': 0, 'generation_score_replaced': False}
    write_json(directory/'COMPLETE.json', result); write_json(directory/'PROGRESS.json', result)
    return result


def worker(role):
    import torch
    from vlm_fp32_adapter.model import load
    from vlm_probe.evaluate import processor_for
    from vlm_human_sft.evaluate import evaluate, lora_hash
    verify(); cohort = read(ROOT/'data/COHORT.json'); reference = read(cohort['baseline_config']['path'])
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    model, _ = load(cohort['model_root'], cohort['models'][role]['adapter'], torch.device('cuda:0'), training=False)
    expected_hash = cohort['models'][role]['lora_sha256']
    if lora_hash(model) != expected_hash:
        raise ValueError('Fixed checkpoint values differ')
    processor = processor_for(Path(cohort['model_root']), read(cohort['reference_config']['path']), 1024)
    aliases = token_aliases(processor.tokenizer)
    write_json(ROOT/f'runs/{role}/ALIASES.json', aliases)
    results = {task: readout_task(model, processor, role, task, cohort, aliases) for task in TASKS}
    if role != 'initial':
        config = {**reference, 'initial_adapter': cohort['models'][role]['adapter'],
                  'human_manifest': cohort['manifests']['train_selected']['path'],
                  'expected_eval_questions': {'human': 340, 'text': 1617}}
        results['new_train_generation'] = evaluate(model, processor, config, 'human', ROOT/f'runs/{role}/train_generation')
    if lora_hash(model) != expected_hash:
        raise ValueError('Inference unexpectedly changed LoRA')
    write_json(ROOT/f'runs/{role}/COMPLETE.json', {'status': 'completed', 'role': role,
        'results': results, 'lora_unchanged': True, 'optimizer_updates': 0,
        'new_generation_qa': 0 if role == 'initial' else 340, 'confirmation_access': False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--role', choices=ROLES)
    args = parser.parse_args()
    if args.prepare:
        prepare(); return
    if args.role:
        worker(args.role); return
    from research_runtime.queue import run_queue
    verify()
    jobs = [{'key': role, 'command': [sys.executable, '-u', str(Path(__file__).resolve()), '--role', role],
             'completion_marker': str(ROOT/f'runs/{role}/COMPLETE.json')} for role in ROLES]
    run_queue(ROOT/'queue', jobs, {'input_lock_sha256': sha_file(ROOT/'INPUT_LOCK.json'),
              'optimizer_updates': 0, 'confirmation_access': False})
    from vlm_fixed_error_readout.analyze import analyze
    result = analyze(ROOT)
    write_json(ROOT/'COMPLETION.json', {'status': 'completed', 'optimizer_updates': 0,
               'analysis_path': str(ROOT/'ANALYSIS.json'), 'new_training_gain_claim': False,
               'confirmation_access': False, 'next': 'choose a targeted hypothesis from fixed-model error categories'})
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
