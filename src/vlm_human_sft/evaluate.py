"""Fixed-profile full evaluation, sharing the tested loader, generator and cache."""
import argparse
import json
import os
from pathlib import Path
import re
import sys
import time

from rarevlm_paths import SOURCE_ROOT


def read(path):
    return json.loads(Path(path).read_text())


def binary_answer(text):
    match = re.match(r'^(yes|no)\b', text.strip(), flags=re.IGNORECASE | re.ASCII)
    return match.group(1).lower() if match else None


def strict_binary(text):
    return bool(re.fullmatch(r'(yes|no)[.,!?]?', text.strip(), flags=re.IGNORECASE | re.ASCII))


def primary_score(prediction, row, task):
    from vlm_deep.common import consensus
    if task == 'human':
        return float(binary_answer(prediction) == row['target'])
    if task != 'text':
        raise ValueError('Unknown task')
    return consensus(prediction, row['answers'])


def lora_hash(model):
    from vlm_object_train.train import tensor_hash
    return tensor_hash((name, parameter) for name, parameter in model.named_parameters() if 'lora_' in name)


def source_files():
    return [Path(__file__), SOURCE_ROOT/'vlm_fp32_adapter/model.py',
            SOURCE_ROOT/'vlm_probe/evaluate.py', SOURCE_ROOT/'vlm_deep/common.py',
            SOURCE_ROOT/'vlm_deep/evaluate.py', SOURCE_ROOT/'vlm_sft/common.py']


def evaluate(model, processor, config, task, directory):
    import torch
    from vlm_deep.common import consensus, content_sha, read_manifest, sha_file, write_json
    from vlm_probe.evaluate import complete_cache, evaluate_batch, package_versions
    from vlm_object_train.train import lock_json
    manifest_path = Path(config[task+'_manifest'])
    manifest = read_manifest(manifest_path)
    rows, images = manifest['splits']['dev'], manifest['images']
    if len(rows) != config['expected_eval_questions'][task]:
        raise ValueError('Incomplete evaluation population')
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    binding = {'source_files': {str(path): sha_file(path) for path in source_files()},
               'manifest_sha256': sha_file(manifest_path), 'lora_sha256': lora_hash(model),
               'gpu_uuid': os.environ['CUDA_VISIBLE_DEVICES'], 'packages': package_versions(),
               'task': task, 'budget': config['evaluation_budget'], 'batch_size': config['evaluation_batch_size'],
               'max_new_tokens': config['max_new_tokens'], 'numerics': 'BF16 base/FP32 LoRA/no global AMP',
               'binary_primary': 'first whole ASCII yes/no token, case-insensitive; otherwise wrong',
               'initialization': config['initial_adapter']}
    lock_json(directory/'INPUT_LOCK.json', binding)
    predictions_path = directory/'predictions.jsonl'
    predictions, _ = complete_cache(predictions_path, rows, images, config['evaluation_batch_size'])
    for row, prediction in zip(rows, predictions):
        if prediction['primary_score'] != primary_score(prediction['prediction'], row, task):
            raise ValueError('Cached primary score differs')
    if (directory/'COMPLETE.json').exists():
        if len(predictions) != len(rows):
            raise ValueError('Completed evaluation is incomplete')
        return read(directory/'COMPLETE.json')
    device = next(model.parameters()).device
    model.eval(); model.config.use_cache = True
    started, previous = time.monotonic(), len(predictions)
    image_token_id = processor.tokenizer.convert_tokens_to_ids('<|image_pad|>')
    with predictions_path.open('a') as stream, torch.inference_mode():
        for start in range(len(predictions), len(rows), config['evaluation_batch_size']):
            chunk = rows[start:start+config['evaluation_batch_size']]
            answers, resources = evaluate_batch(model, processor, chunk, images, manifest_path.parent, device,
                config['evaluation_budget'], image_token_id, config['max_new_tokens'])
            for row, answer in zip(chunk, answers):
                result = {'question_id': row['question_id'], 'image_id': row['image_id'],
                          'question_sha256': content_sha(row['question']),
                          'image_sha256': images[row['image_id']]['sha256'], **answer, **resources,
                          'batch_start': start, 'score': consensus(answer['prediction'], row['answers']),
                          'primary_score': primary_score(answer['prediction'], row, task),
                          'strict_binary_valid': strict_binary(answer['prediction']) if task == 'human' else None}
                stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False)+'\n')
                predictions.append(result)
            stream.flush(); os.fsync(stream.fileno())
            if len(predictions) % 32 == 0 or len(predictions) == len(rows):
                elapsed = time.monotonic()-started
                progress = {'status': 'evaluating', 'task': task, 'completed_questions': len(predictions),
                            'expected_questions': len(rows), 'new_elapsed_seconds': elapsed,
                            'remaining_seconds_estimate': elapsed/max(len(predictions)-previous, 1)*(len(rows)-len(predictions))}
                write_json(directory/'PROGRESS.json', progress); print(json.dumps(progress), flush=True)
    result = {'status': 'completed', 'task': task, 'qa': len(rows),
              'images': len({row['image_id'] for row in rows}),
              'score_percent': 100*sum(row['primary_score'] for row in predictions)/len(rows),
              'strict_invalid_qa': sum(not row['strict_binary_valid'] for row in predictions) if task == 'human' else None,
              'predictions_sha256': sha_file(predictions_path), 'lora_sha256': binding['lora_sha256'],
              'new_elapsed_seconds': time.monotonic()-started, 'goal_complete': False}
    write_json(directory/'PROGRESS.json', result); write_json(directory/'COMPLETE.json', result)
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--task', choices=['human', 'text'])
    args = parser.parse_args(); config = read(args.config)
    if args.task:
        import torch
        from vlm_fp32_adapter.model import load
        from vlm_probe.evaluate import processor_for
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        model, _ = load(config['model_root'], config['initial_adapter'], torch.device('cuda:0'), training=False)
        processor = processor_for(Path(config['model_root']), read(config['reference_config']), config['evaluation_budget'])
        evaluate(model, processor, config, args.task, Path(config['output_root'])/'initial'/args.task)
        return
    from research_runtime.queue import run_queue
    from vlm_deep.common import sha_file
    root = Path(config['output_root']).resolve()
    if not root.is_relative_to('/opt/data/private'):
        raise ValueError('Persistent output required')
    jobs = [{'key': task, 'command': [sys.executable, '-u', '-m', 'vlm_human_sft.evaluate',
              '--config', str(args.config.resolve()), '--task', task],
              'completion_marker': str(root/'initial'/task/'COMPLETE.json')} for task in ['human', 'text']]
    run_queue(root/'baseline_queue', jobs, {'config_sha256': sha_file(args.config),
              'source_sha256': {str(path): sha_file(path) for path in source_files()}})


if __name__ == '__main__':
    main()
