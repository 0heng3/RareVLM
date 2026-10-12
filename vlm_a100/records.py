"""Logical result index, hardware-aware summaries and mixed-runtime verification."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from vlm_deep.common import content_sha, read_manifest, sha_file, write_json
from vlm_deep.evaluate import aggregate, score_prediction
from vlm_deep.report import paired_interval


TASKS = ('dev', 'textvqa', 'docvqa', 'mmstar')
COMPARABILITY = (
    'Frozen scientific config, dataset, effective batch, loss and selection rule are shared. '
    'GPU architecture and physical rank/microbatch layout differ; stochastic trajectories '
    'and floating-point results need not be bitwise identical. Seed summaries include '
    'heterogeneous hardware and do not isolate hardware effects.'
)
TIMING_SCOPE = (
    'Training active_seconds includes checkpoint work and post-training internal dev NLL; '
    'evaluation A100 resources measure new work only and exclude model load and legacy cache. '
    'Do not combine these fields into a hardware speedup estimate.'
)


def _read(path):
    return json.loads(Path(path).read_text())


def build_index(run_root, config):
    """Enumerate protocol keys, selecting exactly one completed artifact per key."""
    root = Path(run_root).resolve()
    trainings, evaluations = [], []
    missing = {'trainings': [], 'evaluations': []}

    def register(kind, identity, candidates):
        candidates = [Path(p) for p in candidates]
        complete = [p for p in candidates if p.is_file()]
        if len(complete) > 1:
            raise ValueError(f'duplicate completed logical result {identity["id"]}: {complete}')
        if complete:
            target = trainings if kind == 'trainings' else evaluations
            target.append({**identity, 'path': str(complete[0])})
        else:
            missing[kind].append({**identity, 'candidate_paths': [str(p) for p in candidates]})

    for quantization in config['numerical_path_matrix']['base_weights']:
        for adapter in config['numerical_path_matrix']['adapter']:
            condition = f'{quantization}_{"base" if adapter == "none" else "pilot"}'
            register('evaluations', {'id': f'numerical/{condition}', 'stage': 'numerical',
                     'method': None, 'seed': None, 'task': 'textvqa', 'split': 'final',
                     'condition': condition, 'quantization': quantization, 'adapter': adapter},
                     [root / 'numerical' / condition / 'EVALUATION.json'])
    for task in ('textvqa', 'docvqa', 'mmstar'):
        register('evaluations', {'id': f'baselines/{task}', 'stage': 'baselines',
                 'method': None, 'seed': None, 'task': task, 'split': 'final'},
                 [root / 'baselines' / task / 'EVALUATION.json'])

    selection_path = root / 'STAGE2_SELECTION.json'
    selected = _read(selection_path)['selected_methods'] if selection_path.is_file() else []
    if selected and (len(selected) != 2 or len(set(selected)) != 2
                     or any(method not in config['methods'] for method in selected)):
        raise ValueError('invalid stage2 method selection')
    for stage, methods, split in [('stage1', list(config['methods']), 'train_10k'),
                                  ('stage2', selected, 'train_all')]:
        for method in methods:
            for seed in config['training_seeds']:
                directory = root / stage / method / f'seed{seed}'
                identity = {'id': f'{stage}/{method}/seed{seed}', 'stage': stage,
                            'method': method, 'seed': seed, 'split': split, 'task': None}
                register('trainings', identity, [directory / 'train' / 'TRAIN_RESULT.json'])
                for task in TASKS:
                    identity = {'id': f'{stage}/{method}/seed{seed}/{task}', 'stage': stage,
                                'method': method, 'seed': seed, 'task': task,
                                'split': 'dev' if task == 'dev' else 'final'}
                    register('evaluations', identity,
                             [directory / f'eval_{task}' / 'EVALUATION.json',
                              directory / f'eval_{task}_a100' / 'EVALUATION.json'])
    expected_train = (len(config['methods']) + 2) * len(config['training_seeds'])
    expected_eval = 4 + 3 + 4 * expected_train
    if not selected:
        missing['trainings'].append({'id': 'stage2_selection_pending',
                                    'count': 2 * len(config['training_seeds'])})
        missing['evaluations'].append({'id': 'stage2_selection_pending',
                                      'count': 8 * len(config['training_seeds'])})
    return {'schema_version': 'rarevlm.logical_run_index.v1', 'run_root': str(root),
            'config_sha256': content_sha(config), 'trainings': trainings,
            'evaluations': evaluations, 'missing': missing,
            'expected_training_runs': expected_train, 'expected_evaluations': expected_eval,
            'completed_training_runs': len(trainings), 'completed_evaluations': len(evaluations),
            'stage2_selected_methods': selected,
            'index_policy': 'explicit protocol keys; preflight and legacy partial artifacts excluded'}


def _training_runtime(result, config):
    compute = result.get('compute', {})
    return {'gpu_name': result.get('gpu_name'), 'world_size': result['world_size'],
            'microbatch': result['microbatch'], 'accumulation': result['accumulation'],
            'effective_batch': result['effective_batch'],
            'precision': compute.get('precision', config['precision']),
            'quantization': compute.get('training_quantization', config['training_quantization']),
            'cuda_matmul_allow_tf32': compute.get('cuda_matmul_allow_tf32'),
            'cudnn_allow_tf32': compute.get('cudnn_allow_tf32'),
            'tf32_record_scope': 'explicit runtime flags' if compute else
                'historical flags not stored; V100 has no TF32 execution',
            'hardware': result.get('hardware'),
            'runtime_layout': result.get('runtime_layout', {
                'world_size': result['world_size'], 'microbatch': result['microbatch'],
                'accumulation': result['accumulation'], 'effective_batch': result['effective_batch']}),
            'source_sha256': result['source_sha256'],
            'runtime_source_sha256': result.get('runtime_source_sha256'),
            'runtime_origin': 'a100_migration' if result.get('runtime_schema_version') else 'historical_v100'}


def _evaluation_runtime(result):
    counts = result.get('prediction_source_counts')
    if counts is None:
        counts = {'historical_v100': result['n']}
    return {'schema_version': result['schema_version'], 'precision': result['precision'],
            'quantization': result['quantization'], 'world_size': result['world_size'],
            'batch_size_per_rank': result['batch_size'],
            'runtime_layout': result.get('runtime_layout'),
            'source_sha256': result['source_sha256'],
            'migration_source_sha256': result.get('migration_source_sha256'),
            'prediction_source_counts': counts,
            'mixed_hardware_predictions': result.get('mixed_hardware_predictions', False),
            'new_work_resources': result.get('resources', []),
            'resource_history_files': result.get('resource_history_files', []),
            'timing_scope': TIMING_SCOPE}


def write_summary(asset_root, run_root, config_path, public_output=None):
    """Write a partial or final report from the canonical logical index."""
    root = Path(run_root).resolve()
    config = _read(config_path)
    index = build_index(root, config)
    write_json(root / 'RUN_INDEX.json', index)
    previous_path = root / 'SUMMARY.json'
    previous = _read(previous_path) if previous_path.is_file() else {}
    old_records = {(r['stage'], r['method'], r['seed'], r['task']): r
                   for r in previous.get('seed_records', [])}
    numerical, baselines = [], {}
    eval_by_key = {}
    for item in index['evaluations']:
        evaluation = _read(item['path'])
        if evaluation['config_sha256'] != index['config_sha256']:
            raise ValueError(f'summary scientific config mismatch: {item["id"]}')
        if item['stage'] == 'numerical':
            numerical.append({'condition': item['condition'], 'score_percent': evaluation['score_percent'],
                              'n': evaluation['n'], 'precision': evaluation['precision'],
                              'quantization': evaluation['quantization'],
                              'result_sha256': sha_file(item['path']),
                              'evaluation_runtime': _evaluation_runtime(evaluation)})
        elif item['stage'] == 'baselines':
            baselines[item['task']] = (item, evaluation)
        else:
            eval_by_key[(item['stage'], item['method'], item['seed'], item['task'])] = (item, evaluation)
    records, baseline_predictions = [], {}
    interval_keys = ('paired_question_mean_delta_pp', 'image_cluster_bootstrap_95_pp',
                     'clusters', 'questions', 'scope')
    for item in index['trainings']:
        train = _read(item['path'])
        if train['config_sha256'] != index['config_sha256']:
            raise ValueError(f'summary scientific config mismatch: {item["id"]}')
        for task in TASKS:
            key = (item['stage'], item['method'], item['seed'], task)
            if key not in eval_by_key:
                continue
            eval_item, evaluation = eval_by_key[key]
            runtime = _training_runtime(train, config)
            record = {'stage': item['stage'], 'method': item['method'], 'seed': item['seed'],
                      'task': task, 'score_percent': evaluation['score_percent'],
                      'question_mean_score_percent': evaluation['question_mean_score_percent'],
                      'n': evaluation['n'], 'images': evaluation['unique_images'],
                      'training_questions': train['question_count'], 'updates': train['completed_updates'],
                      'trainable_parameters': train['trainable_parameters'],
                      'training_active_seconds': train['active_seconds'],
                      'training_peak_allocated_gib': train['peak_gpu_allocated_gib'],
                      'frozen_weights_unchanged': train['frozen_parameters_unchanged'],
                      'train_result_sha256': sha_file(item['path']),
                      'evaluation_sha256': sha_file(eval_item['path']),
                      'training_path': item['path'], 'evaluation_path': eval_item['path'],
                      'gpu_name': runtime['gpu_name'], 'world_size': runtime['world_size'],
                      'microbatch': runtime['microbatch'], 'effective_batch': runtime['effective_batch'],
                      'precision': runtime['precision'], 'quantization': runtime['quantization'],
                      'cuda_matmul_allow_tf32': runtime['cuda_matmul_allow_tf32'],
                      'cudnn_allow_tf32': runtime['cudnn_allow_tf32'],
                      'training_runtime': runtime, 'evaluation_runtime': _evaluation_runtime(evaluation),
                      'hardware_comparability': COMPARABILITY, 'timing_scope': TIMING_SCOPE}
            if task == 'textvqa' and task in baselines:
                previous_record = old_records.get(key, {})
                if (previous_record.get('evaluation_sha256') == record['evaluation_sha256']
                        and all(k in previous_record for k in interval_keys)):
                    record.update({k: previous_record[k] for k in interval_keys})
                    record['paired_interval_origin'] = 'reused unchanged completed evaluation interval'
                else:
                    if task not in baseline_predictions:
                        baseline_predictions[task] = _read(Path(baselines[task][0]['path']).parent / 'predictions.json')
                    adapted = _read(Path(eval_item['path']).parent / 'predictions.json')
                    record.update(paired_interval(baseline_predictions[task], adapted))
                    record['paired_interval_origin'] = 'computed for this completed evaluation'
            records.append(record)
    grouped = defaultdict(list)
    for row in records:
        grouped[(row['stage'], row['method'], row['task'])].append(row)
    summaries = []
    for (stage, method, task), members in sorted(grouped.items()):
        scores = [r['score_percent'] for r in members]
        hardware = sorted({(r['gpu_name'], r['world_size'], r['microbatch']) for r in members})
        summaries.append({'stage': stage, 'method': method, 'task': task,
                          'seeds': [r['seed'] for r in members], 'n_runs': len(scores),
                          'mean_score_percent': statistics.mean(scores),
                          'seed_sd_pp': statistics.stdev(scores) if len(scores) > 1 else None,
                          'baseline_score_percent': baselines[task][1]['score_percent'] if task in baselines else None,
                          'selection_scope': 'dev only; all other tasks excluded from method selection',
                          'training_hardware_layouts': [list(v) for v in hardware],
                          'heterogeneous_training_hardware': len(hardware) > 1,
                          'hardware_comparability': COMPARABILITY})
    selection_path = root / 'STAGE2_SELECTION.json'
    status_path = root / 'STATUS.json'
    result = {'schema_version': 'rarevlm.vlm_deep.summary.a100.v1',
              'status': _read(status_path) if status_path.is_file() else {},
              'scope': 'Full public validation benchmarks; pretraining overlap unknown; fixed one-epoch counted-parameter comparisons.',
              'scientific_config_sha256': index['config_sha256'],
              'logical_run_index': {'path': str(root / 'RUN_INDEX.json'),
                  'completed_training_runs': index['completed_training_runs'],
                  'expected_training_runs': index['expected_training_runs'],
                  'completed_evaluations': index['completed_evaluations'],
                  'expected_evaluations': index['expected_evaluations'], 'missing': index['missing']},
              'numerical_path_matrix': numerical,
              'baseline_metrics': {k: {'score_percent': v[1]['score_percent'], 'n': v[1]['n'],
                  'images': v[1]['unique_images'], 'evaluation_runtime': _evaluation_runtime(v[1])}
                  for k, v in baselines.items()},
              'seed_records': records, 'summaries': summaries,
              'stage2_selection': _read(selection_path) if selection_path.is_file() else None,
              'hardware_comparability': COMPARABILITY, 'timing_scope': TIMING_SCOPE,
              'report_source_sha256': sha_file(Path(__file__)),
              'original_report_source_sha256': sha_file(Path(__file__).resolve().parents[1] / 'vlm_deep/report.py'),
              'weights_and_dataset_images_bundled': False}
    write_json(root / 'SUMMARY.json', result)
    if public_output:
        write_json(Path(public_output), result)
    if records:
        keys = sorted({key for row in records for key in row})
        with (root / 'seed_metrics.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            for row in records:
                writer.writerow({key: json.dumps(value, ensure_ascii=False, sort_keys=True)
                                 if isinstance(value, (dict, list)) else value
                                 for key, value in row.items()})
    return result


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _verify_sources(result):
    package = Path(__file__).resolve().parents[1]
    current = {str(p.relative_to(package / 'vlm_deep')): sha_file(p)
               for p in (package / 'vlm_deep').rglob('*.py')}
    _require(result['source_sha256'] == current, 'frozen original source changed')
    if result.get('runtime_schema_version'):
        expected = {name: sha_file(package / 'vlm_a100' / name)
                    for name in ('__init__.py', 'train.py')}
        _require(result.get('runtime_source_sha256') == expected, 'A100 training source changed')
    if result.get('migration_source_sha256'):
        _require(result['migration_source_sha256'] == {
            'vlm_a100/evaluate.py': sha_file(package / 'vlm_a100/evaluate.py')},
            'A100 evaluation source changed')


def _verify_selection(root, config):
    selection = _read(root / 'STAGE2_SELECTION.json')
    _require(selection.get('selection_uses_final_metrics') is False, 'selection must exclude final metrics')
    ranking = []
    for method in config['methods']:
        scores, nlls, hashes = [], [], {}
        for seed in config['training_seeds']:
            directory = root / 'stage1' / method / f'seed{seed}'
            train_path = directory / 'train/TRAIN_RESULT.json'
            candidates = [directory / 'eval_dev/EVALUATION.json', directory / 'eval_dev_a100/EVALUATION.json']
            completed = [p for p in candidates if p.is_file()]
            _require(len(completed) == 1, 'stage1 dev selection requires one evaluation per seed')
            scores.append(_read(completed[0])['score_percent'])
            nlls.append(_read(train_path)['internal_dev_token_nll'])
            hashes[str(seed)] = {'dev': sha_file(completed[0]), 'train': sha_file(train_path)}
        ranking.append({'method': method, 'mean_dev_score': statistics.mean(scores),
                        'mean_dev_nll': statistics.mean(nlls), 'input_hashes': hashes})
    ranking.sort(key=lambda row: (-row['mean_dev_score'], row['mean_dev_nll'], row['method']))
    _require(selection['selected_methods'] == [r['method'] for r in ranking[:2]], 'stage2 selection changed')
    stored = {r['method']: r for r in selection['dev_ranking']}
    for row in ranking:
        _require(stored[row['method']]['input_hashes'] == row['input_hashes'], 'dev selection source hash changed')
        for key in ('mean_dev_score', 'mean_dev_nll'):
            _require(math.isclose(stored[row['method']][key], row[key], abs_tol=1e-12), 'dev selection summary changed')


def verify(asset_root, run_root, config_path):
    """Verify every completed protocol key once, requiring the full 15/67 delivery."""
    asset_root, root = Path(asset_root).resolve(), Path(run_root).resolve()
    config = _read(config_path)
    index = build_index(root, config)
    write_json(root / 'RUN_INDEX.json', index)
    _require(index['completed_training_runs'] == index['expected_training_runs'],
             f'{index["completed_training_runs"]}/{index["expected_training_runs"]} training runs complete')
    _require(index['completed_evaluations'] == index['expected_evaluations'],
             f'{index["completed_evaluations"]}/{index["expected_evaluations"]} evaluations complete')
    data_path = asset_root / 'prepared/manifest.json'
    manifest = read_manifest(data_path, verify_images=True)
    for split in ('train_all', 'dev', 'final'):
        pixels = {manifest['images'][r['image_id']]['pixel_sha256'] for r in manifest['splits'][split]}
        for other in ('train_all', 'dev', 'final'):
            if other != split:
                _require(not pixels & {manifest['images'][r['image_id']]['pixel_sha256']
                                      for r in manifest['splits'][other]}, 'split pixel leakage')
    train_results = {}
    for item in index['trainings']:
        path = Path(item['path'])
        train = _read(path)
        _require(train['config_sha256'] == index['config_sha256'], 'training config changed')
        _require(train['manifest_sha256'] == manifest['manifest_sha256'], 'training manifest changed')
        _require((train['method'], train['seed'], train['split']) ==
                 (item['method'], item['seed'], item['split']), 'training logical identity changed')
        expected_questions = len(manifest['splits'][item['split']])
        _require(train['question_count'] == train['actual_questions_seen'] == expected_questions,
                 'incomplete or duplicated training question coverage')
        _require(train['completed_updates'] == train['planned_updates'] ==
                 math.ceil(expected_questions / config['effective_batch']), 'training update count changed')
        _require(train['trainable_parameters'] == config['methods'][item['method']]['expected_trainable'] ==
                 sum(train['trainable_names'].values()), 'trainable parameter budget changed')
        _require(train['frozen_parameter_sha256_before'] == train['frozen_parameter_sha256_after']
                 and train['frozen_parameters_unchanged'], 'frozen weights changed')
        _require(train['skipped_updates'] == 0 and train['epochs_completed'] == 1,
                 'training epoch incomplete or updates skipped')
        _require(train['effective_batch'] == config['effective_batch'] ==
                 train['world_size'] * train['microbatch'] * train['accumulation'], 'effective batch changed')
        _require(train['loss'] == config['loss'] and train['checkpoint_selection'] == config['checkpoint_selection'],
                 'loss/checkpoint rule changed')
        if train.get('runtime_schema_version'):
            _require(train.get('run_purpose') == 'research', 'preflight appeared in research index')
            _require((train['world_size'], train['microbatch'], train['accumulation']) == (2, 8, 1),
                     'unvalidated A100 training layout')
            compute = train['compute']
            _require(compute['precision'] == compute['vision_precision'] == 'fp32'
                     and compute['training_quantization'] == 'nf4_language_only'
                     and compute['cuda_matmul_allow_tf32'] is False and compute['cudnn_allow_tf32'] is False,
                     'A100 precision/TF32 changed')
        else:
            _require((train['world_size'], train['microbatch'], train['accumulation']) ==
                     (config['world_size'], config['training_microbatch'], config['training_accumulation']),
                     'historical training layout changed')
        _verify_sources(train)
        input_lock = _read(path.parent / 'TRAIN_INPUT_LOCK.json')
        _require(all(train.get(key) == value for key, value in input_lock.items()), 'training input lock changed')
        for name, digest in train['adapter_files_sha256'].items():
            _require(sha_file(path.parent / 'adapter' / name) == digest, 'trained adapter changed')
        latest_path = path.parent / 'LATEST.json'
        if latest_path.is_file():
            latest = _read(latest_path)
            checkpoint = path.parent / latest['directory']
            _require(sha_file(checkpoint / 'state.pt') == latest['state_sha256'], 'resume checkpoint changed')
            for name, digest in latest['adapter_files_sha256'].items():
                _require(sha_file(checkpoint / 'adapter' / name) == digest, 'resume adapter changed')
        train_results[(item['stage'], item['method'], item['seed'])] = train
    _verify_selection(root, config)
    transfers = {}
    for task in ('docvqa', 'mmstar'):
        path = asset_root / 'transfer' / task / 'manifest.json'
        transfer = _read(path)
        _require(content_sha({k: v for k, v in transfer.items() if k != 'manifest_sha256'}) ==
                 transfer['manifest_sha256'], 'transfer manifest changed')
        for image in transfer['images'].values():
            _require(sha_file(path.parent / image['path']) == image['sha256'], 'transfer image changed')
        transfers[task] = transfer
    total_predictions = 0
    hardware_counts = Counter()
    for item in index['evaluations']:
        path = Path(item['path'])
        evaluation = _read(path)
        task = 'textvqa' if item['task'] == 'dev' else item['task']
        _require(evaluation['config_sha256'] == index['config_sha256'], 'evaluation config changed')
        _require((evaluation['task'], evaluation['split']) == (task, item['split']), 'evaluation identity changed')
        _require(evaluation['precision'] == 'fp32', 'evaluation precision changed')
        _require(evaluation['generation'] == {'max_new_tokens': config['max_new_tokens'], 'do_sample': False,
                 'min_pixels': config['min_pixels'], 'max_pixels': config['max_pixels'],
                 'head_last_position_only': True}, 'generation protocol changed')
        _verify_sources(evaluation)
        input_lock = _read(path.parent / 'EVAL_INPUT_LOCK.json')
        _require(all(evaluation.get(key) == value for key, value in input_lock.items()), 'evaluation input lock changed')
        if item['stage'] in ('stage1', 'stage2'):
            trained = train_results[(item['stage'], item['method'], item['seed'])]
            _require(evaluation['adapter_files_sha256'] == trained['adapter_files_sha256'], 'evaluation adapter changed')
            _require(evaluation['model_provenance_sha256'] == trained['model_provenance_sha256'], 'model provenance changed')
            _require(evaluation['quantization'] == config['training_quantization'], 'trained evaluation quantization changed')
        elif item['stage'] == 'baselines':
            _require(evaluation['adapter_files_sha256'] is None and
                     evaluation['quantization'] == config['training_quantization'], 'baseline condition changed')
        else:
            _require(evaluation['quantization'] == item['quantization'], 'numerical quantization condition changed')
            _require((evaluation['adapter_files_sha256'] is None) == (item['adapter'] == 'none'), 'numerical adapter condition changed')
        data = manifest if task == 'textvqa' else transfers[task]
        references = data['splits'][item['split']] if task == 'textvqa' else data['records']
        _require(evaluation['manifest_sha256'] == data['manifest_sha256'], 'evaluation manifest changed')
        predictions_path = path.parent / 'predictions.json'
        _require(sha_file(predictions_path) == evaluation['predictions_sha256'], 'predictions artifact changed')
        rows = _read(predictions_path)
        lookup = {r['question_id']: r for r in references}
        _require(len(rows) == len(lookup) == evaluation['n'] and
                 [r['question_id'] for r in rows] == [r['question_id'] for r in references],
                 'incomplete, duplicate or reordered benchmark predictions')
        for row in rows:
            reference = lookup[row['question_id']]
            _require(row['image_id'] == reference['image_id'], 'prediction image identity changed')
            _require(row['question_sha256'] == content_sha(reference['question']) and
                     row['image_sha256'] == data['images'][row['image_id']]['sha256'], 'prediction input hashes changed')
            _require(math.isclose(row['score'], score_prediction(row['prediction'], reference, task),
                                 rel_tol=0, abs_tol=1e-12), 'stored task score changed')
        score, groups = aggregate(rows, task)
        _require(math.isclose(score, evaluation['score_percent'], rel_tol=0, abs_tol=1e-10), 'aggregate score changed')
        _require(math.isclose(100 * statistics.mean(r['score'] for r in rows),
                             evaluation['question_mean_score_percent'], rel_tol=0, abs_tol=1e-10), 'question mean changed')
        _require(groups == evaluation['category_scores_percent'], 'category scores changed')
        _require(evaluation['unique_images'] == len({r['image_id'] for r in references}), 'evaluation image count changed')
        if evaluation['schema_version'] == 'rarevlm.vlm_eval.a100.v1':
            runtime = evaluation['runtime_layout']
            _require(evaluation['evaluation_scope'] == 'full_benchmark' and evaluation['status'] == 'complete',
                     'partial/preflight evaluation indexed as complete')
            _require(runtime['world_size'] == evaluation['world_size'] == 2 and
                     runtime['dtype'] == 'fp32' and runtime['tf32_matmul'] is False and runtime['tf32_cudnn'] is False,
                     'A100 evaluation precision/layout changed')
            counts = dict(Counter(row['prediction_source']['kind'] for row in rows))
            _require(counts == evaluation['prediction_source_counts'], 'prediction hardware counts changed')
            _require(set(counts) <= {'generated_a100', 'reused_legacy'}, 'unknown prediction source')
            reuse = evaluation.get('reuse_source')
            if reuse:
                old = Path(reuse['directory'])
                _require(sha_file(old / 'EVAL_INPUT_LOCK.json') == reuse['lock_sha256'], 'legacy input lock changed')
                for name, digest in reuse['prediction_files_sha256'].items():
                    _require(sha_file(old / name) == digest, 'legacy partial artifact changed')
            hardware_counts.update(counts)
        else:
            hardware_counts['historical_v100'] += len(rows)
        total_predictions += len(rows)
    result = {'passed': True, 'schema_version': 'rarevlm.vlm_deep.verification.a100.v1',
              'completed_training_runs': len(index['trainings']),
              'completed_evaluations': len(index['evaluations']),
              'stored_predictions_rescored': total_predictions,
              'official_image_artifacts_verified': len(manifest['images']),
              'transfer_image_artifacts_verified': {k: len(v['images']) for k, v in transfers.items()},
              'full_textvqa_validation_questions': len(manifest['splits']['final']),
              'split_pixels_disjoint': True, 'frozen_weights_and_parameter_budgets_verified': True,
              'dev_only_stage2_selection_verified': True, 'original_and_runtime_sources_verified': True,
              'logical_duplicate_results_rejected': True, 'prediction_hardware_counts': dict(hardware_counts),
              'hardware_comparability': COMPARABILITY, 'timing_scope': TIMING_SCOPE,
              'config_sha256': index['config_sha256'], 'index_sha256': sha_file(root / 'RUN_INDEX.json'),
              'verification_source_sha256': sha_file(Path(__file__))}
    write_json(root / 'VERIFICATION.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['report', 'verify'])
    parser.add_argument('--asset-root', type=Path, required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1] / 'configs/vlm_deep_v2.json')
    parser.add_argument('--public-output', type=Path)
    args = parser.parse_args()
    if args.action == 'report':
        result = write_summary(args.asset_root, args.run_root, args.config, args.public_output)
        index = result['logical_run_index']
        print('REPORT', json.dumps({key: index[key] for key in (
            'completed_training_runs', 'expected_training_runs',
            'completed_evaluations', 'expected_evaluations')}, ensure_ascii=False), flush=True)
    else:
        print('VERIFICATION', json.dumps(verify(args.asset_root, args.run_root, args.config)), flush=True)


if __name__ == '__main__':
    main()
