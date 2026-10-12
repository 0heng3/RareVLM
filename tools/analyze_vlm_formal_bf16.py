"""CPU-only verification and analysis of the twelve frozen BF16/1024 cells.

Formal means complete public benchmark evaluation under a new dev-selected
engineering protocol. It does not imply blind/public-val independence or unknown
pretraining overlap. A running append-only cache produces progress only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
WORKSPACE = Path('/opt/data/private')
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tools.layout_paths import MIGRATION_PATH, verify_source
OUTPUTS = WORKSPACE / 'longtail_vlm/rarevlm_pilot/outputs'
DEFAULT_ASSET = OUTPUTS / 'vlm_deep_v2'
DEFAULT_ROOT = OUTPUTS / 'vlm_formal_bf16_v1'
DEFAULT_CONFIG = REPO / 'configs/vlm_formal_bf16_v1.json'
FONT_PATH = WORKSPACE / '.tools/fonts/NotoSansCJKsc-Regular.otf'
SEEDS = (0, 42, 200)
MODEL_SEEDS = {'base': None, 'qv_seed0': 0, 'qv_seed42': 42, 'qv_seed200': 200}
TASKS = ('textvqa', 'docvqa', 'mmstar')
COUNTS = {'textvqa': (5000, 3166), 'docvqa': (5349, 1285), 'mmstar': (1500, 1430)}
METRICS = {'textvqa': 'TextVQA leave-one-out consensus percent',
           'docvqa': 'DocVQA ANLS percent',
           'mmstar': 'MMStar 18-subcategory macro accuracy percent'}
TASK_LABELS = {'textvqa': 'TextVQA · VQA score', 'docvqa': 'DocVQA · ANLS',
               'mmstar': 'MMStar · 18子类宏平均'}
FREQUENCY_LABELS = ('unseen_0', 'rare_1_5', 'medium_6_20', 'frequent_gt20')
ENCODING_FIELDS = ('question_id', 'image_id', 'question_sha256', 'question_prompt_sha256',
                   'image_sha256', 'image_grid_thw', 'visual_tokens', 'counted_image_tokens',
                   'resized_width', 'resized_height', 'input_tokens', 'padded_input_tokens',
                   'input_truncated', 'batch_start', 'batch_size')
BATCH_FIELDS = ('batch_size', 'padded_input_tokens', 'batch_generation_seconds',
                'batch_preprocessing_seconds', 'batch_h2d_seconds',
                'batch_peak_allocated_gib', 'batch_peak_reserved_gib', 'timing_warmup')
COMMON_LOCK_FIELDS = ('packages', 'model_revision', 'model_config_sha256',
                      'model_provenance_sha256', 'model_weight_provenance_sha256',
                      'processor_and_metadata_sha256', 'source_sha256',
                      'reference_config_sha256', 'formal_config_sha256',
                      'precision', 'vision_precision', 'quantization', 'numerical_path',
                      'min_pixels', 'max_pixels', 'visual_token_budget', 'batch_size',
                      'max_new_tokens', 'do_sample', 'attention_implementation',
                      'head_last_position_only', 'input', 'input_truncation',
                      'tf32_matmul', 'tf32_cudnn', 'autocast')
INTERVAL_SCOPE = ('Pointwise exploratory 95% percentile interval conditional on one fixed '
    'base/checkpoint pair and this public benchmark. Resample entire image clusters and '
    'retain all paired questions. Nine predeclared task×seed comparisons, without '
    'multiplicity correction; no confirmatory significance or equivalence claim. '
    'Training-seed variability is separate; seed×image is never pooled as independent.')
LIMITATIONS = [
    'The dev-selected BF16/1024 protocol changes quantization/storage, compute dtype and visual budget relative to historical NF4/FP32/256. Historical changes are joint-path observations, not pure BF16 or pure budget causal effects.',
    'All adapters were trained using NF4/FP32 and visual budget256; BF16/1024 is an inference change with no new training.',
    'Training qv seed0 used V100, seeds42/200 used A100; microbatch/reduction/dropout execution differs. Three-seed SD includes this heterogeneous lineage and does not isolate hardware effects.',
    'Public val predictions were already measured under the historical protocol; benchmark pretraining overlap is unknown. Formal complete evaluation does not mean a clean unseen holdout.',
    'Answer text frequency is exposure to actual normalized supervised targets in train_all, not semantic visual-class frequency, pretraining frequency, or causally controlled long-tail difficulty.',
    'Slice scores are descriptive with no per-slice confidence intervals. Overall gains do not establish improvement on every unseen/rare slice.',
    'Sampled absence of a co-resident GPU process does not establish exclusivity. Throughput excludes model loading, preflight, verification, disk/controller overhead and is not production service performance.',
    'A high output agreement rate or an interval crossing zero does not prove numerical/accuracy equivalence; no equivalence margin was predeclared.',
]


def now():
    return datetime.now(timezone.utc).isoformat()


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def content_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    if not path.resolve().is_relative_to(WORKSPACE):
        raise ValueError(f'output outside persistent workspace: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def write_csv(path, rows):
    fields = list(rows[0]) if rows else []
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})


def load_cpu_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


NORMALIZER_PATH = REPO / "src" / 'vlm_deep/third_party/vqa_eval_metric.py'
SCORING_PATH = REPO / "src" / 'vlm_deep/scoring.py'
MCQ_PATH = REPO / "src" / 'vlm_deep/third_party/mcq_extract.py'


def scorers():
    normalizer = load_cpu_module('formal_bf16_normalizer', NORMALIZER_PATH).EvalAIAnswerProcessor()
    anls = load_cpu_module('formal_bf16_anls', SCORING_PATH).anls
    extract = load_cpu_module('formal_bf16_mcq', MCQ_PATH).extract_mcq_answer
    return normalizer, anls, extract


def score_prediction(prediction, reference, task, scoring):
    normalize, anls, extract = scoring
    if task == 'textvqa':
        if len(reference['answers']) != 10:
            raise ValueError('TextVQA consensus requires ten human answers')
        answer = normalize(prediction)
        refs = [normalize(str(value)) for value in reference['answers']]
        return sum(min(sum(answer == ref for j, ref in enumerate(refs) if i != j) / 3, 1)
                   for i in range(10)) / 10
    if task == 'docvqa':
        return anls(prediction, reference['answers'])
    if task == 'mmstar':
        return float(extract(prediction, choices=['A', 'B', 'C', 'D']) == reference['target'].strip().upper())
    raise ValueError(task)


def metric(rows, task):
    groups = defaultdict(list)
    if task == 'mmstar':
        for row in rows:
            groups[row['subcategory']].append(float(row['score']))
        if len(groups) != 18:
            raise ValueError('complete MMStar must retain all eighteen subcategories')
        category_scores = {name: 100 * statistics.mean(values) for name, values in sorted(groups.items())}
        return statistics.mean(category_scores.values()), category_scores
    return 100 * statistics.mean(float(row['score']) for row in rows), {}


def task_data(asset_root, task):
    path = asset_root / ('prepared/manifest.json' if task == 'textvqa' else f'transfer/{task}/manifest.json')
    manifest = read_json(path)
    if content_sha({key: value for key, value in manifest.items() if key != 'manifest_sha256'}) != manifest['manifest_sha256']:
        raise ValueError(f'{task} manifest content hash differs')
    rows = manifest['splits']['final'] if task == 'textvqa' else manifest['records']
    if (len(rows), len({row['image_id'] for row in rows})) != COUNTS[task]:
        raise ValueError(f'{task} incomplete frozen manifest')
    if len({str(row['question_id']) for row in rows}) != len(rows):
        raise ValueError(f'{task} duplicate question IDs')
    if task == 'mmstar' and len({row['subcategory'] for row in rows}) != 18:
        raise ValueError('MMStar reference subcategories differ')
    if task == 'textvqa':
        previous = set()
        for split in ('train_all', 'dev', 'final'):
            images = {row['image_id'] for row in manifest['splits'][split]}
            if images & previous:
                raise ValueError('TextVQA train/dev/final image IDs overlap')
            previous |= images
    return {'path': path, 'manifest': manifest, 'rows': rows,
            'manifest_file_sha256': sha_file(path), 'images': manifest['images']}


def verify_sources(lock, hashes):
    source = lock.get('source_sha256')
    if not isinstance(source, dict) or not source:
        raise ValueError('source lock absent')
    for name, expected in source.items():
        receipt = verify_source(name, expected, allow_current=True)
        hashes[receipt['path']] = receipt['current_sha256']


def validate_lock(lock, model_key, task, data, config_path, hashes):
    seed = MODEL_SEEDS[model_key]
    expected = {'model_key': model_key, 'seed': seed, 'role': 'base' if seed is None else 'adapted',
                'task': task, 'split': 'final', 'precision': 'bf16', 'vision_precision': 'bf16',
                'quantization': 'original', 'numerical_path': 'original_bf16',
                'visual_token_budget': 1024, 'min_pixels': 64 * 784, 'max_pixels': 1024 * 784,
                'batch_size': 4, 'max_new_tokens': 32, 'do_sample': False,
                'tf32_matmul': False, 'tf32_cudnn': False,
                'attention_implementation': 'sdpa', 'input_truncation': False,
                'expected_questions': COUNTS[task][0], 'expected_images': COUNTS[task][1],
                'source_split': 'official_benchmark' if task == 'mmstar' else 'official_validation',
                'manifest_sha256': data['manifest']['manifest_sha256'],
                'manifest_file_sha256': data['manifest_file_sha256'],
                'formal_config_sha256': sha_file(config_path),
                'head_last_position_only': True, 'autocast': False,
                'adapter_autocast_dtype': False,
                'explicit_lora_bf16_conversion': seed is not None,
                'input': 'declared image+question+suffix only; multiple-choice options retained; no target, answers or OCR input',
                'model_revision': '66285546d2b821cf421d4f5eb2576359d3770cd3'}
    for key, value in expected.items():
        if lock.get(key) != value:
            raise ValueError(f'unexpected {model_key}/{task} lock: {key}={lock.get(key)!r}, expected={value!r}')
    if lock.get('metric') != METRICS[task]:
        raise ValueError(f'{model_key}/{task} primary metric label differs')
    if lock.get('question_order_sha256') != content_sha([row['question_id'] for row in data['rows']]):
        raise ValueError('question order binding differs')
    suffix = '\nAnswer the question using a single word or phrase.'
    if lock.get('question_prompt_order_sha256') != content_sha([content_sha(row['question'] + row.get('prompt_suffix', suffix)) for row in data['rows']]):
        raise ValueError('question prompt order binding differs')
    subcategories = sorted({row['subcategory'] for row in data['rows']}) if task == 'mmstar' else []
    if lock.get('subcategories') != subcategories:
        raise ValueError('task subcategory taxonomy differs')
    config = read_json(config_path)
    reference_path = REPO / config['reference_config']
    reference = read_json(reference_path)
    if lock.get('reference_config_sha256') != sha_file(reference_path):
        raise ValueError('reference configuration file binding differs')
    if seed is None:
        if lock.get('adapter_files_sha256') is not None or lock.get('training_binding') is not None:
            raise ValueError('base unexpectedly contains an adapter/training binding')
    else:
        binding = lock.get('training_binding') or {}
        if (binding.get('method'), binding.get('seed'), binding.get('split'), binding.get('completed_updates')) != ('qv_r8', seed, 'train_all', 2062):
            raise ValueError('adapter is not the predeclared full qv_r8 checkpoint')
        if not lock.get('adapter_files_sha256'):
            raise ValueError('adapter content binding absent')
        adapter_dir = Path(config['fixed_adapter_dirs'][model_key])
        train_path = adapter_dir.parent / 'TRAIN_RESULT.json'
        if binding.get('path') != str(train_path) or binding.get('sha256') != sha_file(train_path):
            raise ValueError('completed training file binding differs')
        train = read_json(train_path)
        if (train['method'], train['seed'], train['split'], train['completed_updates'], train['planned_updates'], train['actual_questions_seen']) != ('qv_r8', seed, 'train_all', 2062, 2062, 32985):
            raise ValueError('stored training is not the fixed complete checkpoint')
        if not train['frozen_parameters_unchanged'] or train['config_sha256'] != content_sha(reference):
            raise ValueError('training scientific configuration/frozen parameters differ')
        if lock['adapter_files_sha256'] != train['adapter_files_sha256'] or lock['model_provenance_sha256'] != train['model_provenance_sha256']:
            raise ValueError('adapter or model provenance differs from completed training')
        actual_adapter_hashes = {}
        for path in sorted(adapter_dir.iterdir()):
            if path.is_file():
                if str(path) not in hashes:
                    hashes[str(path)] = sha_file(path)
                actual_adapter_hashes[path.name] = hashes[str(path)]
        if actual_adapter_hashes != lock['adapter_files_sha256']:
            raise ValueError('adapter file bytes differ')
        for field in ('gpu_name', 'hardware', 'runtime_layout', 'scientific_config_origin', 'checkpoint_selection'):
            expected_value = train.get(field, 'fixed final epoch' if field == 'checkpoint_selection' else None)
            if binding.get(field) != expected_value:
                raise ValueError(f'training lineage differs: {field}')
    dtypes = lock.get('actual_parameter_dtypes') or {}
    if dtypes.get('params4bit_tensors') != 0 or set(dtypes.get('parameter_tensors_by_dtype', {})) != {'bfloat16'}:
        raise ValueError('loaded parameters are not unquantized BF16')
    if seed is not None and set(dtypes.get('lora_tensors_by_dtype', {})) != {'bfloat16'}:
        raise ValueError('loaded LoRA tensors are not BF16')
    if not str((lock.get('hardware') or {}).get('uuid', '')).startswith('GPU-'):
        raise ValueError('physical GPU UUID binding absent')
    verify_sources(lock, hashes)


def read_cell(root, model_key, task, data, config_path, scoring, hashes):
    directory = root / 'runs' / model_key / task
    prediction_path, lock_path, evaluation_path = (directory / name for name in ('predictions.jsonl', 'INPUT_LOCK.json', 'EVALUATION.json'))
    lock = read_json(lock_path) if lock_path.exists() else None
    if lock:
        validate_lock(lock, model_key, task, data, config_path, hashes)
    raw = prediction_path.read_bytes() if prediction_path.exists() else b''
    lines = raw.splitlines(keepends=True)
    tail_ignored = bool(lines and not lines[-1].endswith(b'\n'))
    if tail_ignored:
        lines = lines[:-1]
    rows = [json.loads(line) for line in lines]
    if len(rows) > len(data['rows']) or (rows and lock is None):
        raise ValueError('invalid saved count or absent input lock')
    lock_sha = content_sha(lock) if lock else None
    source_sha = content_sha(lock['source_sha256']) if lock else None
    for index, row in enumerate(rows):
        reference = data['rows'][index]
        identity = {'question_id': str(reference['question_id']), 'image_id': str(reference['image_id']),
                    'question_sha256': content_sha(reference['question']),
                    'question_prompt_sha256': content_sha(reference['question'] + reference.get('prompt_suffix', '\nAnswer the question using a single word or phrase.')),
                    'image_sha256': data['images'][reference['image_id']]['sha256']}
        if any(str(row.get(key)) != value for key, value in identity.items()):
            raise ValueError(f'{model_key}/{task}/{index} question/image/prompt binding differs')
        expected = {'model_key': model_key, 'seed': MODEL_SEEDS[model_key], 'task': task,
                    'metric': METRICS[task], 'numerical_path': 'original_bf16',
                    'precision': 'bf16', 'quantization': 'original',
                    'visual_token_budget': 1024, 'input_lock_sha256': lock_sha,
                    'source_binding_sha256': source_sha, 'model_revision': lock['model_revision'],
                    'adapter_files_sha256': lock['adapter_files_sha256']}
        for key, value in expected.items():
            if row.get(key) != value:
                raise ValueError(f'{model_key}/{task}/{index} prediction contract differs: {key}')
        for key in ('subcategory', 'category'):
            if row.get(key, '') != reference.get(key, ''):
                raise ValueError(f'{task} category binding differs')
        grid = row['image_grid_thw']
        patches = math.prod(grid)
        visual_tokens = patches // 4
        if len(grid) != 3 or grid[0] != 1 or patches % 4 or not 0 < visual_tokens <= 1024:
            raise ValueError('unexpected visual grid or budget')
        if row['visual_tokens'] != visual_tokens or row['counted_image_tokens'] != visual_tokens:
            raise ValueError('actual visual token accounting differs')
        if (row['resized_height'], row['resized_width']) != (grid[1] * 14, grid[2] * 14):
            raise ValueError('actual resize and grid differ')
        if row['input_truncated'] or row['padded_input_tokens'] < row['input_tokens'] or row['input_exceeds_training_768'] != (row['input_tokens'] > 768):
            raise ValueError('input truncation/padding/training-length accounting differs')
        if row['input_tokens'] < visual_tokens or not 0 <= row['generated_nonpad_tokens'] <= 32:
            raise ValueError('input/generated token accounting differs')
        begin = index // 4 * 4
        if row['batch_start'] != begin or row['batch_size'] != min(4, len(data['rows']) - begin):
            raise ValueError('fixed question batch groups differ')
        score = score_prediction(row['prediction'], reference, task, scoring)
        if not math.isfinite(float(row['score'])) or not math.isclose(row['score'], score, abs_tol=1e-12, rel_tol=0):
            raise ValueError(f'{task} stored task score does not independently reproduce')
    evaluation = read_json(evaluation_path) if evaluation_path.exists() else None
    complete = evaluation is not None
    if complete:
        if lock is None or tail_ignored or len(rows) != len(data['rows']) or evaluation['n'] != len(rows):
            raise ValueError('completed evaluation has incomplete predictions')
        validate_lock(evaluation, model_key, task, data, config_path, hashes)
        if evaluation.get('status') != 'completed' or evaluation.get('input_lock_sha256') != lock_sha or evaluation.get('predictions_sha256') != hashlib.sha256(raw).hexdigest():
            raise ValueError('completed evaluation prediction/content binding differs')
        primary, categories = metric(rows, task)
        if not math.isclose(evaluation['score_percent'], primary, abs_tol=1e-10, rel_tol=0):
            raise ValueError('evaluation primary aggregate differs')
        if evaluation.get('subcategory_score_percent', {}) != categories:
            if set(evaluation.get('subcategory_score_percent', {})) != set(categories) or any(not math.isclose(evaluation['subcategory_score_percent'][name], score, abs_tol=1e-10, rel_tol=0) for name, score in categories.items()):
                raise ValueError('evaluation category aggregates differ')
        question_mean = 100 * statistics.mean(row['score'] for row in rows)
        if not math.isclose(evaluation['question_mean_score_percent'], question_mean, abs_tol=1e-10, rel_tol=0):
            raise ValueError('evaluation diagnostic question mean differs')
    cell = {'rows': rows, 'lock': lock, 'complete': complete, 'evaluation': evaluation}
    progress = {'model_key': model_key, 'seed': MODEL_SEEDS[model_key], 'task': task,
                'complete': complete, 'validated_saved_questions': len(rows),
                'expected_questions': len(data['rows']), 'incomplete_tail_ignored': tail_ignored}
    provenance = {'predictions_file': str(prediction_path), 'predictions_snapshot_sha256': hashlib.sha256(raw).hexdigest(),
                  'input_lock_file': str(lock_path), 'input_lock_sha256': lock_sha,
                  'evaluation_file': str(evaluation_path), 'evaluation_file_sha256': sha_file(evaluation_path) if complete else None}
    return cell, progress, provenance


def verify_common(cells):
    first = cells[('base', 'textvqa')]['lock']
    for (model_key, task), cell in cells.items():
        for field in COMMON_LOCK_FIELDS:
            if cell['lock'].get(field) != first.get(field):
                raise ValueError(f'common twelve-cell contract differs: {model_key}/{task}/{field}')
        within = cells[(model_key, 'textvqa')]['lock']
        for field in ('adapter_files_sha256', 'training_binding', 'actual_parameter_dtypes', 'hardware'):
            if cell['lock'].get(field) != within.get(field):
                raise ValueError(f'within-model checkpoint/numeric/hardware contract differs: {model_key}/{task}/{field}')
        base_rows = cells[('base', task)]['rows']
        for index, (base, row) in enumerate(zip(base_rows, cell['rows'])):
            if any(base[field] != row[field] for field in ENCODING_FIELDS):
                raise ValueError(f'same-task encoded inputs differ: {model_key}/{task}/{index}')


def distribution(values):
    array = np.asarray(values, dtype=float)
    return {'n': len(array), 'min': float(array.min()), 'mean': float(array.mean()),
            'median': float(np.median(array)), 'p05': float(np.quantile(array, .05)),
            'p95': float(np.quantile(array, .95)), 'max': float(array.max())}


def resource_monitor(root):
    path = root / 'RESOURCE_CONTENTION.json'
    raw = read_json(path) if path.exists() else None
    counts = {}
    if raw:
        for field in ('observations_by_model_key', 'observations_by_model', 'observations_by_role'):
            if field in raw:
                counts = raw[field]
                break
    status = {}
    for key in MODEL_SEEDS:
        count = counts.get(key)
        if count is not None and (not isinstance(count, int) or count < 0):
            raise ValueError('invalid resource contention count')
        status[key] = {'resource_timing_status': 'monitor_missing' if count is None else 'contended' if count else 'no_observed_contention',
                       'sampled_contention_observations': count, 'exclusive_gpu_benchmark': False}
    return {'file': str(path), 'file_sha256': sha_file(path) if raw else None,
            'monitor_snapshot': raw, 'models': status,
            'events_file': str(root / 'RESOURCE_CONTENTION.jsonl'),
            'scope': 'Conservatively tag a model if any sampled same-GPU co-resident compute process occurred. Sampled zero does not prove exclusivity; tasks are never preempted.'}


def cell_summary(model_key, task, cell, monitor):
    groups = defaultdict(list)
    for row in cell['rows']:
        groups[(task, row['session_id'], row['batch_start'])].append(row)
    batches = []
    for (task_id, session, start), members in sorted(groups.items()):
        head = members[0]
        if len(members) != head['batch_size'] or any(any(row[field] != head[field] for field in BATCH_FIELDS) for row in members[1:]):
            raise ValueError('partial/inconsistent completed timing batch')
        if head['padded_input_tokens'] != max(row['input_tokens'] for row in members):
            raise ValueError('batch left-padding length differs from actual inputs')
        for field in BATCH_FIELDS:
            if field != 'timing_warmup' and (not math.isfinite(float(head[field])) or float(head[field]) < 0):
                raise ValueError('non-finite/negative batch resource measurement')
        batches.append({'model_key': model_key, 'task': task_id, 'session_id': session,
                        'batch_start': start, **{field: head[field] for field in BATCH_FIELDS}})
    by_session = defaultdict(list)
    for batch in batches:
        by_session[batch['session_id']].append(batch)
    for members in by_session.values():
        first = min(batch['batch_start'] for batch in members)
        if any(bool(batch['timing_warmup']) != (batch['batch_start'] == first) for batch in members):
            raise ValueError('first-session-batch warmup rule differs')
    timed = [batch for batch in batches if not batch['timing_warmup']]
    n = sum(batch['batch_size'] for batch in timed)
    generation = sum(batch['batch_generation_seconds'] for batch in timed)
    preprocessing = sum(batch['batch_preprocessing_seconds'] for batch in timed)
    h2d = sum(batch['batch_h2d_seconds'] for batch in timed)
    if n <= 0 or generation <= 0:
        raise ValueError('no steady-state generation timing')
    primary, categories = metric(cell['rows'], task)
    rows = cell['rows']
    summary = {'model_key': model_key, 'seed': MODEL_SEEDS[model_key], 'task': task, 'metric': METRICS[task],
               'score_percent': primary, 'question_mean_score_percent': 100 * statistics.mean(row['score'] for row in rows),
               'subcategory_score_percent': categories, 'questions': len(rows),
               'unique_image_clusters': len({row['image_id'] for row in rows}),
               'training_binding': cell['lock'].get('training_binding'),
               'hardware': cell['lock']['hardware'], 'visual_tokens': distribution([row['visual_tokens'] for row in rows]),
               'input_tokens': distribution([row['input_tokens'] for row in rows]),
               'generated_nonpad_tokens_including_eos': distribution([row['generated_nonpad_tokens'] for row in rows]),
               'input_exceeds_training_768_questions': sum(row['input_tokens'] > 768 for row in rows),
               'all_batches': len(batches), 'sessions': sorted(by_session),
               'excluded_session_warmup_batches': len(batches) - len(timed),
               'timed_batches': len(timed), 'timed_questions': n,
               'generation_seconds_excluding_warmup': generation,
               'preprocessing_seconds_excluding_warmup': preprocessing, 'h2d_seconds_excluding_warmup': h2d,
               'pure_generation_questions_per_second': n / generation,
               'generation_preprocessing_h2d_questions_per_second': n / (generation + preprocessing + h2d),
               'mean_amortized_generation_seconds_per_question': generation / n,
               'peak_allocated_gib': max(batch['batch_peak_allocated_gib'] for batch in batches),
               'peak_reserved_gib': max(batch['batch_peak_reserved_gib'] for batch in batches),
               **monitor['models'][model_key],
               'timing_scope': 'Deduplicate (task,session_id,batch_start); exclude first batch in every task session from all steady-state phase totals. Peaks include every complete batch. Generation+preprocessing+H2D excludes loading/preflight/file/controller work. Batch amortization is not single-request latency; no production or exclusive-GPU claim.'}
    expected = cell['evaluation']['resources']['pure_generation_questions_per_second']
    if not math.isclose(summary['pure_generation_questions_per_second'], expected, rel_tol=1e-9):
        raise ValueError('worker throughput and independently deduplicated timing differ')
    return summary, batches


def seed_summaries(records):
    output = []
    for task in TASKS:
        base = next(row for row in records if row['model_key'] == 'base' and row['task'] == task)
        members = sorted((row for row in records if row['seed'] is not None and row['task'] == task), key=lambda row: row['seed'])
        if tuple(row['seed'] for row in members) != SEEDS:
            raise ValueError('expected all three training seeds')
        scores = [row['score_percent'] for row in members]
        output.append({'task': task, 'metric': METRICS[task], 'base_score_percent': base['score_percent'],
                       'seeds': list(SEEDS), 'seed_scores_percent': scores, 'mean_score_percent': statistics.mean(scores),
                       'sample_sd_score_percent': statistics.stdev(scores),
                       'seed_adapter_minus_base_pp': [value - base['score_percent'] for value in scores],
                       'mean_adapter_minus_base_pp': statistics.mean(scores) - base['score_percent'],
                       'n_training_seeds': 3, 'seed_uncertainty': 'sample standard deviation (ddof=1), not confidence interval; shared base is evaluated once',
                       'heterogeneous_training_hardware': {str(row['seed']): row['training_binding'] for row in members}})
    return output


def paired_clusters(base, adapted, task, replicates, seed):
    categories = sorted({row['subcategory'] for row in base}) if task == 'mmstar' else ['']
    category_index = {value: index for index, value in enumerate(categories)}
    images = sorted({row['image_id'] for row in base})
    image_index = {value: index for index, value in enumerate(images)}
    sums = np.zeros((len(images), len(categories)), dtype=np.float64)
    counts = np.zeros_like(sums)
    for before, after in zip(base, adapted):
        if before['question_id'] != after['question_id'] or before['image_id'] != after['image_id']:
            raise ValueError('unpaired questions in image bootstrap')
        i = image_index[before['image_id']]
        j = category_index[before['subcategory']] if task == 'mmstar' else 0
        sums[i, j] += float(after['score']) - float(before['score'])
        counts[i, j] += 1
    observed = float(100 * np.mean(sums.sum(axis=0) / counts.sum(axis=0)))
    expected = metric(adapted, task)[0] - metric(base, task)[0]
    if not math.isclose(observed, expected, abs_tol=1e-10, rel_tol=0):
        raise ValueError('paired bootstrap estimator differs from primary benchmark metric')
    rng = np.random.default_rng(seed)
    estimates, missing = [], 0
    while len(estimates) < replicates:
        batch = min(64, replicates - len(estimates))
        sampled = rng.integers(0, len(images), size=(batch, len(images)))
        sampled_counts = counts[sampled].sum(axis=1)
        valid = (sampled_counts > 0).all(axis=1)
        missing += int((~valid).sum())
        if valid.any():
            sampled_sums = sums[sampled].sum(axis=1)
            estimates.extend((100 * np.mean(sampled_sums[valid] / sampled_counts[valid], axis=1)).tolist())
        if missing > max(100000, replicates * 100):
            raise ValueError('too many empty-category bootstrap replicates')
    return {'delta_pp': observed, 'image_cluster_bootstrap_95_pp': np.quantile(estimates, [.025, .975]).tolist(),
            'questions': len(base), 'unique_image_clusters': len(images), 'metric': METRICS[task],
            'bootstrap_replicates': replicates, 'bootstrap_seed': seed,
            'empty_category_replicates_redrawn': missing,
            'empty_category_policy': 'Reject/redraw any resample missing an official category; never replace empty category with zero or silently change macro denominator. This conditions MMStar intervals on all18 categories being represented.',
            'categories_retained': len(categories), 'interval_scope': INTERVAL_SCOPE}


def actual_target(row):
    counts = Counter(str(answer).strip() for answer in row['answers'] if str(answer).strip())
    if not counts:
        raise ValueError('empty raw modal supervised target')
    target = min(counts, key=lambda value: (-counts[value], value))
    if row.get('target') != target:
        raise ValueError('manifest target differs from frozen raw-stripped modal rule')
    return target


def bucket(count):
    return 'unseen_0' if count == 0 else 'rare_1_5' if count <= 5 else 'medium_6_20' if count <= 20 else 'frequent_gt20'


def prepare_slices(data, root, normalize):
    manifest = data['manifest']
    training = manifest['splits']['train_all']
    target_frequency, reference_frequency = Counter(), Counter()
    for row in training:
        target_frequency[normalize(actual_target(row))] += 1
        reference_frequency.update(value for value in {normalize(str(answer)) for answer in row['answers']} if value)
    if len(training) != 32985 or sum(target_frequency.values()) != len(training):
        raise ValueError('full SFT target exposure count differs')
    features = []
    for row in data['rows']:
        target = normalize(actual_target(row))
        reference_counts = Counter(value for value in (normalize(str(answer)) for answer in row['answers']) if value)
        representative = min(reference_counts, key=lambda value: (-reference_counts[value], value)) if reference_counts else ''
        count = target_frequency[target]
        reference_count = reference_frequency[representative]
        features.append({'question_id': str(row['question_id']), 'image_id': row['image_id'],
                         'question_sha256': content_sha(row['question']),
                         'image_sha256': data['images'][row['image_id']]['sha256'],
                         'sft_target_normalized': target, 'train_all_sft_target_qa_frequency': count,
                         'sft_target_answer_frequency': bucket(count),
                         'modal_normalized_reference': representative,
                         'train_all_unique_reference_qa_frequency': reference_count,
                         'reference_answer_frequency': bucket(reference_count)})
    lock = {'schema_version': 'rarevlm.formal_bf16.slices.v1', 'task': 'textvqa', 'split': 'final',
            'manifest_sha256': manifest['manifest_sha256'], 'manifest_file_sha256': data['manifest_file_sha256'],
            'purpose': 'Reference-derived descriptive grouping for saved public-validation outcomes. Never used to choose budget/checkpoint/crops or to form generation inputs.',
            'training_questions': len(training), 'training_split': 'train_all',
            'sft_target_frequency_definition': 'Normalize each actual supervised row.target verified against raw-stripped modal answer (lexical ties), increment once per train_all QA. Retain empty normalized target keys; each training QA contributes exactly once.',
            'reference_frequency_definition': 'Auxiliary coverage: normalize all training reference answers, remove empty strings, deduplicate within each QA, increment each remaining answer once. Eval representative is normalized modal nonempty reference, lexical ties.',
            'evaluation_target_rule': 'Normalize the public-validation raw modal target with same tie rule; look up train_all actual SFT-target QA exposure.',
            'bucket_rules': {'unseen_0': 'count == 0', 'rare_1_5': '1 <= count <= 5', 'medium_6_20': '6 <= count <= 20', 'frequent_gt20': 'count > 20'},
            'normalizer_source_sha256': sha_file(NORMALIZER_PATH),
            'training_target_source_sha256': sha_file(REPO / "src" / 'vlm_deep/common.py'),
            'target_frequencies': dict(sorted(target_frequency.items())),
            'reference_frequencies': dict(sorted(reference_frequency.items())),
            'features': features,
            'limitations': ['Exposure proxy only, not semantic visual-class or pretraining frequency.', 'Post-generation descriptive slices; no slice confidence intervals or selection.', 'Cannot control answer length, annotation ambiguity, image detail or difficulty.']}
    lock['slice_lock_sha256'] = content_sha(lock)
    path = root / 'SLICE_LOCK.json'
    if path.exists():
        if read_json(path) != lock:
            raise ValueError('formal slice lock changed; refusing implicit replacement')
    else:
        write_json(path, lock)
    return lock, path


def slice_summaries(cells, lock):
    rows = []
    for dimension in ('sft_target_answer_frequency', 'reference_answer_frequency'):
        groups = defaultdict(list)
        for index, feature in enumerate(lock['features']):
            groups[feature[dimension]].append(index)
        for label in FREQUENCY_LABELS:
            indices = groups[label]
            for key, seed in MODEL_SEEDS.items():
                scores = [cells[(key, 'textvqa')]['rows'][index]['score'] for index in indices]
                base_scores = [cells[('base', 'textvqa')]['rows'][index]['score'] for index in indices]
                rows.append({'task': 'textvqa', 'dimension': dimension, 'slice': label,
                             'model_key': key, 'seed': seed, 'questions': len(indices),
                             'unique_image_clusters': len({lock['features'][index]['image_id'] for index in indices}),
                             'score_percent': 100 * statistics.mean(scores) if scores else None,
                             'adapter_minus_base_pp': 100 * statistics.mean(a - b for a, b in zip(scores, base_scores)) if scores else None,
                             'uncertainty': 'descriptive point estimate, no per-slice confidence interval'})
    means = []
    for dimension in ('sft_target_answer_frequency', 'reference_answer_frequency'):
        for label in FREQUENCY_LABELS:
            members = [row for row in rows if row['dimension'] == dimension and row['slice'] == label and row['seed'] is not None]
            scores = [row['score_percent'] for row in members]
            deltas = [row['adapter_minus_base_pp'] for row in members]
            means.append({'task': 'textvqa', 'dimension': dimension, 'slice': label,
                          'questions': members[0]['questions'], 'unique_image_clusters': members[0]['unique_image_clusters'],
                          'seeds': list(SEEDS), 'seed_scores_percent': scores,
                          'mean_score_percent': statistics.mean(scores) if all(value is not None for value in scores) else None,
                          'sample_sd_score_percent': statistics.stdev(scores) if all(value is not None for value in scores) else None,
                          'mean_adapter_minus_base_pp': statistics.mean(deltas) if all(value is not None for value in deltas) else None,
                          'uncertainty': 'sample SD across three seed-level slice scores only; no image sampling CI or significance claim'})
    return rows, means


def historical_results(asset_root):
    root = asset_root / 'runs'
    index_path = root / 'RUN_INDEX.json'
    if not index_path.exists():
        return [], {'available': False, 'file': str(index_path)}
    index = read_json(index_path)
    rows = []
    for item in index['evaluations']:
        selected = item['task'] in TASKS and ((item.get('stage') == 'stage2' and item.get('method') == 'qv_r8') or item.get('stage') == 'baselines')
        if not selected:
            continue
        path = Path(item['path'])
        evaluation = read_json(path)
        if evaluation.get('precision') != 'fp32' or evaluation.get('quantization') != 'nf4_language_only':
            raise ValueError('historical selected path differs from NF4/FP32')
        generation = evaluation['generation']
        if generation['max_pixels'] != 256 * 784:
            raise ValueError('historical visual budget differs')
        rows.append({'protocol': 'historical_vlm_deep_v2_NF4_FP32_256', 'task': item['task'],
                     'stage': item.get('stage'), 'method': item.get('method'), 'seed': item.get('seed'),
                     'score_percent': evaluation['score_percent'], 'questions': evaluation['n'],
                     'evaluation_file': str(path), 'evaluation_sha256': sha_file(path),
                     'scope': 'Separate historical protocol table. It changes numerical path and visual budget jointly; cannot attribute the observed difference to BF16 alone.'})
    return rows, {'available': bool(rows), 'index_file': str(index_path), 'index_sha256': sha_file(index_path),
                  'comparison_scope': 'No historical throughput speedup or extra bootstrap tests; separate metric columns/protocol only.'}


def make_figures(output, records, summaries, slices, slice_means, font_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties, fontManager
    fontManager.addfont(str(font_path))
    font_name = FontProperties(fname=str(font_path)).get_name()
    colors = ['#0072B2', '#D55E00', '#009E73']
    markers = ['o', '^', 's']
    files = []
    with plt.rc_context({'font.family': font_name, 'axes.unicode_minus': False, 'svg.fonttype': 'path',
                         'font.size': 10, 'figure.facecolor': 'white', 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.facecolor': 'white'}):
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), layout='constrained')
        for ax, summary in zip(axes, summaries):
            ax.scatter([0], [summary['base_score_percent']], marker='D', color='#555555', s=50, label='底座一次')
            for i, score in enumerate(summary['seed_scores_percent']):
                ax.scatter([1 + (i - 1) * .12], [score], color=colors[i], marker=markers[i], s=45, label=f'seed {SEEDS[i]}')
            ax.errorbar([1], [summary['mean_score_percent']], yerr=[summary['sample_sd_score_percent']], color='black', capsize=5, fmt='_', markersize=18, linewidth=1.5)
            ax.set_xticks([0, 1], ['底座', 'q/v rank8'])
            ax.set_xlim(-.4, 1.4)
            ax.set_title(TASK_LABELS[summary['task']], loc='left', weight='bold')
            ax.set_ylabel('该任务主指标（%）')
            ax.grid(axis='y', alpha=.25)
            ax.annotate(f"{summary['mean_score_percent']:.3f}±{summary['sample_sd_score_percent']:.3f}\nΔ {summary['mean_adapter_minus_base_pp']:+.3f} pp", xy=(1, summary['mean_score_percent']), xytext=(0, 22), textcoords='offset points', ha='center', fontsize=9)
            low = min(summary['base_score_percent'], *summary['seed_scores_percent']) - 2
            high = max(summary['base_score_percent'], summary['mean_score_percent'] + summary['sample_sd_score_percent'], *summary['seed_scores_percent']) + 4
            ax.set_ylim(max(0, low), min(100, high))
        axes[0].legend(fontsize=8, loc='lower right')
        fig.suptitle('正式三任务复核 · 原权重BF16 / 1024视觉token上限', weight='bold', fontsize=14)
        fig.supxlabel('每个彩色点为一个既有训练seed；误差棒为三seed样本SD，非置信区间。各任务指标不合并。', fontsize=9)
        for ext in ('png', 'svg'):
            path = output / f'formal_quality.{ext}'; fig.savefig(path, dpi=180); files.append(path)
        plt.close(fig)
        fig, axes = plt.subplots(2, 3, figsize=(13, 7.2), layout='constrained')
        for col, task in enumerate(TASKS):
            subset = [next(row for row in records if row['task'] == task and row['model_key'] == key) for key in MODEL_SEEDS]
            labels = ['base', 'seed0', 'seed42', 'seed200']
            for index, row in enumerate(subset):
                color = '#555555' if index == 0 else colors[index - 1]
                marker = 'D' if index == 0 else markers[index - 1]
                axes[0, col].scatter([index], [row['pure_generation_questions_per_second']], marker=marker, color=color, s=48)
                axes[1, col].scatter([index], [row['peak_allocated_gib']], marker=marker, color=color, s=48)
            for ax in axes[:, col]:
                ax.set_xticks(range(4), labels, rotation=20)
                ax.grid(axis='y', alpha=.25)
                ax.set_xlim(-.5, 3.5)
                ax.set_ylim(bottom=0)
            axes[0, col].set_title(TASK_LABELS[task], loc='left', weight='bold')
            axes[0, col].set_ylabel('纯生成吞吐（题/秒）')
            axes[1, col].set_ylabel('峰值allocated显存（GiB）')
        fig.suptitle('真实推理成本 · batch4 / 每次会话首batch排除于吞吐', weight='bold', fontsize=14)
        fig.supxlabel('记录资源竞争，采样未发现竞争不代表独占；不包含模型加载、预检与磁盘/controller耗时。', fontsize=9)
        for ext in ('png', 'svg'):
            path = output / f'formal_cost.{ext}'; fig.savefig(path, dpi=180); files.append(path)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(10, 5), layout='constrained')
        dimension = 'sft_target_answer_frequency'
        base = [next(row for row in slices if row['dimension'] == dimension and row['slice'] == label and row['model_key'] == 'base') for label in FREQUENCY_LABELS]
        means = [next(row for row in slice_means if row['dimension'] == dimension and row['slice'] == label) for label in FREQUENCY_LABELS]
        ax.scatter(np.arange(4) - .2, [row['score_percent'] for row in base], color='#555555', marker='D', s=45, label='底座一次')
        for i, seed in enumerate(SEEDS):
            scores = [next(row for row in slices if row['dimension'] == dimension and row['slice'] == label and row['seed'] == seed)['score_percent'] for label in FREQUENCY_LABELS]
            ax.scatter(np.arange(4) + .1 + .07 * (i - 1), scores, color=colors[i], marker=markers[i], s=45, label=f'seed {seed}')
        ax.errorbar(np.arange(4) + .1, [row['mean_score_percent'] for row in means], yerr=[row['sample_sd_score_percent'] for row in means], color='black', fmt='_', markersize=18, capsize=4)
        ax.set_xticks(range(4), [f'{label}\n{row["questions"]}题 / {row["unique_image_clusters"]}图' for label, row in zip(['0次', '1–5次', '6–20次', '>20次'], base)])
        ax.set_ylabel('TextVQA VQA score（%）')
        ax.set_ylim(0, 100); ax.grid(axis='y', alpha=.25); ax.legend(fontsize=9)
        ax.set_title('全量训练监督目标频次切片 · 三seed点与样本SD', loc='left', weight='bold')
        fig.supxlabel('train_all实际SFT目标规范化后按QA计数。频次代理不等同视觉类别或预训练长尾；没有切片CI，不作显著性判断。', fontsize=9)
        for ext in ('png', 'svg'):
            path = output / f'formal_textvqa_frequency.{ext}'; fig.savefig(path, dpi=180); files.append(path)
        plt.close(fig)
    write_json(output / 'FIGURE_PROVENANCE.json', {
        'audience': 'project learning and review', 'journal_submission': None,
        'raw_data': ['cell_metrics.csv', 'seed_summary.csv', 'textvqa_frequency_slices.csv', 'textvqa_frequency_seed_summary.csv'],
        'uncertainty': 'sample SD across three seed-level metric values; image-cluster CIs only in paired_contrasts.csv, never depicted as seed uncertainty',
        'transformations': ['metric-specific full-question aggregation; MMStar18-category macro', 'arithmetic mean and ddof1 SD across three training seeds', 'fixed train_all target-frequency bins', 'deduplicated task/session/batch timings with warmup excluded'],
        'missing_data': 'empty slice is missing, not zero; plots generated only after all12 cells verified complete',
        'font': str(font_path), 'svg_font_policy': 'glyph paths preserve Chinese rendering',
        'software': {'numpy': np.__version__, 'matplotlib': matplotlib.__version__},
        'software_reference': {'title': 'Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents', 'doi': '10.48550/arXiv.2609.00065'},
        'output_files_sha256': {path.name: sha_file(path) for path in files},
        'alt_text': 'Three panels compare the base once and three existing q/v rank8 seed checkpoints under BF16/1024 on TextVQA, DocVQA and MMStar. Cost panels show actual pure generation throughput and peak allocated memory separately. The frequency plot compares each seed with the base across four actual train_all supervised-target exposure bins; SD is between seeds, not image sampling uncertainty.'})
    return [str(path) for path in files]


def findings_text(analysis):
    lines = ['# 原权重BF16＋1024正式三任务复核', '',
             '这里的“正式”指冻结新协议后完整评测公开benchmark，并不表示预训练未见或独立盲测。', '',
             '| 任务 | 底座 | q/v三seed均值±样本SD | 相对底座 |',
             '|---|---:|---:|---:|']
    for row in analysis['seed_summaries']:
        lines.append(f"| {row['task']} | {row['base_score_percent']:.3f} | {row['mean_score_percent']:.3f}±{row['sample_sd_score_percent']:.3f} | {row['mean_adapter_minus_base_pp']:+.3f} pp |")
    lines += ['', 'MMStar主指标为18子类宏平均；三任务分数不合成总分。误差为三个既有训练seed的样本标准差，不是置信区间。', '',
              '图像簇配对区间分别条件于每个固定checkpoint。9项比较是探索性点区间，未做多重比较校正。MMStar重采样时保留图像整簇和18子类宏平均，任何缺失子类的replicate重抽并记录。', '',
              '训练seed0来自V100，seed42/200来自A100；这些模型均在NF4/FP32、256预算下训练，本轮仅改变推理路径，没有BF16训练加速证据。', '',
              'TextVQA频次切片由train_all实际监督目标重新构建，参考答案仅生成后用于计分和描述，未进入推理或选点。整体提高不能替代未见/罕见切片复核；切片没有图像采样CI。', '',
              '历史NF4/FP32/256保留独立表，不能把多因素新旧差异全部归因于BF16或预算。吞吐为会话首batch排除后的纯生成，另报告预处理/H2D/生成合计；没有线上服务或严格独占测速声明。', '',
              '## 限制', '']
    lines += [f'- {value}' for value in analysis['limitations']]
    lines += ['', '## 可复核产物', '',
              '- `ANALYSIS.json`：完整性校验、指标、配对区间、频次、资源竞争与输入来源。',
              '- `cell_metrics.csv` / `seed_summary.csv`：12单元与三种子独立汇总。',
              '- `paired_contrasts.csv`：三个任务×三个固定seed的同图配对区间。',
              '- `textvqa_frequency_slices.csv`：实际监督目标曝光与辅助参考覆盖频次分别报告。',
              '- `formal_quality.png`、`formal_cost.png`、`formal_textvqa_frequency.png`：真实数据图，CSV及provenance配套。', '',
              '软件方法参考：Scientific Agent Skills（K-Dense），https://doi.org/10.48550/arXiv.2609.00065。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset-root', type=Path, default=DEFAULT_ASSET)
    parser.add_argument('--output-root', '--run-root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--replicates', type=int, default=2000)
    parser.add_argument('--bootstrap-seed', type=int, default=20261011)
    parser.add_argument('--font', type=Path, default=FONT_PATH)
    parser.add_argument('--no-figures', action='store_true')
    parser.add_argument('--prepare-only', action='store_true', help='Freeze formal-final frequency slices without reading live predictions.')
    args = parser.parse_args()
    root = args.output_root.resolve()
    output = (args.output or root / 'analysis').resolve()
    if not root.is_relative_to(WORKSPACE) or not output.is_relative_to(WORKSPACE) or output.is_relative_to(root / 'runs'):
        raise ValueError('analysis must remain in persistent workspace outside the run tree')
    if args.replicates < 100:
        raise ValueError('insufficient bootstrap replicates')
    output.mkdir(parents=True, exist_ok=True)
    scoring = scorers()
    datasets = {task: task_data(args.asset_root, task) for task in TASKS}
    slices, slice_path = prepare_slices(datasets['textvqa'], root, scoring[0])
    if args.prepare_only:
        ready = {'schema_version': 'rarevlm.formal_bf16.analysis_ready.v1', 'ready_at': now(),
                 'analysis_source': str(Path(__file__).resolve()), 'analysis_source_sha256': sha_file(__file__),
                 'slice_lock_file': str(slice_path), 'slice_lock_sha256': slices['slice_lock_sha256'],
                 'questions': COUNTS, 'no_gpu': True,
                 'invocation': 'tools/analyze_vlm_formal_bf16.py --output-root <root> --config <config>',
                 'complete_requires': 'all12 fullcells score/identity/input/source/numerical/checkpoint contracts; perseed×task imagecluster bootstrap'}
        write_json(root / 'ANALYSIS_READY.json', ready)
        print(json.dumps({'prepared': True, 'slice_questions': len(slices['features']), 'analysis_source_sha256': ready['analysis_source_sha256']}))
        return 0
    if not args.config.exists():
        raise FileNotFoundError('formal configuration must be frozen before analysis')
    hashes = {str(args.config.resolve()): sha_file(args.config)}
    cells, progress, provenance = {}, [], {}
    for model_key in MODEL_SEEDS:
        for task in TASKS:
            cell, status, origin = read_cell(root, model_key, task, datasets[task], args.config, scoring, hashes)
            cells[(model_key, task)] = cell
            progress.append(status)
            provenance[f'{model_key}:{task}'] = origin
    complete = all(cell['complete'] for cell in cells.values())
    if not complete:
        partial = {'schema_version': 'rarevlm.formal_bf16.analysis.v1', 'updated_at': now(), 'complete': False,
                   'completed_cells': sum(cell['complete'] for cell in cells.values()), 'total_cells': 12,
                   'validated_saved_questions': sum(row['validated_saved_questions'] for row in progress),
                   'expected_questions': 4 * sum(value[0] for value in COUNTS.values()), 'progress': progress,
                   'scope': 'Partial checked progress only; no final seed statistics, effect claims or selection.',
                   'provenance': provenance, 'slice_lock_sha256': slices['slice_lock_sha256'],
                   'architecture_migration': {'path': str(MIGRATION_PATH), 'sha256': sha_file(MIGRATION_PATH),
                       'scope': 'Historical source digests remain original; current layout bytes verified through recorded migration.'}
                       if MIGRATION_PATH.exists() else None}
        write_json(output / 'ANALYSIS.json', partial)
        print(json.dumps({key: partial[key] for key in ('complete', 'completed_cells', 'validated_saved_questions', 'expected_questions')}))
        return 0
    verify_common(cells)
    monitor = resource_monitor(root)
    records, batches, paired = [], [], []
    for (model_key, task), cell in cells.items():
        summary, grouped = cell_summary(model_key, task, cell, monitor)
        records.append(summary); batches.extend(grouped)
    summaries = seed_summaries(records)
    for task_index, task in enumerate(TASKS):
        for seed_index, seed in enumerate(SEEDS):
            key = f'qv_seed{seed}'
            contrast = paired_clusters(cells[('base', task)]['rows'], cells[(key, task)]['rows'], task, args.replicates, args.bootstrap_seed + task_index * 10 + seed_index)
            paired.append({'task': task, 'model_key': key, 'training_seed': seed, **contrast})
    frequency_rows, frequency_means = slice_summaries(cells, slices)
    history, history_origin = historical_results(args.asset_root)
    figures = [] if args.no_figures else make_figures(output, records, summaries, frequency_rows, frequency_means, args.font)
    write_csv(output / 'cell_metrics.csv', records)
    write_csv(output / 'seed_summary.csv', summaries)
    write_csv(output / 'paired_contrasts.csv', paired)
    write_csv(output / 'timing_batches.csv', batches)
    write_csv(output / 'textvqa_frequency_slices.csv', frequency_rows)
    write_csv(output / 'textvqa_frequency_seed_summary.csv', frequency_means)
    write_csv(output / 'historical_protocol_metrics.csv', history)
    analysis = {'schema_version': 'rarevlm.formal_bf16.analysis.v1', 'updated_at': now(), 'complete': True,
                'completed_cells': 12, 'total_cells': 12, 'verified_prediction_questions': 4 * sum(value[0] for value in COUNTS.values()),
                'protocol': {'numerical_path': 'original_bf16', 'visual_token_budget': 1024,
                             'checkpoint_scope': 'base once + existing qv_r8 fulltrain seeds0/42/200; no new training',
                             'config_file': str(args.config.resolve()), 'config_sha256': sha_file(args.config)},
                'integrity_checks': ['all12 complete exact task QA/image counts', 'qid/image/question/prompt/manifest bindings',
                                     'independent task scoring and primary aggregation, MMStar18 macro', 'same-task actual encoded inputs across all4 models',
                                     'actual unquantized BF16 including LoRA', 'checkpoint/processor/revision/package/source/generation contracts',
                                     'timing deduplicated by task/session/batch, first-session batches excluded'],
                'cell_metrics': records, 'seed_summaries': summaries, 'paired_contrasts': paired,
                'textvqa_frequency_slices': frequency_rows, 'textvqa_frequency_seed_summaries': frequency_means,
                'slice_lock_file': str(slice_path), 'slice_lock_sha256': slices['slice_lock_sha256'],
                'resource_contention': monitor, 'historical_protocol_metrics': history,
                'historical_protocol_provenance': history_origin, 'provenance': provenance,
                'analysis_source_sha256': sha_file(__file__), 'source_files_rechecked': hashes,
                'architecture_migration': {'path': str(MIGRATION_PATH), 'sha256': sha_file(MIGRATION_PATH),
                    'scope': 'Historical source digests remain original; current layout bytes verified through recorded migration.'}
                    if MIGRATION_PATH.exists() else None,
                'dataset_provenance': {task: {'manifest_file': str(data['path']), 'manifest_file_sha256': data['manifest_file_sha256'], 'manifest_sha256': data['manifest']['manifest_sha256']} for task, data in datasets.items()},
                'statistics': {'seed_estimator': 'arithmetic mean ± sample SD across seeds0/42/200; no mean CI with n3',
                               'image_intervals': INTERVAL_SCOPE, 'replicates': args.replicates,
                               'multiplicity': 'nine pointwise exploratory comparisons, no adjustment and no confirmatory inference'},
                'limitations': LIMITATIONS, 'figures': figures}
    write_json(output / 'ANALYSIS.json', analysis)
    (output / 'FINDINGS.md').write_text(findings_text(analysis))
    print(json.dumps({'complete': True, 'completed_cells': 12, 'verified_prediction_questions': analysis['verified_prediction_questions'], 'seed_summaries': summaries}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
