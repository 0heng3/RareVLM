"""Once-only seed42 POPE comparison with paired image-cluster uncertainty."""
from __future__ import annotations

import argparse
from collections import Counter
import math
from pathlib import Path

import numpy as np

from vlm_probe.evaluate import now, private_path, read_json, sha_file, write_json
from .scoring import metrics, official_prediction

CONDITIONS = ('initial', 'sft', 'dpo')
CONTRASTS = (('dpo_minus_sft', 'sft', 'dpo'),
    ('dpo_minus_initial', 'initial', 'dpo'), ('sft_minus_initial', 'initial', 'sft'))
METRICS = ('accuracy', 'f1', 'precision', 'recall', 'yes_rate')
BOOTSTRAP_REPLICATES = 6000
BOOTSTRAP_SEED = 20261015
PAIR_FIELDS = ('question_id', 'image_id', 'category', 'label')
AUDIT_FIELDS = ('question_sha256', 'question_prompt_sha256', 'image_sha256',
    'image_grid_thw', 'image_grid_thw_tensor', 'visual_tokens', 'input_tokens',
    'input_ids', 'attention_mask', 'input_ids_sha256', 'attention_mask_sha256',
    'pixel_values', 'padded_input_tokens', 'padded_input_ids_sha256',
    'padded_attention_mask_sha256', 'batch_start', 'batch_size', 'input_truncated',
    'counted_image_tokens')
COMMON_LOCK_FIELDS = ('training_config_sha256', 'training_config_content_sha256',
    'reference_config_sha256', 'source_sha256', 'packages', 'model_revision',
    'model_config_sha256', 'model_provenance_sha256', 'model_weight_provenance_sha256',
    'processor_and_metadata_sha256', 'actual_parameter_dtypes',
    'pope_manifest_sha256', 'pope_manifest_file_sha256', 'official_commit',
    'official_source', 'mirror_source', 'questions', 'unique_images',
    'question_order_sha256', 'question_prompt_order_sha256',
    'numerical_path', 'precision', 'vision_precision', 'quantization',
    'visual_token_budget', 'min_pixels', 'max_pixels', 'batch_size',
    'max_new_tokens', 'do_sample', 'attention_implementation', 'tf32_matmul',
    'tf32_cudnn', 'input_truncation', 'prompt_suffix',
    'current_original_weight_bytes_verified', 'head_last_position_only',
    'explicit_lora_bf16_conversion', 'input')


def assert_same_inputs(left, right, *, require_audit=False):
    if len(left) != len(right):
        raise ValueError('paired POPE lengths differ')
    for before, after in zip(left, right):
        for key in PAIR_FIELDS:
            if before[key] != after[key]:
                raise ValueError(f'paired POPE identity differs: {key}')
        for key in AUDIT_FIELDS:
            if require_audit and (key not in before or key not in after):
                raise ValueError(f'paired POPE actual input audit missing: {key}')
            if (key in before or key in after) and (
                key not in before or key not in after or before[key] != after[key]
            ):
                raise ValueError(f'paired POPE actual input differs: {key}')


def assert_common_locks(cells):
    """Require shared scientific/input locks; adapter and hardware remain per-cell."""
    reference = cells['initial']['lock']
    for condition in CONDITIONS:
        lock = cells[condition]['lock']
        for key in COMMON_LOCK_FIELDS:
            if key not in reference or key not in lock or reference[key] != lock[key]:
                raise ValueError(f'paired POPE common evaluation lock differs: {key}')


def _rates(counts):
    tp, fp, tn, fn = np.moveaxis(counts, -1, 0)
    denominators = {'accuracy': tp + fp + tn + fn, 'f1': 2 * tp + fp + fn,
        'precision': tp + fp, 'recall': tp + fn, 'yes_rate': tp + fp + tn + fn}
    numerators = {'accuracy': tp + tn, 'f1': 2 * tp, 'precision': tp,
        'recall': tp, 'yes_rate': tp + fp}
    return {key: np.divide(numerators[key], denominator,
        out=np.full(denominator.shape, np.nan, dtype=float), where=denominator != 0)
        for key, denominator in denominators.items()}


def paired_contrasts(by_condition, replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED):
    """Resample one shared image universe for all conditions and categories.

    Undefined bootstrap ratios do not get an invented zero or get silently
    dropped: any such replicate makes that metric's interval unavailable.
    Point estimates remain available when their own denominators are defined.
    """
    if replicates != BOOTSTRAP_REPLICATES:
        raise ValueError('predeclared POPE bootstrap requires 6000 replicates')
    if set(by_condition) != set(CONDITIONS):
        raise ValueError('POPE analysis requires initial, sft, and dpo')
    reference = by_condition['initial']
    if not reference or len({row['question_id'] for row in reference}) != len(reference):
        raise ValueError('POPE rows must be nonempty with unique question IDs')
    summaries = {}
    for condition in CONDITIONS:
        rows = by_condition[condition]
        assert_same_inputs(reference, rows)
        summaries[condition] = metrics(rows)
        for row in rows:
            if row.get('score') != int(official_prediction(row['prediction']) == row['label']):
                raise ValueError('saved POPE score differs from official raw-answer parse')
    images = sorted({row['image_id'] for row in reference})
    image_index = {value: index for index, value in enumerate(images)}
    slices = ['overall', *sorted({row['category'] for row in reference})]
    slice_index = {value: index for index, value in enumerate(slices)}
    counts = np.zeros((len(images), len(slices), len(CONDITIONS), 4), dtype=np.int64)
    for condition_index, condition in enumerate(CONDITIONS):
        for row in by_condition[condition]:
            pred = official_prediction(row['prediction'])
            label = row['label']
            slot = 0 if pred == label == 'yes' else 1 if pred == 'yes' else 2 if label == 'no' else 3
            for slice_number in (0, slice_index[row['category']]):
                counts[image_index[row['image_id']], slice_number, condition_index, slot] += 1
    # Multinomial image multiplicities are exactly iid cluster resampling with
    # replacement. The same draw retains all questions from all three categories.
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(len(images), np.full(len(images), 1 / len(images)), size=replicates)
    sampled = (weights @ counts.reshape(len(images), -1)).reshape(
        replicates, len(slices), len(CONDITIONS), 4)
    sampled_rates = _rates(sampled)
    contrasts = {}
    for name, before, after in CONTRASTS:
        before_index, after_index = CONDITIONS.index(before), CONDITIONS.index(after)
        contrast_rows = []
        for slice_number, label in enumerate(slices):
            before_summary = summaries[before]['overall'] if label == 'overall' else summaries[before]['by_category'][label]
            after_summary = summaries[after]['overall'] if label == 'overall' else summaries[after]['by_category'][label]
            values = {}
            for metric in METRICS:
                left, right = before_summary[metric], after_summary[metric]
                point = 100 * (right - left) if left is not None and right is not None else None
                delta = 100 * (sampled_rates[metric][:, slice_number, after_index]
                    - sampled_rates[metric][:, slice_number, before_index])
                undefined = int(np.count_nonzero(~np.isfinite(delta)))
                interval = np.quantile(delta, [.025, .975]).tolist() if not undefined and point is not None else None
                values[metric] = {'before': left, 'after': right, 'delta_pp': point,
                    'image_cluster_bootstrap_95_pp': interval,
                    'one_sided_97_5_lower_pp': interval[0] if interval is not None else None,
                    'undefined_bootstrap_replicates': undefined,
                    'defined_bootstrap_replicates': replicates - undefined,
                    'interval_status': 'defined' if interval is not None else 'unavailable_undefined_ratio'}
            accuracy = values['accuracy']
            contrast_rows.append({'slice': label, 'questions': before_summary['n'],
                'image_clusters': int(np.count_nonzero(counts[:, slice_number, 0].sum(axis=-1))),
                'resampled_image_clusters': len(images), 'checkpoint_axes': 1,
                'metrics': values, 'delta_pp': accuracy['delta_pp'],
                'before_score_percent': 100 * before_summary['accuracy'],
                'after_score_percent': 100 * after_summary['accuracy'],
                'image_cluster_bootstrap_95_pp': accuracy['image_cluster_bootstrap_95_pp'],
                'one_sided_97_5_lower_pp': accuracy['one_sided_97_5_lower_pp'],
                'bootstrap_replicates': replicates, 'bootstrap_seed': seed,
                'uncertainty': 'paired image-cluster percentile bootstrap conditional on one checkpoint per condition; shared image draws across all categories and contrasts; category/secondary-metric intervals pointwise exploratory'})
        contrasts[name] = contrast_rows
    return contrasts, summaries


def quality_gate(contrasts):
    """Two main accuracy tests plus F1/recall point protection, not NI tests."""
    lookup = {name: next(row for row in rows if row['slice'] == 'overall')
        for name, rows in contrasts.items()}
    accuracy_checks, point_checks = {}, {}
    for name in ('dpo_minus_sft', 'dpo_minus_initial'):
        values = lookup[name]['metrics']
        point, lower = values['accuracy']['delta_pp'], values['accuracy']['one_sided_97_5_lower_pp']
        accuracy_checks[f'{name}_accuracy_at_least_0_5pp'] = point is not None and point >= .5
        accuracy_checks[f'{name}_accuracy_one_sided_97_5_lower_gt0'] = lower is not None and lower > 0
        for metric in ('f1', 'recall'):
            value = values[metric]['delta_pp']
            point_checks[f'{name}_{metric}_point_at_least_minus_0_5pp'] = value is not None and value >= -.5
    passed = all(accuracy_checks.values()) and all(point_checks.values())
    return {'quality_passed': passed, 'pope_quality_passed': passed,
        'accuracy_checks': accuracy_checks, 'point_protection_checks': point_checks,
        'accuracy_family': 'two primary overall accuracy contrasts; one-sided alpha=.05/2=.025 each',
        'point_protection_interpretation': 'F1 and recall point differences only; not noninferiority or equivalence',
        'automatic_new_training': False, 'automatic_official_evaluation': False}


def text_protection(path=None, *, expected_config_sha256=None):
    """Read, never modify, the separately frozen seed42 Text analysis."""
    pending = {'status': 'pending_text_analysis', 'passed': None,
        'one_sided_97_5_lower_pp': None, 'margin_pp': .5,
        'criterion': 'Text dpo_minus_initial overall one-sided 97.5% lower strictly > -0.5pp'}
    if path is None:
        return pending
    path = private_path(path)
    if path.is_dir():
        candidates = (path / 'ANALYSIS.json', path / 'pilot_seed42/ANALYSIS.json',
            path / 'analysis/pilot_seed42/ANALYSIS.json')
        path = next((value for value in candidates if value.exists()), candidates[0])
    pending['analysis_file'] = str(path)
    if not path.exists():
        return pending
    payload = read_json(path)
    pending['analysis_sha256'] = sha_file(path)
    if payload.get('schema_version') != 'rarevlm.posttrain.dev_analysis.v1':
        raise ValueError('Text protection requires the frozen posttrain dev analysis schema')
    if payload.get('seeds') != [42]:
        raise ValueError('Text protection must use the seed42 exploration analysis')
    pending['schema_version'] = payload['schema_version']
    pending['config_sha256'] = payload.get('config_sha256')
    if expected_config_sha256 is not None and payload.get('config_sha256') != expected_config_sha256:
        raise ValueError('Text protection config_sha256 differs from this posttraining study')
    pending['configuration_binding_verified'] = expected_config_sha256 is not None
    if payload.get('status') != 'complete':
        return pending
    rows = payload.get('contrasts', {}).get('dpo_minus_initial', [])
    matches = [row for row in rows if row.get('slice') == 'overall']
    if len(matches) != 1:
        raise ValueError('completed Text analysis lacks one unique dpo_minus_initial overall contrast')
    lower = matches[0].get('one_sided_97_5_lower_pp')
    if lower is None:
        return pending
    if isinstance(lower, bool) or not isinstance(lower, (int, float)) or not math.isfinite(lower):
        raise ValueError('Text protection lower bound must be finite or None')
    return {**pending, 'status': 'available', 'passed': lower > -.5,
        'one_sided_97_5_lower_pp': lower,
        'original_text_source_data': payload.get('source_data'),
        'original_text_quality_gate': payload.get('quality_gate', payload.get('gate')),
        'original_text_gate_unchanged': True}


def promotion(pope_gate, text_guard):
    passed = None if text_guard['passed'] is None else bool(pope_gate['quality_passed'] and text_guard['passed'])
    return {'status': 'pending_text_analysis' if passed is None else ('passed' if passed else 'failed'),
        'passed': passed, 'pope_quality_passed': pope_gate['quality_passed'],
        'text_protection_passed': text_guard['passed'],
        'automatic_new_training': False, 'automatic_official_evaluation': False,
        'interpretation': 'single-checkpoint screening only; no automatic training, seed expansion, or formal evaluation; Text original quality gate remains independent'}


def assert_text_checkpoints(cells, text_guard):
    """Bind Text protection to these exact adapters and unchanged source files."""
    if text_guard['passed'] is None:
        return {**text_guard, 'checkpoint_binding_verified': False}
    sources = text_guard.get('original_text_source_data')
    if not isinstance(sources, dict):
        raise ValueError('Text checkpoint protection lacks source_data provenance')
    checked = {}
    for condition in CONDITIONS:
        source = sources.get(f'{condition}_42')
        if not isinstance(source, dict):
            raise ValueError(f'Text checkpoint protection lacks {condition}_42 provenance')
        expected = cells[condition]['lock']['training_binding']['sha256']
        if source.get('training_result_sha256') != expected:
            raise ValueError(f'Text/POPE checkpoint training result differs: {condition}')
        checked[condition] = {'training_result_sha256': expected}
        for kind in ('predictions', 'evaluation'):
            filename = source.get(f'{kind}_file')
            digest = source.get(f'{kind}_sha256')
            if not isinstance(filename, str) or not isinstance(digest, str):
                raise ValueError(f'Text {condition} {kind} source provenance missing')
            path = private_path(filename)
            if not path.is_file() or sha_file(path) != digest:
                raise ValueError(f'Text {condition} {kind} source file changed')
            checked[condition][f'{kind}_sha256'] = digest
    return {**text_guard, 'checkpoint_binding_verified': True,
        'source_files_rehashed': True, 'checkpoint_binding': checked}


def analyze(config_path, output_root, seed=42, text_analysis=None):
    if seed != 42:
        raise ValueError('this once-only POPE analysis is restricted to seed42')
    from .evaluate import read_cell
    config_path, output_root = private_path(config_path), private_path(output_root)
    config = read_json(config_path)
    settings = config.get('analysis', {})
    if settings.get('bootstrap_replicates', BOOTSTRAP_REPLICATES) != BOOTSTRAP_REPLICATES:
        raise ValueError('predeclared POPE bootstrap requires 6000 replicates')
    bootstrap_seed = settings.get('bootstrap_seed', BOOTSTRAP_SEED)
    cells = {condition: read_cell(output_root, condition, seed, config_path, config)
        for condition in CONDITIONS}
    destination = output_root / 'analysis/pope_pilot_seed42'
    guard = text_protection(text_analysis, expected_config_sha256=sha_file(config_path))
    result = {'schema_version': 'rarevlm.posttrain.pope.analysis.v1', 'updated_at': now(),
        'status': 'waiting_pope_predictions', 'complete': all(cell['complete'] for cell in cells.values()),
        'scope': 'seed42 single-checkpoint once-only external POPE exploration', 'seeds': [42],
        'completed_cells': sum(cell['complete'] for cell in cells.values()), 'expected_cells': 3,
        'expected_predictions': 27000, 'config_sha256': sha_file(config_path),
        'analysis_source_sha256': sha_file(Path(__file__)),
        'text_protection': guard, 'overall_promotion': {'status': 'pending_pope_predictions', 'passed': None}}
    if not result['complete']:
        result['pending_conditions'] = [condition for condition, cell in cells.items() if not cell['complete']]
        write_json(destination / 'ANALYSIS.json', result)
        return result
    by_condition = {condition: cells[condition]['rows'] for condition in CONDITIONS}
    reference = by_condition['initial']
    assert_common_locks(cells)
    guard = assert_text_checkpoints(cells, guard)
    result['text_protection'] = guard
    for condition in CONDITIONS:
        assert_same_inputs(reference, by_condition[condition], require_audit=True)
    if (len(reference) != 9000 or len({row['image_id'] for row in reference}) != 500
        or Counter(row['category'] for row in reference) != {'random': 3000, 'popular': 3000, 'adversarial': 3000}):
        raise ValueError('external POPE must contain all 9000 questions on 500 shared images')
    expected_images = {row['image_id'] for row in reference}
    for category in ('random', 'popular', 'adversarial'):
        subset = [row for row in reference if row['category'] == category]
        if ({row['image_id'] for row in subset} != expected_images
            or set(Counter(row['image_id'] for row in subset).values()) != {6}
            or Counter(row['label'] for row in subset) != {'yes': 1500, 'no': 1500}):
            raise ValueError('POPE category must retain shared 500 images, six QA/image and official label balance')
    contrasts, summaries = paired_contrasts(by_condition, seed=bootstrap_seed)
    gate = quality_gate(contrasts)
    result.update(status='complete', validated_predictions=27000, image_clusters=500,
        condition_summary=summaries, contrasts=contrasts, quality_gate=gate,
        pope_quality_gate=gate, overall_promotion=promotion(gate, guard),
        source_data={condition: cells[condition]['provenance'] for condition in CONDITIONS},
        pairing_audit={'shared_lock_fields': list(COMMON_LOCK_FIELDS),
            'matched_record_fields': [*PAIR_FIELDS, *AUDIT_FIELDS],
            'all_fields_required': True, 'all_conditions_matched': True},
        bootstrap={'replicates': BOOTSTRAP_REPLICATES, 'seed': bootstrap_seed,
            'unit': 'image_id shared across all categories', 'image_clusters': 500,
            'undefined_ratio_policy': 'None; no invented zeros and no removal of undefined draws; CI unavailable if any draw is undefined'},
        limitations=['one fixed checkpoint per condition; no training-seed stability claim',
            'public benchmark pretraining/training overlap unknown; not a new blind benchmark',
            'category and secondary-metric intervals pointwise exploratory; no simultaneous family claim',
            'strict format validity and specificity diagnostic only; all answers remain in official denominator',
            'F1/recall point protection does not establish noninferiority',
            'Text original quality gate retained independently; no automatic expansion or formal evaluation'])
    write_json(destination / 'ANALYSIS.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--seed42', action='store_true', help='Fixed seed42 scope (the sole supported analysis)')
    parser.add_argument('--text-analysis', type=Path)
    args = parser.parse_args()
    result = analyze(args.config, args.output_root, seed=42, text_analysis=args.text_analysis)
    print(result['status'], 'completed cells:', result['completed_cells'], '/', result['expected_cells'],
        'overall promotion:', result['overall_promotion']['status'])


if __name__ == '__main__':
    main()
