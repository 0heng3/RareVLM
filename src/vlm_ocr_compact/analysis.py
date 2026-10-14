"""Independent compact-context analysis after frozen generic OCR analysis.

The compact candidate was fixed after inspecting the completed v1 result.
Both central95% and central97.5% image-cluster percentile intervals are kept.
The latter is a conservative advancement screen for two specified OCR primary
comparisons, not a correction for adaptive selection or all historical work.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True

from rarevlm_paths import PROJECT_ROOT, SOURCE_ROOT
from vlm_deep.common import NORMALIZER, training_target
from vlm_ocr.analysis import LABELS, EXPECTED_COUNTS, prepare_slices
from vlm_ocr.evaluate import (CONDITIONS, prepared_binding, read_json,
                             source_hashes, validate_config as validate_generic)
from vlm_probe.evaluate import (content_sha, now, private_path, read_manifest,
                               sha_file, write_json)

DEFAULT_CONFIG = PROJECT_ROOT / 'configs/vlm_ocr_compact_v1.json'
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20261013
CONFIDENCE_LEVELS = (.95, .975)
TARGET = 'predicted_ocr'


def configuration(path):
    config = validate_generic(read_json(path))
    required = {'two_sided_confidence': .975, 'exploratory_ci': .95,
                'specified_ocr_primary_candidates': 2}
    for key, value in required.items():
        if config.get('compact_statistics', {}).get(key) != value:
            raise ValueError(f'compact statistical declaration differs: {key}')
    config.update(_config_path=str(path), _config_file_sha256=sha_file(path))
    return config


def readiness(asset_root, output_root, config_path):
    config = configuration(config_path)
    slices = prepare_slices(asset_root, output_root)
    prepared, _ = prepared_binding(config, output_root)
    sources = source_hashes()
    sources.update({'vlm_ocr/analysis.py': sha_file(SOURCE_ROOT/'vlm_ocr/analysis.py'),
                    'vlm_ocr_compact/analysis.py': sha_file(Path(__file__))})
    ready = {'schema_version': 'rarevlm.ocr_compact.analysis_ready.v1',
             'status': 'prepared', 'created_at': now(),
             'config_sha256': sha_file(config_path),
             'analysis_source_sha256': sha_file(Path(__file__)),
             'source_sha256': sources, 'prepared_binding': prepared,
             'slice_lock_sha256': slices['slice_lock_sha256'],
             'expected_predictions': 6468, 'bootstrap_replicates': 2000,
             'bootstrap_seed': 20261013, 'central_confidence_levels': [.95, .975],
             'primary_comparison': 'predicted_ocr minus native baseline overall',
             'advancement_interval': 'central97.5% percentile[1.25%,98.75%]',
             'gate_thresholds': {'overall_min_delta_pp': .5, 'ci97_5_lower_gt_pp': 0.,
                                 'rare_min_delta_pp': 0., 'frequent_min_delta_pp': -.5},
             'scope': 'fixed compact candidate chosen after completed v1;historically explored dev;conservative two-OCR-candidate screen;not adaptive-selection correction or study-wide FWER'}
    write_json(output_root/'COMPACT_ANALYSIS_READY.json', ready)
    return ready


def independent_consensus(prediction, answers):
    """Recompute ten-reference leave-one-out scoring, not saved aggregates."""
    if len(answers) != 10:
        raise ValueError('TextVQA requires all ten frozen reference answers')
    prediction = NORMALIZER(prediction)
    references = [NORMALIZER(a) for a in answers]
    matches = sum(a == prediction for a in references)
    return sum(min((matches - int(a == prediction))/3., 1.)
               for a in references)/10.


def frequency_labels(manifest):
    train, dev = manifest['splits']['train_all'], manifest['splits']['dev']
    if len(train) != 32985 or len(dev) != 1617:
        raise ValueError('frozen train_all/dev counts changed')
    frequency = Counter()
    for row in train:
        if row['target'] != training_target(row['answers']):
            raise ValueError('raw-modal training target differs')
        frequency[NORMALIZER(row['target'])] += 1
    labels = []
    for row in dev:
        if row['target'] != training_target(row['answers']):
            raise ValueError('raw-modal dev target differs')
        f = frequency[NORMALIZER(row['target'])]
        labels.append(LABELS[0] if f == 0 else LABELS[1] if f <= 5
                      else LABELS[2] if f <= 20 else LABELS[3])
    if dict(Counter(labels)) != EXPECTED_COUNTS:
        raise ValueError('actual-target frequency slice counts changed')
    return labels


def primary_bootstrap(before, after, rows):
    import numpy as np
    if len(before) != len(after) or len(before) != len(rows):
        raise ValueError('unpaired primary arrays')
    clusters = {}
    for left, right, row in zip(before, after, rows):
        entry = clusters.setdefault(row['image_id'], [0., 0.])
        entry[0] += right - left
        entry[1] += 1
    matrix = np.asarray(list(clusters.values()), dtype=float)
    if len(matrix) != 1024:
        raise ValueError('primary requires1024 image clusters')
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values = []
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled = matrix[rng.integers(0, len(matrix), size=len(matrix))].sum(axis=0)
        values.append(100*sampled[0]/sampled[1])
    intervals = {}
    for confidence in CONFIDENCE_LEVELS:
        alpha = (1 - confidence)/2
        key = '95' if confidence == .95 else '97_5'
        intervals[key] = np.quantile(values, [alpha, 1-alpha]).tolist()
    return {'comparison': 'predicted_ocr_minus_baseline', 'slice': 'overall',
            'questions': len(rows), 'image_clusters': len(matrix),
            'reference_score_percent': 100*statistics.mean(before),
            'candidate_score_percent': 100*statistics.mean(after),
            'delta_pp': 100*statistics.mean(r-l for l, r in zip(before, after)),
            'image_cluster_bootstrap_95_pp': intervals['95'],
            'image_cluster_bootstrap_97_5_pp': intervals['97_5'],
            'bootstrap_replicates': BOOTSTRAP_REPLICATES,
            'bootstrap_seed': BOOTSTRAP_SEED,
            'central_interval_quantiles': {'95': [.025, .975], '97_5': [.0125, .9875]},
            'uncertainty': 'conditional on one fixed checkpoint and historically explored dev;paired same-image QA resampled together;not training-seed stability'}


def quality_gate(primary, rare_delta, frequent_delta, config):
    expected = {'overall_min_delta_vs_baseline_pp': .5, 'primary_ci_lower_min_pp': 0.,
                'rare_min_delta_vs_baseline_pp': 0., 'frequent_min_delta_vs_baseline_pp': -.5}
    for key, value in expected.items():
        if config.get('dev_gate', {}).get(key) != value:
            raise ValueError(f'compact gate threshold changed: {key}')
    checks = {'overall_delta_ge_0_5_pp': primary['delta_pp'] >= .5,
              'primary_image_cluster_97_5_ci_lower_gt_zero': primary['image_cluster_bootstrap_97_5_pp'][0] > 0.,
              'rare_point_delta_ge_zero': rare_delta >= 0.,
              'frequent_point_delta_ge_minus_0_5_pp': frequent_delta >= -.5}
    return {'passed': all(checks.values()), 'checks': checks, 'thresholds': expected,
            'primary_interval_confidence': .975, 'primary_interval_is_two_sided': True,
            'specified_ocr_primary_candidates': 2,
            'decision': 'review unchanged candidate for separately declared confirmation' if all(checks.values()) else 'retain complete negative result;no automatic tuning or expansion',
            'automatic_new_training': False, 'automatic_formal_evaluation': False,
            'interpretation': 'more conservative screen for the two specified OCR overall comparisons;v2 chosen after v1;does not correct adaptive selection or all historical comparisons;rare/frequent point guards are not noninferiority tests'}


def read_predictions(output_root, condition, rows, images, config_path, config, contract):
    directory = output_root/'runs'/condition/'dev'
    path = directory/'predictions.jsonl'
    data = path.read_bytes()
    if not data.endswith(b'\n'):
        raise ValueError('completed cache has an interrupted final line')
    records = [json.loads(line) for line in data.splitlines()]
    if len(records) != 1617 or len({r['question_id'] for r in records}) != 1617:
        raise ValueError('compact cell must contain1617 unique QA')
    lock = read_json(directory/'INPUT_LOCK.json')
    if (not lock or lock.get('condition') != condition
            or lock.get('ocr_config_sha256') != sha_file(config_path)
            or lock.get('source_sha256') != source_hashes()):
        raise ValueError('compact full-cell source/config/condition binding changed')
    lock_sha = content_sha(lock)
    scores = []
    for record, row in zip(records, rows):
        if (record['question_id'], record['image_id']) != (row['question_id'], row['image_id']):
            raise ValueError('compact QA/image order differs')
        if (record['condition'] != condition or record['input_lock_sha256'] != lock_sha
                or record['image_sha256'] != images[row['image_id']]['sha256']):
            raise ValueError('compact prediction identity differs')
        value = independent_consensus(record['prediction'], row['answers'])
        if not math.isclose(value, record['score'], rel_tol=0, abs_tol=1e-12):
            raise ValueError('independent compact consensus differs')
        frozen = contract[(condition, str(row['question_id']))]
        for key in ('question_sha256', 'question_prompt_sha256', 'image_grid_thw',
                    'visual_tokens', 'counted_image_tokens', 'per_image_visual_tokens',
                    'image_slots', 'rgb_frame_sha256', 'pixel_values_sha256',
                    'pixel_values_shape', 'input_ids_sha256', 'encoded_input_sha256',
                    'context_sha256', 'input_tokens', 'padded_input_tokens',
                    'input_truncated', 'ocr_text_tokens', 'ocr_prefix_tokens'):
            if record[key] != frozen[key]:
                raise ValueError(f'compact recorded input differs from CPU contract: {key}')
        if record['input_truncated'] or not 1 <= record['visual_tokens'] <= 1024:
            raise ValueError('compact native input truncated or over cap')
        scores.append(value)
    evaluation = read_json(directory/'EVALUATION.json')
    if (not evaluation or evaluation.get('status') != 'completed'
            or evaluation['predictions_sha256'] != sha_file(path)
            or evaluation['input_lock_sha256'] != lock_sha
            or not math.isclose(evaluation['score_percent'], 100*statistics.mean(scores), rel_tol=0, abs_tol=1e-10)):
        raise ValueError('compact completed evaluation differs from independent scores')
    return records, scores, lock, {'path': str(path), 'sha256': sha_file(path),
                                   'input_lock_sha256': lock_sha}


def analyze(asset_root, output_root, config_path):
    config = configuration(config_path)
    ready = read_json(output_root/'COMPACT_ANALYSIS_READY.json')
    if (not ready or ready['config_sha256'] != sha_file(config_path)
            or ready['analysis_source_sha256'] != sha_file(Path(__file__))):
        raise ValueError('compact analysis readiness missing or changed')
    prepared, _ = prepared_binding(config, output_root)
    if ready['prepared_binding'] != prepared:
        raise ValueError('compact prepared inputs changed')
    for name, digest in ready['source_sha256'].items():
        if sha_file(SOURCE_ROOT/name) != digest:
            raise ValueError('compact analysis/helper source changed')
    generic_path = output_root/'analysis/ANALYSIS.json'
    generic = read_json(generic_path)
    if (not generic or not generic.get('complete') or generic.get('validated_predictions') != 6468
            or generic.get('completed_cells') != 4 or generic.get('config_sha256') != sha_file(config_path)
            or generic.get('analysis_source_sha256') != sha_file(SOURCE_ROOT/'vlm_ocr/analysis.py')):
        raise ValueError('complete frozen generic compact analysis required')
    manifest_path = asset_root/'prepared/manifest.json'
    manifest = read_manifest(manifest_path)
    rows, images = manifest['splits']['dev'], manifest['images']
    if len(rows) != 1617 or len({r['image_id'] for r in rows}) != 1024:
        raise ValueError('compact manifest QA/image counts changed')
    labels = frequency_labels(manifest)
    full = read_json(Path(prepared['input_contract']['path']))
    contract = {(r['condition'], str(r['question_id'])): r for r in full['records']}
    if len(contract) != 6468:
        raise ValueError('compact full input contract incomplete')
    cells = {c: read_predictions(output_root, c, rows, images, config_path, config, contract)
             for c in CONDITIONS}
    for left, right in (('baseline', TARGET), ('empty_template', 'shuffled_ocr')):
        if cells[left][2]['hardware']['uuid'] != cells[right][2]['hardware']['uuid']:
            raise ValueError('paired worker conditions used different GPUs')
    for index in range(len(rows)):
        base = cells['baseline'][0][index]
        for condition in CONDITIONS[1:]:
            other = cells[condition][0][index]
            for key in ('question_id', 'image_id', 'question_sha256', 'question_prompt_sha256',
                        'image_sha256', 'rgb_frame_sha256', 'pixel_values_sha256',
                        'pixel_values_shape', 'image_grid_thw', 'visual_tokens',
                        'per_image_visual_tokens', 'image_slots'):
                if base[key] != other[key]:
                    raise ValueError(f'compact four-arm actual image identity differs: {key}')
    summaries = {}
    for condition, (_, scores, lock, provenance) in cells.items():
        slices = {label: {'questions': labels.count(label),
                          'score_percent': 100*statistics.mean(v for v, l in zip(scores, labels) if l == label)}
                  for label in LABELS}
        score = 100*statistics.mean(scores)
        prior = generic['condition_summary'][condition]
        if not math.isclose(score, prior['score_percent'], rel_tol=0, abs_tol=1e-10):
            raise ValueError('compact independent overall differs from generic analysis')
        for label in LABELS:
            if not math.isclose(slices[label]['score_percent'], prior['slices'][label]['score_percent'], rel_tol=0, abs_tol=1e-10):
                raise ValueError('compact independent slice differs from generic analysis')
        summaries[condition] = {'score_percent': score, 'slices': slices,
                                'resource': prior['resource'], 'hardware': lock['hardware'],
                                'provenance': provenance}
    primary = primary_bootstrap(cells['baseline'][1], cells[TARGET][1], rows)
    old_primary = generic['primary_contrast']
    if not math.isclose(primary['delta_pp'], old_primary['delta_pp'], rel_tol=0, abs_tol=1e-10):
        raise ValueError('compact independently paired delta differs')
    for a, b in zip(primary['image_cluster_bootstrap_95_pp'], old_primary['image_cluster_bootstrap_95_pp']):
        if not math.isclose(a, b, rel_tol=0, abs_tol=1e-10):
            raise ValueError('compact independently paired95% interval differs')
    rare = summaries[TARGET]['slices']['rare_1_5']['score_percent'] - summaries['baseline']['slices']['rare_1_5']['score_percent']
    frequent = summaries[TARGET]['slices']['frequent_gt20']['score_percent'] - summaries['baseline']['slices']['frequent_gt20']['score_percent']
    old_cost = generic['preparation_cost']
    cost = {'historical_ocr_cache_reused': True, 'new_ocr_inference_executions': 0,
            'historical_detector_predict_api_seconds_once_per_unique_image': old_cost['historical_cached_detector_predict_api_seconds_once_per_unique_image'],
            'historical_recognizer_predict_api_seconds_once_per_unique_image': old_cost['new_recognition_predict_api_seconds_once_per_unique_image'],
            'ocr_ready_path': prepared['ocr_manifest']['path'],
            'new_context_and_full_CPU_input_validation_seconds': full.get('elapsed_seconds'),
            'scope': 'v1 detector/recognizer outputs reused;historical OCR cost is not new compact inference;cached VLM phases reported per cell;no fresh end-to-end or service speedup claim'}
    result = {'schema_version': 'rarevlm.ocr_compact.analysis.v1', 'status': 'complete',
              'complete': True, 'updated_at': now(), 'completed_cells': 4,
              'validated_predictions': 6468, 'expected_predictions': 6468,
              'config_sha256': sha_file(config_path),
              'analysis_source_sha256': sha_file(Path(__file__)),
              'generic_analysis': {'path': str(generic_path), 'sha256': sha_file(generic_path)},
              'slice_lock_sha256': ready['slice_lock_sha256'], 'prepared_binding': prepared,
              'condition_summary': summaries, 'primary_contrast': primary,
              'contrasts': generic['contrasts'], 'gate': quality_gate(primary, rare, frequent, config),
              'generic_registered_95_gate_diagnostic': generic['gate'],
              'preparation_cost': cost,
              'statistical_declaration': {'central95_quantiles': [.025, .975],
                  'central97_5_quantiles': [.0125, .9875], 'specified_ocr_primary_candidates': 2,
                  'authoritative_advancement_interval': 'central97.5%',
                  'v1_registered95_interval_is_unchanged': True,
                  'scope': 'two specified OCR overall primary comparisons only;v2 fixed after observing complete v1;conservative screen does not correct adaptive selection or all historical comparisons'},
              'limitations': ['historically explored internal dev, not a new blind holdout',
                  'single fixed existing checkpoint;no training-seed replication',
                  '95/97.5 percentile intervals conditional on this checkpoint;secondary/frequency intervals are exploratory unadjusted',
                  'rare/frequent advancement guards are point estimates, not equivalence/noninferiority tests',
                  'compact carrier and word-only packing jointly change prompt semantics,text coverage and input length;no isolated coordinate-removal attribution',
                  'image pixels/grid/visual tokens match per QA;total textual information/input length/compute can differ',
                  'historical detector/recognizer preparation is cached;VLM timings exclude fresh OCR;not online speedup',
                  'public pretraining overlap unknown;no claim of unseen pretraining examples'],
              'scope': 'one frozen compact text-only OCR candidate;full internal dev;zero training;no automatic official evaluation or resume update'}
    directory = output_root/'compact_analysis'
    write_json(directory/'ANALYSIS.json', result)
    # Reader compatibility points at the same independent result; neither file
    # overwrites the generic report or any v1 result.
    write_json(output_root/'analysis/COMPACT_ANALYSIS.json', result)
    lines = ['# Compact OCR context results', '',
             f'All four cells complete;6,468 predictions independently re-scored. Conservative gate passed: {result["gate"]["passed"]}.', '',
             '| condition | score (%) |', '|---|---:|']
    lines += [f'| {c} | {s["score_percent"]:.6f} |' for c, s in summaries.items()]
    lines += ['', f'Primary delta: {primary["delta_pp"]:+.6f} pp.',
              f'Central95% image-cluster interval: {primary["image_cluster_bootstrap_95_pp"]}.',
              f'Central97.5% image-cluster interval: {primary["image_cluster_bootstrap_97_5_pp"]}.', '',
              'The97.5% interval is a conservative screen for two specified OCR primary comparisons. This candidate was fixed after v1 results;the screen does not correct adaptive selection or study-wide multiplicity.', '',
              'Rare/frequent gate guards are point estimates. Cached VLM timing excludes the reused historical detector/recognizer cost.', '',
              'No automatic replication, official benchmark evaluation, or resume change.']
    (directory/'findings.md').write_text('\n'.join(lines)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    args.asset_root = private_path(args.asset_root)
    args.output_root = private_path(args.output_root)
    args.config = private_path(args.config)
    result = readiness(args.asset_root, args.output_root, args.config) if args.prepare_only else analyze(args.asset_root, args.output_root, args.config)
    print(json.dumps({k: result.get(k) for k in ('status', 'complete', 'validated_predictions', 'gate', 'analysis_source_sha256')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
