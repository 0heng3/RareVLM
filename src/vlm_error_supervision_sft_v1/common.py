"""New study inputs/evaluation; use the established supervised-batch implementation unchanged."""
from collections import defaultdict
from pathlib import Path
import json
import time

from rarevlm_paths import PROJECT_ROOT
from vlm_deep.common import NORMALIZER, write_json
from vlm_region_preference_probe_v1.common import make_batch

ROOT = PROJECT_ROOT.parents[1] / 'outputs/vlm_error_supervision_sft_v1'
MINING = ROOT.parent / 'vlm_error_supervision_mining_v1'
CONFIG = PROJECT_ROOT / 'configs/vlm_error_supervision_sft_v1.json'


def read(path):
    return json.loads(Path(path).read_text())


def summarize(rows):
    return {'qa': len(rows), 'score_points': sum(q['vqa_score'] for q in rows),
            'vqa_percent': 100 * sum(q['vqa_score'] for q in rows) / len(rows),
            'full_score_qa': sum(q['vqa_score'] >= .999999 for q in rows),
            'prefix_missing_words': sum(bool(NORMALIZER(q['prediction'])) and
                NORMALIZER(q['target']).startswith(NORMALIZER(q['prediction']) + ' ') for q in rows)}


def evaluate(model, processor, directory, names=('hard', 'validation', 'retention')):
    import torch
    from vlm_probe.evaluate import evaluate_batch
    from vlm_attention_scope_repair_v1.generation import raw_greedy_config
    from vlm_deep.evaluate import score_prediction
    data = read(ROOT / 'data/MANIFEST.json')
    c = read(CONFIG)
    model.eval()
    model.config.use_cache = True
    model.generation_config = raw_greedy_config(model.generation_config, 32)
    summary = {}
    started = time.monotonic()
    with torch.inference_mode():
        for name in names:
            rows = data[name + '_eval']
            path = directory / name / 'PREDICTIONS.json'
            predictions = read(path) if path.exists() else []
            if len(predictions) % 4 or [q['question_id'] for q in predictions] != [q['question_id'] for q in rows[:len(predictions)]]:
                raise ValueError('Actual-error study generation cache grouping differs')
            for start in range(len(predictions), len(rows), 4):
                chunk = rows[start:start + 4]
                answers, resources = evaluate_batch(model, processor, chunk, data['images'], ROOT,
                    next(model.parameters()).device, 1024, processor.tokenizer.convert_tokens_to_ids('<|image_pad|>'), 32)
                for q, answer in zip(chunk, answers):
                    if 'expected_prefix_tokens' in q and (answer['input_tokens'] != q['expected_prefix_tokens']
                            or answer['image_grid_thw'] != q['expected_grid']):
                        raise ValueError('Final frame differs from actual initial-generation frame')
                    predictions.append({'question_id': q['question_id'], 'image_id': q['image_id'],
                        'target': q['target'], 'stratum': q['stratum'], 'answer_frequency': q['answer_frequency'],
                        **answer, 'vqa_score': score_prediction(answer['prediction'], q, 'textvqa'), **resources})
                write_json(path, predictions)
                write_json(directory / 'EVAL_PROGRESS.json', {'status': 'evaluating', 'cohort': name,
                    'completed_qa': len(predictions), 'expected_qa': len(rows), 'active_seconds': time.monotonic() - started})
                if time.monotonic() - started > c['evaluation_cap_seconds']:
                    raise TimeoutError('Registered Final generation cap reached')
            summary[name] = summarize(predictions)
    write_json(directory / 'EVAL_COMPLETE.json', {'status': 'completed', 'metrics': summary})
    return summary


def paired(before, after):
    import numpy as np
    if [q['question_id'] for q in before] != [q['question_id'] for q in after]:
        raise ValueError('Paired generation IDs differ')
    groups = defaultdict(lambda: [0., 0])
    for a, b in zip(before, after):
        if a['image_id'] != b['image_id']:
            raise ValueError('Paired image IDs differ')
        groups[a['image_id']][0] += b['vqa_score'] - a['vqa_score']
        groups[a['image_id']][1] += 1
    values = np.array([groups[k] for k in sorted(groups)], dtype=np.float64)
    rng = np.random.default_rng(20261020)
    indices = rng.integers(0, len(values), size=(6000, len(values)))
    samples = 100 * values[indices, 0].sum(1) / values[indices, 1].sum(1)
    return {'delta_pp': float(100 * values[:, 0].sum() / values[:, 1].sum()),
            'ci95': np.quantile(samples, [.025, .975]).tolist(), 'image_clusters': len(values),
            'replicates': 6000, 'seed': 20261020, 'scope': 'Conditional on these fixed checkpoints and this adaptation cohort.'}
