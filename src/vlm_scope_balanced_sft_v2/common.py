import json
from pathlib import Path
import torch
from rarevlm_paths import PROJECT_ROOT
from vlm_deep.common import write_json, NORMALIZER
from vlm_region_preference_probe_v1.common import make_batch
from vlm_attention_scope_repair_v1.generation import raw_greedy_config
from vlm_probe.evaluate import evaluate_batch
from vlm_deep.evaluate import score_prediction

ROOT = PROJECT_ROOT.parents[1] / 'outputs/vlm_scope_balanced_sft_v2'
PARENT = ROOT.parent / 'vlm_region_preference_probe_v1'
CONFIG = PROJECT_ROOT / 'configs/vlm_scope_balanced_sft_v2.json'
read = lambda path: json.loads(Path(path).read_text())


def evaluate(model, processor, directory):
    data = read(ROOT / 'data/EVAL.json')
    model.eval()
    model.config.use_cache = True
    model.generation_config = raw_greedy_config(model.generation_config, 32)
    summary = {}
    with torch.inference_mode():
        for name, key in (('hard', 'hard_eval'), ('retention', 'retention_eval')):
            rows = data[key]
            path = directory / name / 'PREDICTIONS.json'
            predictions = read(path) if path.exists() else []
            if len(predictions) % 4 or [q['question_id'] for q in predictions] != [q['question_id'] for q in rows[:len(predictions)]]:
                raise ValueError('Evaluation cache order differs')
            for start in range(len(predictions), len(rows), 4):
                chunk = rows[start:start + 4]
                answers, _ = evaluate_batch(model, processor, chunk, data['images'], ROOT, next(model.parameters()).device,
                                           1024, processor.tokenizer.convert_tokens_to_ids('<|image_pad|>'), 32)
                for q, answer in zip(chunk, answers):
                    if 'expected_input_grid' in q and (answer['image_grid_thw'] != q['expected_input_grid'] or answer['input_tokens'] != q['expected_input_token_count']):
                        raise ValueError('New retained input grid/length differs from CPU encoding')
                    p, target = NORMALIZER(answer['prediction']), NORMALIZER(q['target'])
                    predictions.append({'question_id': q['question_id'], 'image_id': q['image_id'], 'target': q['target'],
                        'pair_id': q.get('pair_id'), 'scope_stratum': q.get('scope_stratum'), **answer,
                        'raw_exact': answer['prediction'] == q['target'],
                        'casefold_exact': answer['prediction'].casefold() == q['target'].casefold(),
                        'prefix_truncated': bool(p) and target.startswith(p + ' '),
                        'vqa_score': score_prediction(answer['prediction'], q, 'textvqa') if name == 'retention' else None})
                write_json(path, predictions)
                write_json(directory / 'EVAL_PROGRESS.json', {'status': 'evaluating', 'cohort': name,
                           'completed_qa': len(predictions), 'expected_qa': len(rows)})
            if name == 'hard':
                pairs = []
                for i in range(0, len(predictions), 2):
                    a, b = predictions[i:i + 2]
                    if a['pair_id'] != b['pair_id']:
                        raise ValueError('Region pairs differ')
                    pairs.append({'pair_id': a['pair_id'], 'both_correct': a['casefold_exact'] and b['casefold_exact'],
                                  'same_answer': a['prediction'].casefold() == b['prediction'].casefold()})
                write_json(directory / name / 'PAIRS.json', pairs)
                summary[name] = {'qa': len(predictions), 'raw_correct': sum(q['raw_exact'] for q in predictions),
                    'casefold_correct': sum(q['casefold_exact'] for q in predictions), 'pairs': len(pairs),
                    'both_correct': sum(q['both_correct'] for q in pairs), 'same_answer': sum(q['same_answer'] for q in pairs)}
            else:
                def metrics(values):
                    score = sum(q['vqa_score'] for q in values)
                    return {'qa': len(values), 'score_points': score, 'score_percent': 100 * score / len(values),
                            'prefix_truncated': sum(q['prefix_truncated'] for q in values)}
                summary[name] = {**metrics(predictions),
                    **{stratum: metrics([q for q in predictions if q['scope_stratum'] == stratum]) for stratum in ('short', 'multiword')}}
    write_json(directory / 'EVAL_COMPLETE.json', {'status': 'completed', 'metrics': summary,
               'scope': 'Training-image-disjoint64 retained QA and training28 regions; not main benchmark'})
    return summary
