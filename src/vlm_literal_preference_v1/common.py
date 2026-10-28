from pathlib import Path
import json
from rarevlm_paths import PROJECT_ROOT
from vlm_deep.common import collate, content_sha, encode
from vlm_grpo_iteration_v1.common import chosen_sample
from vlm_probe.evaluate import inference_row
from vlm_sampled_signal_v1.trajectory import attach_completion

ROOT = PROJECT_ROOT.parents[1]/'outputs/vlm_literal_preference_v1'
CONFIG = PROJECT_ROOT/'configs/vlm_literal_preference_v1.json'


def read(path):
    return json.loads(Path(path).read_text())


def sample(processor, data, q, rejected=False):
    if not rejected:
        return chosen_sample(processor, q, data['images'])
    prefix = encode(processor, inference_row(q), Path(data['images'][q['image_id']]['path']), False)
    if content_sha(prefix['input_ids'].tolist()) != q['inference_input_sha256']:
        raise ValueError('Actual negative prompt differs from initial generation')
    if not q['initial_prediction']['terminated']:
        raise ValueError('Length-censored negative cannot establish a complete preference')
    return attach_completion(prefix, q['rejected_completion_ids'])


def make_batch(processor, data, rows, cpu):
    preference = rows[0]['study_source'] == 'preference'
    if any((q['study_source'] == 'preference') != preference for q in rows):
        raise ValueError('Mixed-source microbatch')
    chosen = [sample(processor, data, q) for q in rows]
    rejected = [sample(processor, data, q, True) for q in rows] if preference else []
    for kind, samples in [('chosen', chosen), ('rejected', rejected)]:
        for q, item in zip(rows, samples):
            expected = cpu[q['question_id']+':'+kind]
            if any(content_sha(item[k].tolist()) != expected[k+'_sha256'] for k in ('input_ids', 'labels')):
                raise ValueError('Registered assistant input/labels differ')
    batch = collate(chosen+rejected, processor.tokenizer.pad_token_id)
    binding = {'question_ids': [q['question_id'] for q in rows],
               'padded_input_ids_sha256': content_sha(batch['input_ids'].tolist()),
               'padded_labels_sha256': content_sha(batch['labels'].tolist()),
               'grid_sha256': content_sha(batch['image_grid_thw'].tolist())}
    return batch, binding, chosen, rejected
