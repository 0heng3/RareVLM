from pathlib import Path
import json
import torch
from rarevlm_paths import PROJECT_ROOT
from vlm_deep.common import encode, collate, content_sha
from vlm_probe.evaluate import inference_row, processor_for
from vlm_sampled_signal_v1.trajectory import attach_completion

ROOT = PROJECT_ROOT.parents[1] / 'outputs/vlm_grpo_iteration_v1'
OLD = ROOT.parent / 'vlm_error_supervision_sft_v1'
SAMPLED = ROOT.parent / 'vlm_sampled_reward_signal_v1'


def read(path):
    return json.loads(Path(path).read_text())


def actor(training=False, adapter=None):
    from vlm_fp32_readout_v1.model import load
    from vlm_human_sft.evaluate import lora_hash
    plan = read(ROOT / 'PLAN.json')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, binding = load(plan['model_root'], adapter or plan['adapter'], torch.device('cuda:0'), training=training, dropout=0.)
    if adapter is None and lora_hash(model) != plan['lora_sha256']:
        raise ValueError('Wrong original policy')
    processor = processor_for(Path(plan['model_root']), read(plan['reference_config']), 1024)
    return model, processor, binding


def chosen_sample(processor, row, images):
    item = encode(processor, row, Path(images[row['image_id']]['path']), True)
    eos = processor.tokenizer.convert_tokens_to_ids('<|im_end|>')
    positions = ((item['labels'][0] != -100) & item['input_ids'][0].eq(eos)).nonzero().flatten()
    if len(positions) != 1:
        raise ValueError('Need one actual supervised template EOS')
    cut = int(positions[0]) + 1
    return {k: v[:, :cut] if k in ('input_ids', 'attention_mask', 'labels') else v for k,v in item.items()}


def chosen_batch(processor, rows, images, cpu):
    samples = [chosen_sample(processor, q, images) for q in rows]
    for row, item in zip(rows, samples):
        if content_sha(item['input_ids'].tolist()) != cpu[row['question_id']]['input_ids'] or content_sha(item['labels'].tolist()) != cpu[row['question_id']]['labels']:
            raise ValueError('New chosen/EOS input changed')
    return collate(samples, processor.tokenizer.pad_token_id)


def rollout_batch(processor, group, row, images):
    prefix = encode(processor, inference_row(row), Path(images[row['image_id']]['path']), False)
    if prefix['input_ids'][0].tolist() != group['prompt_ids']:
        raise ValueError('Behavior prompt changed')
    return collate([attach_completion(prefix, q['completion_ids']) for q in group['samples']], processor.tokenizer.pad_token_id)
