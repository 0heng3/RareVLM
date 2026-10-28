"""Explicit raw greedy policy for NEW studies, preserving token identities.

The historical evaluator inherited repetition_penalty=1.05 from the base model.
This wrapper deliberately establishes a new policy and requires a fresh initial
baseline. It must never be used to overwrite an old study's evaluation cache.
"""

from copy import deepcopy
from pathlib import Path


def raw_greedy_config(inherited, max_new_tokens=32):
    from transformers import GenerationConfig

    if type(max_new_tokens) is not int or max_new_tokens < 1:
        raise ValueError('max_new_tokens must be a positive integer')
    # Build from declared fields, rather than copying hidden processors such as
    # repetition/sequence biases or bad-word suppression from a base checkpoint.
    return GenerationConfig(
        bos_token_id=deepcopy(inherited.bos_token_id),
        eos_token_id=deepcopy(inherited.eos_token_id),
        pad_token_id=deepcopy(inherited.pad_token_id),
        do_sample=False, num_beams=1, repetition_penalty=1.0,
        temperature=1.0, top_p=1.0, top_k=50,
        max_new_tokens=max_new_tokens, use_cache=True)


def evaluate(model, processor, config, task, directory):
    """Use the validated evaluator under a separately locked generation policy."""
    from vlm_deep.common import content_sha, sha_file
    from vlm_human_sft.evaluate import evaluate as historical_evaluate
    from vlm_object_train.train import lock_json

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    inherited = model.generation_config
    policy = raw_greedy_config(inherited, config['max_new_tokens'])
    policy_path = directory / 'GENERATION_POLICY.json'
    if (directory / 'INPUT_LOCK.json').exists() and not policy_path.exists():
        raise ValueError('Refusing to attach a new generation policy to a historical cache')
    lock_json(policy_path, {'policy': policy.to_dict(),
                           'policy_sha256': content_sha(policy.to_dict()),
                           'source_sha256': sha_file(Path(__file__)),
                           'fresh_same_policy_initial_baseline_required': True})
    model.generation_config = policy
    try:
        return historical_evaluate(model, processor, config, task, directory)
    finally:
        model.generation_config = inherited
