"""Exact frozen object/replay encoding and padded microbatch identity."""
from vlm_deep.common import collate
from vlm_object_train.train import batch_hash, encode_qa


def encode_microbatch(processor, rows, indices, config):
    encoded = [encode_qa(processor, rows[index], config) for index in indices]
    samples, contracts = zip(*encoded)
    batch = collate(samples, processor.tokenizer.pad_token_id)
    micro_sha = batch_hash(batch)
    padded_length = int(batch['input_ids'].shape[1])
    contracts = [{**contract, 'microbatch_input_sha256': micro_sha,
                  'microbatch_question_position': position, 'microbatch_size': len(indices),
                  'microbatch_sequence_length': padded_length}
                 for position, contract in enumerate(contracts)]
    return batch, contracts, samples


def cache_key(update, microbatch):
    return f'update_{update:06d}_replay_{microbatch:03d}'
