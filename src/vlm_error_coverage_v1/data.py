"""Keep physical generation microbatches intact when sharding the fixed pool."""


def assigned_batches(rows, shard, batch_size=4, shards=2):
    if len(rows) % (batch_size*shards) or not 0 <= shard < shards:
        raise ValueError('Require complete fixed batches on every shard')
    batches=[rows[start:start+batch_size] for start in range(0,len(rows),batch_size)]
    if any(len({q['fold'] for q in group})!=1 for group in batches):
        raise ValueError('Generation batch crosses training/validation fold')
    return [group for i,group in enumerate(batches) if i%shards==shard]
