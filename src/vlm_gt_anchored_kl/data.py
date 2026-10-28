"""GT-only replay preparation. No generated-answer file or rollout is consumed."""
from __future__ import annotations
import json,os,time
from pathlib import Path
for _k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[_k]='2'
import torch
from vlm_deep.common import collate,content_sha,encode,sha_file
from vlm_object_train.data import private_path,read_training_rows
from vlm_object_train.schedule import build_schedule
from vlm_object_train.train import batch_hash,learning_rate,tensor_hash
from . import CONDITIONS
PLAN_SCHEMA='rarevlm.gt_anchored_kl.plan.v1'
def read_json(path):
    return json.loads(Path(path).read_text())



def atomic_json(path, value):
    path = private_path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)



def immutable_json(path, value):
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f'immutable JSON differs: {path}')
    else:
        atomic_json(path, value)
    return value



def sealed(value, key='manifest_sha256'):
    return {**value, key: content_sha(value)}



def check_seal(value, key='manifest_sha256'):
    if value.get(key) != content_sha({k: v for k, v in value.items() if k != key}):
        raise ValueError(f'changed {key}')
    return value



def load_rows(config):
    from .contracts import validate_inherited_admissions
    validate_inherited_admissions(config)
    objects, replays, _ = read_training_rows(config)
    if len(objects) != 2888 or len(replays) != 32985:
        raise ValueError('full frozen object and train_all pools required')
    return objects, replays



def build_plan(config, seed):
    if seed not in (0, 42, 200) or config['epochs'] != 1 or config['object_effective_batch'] != 16 or config['replay_per_update'] != 32:
        raise ValueError('fixed order seeds/one epoch/object16/replay32 required')
    objects, replays = load_rows(config)
    blocks = build_schedule(objects, replays, seed, 1, 16, 32)
    oq = [[str(objects[i]['question_id']) for i in b['object_indices']] for b in blocks]
    rq = [[str(replays[i]['question_id']) for i in b['replay_indices']] for b in blocks]
    schedule = {'object_question_ids_by_update': oq, 'replay_question_ids_by_update': rq,
        'learning_rates_by_update': [learning_rate(i, len(blocks), config) for i in range(len(blocks))],
        'training_microbatch_questions': 4,
        'object_microbatches_by_update': [[q[i:i + 4] for i in range(0, len(q), 4)] for q in oq],
        'replay_microbatches_by_update': [[q[i:i + 4] for i in range(0, len(q), 4)] for q in rq]}
    flat = [q for block in rq for q in block]
    if len(blocks) != 181 or len(flat) != 5792 or len(set(flat)) != 5792:
        raise ValueError('complete unique5792/181-step stream required')
    if seed == 42:
        old_file = private_path(config['historical_coverage_study_root']) / 'runs/wide32/seed42/train/SCHEDULE.json'
        if read_json(old_file) != schedule:
            raise ValueError('seed42 stream differs from completed frozen wide32 schedule')
    value = {'schema_version': PLAN_SCHEMA, 'seed': seed, 'config_sha256': content_sha(config),
        'blocks': blocks, 'schedule': schedule,
        'counts': {'objects': len(objects), 'replay_pool': len(replays), 'updates': 181,
            'replay_exposures': 5792, 'unique_replay_QA': 5792, 'new_teacher_generation_QA': 0},
        'object_order_sha256': content_sha(oq), 'replay_order_sha256': content_sha(rq),
        'source_files_sha256': {str(private_path(config[key])): sha_file(private_path(config[key]))
            for key in ('train_manifest', 'replay_manifest', 'study_protocol')}}
    return sealed(value, 'plan_sha256')



def check_sample(sample, config):
    labels, roles = sample['labels'], sample['answer_roles']
    if (roles.dtype != torch.long or roles.shape != labels.shape or labels.shape != sample['input_ids'].shape
            or not torch.equal(labels != -100, roles != 0) or not bool(((roles >= 0) & (roles <= 4)).all())
            or not torch.equal(labels[labels != -100], sample['input_ids'][labels != -100])
            or not bool((labels != -100).any()) or int(labels[0, 0]) != -100
            or sample['input_ids'].shape[1] > config['max_sequence_length']):
        raise ValueError('assistant IDs/labels/roles/mask/length contract differs')
    grid = sample['image_grid_thw']
    if grid.shape != (1, 3) or int(grid[0, 0]) != 1 or int(grid.prod()) % 4 or not 0 < int(grid.prod()) // 4 <= config['visual_token_budget']:
        raise ValueError('one full static image within fixed visual budget required')
    return sample



def collate_training(samples, pad_token_id):
    if not samples:
        raise ValueError('empty training microbatch')
    batch = collate(samples, pad_token_id)
    roles = torch.zeros_like(batch['input_ids'], dtype=torch.long)
    for i, sample in enumerate(samples):
        roles[i, -sample['answer_roles'].shape[1]:] = sample['answer_roles'][0]
    batch['answer_roles'] = roles
    if not torch.equal(batch['labels'] != -100, roles != 0):
        raise ValueError('left padding changed role/label coverage')
    return batch



def sample_contract(sample, row, config):
    check_sample(sample, config)
    standard = {key: value for key, value in sample.items() if key != 'answer_roles'}
    valid = sample['labels'] != -100
    return {'question_id': str(row['question_id']), 'image_id': str(row['image_id']),
        'input_sha256': batch_hash(standard), 'input_tokens': int(sample['attention_mask'].sum()),
        'sequence_length': int(sample['input_ids'].shape[1]), 'visual_tokens': int(sample['image_grid_thw'].prod()) // 4,
        'assistant_tokens': int(valid.sum()), 'truncated': False,
        'answer_roles_sha256': tensor_hash([('answer_roles', sample['answer_roles'])]),
        'assistant_target_ids_sha256': tensor_hash([('assistant_target_ids', sample['labels'][valid])])}



def load_teacher_replies(config,seed):
    if type(seed) is not int or seed not in (0,42,200):raise ValueError('unregistered seed')
    return {}  # compatibility field: no generated replies exist or are required

def encode_training_sample(processor,row,config,*,source,condition,teacher_replies=None):
    if source not in ('object','replay') or condition not in CONDITIONS:raise ValueError('GT-only source/condition')
    sample=encode(processor,row,row['image_path'],True)
    answer=sample['labels'][sample['labels']!=-100].tolist()
    eos=len(answer)-2
    if eos<0 or answer[eos:]!=[151645,198]:raise ValueError('GT EOS/template suffix changed')
    roles=([1]+[2]*(eos-1) if eos else [])+[3,4]
    sample['answer_roles']=torch.zeros_like(sample['labels'],dtype=torch.long)
    sample['answer_roles'][sample['labels']!=-100]=torch.tensor(roles,dtype=torch.long)
    return check_sample(sample,config)
