import copy
import json
import random
import sys
import types
from pathlib import Path

import pytest
import torch

from vlm_alignment_signal.loss import completion_logps, sequence_logps
from vlm_object_behavior_replay import train
from vlm_object_behavior_replay.loss import completion_ce, normalized_qa_loss, sequence_ce


class TinyActor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(7, 7, bias=False)
        self.frozen = torch.nn.Parameter(torch.tensor([2.]), requires_grad=False)
        self.register_buffer('seen', torch.tensor(0))
        self.calls = 0

    def get_output_embeddings(self):
        return self.head

    def forward(self, input_ids, attention_mask, use_cache=False):
        assert use_cache is False
        self.calls += 1
        self.seen.add_(1)
        self.rope_deltas = torch.tensor([[self.calls]])
        hidden = torch.nn.functional.one_hot(input_ids, 7).float()
        return types.SimpleNamespace(logits=self.head(hidden))

    def save_pretrained(self, directory, safe_serialization=True):
        directory = Path(directory)
        directory.mkdir(parents=True)
        (directory/'adapter.json').write_text(json.dumps(self.head.weight.detach().tolist()))


def sample(i=0):
    # A prompt token plus content/EOS/template, with variable supervised lengths.
    ids = torch.tensor([[0, 1+(i % 3), 4, 5]])
    labels = ids.clone()
    labels[:,0] = -100
    roles = torch.tensor([[0, 1, 3, 4]])
    if i % 3 == 1:
        labels[:,:2] = -100
        roles = torch.tensor([[0, 0, 3, 4]])  # Empty-content realEOS: no first-content role.
    elif i % 3 == 2:
        roles = torch.tensor([[0, 1, 2, 2]])  # Capped target withoutEOS/template.
    return {'input_ids':ids, 'attention_mask':torch.ones_like(ids), 'labels':labels, 'answer_roles':roles}


def collate(samples, pad_token_id=0):
    return {key:torch.cat([s[key] for s in samples]) for key in samples[0]}


def test_single_forward_matches_frozen_ce_and_gradients_with_exclusive_roles():
    torch.manual_seed(11)
    model = TinyActor()
    old = copy.deepcopy(model)
    batch = collate([sample(i) for i in range(4)])
    ce,counts,roles = sequence_ce(model,batch)
    logps,old_counts = sequence_logps(old,{k:v for k,v in batch.items() if k!='answer_roles'})
    expected = -logps/old_counts
    assert model.calls == 1
    assert torch.equal(ce,expected) and torch.equal(counts,old_counts)
    ce.mean().backward()
    expected.mean().backward()
    assert torch.equal(model.head.weight.grad,old.head.weight.grad)
    assert sum(v['token_count'] for v in roles.values()) == int(counts.sum())
    assert roles['first_content']['token_count'] == 3
    assert roles['actual_eos']['token_count'] == 3
    assert roles['later_content']['token_count'] == 2
    assert roles['template_suffix']['token_count'] == 3
    bad = batch['answer_roles'].clone()
    bad[0,0] = 1
    with pytest.raises(ValueError,match='cover'):
        completion_ce(torch.randn(4,4,7),batch['labels'],bad)


def test_variable_length_per_qa_micro4_object_tail8_replay32_gradient():
    parameter = torch.tensor(.17,requires_grad=True)
    baseline = parameter.detach().clone().requires_grad_()
    batches = {'object':[sample(i) for i in range(8)], 'replay':[sample(i+1) for i in range(32)]}
    basis = torch.arange(4*7,dtype=torch.float32).reshape(1,4,7)/13
    for source,rows in batches.items():
        for start in range(0,len(rows),4):
            b=collate(rows[start:start+4])
            ce,_,_ = completion_ce(parameter*basis.expand(len(b['labels']),-1,-1),b['labels'],b['answer_roles'])
            normalized_qa_loss(ce,source,8,32,1.).backward()
    expected = 0
    for rows in batches.values():
        b=collate(rows)
        logps,counts = completion_logps(baseline*basis.expand(len(rows),-1,-1),b['labels'])
        expected = expected+(-logps/counts).mean()
    expected.backward()
    assert torch.allclose(parameter.grad,baseline.grad,atol=3e-7,rtol=1e-6)
    assert not torch.isclose(expected.detach(),sum(-completion_logps(baseline.detach()*basis.expand(len(rows),-1,-1),collate(rows)['labels'])[0].sum()/completion_logps(baseline.detach()*basis.expand(len(rows),-1,-1),collate(rows)['labels'])[1].sum() for rows in batches.values()))


def fake_stream(blocks):
    objects=[{'question_id':str(i),'sample':sample(i)} for i in range(sum(len(b['object_indices']) for b in blocks))]
    replays=[{'question_id':str(i),'sample':sample(i+1)} for i in range(sum(len(b['replay_indices']) for b in blocks))]
    def encode(processor,rows,selected,config,**kwargs):
        samples=[rows[i]['sample'] for i in selected]
        contracts=[{'question_id':rows[i]['question_id'],'assistant_tokens':int((s['labels']!=-100).sum()),
                    'input_tokens':4,'visual_tokens':1} for i,s in zip(selected,samples)]
        return collate(samples),contracts,samples
    context={'teacher_replies':{},'plan':{'seed':42,'blocks':blocks},'contracts':{'object':[],'replay':[]}}
    for b in blocks:
        for source,rows in [('object',objects),('replay',replays)]:
            for start in range(0,len(b[f'{source}_indices']),4):
                context['contracts'][source].extend(encode(None,rows,b[f'{source}_indices'][start:start+4],{})[1])
    return objects,replays,context,encode


@pytest.mark.parametrize('fail', [False,True])
def test_preflight_restores_fp32_rng_buffers_rope_and_discards_adam(tmp_path,monkeypatch,fail):
    block={'object_indices':list(range(16)),'replay_indices':list(range(32))}
    objects,replays,context,encoder=fake_stream([block])
    monkeypatch.setattr(train,'encode_group',encoder)
    monkeypatch.setattr(train,'verify_prepared_files_unchanged',lambda *a,**k:True)
    data=types.ModuleType('vlm_object_behavior_replay.data')
    def encode_sample(processor,row,config,**kwargs):
        if fail:
            torch.rand(3)
            random.random()
            raise ValueError('injected_input_failure')
        return row['sample']
    data.encode_training_sample=encode_sample
    data.collate_training=collate
    monkeypatch.setitem(sys.modules,'vlm_object_behavior_replay.data',data)
    model=TinyActor()
    before=train.trainable_hash(model)
    rng=train.rng_hash(train.rng_state())
    config={'preflight_lr':1e-4,'lr':5e-6,'weight_decay':0.,'gradient_clip_norm':1.,'replay_lambda':1.,'training_microbatch':4,'preflight_memory_cap_gib':65}
    processor=types.SimpleNamespace(tokenizer=types.SimpleNamespace(pad_token_id=0))
    call=lambda:train.preflight(model,list(p for p in model.parameters() if p.requires_grad),processor,objects,replays,block,config,'teacher_ce32',tmp_path,torch.device('cpu'),context)
    if fail:
        with pytest.raises(ValueError,match='injected_input_failure'):
            call()
        result=json.loads((tmp_path/'PREFLIGHT.json').read_text())
        assert result['status']=='failed' and result['discarded_engineering_optimizer_updates']==0
        assert result['engineering_actor_forward_backward_questions']==0
        assert result['engineering_actor_forward_only_probe_questions']==0
    else:
        result=call()
        assert result['status']=='passed' and result['discarded_engineering_optimizer_updates']==1
        assert result['record']['initial_replay_mean_ce']>0
        assert result['record']['object_gradient_l2']>0 and result['record']['replay_gradient_l2']>0
    assert before==train.trainable_hash(model)
    assert rng==train.rng_hash(train.rng_state())
    assert model.seen.item()==0 and not hasattr(model,'rope_deltas')
    assert all(p.grad is None for p in model.parameters())
    assert result['model_runtime_buffers_and_rope_cache_restored']
    assert result['rng_restored'] and result['initial_parameters_restored_exactly']


def test_exact_adam_checkpoint_resume_keeps_new_role_and_input_cursors(tmp_path,monkeypatch):
    blocks=[{'object_indices':list(range(4)),'replay_indices':list(range(4))},
            {'object_indices':list(range(4,8)),'replay_indices':list(range(4,8))}]
    objects,replays,context,encoder=fake_stream(blocks)
    monkeypatch.setattr(train,'encode_group',encoder)
    torch.manual_seed(7)
    base=TinyActor()
    full=copy.deepcopy(base)
    part=copy.deepcopy(base)
    opt1=torch.optim.AdamW([full.head.weight],lr=5e-6,weight_decay=0.)
    opt2=torch.optim.AdamW([part.head.weight],lr=5e-6,weight_decay=0.)
    config={'training_microbatch':4,'replay_lambda':1.,'preflight_memory_cap_gib':65}
    processor=types.SimpleNamespace(tokenizer=types.SimpleNamespace(pad_token_id=0))
    def update(model,opt,stats,step):
        opt.zero_grad(set_to_none=True)
        train.backward_update(model,processor,objects,replays,blocks[step],'teacher_ce32',config,torch.device('cpu'),context,stats,update=step)
        torch.nn.utils.clip_grad_norm_([model.head.weight],1.)
        opt.step()
    stats1,stats2=train.initial_stats(),train.initial_stats()
    for step in range(2):update(full,opt1,stats1,step)
    update(part,opt2,stats2,0)
    train.save_checkpoint(tmp_path,part,opt2,1,stats2,'lineage','schedule')
    restored=copy.deepcopy(base)
    opt3=torch.optim.AdamW([restored.head.weight],lr=5e-6,weight_decay=0.)
    step,stats3=train.resume_checkpoint(tmp_path,restored,opt3,'lineage','schedule',2)
    assert step==1
    train.validate_exposure_cursor(stats3,blocks,step,context)
    update(restored,opt3,stats3,1)
    assert torch.equal(full.head.weight,restored.head.weight)
    assert stats1==stats3
    train.validate_exposure_cursor(stats3,blocks,2,context)
    bad=copy.deepcopy(stats3);bad['replay_input_contracts'].pop()
    with pytest.raises(ValueError,match='cursor'):
        train.validate_exposure_cursor(bad,blocks,2,context)
    with pytest.raises(ValueError,match='binding'):
        train.resume_checkpoint(tmp_path,restored,opt3,'other','schedule',2)


def test_actual_teacher_ids_roles_are_bound_before_forward(monkeypatch):
    from vlm_object_behavior_replay.inputs import encode_group
    data=types.ModuleType('vlm_object_behavior_replay.data')
    data.encode_training_sample=lambda *a,**k:sample(0)
    data.collate_training=collate
    data.sample_contract=lambda sample,row,config:{'question_id':row['question_id']}
    monkeypatch.setitem(sys.modules,'vlm_object_behavior_replay.data',data)
    replies={'q':{'training_target_ids':[1,4,5],'training_answer_roles':[1,3,4]}}
    processor=types.SimpleNamespace(tokenizer=types.SimpleNamespace(pad_token_id=0))
    batch,contracts,_=encode_group(processor,[{'question_id':'q'}],[0],{},source='replay',condition='teacher_ce32',teacher_replies=replies)
    assert contracts[0]['supervised_target_ids']==[1,4,5]
    assert contracts[0]['supervised_answer_roles']==[1,3,4]
    assert contracts[0]['target_source']=='fixed_initial_greedy_ids'
    assert batch['answer_roles'].shape==batch['input_ids'].shape
    replies['q']['training_target_ids']=[1,4,6]
    with pytest.raises(ValueError,match='changed/re-tokenized'):
        encode_group(processor,[{'question_id':'q'}],[0],{},source='replay',condition='teacher_ce32',teacher_replies=replies)


def test_fixed_precision_budget_and_control_admission_reject_fallback(monkeypatch):
    root=Path(__file__).resolve().parents[1]
    config=json.loads((root/'configs/vlm_object_behavior_replay_v1.json').read_text())
    train.validate_config(config)
    for key,value in [('training_microbatch',2),('preflight_memory_cap_gib',66),('replay_lambda',4.),('replay_per_update',8)]:
        with pytest.raises(ValueError):train.validate_config({**config,key:value})
    for condition,seed in [('gt_ce32',0),('gt_ce32',200),('teacher_ce32',1)]:
        with pytest.raises(ValueError):train.validate_condition(condition,seed)
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda *a:66*2**30)
    monkeypatch.setattr(torch.cuda,'max_memory_reserved',lambda *a:66*2**30)
    with pytest.raises(RuntimeError,match='cap_exceeded'):
        train.enforce_memory_cap(torch.device('cuda'),config,'preflight')
