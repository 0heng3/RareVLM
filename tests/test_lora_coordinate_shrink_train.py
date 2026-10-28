"""CPU mathematical and restoration contracts; never load a research model."""
import copy
import json
import math
import random
import types
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from vlm_lora_coordinate_shrink import train
from vlm_lora_coordinate_shrink.loss import (assistant_logprobs, completion_ce, normalized_qa_loss,
    per_question_kl, replay_diagnostics, sequence_ce)
from vlm_object_behavior_replay.loss import sequence_ce as parent_ce


class ContractActor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_head = torch.nn.Linear(7, 7, bias=False)
        self.frozen = torch.nn.Parameter(torch.tensor([2.]), requires_grad=False)
        self.register_buffer('seen', torch.tensor(0))
        self.calls = 0
        self.is_gradient_checkpointing = True
        self.config = types.SimpleNamespace(use_cache=False)

    def get_output_embeddings(self):
        return self.lora_head

    def gradient_checkpointing_disable(self):
        self.is_gradient_checkpointing = False

    def disable_input_require_grads(self):
        pass

    def forward(self, input_ids, attention_mask, use_cache=False):
        assert not use_cache
        self.calls += 1
        self.seen.add_(1)
        self.rope_deltas = torch.tensor([[self.calls]])
        return types.SimpleNamespace(logits=self.lora_head(F.one_hot(input_ids,7).float()))

    def save_pretrained(self, directory, safe_serialization=True):
        directory=Path(directory)
        directory.mkdir(parents=True)
        (directory/'adapter.json').write_text(json.dumps(self.lora_head.weight.detach().tolist()))


def sample(i=0):
    # Variable prompt lengths, left padding, ordinary content, actual EOS/template,
    # empty-content EOS, and capped content without EOS all retain their labels.
    ids=torch.tensor([[0,1+(i%3),4,5]])
    labels=ids.clone(); labels[:,0]=-100
    roles=torch.tensor([[0,1,3,4]])
    mask=torch.ones_like(ids)
    if i%3==1:
        labels[:,:2]=-100; roles=torch.tensor([[0,0,3,4]]); mask[:,0]=0
    elif i%3==2:
        roles=torch.tensor([[0,1,2,2]])
    return {'input_ids':ids,'attention_mask':mask,'labels':labels,'answer_roles':roles}


def collate(samples):
    return {key:torch.cat([s[key] for s in samples]) for key in samples[0]}


def frozen_reference(actor):
    ref=copy.deepcopy(actor)
    ref.requires_grad_(False);ref.eval();ref.gradient_checkpointing_disable()
    state=train.reference_state(ref)
    binding={'initial_lora_tensor_sha256':state['lora_tensor_sha256'],
             'initial_base_parameter_sha256':state['base_parameter_sha256'],
             'initial_all_parameter_sha256':state['all_parameter_sha256']}
    return ref,binding


def test_unchanged_object_ce_and_causal_full_vocabulary_rows():
    torch.manual_seed(11)
    actor=ContractActor();old=copy.deepcopy(actor)
    batch=collate([sample(i) for i in range(4)])
    ce,counts,roles=sequence_ce(actor,batch)
    old_ce,old_counts,old_roles=parent_ce(old,batch)
    assert torch.equal(ce,old_ce) and torch.equal(counts,old_counts) and roles==old_roles
    ce.mean().backward();old_ce.mean().backward()
    assert torch.equal(actor.lora_head.weight.grad,old.lora_head.weight.grad)
    logp,n,targets,role_ids=assistant_logprobs(actor,batch,7)
    full=actor.lora_head(F.one_hot(batch['input_ids'],7).float())[:,:-1,:].float()
    valid=batch['labels'][:,1:]!=-100
    assert torch.equal(logp,F.log_softmax(full[valid],-1))
    assert torch.equal(targets,batch['labels'][:,1:][valid])
    assert torch.equal(role_ids,batch['answer_roles'][:,1:][valid])
    assert int(n.sum())==11 and logp.shape==(11,7)
    with pytest.raises(ValueError,match='vocabulary'):
        assistant_logprobs(actor,batch,6)
    bad={**batch,'answer_roles':batch['answer_roles'].clone()};bad['answer_roles'][0,0]=1
    with pytest.raises(ValueError,match='roles'):
        assistant_logprobs(actor,bad,7)
    # A supervised first position has no causal predecessor and must be rejected.
    bad={**batch,'labels':batch['labels'].clone(),'answer_roles':batch['answer_roles'].clone()}
    bad['labels'][0,0]=0;bad['answer_roles'][0,0]=2
    with pytest.raises(ValueError,match='causal'):
        assistant_logprobs(actor,bad,7)


def test_forward_kl_direction_vocab_sum_qa_mean_and_teacher_detachment():
    actor_logits=torch.tensor([[.3,-.2,1.],[-.8,.7,.1],[.4,.2,-.4]],requires_grad=True)
    teacher_logits=torch.tensor([[-.2,.5,.1],[.8,-.4,.1],[.3,-.7,.8]],requires_grad=True)
    actor=F.log_softmax(actor_logits,-1);teacher=F.log_softmax(teacher_logits,-1)
    counts=torch.tensor([1,2])
    qa,token=per_question_kl(actor,teacher,counts)
    expected=(teacher.detach().exp()*(teacher.detach()-actor)).sum(-1)
    assert torch.allclose(token,expected) and torch.allclose(qa,torch.stack([expected[0],expected[1:].mean()]))
    reverse=(actor.exp()*(actor-teacher.detach())).sum(-1)
    assert not torch.allclose(token,reverse)
    assert not torch.allclose(token,expected/3)
    assert not torch.isclose(qa.mean(),token.mean())
    qa.mean().backward()
    assert teacher_logits.grad is None
    expected_grad=(actor.detach().exp()-teacher.detach().exp())*torch.tensor([[.5],[.25],[.25]])
    assert torch.allclose(actor_logits.grad,expected_grad,atol=8e-8)
    with pytest.raises(ValueError,match='FP32'):
        per_question_kl(actor.double(),teacher.double(),counts)
    with pytest.raises(ValueError):
        per_question_kl(actor,teacher,torch.tensor([3,0]))


def test_initial_self_distribution_zero_gradient_and_detached_role_ce():
    torch.manual_seed(3)
    logits=torch.randn(5,7,requires_grad=True)
    logp=F.log_softmax(logits,-1)
    qa,token=per_question_kl(logp,logp.clone(),torch.tensor([2,3]))
    assert qa.abs().max()==0
    qa.mean().backward()
    assert logits.grad.abs().max()<1e-7
    ce,ce_roles,kl_roles=replay_diagnostics(logp,torch.tensor([1,4,2,4,5]),
        torch.tensor([2,3]),torch.tensor([1,3,1,3,4]),token)
    assert not ce.requires_grad and ce.mean()>0
    assert sum(v['token_count'] for v in ce_roles.values())==5
    assert sum(v['token_count'] for v in kl_roles.values())==5
    assert ce_roles['actual_eos']['questions_with_role']==2


def test_micro4_last_object8_replay32_matches_independent_qa_gradient():
    torch.manual_seed(5)
    actor=ContractActor();expected_actor=copy.deepcopy(actor)
    ref,_=frozen_reference(actor)
    # Perturb only actor, so preservation gradients are nonzero.
    with torch.no_grad():actor.lora_head.weight.add_(.03*torch.randn_like(actor.lora_head.weight))
    expected_actor.load_state_dict(actor.state_dict())
    rows={'object':[sample(i) for i in range(8)],'replay':[sample(i+1) for i in range(32)]}
    expected_loss=0
    for source,samples in rows.items():
        for start in range(0,len(samples),4):
            batch=collate(samples[start:start+4])
            if source=='object': values,_,_=sequence_ce(actor,batch)
            else:
                a,c,_,_=assistant_logprobs(actor,batch,7)
                with torch.no_grad():t,_,_,_=assistant_logprobs(ref,batch,7)
                values,_=per_question_kl(a,t,c)
            normalized_qa_loss(values,source,8,32,1.).backward()
        batch=collate(samples)
        if source=='object':values,_,_=sequence_ce(expected_actor,batch)
        else:
            a,c,_,_=assistant_logprobs(expected_actor,batch,7)
            with torch.no_grad():t,_,_,_=assistant_logprobs(ref,batch,7)
            values,_=per_question_kl(a,t,c)
        expected_loss=expected_loss+values.mean()
    expected_loss.backward()
    assert torch.allclose(actor.lora_head.weight.grad,expected_actor.lora_head.weight.grad,atol=8e-8,rtol=1e-6)
    assert all(p.grad is None for p in ref.parameters())


def fake_stream(blocks):
    objects=[{'question_id':str(i),'sample':sample(i)} for i in range(sum(len(b['object_indices']) for b in blocks))]
    replays=[{'question_id':str(i),'sample':sample(i+1)} for i in range(sum(len(b['replay_indices']) for b in blocks))]
    def encode(processor,rows,selected,config,**kwargs):
        samples=[rows[i]['sample'] for i in selected]
        contracts=[{'question_id':rows[i]['question_id'],'assistant_tokens':int((s['labels']!=-100).sum()),
                    'input_tokens':int(s['attention_mask'].sum()),'visual_tokens':1} for i,s in zip(selected,samples)]
        return collate(samples),contracts,samples
    context={'teacher_replies':{},'plan':{'seed':42,'blocks':blocks},'contracts':{'object':[],'replay':[]}}
    for b in blocks:
        for source,rows in [('object',objects),('replay',replays)]:
            for start in range(0,len(b[f'{source}_indices']),4):
                context['contracts'][source].extend(encode(None,rows,b[f'{source}_indices'][start:start+4],{})[1])
    return objects,replays,context,encode


@pytest.mark.parametrize('mismatch',[False,True])
def test_embedded_preflight_zero_initial_kl_or_reject_restores_all_state(tmp_path,monkeypatch,mismatch):
    block={'object_indices':list(range(16)),'replay_indices':list(range(32))}
    objects,replays,context,encoder=fake_stream([block])
    monkeypatch.setattr(train,'encode_group',encoder)
    monkeypatch.setattr(train,'verify_prepared_files_unchanged',lambda *a,**k:True)
    torch.manual_seed(21)
    model=ContractActor();ref,binding=frozen_reference(model)
    if mismatch:
        with torch.no_grad():ref.lora_head.weight[0,0].add_(.4)
        state=train.reference_state(ref)
        binding.update(initial_lora_tensor_sha256=state['lora_tensor_sha256'],initial_all_parameter_sha256=state['all_parameter_sha256'])
    before=train.trainable_hash(model);rng=train.rng_hash(train.rng_state())
    config={'preflight_lr':1e-4,'lr':5e-6,'weight_decay':0.,'gradient_clip_norm':1.,'replay_lambda':1.,
        'training_microbatch':4,'preflight_memory_cap_gib':65,
        'teacher_reference':{'initial_max_abs_per_question_KL':1e-5,'initial_replay_to_object_gradient_norm_max_ratio':.01}}
    processor=types.SimpleNamespace(tokenizer=types.SimpleNamespace(pad_token_id=0))
    call=lambda:train.preflight(model,[p for p in model.parameters() if p.requires_grad],processor,objects,replays,block,
        config,'lora_half32',tmp_path,torch.device('cpu'),context,ref,binding,7,lineage_sha256='lineage')
    if mismatch:
        with pytest.raises(ValueError,match='parity/gradient'):call()
        result=json.loads((tmp_path/'PREFLIGHT.json').read_text())
        assert result['status']=='failed' and result['discarded_engineering_optimizer_updates']==0
        assert result['record']['initial_max_abs_per_question_kl']>1e-5
    else:
        result=call()
        assert result['status']=='passed'
        assert result['record']['initial_max_abs_per_question_kl']==0
        assert result['record']['initial_replay_to_object_gradient_norm_ratio']<1e-5
        assert result['record']['post_update_mean_replay_kl']>0
        assert result['record']['post_update_replay_gradient_l2']>0
        assert result['engineering_actor_forward_backward_questions']==80
        assert result['engineering_reference_forward_questions']==64
        train.verify_preflight_record(result,before,'lineage')
        bad=copy.deepcopy(result);bad['record']['initial_replay_to_object_gradient_norm_ratio']=.010001
        with pytest.raises(ValueError,match='admission'):train.verify_preflight_record(bad,before,'lineage')
    assert train.trainable_hash(model)==before and train.rng_hash(train.rng_state())==rng
    assert model.seen.item()==ref.seen.item()==0
    assert not hasattr(model,'rope_deltas') and not hasattr(ref,'rope_deltas')
    assert all(p.grad is None for p in list(model.parameters())+list(ref.parameters()))
    assert result['initial_parameters_restored_exactly'] and result['rng_restored']
    assert result['reference_runtime_buffers_and_rope_cache_restored']


def test_fixed_reference_construction_rng_storage_and_initial_source(monkeypatch):
    torch.manual_seed(8)
    actor=ContractActor()
    initial=train.trainable_hash(actor);before=train.rng_hash(train.rng_state())
    monkeypatch.setattr(train,'INITIAL_MASTER_SHA256',initial)
    def loader(*args):
        torch.rand(7);random.random()
        return copy.deepcopy(actor),[],{'lora_elements_by_dtype':{'float32':49}}
    monkeypatch.setattr(train,'load_actor',loader)
    ref,binding=train.load_reference({'teacher_reference':{}},{},torch.device('cpu'),actor,initial,train.frozen_hash(actor))
    assert train.rng_hash(train.rng_state())==before
    assert not train.storage_is_shared(actor,ref)
    assert binding['initial_lora_tensor_sha256']==initial
    train.assert_reference_fixed(ref,binding)
    with torch.no_grad():ref.lora_head.weight[0,0].add_(.1)
    with pytest.raises(ValueError,match='hash changed'):train.assert_reference_fixed(ref,binding)
    monkeypatch.setattr(train,'load_actor',lambda *a:(actor,[],{}))
    with pytest.raises(ValueError,match='storage'):train.load_reference({'teacher_reference':{}},{},torch.device('cpu'),actor,initial,train.frozen_hash(actor))
    assert actor.lora_head.weight.requires_grad


def test_checkpoint_resume_preserves_kl_reference_input_and_adam_cursor(tmp_path,monkeypatch):
    blocks=[{'object_indices':list(range(4)),'replay_indices':list(range(4))},
            {'object_indices':list(range(4,8)),'replay_indices':list(range(4,8))}]
    objects,replays,context,encoder=fake_stream(blocks)
    monkeypatch.setattr(train,'encode_group',encoder)
    torch.manual_seed(7)
    base=ContractActor();full=copy.deepcopy(base);part=copy.deepcopy(base);ref,_=frozen_reference(base)
    opt1=torch.optim.AdamW([full.lora_head.weight],lr=5e-6,weight_decay=0.)
    opt2=torch.optim.AdamW([part.lora_head.weight],lr=5e-6,weight_decay=0.)
    config={'training_microbatch':4,'replay_lambda':1.,'preflight_memory_cap_gib':65}
    def update(model,opt,stats,step):
        opt.zero_grad(set_to_none=True)
        train.backward_update(model,None,objects,replays,blocks[step],'lora_half32',config,torch.device('cpu'),context,
            stats,update=step,reference_model=ref,vocabulary_size=7)
        torch.nn.utils.clip_grad_norm_([model.lora_head.weight],1.)
        opt.step()
    stats1,stats2=train.initial_stats(),train.initial_stats()
    for step in range(2):update(full,opt1,stats1,step)
    update(part,opt2,stats2,0)
    train.save_checkpoint(tmp_path,part,opt2,1,stats2,'lineage','schedule')
    restored=copy.deepcopy(base);opt3=torch.optim.AdamW([restored.lora_head.weight],lr=5e-6,weight_decay=0.)
    step,stats3=train.resume_checkpoint(tmp_path,restored,opt3,'lineage','schedule',2)
    train.validate_exposure_cursor(stats3,blocks,step,context)
    update(restored,opt3,stats3,step)
    assert torch.equal(full.lora_head.weight,restored.lora_head.weight)
    stats1.pop('reference_forward_seconds');stats3.pop('reference_forward_seconds')
    assert stats1==stats3
    train.validate_exposure_cursor(stats3,blocks,2,context)
    bad=copy.deepcopy(stats3);bad['reference_forward_questions']-=1
    with pytest.raises(ValueError,match='reference'):train.validate_exposure_cursor(bad,blocks,2,context)
    with pytest.raises(ValueError,match='binding'):train.resume_checkpoint(tmp_path,restored,opt3,'other','schedule',2)


def test_fixed_precision_reference_budget_and_no_standalone_or_control_fallback(monkeypatch):
    root=Path(__file__).resolve().parents[1]
    config=json.loads((root/'configs/vlm_lora_coordinate_shrink_v1.json').read_text())
    train.validate_config(config)
    for key,value in [('training_microbatch',2),('preflight_memory_cap_gib',66),('replay_lambda',4.),('replay_per_update',8)]:
        with pytest.raises(ValueError):train.validate_config({**config,key:value})
    for key,value in [('mode','train'),('full_vocabulary',False),('initial_max_abs_per_question_KL',1e-4),('grad_enabled',True)]:
        bad=copy.deepcopy(config);bad['teacher_reference'][key]=value
        with pytest.raises(ValueError):train.validate_config(bad)
    for condition,seed in [('teacher_ce32',42),('gt_ce32',42),('lora_half32',42),('lora_half32',1)]:
        with pytest.raises(ValueError):train.validate_condition(condition,seed)
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda *a:66*2**30)
    monkeypatch.setattr(torch.cuda,'max_memory_reserved',lambda *a:66*2**30)
    with pytest.raises(RuntimeError,match='cap_exceeded'):
        train.enforce_memory_cap(torch.device('cuda'),config,'reference_forward')


def test_parent_parameter_anchor_rejects_unbound_or_wrong_frozen_identity(tmp_path):
    path=tmp_path/'independent_review/TRAINING_COMPLETION_REVIEW.json'
    path.parent.mkdir()
    review={'status':'passed_training_engineering_only',
        'shared_initial':{'canonical_master_sha256':train.INITIAL_MASTER_SHA256},
        'conditions':[{'condition':'teacher_ce32','formal_updates':181,
            'initial_canonical_master_sha256':train.INITIAL_MASTER_SHA256,
            'saved_frozen_base_vision_before_after_sha256':'47486b9a45c127a55223b496f9847e45b2a181dd225697d7d98b274a67ade929'}]}
    path.write_text(json.dumps(review))
    config={'parent_output_root':str(tmp_path),'parent_proof_files_sha256':{str(path):train.sha_file(path)}}
    anchor=train.parent_parameter_anchor(config)
    assert anchor['initial_canonical_master_sha256']==train.INITIAL_MASTER_SHA256
    review['conditions'][0]['saved_frozen_base_vision_before_after_sha256']='changed'
    path.write_text(json.dumps(review))
    with pytest.raises(ValueError,match='file SHA'):train.parent_parameter_anchor(config)
    config['parent_proof_files_sha256'][str(path)]=train.sha_file(path)
    with pytest.raises(ValueError,match='frozen base'):train.parent_parameter_anchor(config)


def test_stage32_numerical_loss_and_training_core_are_explicit_copies():
    import ast
    root=Path(__file__).resolve().parents[1]/'src'
    for file,names in [('loss.py',['completion_ce','sequence_ce','assistant_logprobs','per_question_kl','replay_diagnostics']),
            ('train.py',['enforce_memory_cap','backward_update','runtime_matches','gradient_norm','load_reference','assert_reference_fixed'])]:
        old=ast.parse((root/'vlm_generated_prefix_kl'/file).read_text())
        new=ast.parse((root/'vlm_lora_coordinate_shrink'/file).read_text())
        for name in names:
            a=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name==name)
            b=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name==name)
            assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False),name


def test_seed42_cli_rejected_before_admission_or_model_load(tmp_path,monkeypatch):
    from vlm_lora_coordinate_shrink import data
    root=Path(__file__).resolve().parents[1]
    config_file=root/'configs/vlm_lora_coordinate_shrink_v1.json'
    calls=[]
    monkeypatch.setattr(data,'require_replication_admission',lambda *a:calls.append('admission'))
    monkeypatch.setattr(train,'load_actor',lambda *a:calls.append('model'))
    args=types.SimpleNamespace(config=str(config_file),condition='lora_half32',seed=42,output_root=str(tmp_path),resume=False)
    with pytest.raises(ValueError,match='historical only'):train.run(args)
    assert calls==[]
    import sys
    monkeypatch.setattr(sys,'argv',['producer','--config',str(config_file),'--condition','lora_half32',
        '--seed','42','--output-root',str(tmp_path)])
    with pytest.raises(ValueError,match='historical only'):train.main()
    assert not (tmp_path/'producer_runs').exists() and calls==[]
