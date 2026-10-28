"""Coverage budgets, immutable reference mapping, real QA means and resume."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import torch

from vlm_object_replay_coverage import train as trainer
from vlm_object_replay_coverage.reference import (TeacherSource, create_teacher_source,
    make_reference_map, schedule_for_condition, source_microbatch)
from vlm_object_kl import inputs
from vlm_object_kl.cache import TeacherCache, build_teacher_cache
from vlm_object_train.schedule import build_schedule as old_schedule, coverage
from test_object_presence_training import (FakeProcessor, TinyLM, rows, variable_fake_encode,
    config as object_config)


def config():
    return {**object_config(), 'schema_version': trainer.SCHEMA, 'replay_per_update': 32,
        'kl_temperature': 1., 'kl_direction': 'initial_to_actor', 'teacher_logprob_dtype': 'float32',
        'teacher_full_vocabulary': True, 'initial_kl_abs_tolerance': 1e-5,
        'teacher_cache_disk_reserve_bytes': 1073741824, 'preflight_memory_cap_gib': 65.,
        'teacher_reference_study_root': '/opt/data/private/test-only-original-KL-bank',
        'teacher_reference_condition': 'object_kl', 'standalone_reference_scope': 'engineering_first_update_only'}


def binding(model, blocks, replays, replay_per_update):
    qids = [[str(replays[i]['question_id']) for i in b['replay_indices']] for b in blocks]
    return {'initial_adapter_tensor_sha256': trainer.trainable_hash(model),
        'frozen_parameter_sha256_before': trainer.frozen_hash(model),
        'training_microbatch_questions': 4, 'replay_questions_per_update': replay_per_update,
        'planned_updates': len(blocks), 'planned_replay_exposures': len(blocks) * replay_per_update,
        'replay_order_sha256': trainer.content_sha(qids),
        'replay_microbatch_order_sha256': trainer.content_sha([[ids[i:i+4] for i in range(0,len(ids),4)] for ids in qids]),
        'hardware': {'uuid': 'CPU-contract-no-GPU'}, 'initial_adapter_files_sha256': {'test': 'not-a-production-checkpoint'}}


@pytest.fixture
def prepared(monkeypatch):
    monkeypatch.setattr(inputs, 'encode_qa', variable_fake_encode)
    torch.manual_seed(91)
    model = TinyLM(); model.config = SimpleNamespace(vocab_size=9)
    return model, rows(24), rows(100, 'replay'), config()


def test_equal_5792_exposures_but_distinct5792_vs1448_and_same_object_stream():
    objects, replays = rows(2888), rows(32985,'replay')
    wide = schedule_for_condition(objects,replays,42,condition='wide32')
    repeat = schedule_for_condition(objects,replays,42,condition='repeat8x4')
    old = old_schedule(objects,replays,42)
    assert len(wide)==len(repeat)==181 and len(wide[-1]['object_indices'])==8
    assert [b['object_indices'] for b in wide]==[b['object_indices'] for b in repeat]==[b['object_indices'] for b in old]
    wi=[i for b in wide for i in b['replay_indices']];ri=[i for b in repeat for i in b['replay_indices']]
    assert len(wi)==len(ri)==5792 and len(set(wi))==5792 and len(set(ri))==1448
    assert all(b['replay_indices']==o['replay_indices']*4 for b,o in zip(repeat,old))
    assert coverage(replays,wi)['unique_questions_consumed']==5792
    assert coverage(replays,ri)['unique_questions_consumed']==1448
    assert wide==schedule_for_condition(objects,replays,42,condition='wide32')
    with pytest.raises(ValueError,match='seed42'):
        schedule_for_condition(objects,replays,0,condition='repeat8x4')


def test_repeat_reference_maps8_actor_microbatches_to2_original_without_copying_probabilities(tmp_path,prepared):
    model,objects,replays,settings=prepared
    old_blocks=old_schedule(objects,replays,42)
    b=binding(model,old_blocks,replays,8)
    d=tmp_path/'original-bank'
    manifest=build_teacher_cache(model,FakeProcessor(),replays,old_blocks,settings,d,b,torch.device('cpu'))
    bank=TeacherCache(d,b,trainer.sha_file(d/'MANIFEST.json'))
    new_blocks=schedule_for_condition(objects,replays,42,condition='repeat8x4')
    mapping=make_reference_map('repeat8x4',new_blocks,replays,manifest)
    teacher=TeacherSource(bank,{'scope':'formal_full_stream'},mapping)
    before=trainer.file_hashes(d)
    for update,block in enumerate(new_blocks):
        for micro in range(8):
            selected=block['replay_indices'][micro*4:micro*4+4]
            batch,contracts,_=inputs.encode_microbatch(FakeProcessor(),replays,selected,settings)
            actual,counts=teacher.load(update,micro,batch,contracts,torch.device('cpu'))
            direct,direct_counts=bank.load(update,micro%2,batch,contracts,torch.device('cpu'))
            assert torch.equal(actual,direct) and torch.equal(counts,direct_counts)
    assert trainer.file_hashes(d)==before
    assert [source_microbatch('repeat8x4',i) for i in range(8)]==[0,1,0,1,0,1,0,1]
    with pytest.raises(ValueError):source_microbatch('repeat8x4',8)
    altered=deepcopy(contracts);altered[0]['microbatch_input_sha256']='changed-labels-padding-pixels-or-grid'
    with pytest.raises(ValueError,match='differ'):
        teacher.load(update,7,batch,altered,torch.device('cpu'))


def test_standalone_fresh_reference_builds_only_first32_and_exactly_restores_initial(tmp_path,prepared):
    model,objects,replays,settings=prepared
    blocks=schedule_for_condition(objects,replays,42,condition='wide32')
    b=binding(model,blocks,replays,32);state=deepcopy(model.state_dict());random_before=trainer.rng_hash(trainer.rng_state())
    teacher,record,manifest=create_teacher_source(model,FakeProcessor(),replays,blocks,settings,tmp_path,b,
        torch.device('cpu'),condition='wide32',preflight_only=True)
    assert record['scope']=='engineering_first_update_only'
    assert record['actual_new_teacher_forward_questions']==manifest['teacher_forward_questions']==32
    assert record['actor_replay_questions_planned']==record['actor_replay_unique_questions_planned']==32
    assert len(manifest['entries'])==8 and len(blocks)==2
    assert trainer.rng_hash(trainer.rng_state())==random_before
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in state.items())
    assert model.rope_deltas is None
    result=trainer.preflight(model,[p for p in model.parameters() if p.requires_grad],FakeProcessor(),objects,replays,
        blocks[0],settings,'wide32',tmp_path,torch.device('cpu'),teacher_cache=teacher)
    assert result['status']=='passed' and result['discarded_engineering_optimizer_updates']==1
    assert result['actual_new_teacher_forward_questions']==32 and result['reference_scope']=='engineering_first_update_only'
    assert len(result['record']['initial_replay_kl_per_question'])==32
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in state.items())


def fresh(tmp_path,prepared):
    model,objects,replays,settings=prepared
    blocks=schedule_for_condition(objects,replays,42,condition='wide32')
    b=binding(model,blocks,replays,32)
    teacher,_,_=create_teacher_source(model,FakeProcessor(),replays,blocks,settings,tmp_path,b,
        torch.device('cpu'),condition='wide32',preflight_only=False)
    return blocks,teacher


def test_short_tail8_object_and32_replay_have_independent_QA_mean_gradients(tmp_path,prepared):
    model,objects,replays,settings=prepared
    blocks,teacher=fresh(tmp_path,prepared)
    with torch.no_grad():model.head.lora_weight.add_(torch.randn_like(model.head.lora_weight)*.03)
    accumulated=model;expected=deepcopy(model)
    stats=trainer.initial_stats()
    trainer.backward_update(accumulated,FakeProcessor(),objects,replays,blocks[-1],'wide32',settings,
        torch.device('cpu'),stats,teacher_cache=teacher,update=1)
    # Reassemble exactly the independent object mean + full32-question KL mean,
    # without calling the microbatch normalization helper under test.
    objective=0.
    for source,data,indices in [('object',objects,blocks[-1]['object_indices']),('replay',replays,blocks[-1]['replay_indices'])]:
        for micro,start in enumerate(range(0,len(indices),4)):
            batch,contracts,_=inputs.encode_microbatch(FakeProcessor(),data,indices[start:start+4],settings)
            if source=='object':
                lp,counts=trainer.sequence_logps(expected,batch);objective+=(-lp/counts).sum()/8
            else:
                actor,counts,_=trainer.assistant_logprobs(expected,batch)
                reference,reference_counts=teacher.load(1,micro,batch,contracts,torch.device('cpu'))
                objective+=trainer.per_question_kl(actor,reference,counts).sum()/32
    objective.backward()
    assert torch.allclose(accumulated.head.lora_weight.grad,expected.head.lora_weight.grad,atol=3e-7,rtol=3e-6)
    assert stats['object_exposures']==8 and stats['replay_exposures']==32
    assert stats['object_microbatches']==2 and stats['replay_microbatches']==8


def test_full32_replay_continuous_and_checkpoint_resume_have_identical_Adam_cursor(tmp_path,prepared):
    initial,objects,replays,settings=prepared
    blocks,teacher=fresh(tmp_path/'bank',prepared)
    def optimizer(model):return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=settings['lr'],weight_decay=0.)
    def execute(model,opt,stats,begin,end):
        for step in range(begin,end):
            opt.zero_grad(set_to_none=True)
            for group in opt.param_groups:group['lr']=trainer.learning_rate(step,len(blocks),settings)
            trainer.backward_update(model,FakeProcessor(),objects,replays,blocks[step],'wide32',settings,torch.device('cpu'),stats,teacher_cache=teacher,update=step)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
            opt.step()
    full,part=deepcopy(initial),deepcopy(initial);opt_a,opt_b=optimizer(full),optimizer(part)
    sa,sb=trainer.initial_stats(),trainer.initial_stats();execute(full,opt_a,sa,0,2);execute(part,opt_b,sb,0,1)
    checkpoint=tmp_path/'checkpoints';checkpoint.mkdir()
    trainer.save_checkpoint(checkpoint,part,opt_b,1,sb,'locked-source-reference-UUID','schedule32')
    restored=deepcopy(initial);opt_c=optimizer(restored)
    cursor,sc=trainer.resume_checkpoint(checkpoint,restored,opt_c,'locked-source-reference-UUID','schedule32',2)
    trainer.validate_exposure_cursor(sc,blocks,cursor,objects,replays,'wide32')
    execute(restored,opt_c,sc,cursor,2)
    assert trainer.trainable_hash(full)==trainer.trainable_hash(restored) and sa==sc
    assert sc['object_exposures']==24 and sc['replay_exposures']==64
    trainer.validate_exposure_cursor(sc,blocks,2,objects,replays,'wide32')
    with pytest.raises(ValueError,match='binding changed'):
        trainer.resume_checkpoint(checkpoint,restored,opt_c,'changed-reference','schedule32',2)


def test_fixed_config_accepts32_and_rejects_hidden_budget_or_reference_scope_changes():
    assert trainer.validate_config(config())==config()
    for key,value in [('replay_per_update',8),('replay_lambda',2.),('training_microbatch',1),
                       ('standalone_reference_scope','full5792'),('teacher_reference_condition','actor')]:
        with pytest.raises(ValueError):trainer.validate_config({**config(),key:value})
