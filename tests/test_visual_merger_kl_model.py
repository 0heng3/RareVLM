"""CPU/meta structural contracts; never load the 3B checkpoint or a CUDA model."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vlm_visual_merger_kl import model as m


@pytest.fixture(scope='module')
def skeleton():
    from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
    root=Path(__file__).resolve().parents[1]
    config=json.loads((root/'configs/vlm_visual_merger_kl_v1.json').read_text())
    cfg=Qwen2_5_VLConfig.from_pretrained(config['model_root'],local_files_only=True)
    with torch.device('meta'):
        bare=Qwen2_5_VLForConditionalGeneration(cfg)
        actor=m.build_composite_adapter(bare)
    return actor


def test_real_architecture_meta_exact_inventory_active_adapter_and_dtype(skeleton):
    language,visual,all_lora=m.lora_partition(skeleton,True)
    assert len(language)==144 and len(visual)==4 and len(all_lora)==148
    assert sum(p.numel() for p in visual.values())==1843200
    assert sum(p.numel() for p in all_lora.values())==3686400
    assert all(not p.requires_grad for p in language.values())
    assert all(p.dtype==torch.float32 and p.requires_grad for p in visual.values())
    assert all(p.device.type=='meta' for p in skeleton.parameters())
    with pytest.raises(ValueError,match='injected once'):
        m.build_composite_adapter(skeleton)
    modules=dict(skeleton.named_modules())
    first=modules['base_model.model.visual.merger.mlp.0']
    assert first.r['default']==96 and first.lora_alpha['default']==192
    assert first.scaling['default']==2 and first.active_adapters==['default']
    assert modules['base_model.model.visual.merger.mlp.2'].r['default']==120


def test_saved_composite_config_exact_targets_patterns_and_no_extra_mechanism(skeleton):
    config=skeleton.peft_config['default'].to_dict()
    m.validate_composite_adapter_config(config)
    for key,value in [('target_modules',['q_proj','v_proj']),('rank_pattern',{}),('alpha_pattern',{}),
                      ('use_dora',True),('use_rslora',True),('modules_to_save',['lm_head']),('lora_dropout',.1)]:
        bad=copy.deepcopy(config);bad[key]=value
        with pytest.raises(ValueError):m.validate_composite_adapter_config(bad)
    bad=copy.deepcopy(config);bad['target_modules']=list(bad['target_modules'])+['visual.blocks.0.attn.qkv']
    with pytest.raises(ValueError):m.validate_composite_adapter_config(bad)


def test_language_copy_exact_keyset_shape_and_bf16_canonical_rounding(skeleton):
    language,_,_=m.lora_partition(skeleton)
    class Parameters:
        def named_parameters(self):return iter(self.weights.items())
        def named_modules(self):return iter([])
    cpu=Parameters()
    cpu.weights={n:torch.nn.Parameter(torch.empty(tuple(p.shape)),requires_grad=False)
                 for n,p in skeleton.named_parameters() if 'lora_' in n}
    generator=torch.Generator().manual_seed(8)
    raw={m.safe_name(n):torch.randn(tuple(p.shape),generator=generator)*.01 for n,p in language.items()}
    digest=m.copy_language_initial(cpu,raw,expected_sha256=None)
    expected=[(n,raw[m.safe_name(n)].bfloat16().float()) for n in language]
    assert digest==m.tensor_hash(expected)
    assert all(torch.equal(cpu.weights[n],value) for n,value in expected)
    with pytest.raises(ValueError,match='keyset'):
        m.copy_language_initial(cpu,{**raw,'unrelated':torch.ones(1)},None)
    bad=dict(raw);bad[next(iter(bad))]=torch.ones(1)
    with pytest.raises(ValueError,match='shape/finite'):
        m.copy_language_initial(cpu,bad,None)
    with pytest.raises(ValueError,match='canonical'):
        m.copy_language_initial(cpu,raw,'wrong')


def test_only_declared_base_names_are_canonicalized_and_value_changes_detected():
    names=['base_model.model.visual.merger.mlp.0.base_layer.weight',
           'base_model.model.visual.merger.mlp.2.base_layer.bias',
           'base_model.model.model.layers.0.self_attn.q_proj.base_layer.weight']
    assert m.canonical_base_name(names[0])=='base_model.model.visual.merger.mlp.0.weight'
    assert m.canonical_base_name(names[1])=='base_model.model.visual.merger.mlp.2.bias'
    assert m.canonical_base_name(names[2])==names[2]
    class Parameters:
        def __init__(self,items):self.items=items
        def named_parameters(self):return iter(self.items)
    actor=Parameters([(n,torch.ones(2,dtype=torch.bfloat16)*i) for i,n in enumerate(names)])
    ref=Parameters([(m.canonical_base_name(n),p.clone()) for n,p in actor.items])
    a,r=m.base_identity(actor),m.base_identity(ref)
    assert a['raw_named_base_parameter_sha256']!=r['raw_named_base_parameter_sha256']
    assert a['canonical_base_parameter_sha256']==r['canonical_base_parameter_sha256']
    ref.items[0][1].add_(1)
    assert m.base_identity(actor)['canonical_base_parameter_sha256']!=m.base_identity(ref)['canonical_base_parameter_sha256']
    collision=Parameters(actor.items+[(m.canonical_base_name(names[0]),torch.ones(2))])
    with pytest.raises(ValueError,match='collision'):m.canonical_base_parameters(collision)


def test_zero_B_keeps_frozen_language_active_and_backpropagates_through_frozen_decoder():
    from peft import LoraConfig, get_peft_model
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__();self.merger=torch.nn.Linear(3,3);self.q_proj=torch.nn.Linear(3,3)
        def forward(self,x):return self.q_proj(self.merger(x)).square().mean()
    torch.manual_seed(2)
    actor=get_peft_model(Tiny(),LoraConfig(r=2,lora_alpha=4,lora_dropout=0.,target_modules=['merger','q_proj']))
    with torch.no_grad():actor.base_model.model.q_proj.lora_B.default.weight.fill_(.07)
    actor.requires_grad_(False)
    for n,p in actor.named_parameters():
        if '.merger.lora_' in n:p.requires_grad_(True)
    ref=copy.deepcopy(actor).requires_grad_(False)
    x=torch.randn(4,3)
    assert torch.equal(actor(x),ref(x))
    before={n:p.detach().clone() for n,p in actor.named_parameters() if '.q_proj.lora_' in n}
    actor(x).backward()
    a=actor.base_model.model.merger.lora_A.default.weight
    b=actor.base_model.model.merger.lora_B.default.weight
    assert a.grad is not None and torch.count_nonzero(a.grad)==0
    assert b.grad is not None and torch.linalg.vector_norm(b.grad)>0
    assert all(p.grad is None for n,p in actor.named_parameters() if '.q_proj.' in n)
    assert all(torch.equal(dict(actor.named_parameters())[n],p) for n,p in before.items())
    for layer in (actor.base_model.model.merger,actor.base_model.model.q_proj):
        assert layer.active_adapters==['default'] and not layer.disable_adapters


def test_checkpoint_saves_full148_but_only_four_master_and_adam_states(skeleton,tmp_path):
    from safetensors.torch import load_file
    from vlm_object_train.train import save_checkpoint,resume_checkpoint,trainable_hash
    # Materialize only synthetic LoRA factors; the real-sized base stays on meta.
    actor=copy.deepcopy(skeleton)
    for index,(name,p) in enumerate(list(actor.named_parameters())):
        if 'lora_' in name:
            module_name,attribute=name.rsplit('.',1)
            module=actor.get_submodule(module_name)
            setattr(module,attribute,torch.nn.Parameter(torch.full(tuple(p.shape),.001*(index%3+1)),
                                                       requires_grad=p.requires_grad))
    language,visual,all_lora=m.lora_partition(actor,True)
    frozen=m.tensor_hash(language.items())
    optimizer=torch.optim.AdamW(list(visual.values()),lr=5e-6,weight_decay=0.)
    for p in visual.values():p.grad=torch.ones_like(p)
    optimizer.step();optimizer.zero_grad(set_to_none=True)
    expected=trainable_hash(actor)
    save_checkpoint(tmp_path,actor,optimizer,1,{'committed':True},'lineage','schedule')
    latest=json.loads((tmp_path/'LATEST.json').read_text())
    checkpoint=tmp_path/latest['directory']
    saved=load_file(str(checkpoint/'adapter/adapter_model.safetensors'))
    assert len(saved)==148 and sum(p.numel() for p in saved.values())==3686400
    for name,p in all_lora.items():assert torch.equal(saved[m.safe_name(name)],p)
    state=torch.load(checkpoint/'state.pt',map_location='cpu',weights_only=False)
    assert set(state['trainable_parameters'])==set(m.VISUAL_NAMES)
    assert len(state['optimizer']['state'])==4
    assert all(v['step']==1 for v in state['optimizer']['state'].values())
    with torch.no_grad():
        for p in visual.values():p.add_(1.)
    other=torch.optim.AdamW(list(visual.values()),lr=5e-6,weight_decay=0.)
    step,stats=resume_checkpoint(tmp_path,actor,other,'lineage','schedule',181)
    assert step==1 and stats=={'committed':True} and trainable_hash(actor)==expected
    assert m.tensor_hash(language.items())==frozen and len(other.state)==4


def test_metadata_consumer_binds_visual_initial_frozen_language_reference_and_native_identity():
    from vlm_visual_merger_kl import train
    config={'visual_adaptation':{k:v for k,v in m.architecture().items()
            if k not in ('scaling','new_initial_B_zero','base_name_mapping_scope')}}
    dtype={'parameter_tensors_by_dtype':{'bfloat16':824,'float32':148},
        'parameter_stored_elements_by_dtype':{'bfloat16':3754622976,'float32':3686400},
        'lora_tensors_by_dtype':{'float32':148},'lora_elements_by_dtype':{'float32':3686400},
        'trainable_parameters':1843200,'params4bit_tensors':0,'buffer_tensors_by_dtype':{'float32':38}}
    common={'visual_fp32_tensor_sha256':'initial-visual4'}
    mapping={'exact_declared_bijection_and_values':True,'reference_original_base_parameter_sha256':m.REFERENCE_BASE_SHA256,
             'canonical_base_parameter_sha256':'mappedbase'}
    sig={'language_fp32_tensor_sha256':m.LANGUAGE_MASTER_SHA256,'visual_fp32_tensor_sha256':'initial-visual4'}
    initial={'architecture':m.architecture(),'adapter_signatures':sig,'common_initialization_identity':common,
        'common_initialization_identity_sha256':m.content_sha(common),'common_initialization_manifest_file':'common/MANIFEST.json',
        'common_initialization_manifest_file_sha256':'manifestSHA','frozen_base_mapping':mapping,
        'actual_parameter_dtypes':dtype,'language_gradients_absent':True}
    finalsig={**sig,'visual_fp32_tensor_sha256':'final-visual4','full_adapter_native_bf16_tensor_sha256':'native148',
              'full_adapter_bf16_promoted_fp32_tensor_sha256':'promoted148'}
    e={'architecture':m.architecture(),'frozen_language_canonical_fp32_tensor_sha256':m.LANGUAGE_MASTER_SHA256,
       'full_adapter_native_bf16_tensor_sha256':'native148','full_adapter_bf16_promoted_fp32_tensor_sha256':'promoted148',
       'canonical_base_parameter_sha256':'mappedbase'}
    after={**initial,'adapter_signatures':finalsig,'evaluation_identity':e}
    ref={'initial_lora_tensor_sha256':m.LANGUAGE_MASTER_SHA256,'initial_base_parameter_sha256':m.REFERENCE_BASE_SHA256,
         'initial_all_parameter_sha256':'ddcbe50ae8036d83cfff644bda9a23b2aa647652013f83bd75209274061677a8',
         'all_parameters_frozen':True,'parameter_storage_shared_with_actor':False,
         'actor_rng_restored_after_construction':True,'execution_path_identity_claim':False}
    refafter={'lora_tensor_sha256':ref['initial_lora_tensor_sha256'],'base_parameter_sha256':ref['initial_base_parameter_sha256'],
              'all_parameter_sha256':ref['initial_all_parameter_sha256'],'all_parameters_frozen':True,
              'gradients_absent':True,'training':False,'gradient_checkpointing':False}
    gradients={name:{'gradient_l2':0.,'finite':True,'nonzero_elements':0} for name in m.VISUAL_NAMES}
    pref={'schema_version':train.SCHEMA,'status':'passed','condition':train.CANDIDATE,
        'discarded_engineering_optimizer_updates':1,'formal_optimizer_updates':0,
        'engineering_actor_forward_backward_questions':80,'engineering_actor_training_forward_started_questions':80,
        'engineering_actor_forward_only_probe_questions':0,'teacher_generation_questions':0,
        'engineering_reference_forward_questions':64,'engineering_reference_forward_started_questions':64,
        'reference_rng_unchanged_microbatches':16,'reference_rng_unchanged':True,'reference_parameters_unchanged':True,
        'reference_runtime_buffers_and_rope_cache_restored':True,'initial_parameters_restored_exactly':True,'rng_restored':True,
        'frozen_language_restored_exactly':True,'frozen_language_gradients_absent':True,
        'frozen_language_sha256_before':m.LANGUAGE_MASTER_SHA256,'frozen_language_sha256_after':m.LANGUAGE_MASTER_SHA256,
        'model_runtime_buffers_and_rope_cache_restored':True,'engineering_optimizer_discarded':True,
        'frozen_parameters_unchanged':True,'prepared_inputs_and_teacher_replies_unchanged':True,
        'memory_cap_gib':65,'memory_cap_includes_actor_and_reference':True,'memory_cap_passed':True,
        'trainable_sha256_before':'initial-visual4','lineage_sha256':'lineage','peak_allocated_gib':10.,'peak_reserved_gib':11.,
        'record':{'object_gradient_l2':1.,'replay_gradient_l2':0.,'initial_replay_to_object_gradient_norm_ratio':0.,
                  'initial_max_abs_per_question_kl':0.,'initial_mean_replay_kl':0.,'post_update_mean_replay_kl':.01,
                  'post_update_replay_gradient_l2':.2,'post_update_max_abs_per_question_kl':.02,
                  'gradient_l2_before_clip':1.,'actual_object_questions':16,'actual_replay_questions':32,
                  'trainable_sha256_after_temporary_update':'temp','object_factor_gradients':gradients,
                  'initial_replay_factor_gradients':gradients,'post_update_replay_factor_gradients':gradients}}
    result={'model_metadata_initial':initial,'model_metadata_after':after,'initial_adapter_tensor_sha256':'initial-visual4',
        'frozen_language_parameters_unchanged':True,'frozen_language_gradients_absent':True,
        'trainable_parameters':1843200,'trainable_names':{n:torch.Size(s).numel() for n,s in zip(m.VISUAL_NAMES,[(96,5120),(5120,96),(120,5120),(2048,120)])},
        'dtypes':dtype,'final_lora_tensor_sha256':'final-visual4','teacher_reference':ref,'teacher_reference_after':refafter,
        'frozen_parameter_sha256_before':'allfrozen','frozen_parameter_sha256_after':'allfrozen','frozen_parameters_unchanged':True,
        'reference_parameters_unchanged':True,'completed_updates':181,'actual_object_exposures':2888,'actual_replay_exposures':5792,
        'actual_questions_seen':8680,'formal_reference_forward_questions':5792,'formal_reference_forward_started_questions':5792,
        'reference_rng_unchanged_microbatches':1448,'replay_hard_ce_backward':False,'teacher_probability_cache_used':False,
        'preflight':pref,'lineage_sha256':'lineage','last_factor_gradients':gradients,'peak_gpu_allocated_gib':10.,'peak_gpu_reserved_gib':11.}
    lock={'model_metadata_initial':initial,'teacher_reference':ref}
    reviewed=m.verify_training_model_metadata(result,lock,config)
    assert set(reviewed)=={'identity','telemetry'} and reviewed['telemetry']=={}
    assert reviewed['identity']['evaluation_identity']==e
    for mutate in (lambda x:x.update(initial_adapter_tensor_sha256=m.LANGUAGE_MASTER_SHA256),
                   lambda x:x['model_metadata_after']['adapter_signatures'].update(language_fp32_tensor_sha256='changed'),
                   lambda x:x['teacher_reference_after'].update(base_parameter_sha256='changed'),
                   lambda x:x.update(formal_reference_forward_questions=5791),
                   lambda x:x.update(peak_gpu_reserved_gib=66.)):
        bad=copy.deepcopy(result);mutate(bad)
        with pytest.raises(ValueError):m.verify_training_model_metadata(bad,lock,config)
