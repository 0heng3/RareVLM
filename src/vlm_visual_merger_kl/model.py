"""One active composite adapter: frozen SFT language plus trainable merger.

No model is constructed at import time.  Hashes retain runtime parameter order;
only the two declared merger base-layer names are canonicalized for base proof.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

from vlm_deep.common import content_sha, sha_file
from vlm_object_train.train import tensor_hash

LANGUAGE_MASTER_SHA256 = '6a44359406c53261b254c3a19393c63187b34af09baec539d6b235ca148c0fa7'
REFERENCE_BASE_SHA256 = '47486b9a45c127a55223b496f9847e45b2a181dd225697d7d98b274a67ade929'
TARGETS = {'visual.merger.mlp.0': (96, 192, 5120, 5120),
           'visual.merger.mlp.2': (120, 240, 5120, 2048)}
VISUAL_NAMES = tuple('base_model.model.'+target+'.lora_'+factor+'.default.weight'
                     for target in TARGETS for factor in ('A','B'))


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def safe_name(name):
    return name.replace('.lora_A.default.', '.lora_A.').replace('.lora_B.default.', '.lora_B.')


def validate_composite_adapter_config(config,model_root=None):
    """Validate saved PEFT metadata before any real model/adapter construction."""
    targets={f'model.layers.{i}.self_attn.{q}' for i in range(36) for q in ('q_proj','v_proj')} | set(TARGETS)
    # PEFT0.15.2 _find_minimal_target_modules serializes the exact same
    # real-model targets as these four suffixes. Runtime inventory is rechecked.
    observed=config.get('target_modules')
    _require(isinstance(observed,(list,tuple,set)) and len(observed)==len(set(observed))
             and set(observed) in (targets,{'mlp.0','mlp.2','q_proj','v_proj'}),
             'saved composite exact language72 plus merger2 targets differ')
    for key,value in {'r':8,'lora_alpha':16,'lora_dropout':0.,'bias':'none','task_type':'CAUSAL_LM',
                      'peft_type':'LORA','fan_in_fan_out':False,'use_rslora':False,'use_dora':False}.items():
        _require(config.get(key)==value and isinstance(config.get(key),bool)==isinstance(value,bool),
                 'saved composite PEFT contract differs: '+key)
    _require(config.get('rank_pattern')=={t:v[0] for t,v in TARGETS.items()}
             and config.get('alpha_pattern')=={t:v[1] for t,v in TARGETS.items()},
             'saved merger rank/alpha patterns differ')
    for key in ('modules_to_save','layers_to_transform','layers_pattern','layer_replication','exclude_modules'):
        _require(config.get(key) is None,'unsupported extra composite adapter mechanism: '+key)
    _require(config.get('lora_bias',False) is False and config.get('loftq_config',{})=={},
             'saved composite changes bias/quantization mechanism')
    _require(config.get('init_lora_weights') is True and config.get('megatron_config') is None
             and config.get('trainable_token_indices') is None and config.get('eva_config') is None
             and config.get('corda_config') is None,'unsupported composite initialization/additional mechanism')
    if model_root is not None:
        _require(Path(config['base_model_name_or_path']).resolve()==Path(model_root).resolve(),
                 'saved composite base model path differs')
    return {'target_modules':sorted(targets),'rank_pattern':dict(config['rank_pattern']),
            'alpha_pattern':dict(config['alpha_pattern']),'global_rank':8,'global_alpha':16}


def lora_partition(model, require_training=None):
    """Strict placement/shape/active-adapter inventory; works on meta models."""
    import torch
    all_lora = {n:p for n,p in model.named_parameters() if 'lora_' in n}
    visual = {n:p for n,p in all_lora.items() if '.visual.' in n}
    language = {n:p for n,p in all_lora.items() if '.visual.' not in n}
    _require(set(visual) == set(VISUAL_NAMES), 'exact four visual merger factor names required')
    for target,(rank,alpha,inp,out) in TARGETS.items():
        for factor,shape in (('A',(rank,inp)),('B',(out,rank))):
            name = 'base_model.model.'+target+'.lora_'+factor+'.default.weight'
            _require(tuple(visual[name].shape) == shape, 'visual LoRA shape differs: '+name)
    _require(len(language)==144 and sum(p.numel() for p in language.values())==1843200,
             'frozen language must contain exactly144/1843200 q/v factors')
    for name,p in language.items():
        _require('.model.layers.' in name and any('.'+q+'.' in name for q in ('q_proj','v_proj'))
                 and ('.lora_A.default.weight' in name or '.lora_B.default.weight' in name),
                 'unexpected language LoRA target: '+name)
        _require((p.shape[0] if '.lora_A.' in name else p.shape[1])==8, 'language rank must remain8')
    _require(len(all_lora)==148 and sum(p.numel() for p in all_lora.values())==3686400,
             'composite adapter inventory differs')
    for name,module in model.named_modules():
        if hasattr(module,'lora_A') and len(module.lora_A):
            _require(set(module.lora_A)=={'default'} and list(module.active_adapters)==['default']
                     and not module.disable_adapters and not module.merged_adapters,
                     'one unmerged always-active default adapter is required')
            suffix = next((t for t in TARGETS if name.endswith(t)),None)
            rank,alpha = TARGETS[suffix][:2] if suffix else (8,16)
            _require(module.r['default']==rank and module.lora_alpha['default']==alpha
                     and module.scaling['default']==2. and getattr(module.lora_dropout['default'],'p',0.)==0.,
                     'rank/alpha/scaling/dropout differs')
    if require_training is not None:
        expected = set(VISUAL_NAMES) if require_training else set()
        _require({n for n,p in model.named_parameters() if p.requires_grad}==expected,
                 'only four visual masters may train, or all parameters must freeze for eval')
    return language,visual,all_lora


def build_composite_adapter(base):
    import torch
    from peft import LoraConfig, get_peft_model
    _require(not hasattr(base,'peft_config'), 'composite must be injected once into the bare base')
    modules = dict(base.named_modules())
    language = [n for n,m in modules.items() if '.layers.' in n and 'visual' not in n
                and n.endswith(('.q_proj','.v_proj'))]
    _require(len(language)==72 and all(isinstance(modules[n],torch.nn.Linear) for n in language),
             'bare model language q/v inventory differs')
    for target,(_,_,inp,out) in TARGETS.items():
        layer=modules.get(target)
        _require(isinstance(layer,torch.nn.Linear) and (layer.in_features,layer.out_features)==(inp,out),
                 'actual merger Linear dimensions differ: '+target)
    model=get_peft_model(base,LoraConfig(r=8,lora_alpha=16,lora_dropout=0.,bias='none',task_type='CAUSAL_LM',
        target_modules=language+list(TARGETS),rank_pattern={n:v[0] for n,v in TARGETS.items()},
        alpha_pattern={n:v[1] for n,v in TARGETS.items()},use_rslora=False,use_dora=False))
    model.requires_grad_(False)
    for name,p in model.named_parameters():
        if 'lora_' in name:
            p.data=p.data.float()
            p.requires_grad_(name in VISUAL_NAMES)
    lora_partition(model,True)
    return model


def copy_language_initial(model, raw_tensors, expected_sha256=LANGUAGE_MASTER_SHA256):
    """Exact key/shape copy, reproducing original BF16 load then FP32 master."""
    import torch
    language,_,_=lora_partition(model)
    expected={safe_name(n):n for n in language}
    _require(len(expected)==144 and set(raw_tensors)==set(expected), 'raw SFT language keyset differs')
    with torch.no_grad():
        for rawname,name in expected.items():
            value=raw_tensors[rawname]
            _require(value.is_floating_point() and tuple(value.shape)==tuple(language[name].shape)
                     and bool(torch.isfinite(value).all()), 'raw SFT factor shape/finite differs')
            language[name].copy_(value.to(dtype=torch.bfloat16).float())
    actual=tensor_hash(language.items())
    if expected_sha256 is not None:
        _require(actual==expected_sha256, 'language BF16-round FP32 canonical identity differs')
    return actual


def copy_visual_initial(model,tensors,identity):
    import torch
    from .initialization import validate_visual_tensors
    validate_visual_tensors(tensors)
    _,visual,_=lora_partition(model)
    _require(identity['tensor_order']==list(VISUAL_NAMES), 'common initializer runtime order differs')
    _require(tensor_hash((n,tensors[n]) for n in VISUAL_NAMES)==identity['visual_fp32_tensor_sha256'],
             'common initializer tensor digest differs')
    with torch.no_grad():
        for name in VISUAL_NAMES:
            visual[name].copy_(tensors[name])
    _require(tensor_hash(visual.items())==identity['visual_fp32_tensor_sha256'],
             'loaded visual masters differ from common initialization')


def canonical_base_name(name):
    for target in TARGETS:
        prefix='base_model.model.'+target+'.base_layer.'
        if name.startswith(prefix):
            return 'base_model.model.'+target+'.'+name[len(prefix):]
    return name


def canonical_base_parameters(model):
    result={}
    for name,p in model.named_parameters():
        if 'lora_' in name:
            continue
        canonical=canonical_base_name(name)
        _require(canonical not in result, 'base-name mapping collision')
        result[canonical]=p
    return result


def base_identity(model):
    named=[(n,p) for n,p in model.named_parameters() if 'lora_' not in n]
    mapped=canonical_base_parameters(model)
    return {'raw_named_base_parameter_sha256':tensor_hash(named),
        'canonical_base_parameter_sha256':tensor_hash(sorted(mapped.items())),
        'tensor_count':len(mapped),'elements':sum(p.numel() for p in mapped.values()),
        'remapped_names':{n:canonical_base_name(n) for n,p in named if n!=canonical_base_name(n)}}


def prove_base_mapping(actor,reference):
    """Values/names/dtypes bind to the independently loaded accepted SFT base."""
    a,r=base_identity(actor),base_identity(reference)
    _require(r['raw_named_base_parameter_sha256']==REFERENCE_BASE_SHA256,
             'reference base differs from accepted parent4748 identity')
    _require(a['canonical_base_parameter_sha256']==r['canonical_base_parameter_sha256']
             and a['tensor_count']==r['tensor_count'] and a['elements']==r['elements'],
             'actor frozen base values differ after exact declared name mapping')
    expected={ 'base_model.model.'+t+'.base_layer.'+suffix:'base_model.model.'+t+'.'+suffix
               for t in TARGETS for suffix in ('weight','bias') }
    _require(a['remapped_names']==expected and not r['remapped_names'], 'unexpected base wrapper remapping')
    return {**a,'reference_original_base_parameter_sha256':r['raw_named_base_parameter_sha256'],
            'reference_canonical_base_parameter_sha256':r['canonical_base_parameter_sha256'],
            'exact_declared_bijection_and_values':True}


def architecture():
    return {'active_adapter':'default','always_active_composite':True,'language_frozen_tensors':144,
        'language_frozen_elements':1843200,'new_trainable_tensors':4,'new_trainable_elements':1843200,
        'total_adapter_tensors':148,'total_adapter_elements':3686400,
        'targets':{t:{'r':v[0],'lora_alpha':v[1],'A_shape':[v[0],v[2]],'B_shape':[v[3],v[0]]}
                   for t,v in TARGETS.items()},'scaling':2.,'new_initial_B_zero':True,
        'base_name_mapping_scope':'only visual.merger.mlp.0/2.base_layer to original paths'}


def adapter_signatures(model):
    import torch
    language,visual,all_lora=lora_partition(model)
    _require(all(p.dtype==torch.float32 and bool(torch.isfinite(p).all()) for p in all_lora.values()),
             'training full148 adapter must contain finite FP32 values')
    return {'full_adapter_fp32_tensor_sha256':tensor_hash(all_lora.items()),
        'full_adapter_native_bf16_tensor_sha256':tensor_hash((n,p.to(torch.bfloat16)) for n,p in all_lora.items()),
        'full_adapter_bf16_promoted_fp32_tensor_sha256':tensor_hash((n,p.to(torch.bfloat16).float()) for n,p in all_lora.items()),
        'language_fp32_tensor_sha256':tensor_hash(language.items()),
        'language_native_bf16_tensor_sha256':tensor_hash((n,p.to(torch.bfloat16)) for n,p in language.items()),
        'visual_fp32_tensor_sha256':tensor_hash(visual.items())}


def evaluation_identity(model,config,expected_training_signatures=None,base=None):
    import torch
    from vlm_precision.evaluate import parameter_dtypes
    language,visual,all_lora=lora_partition(model)
    # This function also creates the anticipated native eval signature from FP32.
    native=tensor_hash((n,p.to(torch.bfloat16)) for n,p in all_lora.items())
    promoted=tensor_hash((n,p.to(torch.bfloat16).float()) for n,p in all_lora.items())
    language_promoted=tensor_hash((n,p.to(torch.bfloat16).float()) for n,p in language.items())
    _require(language_promoted==LANGUAGE_MASTER_SHA256,'evaluation would change the fixed language SFT')
    dtypes=parameter_dtypes(model)
    if expected_training_signatures is not None:
        _require(native==expected_training_signatures['full_adapter_native_bf16_tensor_sha256']
                 and promoted==expected_training_signatures['full_adapter_bf16_promoted_fp32_tensor_sha256'],
                 'native evaluation adapter rounding differs')
    # Anticipated dtypes are constructed from actual dimensions, not old totals.
    expected_dtypes=copy.deepcopy(dtypes)
    expected_dtypes.update(parameter_tensors_by_dtype={'bfloat16':sum(dtypes['parameter_tensors_by_dtype'].values())},
        parameter_stored_elements_by_dtype={'bfloat16':sum(dtypes['parameter_stored_elements_by_dtype'].values())},
        lora_tensors_by_dtype={'bfloat16':148},lora_elements_by_dtype={'bfloat16':3686400},trainable_parameters=0)
    _require(expected_dtypes['parameter_tensors_by_dtype']=={'bfloat16':972}
             and expected_dtypes['parameter_stored_elements_by_dtype']=={'bfloat16':3758309376}
             and expected_dtypes['buffer_tensors_by_dtype']=={'float32':38}
             and expected_dtypes['params4bit_tensors']==0,'actual native composite inventory differs from declared increment')
    b=base or base_identity(model)
    return {'architecture':architecture(),'actual_parameter_dtypes':expected_dtypes,
        'full_adapter_native_bf16_tensor_sha256':native,'full_adapter_bf16_promoted_fp32_tensor_sha256':promoted,
        'frozen_language_canonical_fp32_tensor_sha256':language_promoted,
        'canonical_base_parameter_sha256':b['canonical_base_parameter_sha256'],
        'raw_named_base_parameter_sha256':b['raw_named_base_parameter_sha256'],
        'base_remapped_names':b['remapped_names'],'all_parameters_frozen':True,
        'numerical_path':'original_bf16','autocast':False,'new_adapter_tensors_vs_language_control':4,
        'new_adapter_elements_vs_language_control':1843200}


def load_training_actor(config,reference,device):
    import torch
    from safetensors.torch import load_file
    from transformers import Qwen2_5_VLForConditionalGeneration
    from vlm_precision.evaluate import parameter_dtypes
    from .initialization import load_common_initialization, initialization_paths
    base=Qwen2_5_VLForConditionalGeneration.from_pretrained(str(config['model_root']),local_files_only=True,
        torch_dtype=torch.bfloat16,quantization_config=None,attn_implementation='sdpa',low_cpu_mem_usage=True,
        device_map={'':str(device)})
    model=build_composite_adapter(base)
    copy_language_initial(model,load_file(str(Path(config['initial_adapter'])/'adapter_model.safetensors'),device='cpu'))
    tensors,manifest=load_common_initialization(config,require_prepared=True)
    copy_visual_initial(model,tensors,manifest['identity'])
    for module in model.modules():
        if isinstance(module,torch.nn.Dropout): module.p=0.
    model.config.use_cache=False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.enable_input_require_grads()
    model.train()
    language,visual,all_lora=lora_partition(model,True)
    for n,p in model.named_parameters():
        _require(p.dtype==(torch.float32 if 'lora_' in n else torch.bfloat16), 'actor numerical storage dtype differs: '+n)
    path=initialization_paths(config)['manifest']
    identity={'architecture':architecture(),'common_initialization_identity':manifest['identity'],
        'common_initialization_manifest_file':str(path),'common_initialization_manifest_file_sha256':sha_file(path),
        'common_initialization_identity_sha256':manifest['identity_sha256'],
        'adapter_signatures':adapter_signatures(model),'language_gradients_absent':all(p.grad is None for p in language.values()),
        'actual_parameter_dtypes':parameter_dtypes(model)}
    return model,list(visual.values()),parameter_dtypes(model),identity


def load_evaluation_actor(model_root,reference,device,adapter_dir,config):
    import torch
    from peft import PeftModel
    from transformers import Qwen2_5_VLForConditionalGeneration
    from vlm_precision.evaluate import parameter_dtypes
    validate_composite_adapter_config(json.loads((Path(adapter_dir)/'adapter_config.json').read_text()),model_root)
    base=Qwen2_5_VLForConditionalGeneration.from_pretrained(str(model_root),local_files_only=True,
        torch_dtype=torch.bfloat16,quantization_config=None,attn_implementation='sdpa',low_cpu_mem_usage=True,
        device_map={'':str(device)})
    model=PeftModel.from_pretrained(base,str(adapter_dir),is_trainable=False,autocast_adapter_dtype=False)
    for name,p in model.named_parameters():
        if 'lora_' in name: p.data=p.data.to(torch.bfloat16)
    model.requires_grad_(False);model.eval();model.config.use_cache=True
    model.generation_config.temperature=1.;model.generation_config.top_p=1.;model.generation_config.top_k=50
    lora_partition(model,False)
    _require(all(p.dtype==torch.bfloat16 for p in model.parameters()),'native evaluation contains non-BF16 weight')
    identity=evaluation_identity(model,config)
    _require(parameter_dtypes(model)==identity['actual_parameter_dtypes'],'actual eval dtype inventory differs')
    return model,parameter_dtypes(model),identity


def verify_training_model_metadata(result,train_input_lock,config):
    """Pure metadata validation. File/tensor-byte checks belong to preparation/final audit."""
    initial=train_input_lock['model_metadata_initial'];after=result['model_metadata_after']
    _require(result['model_metadata_initial']==initial,'model initial metadata differs from immutable train lock')
    _require(initial['architecture']==after['architecture']==architecture()==config['visual_adaptation'] | {
        'scaling':2.,'new_initial_B_zero':True,'base_name_mapping_scope':'only visual.merger.mlp.0/2.base_layer to original paths'},
        'composite architecture metadata differs')
    _require(initial['adapter_signatures']['language_fp32_tensor_sha256']==LANGUAGE_MASTER_SHA256
             ==after['adapter_signatures']['language_fp32_tensor_sha256'], 'frozen language identity changed')
    _require(initial['adapter_signatures']['visual_fp32_tensor_sha256']==initial['common_initialization_identity']['visual_fp32_tensor_sha256'],
             'initial visual master differs from common initializer')
    _require(result['initial_adapter_tensor_sha256']==initial['adapter_signatures']['visual_fp32_tensor_sha256'],
             'train lock initial master must describe the four visual factors')
    _require(initial['common_initialization_identity_sha256']==content_sha(initial['common_initialization_identity']),
             'common initializer identity seal differs')
    _require(after['adapter_signatures']['visual_fp32_tensor_sha256']!=initial['adapter_signatures']['visual_fp32_tensor_sha256'],
             'Final visual masters did not change')
    _require(after['frozen_base_mapping']==initial['frozen_base_mapping']
             and initial['frozen_base_mapping']['exact_declared_bijection_and_values'] is True
             and initial['frozen_base_mapping']['reference_original_base_parameter_sha256']==REFERENCE_BASE_SHA256,
             'frozen base mapping/value identity changed')
    _require(result['frozen_language_parameters_unchanged'] is True and result['frozen_language_gradients_absent'] is True
             and initial['language_gradients_absent'] is True and after['language_gradients_absent'] is True,
             'frozen language acquired gradients/updates')
    _require(result['trainable_parameters']==1843200 and len(result['trainable_names'])==4
             and set(result['trainable_names'])==set(VISUAL_NAMES) and sum(result['trainable_names'].values())==1843200,
             'optimizer trainable inventory differs')
    _require(after['actual_parameter_dtypes']==initial['actual_parameter_dtypes']==result['dtypes'],
             'training dtype strategy changed')
    dtype=initial['actual_parameter_dtypes']
    _require(dtype['parameter_tensors_by_dtype']=={'bfloat16':824,'float32':148}
             and dtype['parameter_stored_elements_by_dtype']=={'bfloat16':3754622976,'float32':3686400}
             and dtype['lora_tensors_by_dtype']=={'float32':148}
             and dtype['lora_elements_by_dtype']=={'float32':3686400}
             and dtype['trainable_parameters']==1843200 and dtype['params4bit_tensors']==0
             and dtype['buffer_tensors_by_dtype']=={'float32':38},'actual training dtype inventory differs')
    _require(result['final_lora_tensor_sha256']==after['adapter_signatures']['visual_fp32_tensor_sha256'],
             'Final master digest is not four visual masters')
    for key in ('common_initialization_identity','common_initialization_identity_sha256',
                'common_initialization_manifest_file','common_initialization_manifest_file_sha256'):
        _require(after[key]==initial[key],'common initialization provenance changed')
    e=after['evaluation_identity']
    _require(e['architecture']==architecture() and e['frozen_language_canonical_fp32_tensor_sha256']==LANGUAGE_MASTER_SHA256
             and e['full_adapter_native_bf16_tensor_sha256']==after['adapter_signatures']['full_adapter_native_bf16_tensor_sha256']
             and e['full_adapter_bf16_promoted_fp32_tensor_sha256']==after['adapter_signatures']['full_adapter_bf16_promoted_fp32_tensor_sha256']
             and e['canonical_base_parameter_sha256']==initial['frozen_base_mapping']['canonical_base_parameter_sha256'],
             'anticipated native evaluation identity differs')
    ref=result['teacher_reference'];ref_after=result['teacher_reference_after']
    _require(ref==train_input_lock['teacher_reference']
             and ref['initial_lora_tensor_sha256']==LANGUAGE_MASTER_SHA256
             and ref['initial_base_parameter_sha256']==REFERENCE_BASE_SHA256
             and ref['initial_all_parameter_sha256']=='ddcbe50ae8036d83cfff644bda9a23b2aa647652013f83bd75209274061677a8'
             and ref['all_parameters_frozen'] is True and ref['parameter_storage_shared_with_actor'] is False
             and ref['actor_rng_restored_after_construction'] is True and ref['execution_path_identity_claim'] is False,
             'fixed independent SFT-only reference identity differs')
    for a,b in (('lora_tensor_sha256','initial_lora_tensor_sha256'),
                ('base_parameter_sha256','initial_base_parameter_sha256'),
                ('all_parameter_sha256','initial_all_parameter_sha256')):
        _require(ref_after[a]==ref[b],'Final fixed reference signature changed')
    _require(ref_after['all_parameters_frozen'] is True and ref_after['gradients_absent'] is True
             and ref_after['training'] is False and ref_after['gradient_checkpointing'] is False,
             'reference acquired gradients/train/checkpoint mode')
    _require(result['frozen_parameter_sha256_before']==result['frozen_parameter_sha256_after']
             and result['frozen_parameters_unchanged'] is True and result['reference_parameters_unchanged'] is True
             and result['completed_updates']==181 and result['actual_object_exposures']==2888
             and result['actual_replay_exposures']==5792 and result['actual_questions_seen']==8680
             and result['formal_reference_forward_questions']==5792
             and result['formal_reference_forward_started_questions']==5792
             and result['reference_rng_unchanged_microbatches']==1448
             and result['replay_hard_ce_backward'] is False and result['teacher_probability_cache_used'] is False,
             'training/frozen/reference full stream metadata differs')
    from .train import verify_preflight_record
    verify_preflight_record(result['preflight'],result['initial_adapter_tensor_sha256'],result['lineage_sha256'])
    for gradients in (result['last_factor_gradients'],result['preflight']['record']['object_factor_gradients'],
                      result['preflight']['record']['initial_replay_factor_gradients'],
                      result['preflight']['record']['post_update_replay_factor_gradients']):
        _require(set(gradients)==set(VISUAL_NAMES),'four factor gradient inventory differs')
        for value in gradients.values():
            _require(value['finite'] is True and math.isfinite(value['gradient_l2']) and value['gradient_l2']>=0
                     and type(value['nonzero_elements']) is int and value['nonzero_elements']>=0,
                     'factor gradient diagnostics must be finite; initial A zero is allowed')
    _require(all(math.isfinite(result[k]) and 0<=result[k]<=65
                 for k in ('peak_gpu_allocated_gib','peak_gpu_reserved_gib')),
             'combined actor/reference training peak exceeds65GiB')
    return {'identity':{'architecture':architecture(),'initial':initial,'final':after,'evaluation_identity':e},'telemetry':{}}
