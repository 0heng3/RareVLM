"""Final-only object CE plus online fixed-initial generated-prefix full KL."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import time

for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '2'
os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')
import torch

from rarevlm_paths import SOURCE_ROOT
from vlm_deep.common import content_sha, sha_file, to_device, write_json
from vlm_object_train import train as legacy
from vlm_object_train.data import OBJECT_SUFFIX, PROMPT_SUFFIX, private_path
from vlm_object_train.schedule import coverage
from vlm_object_train.train import (autocast, check_gradients, cpu_state, file_hashes, frozen_hash,
    learning_rate, load_actor, lock_json, now, package_lock, restore_rng, restore_runtime,
    resume_checkpoint, rng_hash, rng_state, runtime_state, save_checkpoint, sync, tensor_hash, trainable_hash)
from . import SCHEMA, CANDIDATE, CONDITIONS
from .inputs import (encode_group, legacy_contracts, paths as input_paths,
                     verify_prepared_training_inputs, verify_prepared_files_unchanged)
from .loss import (accumulate_roles, empty_role_statistics, normalized_qa_loss,
                   sequence_ce, summarize_roles, assistant_logprobs, per_question_kl, per_question_gt_ce,
                   replay_diagnostics, empty_kl_role_statistics, accumulate_kl_roles, summarize_kl_roles)


# Explicit immutable numerical provenance; parent files are never patched.
PARENT_SOURCE_SHA256 = {'train.py': 'f9914c0a135adaaab5a949787c4621d9a4a048de64fc9f6f3e35d42178f059f8', 'loss.py': '1c8aaabb4ab2af4000b5fbdf61c976dd43f99fe5e4c4345ff87829ba08a88aec'}
INITIAL_MASTER_SHA256 = '6a44359406c53261b254c3a19393c63187b34af09baec539d6b235ca148c0fa7'

def validate_config(config):
    fixed = {'schema_version':SCHEMA, 'object_prompt_suffix':OBJECT_SUFFIX,
             'replay_prompt_suffix':PROMPT_SUFFIX, 'object_effective_batch':16,
             'replay_per_update':32, 'replay_lambda':1., 'training_microbatch':4,
             'training_accumulation':4, 'expected_trainable':1843200,
             'checkpoint_every_updates':25, 'epochs':1, 'lr':5e-6, 'preflight_lr':1e-4,
             'weight_decay':0., 'warmup_ratio':.1, 'gradient_clip_norm':1.,
             'gradient_checkpointing':True, 'visual_token_budget':1024,
             'max_sequence_length':2048, 'preflight_memory_cap_gib':65}
    for key,value in fixed.items():
        if config.get(key) != value or isinstance(config.get(key),bool) != isinstance(value,bool):
            raise ValueError(f'behavior replay fixed contract differs: {key}')
    if (config.get('candidates') != [CANDIDATE] or config.get('control_can_be_selected') is not False
            or config.get('replay_objective') != 'forward_kl_plus_GT_CE'):
        raise ValueError('only gt_ce_kl32 full forward KL can be selected')
    ref_fixed = {'kind':'online_independent_fixed_initial','base_precision':'bf16','vision_precision':'bf16',
        'lora_master_precision':'fp32','autocast':'bf16','mode':'eval','gradient_checkpointing':False,
        'grad_enabled':False,'all_parameters_frozen':True,'share_parameter_storage_with_actor':False,
        'restore_actor_RNG_around_construction':True,'initial_runtime_master_sha256':INITIAL_MASTER_SHA256,
        'temperature':1.,'direction':'teacher_to_actor_forward_KL','full_vocabulary':True,
        'vocabulary_reduction':'sum','within_question_reduction':'assistant_token_mean',
        'source_reduction':'question_mean','repetition_penalty_or_generation_logits_processor':False,
        'probability_bank':False,'formal_reference_forward_QA_per_seed':5792,
        'initial_max_abs_per_question_KL':1e-5,'initial_replay_to_object_gradient_norm_max_ratio':.01}
    for key,value in ref_fixed.items():
        if (config.get('teacher_reference',{}).get(key) != value
                or isinstance(config.get('teacher_reference',{}).get(key),bool) != isinstance(value,bool)):
            raise ValueError(f'fixed online reference contract differs: {key}')
    if config.get('replay_gt_ce_weights') != {'gt_kl32':0.,'gt_ce_kl32':1.}:
        raise ValueError('single fixed candidate gamma1/control gamma0 required')
    evaluation = config.get('evaluation',{})
    for key,value in {'numerical_path':'original_bf16','precision':'bf16','vision_precision':'bf16',
                      'quantization':'original','visual_token_budget':1024,'batch_size':4,
                      'max_new_tokens':32,'do_sample':False,'attention_implementation':'sdpa',
                      'tf32_matmul':False,'tf32_cudnn':False,'input_truncation':False}.items():
        if evaluation.get(key) != value:
            raise ValueError(f'behavior numerical/evaluation contract differs: {key}')
    diagnostics = config.get('position_diagnostics',{})
    if (diagnostics.get('enabled') is not True or diagnostics.get('mutually_exclusive') is not True
            or diagnostics.get('changes_loss_weights') is not False or diagnostics.get('extra_model_calls') != 0):
        raise ValueError('position diagnostics must be exclusive and weight-neutral without extra calls')
    for key in ('train_manifest','replay_manifest','reference_config','model_root','initial_adapter',
                'data_admission','dataset_protocol','teacher_reply_root','output_root'):
        if not config.get(key):
            raise ValueError(f'missing required path: {key}')
        private_path(config[key])
    return config


def validate_condition(condition, seed):
    if condition not in CONDITIONS or type(seed) is not int or seed not in (0,42,200) or (condition=='gt_kl32' and seed!=42):
        raise ValueError('only gt_ce_kl32 with seed0/42/200 is allowed')


def source_hashes():
    result = legacy.source_hashes()
    for path in sorted(Path(__file__).parent.glob('*.py')):
        result[str(path.relative_to(SOURCE_ROOT))] = sha_file(path)
    for name,digest in PARENT_SOURCE_SHA256.items():
        path = SOURCE_ROOT/'vlm_object_behavior_replay'/name
        if sha_file(path) != digest:
            raise ValueError('immutable parent numerical source changed')
        result[str(path.relative_to(SOURCE_ROOT))] = digest
    return result


def parent_parameter_anchor(config):
    review_file = private_path(config['parent_output_root'])/'independent_review/TRAINING_COMPLETION_REVIEW.json'
    digest = config['parent_proof_files_sha256'].get(str(review_file))
    if not digest or sha_file(review_file) != digest:
        raise ValueError('accepted parent parameter review file SHA differs')
    review = json.loads(review_file.read_text())
    arms = [row for row in review.get('conditions',[]) if row.get('condition') == 'gt_ce32']
    if (review.get('status') != 'passed_training_engineering_only' or len(arms) != 1
            or arms[0].get('formal_updates') != 181
            or arms[0].get('initial_canonical_master_sha256') != INITIAL_MASTER_SHA256
            or review.get('shared_initial',{}).get('canonical_master_sha256') != INITIAL_MASTER_SHA256):
        raise ValueError('accepted parent initial-parameter review schema/identity differs')
    base_sha = arms[0].get('saved_frozen_base_vision_before_after_sha256')
    if base_sha != '47486b9a45c127a55223b496f9847e45b2a181dd225697d7d98b274a67ade929':
        raise ValueError('accepted parent frozen base/vision identity differs')
    return {'review_file':str(review_file),'review_file_sha256':digest,'condition':'gt_ce32',
        'initial_canonical_master_sha256':INITIAL_MASTER_SHA256,'frozen_base_vision_sha256':base_sha}


def initial_stats():
    result = {'training_seconds':0., 'last_gradient_l2':None, 'replay_rng_unchanged_checks':0,
              'replay_kl_sum':0.,'replay_max_abs_per_question_kl':0.,
              'replay_kl_role_statistics':empty_kl_role_statistics(),
              'reference_forward_started_questions':0,'reference_forward_questions':0,
              'reference_forward_seconds':0.,'reference_rng_unchanged_microbatches':0}
    for source in ('object','replay'):
        result.update({f'{source}_exposures':0,f'{source}_ce_sum':0.,f'{source}_input_contracts':[],
                       f'{source}_active_seconds':0.,f'{source}_role_statistics':empty_role_statistics(),
                       f'{source}_forward_started_questions':0})
        for name in ('input_tokens','visual_tokens','answer_tokens','microbatches','padding_tokens'):
            result[f'{source}_{name}'] = 0
    return result


def validate_exposure_cursor(stats, blocks, completed, context):
    if type(completed) is not int or not 0 <= completed <= len(blocks):
        raise ValueError('formal update cursor invalid')
    for source in ('object','replay'):
        count = sum(len(block[f'{source}_indices']) for block in blocks[:completed])
        if (stats[f'{source}_exposures'] != count
                or stats[f'{source}_forward_started_questions'] != count
                or stats[f'{source}_input_contracts'] != context['contracts'][source][:count]):
            raise ValueError('formal exposure/actual-input cursor differs from frozen complete stream')
        roles = stats[f'{source}_role_statistics']
        if sum(v['token_count'] for v in roles.values()) != stats[f'{source}_answer_tokens']:
            raise ValueError('formal role-token cursor differs from assistant token count')
    if stats['replay_rng_unchanged_checks'] != completed:
        raise ValueError('replay RNG-check cursor differs from completed updates')
    replay_count = sum(len(b['replay_indices']) for b in blocks[:completed])
    replay_microbatches = sum(math.ceil(len(b['replay_indices'])/4) for b in blocks[:completed])
    if (stats['reference_forward_questions'] != replay_count
            or stats['reference_forward_started_questions'] != replay_count
            or stats['reference_rng_unchanged_microbatches'] != replay_microbatches
            or sum(v['token_count'] for v in stats['replay_kl_role_statistics'].values()) != stats['replay_answer_tokens']):
        raise ValueError('reference/KL-role cursor differs from completed replay stream')


def enforce_memory_cap(device, config, phase):
    if device.type == 'cuda':
        allocated, reserved = torch.cuda.max_memory_allocated(device)/2**30, torch.cuda.max_memory_reserved(device)/2**30
        if max(allocated,reserved) > config['preflight_memory_cap_gib']:
            raise RuntimeError(f'{phase}_memory_cap_exceeded: allocated={allocated:.6f}GiB reserved={reserved:.6f}GiB cap=65GiB; no batch fallback')


def lora_hash(model):
    return tensor_hash((name,p) for name,p in model.named_parameters() if 'lora_' in name)


def reference_base_hash(model):
    # reference is all-frozen, so legacy.frozen_hash(reference) would include LoRA.
    return tensor_hash((name,p) for name,p in model.named_parameters() if 'lora_' not in name)


def reference_state(model):
    return {'lora_tensor_sha256':lora_hash(model),'base_parameter_sha256':reference_base_hash(model),
            'all_parameter_sha256':tensor_hash(model.named_parameters()),
            'all_parameters_frozen':all(not p.requires_grad for p in model.parameters()),
            'gradients_absent':all(p.grad is None for p in model.parameters()),
            'training':model.training,'gradient_checkpointing':bool(getattr(model,'is_gradient_checkpointing',False))}


def storage_is_shared(actor, reference):
    pointers = {(p.device.type,p.device.index,p.untyped_storage().data_ptr()) for p in actor.parameters() if p.numel()}
    return any((p.device.type,p.device.index,p.untyped_storage().data_ptr()) in pointers
               for p in reference.parameters() if p.numel())


def load_reference(config, model_config, device, actor, initial_sha, actor_frozen_sha):
    """Independent fixed q0. Its eval/no-grad path is admitted by real parity."""
    saved_rng = rng_state()
    try:
        reference,_,dtypes = load_actor(config,model_config,device)
        if trainable_hash(reference) != initial_sha or initial_sha != INITIAL_MASTER_SHA256:
            raise ValueError('independent reference initial FP32 LoRA identity differs')
        if storage_is_shared(actor,reference):
            raise ValueError('reference shares parameter storage with actor')
        reference.requires_grad_(False)
        reference.gradient_checkpointing_disable()
        reference.disable_input_require_grads()
        reference.eval()
        reference.config.use_cache = False
        state = reference_state(reference)
        if (state['base_parameter_sha256'] != actor_frozen_sha or not state['all_parameters_frozen']
                or not state['gradients_absent'] or state['training'] or state['gradient_checkpointing']):
            raise ValueError('independent reference base/frozen/eval/checkpoint contract differs')
    finally:
        restore_rng(saved_rng)
    if rng_hash(rng_state()) != rng_hash(saved_rng):
        raise ValueError('reference construction advanced actor RNG')
    return reference,{**config['teacher_reference'],
        'initial_lora_tensor_sha256':state['lora_tensor_sha256'],
        'initial_base_parameter_sha256':state['base_parameter_sha256'],
        'initial_all_parameter_sha256':state['all_parameter_sha256'],
        'all_parameters_frozen':True,'parameter_storage_shared_with_actor':False,
        'dtypes':dtypes,'parameter_count':sum(p.numel() for p in reference.parameters()),
        'actor_rng_restored_after_construction':True,
        'execution_path_identity_claim':False}


def assert_reference_fixed(reference, initial_state):
    state = reference_state(reference)
    for key,oldkey in (('lora_tensor_sha256','initial_lora_tensor_sha256'),
                       ('base_parameter_sha256','initial_base_parameter_sha256'),
                       ('all_parameter_sha256','initial_all_parameter_sha256')):
        if state[key] != initial_state[oldkey]:
            raise ValueError('fixed online reference parameter hash changed')
    if (not state['all_parameters_frozen'] or not state['gradients_absent']
            or state['training'] or state['gradient_checkpointing']):
        raise ValueError('reference acquired gradients/train mode/checkpointing')
    return state


def backward_update(model, processor, objects, replays, block, condition, config, device, context,
                    stats=None, teaching_records=None, sources=('object','replay'), update=0,
                    reference_model=None, vocabulary_size=None,replay_component="joint"):
    obj,rep = block['object_indices'],block['replay_indices']
    for source,rows,indices in (('object',objects,obj),('replay',replays,rep)):
        if source not in sources:
            continue
        replay_rng = rng_hash(rng_state()) if source == 'replay' else None
        for start in range(0,len(indices),config['training_microbatch']):
            selected = indices[start:start+4]
            batch,contracts,samples = encode_group(processor,rows,selected,config,source=source,
                condition=condition,teacher_replies=context['teacher_replies'])
            offset = sum(len(b[f'{source}_indices']) for b in context['plan']['blocks'][:update])+start
            if contracts != context['contracts'][source][offset:offset+len(selected)]:
                raise ValueError('actual forward input/target IDs/roles differ from frozen training stream')
            new_records = []
            if teaching_records is not None:
                for position,(index,contract,sample) in enumerate(zip(selected,contracts,samples)):
                    row = rows[index]
                    record = {'source':source,**contract,'question':row['question'],
                        'original_gt_for_provenance_only':row['target'],'prompt_suffix':row['prompt_suffix'],
                        'image_path':row['image_path'],'input_ids':batch['input_ids'][position].tolist(),
                        'labels':batch['labels'][position].tolist(),'answer_roles':batch['answer_roles'][position].tolist(),
                        'image_grid_thw':sample['image_grid_thw'].tolist(),
                        'pixel_values_shape':list(sample['pixel_values'].shape)}
                    teaching_records.append(record)
                    new_records.append(record)
            padding = batch['input_ids'].numel()-int(batch['attention_mask'].sum())
            batch = to_device(batch,device)
            if source == 'replay':
                if reference_model is None or reference_model.training or any(p.requires_grad for p in reference_model.parameters()):
                    raise ValueError('replay requires an independent all-frozen eval reference')
                reference_rng = rng_hash(rng_state())
                sync(device)
                begun = time.monotonic()
                if stats is not None:
                    stats['reference_forward_started_questions'] += len(selected)
                with torch.no_grad(),autocast(device):
                    teacher,teacher_counts,teacher_targets,teacher_roles = assistant_logprobs(reference_model,batch,vocabulary_size)
                teacher = teacher.detach()
                sync(device)
                duration = time.monotonic()-begun
                if (teacher.requires_grad or any(p.grad is not None for p in reference_model.parameters())
                        or rng_hash(rng_state()) != reference_rng):
                    raise ValueError('fixed reference acquired gradient or advanced RNG')
                if stats is not None:
                    stats['reference_forward_questions'] += len(selected)
                    stats['reference_forward_seconds'] += duration
                    stats['reference_rng_unchanged_microbatches'] += 1
                enforce_memory_cap(device,config,'reference_forward')
            if stats is not None:
                stats[f'{source}_forward_started_questions'] += len(selected)
            with torch.enable_grad(),autocast(device):
                if source == 'object':
                    ce,counts,role_stats = sequence_ce(model,batch)
                    objective = ce
                else:
                    actor,counts,targets,roles = assistant_logprobs(model,batch,vocabulary_size)
                    if (not torch.equal(counts,teacher_counts) or not torch.equal(targets,teacher_targets)
                            or not torch.equal(roles,teacher_roles)):
                        raise ValueError('actor/reference assistant masks, shifted targets or roles differ')
                    kl_objective,token_kl = per_question_kl(actor,teacher,counts)
                    gt_ce = per_question_gt_ce(actor,targets,counts)
                    gamma = config['replay_gt_ce_weights'][condition]
                    if replay_component=='joint': objective=kl_objective+gamma*gt_ce
                    elif replay_component=='kl_only': objective=kl_objective
                    elif replay_component=='ce_only': objective=gt_ce
                    else: raise ValueError('unknown replay gradient probe component')
                    ce,role_stats,kl_roles = replay_diagnostics(actor,targets,counts,roles,token_kl)
                if objective.numel() != len(selected) or counts.tolist() != [c['assistant_tokens'] for c in contracts]:
                    raise ValueError('actual per-QA assistant count differs from frozen contract')
                loss = normalized_qa_loss(objective,source,len(obj),len(rep),config['replay_lambda'])
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('nonfinite object CE/GT-prefix forward KL')
            loss.backward()
            enforce_memory_cap(device,config,'training')
            if new_records:
                for record,value,hard_ce in zip(new_records,objective.detach().tolist(),ce.detach().tolist()):
                    record['per_question_objective'] = value
                    record['per_question_hard_ce_diagnostic'] = hard_ce
                    record['hard_ce_backpropagated'] = source == 'object' or (replay_component=='ce_only' or (replay_component=='joint' and gamma>0))
            if stats is not None:
                stats[f'{source}_exposures'] += len(selected)
                stats[f'{source}_input_tokens'] += sum(c['input_tokens'] for c in contracts)
                stats[f'{source}_visual_tokens'] += sum(c['visual_tokens'] for c in contracts)
                stats[f'{source}_answer_tokens'] += int(counts.sum())
                stats[f'{source}_ce_sum'] += float(ce.detach().sum())
                stats[f'{source}_input_contracts'].extend(contracts)
                stats[f'{source}_microbatches'] += 1
                stats[f'{source}_padding_tokens'] += padding
                accumulate_roles(stats[f'{source}_role_statistics'],role_stats)
                if source == 'replay':
                    stats['replay_kl_sum'] += float(kl_objective.detach().sum())
                    stats['replay_max_abs_per_question_kl'] = max(stats['replay_max_abs_per_question_kl'],float(kl_objective.detach().abs().max()))
                    accumulate_kl_roles(stats['replay_kl_role_statistics'],kl_roles)
        if replay_rng is not None:
            if rng_hash(rng_state()) != replay_rng:
                raise ValueError('replay/reference advanced RNG; object stochastic path would differ')
            if stats is not None:
                stats['replay_rng_unchanged_checks'] += 1


def runtime_matches(snapshot):
    for module,buffers,attributes in snapshot:
        if set(module._buffers) != set(buffers):
            return False
        for key,(original,value) in buffers.items():
            current = module._buffers[key]
            if current is not original or (value is not None and not torch.equal(current.detach().cpu(),value)):
                return False
        for key,(existed,value,_) in attributes.items():
            if (key in module.__dict__) != existed:
                return False
            current = module.__dict__.get(key)
            if isinstance(value,torch.Tensor):
                if not isinstance(current,torch.Tensor) or not torch.equal(current.detach().cpu(),value):
                    return False
            elif current != value:
                return False
    return True


def gradient_norm(parameters):
    return math.sqrt(sum(float(p.grad.detach().double().square().sum()) for p in parameters if p.grad is not None))


def preflight(model, parameters, processor, objects, replays, block, config, condition, directory,
              device, context, reference_model, reference_binding, vocabulary_size, frozen_sha256=None, lineage_sha256=None):
    """One embedded discarded joint Adam step, plus a post-step restoring-KL probe."""
    initial = {name:cpu_state(p) for name,p in model.named_parameters() if p.requires_grad}
    before = trainable_hash(model)
    frozen_before = frozen_sha256 or frozen_hash(model)
    rng_before,runtime_before = rng_state(),runtime_state(model)
    reference_runtime_before = runtime_state(reference_model)
    optimizer = torch.optim.AdamW(parameters,lr=config['preflight_lr'],weight_decay=config['weight_decay'])
    error,completed,record = None,0,{}
    probe_stats,ce_probe_stats,after_stats = initial_stats(),initial_stats(),initial_stats()
    try:
        optimizer.zero_grad(set_to_none=True)
        backward_update(model,processor,objects,replays,block,condition,config,device,context,
                        probe_stats,sources=('object',),reference_model=reference_model,vocabulary_size=vocabulary_size)
        check_gradients(model,parameters)
        object_norm = gradient_norm(parameters)
        if not math.isfinite(object_norm) or object_norm <= 0:
            raise ValueError('initial object gradient must be finite and positive')
        object_grad = [p.grad.detach().clone() if p.grad is not None else None for p in parameters]
        optimizer.zero_grad(set_to_none=True)
        backward_update(model,processor,objects,replays,block,condition,config,device,context,
                        probe_stats,sources=('replay',),reference_model=reference_model,vocabulary_size=vocabulary_size,replay_component='kl_only')
        check_gradients(model,parameters)  # Zero KL gradient is valid; absent/nonfinite is not.
        replay_norm = gradient_norm(parameters)
        ratio = replay_norm/object_norm
        record.update({'actual_object_questions':len(block['object_indices']),
            'actual_replay_questions':len(block['replay_indices']),
            'object_gradient_l2':object_norm,'replay_gradient_l2':replay_norm,
            'initial_replay_to_object_gradient_norm_ratio':ratio,
            'initial_max_abs_per_question_kl':probe_stats['replay_max_abs_per_question_kl'],
            'initial_mean_replay_kl':probe_stats['replay_kl_sum']/probe_stats['replay_exposures'],
            'initial_object_mean_ce':probe_stats['object_ce_sum']/probe_stats['object_exposures'],
            'initial_replay_mean_hard_ce_diagnostic':probe_stats['replay_ce_sum']/probe_stats['replay_exposures'],
            'initial_replay_gradient_may_be_zero':True,
            'source_gradient_measurement':'separate source backward then FP32 sum for one discarded step only'})
        if (probe_stats['replay_max_abs_per_question_kl'] > config['teacher_reference']['initial_max_abs_per_question_KL']
                or not math.isfinite(ratio)
                or ratio > config['teacher_reference']['initial_replay_to_object_gradient_norm_max_ratio']):
            raise ValueError('initial online reference parity/gradient admission failed; no tolerance fallback')
        kl_grad=[p.grad.detach().clone() if p.grad is not None else None for p in parameters]
        optimizer.zero_grad(set_to_none=True)
        backward_update(model,processor,objects,replays,block,condition,config,device,context,
                        ce_probe_stats,sources=('replay',),reference_model=reference_model,
                        vocabulary_size=vocabulary_size,replay_component='ce_only')
        check_gradients(model,parameters)
        ce_norm=gradient_norm(parameters)
        if not math.isfinite(ce_norm) or ce_norm<=0:raise ValueError('GT replay CE gradient must be finite and positive')
        gamma=config['replay_gt_ce_weights'][condition]
        for parameter,kg in zip(parameters,kl_grad):
            cg=parameter.grad
            parameter.grad=(torch.zeros_like(parameter) if kg is None else kg.clone())
            if cg is not None:parameter.grad.add_(cg,alpha=gamma)
        record.update(initial_gt_ce_gradient_l2=ce_norm,replay_gt_ce_weight=gamma,
                      initial_joint_replay_gradient_l2=gradient_norm(parameters),
                      initial_parity_gradient_component='KL_only',
                      source_gradient_measurement='separate object/KL/GTCE backwards; object+KL+gamma*CE for one discarded step')
        for parameter,gradient in zip(parameters,object_grad):
            if gradient is not None:
                if parameter.grad is None:
                    parameter.grad = gradient
                else:
                    parameter.grad.add_(gradient)
        check_gradients(model,parameters)
        norm = torch.nn.utils.clip_grad_norm_(parameters,config['gradient_clip_norm'],error_if_nonfinite=True)
        if float(norm) <= 0:
            raise ValueError('temporary update has zero joint gradient')
        enforce_memory_cap(device,config,'preflight')
        optimizer.step()
        completed += 1
        after = trainable_hash(model)
        record.update({'gradient_l2_before_clip':float(norm),'finite_gradients':True,
                       'trainable_sha256_after_temporary_update':after})
        if before == after:
            raise ValueError('temporary update did not change live FP32 LoRA')
        optimizer.zero_grad(set_to_none=True)
        backward_update(model,processor,objects,replays,block,condition,config,device,context,
                        after_stats,sources=('replay',),reference_model=reference_model,vocabulary_size=vocabulary_size,replay_component='kl_only')
        check_gradients(model,parameters)
        restoring_norm = gradient_norm(parameters)
        after_kl = after_stats['replay_kl_sum']/after_stats['replay_exposures']
        record.update({'post_update_mean_replay_kl':after_kl,
                       'post_update_max_abs_per_question_kl':after_stats['replay_max_abs_per_question_kl'],
                       'post_update_replay_gradient_l2':restoring_norm})
        if not math.isfinite(after_kl) or after_kl <= 0 or not math.isfinite(restoring_norm) or restoring_norm <= 0:
            raise ValueError('temporary update did not produce finite positive KL and restoring gradient')
        assert_reference_fixed(reference_model,reference_binding)
        enforce_memory_cap(device,config,'preflight')
    except Exception as exc:
        error = exc
    finally:
        with torch.no_grad():
            for name,parameter in model.named_parameters():
                if name in initial:
                    parameter.copy_(initial[name])
        model.zero_grad(set_to_none=True)
        optimizer.state.clear()
        restore_runtime(runtime_before)
        restore_runtime(reference_runtime_before)
        restore_rng(rng_before)
    restored = trainable_hash(model) == before
    rng_restored = rng_hash(rng_state()) == rng_hash(rng_before)
    runtime_restored = runtime_matches(runtime_before)
    reference_runtime_restored = runtime_matches(reference_runtime_before)
    frozen_after = frozen_hash(model)
    reference_after = None
    prepared_unchanged = False
    try:
        reference_after = assert_reference_fixed(reference_model,reference_binding)
        verify_prepared_files_unchanged(config,context['plan']['seed'],condition,context)
        prepared_unchanged = True
    except Exception as exc:
        if error is None:
            error = exc
    reference_fixed = reference_after is not None
    allocated = torch.cuda.max_memory_allocated(device)/2**30 if device.type == 'cuda' else 0.
    reserved = torch.cuda.max_memory_reserved(device)/2**30 if device.type == 'cuda' else 0.
    if not all((restored,rng_restored,runtime_restored,reference_runtime_restored,reference_fixed,frozen_after==frozen_before)):
        error = ValueError('preflight exact master/RNG/runtime/reference/frozen restore failed')
    if error is None and max(allocated,reserved) > 65:
        error = RuntimeError(f'preflight_memory_cap_exceeded: allocated={allocated} reserved={reserved} cap=65GiB')
    actor_completed = sum(st[f'{source}_exposures'] for st in (probe_stats,ce_probe_stats,after_stats) for source in ('object','replay'))
    actor_started = sum(st[f'{source}_forward_started_questions'] for st in (probe_stats,ce_probe_stats,after_stats) for source in ('object','replay'))
    result = {'schema_version':SCHEMA,'status':'failed' if error else 'passed','condition':condition,'completed_at':now(),
        'scope':'single embedded engineering admission; discarded temporary update and post-update KL backward only',
        'discarded_engineering_optimizer_updates':completed,'formal_optimizer_updates':0,
        'engineering_actor_forward_backward_questions':actor_completed,
        'engineering_actor_training_forward_started_questions':actor_started,
        'engineering_actor_forward_only_probe_questions':0,'teacher_generation_questions':0,
        'engineering_reference_forward_questions':sum(st['reference_forward_questions'] for st in (probe_stats,ce_probe_stats,after_stats)),
        'engineering_reference_forward_started_questions':sum(st['reference_forward_started_questions'] for st in (probe_stats,ce_probe_stats,after_stats)),
        'engineering_reference_forward_seconds':sum(st['reference_forward_seconds'] for st in (probe_stats,ce_probe_stats,after_stats)),
        'reference_rng_unchanged_microbatches':sum(st['reference_rng_unchanged_microbatches'] for st in (probe_stats,ce_probe_stats,after_stats)),
        'reference_rng_unchanged':all(st['reference_forward_started_questions'] == st['reference_forward_questions']
            == st['reference_rng_unchanged_microbatches']*4 for st in (probe_stats,ce_probe_stats,after_stats)),
        'reference_parameters_unchanged':reference_fixed,'reference_state_after':reference_after,
        'reference_runtime_buffers_and_rope_cache_restored':reference_runtime_restored,
        'teacher_probability_cache_used':False,'engineering_learning_rate':config['preflight_lr'],
        'formal_learning_rate':config['lr'],'initial_parameters_restored_exactly':restored,
        'rng_restored':rng_restored,'model_runtime_buffers_and_rope_cache_restored':runtime_restored,
        'engineering_optimizer_discarded':True,'frozen_parameters_unchanged':frozen_after==frozen_before,
        'prepared_inputs_and_teacher_replies_unchanged':prepared_unchanged,
        'lineage_sha256':lineage_sha256,
        'frozen_parameter_sha256_before':frozen_before,'frozen_parameter_sha256_after':frozen_after,
        'trainable_sha256_before':before,'trainable_parameters':sum(p.numel() for p in parameters),
        'training_microbatch_questions':4,'training_accumulation':4,'object_effective_batch':16,
        'peak_allocated_gib':allocated,'peak_reserved_gib':reserved,'memory_cap_gib':65,
        'memory_cap_includes_actor_and_reference':True,'memory_cap_passed':max(allocated,reserved)<=65,'record':record}
    if error:
        result['error_type'],result['error'] = type(error).__name__,str(error)
    write_json(directory/'PREFLIGHT.json',result)
    with (directory/'PREFLIGHT_HISTORY.jsonl').open('a') as stream:
        stream.write(json.dumps(result,ensure_ascii=False,allow_nan=False)+'\n')
        stream.flush()
        os.fsync(stream.fileno())
    if error:
        raise error
    return result


def verify_preflight_record(result, initial_sha, lineage_sha, condition):
    fixed = {'schema_version':SCHEMA,'status':'passed','condition':condition,
        'discarded_engineering_optimizer_updates':1,'formal_optimizer_updates':0,
        'engineering_actor_forward_backward_questions':112,
        'engineering_actor_training_forward_started_questions':112,
        'engineering_actor_forward_only_probe_questions':0,'teacher_generation_questions':0,
        'engineering_reference_forward_questions':96,'engineering_reference_forward_started_questions':96,
        'reference_rng_unchanged_microbatches':24,'reference_rng_unchanged':True,
        'reference_parameters_unchanged':True,'reference_runtime_buffers_and_rope_cache_restored':True,
        'initial_parameters_restored_exactly':True,'rng_restored':True,
        'model_runtime_buffers_and_rope_cache_restored':True,'engineering_optimizer_discarded':True,
        'frozen_parameters_unchanged':True,'prepared_inputs_and_teacher_replies_unchanged':True,
        'memory_cap_gib':65,'memory_cap_includes_actor_and_reference':True,'memory_cap_passed':True,
        'trainable_sha256_before':initial_sha,'lineage_sha256':lineage_sha}
    for key,value in fixed.items():
        if result.get(key) != value or isinstance(result.get(key),bool) != isinstance(value,bool):
            raise ValueError(f'embedded preflight receipt differs: {key}')
    record = result['record']
    if (record.get('initial_parity_gradient_component')!='KL_only' or record.get('replay_gt_ce_weight')!=(1. if condition==CANDIDATE else 0.)
        or not math.isfinite(record.get('initial_gt_ce_gradient_l2',float('nan'))) or record['initial_gt_ce_gradient_l2']<=0):
        raise ValueError('separate GT CE gradient and fixed gamma evidence required')
    for key in ('object_gradient_l2','replay_gradient_l2','initial_replay_to_object_gradient_norm_ratio',
                'initial_max_abs_per_question_kl','initial_mean_replay_kl','post_update_mean_replay_kl',
                'post_update_replay_gradient_l2','post_update_max_abs_per_question_kl','gradient_l2_before_clip'):
        if not math.isfinite(record[key]):
            raise ValueError('preflight has nonfinite measured diagnostics')
    if (record['object_gradient_l2'] <= 0 or record['replay_gradient_l2'] < 0
            or not 0 <= record['initial_replay_to_object_gradient_norm_ratio'] <= .01
            or record['initial_replay_to_object_gradient_norm_ratio'] != record['replay_gradient_l2']/record['object_gradient_l2']
            or not 0 <= record['initial_max_abs_per_question_kl'] <= 1e-5
            or record.get('actual_object_questions') != 16 or record.get('actual_replay_questions') != 32
            or record['gradient_l2_before_clip'] <= 0
            or record['post_update_max_abs_per_question_kl'] <= 0
            or record['post_update_mean_replay_kl'] <= 0 or record['post_update_replay_gradient_l2'] <= 0
            or record['trainable_sha256_after_temporary_update'] == initial_sha
            or max(result['peak_allocated_gib'],result['peak_reserved_gib']) > 65):
        raise ValueError('preflight measured parity/gradient/update/memory admission failed')
    return True


def run(args):
    import numpy as np
    from vlm_joint.evaluate import visible_gpu_binding
    from vlm_probe.evaluate import processor_for
    session_started = time.monotonic()
    config_path = private_path(args.config)
    config = validate_config(json.loads(config_path.read_text()))
    validate_condition(args.condition,args.seed)
    parameter_anchor = parent_parameter_anchor(config)
    context = verify_prepared_training_inputs(config,args.seed,args.condition)
    objects,replays = context['objects'],context['replays']
    plan,blocks = context['plan'],context['plan']['blocks']
    schedule,schedule_sha = plan['schedule'],content_sha(plan['schedule'])
    directory = private_path(args.output_root)/'runs'/args.condition/f'seed{args.seed}'/'train'
    directory.mkdir(parents=True,exist_ok=True)
    if (directory/'LATEST.json').exists() and not args.resume:
        raise ValueError('checkpoint exists; explicitly use --resume')
    if args.resume and not (directory/'LATEST.json').exists():
        raise ValueError('--resume requires an existing exact checkpoint')
    if int(os.environ.get('WORLD_SIZE','1')) != 1:
        raise ValueError('single visible GPU required')
    hardware = visible_gpu_binding()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError('original BF16 CUDA device required')
    torch.cuda.set_device(0)
    device = torch.device('cuda',0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    reference_path = private_path(config['reference_config'])
    reference = json.loads(reference_path.read_text())
    model_root,adapter_root = private_path(config['model_root']),private_path(config['initial_adapter'])
    adapter_config = json.loads((adapter_root/'adapter_config.json').read_text())
    if (adapter_config['r'] != 8 or adapter_config['lora_alpha'] != 16 or adapter_config['bias'] != 'none'
            or set(adapter_config['target_modules']) != {'q_proj','v_proj'}):
        raise ValueError('initial adapter must be fixed languageq/v rank8alpha16')
    p = input_paths(config,args.seed,args.condition)
    static_lock = {'schema_version':SCHEMA,'condition':args.condition,'seed':args.seed,
        'config_sha256':content_sha(config),'config_file_sha256':sha_file(config_path),
        'reference_config_sha256':sha_file(reference_path),'source_sha256':source_hashes(),
        'packages':package_lock(),'hardware':hardware,
        'data_binding':{key:context['manifest'][key] for key in ('data_admission_file_sha256',
                         'train_manifest_file_sha256','replay_manifest_file_sha256',
                         'plan_file_sha256','prompt_manifest_file_sha256')},
        'train_manifest_sha256':sha_file(private_path(config['train_manifest'])),
        'train_manifest_file_sha256':sha_file(private_path(config['train_manifest'])),
        'replay_manifest_sha256':sha_file(private_path(config['replay_manifest'])),
        'model_files_sha256':file_hashes(model_root),
        'model_provenance_sha256':sha_file(model_root.parent/'model_provenance.json'),
        'initial_adapter_files_sha256':file_hashes(adapter_root),'initial_adapter_dir':str(adapter_root),
        'object_question_count':len(objects),'question_count':len(objects),
        'unique_training_images':len({str(row['image_id']) for row in objects}),
        'planned_updates':len(blocks),'planned_object_exposures':2888,'planned_replay_exposures':5792,
        'object_order_sha256':content_sha(schedule['object_question_ids_by_update']),
        'order_sha256':content_sha(schedule['object_question_ids_by_update']),
        'replay_order_sha256':content_sha(schedule['replay_question_ids_by_update']),
        'object_microbatch_order_sha256':content_sha(schedule['object_microbatches_by_update']),
        'replay_microbatch_order_sha256':content_sha(schedule['replay_microbatches_by_update']),
        'schedule_sha256':schedule_sha,'training_microbatch_questions':4,'training_accumulation':4,
        'object_effective_batch':16,'replay_questions_per_update':32,'replay_lambda':1.,
        'loss':'mean_QA CE_object + mean_QA fullvocab KL_initial_actor_GTprefix + gamma*mean_QA CE_GT_replay',
        'replay_objective':'forward_kl_plus_GT_CE','kl_temperature':1.,
        'kl_direction':'teacher_to_actor_forward_KL','kl_vocabulary_reduction':'sum',
        'teacher_reference':config['teacher_reference'],
        'parent_input_layout_condition':'gt_ce32','parent_config_file':config['parent_config_file'],
        'parent_proof_files_sha256':config['parent_proof_files_sha256'],
        'explicit_parent_numeric_source_sha256':PARENT_SOURCE_SHA256,
        'parent_parameter_anchor':parameter_anchor,
        'normalization':'independent actual object/replay QA means; last object block denominator8',
        'checkpoint_selection':'fixed Final@1 epoch; no dev selection',
        'numerical_path':'original_bf16','base_precision':'bf16','lora_parameter_precision':'fp32',
        'lora_initialization':'initial SFT adapter rounded to BF16 then exactly promoted to FP32',
        'autocast':'bf16','dropout':0.,'visual_token_budget':1024,'max_sequence_length':2048,
        'object_prompt_suffix':config['object_prompt_suffix'],'replay_prompt_suffix':config['replay_prompt_suffix'],
        'input_truncation':False,'replay_hard_ce_backward':args.condition==CANDIDATE,'replay_gt_ce_weight':config['replay_gt_ce_weights'][args.condition],'teacher_probability_cache_used':False,
        'teacher_reply_generation_new_training_calls':0,'teacher_replies_manifest_file_sha256':sha_file(p['teacher']),
        'teacher_replies_lock_file_sha256':sha_file(p['teacher_lock']),
        'teacher_replies_manifest_path':str(p['teacher']),'teacher_replies_lock_path':str(p['teacher_lock']),
        'prepared_training_inputs_file_sha256':context['manifest_file_sha256'],
        'prepared_training_inputs_content_sha256':content_sha(context['manifest']),
        'prepared_training_contracts_file_sha256':context['manifest']['contracts_file_sha256'],
        'prepared_input_streams':context['manifest']['streams'],
        'replay_target_source':context['manifest']['replay_target_source'],
        'position_diagnostics_extra_model_calls':0,'position_diagnostics_change_loss_weights':False,
        'teacher_generation_accounting_reference':{key:value for key,value in context['teacher_replies_manifest'].items()
                                                    if key != 'rows'},
        'teacher_generation_accounting_scope':'GT targets only; no rollout generation in any seed; legacy field names retained explicitly',
        'equal_compute_claim':False,
        'helper_provenance':['frozen object load_actor/init/Adam/LR/checkpoint/RNG/runtime helpers',
                             'explicit immutable parent CE copy; new full-vocabulary forwardKL on identical GT-prefix inputs']}
    lock_json(directory/'STATIC_INPUT_LOCK.json',static_lock)
    lock_json(directory/'SCHEDULE.json',schedule)
    torch.cuda.reset_peak_memory_stats(device)
    loading_start = time.monotonic()
    model,parameters,dtypes = load_actor(config,reference,device)
    processor = processor_for(model_root,reference,config['visual_token_budget'])
    sync(device)
    loading_seconds = time.monotonic()-loading_start
    initial = {name:cpu_state(parameter) for name,parameter in model.named_parameters() if parameter.requires_grad}
    initial_sha = trainable_hash(model)
    if len(parameters) != 144 or any(p.dtype != torch.float32 for p in parameters):
        raise ValueError('actor must retain exactly144 FP32 language q/v LoRA master tensors')
    if initial_sha != config['teacher_reference']['initial_runtime_master_sha256']:
        raise ValueError('actor initial FP32 master differs from fixed teacher initial master')
    frozen_before = frozen_hash(model)
    if frozen_before != parameter_anchor['frozen_base_vision_sha256']:
        raise ValueError('actual actor frozen base/vision differs from accepted parent identity')
    reference_load_started = time.monotonic()
    reference_model,reference_binding = load_reference(config,reference,device,model,initial_sha,frozen_before)
    sync(device)
    reference_load_seconds = time.monotonic()-reference_load_started
    vocabulary_size = int(model.get_output_embeddings().weight.shape[0])
    if vocabulary_size != 151936 or vocabulary_size != int(reference_model.get_output_embeddings().weight.shape[0]):
        raise ValueError('actor/reference full model head vocabulary differs')
    enforce_memory_cap(device,config,'two_model_load')
    lock = {**static_lock,'initial_adapter_tensor_sha256':initial_sha,
            'frozen_parameter_sha256_before':frozen_before,'dtypes':dtypes,
            'teacher_reference':reference_binding,'full_vocabulary_size':vocabulary_size,
            'trainable_parameters':sum(p.numel() for p in parameters),
            'trainable_names':{name:p.numel() for name,p in model.named_parameters() if p.requires_grad}}
    lock_json(directory/'TRAIN_INPUT_LOCK.json',lock)
    lineage_sha = content_sha(lock)
    preflight_seconds = 0.
    preflight_reused = (directory/'PREFLIGHT.json').is_file()
    if args.resume and not preflight_reused:
        raise ValueError('resume requires the original completed embedded preflight receipt')
    if preflight_reused:
        preflight_result = json.loads((directory/'PREFLIGHT.json').read_text())
        verify_preflight_record(preflight_result,initial_sha,lineage_sha,args.condition)
    else:
        preflight_started = time.monotonic()
        preflight_result = preflight(model,parameters,processor,objects,replays,blocks[0],config,args.condition,
            directory,device,context,reference_model,reference_binding,vocabulary_size,
            frozen_sha256=frozen_before,lineage_sha256=lineage_sha)
        sync(device)
        preflight_seconds = time.monotonic()-preflight_started
        verify_preflight_record(preflight_result,initial_sha,lineage_sha,args.condition)
    preflight_file_sha = sha_file(directory/'PREFLIGHT.json')
    optimizer = torch.optim.AdamW(parameters,lr=config['lr'],weight_decay=config['weight_decay'])
    stats,completed = initial_stats(),0
    stats['embedded_preflight_file_sha256'] = preflight_file_sha
    if args.resume:
        completed,stats = resume_checkpoint(directory,model,optimizer,lineage_sha,schedule_sha,len(blocks))
        validate_exposure_cursor(stats,blocks,completed,context)
        if stats.get('embedded_preflight_file_sha256') != preflight_file_sha:
            raise ValueError('resume embedded preflight file SHA differs from checkpoint')
    if (directory/'TRAIN_RESULT.json').exists():
        result = json.loads((directory/'TRAIN_RESULT.json').read_text())
        if result['lineage_sha256'] != lineage_sha or file_hashes(directory/'adapter') != result['adapter_files_sha256']:
            raise ValueError('completed result adapter/lineage changed')
        return result
    write_json(directory/'PROGRESS.json',{'status':'training','completed_updates':completed,
        'planned_updates':len(blocks),'actual_object_exposures':stats['object_exposures'],
        'actual_replay_exposures':stats['replay_exposures'],'updated_at':now()})
    torch.cuda.reset_peak_memory_stats(device)
    for step in range(completed,len(blocks)):
        sync(device)
        begun = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:
            group['lr'] = learning_rate(step,len(blocks),config)
        teaching = [] if step == 0 else None
        for source in ('object','replay'):
            phase = time.monotonic()
            backward_update(model,processor,objects,replays,blocks[step],args.condition,config,device,context,
                            stats,teaching,sources=(source,),update=step,
                            reference_model=reference_model,vocabulary_size=vocabulary_size)
            sync(device)
            stats[f'{source}_active_seconds'] += time.monotonic()-phase
        check_gradients(model,parameters)
        norm = torch.nn.utils.clip_grad_norm_(parameters,config['gradient_clip_norm'],error_if_nonfinite=True)
        optimizer.step()
        sync(device)
        enforce_memory_cap(device,config,'formal_training')
        stats['training_seconds'] += time.monotonic()-begun
        stats['last_gradient_l2'] = float(norm)
        if teaching is not None:
            write_json(directory/'FIRST_UPDATE.json',{'scope':'actual first formal update inputs and exact supervised IDs',
                'completed_updates':1,'learning_rate':optimizer.param_groups[0]['lr'],
                'gradient_l2_before_clip':float(norm),'object_count':len(blocks[step]['object_indices']),
                'replay_count':len(blocks[step]['replay_indices']),'records':teaching,
                'object_mean_ce':stats['object_ce_sum']/stats['object_exposures'],
                'replay_mean_kl':stats['replay_kl_sum']/stats['replay_exposures'],
                'replay_hard_ce_diagnostic':stats['replay_ce_sum']/stats['replay_exposures'],
                'replay_hard_ce_backward':args.condition==CANDIDATE,'replay_gt_ce_weight':config['replay_gt_ce_weights'][args.condition],'full_vocabulary_size':vocabulary_size,
                'reference_forward_questions':stats['reference_forward_questions']})
        progress = {'status':'training','completed_updates':step+1,'planned_updates':len(blocks),
            'actual_object_exposures':stats['object_exposures'],'actual_replay_exposures':stats['replay_exposures'],
            'training_active_seconds':stats['training_seconds'],'gradient_l2':float(norm),
            'formal_reference_forward_questions':stats['reference_forward_questions'],
            'reference_forward_seconds':stats['reference_forward_seconds'],'updated_at':now()}
        if step == 0 or (step+1)%25 == 0 or step+1 == len(blocks):
            print('OBJECT_TRAIN_UPDATE',json.dumps({'condition':args.condition,**progress}),flush=True)
        if (step+1)%25 == 0 or step+1 == len(blocks):
            validate_exposure_cursor(stats,blocks,step+1,context)
            save_checkpoint(directory,model,optimizer,step+1,stats,lineage_sha,schedule_sha)
            write_json(directory/'PROGRESS.json',progress)
    validate_exposure_cursor(stats,blocks,len(blocks),context)
    verify_prepared_files_unchanged(config,args.seed,args.condition,context)
    reference_after = assert_reference_fixed(reference_model,reference_binding)
    frozen_after = frozen_hash(model)
    if frozen_before != frozen_after:
        raise ValueError('frozen base/vision changed during formal training')
    delta_sq,maximum,changed = 0.,0.,0
    for name,parameter in model.named_parameters():
        if parameter.requires_grad:
            value = parameter.detach().cpu()
            if not bool(torch.isfinite(value).all()):
                raise FloatingPointError('nonfinite Final FP32 master')
            delta = value-initial[name]
            delta_sq += float(delta.double().square().sum())
            maximum = max(maximum,float(delta.abs().max()))
            changed += int(bool((delta != 0).any()))
    if changed == 0:
        raise ValueError('Final LoRA did not update')
    model.save_pretrained(directory/'adapter',safe_serialization=True)
    inputs_file = directory/'TRAINED_INPUT_CONTRACTS.json'
    write_json(inputs_file,{source:stats[f'{source}_input_contracts'] for source in ('object','replay')})
    consumed = [index for block in blocks for index in block['replay_indices']]
    result = {**lock,'lineage_sha256':lineage_sha,'status':'completed','completed_at':now(),
        'completed_updates':len(blocks),'epochs_completed':1,'skipped_updates':0,
        'actual_questions_seen':stats['object_exposures']+stats['replay_exposures'],
        'actual_object_exposures':stats['object_exposures'],'actual_replay_exposures':stats['replay_exposures'],
        'trained_input_contracts_file_sha256':sha_file(inputs_file),
        'replay_coverage':coverage(replays,consumed),'replay_rng_unchanged_updates':stats['replay_rng_unchanged_checks'],
        'mean_object_ce':stats['object_ce_sum']/stats['object_exposures'],
        'mean_replay_ce':stats['replay_ce_sum']/stats['replay_exposures'],
        'mean_replay_ce_scope':'same actor logits; GT CE differentiable in gamma1 candidate, gamma0 control diagnostic only',
        'mean_replay_kl':stats['replay_kl_sum']/stats['replay_exposures'],
        'replay_kl_position_role_statistics':summarize_kl_roles(stats['replay_kl_role_statistics']),
        'teacher_reference_after':reference_after,'reference_parameters_unchanged':True,
        'formal_reference_forward_questions':stats['reference_forward_questions'],
        'formal_reference_forward_started_questions':stats['reference_forward_started_questions'],
        'reference_rng_unchanged_microbatches':stats['reference_rng_unchanged_microbatches'],
        'reference_forward_seconds':stats['reference_forward_seconds'],
        'formal_reference_forward_accounting_scope':'completed committed optimizer-update stream; engineering separately; abandoned uncheckpointed attempts are not included',
        'position_role_statistics':{source:summarize_roles(stats[f'{source}_role_statistics']) for source in ('object','replay')},
        'teacher_replies_manifest_file_sha256_after':sha_file(p['teacher']),
        'teacher_replies_lock_file_sha256_after':sha_file(p['teacher_lock']),
        'prepared_inputs_and_teacher_replies_unchanged':True,
        'training_active_seconds':stats['training_seconds'],'model_load_seconds':loading_seconds,
        'reference_model_load_seconds_current_session':reference_load_seconds,
        'preflight_seconds_current_session':preflight_seconds,'preflight':preflight_result,
        'preflight_file_sha256':preflight_file_sha,'embedded_preflight_reused_on_resume':bool(args.resume),'embedded_preflight_reused':preflight_reused,
        'session_wall_seconds':time.monotonic()-session_started,
        'lora_delta_l2':math.sqrt(delta_sq),'lora_delta_max_abs':maximum,'changed_lora_tensors':changed,
        'final_lora_tensor_sha256':trainable_hash(model),'frozen_parameter_sha256_after':frozen_after,
        'frozen_parameters_unchanged':True,'resumed_at_update':completed,
        'peak_gpu_allocated_gib':max(torch.cuda.max_memory_allocated(device)/2**30,preflight_result['peak_allocated_gib']),
        'peak_gpu_reserved_gib':max(torch.cuda.max_memory_reserved(device)/2**30,preflight_result['peak_reserved_gib']),
        'adapter_files_sha256':file_hashes(directory/'adapter'),
        'compute_accounting':{'equal_compute_claim':False,'measured_flops':None,
            'teacher_generation_new_calls_in_training':0,'teacher_probability_forward_questions':stats['reference_forward_questions'],
            'teacher_probability_file_bytes':0,'position_diagnostics_extra_model_calls':0,
            'formal_reference_forward_seconds':stats['reference_forward_seconds'],
            'reference_forward_scope':'online no_grad reference only, synchronized; included in replay active seconds; committed stream, not abandoned attempts',
            'embedded_preflight_file_sha256':preflight_file_sha,
            'engineering_reference_forward_questions':preflight_result['engineering_reference_forward_questions'],
            'engineering_reference_forward_seconds':preflight_result['engineering_reference_forward_seconds'],
            'engineering_actor_forward_backward_questions':preflight_result['engineering_actor_forward_backward_questions'],
            'discarded_engineering_optimizer_updates':preflight_result['discarded_engineering_optimizer_updates'],
            'teacher_generation_cost_source_manifest':str(p['teacher']),
            'teacher_generation_cost_source_lock':str(p['teacher_lock']),
            'object_forward_backward_questions':stats['object_exposures'],
            'extra_replay_forward_backward_questions':stats['replay_exposures'],
            'object_active_seconds':stats['object_active_seconds'],'extra_replay_active_seconds':stats['replay_active_seconds']}}
    for source in ('object','replay'):
        rows = stats[f'{source}_input_contracts']
        result[f'{source}_input_contract_sha256'] = content_sha(rows)
        result[f'{source}_legacy_input_contract_sha256'] = content_sha(legacy_contracts(rows))
        result[f'{source}_supervised_target_ids_stream_sha256'] = content_sha([r['supervised_target_ids'] for r in rows])
        result[f'{source}_supervised_answer_roles_stream_sha256'] = content_sha([r['supervised_answer_roles'] for r in rows])
        for name in ('input_tokens','visual_tokens','answer_tokens','microbatches','padding_tokens'):
            result[f'training_{source}_{name}'] = stats[f'{source}_{name}']
            result['compute_accounting'][f'{source}_{name}'] = stats[f'{source}_{name}']
    write_json(directory/'TRAIN_RESULT.json',result)
    write_json(directory/'PROGRESS.json',{'status':'completed','completed_updates':len(blocks),
        'planned_updates':len(blocks),'actual_object_exposures':2888,'actual_replay_exposures':5792,
        'training_active_seconds':stats['training_seconds'],'updated_at':now()})
    print('TRAINING_COMPLETED',args.condition,args.seed,flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--condition',required=True,choices=CONDITIONS)
    parser.add_argument('--seed',required=True,type=int)
    parser.add_argument('--output-root',required=True)
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    try:
        run(args)
    except Exception as exc:
        directory = private_path(args.output_root)/'runs'/args.condition/f'seed{args.seed}'/'train'
        directory.mkdir(parents=True,exist_ok=True)
        allocated=reserved=0.
        if torch.cuda.is_initialized():
            allocated=torch.cuda.max_memory_allocated()/2**30
            reserved=torch.cuda.max_memory_reserved()/2**30
        write_json(directory/'TRAINING_FAILURE.json',{'schema_version':SCHEMA,'status':'failed',
            'condition':args.condition,'seed':args.seed,'failed_at':now(),
            'error_type':type(exc).__name__,'error':str(exc),'peak_allocated_gib':allocated,
            'peak_reserved_gib':reserved,'memory_cap_gib':65,'runtime_fallback':False})
        raise


if __name__ == '__main__':
    main()
