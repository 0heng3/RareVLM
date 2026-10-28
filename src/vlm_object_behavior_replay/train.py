"""Final-only object/behavior hard CE, with immutable generated-token targets."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import time

for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
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
from . import SCHEMA, CANDIDATE, CONTROL, CONDITIONS
from .inputs import (encode_group, legacy_contracts, paths as input_paths,
                     verify_prepared_training_inputs, verify_prepared_files_unchanged)
from .loss import (accumulate_roles, empty_role_statistics, normalized_qa_loss,
                   sequence_ce, summarize_roles)


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
    if (config.get('candidates') != [CANDIDATE] or config.get('exploratory_control') != CONTROL
            or config.get('control_can_be_selected') is not False):
        raise ValueError('only teacher_ce32 can be selected; gt_ce32 is exploratory only')
    generation = config.get('teacher_generation',{})
    for key,value in {'shards':2,'batch_size':4,'max_new_tokens':32,'do_sample':False,
                      'plan_questions_per_seed':5792,'qa_per_shard':2896,
                      'microbatches_per_shard':724,'retokenize_teacher_answers':False,
                      'actual_generated_ids_required':True,
                      'filter_or_resample_teacher_answers':False,
                      'initial_runtime_master_sha256':'6a44359406c53261b254c3a19393c63187b34af09baec539d6b235ca148c0fa7'}.items():
        if generation.get(key) != value:
            raise ValueError(f'teacher generation fixed contract differs: {key}')
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
    if condition not in CONDITIONS or type(seed) is not int or seed not in (0,42,200) or (condition == CONTROL and seed != 42):
        raise ValueError('teacher_ce32 allows0/42/200; gt_ce32 is seed42 exploratory only')


def source_hashes():
    result = legacy.source_hashes()
    for path in sorted(Path(__file__).parent.glob('*.py')):
        result[str(path.relative_to(SOURCE_ROOT))] = sha_file(path)
    return result


def initial_stats():
    result = {'training_seconds':0., 'last_gradient_l2':None, 'replay_rng_unchanged_checks':0}
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


def enforce_memory_cap(device, config, phase):
    if device.type == 'cuda':
        allocated, reserved = torch.cuda.max_memory_allocated(device)/2**30, torch.cuda.max_memory_reserved(device)/2**30
        if max(allocated,reserved) > config['preflight_memory_cap_gib']:
            raise RuntimeError(f'{phase}_memory_cap_exceeded: allocated={allocated:.6f}GiB reserved={reserved:.6f}GiB cap=65GiB; no batch fallback')


def backward_update(model, processor, objects, replays, block, condition, config, device, context,
                    stats=None, teaching_records=None, sources=('object','replay'), update=0):
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
                raise ValueError('actual forward input/target IDs/roles differ from CPU frozen training stream')
            if teaching_records is not None:
                for position,(index,contract,sample) in enumerate(zip(selected,contracts,samples)):
                    row = rows[index]
                    teaching_records.append({'source':source,**contract,'question':row['question'],
                        'original_gt_for_provenance_only':row['target'],'prompt_suffix':row['prompt_suffix'],
                        'image_path':row['image_path'],'input_ids':batch['input_ids'][position].tolist(),
                        'labels':batch['labels'][position].tolist(),'answer_roles':batch['answer_roles'][position].tolist(),
                        'image_grid_thw':sample['image_grid_thw'].tolist(),
                        'pixel_values_shape':list(sample['pixel_values'].shape)})
            padding = batch['input_ids'].numel()-int(batch['attention_mask'].sum())
            batch = to_device(batch,device)
            if stats is not None:
                stats[f'{source}_forward_started_questions'] += len(selected)
            with torch.enable_grad(),autocast(device):
                ce,counts,role_stats = sequence_ce(model,batch)
                if ce.numel() != len(selected) or counts.tolist() != [c['assistant_tokens'] for c in contracts]:
                    raise ValueError('actual per-QA assistant count differs from frozen contract')
                loss = normalized_qa_loss(ce,source,len(obj),len(rep),config['replay_lambda'])
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('nonfinite hard CE')
            loss.backward()
            enforce_memory_cap(device,config,'training')
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
        if replay_rng is not None:
            if rng_hash(rng_state()) != replay_rng:
                raise ValueError('replay advanced RNG; frozen dropout0 stochastic-path contract differs')
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


def preflight(model, parameters, processor, objects, replays, block, config, condition, directory,
              device, context, frozen_sha256=None):
    """One discarded source-checked temporary Adam update; restore every state."""
    initial = {name:cpu_state(p) for name,p in model.named_parameters() if p.requires_grad}
    before = trainable_hash(model)
    frozen_before = frozen_sha256 or frozen_hash(model)
    rng_before,runtime_before = rng_state(),runtime_state(model)
    optimizer = torch.optim.AdamW(parameters,lr=config['preflight_lr'],weight_decay=config['weight_decay'])
    error,completed,record = None,0,{}
    probe_stats = initial_stats()
    probe_forward_started = probe_forward_completed = 0
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    try:
        from .data import encode_training_sample,collate_training
        sample = encode_training_sample(processor,objects[block['object_indices'][0]],config,
            source='object',condition=condition,teacher_replies=context['teacher_replies'])
        probe = to_device(collate_training([sample],processor.tokenizer.pad_token_id),device)
        probe_forward_started += 1
        with torch.enable_grad(),autocast(device):
            before_ce,_,_ = sequence_ce(model,probe)
        probe_forward_completed += 1
        before_ce = before_ce.detach()
        optimizer.zero_grad(set_to_none=True)
        backward_update(model,processor,objects,replays,block,condition,config,device,context,
                        probe_stats,sources=('object',),update=0)
        check_gradients(model,parameters)
        object_grad = [p.grad.detach().clone() if p.grad is not None else None for p in parameters]
        object_norm = math.sqrt(sum(float(g.double().square().sum()) for g in object_grad if g is not None))
        optimizer.zero_grad(set_to_none=True)
        backward_update(model,processor,objects,replays,block,condition,config,device,context,
                        probe_stats,sources=('replay',),update=0)
        check_gradients(model,parameters)
        replay_norm = math.sqrt(sum(float(p.grad.detach().double().square().sum()) for p in parameters if p.grad is not None))
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
        probe_forward_started += 1
        with torch.enable_grad(),autocast(device):
            after_ce,_,_ = sequence_ce(model,probe)
        probe_forward_completed += 1
        delta = float((after_ce.detach()-before_ce).abs().max())
        after = trainable_hash(model)
        if before == after or not math.isfinite(delta) or delta <= 0:
            raise ValueError('temporary update did not change live LoRA/assistant CE')
        record = {'actual_object_questions':len(block['object_indices']),
                  'actual_replay_questions':len(block['replay_indices']),
                  'object_gradient_l2':object_norm,'replay_gradient_l2':replay_norm,
                  'gradient_l2_before_clip':float(norm),'finite_gradients':True,
                  'source_gradient_measurement':'separate source backward, then FP32 sum for discarded step only',
                  'initial_object_mean_ce':probe_stats['object_ce_sum']/probe_stats['object_exposures'],
                  'initial_replay_mean_ce':probe_stats['replay_ce_sum']/probe_stats['replay_exposures'],
                  'initial_ce_required_zero':False,'probe_ce_abs_difference_after_temporary_update':delta,
                  'trainable_sha256_after_temporary_update':after}
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
        restore_rng(rng_before)
    restored = trainable_hash(model) == before
    rng_restored = rng_hash(rng_state()) == rng_hash(rng_before)
    runtime_restored = runtime_matches(runtime_before)
    frozen_after = frozen_hash(model)
    verify_prepared_files_unchanged(config,context['plan']['seed'],condition,context)
    allocated = torch.cuda.max_memory_allocated(device)/2**30 if device.type == 'cuda' else 0.
    reserved = torch.cuda.max_memory_reserved(device)/2**30 if device.type == 'cuda' else 0.
    if not restored or not rng_restored or not runtime_restored or frozen_after != frozen_before:
        error = ValueError('preflight exact master/RNG/runtime/frozen restore failed')
    if error is None and max(allocated,reserved) > 65:
        error = RuntimeError(f'preflight_memory_cap_exceeded: allocated={allocated} reserved={reserved} cap=65GiB')
    result = {'status':'failed' if error else 'passed','condition':condition,'completed_at':now(),
              'scope':'engineering only; temporary optimizer discarded before fixed Final training',
              'discarded_engineering_optimizer_updates':completed,'formal_optimizer_updates':0,
              'engineering_actor_forward_backward_questions':probe_stats['object_exposures']+probe_stats['replay_exposures'],
              'engineering_actor_training_forward_started_questions':probe_stats['object_forward_started_questions']+probe_stats['replay_forward_started_questions'],
              'engineering_actor_forward_only_probe_questions':probe_forward_completed,
              'engineering_actor_forward_only_probe_started_questions':probe_forward_started,'teacher_generation_questions':0,
              'teacher_probability_cache_used':False,'engineering_learning_rate':config['preflight_lr'],
              'formal_learning_rate':config['lr'],'initial_parameters_restored_exactly':restored,
              'rng_restored':rng_restored,'model_runtime_buffers_and_rope_cache_restored':runtime_restored,
              'engineering_optimizer_discarded':True,'frozen_parameters_unchanged':frozen_after==frozen_before,
              'prepared_inputs_and_teacher_replies_unchanged':True,
              'frozen_parameter_sha256_before':frozen_before,'frozen_parameter_sha256_after':frozen_after,
              'trainable_sha256_before':before,'trainable_parameters':sum(p.numel() for p in parameters),
              'training_microbatch_questions':4,'training_accumulation':4,'object_effective_batch':16,
              'peak_allocated_gib':allocated,'peak_reserved_gib':reserved,'memory_cap_gib':65,
              'memory_cap_passed':max(allocated,reserved)<=65,'record':record}
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


def run(args):
    import numpy as np
    from vlm_joint.evaluate import visible_gpu_binding
    from vlm_probe.evaluate import processor_for
    session_started = time.monotonic()
    config_path = private_path(args.config)
    config = validate_config(json.loads(config_path.read_text()))
    validate_condition(args.condition,args.seed)
    if args.resume and args.preflight_only:
        raise ValueError('--resume and --preflight-only are mutually exclusive')
    context = verify_prepared_training_inputs(config,args.seed,args.condition)
    objects,replays = context['objects'],context['replays']
    plan,blocks = context['plan'],context['plan']['blocks']
    schedule,schedule_sha = plan['schedule'],content_sha(plan['schedule'])
    directory = private_path(args.output_root)/'runs'/args.condition/f'seed{args.seed}'/'train'
    directory.mkdir(parents=True,exist_ok=True)
    if (directory/'LATEST.json').exists() and not args.resume and not args.preflight_only:
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
    torch.set_num_threads(2)
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
        'loss':'mean_QA mean_assistant_tokens hardCE(object) + mean_QA mean_assistant_tokens hardCE(replay)',
        'normalization':'independent actual object/replay QA means; last object block denominator8',
        'checkpoint_selection':'fixed Final@1 epoch; no dev selection',
        'numerical_path':'original_bf16','base_precision':'bf16','lora_parameter_precision':'fp32',
        'lora_initialization':'initial SFT adapter rounded to BF16 then exactly promoted to FP32',
        'autocast':'bf16','dropout':0.,'visual_token_budget':1024,'max_sequence_length':2048,
        'object_prompt_suffix':config['object_prompt_suffix'],'replay_prompt_suffix':config['replay_prompt_suffix'],
        'input_truncation':False,'replay_hard_ce_backward':True,'teacher_probability_cache_used':False,
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
        'teacher_generation_accounting_scope':'shared completed rollout reference, not new generation in this actor-training run',
        'equal_compute_claim':False,
        'helper_provenance':['frozen object load_actor/init/Adam/LR/checkpoint/RNG/runtime helpers',
                             'new data exact generated-ID encoder and loss equivalent causal FP32 logsoftmax CE']}
    lock_json(directory/'STATIC_INPUT_LOCK.json',static_lock)
    lock_json(directory/'SCHEDULE.json',schedule)
    loading_start = time.monotonic()
    model,parameters,dtypes = load_actor(config,reference,device)
    processor = processor_for(model_root,reference,config['visual_token_budget'])
    sync(device)
    loading_seconds = time.monotonic()-loading_start
    initial = {name:cpu_state(parameter) for name,parameter in model.named_parameters() if parameter.requires_grad}
    initial_sha = trainable_hash(model)
    if initial_sha != config['teacher_generation']['initial_runtime_master_sha256']:
        raise ValueError('actor initial FP32 master differs from fixed teacher initial master')
    frozen_before = frozen_hash(model)
    lock = {**static_lock,'initial_adapter_tensor_sha256':initial_sha,
            'frozen_parameter_sha256_before':frozen_before,'dtypes':dtypes,
            'trainable_parameters':sum(p.numel() for p in parameters),
            'trainable_names':{name:p.numel() for name,p in model.named_parameters() if p.requires_grad}}
    lock_json(directory/'TRAIN_INPUT_LOCK.json',lock)
    lineage_sha = content_sha(lock)
    preflight_started = time.monotonic()
    preflight_result = preflight(model,parameters,processor,objects,replays,blocks[0],config,args.condition,
                                directory,device,context,frozen_sha256=frozen_before)
    sync(device)
    preflight_seconds = time.monotonic()-preflight_started
    if args.preflight_only:
        print('PREFLIGHT_PASSED',args.condition,args.seed,flush=True)
        return preflight_result
    optimizer = torch.optim.AdamW(parameters,lr=config['lr'],weight_decay=config['weight_decay'])
    stats,completed = initial_stats(),0
    if args.resume:
        completed,stats = resume_checkpoint(directory,model,optimizer,lineage_sha,schedule_sha,len(blocks))
        validate_exposure_cursor(stats,blocks,completed,context)
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
                            stats,teaching,sources=(source,),update=step)
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
                'replay_count':len(blocks[step]['replay_indices']),'records':teaching})
        progress = {'status':'training','completed_updates':step+1,'planned_updates':len(blocks),
            'actual_object_exposures':stats['object_exposures'],'actual_replay_exposures':stats['replay_exposures'],
            'training_active_seconds':stats['training_seconds'],'gradient_l2':float(norm),'updated_at':now()}
        if step == 0 or (step+1)%25 == 0 or step+1 == len(blocks):
            print('OBJECT_TRAIN_UPDATE',json.dumps({'condition':args.condition,**progress}),flush=True)
        if (step+1)%25 == 0 or step+1 == len(blocks):
            validate_exposure_cursor(stats,blocks,step+1,context)
            save_checkpoint(directory,model,optimizer,step+1,stats,lineage_sha,schedule_sha)
            write_json(directory/'PROGRESS.json',progress)
    validate_exposure_cursor(stats,blocks,len(blocks),context)
    verify_prepared_files_unchanged(config,args.seed,args.condition,context)
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
        'mean_replay_ce_scope':'actual supervised hard CE; backpropagated with independent QA mean lambda1',
        'position_role_statistics':{source:summarize_roles(stats[f'{source}_role_statistics']) for source in ('object','replay')},
        'teacher_replies_manifest_file_sha256_after':sha_file(p['teacher']),
        'teacher_replies_lock_file_sha256_after':sha_file(p['teacher_lock']),
        'prepared_inputs_and_teacher_replies_unchanged':True,
        'training_active_seconds':stats['training_seconds'],'model_load_seconds':loading_seconds,
        'preflight_seconds_current_session':preflight_seconds,'preflight':preflight_result,
        'session_wall_seconds':time.monotonic()-session_started,
        'lora_delta_l2':math.sqrt(delta_sq),'lora_delta_max_abs':maximum,'changed_lora_tensors':changed,
        'final_lora_tensor_sha256':trainable_hash(model),'frozen_parameter_sha256_after':frozen_after,
        'frozen_parameters_unchanged':True,'resumed_at_update':completed,
        'peak_gpu_allocated_gib':torch.cuda.max_memory_allocated(device)/2**30,
        'peak_gpu_reserved_gib':torch.cuda.max_memory_reserved(device)/2**30,
        'adapter_files_sha256':file_hashes(directory/'adapter'),
        'compute_accounting':{'equal_compute_claim':False,'measured_flops':None,
            'teacher_generation_new_calls_in_training':0,'teacher_probability_forward_questions':0,
            'teacher_probability_file_bytes':0,'position_diagnostics_extra_model_calls':0,
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
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--preflight-only',action='store_true')
    group.add_argument('--resume',action='store_true')
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
