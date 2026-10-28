"""Vision-only matched CE / CE+DPO; reuse fixed data, losses and runtime."""
import argparse
import importlib
import json
import os
from pathlib import Path
import random
import sys
import time

from rarevlm_paths import PROJECT_ROOT
from vlm_deep.common import content_sha, sha_file, write_json
from vlm_grpo_iteration_v1.common import read
from vlm_native_scope_v1.model import ROOT as PARENT, load, PARAMETERS
from vlm_literal_preference_v1.common import make_batch

ROOT = PARENT.parent / 'vlm_visual_preference_v1'
PROTOCOL = PROJECT_ROOT / 'docs/VLM_VISUAL_PREFERENCE_V1_PROTOCOL.md'
ROLES = ('ce', 'ce_dpo')
INTERNAL = ('native_hard', 'literal_validation', 'seen_validation', 'retention')


def prepare():
    if (ROOT / 'STUDY_INPUT_LOCK.json').exists():
        raise FileExistsError('Study already frozen')
    diagnosis = read(PARENT.parent / 'vlm_native_scope_readout_v1/COMPLETION.json')
    if diagnosis['status'] != 'completed' or diagnosis['summaries']['visual']['residual_chosen_gt_current_wrong_sum'] >= 45:
        raise ValueError('No evidence of current wrong-answer likelihood headroom')
    for name in ('MANIFEST.json', 'CPU_INPUTS.json', 'SCHEDULE.json'):
        path = ROOT / 'data' / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes((PARENT / 'data' / name).read_bytes())
    config = {**read(PARENT / 'CONFIG.json'), 'output_root': str(ROOT), 'roles': list(ROLES),
              'algorithm': 'visual_CE_plus_reference_DPO', 'beta': .1, 'dpo_weight': {'ce': 0., 'ce_dpo': 1.},
              'automatic_main_evaluation': True, 'training_cap_seconds': 2400}
    write_json(ROOT / 'CONFIG.json', config)
    paths = [Path(__file__), PROTOCOL, ROOT / 'CONFIG.json', PARENT / 'STUDY_INPUT_LOCK.json',
             PARENT / 'COMPLETION.json', PARENT.parent / 'vlm_native_scope_readout_v1/COMPLETION.json']
    paths += [ROOT / 'data' / n for n in ('MANIFEST.json', 'CPU_INPUTS.json', 'SCHEDULE.json')]
    paths += [PROJECT_ROOT / 'src' / n for n in ('vlm_native_scope_v1/model.py', 'vlm_batch_stable_v1/model.py',
              'vlm_literal_preference_v1/common.py', 'vlm_alignment_signal/loss.py', 'research_runtime/queue.py')]
    paths += [PROJECT_ROOT / 'tools' / n for n in ('run_native_scope_v1.py', 'run_native_stable_kl_v1.py', 'run_native_correction_sft_dpo_v2.py')]
    paths += [Path(config['initial_adapter']) / n for n in ('adapter_config.json', 'adapter_model.safetensors')]
    reuse = {}
    for name in INTERNAL + ('transfer_dev',):
        source = PARENT / 'baseline' / name / 'PREDICTIONS.json'; destination = ROOT / 'baseline' / name / 'PREDICTIONS.json'
        destination.parent.mkdir(parents=True, exist_ok=True); destination.write_bytes(source.read_bytes()); paths += [source, destination]
        reuse[name] = {'source': str(source), 'sha256': sha_file(source), 'qa': len(read(source))}
    # Monitor-compatible baseline partitions are exact reused artifacts, not new inference.
    for part, names in [('a', ('native_hard', 'literal_validation', 'retention')), ('b', ('seen_validation',))]:
        for name in names:
            target = ROOT / ('baseline_' + part) / name / 'PREDICTIONS.json'; target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / 'baseline' / name / 'PREDICTIONS.json').read_bytes())
        write_json(ROOT / ('baseline_' + part) / 'COMPLETE.json', {'status': 'completed', 'cohorts': list(names), 'new_generations': 0, 'reused': True})
    write_json(ROOT / 'baseline_TRANSFER_COMPLETE.json', {'status': 'completed', 'cohorts': ['transfer_dev'], 'new_generations': 0, 'reused': True})
    write_json(ROOT / 'BASELINE_REUSE.json', {'status': 'completed', 'source': str(PARENT), 'cohorts': reuse, 'new_generations': 0, 'policy_unchanged': True})
    paths.append(ROOT / 'BASELINE_REUSE.json')
    (ROOT / 'DATA_RECEIPT.json').write_bytes((PARENT / 'DATA_RECEIPT.json').read_bytes())
    parent = read(PARENT / 'STUDY_INPUT_LOCK.json')
    write_json(ROOT / 'STUDY_INPUT_LOCK.json', {'status': 'frozen', 'files': {str(p): sha_file(p) for p in paths},
               'initial_lora_sha256': parent['initial_lora_sha256'], 'parent_input_lock_sha256': sha_file(PARENT / 'STUDY_INPUT_LOCK.json'),
               'roles': list(ROLES), 'new_updates': 64, 'chosen_qa_per_role': 512, 'negative_sequences_per_role': 256,
               'reference_sequences': 256, 'new_generations_phase1': 1920, 'conditional_generations': 3234,
               'optimizer_parameters_per_role': PARAMETERS, 'automatic_more_training': False})


def verify():
    lock = read(ROOT / 'STUDY_INPUT_LOCK.json')
    for path, digest in lock['files'].items():
        if sha_file(path) != digest:
            raise ValueError('Frozen input changed: ' + path)
    return lock


def evaluate(model, processor, role, names, phase):
    evaluator = importlib.import_module('run_native_correction_sft_dpo_v2'); evaluator.ROOT = ROOT
    evaluator.evaluate(model, processor, ROOT / 'runs' / role / phase, names, read(ROOT / 'data/MANIFEST.json'))
    write_json(ROOT / 'runs' / role / (phase + '_COMPLETE.json'), {'status': 'completed', 'role': role, 'cohorts': list(names)})


def reference():
    import torch
    from vlm_alignment_signal.loss import sequence_logps
    from vlm_human_sft.evaluate import lora_hash
    from vlm_sft.common import to_device
    lock = verify(); c = read(ROOT / 'CONFIG.json'); data = read(ROOT / 'data/MANIFEST.json'); cpu = read(ROOT / 'data/CPU_INPUTS.json')
    rows = {q['question_id']: q for q in data['rows']}; model, processor, _ = load(c['initial_adapter'], 'visual', False)
    model.eval(); model.requires_grad_(False); before = lora_hash(model)
    if before != lock['initial_lora_sha256']:
        raise ValueError('Reference identity mismatch')
    model.config.use_cache = False; memo = {}; batches = {}; started = time.monotonic()
    with torch.inference_mode():
        for step, update in enumerate(read(ROOT / 'data/SCHEDULE.json')['updates']):
            for micro, ids in enumerate(update['micros']):
                chunk = [rows[q] for q in ids]
                if chunk[0]['study_source'] != 'preference':
                    continue
                batch, binding, _, _ = make_batch(processor, data, chunk, cpu); key = content_sha(binding)
                if key not in memo:
                    logp, counts = sequence_logps(model, to_device(batch, next(model.parameters()).device))
                    if not bool(torch.isfinite(logp).all()):
                        raise FloatingPointError('Reference nonfinite')
                    memo[key] = {'binding': binding, 'chosen_logps': logp[:4].cpu().tolist(), 'rejected_logps': logp[4:].cpu().tolist(), 'counts': counts.cpu().tolist()}
                batches[f'{step+1}:{micro}'] = memo[key]
                write_json(ROOT / 'reference/PROGRESS.json', {'status': 'reference', 'completed_batches': len(batches), 'expected': 64, 'unique_forward_batches': len(memo), 'active_seconds': time.monotonic()-started})
                if time.monotonic()-started > 600 or torch.cuda.max_memory_allocated()/2**30 > 65:
                    raise RuntimeError('Reference budget')
    if len(memo) != 32 or len(batches) != 64 or lora_hash(model) != before:
        raise ValueError('Reference count/identity')
    write_json(ROOT / 'reference/CACHE.json', {'reference_lora_sha256': before, 'batches': batches})
    write_json(ROOT / 'reference/COMPLETE.json', {'status': 'completed', 'unique_forward_batches': 32, 'forward_sequences': 256, 'optimizer_updates': 0, 'active_seconds': time.monotonic()-started, 'cache_sha256': sha_file(ROOT / 'reference/CACHE.json')})


def train(role):
    import numpy as np
    import torch
    from safetensors.torch import load_file
    from vlm_alignment_signal.loss import sequence_logps, dpo_loss
    from vlm_human_sft.evaluate import lora_hash
    from vlm_object_train.train import learning_rate, lock_json, save_checkpoint, resume_checkpoint, tensor_hash
    from vlm_sft.common import to_device
    from run_native_scope_v1 import gradient_check, mode
    lock = verify(); c = read(ROOT / 'CONFIG.json'); data = read(ROOT / 'data/MANIFEST.json'); cpu = read(ROOT / 'data/CPU_INPUTS.json')
    rows = {q['question_id']: q for q in data['rows']}; updates = read(ROOT / 'data/SCHEDULE.json')['updates']; refdone = read(ROOT / 'reference/COMPLETE.json')
    if sha_file(ROOT / 'reference/CACHE.json') != refdone['cache_sha256']:
        raise ValueError('Reference changed')
    refs = read(ROOT / 'reference/CACHE.json')['batches']; directory = ROOT / 'runs' / role / 'train'
    closed = directory / 'TRAIN_COMPLETE.json'
    if closed.exists():
        result = read(closed); model, processor, _ = load(result['adapter'], 'visual', False); return model, processor, result
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    model, processor, binding = load(c['initial_adapter'], 'visual', True)
    if lora_hash(model) != lock['initial_lora_sha256']:
        raise ValueError('Initial model mismatch')
    frozen = {n:p for n,p in model.named_parameters() if 'lora_' in n and not p.requires_grad}; frozen_hash = tensor_hash(frozen.items())
    params = [p for p in model.parameters() if p.requires_grad]
    lineage = {'study_sha256': sha_file(ROOT/'STUDY_INPUT_LOCK.json'), 'schedule_sha256': sha_file(ROOT/'data/SCHEDULE.json'), 'reference_sha256': refdone['cache_sha256'], 'role': role, 'gpu_uuid': os.environ['CUDA_VISIBLE_DEVICES']}
    lineage_sha = content_sha(lineage); lock_json(directory/'INPUT_LOCK.json', lineage); optimizer = torch.optim.AdamW(params, lr=c['lr'], weight_decay=0.)
    stats = {'questions':0, 'chosen_ce_sum':0., 'loss_sum':0., 'preference_dpo_sum':0., 'negative_forward_sequences':0, 'source_counts':{}, 'active_seconds':0., 'trace_bytes':0, 'peak_allocated_gib':0., 'peak_reserved_gib':0.}; completed = 0
    if (directory/'LATEST.json').exists():
        completed,stats = resume_checkpoint(directory,model,optimizer,lineage_sha,lineage['schedule_sha256'],32)
    path = directory/'INPUT_TRACES.jsonl'; path.touch(exist_ok=True)
    if path.stat().st_size < stats['trace_bytes']:
        raise ValueError('Trace shorter than checkpoint')
    prior = stats['active_seconds']; begun = time.monotonic(); durations = []; torch.cuda.reset_peak_memory_stats()
    with path.open('r+b') as trace:
        trace.truncate(stats['trace_bytes']); trace.seek(0,os.SEEK_END)
        def checkpoint(step):
            stats['active_seconds'] = prior+time.monotonic()-begun; stats['trace_bytes'] = trace.tell(); trace.flush(); os.fsync(trace.fileno()); save_checkpoint(directory,model,optimizer,step,stats,lineage_sha,lineage['schedule_sha256'])
        for step in range(completed,32):
            begin = time.monotonic(); mode(model,'visual'); optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups:
                group['lr'] = learning_rate(step,32,c)
            ce_total = 0.; dpo_total = 0.; objective = 0.; margins = []
            for micro,ids in enumerate(updates[step]['micros']):
                chunk = [rows[q] for q in ids]; batch,key,chosen,rejected = make_batch(processor,data,chunk,cpu)
                logp,counts = sequence_logps(model,to_device(batch,next(model.parameters()).device)); ce = -logp[:4]/counts[:4]; dpo = torch.zeros_like(ce)
                if rejected:
                    ref = refs[f'{step+1}:{micro}']
                    if key != ref['binding'] or counts.cpu().tolist() != ref['counts']:
                        raise ValueError('Reference batch mismatch')
                    refc = torch.tensor(ref['chosen_logps'],device=logp.device,dtype=torch.float32); refr = torch.tensor(ref['rejected_logps'],device=logp.device,dtype=torch.float32)
                    relative = (logp[:4]-logp[4:])-(refc-refr)
                    if step == 0:
                        margin = float(relative.detach().abs().max()); margins.append(margin)
                        if margin > .05:
                            raise ValueError('Initial reference margin mismatch')
                    dpo = dpo_loss(logp[:4],logp[4:],refc,refr,.1); stats['negative_forward_sequences'] += 4
                losses = ce+c['dpo_weight'][role]*dpo
                if losses.dtype != torch.float32 or not bool(torch.isfinite(losses).all()):
                    raise FloatingPointError('Nonfinite objective')
                (losses.sum()/16).backward(); ces = ce.detach().cpu().tolist(); ds = dpo.detach().cpu().tolist(); ls = losses.detach().cpu().tolist()
                for i,(q,item,cv,dv,lv) in enumerate(zip(chunk,chosen,ces,ds,ls)):
                    entry = {'update':step+1,'microbatch':micro,'question_id':q['question_id'],'image_id':q['image_id'],'source':q['study_source'],'target':q['target'],'chosen_ce':cv,'dpo_loss':dv,'qa_objective':lv,'dpo_weight':c['dpo_weight'][role] if rejected else 0.,'input_ids':item['input_ids'][0].tolist(),'labels':item['labels'][0].tolist()}
                    if rejected:
                        entry.update(rejected_target=q['rejected_target'],rejected_input_ids=rejected[i]['input_ids'][0].tolist(),rejected_labels=rejected[i]['labels'][0].tolist(),reference_batch_key=f'{step+1}:{micro}',actual_padded_batch_binding=key)
                    trace.write((json.dumps(entry,ensure_ascii=False,allow_nan=False)+'\n').encode()); stats['source_counts'][q['study_source']] = stats['source_counts'].get(q['study_source'],0)+1
                ce_total += sum(ces); dpo_total += sum(ds); objective += sum(ls)
                del batch,logp,counts,ce,dpo,losses,chosen,rejected
            checks = gradient_check(model,'visual') if step == 0 else {}; norm = float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)); peak = torch.cuda.max_memory_allocated()/2**30
            if norm <= 0 or peak > 65:
                raise ValueError('Gradient/memory before optimizer')
            optimizer.step(); torch.cuda.synchronize(); stats['questions'] += 16; stats['chosen_ce_sum'] += ce_total; stats['preference_dpo_sum'] += dpo_total; stats['loss_sum'] += objective
            stats['active_seconds'] = prior+time.monotonic()-begun; stats['peak_allocated_gib'] = max(stats['peak_allocated_gib'],peak); stats['peak_reserved_gib'] = max(stats['peak_reserved_gib'],torch.cuda.max_memory_reserved()/2**30)
            if step == 0 or (step+1)%8 == 0:
                checkpoint(step+1)
            if step == 0:
                saved = load_file(str(directory/read(directory/'LATEST.json')['directory']/'adapter/adapter_model.safetensors')); expected = {n.replace('.default.','.'):v for n,v in model.named_parameters() if 'lora_' in n}
                if set(saved) != set(expected) or any(not torch.equal(saved[n],v.detach().cpu()) for n,v in expected.items()):
                    raise ValueError('Saved FP32 master differs')
            durations.append(time.monotonic()-begin); speed = sum(durations[-8:])/len(durations[-8:]); progress = {'status':'training','role':role,'completed_updates':step+1,'planned_updates':32,'mean_chosen_ce_this_update':ce_total/16,'mean_objective_this_update':objective/16,'mean_dpo_per_preference_qa':dpo_total/8,'gradient_norm':norm,'recent_seconds_per_update':speed,'remaining_training_seconds_estimate':speed*(31-step),**stats}; write_json(directory/'PROGRESS.json',progress)
            if step == 0:
                write_json(directory/'FIRST_UPDATE.json',{**progress,**checks,'binding':binding,'initial_reference_margin_max':max(margins),'saved_master_values_exact':True})
            if stats['active_seconds'] > 2400:
                checkpoint(step+1); raise RuntimeError('Training cap')
    if stats['questions'] != 512 or stats['negative_forward_sequences'] != 256 or stats['source_counts'] != {'common':256,'preference':256} or tensor_hash(frozen.items()) != frozen_hash:
        raise ValueError('Exposures/frozen factors changed')
    destination = directory/'adapter'; model.save_pretrained(destination,safe_serialization=True); result = {'status':'completed','role':role,'updates':32,'adapter':str(destination),'final_lora_sha256':lora_hash(model),'input_trace_sha256':sha_file(path),'frozen_factors_unchanged':True,'optimized_parameters':PARAMETERS,**stats}; write_json(closed,result); write_json(directory/'PROGRESS.json',result); return model,processor,result


def analyze(transfer=False):
    from vlm_grpo_iteration_v1.study import paired
    data = read(ROOT/'data/MANIFEST.json'); names = list(INTERNAL)+(['transfer_dev'] if transfer else [])
    rec = {'start':{n:read(ROOT/'baseline'/n/'PREDICTIONS.json') for n in names}}
    for role in ROLES:
        rec[role] = {n:read(ROOT/'runs'/role/('transfer' if n=='transfer_dev' else 'internal')/n/'PREDICTIONS.json') for n in names}
    for records in rec.values():
        for group in ('singleword','multiword','rare_1_5'):
            ids = {q['question_id'] for q in data['evaluations']['seen_validation'] if (q['stratum']==group if group!='rare_1_5' else 1<=q['answer_frequency']<=5)}; records[group] = [q for q in records['seen_validation'] if q['question_id'] in ids]
    def norm(rows,name):
        return [{**q,'image_id':q['filename_family'],'vqa_score':float(q['iiit_alnum_exact'])} for q in rows] if name=='literal_validation' else rows
    metrics = {r:{n:100*sum(q['vqa_score'] for q in norm(rows,n))/len(rows) for n,rows in records.items()} for r,records in rec.items()}
    vs = {r:{n:paired(norm(rec['start'][n],n),norm(rows,n)) for n,rows in rec[r].items()} for r in ROLES}; advantage = {n:paired(norm(rec['ce'][n],n),norm(rows,n)) for n,rows in rec['ce_dpo'].items()}
    checks = {}
    for r in ROLES:
        checks[r] = {'hard_learning':sum(q['vqa_score']>=1-1e-9 for q in rec[r]['native_hard'])>=32 and vs[r]['native_hard']['delta_pp']>=20,'ocr_point_protection':vs[r]['literal_validation']['delta_pp']>=-.5,'native_point_protections':all(vs[r][n]['delta_pp']>=-.5 for n in ('seen_validation','singleword','multiword','rare_1_5','retention'))}
    matched = all(advantage[n]['delta_pp']>=-.5 for n in ('literal_validation','seen_validation','rare_1_5'))
    expand = all(checks['ce_dpo'].values()) and matched
    result = {'status':'completed','metrics':metrics,'vs_start':vs,'screen_checks':checks,'screen_passed':{r:all(q.values()) for r,q in checks.items()},'candidate_vs_ce':advantage,'matched_control_point_protection':matched,'expand_transfer':expand,'goal_complete':False,'scope':'Single-seed internal matched vision CE/DPO pilot; no external or stable ability claim.'}
    if transfer:
        result['native_positive_signal'] = {'ce_dpo':expand and all(q['delta_pp']>=.5 and q['ci95'][0]>0 for q in (vs['ce_dpo']['transfer_dev'],advantage['transfer_dev']))}
    write_json(ROOT/('TRANSFER_ANALYSIS.json' if transfer else 'SCREEN_ANALYSIS.json'),result); return result


def worker(arg):
    verify(); kind,role = arg.split(':')
    if kind == 'reference':
        reference(); return
    if kind == 'internal':
        model,processor,_ = train(role); evaluate(model,processor,role,INTERNAL,'internal')
    else:
        final = read(ROOT/'runs'/role/'train/TRAIN_COMPLETE.json'); model,processor,_ = load(final['adapter'],'visual',False); evaluate(model,processor,role,('transfer_dev',),'transfer')


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare',action='store_true'); parser.add_argument('--worker'); args = parser.parse_args()
    if args.prepare:
        prepare(); return
    if args.worker:
        worker(args.worker); return
    from research_runtime.queue import run_queue
    verify(); identity = {'study_sha256':sha_file(ROOT/'STUDY_INPUT_LOCK.json')}; entry = str(Path(__file__).resolve())
    def job(kind,role,marker):
        return {'key':role,'command':[sys.executable,'-u',entry,'--worker',kind+':'+role],'completion_marker':str(marker)}
    run_queue(ROOT/'reference_queue',[job('reference','reference',ROOT/'reference/COMPLETE.json')],identity)
    run_queue(ROOT/'training_queue',[job('internal',r,ROOT/'runs'/r/'internal_COMPLETE.json') for r in ROLES],identity)
    result = analyze()
    if not result['expand_transfer']:
        write_json(ROOT/'COMPLETION.json',{**result,'phase':'screen_negative_no_expansion','new_updates':64,'new_generations':1920,'automatic_more_training':False}); return
    run_queue(ROOT/'transfer_queue',[job('transfer',r,ROOT/'runs'/r/'transfer_COMPLETE.json') for r in ROLES],identity); result = analyze(True)
    write_json(ROOT/'COMPLETION.json',{**result,'phase':'pilot_complete' if any(result['native_positive_signal'].values()) else 'transfer_negative_no_expansion','new_updates':64,'new_generations':5154,'automatic_more_training':False})


if __name__ == '__main__':
    main()
