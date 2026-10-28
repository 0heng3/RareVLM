import time
import torch
from vlm_deep.common import write_json, sha_file
from vlm_fp32_adapter.model import restore_fp32_adapter
from vlm_human_sft.evaluate import lora_hash
from vlm_object_train.train import learning_rate
from .common import ROOT, PARENT, actor, read, gradients, flat, project


def run():
    plan = read(ROOT/'PLAN.json'); old = read(PARENT/'PLAN.json')
    data = read(PARENT/'DATA.json'); cpu = read(PARENT/'CPU_INPUTS.json'); schedule = read(PARENT/'SCHEDULE.json')
    model, processor, _ = actor(training=True)
    parameters = [p for p in model.parameters() if p.requires_grad]
    points = []; begin = time.monotonic(); discarded = 0
    for checkpoint in old['calibration_checkpoints']:
        restore_fp32_adapter(model, checkpoint['adapter'])
        if lora_hash(model) != checkpoint['trainable_sha256']:
            raise ValueError('Historical calibration checkpoint mismatch')
        optimizer = torch.optim.AdamW(parameters, lr=2e-4, weight_decay=0.)
        state = torch.load(checkpoint['state_path'], map_location='cpu', weights_only=False)
        if int(state['completed_updates']) != checkpoint['step']:
            raise ValueError('Historical optimizer cursor mismatch')
        if list(state['trainable_parameters']) != [n for n,p in model.named_parameters() if p.requires_grad]:
            raise ValueError('Optimizer parameter coordinate order differs')
        optimizer.load_state_dict(state['optimizer']); optimizer.zero_grad(set_to_none=True)
        step = checkpoint['step']
        for group in optimizer.param_groups:
            group['lr'] = learning_rate(step, 64, {'lr': 2e-4, 'warmup_ratio': .125})
        _, vectors, totals, _ = gradients(model, processor, schedule[step], data, cpu)
        before = flat(parameters).clone()
        torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        optimizer.step(); discarded += 1
        delta = flat(parameters)-before
        projected, detail = project(delta, vectors)
        cosines = {s: float(torch.dot(g.double(), delta.double()) / (g.double().norm()*delta.double().norm())) for s,g in vectors.items()}
        conflict = cosines['replay'] > .01 and cosines['hard'] < 0.
        points.append({'checkpoint_step': step, 'next_update': step+1, 'ce_qa_mean': {k:v/8 for k,v in totals.items()},
                       'displacement_cosines': cosines, 'conflict_for_admission': conflict,
                       'source_gradient_cosine': float(torch.nn.functional.cosine_similarity(vectors['hard'].double(), vectors['replay'].double(), dim=0)),
                       'correction_fraction': detail['correction_norm']/detail['proposed_norm'], **detail})
        write_json(ROOT/'DIAGNOSTIC_PROGRESS.json', {'status':'diagnosing', 'points':points, 'discarded_optimizer_updates':discarded})
        del optimizer, state, before, delta, projected, vectors
        if torch.cuda.max_memory_allocated()/2**30 > 65:
            raise RuntimeError('65GiB diagnostic allocation budget')
        if time.monotonic()-begin > 180:
            raise TimeoutError('180sec diagnostic active budget')
    restore_fp32_adapter(model, old['initial_adapter'])
    if lora_hash(model) != old['initial_lora_sha256']:
        raise ValueError('Original SFT not restored')
    result = {'status':'completed', 'plan_sha256':sha_file(ROOT/'PLAN.json'), 'points':points,
              'training_allowed':sum(p['conflict_for_admission'] for p in points)>=2,
              'rule':'At least two of three training-only checkpoints have replay displacement cosine > .01 and hard cosine < 0.',
              'new_formal_optimizer_updates':0, 'discarded_diagnostic_optimizer_updates':discarded,
              'initial_restored_sha256':lora_hash(model), 'active_seconds':time.monotonic()-begin,
              'validation_used':False, 'scope':'local directional mechanism; no independent capability conclusion'}
    write_json(ROOT/'DIAGNOSTIC.json',result)
    return result
