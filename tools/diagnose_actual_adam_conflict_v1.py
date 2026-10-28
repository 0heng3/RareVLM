"""Measure saved updates, reconstruct first-step gradients, test one local repair.

One disposable GPU Adam step at the registered peak LR, then a projected copy
of that SAME displacement. No training run, adapter export, evaluation tuning
or confirmation. Original adapters are restored even after a diagnostic error.
"""

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time

PACKAGE = Path(__file__).resolve().parents[1]
OUTPUTS = PACKAGE.parents[1] / 'outputs'
PARENT = OUTPUTS / 'vlm_attention_scope_v1'
ROOT = OUTPUTS / 'vlm_actual_adam_conflict_v1'
PROBES = ['vqa2_train2014:444344000', 'vqa2_train2014:196971004',
          'vqa2_train2014:360575013', 'vqa2_train2014:178520002']
SATELLITE = 'vqa2_train2014:574443003'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    from vlm_deep.common import write_json
    write_json(Path(path), value)


def prepare():
    from prepare_semantic_admission_v1 import write_once
    from vlm_attention_scope_v1.study import verify as verify_parent
    verify_parent()
    if (ROOT / 'INPUT_LOCK.json').exists():
        verify(); return
    config = read(PACKAGE / 'configs/vlm_attention_scope_v1.json')
    manifest = read(config['manifest']); rows = {q['question_id']: q for q in manifest['splits']['train_all']}
    repaired = read(OUTPUTS / 'vlm_attention_scope_repair_v1/TRAINING_MANIFEST.json')
    admitted = {q['question_id'] for q in repaired['splits']['train_all']}
    if not set(PROBES) <= admitted or [rows[q]['target'] for q in PROBES] != ['no','no','yes','yes']:
        raise ValueError('All four visually supported probes must belong to the repaired training pool')
    micros = read(config['schedule_path'])['updates'][0]['micros']
    first_ids = [q for micro in micros for q in micro['question_ids']]
    selected = {q: rows[q] for q in list(dict.fromkeys(PROBES + first_ids))}
    states = {'initial': config['initial_adapter']}
    train = PARENT / 'runs/qv/train'
    for step in (1,25,200):
        options = list(train.glob(f'checkpoint_{step:06d}_*/adapter'))
        if len(options) != 1:
            raise ValueError('Need exactly one historical checkpoint for each fixed cursor')
        states['step'+str(step)] = str(options[0])
    cohort = {'qa': selected, 'images': {q['image_id']: manifest['images'][q['image_id']] for q in selected.values()},
              'probes': PROBES, 'first_update_micros': micros, 'states': states,
              'baseline_config': config['baseline_config'], 'historical_train_config': config,
              'first_optimizer_state': str(Path(states['step1']).parent/'state.pt'),
              'historical_first_update': str(train/'FIRST_UPDATE.json')}
    write_once(ROOT/'COHORT.json', cohort)
    protocol = {'scope': 'Train-only local causal diagnosis; not independent capability evidence',
        'historical_checkpoints': [0,1,25,200], 'probe_qa':4, 'first_update_qa':16,
        'diagnostic_optimizer_steps':1, 'temporary_counterfactual_states':2,
        'formal_training_updates':0, 'new_exported_adapters':0, 'confirmation_used':False,
        'local_control_lr':config['lr'], 'control_lr_note':'Registered peak LR, not historical first warmup step',
        'projection_constraints':['probe_no_alias_nll','probe_yes_alias_nll','text_mean_qa_ce'],
        'minimum_descent_cosine':.05, 'max_worker_seconds':300,
        'local_acceptance_only': {'no_nll_improvement_vs_control':.01,
            'no_nll_improvement_vs_initial':.01,'yes_nll_max_increase_vs_initial':.01,
            'text_ce_max_increase_vs_initial':.005},
        'automatic_training_promotion':False,
        'no_parameter_sweep_or_retry_scales':True}
    write_once(ROOT/'PROTOCOL.json', protocol)
    paths = [Path(__file__), ROOT/'COHORT.json', ROOT/'PROTOCOL.json',
             Path(cohort['first_optimizer_state']), Path(cohort['historical_first_update']),
             Path(config['manifest']), Path(config['schedule_path']),
             OUTPUTS/'vlm_attention_scope_repair_v1/TRAINING_MANIFEST.json',
             PACKAGE/'src/vlm_adam_conflict_v1/projection.py',
             PACKAGE/'tests/test_vlm_adam_conflict_v1.py',
             PACKAGE/'src/vlm_gradient_mechanism_v1/components.py',
             PACKAGE/'src/vlm_gradient_mechanism_v1/metrics.py',
             PACKAGE/'src/vlm_attention_scope_repair_v1/generation.py']
    for adapter in states.values():
        paths.extend(Path(adapter)/f for f in ('adapter_model.safetensors','adapter_config.json'))
    write_once(ROOT/'INPUT_LOCK.json', {'files':{str(p):sha(p) for p in paths},
        'historical_sources_immutable':True, 'new_optimizer_steps_discarded':1})
    print(json.dumps({'status':'prepared',**protocol},ensure_ascii=False),flush=True)


def verify():
    from vlm_attention_scope_v1.study import verify as verify_parent
    verify_parent()
    for path, expected in read(ROOT/'INPUT_LOCK.json')['files'].items():
        if sha(path) != expected:
            raise ValueError('Fixed diagnostic source changed: '+path)


def worker():
    import torch
    from peft.tuners.lora.layer import Linear
    from vlm_deep.common import collate, encode, to_device
    from vlm_runtime_metadata_v1.model import load
    from vlm_fp32_adapter.model import restore_fp32_adapter
    from vlm_attention_scope_v1.model import apply_scope
    from vlm_probe.evaluate import processor_for, inference_row
    from vlm_human_sft.evaluate import lora_hash, binary_answer
    from vlm_gradient_mechanism_v1.components import components_from_logits
    from vlm_gradient_mechanism_v1.metrics import capture_gradients
    from vlm_attention_scope_repair_v1.generation import raw_greedy_config
    from vlm_adam_conflict_v1.projection import project_displacement
    from run_fixed_error_readout_v1 import token_aliases

    verify(); started=time.monotonic(); p=read(ROOT/'PROTOCOL.json');c=read(ROOT/'COHORT.json')
    base=read(c['baseline_config']); torch.set_num_threads(2)
    torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model,binding=load(base['model_root'],c['states']['initial'],torch.device('cuda:0'),training=True,dropout=0.)
    apply_scope(model,'qv'); initial_hash=lora_hash(model)
    named=[(n,v) for n,v in model.named_parameters() if v.requires_grad];params=[v for _,v in named]
    before={n:v.detach().cpu().clone() for n,v in named}
    flat_before=torch.cat([v.double().reshape(-1) for v in before.values()])
    processor=processor_for(Path(base['model_root']),read(base['reference_config']),1024)
    aliases=token_aliases(processor.tokenizer)['groups']; device=torch.device('cuda:0')
    dtype_evidence={};hooks=[]
    first_layer=next(m for n,m in model.named_modules() if n.endswith('self_attn.q_proj') and hasattr(m,'lora_A'))
    def capture(tag):
        def hook(module,args,output):
            dtype_evidence.setdefault(tag,{'input':str(args[0].dtype),'output':str(output.dtype)})
        return hook
    for tag,module in [('lora_A',first_layer.lora_A['default']),('lora_B',first_layer.lora_B['default']),('projection_output',first_layer)]:
        hooks.append(module.register_forward_hook(capture(tag)))

    def batches(ids, supervised=True):
        return collate([encode(processor,c['qa'][q],Path(c['images'][c['qa'][q]['image_id']]['path']),supervised)
                        for q in ids],processor.tokenizer.pad_token_id)
    probe_cpu=batches(PROBES)
    micros=[(micro,batches(micro['question_ids'])) for micro in c['first_update_micros']]
    trace_path=PARENT/'runs/qv/train/INPUT_TRACES.jsonl'
    with trace_path.open() as stream:
        old_trace=[json.loads(next(stream)) for _ in range(16)]
    for (micro,batch),start in zip(micros,range(0,16,4)):
        for i,q in enumerate(micro['question_ids']):
            old=old_trace[start+i]
            if old['question_id'] != q or batch['input_ids'][i].tolist()!=old['input_ids'] or batch['labels'][i].tolist()!=old['labels']:
                raise ValueError('Exact historical first-update input/label batch mismatch')

    def budget():
        if time.monotonic()-started>p['max_worker_seconds']:
            raise TimeoutError('Bounded optimizer diagnosis exceeded budget')

    def forward(cpu):
        budget();batch=to_device(cpu,device);labels=batch['labels'];count=labels.ne(-100).sum(1)
        keep=int(count.max())+1
        handle=model.get_output_embeddings().register_forward_pre_hook(lambda m,a:(a[0][:,-keep:],))
        try:
            with torch.autocast('cuda',enabled=False):
                out=model(**{k:v for k,v in batch.items() if k!='labels'},use_cache=False)
        finally:
            handle.remove()
        result=components_from_logits(out.logits,labels)
        shifted=out.logits[:,:-1].float();first=result['first_prediction_positions_in_tail']
        first_logits=shifted[torch.arange(labels.shape[0],device=device),first]
        z=torch.logsumexp(first_logits,-1)
        for word in ('yes','no'):
            result[word+'_alias_nll']=z-torch.logsumexp(first_logits[:,aliases[word]['ids']],-1)
        result['canonical_margin']=first_logits[:,aliases['yes']['canonical']]-first_logits[:,aliases['no']['canonical']]
        return result

    gradients={};loss_evidence=[]
    def add_gradient(name,value,retain_graph):
        g=torch.autograd.grad(value,params,retain_graph=retain_graph)
        snap=capture_gradients(named,g)
        if snap.missing_gradients:raise ValueError('Disconnected trainable gradient')
        gradients[name]=gradients.get(name,torch.zeros_like(snap.values))+snap.values

    measurements={};diagnostic_steps=0
    try:
        model.train();model.get_base_model().visual.eval();model.config.use_cache=False
        for micro,cpu in micros:
            out=forward(cpu);ids=micro['question_ids']
            if micro['source']=='text':pieces={'text':out['qa_ce'].sum()/16}
            else:
                sat=[i for i,q in enumerate(ids) if q==SATELLITE]
                other=[i for i,q in enumerate(ids) if q!=SATELLITE]
                pieces={'other_human':out['qa_ce'][other].sum()/16}
                if sat:pieces['satellite']=out['qa_ce'][sat].sum()/16
            for i,(name,value) in enumerate(pieces.items()):
                add_gradient(name,value,i<len(pieces)-1)
            loss_evidence.extend({'question_id':q,'source':micro['source'],'qa_ce':float(v)}
                                 for q,v in zip(ids,out['qa_ce'].detach().cpu()))
            del out,pieces
        out=forward(probe_cpu)
        add_gradient('probe_no',out['no_alias_nll'][:2].mean(),True)
        add_gradient('probe_yes',out['yes_alias_nll'][2:].mean(),False)
        del out
        objective=gradients['text']+gradients['other_human']+gradients['satellite']
        state=torch.load(c['first_optimizer_state'],map_location='cpu',weights_only=False)
        if list(state['trainable_parameters'])!=[n for n,_ in named]:raise ValueError('Named optimizer order mismatch')
        group=state['optimizer']['param_groups'][0]; beta1=group['betas'][0]
        applied=torch.cat([(state['optimizer']['state'][pid]['exp_avg'].double()/(1-beta1)).reshape(-1)
                           for pid in group['params']])
        coefficient=min(1.,1./(float(objective.norm())+1e-6))
        relative_error=float((objective*coefficient-applied).norm()/applied.norm())
        if relative_error>1e-4:raise ValueError('Recomputed clipped gradient does not match actual saved Adam moments')
        actual_first=torch.cat([state['trainable_parameters'][n].double().reshape(-1) for n,_ in named])-flat_before
        coordinate=[{'name':n,'shape':list(v.shape),'numel':v.numel()} for n,v in named]
        torch.save({'coordinates':coordinate,'gradients':gradients,'actual_first_displacement':actual_first},ROOT/'NAMED_GRADIENTS.pt')
        gradient_report={'relative_l2_error_vs_saved_first_moment':relative_error,
            'raw_norm':float(objective.norm()),'historical_logged_norm':read(c['historical_first_update'])['gradient_norm'],
            'historical_first_delta_local_predictions':{name:float(g@actual_first) for name,g in gradients.items()},
            'source_dot_probe_no':{name:float(gradients[name]@gradients['probe_no']) for name in ('text','satellite','other_human')},
            'scope':'Source dots are raw SGD geometry; actual-delta projections are local first-order predictions only'}
        for h in hooks:h.remove()

        def measure(label):
            model.eval();model.config.use_cache=False
            with torch.no_grad():
                out=forward(probe_cpu)
                v={'probe_no_alias_nll':float(out['no_alias_nll'][:2].mean()),
                   'probe_yes_alias_nll':float(out['yes_alias_nll'][2:].mean()),
                   'probe_canonical_margins':out['canonical_margin'].cpu().tolist(),
                   'probe_canonical_first_nll':out['first_answer_ce'].cpu().tolist()}
                scores={'text':[],'human':[]}
                for micro,cpu in micros:scores[micro['source']].extend(forward(cpu)['qa_ce'].cpu().tolist())
                v.update({source+'_mean_qa_ce':sum(values)/len(values) for source,values in scores.items()})
                prompt_cpu=collate([encode(processor,inference_row(c['qa'][q]),Path(c['images'][c['qa'][q]['image_id']]['path']),False)
                                    for q in PROBES],processor.tokenizer.pad_token_id)
                prompt=to_device(prompt_cpu,device)
                generated=model.generate(**prompt,generation_config=raw_greedy_config(model.generation_config,1))
                texts=[processor.tokenizer.decode(row[prompt['input_ids'].shape[1]:],skip_special_tokens=True).strip() for row in generated]
                v['generated_first_tokens']=texts
                v['generated_first_binary_correct']=[binary_answer(t)==c['qa'][q]['target'] for t,q in zip(texts,PROBES)]
            measurements[label]=v
            write(ROOT/'PROGRESS.json',{'status':'measuring_actual_updates','state':label,
                'elapsed_seconds':time.monotonic()-started,'diagnostic_optimizer_steps':diagnostic_steps,
                'formal_training_updates':0})
            return v

        for label,adapter in c['states'].items():
            restore_fp32_adapter(model,adapter)
            if label=='step1' and not all(torch.equal(v.detach().cpu(),state['trainable_parameters'][n]) for n,v in named):
                raise ValueError('Saved step1 FP32 masters differ from restored adapter')
            measure(label)
        # ONE disposable real GPU Adam step, fixed peak LR. This is explicitly
        # not replaying the historical first warmup step at another LR.
        restore_fp32_adapter(model,c['states']['initial'])
        optimizer=torch.optim.AdamW(params,lr=p['local_control_lr'],betas=tuple(group['betas']),
                                    eps=group['eps'],weight_decay=0.)
        offset=0
        for _,v in named:
            v.grad=objective[offset:offset+v.numel()].reshape(v.shape).to(device=device,dtype=v.dtype).clone();offset+=v.numel()
        torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
        optimizer.step();diagnostic_steps+=1;optimizer.zero_grad(set_to_none=True)
        raw_delta=torch.cat([v.detach().cpu().double().reshape(-1) for _,v in named])-flat_before
        measure('local_peak_raw_adam')
        constraints={'probe_no_alias_nll':gradients['probe_no'],
                     'probe_yes_alias_nll':gradients['probe_yes'],
                     'text_mean_qa_ce':gradients['text']*2}
        candidate,projection=project_displacement(raw_delta,constraints,p['minimum_descent_cosine'])
        offset=0
        with torch.no_grad():
            for n,v in named:
                increment=candidate[offset:offset+v.numel()].reshape(v.shape)
                v.copy_((before[n].double()+increment).float().to(device));offset+=v.numel()
        projected_actual=torch.cat([v.detach().cpu().double().reshape(-1) for _,v in named])-flat_before
        projection['realized_fp32_projections']={name:float(g@projected_actual) for name,g in constraints.items()}
        measure('local_peak_projected_adam')
        gate=p['local_acceptance_only'];a=measurements['initial'];b=measurements['local_peak_raw_adam'];v=measurements['local_peak_projected_adam']
        gates={'no_vs_raw':v['probe_no_alias_nll']<=b['probe_no_alias_nll']-gate['no_nll_improvement_vs_control'],
               'no_vs_initial':v['probe_no_alias_nll']<=a['probe_no_alias_nll']-gate['no_nll_improvement_vs_initial'],
               'yes_kept':v['probe_yes_alias_nll']<=a['probe_yes_alias_nll']+gate['yes_nll_max_increase_vs_initial'],
               'text_kept':v['text_mean_qa_ce']<=a['text_mean_qa_ce']+gate['text_ce_max_increase_vs_initial']}
        result={'status':'completed','measurements':measurements,'gradient_reconstruction':gradient_report,
            'first_update_qa_losses':loss_evidence,'actual_dtype_evidence':dtype_evidence,
            'installed_peft_linear_forward_sha256':hashlib.sha256(inspect.getsource(Linear.forward).encode()).hexdigest(),
            'projection':projection,'local_gates':gates,'local_candidate_passed':all(gates.values()),
            'diagnostic_optimizer_steps':diagnostic_steps,'formal_training_updates':0,
            'new_exported_adapters':0,'new_capability_gain_claim':False,
            'worker_seconds':time.monotonic()-started,'automatic_training_promotion':False}
    finally:
        for h in hooks:h.remove()
        restore_fp32_adapter(model,c['states']['initial'])
        if lora_hash(model)!=initial_hash:raise ValueError('Disposable diagnostic failed to restore initial weights')
    verify();write(ROOT/'ANALYSIS.json',result)
    write(ROOT/'WORKER_COMPLETE.json',{'status':'completed','initial_weights_restored':True,
        'discarded_diagnostic_optimizer_steps':diagnostic_steps,'formal_training_updates':0})
    print(json.dumps(result,ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--worker',action='store_true')
    args=parser.parse_args()
    if args.prepare:prepare();return
    if args.worker:worker();return
    verify()
    from research_runtime.queue import run_queue
    run_queue(ROOT/'queue',[{'key':'actual_adam','command':[sys.executable,'-u',str(Path(__file__).resolve()),'--worker'],
        'completion_marker':str(ROOT/'WORKER_COMPLETE.json')}],{'input_lock_sha256':sha(ROOT/'INPUT_LOCK.json'),
        'formal_training_updates':0,'discarded_diagnostic_optimizer_steps':1})
    write(ROOT/'COMPLETION.json',{'status':'completed','formal_training_updates':0,
        'discarded_diagnostic_optimizer_steps':1,'new_capability_gain_claim':False,
        'automatic_training_started':False,'local_candidate_passed':read(ROOT/'ANALYSIS.json')['local_candidate_passed']})


if __name__=='__main__':main()
