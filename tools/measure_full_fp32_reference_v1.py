"""Small full-FP32 reference to isolate BF16 finite-step linearization effects.

All original BF16 base values are exactly promoted, not recovered from another
pretrained model. Four training probes, original states 0/1/200, two scalar
initial backward passes, zero optimizer updates and no exported model.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

PACKAGE=Path(__file__).resolve().parents[1]
OUTPUTS=PACKAGE.parents[1]/'outputs'
PARENT=OUTPUTS/'vlm_actual_adam_conflict_v1'
ROOT=OUTPUTS/'vlm_full_fp32_reference_v1'


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    from vlm_deep.common import write_json
    write_json(Path(path),value)


def prepare():
    from prepare_semantic_admission_v1 import write_once
    from vlm_attention_scope_v1.study import verify as verify_parent
    verify_parent()
    if (ROOT/'INPUT_LOCK.json').exists():verify();return
    c=read(OUTPUTS/'vlm_fp32_readout_diagnostic_v1/COHORT.json')
    c['states']={k:v for k,v in c['states'].items() if k in ('initial','step1','step200')}
    write_once(ROOT/'COHORT.json',c)
    write_once(ROOT/'PROTOCOL.json',{'scope':'Train-only numerical reference, not performance promotion',
        'qa':4,'states':[0,1,200],'all_floating_weights_and_operations':'float32, TF32 disabled',
        'base_values':'Exact FP32 promotion of the same stored BF16 base, no recovered pretrained tails',
        'scalar_backward_passes':2,'optimizer_steps':0,'max_worker_seconds':300,
        'automatic_training_promotion':False,'no_hyperparameter_or_scale_retry':True})
    paths=[Path(__file__),ROOT/'COHORT.json',ROOT/'PROTOCOL.json',PARENT/'NAMED_GRADIENTS.pt',
        PACKAGE/'src/vlm_fp32_adapter/model.py',PACKAGE/'src/vlm_gradient_mechanism_v1/metrics.py']
    for adapter in c['states'].values():paths.extend(Path(adapter)/n for n in ('adapter_model.safetensors','adapter_config.json'))
    write_once(ROOT/'INPUT_LOCK.json',{'files':{str(p):sha(p) for p in paths},'optimizer_steps':0})


def verify():
    from vlm_attention_scope_v1.study import verify as verify_parent
    verify_parent()
    for path,expected in read(ROOT/'INPUT_LOCK.json')['files'].items():
        if sha(path)!=expected:raise ValueError('Fixed full-FP32 reference source changed: '+path)


def worker():
    import torch
    from vlm_deep.common import collate,encode,to_device
    from vlm_runtime_metadata_v1.model import load
    from vlm_fp32_adapter.model import restore_fp32_adapter
    from vlm_attention_scope_v1.model import apply_scope
    from vlm_gradient_mechanism_v1.metrics import capture_gradients
    from vlm_probe.evaluate import processor_for
    from vlm_human_sft.evaluate import lora_hash
    from run_fixed_error_readout_v1 import token_aliases
    verify();started=time.monotonic();c=read(ROOT/'COHORT.json');p=read(ROOT/'PROTOCOL.json')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base=read(c['baseline_config']);device=torch.device('cuda:0')
    model,_=load(base['model_root'],c['states']['initial'],device,training=True,dropout=0.)
    apply_scope(model,'qv');original_hash=lora_hash(model);model.float()
    if any(v.dtype!=torch.float32 for v in model.parameters() if v.is_floating_point()):
        raise ValueError('Full-FP32 reference still has non-FP32 model parameters')
    named=[(n,v) for n,v in model.named_parameters() if v.requires_grad]
    processor=processor_for(Path(base['model_root']),read(base['reference_config']),1024)
    alias=token_aliases(processor.tokenizer)['groups']
    rows=[c['qa'][q] for q in c['probes']]
    batch=to_device(collate([encode(processor,q,Path(c['images'][q['image_id']]['path']),True) for q in rows],processor.tokenizer.pad_token_id),device)
    if not bool(batch['labels'].ne(-100).sum(1).eq(3).all()):raise ValueError('Reference target length differs')
    inputs={k:v for k,v in batch.items() if k!='labels'}
    def forward():
        if time.monotonic()-started>p['max_worker_seconds']:raise TimeoutError('Full-FP32 reference exceeded budget')
        head=model.get_output_embeddings();handle=head.register_forward_pre_hook(lambda m,a:(a[0][:,-4:],))
        try:
            with torch.autocast('cuda',enabled=False):logits=model(**inputs,use_cache=False).logits[:,0]
        finally:handle.remove()
        if logits.dtype!=torch.float32:raise ValueError('Reference logits are not FP32')
        z=torch.logsumexp(logits,-1)
        return logits,{word:z-torch.logsumexp(logits[:,alias[word]['ids']],-1) for word in ('yes','no')}
    def summarize(logits,losses):
        return {'probe_no_alias_nll':float(losses['no'][:2].mean()),
            'probe_yes_alias_nll':float(losses['yes'][2:].mean()),
            'canonical_margins':(logits[:,alias['yes']['canonical']]-logits[:,alias['no']['canonical']]).detach().cpu().tolist(),
            'raw_argmax_tokens':[processor.tokenizer.decode([int(i)]) for i in logits.argmax(-1)]}
    parent=torch.load(PARENT/'NAMED_GRADIENTS.pt',map_location='cpu',weights_only=True)
    if [q['name'] for q in parent['coordinates']]!=[n for n,_ in named]:raise ValueError('Reference gradient coordinates differ')
    gradients={};predictions={};values={}
    try:
        model.train();model.get_base_model().visual.eval();model.config.use_cache=False
        logits,losses=forward();values['initial']=summarize(logits,losses)
        for i,(name,value) in enumerate([('probe_no',losses['no'][:2].mean()),('probe_yes',losses['yes'][2:].mean())]):
            grads=torch.autograd.grad(value,[v for _,v in named],retain_graph=i==0)
            snap=capture_gradients(named,grads);gradients[name]=snap.values
            predictions[name]=float(snap.values@parent['actual_first_displacement'])
        del logits,losses
        for state in ('step1','step200'):
            restore_fp32_adapter(model,c['states'][state])
            with torch.no_grad():
                logits,losses=forward();values[state]=summarize(logits,losses)
            write(ROOT/'PROGRESS.json',{'status':'measuring_full_fp32_reference','checkpoint':state,
                'elapsed_seconds':time.monotonic()-started,'optimizer_steps':0})
        comparison={word:{'first_order_prediction':predictions['probe_'+word],
            'actual_first_step_change':values['step1']['probe_'+word+'_alias_nll']-values['initial']['probe_'+word+'_alias_nll'],
            'actual_final_change':values['step200']['probe_'+word+'_alias_nll']-values['initial']['probe_'+word+'_alias_nll']}
            for word in ('no','yes')}
        torch.save({'coordinates':parent['coordinates'],'gradients':gradients},ROOT/'NAMED_GRADIENTS.pt')
        result={'status':'completed','measurements':values,'first_step_comparison':comparison,
            'worker_seconds':time.monotonic()-started,'optimizer_steps':0,'formal_training_updates':0,
            'new_capability_gain_claim':False,'automatic_training_promotion':False,
            'scope':'Full-FP32 reference does not replace historical BF16 evaluation or prove full-training causality'}
    finally:
        restore_fp32_adapter(model,c['states']['initial'])
        if lora_hash(model)!=original_hash:raise ValueError('Reference failed to restore original adapter')
    verify();write(ROOT/'ANALYSIS.json',result)
    write(ROOT/'WORKER_COMPLETE.json',{'status':'completed','optimizer_steps':0,'adapter_restored':True})
    print(json.dumps(result,ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare',action='store_true');parser.add_argument('--worker',action='store_true')
    a=parser.parse_args()
    if a.prepare:prepare();return
    if a.worker:worker();return
    verify()
    from research_runtime.queue import run_queue
    run_queue(ROOT/'queue',[{'key':'full_fp32_reference','command':[sys.executable,'-u',str(Path(__file__).resolve()),'--worker'],
        'completion_marker':str(ROOT/'WORKER_COMPLETE.json')}],{'input_lock_sha256':sha(ROOT/'INPUT_LOCK.json'),'optimizer_steps':0})
    write(ROOT/'COMPLETION.json',{'status':'completed','optimizer_steps':0,'new_capability_gain_claim':False,'automatic_training_started':False})


if __name__=='__main__':main()
