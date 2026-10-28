"""Isolate output-head precision using identical saved-checkpoint hidden states.

Four training probes, checkpoints 0/1/25/200, BF16/FP32 heads, dense/tail
projection. Two initial-point scalar backward passes, zero optimizer steps.
No model/adapter export, parameter sweep or heldout/confirmation inputs.
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
ROOT=OUTPUTS/'vlm_fp32_readout_diagnostic_v1'


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
    cohort=read(PARENT/'COHORT.json')
    cohort['qa']={q:cohort['qa'][q] for q in cohort['probes']}
    cohort['images']={q['image_id']:cohort['images'][q['image_id']] for q in cohort['qa'].values()}
    write_once(ROOT/'COHORT.json',cohort)
    protocol={'scope':'Train-only head-precision isolation; not model capability evidence',
        'checkpoints':[0,1,25,200],'qa_per_checkpoint':4,
        'head_variants':['bf16_tail','bf16_dense','fp32_tail','fp32_dense'],
        'controlled_hidden_states':'Identical BF16 decoder hidden tensors per checkpoint',
        'gradient_variants':'Initial FP32 head no/yes alias NLL, named q/v coordinates',
        'scalar_backward_passes':2,'optimizer_steps':0,'formal_training_updates':0,
        'max_worker_seconds':180,'automatic_training_promotion':False,
        'question':'Does BF16 output-head projection/slicing explain the first-step derivative/finite-change discrepancy?',
        'limitations':'Decoder remains BF16; FP32 head alone cannot guarantee smooth loss or capability gains'}
    write_once(ROOT/'PROTOCOL.json',protocol)
    paths=[Path(__file__),ROOT/'COHORT.json',ROOT/'PROTOCOL.json',PARENT/'ANALYSIS.json',PARENT/'NAMED_GRADIENTS.pt',
           PACKAGE/'src/vlm_fp32_readout_v1/model.py',PACKAGE/'tests/test_vlm_fp32_readout_v1.py',
           PACKAGE/'src/vlm_gradient_mechanism_v1/metrics.py']
    for adapter in cohort['states'].values():paths.extend(Path(adapter)/n for n in ('adapter_model.safetensors','adapter_config.json'))
    write_once(ROOT/'INPUT_LOCK.json',{'files':{str(p):sha(p) for p in paths},'optimizer_steps':0})
    print(json.dumps(protocol,ensure_ascii=False),flush=True)


def verify():
    from vlm_attention_scope_v1.study import verify as verify_parent
    verify_parent()
    for path,expected in read(ROOT/'INPUT_LOCK.json')['files'].items():
        if sha(path)!=expected:raise ValueError('Frozen readout diagnostic source changed: '+path)


def worker():
    import torch
    import torch.nn.functional as F
    from vlm_deep.common import collate,encode,to_device
    from vlm_runtime_metadata_v1.model import load
    from vlm_fp32_adapter.model import restore_fp32_adapter
    from vlm_attention_scope_v1.model import apply_scope
    from vlm_fp32_readout_v1.model import install_fp32_readout
    from vlm_gradient_mechanism_v1.metrics import capture_gradients
    from vlm_probe.evaluate import processor_for
    from vlm_human_sft.evaluate import lora_hash
    from run_fixed_error_readout_v1 import token_aliases

    verify();started=time.monotonic();c=read(ROOT/'COHORT.json');protocol=read(ROOT/'PROTOCOL.json')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base=read(c['baseline_config']);device=torch.device('cuda:0')
    model,_=load(base['model_root'],c['states']['initial'],device,training=True,dropout=0.)
    apply_scope(model,'qv');original_hash=lora_hash(model);old_head=model.get_output_embeddings()
    embedding=model.get_input_embeddings();embedding_ptr=embedding.weight.data_ptr()
    processor=processor_for(Path(base['model_root']),read(base['reference_config']),1024)
    aliases=token_aliases(processor.tokenizer)['groups']
    rows=[c['qa'][q] for q in c['probes']]
    batch=to_device(collate([encode(processor,q,Path(c['images'][q['image_id']]['path']),True) for q in rows],processor.tokenizer.pad_token_id),device)
    if not bool(batch['labels'].ne(-100).sum(1).eq(3).all()):raise ValueError('Probe target lengths changed')
    inputs={k:v for k,v in batch.items() if k!='labels'}
    named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    def metrics(first):
        first=first.float();z=torch.logsumexp(first,-1)
        no=z-torch.logsumexp(first[:,aliases['no']['ids']],-1)
        yes=z-torch.logsumexp(first[:,aliases['yes']['ids']],-1)
        return {'probe_no_alias_nll':float(no[:2].mean()),'probe_yes_alias_nll':float(yes[2:].mean()),
            'canonical_margins':(first[:,aliases['yes']['canonical']]-first[:,aliases['no']['canonical']]).cpu().tolist(),
            'raw_argmax_tokens':[processor.tokenizer.decode([int(i)]) for i in first.argmax(-1)],
            'canonical_yes_logits':first[:,aliases['yes']['canonical']].cpu().tolist(),
            'canonical_no_logits':first[:,aliases['no']['canonical']].cpu().tolist()}
    measurements={};saved_logits={};weight32=old_head.weight.detach().float()
    try:
        model.eval();model.config.use_cache=False
        for state,adapter in c['states'].items():
            if time.monotonic()-started>protocol['max_worker_seconds']:raise TimeoutError('Bounded readout diagnosis exceeded budget')
            restore_fp32_adapter(model,adapter);captured=[]
            def tail(module,args):
                captured.append(args[0].detach())
                return (args[0][:,-4:],)
            handle=old_head.register_forward_pre_hook(tail)
            try:
                with torch.no_grad(),torch.autocast('cuda',enabled=False):out=model(**inputs,use_cache=False)
            finally:handle.remove()
            hidden=captured[0]
            measurements[state]={};saved_logits[state]={}
            with torch.no_grad(),torch.autocast('cuda',enabled=False):
                first=out.logits[:,0].float()
                measurements[state]['bf16_tail']=metrics(first);saved_logits[state]['bf16_tail']=first.cpu()
                del out,first
                for variant in ('bf16_dense','fp32_tail','fp32_dense'):
                    dense=variant.endswith('dense');full=hidden if dense else hidden[:,-4:]
                    if variant.startswith('fp32'):logits=F.linear(full.float(),weight32)
                    else:logits=F.linear(full,old_head.weight)
                    first=logits[:,-4 if dense else 0].float()
                    measurements[state][variant]=metrics(first);saved_logits[state][variant]=first.cpu()
                    del logits,first
            del captured,hidden
            write(ROOT/'PROGRESS.json',{'status':'measuring_head_variants','checkpoint':state,'optimizer_steps':0,'elapsed_seconds':time.monotonic()-started})
        del weight32
        restore_fp32_adapter(model,c['states']['initial']);binding=install_fp32_readout(model)
        model.train();model.get_base_model().visual.eval();model.config.use_cache=False
        handle=model.get_output_embeddings().register_forward_pre_hook(lambda m,a:(a[0][:,-4:],))
        try:
            with torch.autocast('cuda',enabled=False):out=model(**inputs,use_cache=False)
        finally:handle.remove()
        logits=out.logits[:,0];z=torch.logsumexp(logits,-1)
        losses={'probe_no':(z-torch.logsumexp(logits[:,aliases['no']['ids']],-1))[:2].mean(),
                'probe_yes':(z-torch.logsumexp(logits[:,aliases['yes']['ids']],-1))[2:].mean()}
        parent_vectors=torch.load(PARENT/'NAMED_GRADIENTS.pt',map_location='cpu',weights_only=True)
        if [q['name'] for q in parent_vectors['coordinates']] != [n for n,_ in named]:raise ValueError('Gradient coordinates differ')
        projections={};new_vectors={}
        for index,(name,loss) in enumerate(losses.items()):
            grad=torch.autograd.grad(loss,[v for _,v in named],retain_graph=index==0)
            snapshot=capture_gradients(named,grad);new_vectors[name]=snapshot.values
            projections[name]=float(snapshot.values@parent_vectors['actual_first_displacement'])
        torch.save({'coordinates':parent_vectors['coordinates'],'gradients':new_vectors},ROOT/'FP32_HEAD_NAMED_GRADIENTS.pt')
        torch.save(saved_logits,ROOT/'FIRST_POSITION_LOGITS.pt')
        parent=read(PARENT/'ANALYSIS.json');comparison={}
        for group in ('no','yes'):
            field='probe_'+group+'_alias_nll';key='probe_'+group
            old_prediction=parent['gradient_reconstruction']['historical_first_delta_local_predictions'][key]
            old_actual=measurements['step1']['bf16_tail'][field]-measurements['initial']['bf16_tail'][field]
            new_actual=measurements['step1']['fp32_tail'][field]-measurements['initial']['fp32_tail'][field]
            comparison[group]={'bf16_first_order_prediction':old_prediction,'bf16_actual_change':old_actual,
                'fp32_head_first_order_prediction':projections[key],'fp32_head_actual_change':new_actual,
                'bf16_linearization_residual':abs(old_actual-old_prediction),
                'fp32_head_linearization_residual':abs(new_actual-projections[key])}
        result={'status':'completed','measurements':measurements,'first_step_comparison':comparison,
            'readout_policy_binding':binding,'optimizer_steps':0,'formal_training_updates':0,
            'new_capability_gain_claim':False,'worker_seconds':time.monotonic()-started,
            'scope':'Identical BF16 hidden states isolate head arithmetic; decoder numerical effects remain possible'}
    finally:
        restore_fp32_adapter(model,c['states']['initial']);model.set_output_embeddings(old_head)
        if lora_hash(model)!=original_hash or embedding.weight.data_ptr()!=embedding_ptr:
            raise ValueError('Read-only head diagnosis failed to restore original model')
    verify();write(ROOT/'ANALYSIS.json',result)
    write(ROOT/'WORKER_COMPLETE.json',{'status':'completed','original_head_and_adapter_restored':True,'optimizer_steps':0})
    print(json.dumps(result,ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare',action='store_true');parser.add_argument('--worker',action='store_true')
    args=parser.parse_args()
    if args.prepare:prepare();return
    if args.worker:worker();return
    verify()
    from research_runtime.queue import run_queue
    run_queue(ROOT/'queue',[{'key':'head_precision','command':[sys.executable,'-u',str(Path(__file__).resolve()),'--worker'],
        'completion_marker':str(ROOT/'WORKER_COMPLETE.json')}],{'input_lock_sha256':sha(ROOT/'INPUT_LOCK.json'),'optimizer_steps':0})
    write(ROOT/'COMPLETION.json',{'status':'completed','optimizer_steps':0,'formal_training_updates':0,
        'new_capability_gain_claim':False,'automatic_training_started':False})


if __name__=='__main__':main()
