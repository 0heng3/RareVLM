"""Localize observed BF16 tail/full gradient difference; zero optimizer updates."""
import json
from pathlib import Path
import sys

from diagnose_sft_failure import OUT, CONFIG, assign, load, read, setup, weights


def worker():
    torch, c, m = setup()
    import torch.nn.functional as F
    from vlm_deep.common import answer_loss, collate, encode, to_device, write_json
    fit = read(OUT / 'FIT.json')['fixed_sample_loss']['initial']['train_all']['records']
    # Training-only examples; no generated benchmark or post-hoc model selection.
    ids = [fit[0]['question_id'], max(fit, key=lambda r: r['ce'])['question_id']]
    selected = [next(r for r in m['splits']['train_all'] if r['question_id'] == qid) for qid in ids]
    model, processor = load(c)
    assign(model, weights(c, 'initial'), 'rounded_fp32')
    params = [p for n,p in model.named_parameters() if 'lora_' in n]
    for p in params:p.requires_grad_(True)
    root = Path(c['manifest']).parent
    batch = to_device(collate([encode(processor,r,root/m['images'][r['image_id']]['path'],True) for r in selected],processor.tokenizer.pad_token_id),torch.device('cuda:0'))
    head=model.get_output_embeddings(); original=head.forward
    def fp32_head(hidden):
        with torch.autocast('cuda', enabled=False):
            return F.linear(hidden.float(), head.weight.float(), head.bias.float() if head.bias is not None else None)
    values={};grads={}
    try:
        for name,tail,use32 in [('tail_repeat1',True,False),('tail_repeat2',True,False),('full_bf16',False,False),('tail_fp32head',True,True),('full_fp32head',False,True)]:
            head.forward=fp32_head if use32 else original
            with torch.autocast('cuda',dtype=torch.bfloat16):
                losses,_,_=answer_loss(model,batch,tail=tail);loss=losses.mean()
            values[name]=losses.detach().float().cpu().tolist()
            grads[name]=[v.detach().float().cpu() for v in torch.autograd.grad(loss,params)]
            print(name,values[name],flush=True)
    finally:head.forward=original
    def compare(a,b):
        left,right=grads[a],grads[b]
        delta=sum(float((x.double()-y.double()).square().sum()) for x,y in zip(left,right))**.5
        norm=sum(float(y.double().square().sum()) for y in right)**.5
        return {'gradient_difference_l2':delta,'reference_gradient_l2':norm,'relative_gradient_difference':delta/norm,
            'max_parameter_gradient_abs_difference':max(float((x-y).abs().max()) for x,y in zip(left,right)),
            'maximum_loss_difference':max(abs(x-y) for x,y in zip(values[a],values[b]))}
    result={'status':'completed','question_ids':ids,'losses':values,'comparisons':{
        'repeat_same_bf16':compare('tail_repeat1','tail_repeat2'),'tail_vs_full_bf16':compare('tail_repeat1','full_bf16'),
        'tail_vs_full_fp32_head':compare('tail_fp32head','full_fp32head')},'optimizer_updates':0,'scope':'two training cases; numerical localization, no performance experiment'}
    write_json(OUT/'GRADIENT_MATH.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    if '--worker' in sys.argv:worker()
    else:
        from research_runtime.queue import run_queue
        from vlm_deep.common import sha_file
        run_queue(OUT/'gradient_queue',[{'key':'gradient_math','command':[sys.executable,'-u',str(Path(__file__).resolve()),'--worker'],
            'completion_marker':str(OUT/'GRADIENT_MATH.json')}],{'source_sha256':sha_file(Path(__file__)),'zero_optimizer_updates':True})
