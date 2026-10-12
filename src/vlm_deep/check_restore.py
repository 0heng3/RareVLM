"""Restore a real DDP-produced adapter/optimizer/RNG and reproduce dev NLL."""
import argparse
import json
from pathlib import Path
from rarevlm_paths import PROJECT_ROOT
from .common import answer_loss,collate,encode,read_manifest,sha_file,to_device,write_json
from .model import load
from .train import restore_rng


def main():
    import torch
    from peft import get_peft_model_state_dict,set_peft_model_state_dict
    from safetensors.torch import load_file
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--asset-root',type=Path,required=True)
    p.add_argument('--pilot-root',type=Path,required=True)
    p.add_argument('--config',type=Path,default=PROJECT_ROOT/'configs/vlm_deep_v2.json')
    args=p.parse_args();c=json.loads(args.config.read_text());torch.set_num_threads(2)
    directory=args.asset_root/'ddp_contract';trained=json.loads((directory/'TRAIN_RESULT.json').read_text())
    latest=json.loads((directory/'LATEST.json').read_text());checkpoint=directory/latest['directory']
    assert sha_file(checkpoint/'state.pt')==latest['state_sha256']
    for name,digest in latest['adapter_files_sha256'].items():assert sha_file(checkpoint/'adapter'/name)==digest
    state=torch.load(checkpoint/'state.pt',map_location='cpu',weights_only=False)
    model,processor=load(args.pilot_root/'model',c,'cuda:0','qv_r8',training=True)
    tensors=load_file(str(checkpoint/'adapter/adapter_model.safetensors'))
    set_peft_model_state_dict(model,tensors)
    restored=get_peft_model_state_dict(model)
    for name,t in tensors.items():assert torch.equal(t,restored[name].detach().cpu())
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=c['lr'])
    optimizer.load_state_dict(state['optimizer'])
    assert len(optimizer.state)>0 and state['completed_updates']==3
    restore_rng(state['rank_states'][0]['rng'])
    assert torch.equal(torch.get_rng_state(),state['rank_states'][0]['rng']['cpu'])
    assert torch.equal(torch.cuda.get_rng_state(),state['rank_states'][0]['rng']['cuda'])
    manifest=args.asset_root/'contract_data/prepared/manifest.json';m=read_manifest(manifest);total=0.;count=0
    model.eval()
    with torch.inference_mode():
        for r in m['splits']['dev']:
            b=to_device(collate([encode(processor,r,manifest.parent/m['images'][r['image_id']]['path'],True)],processor.tokenizer.pad_token_id),'cuda:0')
            _,nll,n=answer_loss(model,b);total+=float(nll);count+=n
    result={'passed':True,'adapter_tensors_identical_after_restore':True,'optimizer_restored':True,'cpu_cuda_rng_restored':True,
        'checkpoint_updates':state['completed_updates'],'questions_seen':sum(r['stats']['questions'] for r in state['rank_states']),
        'restored_dev_nll':total/count,'original_dev_nll':trained['internal_dev_token_nll'],
        'scope':'real DDP checkpoint restoration contract; not research benchmark evidence'}
    assert abs(result['restored_dev_nll']-result['original_dev_nll'])<1e-5
    write_json(args.asset_root/'CHECKPOINT_RESTORE_CONTRACT.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
