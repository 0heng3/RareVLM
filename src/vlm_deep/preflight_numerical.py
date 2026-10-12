"""Original FP32 inference memory and last-logit generation equivalence."""
import argparse
import json
from pathlib import Path
from rarevlm_paths import PROJECT_ROOT
from .common import collate,content_sha,encode,to_device,write_json
from .model import load


def main():
    import torch
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--asset-root',type=Path,required=True);p.add_argument('--pilot-root',type=Path,required=True)
    p.add_argument('--config',type=Path,default=PROJECT_ROOT/'configs/vlm_deep_v2.json')
    args=p.parse_args();c=json.loads(args.config.read_text());torch.set_num_threads(2)
    old=json.loads((args.pilot_root/'prepared/manifest.json').read_text());row=old['splits']['train'][0]
    model,processor=load(args.pilot_root/'model',c,'cuda:0',quantization='original')
    model.generation_config.temperature=1.;model.generation_config.top_p=1.;model.generation_config.top_k=50
    batch=to_device(collate([encode(processor,row,args.pilot_root/'prepared'/row['image'],False)],processor.tokenizer.pad_token_id),'cuda:0')
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        full=model.generate(**batch,max_new_tokens=8,do_sample=False,use_cache=True)
        hook=model.get_output_embeddings().register_forward_pre_hook(lambda module,items:(items[0][:,-1:,:],))
        try:optimized=model.generate(**batch,max_new_tokens=8,do_sample=False,use_cache=True)
        finally:hook.remove()
    if not torch.equal(full,optimized):raise ValueError('last-logit optimization changed training-image generation')
    result={'passed':True,'config_sha256':content_sha(c),'quantization':'original','precision':'fp32',
        'last_logit_generation_equal':True,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
        'gpu':torch.cuda.get_device_name(),'scope':'training-image numerical-path preflight; no benchmark scoring'}
    write_json(args.asset_root/'preflight/original_fp32.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
