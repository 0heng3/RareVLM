"""Local image/question inference CLI; no reference answer accepted."""
import argparse
import json
import time
from pathlib import Path
from .common import user_messages
from .model import load_stack


def main():
    import torch
    from PIL import Image
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir',type=Path,required=True);p.add_argument('--adapter-dir',type=Path)
    p.add_argument('--image',type=Path,required=True);p.add_argument('--question',required=True)
    p.add_argument('--config',type=Path,default=Path(__file__).resolve().parents[1]/'configs/vlm_sft_textvqa.json')
    p.add_argument('--device',default='cuda:0')
    args=p.parse_args();config=json.loads(args.config.read_text())
    model,processor=load_stack(args.model_dir,config,args.device,adapter_dir=args.adapter_dir)
    text=processor.apply_chat_template(user_messages(args.question),tokenize=False,add_generation_prompt=True)
    batch=processor(text=[text],images=[Image.open(args.image).convert('RGB')],return_tensors='pt').to(args.device)
    with torch.inference_mode():
        torch.cuda.synchronize();start=time.monotonic()
        generated=model.generate(**batch,max_new_tokens=config['max_new_tokens'],do_sample=False,use_cache=True)
        torch.cuda.synchronize();duration=time.monotonic()-start
    prediction=processor.tokenizer.decode(generated[0,batch['input_ids'].shape[1]:],skip_special_tokens=True)
    print(json.dumps({'answer':prediction.strip(),'generation_seconds':duration,'image':str(args.image)},ensure_ascii=False))


if __name__=='__main__':main()
