import time
import torch
from vlm_deep.common import write_json
from vlm_human_sft.evaluate import lora_hash
from vlm_fit_positive_control_v1.study import generated
from .common import ROOT,actor,read


def worker(shard):
    pool=read(ROOT/'POOL.json');plan=read(ROOT/'PLAN.json');batches=[pool['rows'][i:i+4] for i in range(0,len(pool['rows']),4)]
    batches=[b for i,b in enumerate(batches) if i%2==shard];flat=[q for b in batches for q in b]
    model,processor,_=actor();model.eval();model.config.use_cache=True
    if lora_hash(model)!=plan['initial_lora_sha256']:raise ValueError('Wrong initial miner')
    directory=ROOT/f'mining_{shard}';path=directory/'PREDICTIONS.json';preds=read(path) if path.exists() else []
    if len(preds)%4 or [q['question_id'] for q in preds]!=[q['question_id'] for q in flat[:len(preds)]]:raise ValueError('Whole mining batches differ')
    previous=read(directory/'PROGRESS.json').get('active_seconds',0.) if (directory/'PROGRESS.json').exists() else 0.
    begin=time.monotonic();torch.cuda.reset_peak_memory_stats()
    for chunk in batches[len(preds)//4:]:
        answers=generated(model,processor,chunk,pool);preds+=answers
        temporary=path.with_suffix('.tmp');write_json(temporary,preds);temporary.replace(path)
        active=previous+time.monotonic()-begin;peak=torch.cuda.max_memory_allocated()/2**30
        write_json(directory/'PROGRESS.json',{'status':'generating_training_donors','completed_qa':len(preds),'expected_qa':len(flat),'active_seconds':active,'peak_allocated_gib':peak})
        if active>600 or peak>65:raise RuntimeError('Mining budget exceeded')
    if lora_hash(model)!=plan['initial_lora_sha256']:raise ValueError('Mining altered weights')
    write_json(directory/'COMPLETE.json',{'status':'completed','qa':len(preds),'formal_optimizer_updates':0,'active_seconds':previous+time.monotonic()-begin})
