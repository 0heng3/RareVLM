"""Matched full-dev FP32-LoRA diagnostic, actual training initial vs Final."""
import json
import os
from pathlib import Path
import sys

from diagnose_sft_failure import OUT, CONFIG, assign, load, read, setup, weights


def worker(name):
    torch,c,m=setup()
    from vlm_deep.common import consensus,content_sha,write_json
    from vlm_probe.evaluate import complete_cache,evaluate_batch
    model,processor=load(c)
    # Match the actual rounded-FP32 training initial, rather than inventing a
    # new raw-FP32 initializer and attributing that difference to training.
    assign(model,weights(c,name),'rounded_fp32' if name=='initial' else 'master_fp32')
    directory=OUT/('full_fp32_'+name);directory.mkdir(exist_ok=True)
    path=directory/'predictions.jsonl';dev=m['splits']['dev'];root=Path(c['manifest']).parent
    records,_=complete_cache(path,dev,m['images'],4)
    image_token=processor.tokenizer.convert_tokens_to_ids('<|image_pad|>')
    with path.open('a') as stream,torch.inference_mode():
        for begin in range(len(records),len(dev),4):
            chunk=dev[begin:begin+4]
            answers,resources=evaluate_batch(model,processor,chunk,m['images'],root,torch.device('cuda:0'),1024,image_token,32)
            for row,answer in zip(chunk,answers):
                result={'question_id':row['question_id'],'image_id':row['image_id'],
                    'question_sha256':content_sha(row['question']),'image_sha256':m['images'][row['image_id']]['sha256'],
                    **answer,'score':consensus(answer['prediction'],row['answers']),**resources,'batch_start':begin}
                records.append(result);stream.write(json.dumps(result,ensure_ascii=False,allow_nan=False)+'\n')
            stream.flush();os.fsync(stream.fileno())
            write_json(directory/'PROGRESS.json',{'status':'running','completed_qa':len(records),'total_qa':len(dev)})
    value={'status':'completed','qa':len(records),'score_percent':100*sum(r['score'] for r in records)/len(records),
        'initial_state':'actual BF16-rounded then FP32 training initializer' if name=='initial' else 'saved FP32 Final masters',
        'base_precision':'bf16','lora_precision':'fp32','autocast':False,'optimizer_updates':0,
        'scope':'new numerical-path diagnostic; original completed study unchanged'}
    write_json(directory/'COMPLETE.json',value);print(json.dumps(value),flush=True)


if __name__=='__main__':
    if len(sys.argv)>1:worker(sys.argv[1])
    else:
        from research_runtime.queue import run_queue
        from vlm_deep.common import sha_file,write_json
        jobs=[{'key':name,'command':[sys.executable,'-u',str(Path(__file__).resolve()),name],
            'completion_marker':str(OUT/('full_fp32_'+name)/'COMPLETE.json')} for name in ['initial','high1024']]
        write_json(OUT/'FP32_FULL_PLAN.json',{'status':'registered_before_generation','conditions':['actual rounded-FP32 initial','Final FP32 masters'],
            'qa_per_condition':1617,'original_batch_size':4,'original_order':True,'visual_budget':1024,'greedy_new_tokens':32,
            'base':'original BF16','lora':'FP32','optimizer_updates':0,'purpose':'measure the evaluation-path interaction with the existing training effect; no recovery of the original failed gate'})
        run_queue(OUT/'fp32_full_queue',jobs,{'source_sha256':sha_file(Path(__file__)),'zero_optimizer_updates':True})
