"""Shared immutable inputs and real generation metrics for the32-step preference probe."""
from pathlib import Path
import json
import torch
from rarevlm_paths import PROJECT_ROOT
from vlm_deep.common import encode,collate,content_sha,write_json
from vlm_sft.common import to_device
from vlm_probe.evaluate import evaluate_batch
from vlm_attention_scope_repair_v1.generation import raw_greedy_config
from vlm_deep.evaluate import score_prediction

ROOT=PROJECT_ROOT.parents[1]/'outputs/vlm_region_preference_probe_v1'
CONFIG=PROJECT_ROOT/'configs/vlm_region_preference_probe_v1.json'
def read(path):return json.loads(Path(path).read_text())


def samples(processor,data,rows,cpu,rejected=False):
    result=[]
    for row in rows:
        kind='rejected' if rejected else 'chosen'
        q={**row,'target':row['rejected_target'] if rejected else row['target']}
        item=encode(processor,q,Path(data['images'][q['image_id']]['path']),True)
        expected=cpu[q['question_id']+':'+kind]
        if any(content_sha(item[key].tolist())!=expected[key+'_sha256'] for key in ('input_ids','labels')):
            raise ValueError('Actual supervised input differs from registered CPU encoding')
        result.append(item)
    return result


def binding(batch,ids):
    return {'question_ids':list(ids),'padded_input_ids_sha256':content_sha(batch['input_ids'].tolist()),
        'padded_labels_sha256':content_sha(batch['labels'].tolist()),
        'grid_sha256':content_sha(batch['image_grid_thw'].tolist())}


def make_batch(processor,data,rows,cpu):
    chosen=samples(processor,data,rows,cpu)
    pref=rows[0]['study_source']=='preference'
    if any((q['study_source']=='preference')!=pref for q in rows):raise ValueError('Mixed-source microbatch')
    rejected=samples(processor,data,rows,cpu,True) if pref else []
    batch=collate(chosen+rejected,processor.tokenizer.pad_token_id)
    return batch,binding(batch,[q['question_id'] for q in rows]),chosen,rejected


def evaluate(model,processor,directory):
    data=read(ROOT/'data/MANIFEST.json')
    model.eval();model.config.use_cache=True;model.generation_config=raw_greedy_config(model.generation_config,32)
    summary={}
    with torch.inference_mode():
        for name,key in [('hard','hard_eval'),('retention','retention_eval')]:
            rows=data[key];path=directory/name/'PREDICTIONS.json';predictions=read(path) if path.exists() else []
            if len(predictions)%4 or [q['question_id'] for q in predictions]!=[q['question_id'] for q in rows[:len(predictions)]]:
                raise ValueError('Generation cache grouping differs')
            for start in range(len(predictions),len(rows),4):
                chunk=rows[start:start+4]
                answers,_=evaluate_batch(model,processor,chunk,data['images'],ROOT,next(model.parameters()).device,
                    1024,processor.tokenizer.convert_tokens_to_ids('<|image_pad|>'),32)
                for q,a in zip(chunk,answers):
                    predictions.append({'question_id':q['question_id'],'image_id':q['image_id'],'target':q['target'],**a,
                        'pair_id':q.get('pair_id'),'pair_side':q.get('pair_side'),
                        'raw_exact':a['prediction']==q['target'],
                        'casefold_exact':a['prediction'].casefold()==q['target'].casefold(),
                        'vqa_score':score_prediction(a['prediction'],q,'textvqa') if name=='retention' else None})
                write_json(path,predictions)
                write_json(directory/'EVAL_PROGRESS.json',{'status':'evaluating','cohort':name,'completed_qa':len(predictions),'expected_qa':len(rows)})
            if name=='hard':
                pair_rows=[]
                for start in range(0,len(predictions),2):
                    a,b=predictions[start:start+2]
                    if a['pair_id']!=b['pair_id']:raise ValueError('Unpaired region evaluation')
                    pair_rows.append({'pair_id':a['pair_id'],'both_correct':a['casefold_exact'] and b['casefold_exact'],
                        'same_answer':a['prediction'].casefold()==b['prediction'].casefold()})
                write_json(directory/name/'PAIRS.json',pair_rows)
                summary[name]={'qa':len(rows),'raw_correct':sum(q['raw_exact'] for q in predictions),
                    'casefold_correct':sum(q['casefold_exact'] for q in predictions),'pairs':len(pair_rows),
                    'both_correct':sum(q['both_correct'] for q in pair_rows),'same_answer':sum(q['same_answer'] for q in pair_rows)}
            else:
                score=sum(q['vqa_score'] for q in predictions)
                summary[name]={'qa':len(rows),'score_points':score,'score_percent':100*score/len(rows)}
    write_json(directory/'EVAL_COMPLETE.json',{'status':'completed','scope':'Training-only positive control, not main benchmark','metrics':summary})
    return summary
