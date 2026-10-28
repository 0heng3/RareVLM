import time
from vlm_deep.common import write_json
from .common import ROOT,read


def evaluate(model,processor,role):
    import torch
    from vlm_probe.evaluate import evaluate_batch
    from vlm_deep.evaluate import score_prediction
    from vlm_attention_scope_repair_v1.generation import raw_greedy_config
    data=read(ROOT/'DATA.json');directory=ROOT/'runs'/role;model.eval();model.config.use_cache=True
    model.generation_config=raw_greedy_config(model.generation_config,32);begun=time.monotonic()
    with torch.inference_mode():
        for name,rows in data['evaluations'].items():
            path=directory/name/'PREDICTIONS.json';preds=read(path) if path.exists() else []
            if len(preds)%4 or [q['question_id'] for q in preds]!=[q['question_id'] for q in rows[:len(preds)]]:raise ValueError('Evaluation batch grouping differs')
            for start in range(len(preds),len(rows),4):
                chunk=rows[start:start+4]
                answers,_=evaluate_batch(model,processor,chunk,data['images'],ROOT,next(model.parameters()).device,1024,processor.tokenizer.convert_tokens_to_ids('<|image_pad|>'),32)
                for q,a in zip(chunk,answers):
                    if a['input_tokens']!=q['expected_prefix_tokens'] or a['image_grid_thw']!=q['expected_grid']:raise ValueError('Actual inference frame differs')
                    preds.append({'question_id':q['question_id'],'image_id':q['image_id'],'target':q['target'],'stratum':q['stratum'],
                        'frequency_slice':q.get('frequency_slice'),'evaluation_scored':q['evaluation_scored'],**a,'vqa_score':score_prediction(a['prediction'],q,'textvqa')})
                write_json(path,preds);write_json(directory/'EVAL_PROGRESS.json',{'status':'evaluating','cohort':name,'completed_qa':len(preds),'expected_qa':len(rows),'active_seconds':time.monotonic()-begun})
                if time.monotonic()-begun>1200:raise TimeoutError('1200sec fixed evaluation budget')
