# Exact parent evaluation body; only this namespace binds a different output ROOT.
import time
import torch
from vlm_deep.common import write_json
from vlm_fit_positive_control_v1.study import generated
from .common import ROOT,PARENT,read


def evaluate(model,processor):
    data=read(PARENT/'DATA.json');directory=ROOT/'runs/candidate';model.eval();model.config.use_cache=True
    prior=read(directory/'EVAL_PROGRESS.json').get('active_seconds_total',0.) if (directory/'EVAL_PROGRESS.json').exists() else 0.
    begin=time.monotonic();torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for name,rows in data['evaluations'].items():
            path=directory/name/'PREDICTIONS.json';preds=read(path) if path.exists() else []
            if len(preds)>len(rows) or (len(preds)!=len(rows) and len(preds)%4) or [q['question_id'] for q in preds]!=[q['question_id'] for q in rows[:len(preds)]]:raise ValueError('Fixed evaluation grouping differs')
            for start in range(len(preds),len(rows),4):
                chunk=rows[start:start+4];answers=generated(model,processor,chunk,data)
                preds += [{**a,'stratum':q['stratum'],'frequency_slice':q.get('frequency_slice'),'evaluation_scored':q['evaluation_scored'],'expected_grid_checked':q['expected_grid']} for q,a in zip(chunk,answers)]
                temporary=path.with_suffix('.tmp');write_json(temporary,preds);temporary.replace(path)
                active=prior+time.monotonic()-begin;peak=torch.cuda.max_memory_allocated()/2**30
                write_json(directory/'EVAL_PROGRESS.json',{'status':'evaluating','cohort':name,'completed_qa':len(preds),'expected_qa':len(rows),'active_seconds_total':active,'peak_allocated_gib':peak})
                if active>2400 or peak>65:raise RuntimeError('Evaluation budget exceeded')
