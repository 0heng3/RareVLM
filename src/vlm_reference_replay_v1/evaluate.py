"""Versioned metadata reader; original frozen trainer/evaluation remain intact."""
import time
from vlm_deep.common import write_json
from .common import ROOT,read


def evaluate(model,processor,role):
    import torch
    from vlm_fit_positive_control_v1.study import generated
    data=read(ROOT/'DATA.json');directory=ROOT/'runs'/role;model.eval();model.config.use_cache=True
    prior_seconds=read(directory/'EVAL_PROGRESS.json').get('active_seconds_total',0.) if (directory/'EVAL_PROGRESS.json').exists() else 0.
    begin=time.monotonic();torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for name,rows in data['evaluations'].items():
            path=directory/name/'PREDICTIONS.json';preds=read(path) if path.exists() else []
            if len(preds)>len(rows) or (len(preds)!=len(rows) and len(preds)%4) or [q['question_id'] for q in preds]!=[q['question_id'] for q in rows[:len(preds)]]:
                raise ValueError('Evaluation grouping/partial final batch differs')
            for start in range(len(preds),len(rows),4):
                chunk=rows[start:start+4];answers=generated(model,processor,chunk,data)
                for q,answer in zip(chunk,answers):
                    preds.append({**answer,'stratum':q['stratum'],'frequency_slice':q.get('frequency_slice'),
                                  'evaluation_scored':q['evaluation_scored'],'expected_grid_checked':q['expected_grid']})
                temporary=path.with_suffix('.tmp');write_json(temporary,preds);temporary.replace(path)
                peak=torch.cuda.max_memory_allocated()/2**30
                write_json(directory/'EVAL_PROGRESS.json',{'status':'evaluating','cohort':name,'completed_qa':len(preds),'expected_qa':len(rows),
                           'active_seconds_this_attempt':time.monotonic()-begin,'active_seconds_total':prior_seconds+time.monotonic()-begin,'peak_allocated_gib':peak})
                if prior_seconds+time.monotonic()-begin>2400:raise TimeoutError('2400sec role evaluation budget')
                if peak>65:raise RuntimeError('65GiB evaluation allocation budget exceeded')
