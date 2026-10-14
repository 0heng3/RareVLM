"""Execute only pinned upstream scoring nodes against saved full predictions."""
from __future__ import annotations
import ast
import csv
import json
from pathlib import Path
import re
import statistics

from rarevlm_paths import PROJECT_ROOT, SOURCE_ROOT
from vlm_deep.evaluate import score_prediction
from vlm_formal.evaluate import task_data
from vlm_probe.evaluate import now, sha_file, write_json


def scoring_audit(root):
    directory=root/'reading';directory.mkdir(exist_ok=True)
    report=directory/'TEXTVQA_SCORING_AUDIT.json'
    if report.exists():return json.loads(report.read_text())
    upstream=root/'upstream/EvolvingLMMs-Lab__lmms-eval/lmms_eval/tasks'
    processor_path=upstream/'_task_utils/vqa_eval_metric.py';score_path=upstream/'textvqa/utils.py'
    namespace={'re':re,'statistics':statistics}
    for path,name in [(processor_path,'EvalAIAnswerProcessor'),(score_path,'textvqa_process_results')]:
        tree=ast.parse(path.read_text())
        nodes=[node for node in tree.body if getattr(node,'name',None)==name]
        if len(nodes)!=1:raise ValueError('Pinned scorer AST node missing')
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    upstream_score=namespace['textvqa_process_results']
    config_path=PROJECT_ROOT/'configs/vlm_strong_confirm_v1.json';cfg=json.loads(config_path.read_text())
    rows,_,_,binding=task_data(Path(cfg['asset_root']),'textvqa',cfg)
    by_id={str(r['question_id']):r for r in rows};cells=[]
    predictions_root=Path(cfg['output_root'])
    with (directory/'textvqa_scoring_comparison.csv').open('w') as stream:
        writer=csv.writer(stream);writer.writerow(['condition','question_id','saved_score','rarevlm_rescore','upstream_rescore','delta'])
        for condition in ['dense4','segmented16']:
            path=predictions_root/'runs/textvqa'/condition/'predictions.jsonl'
            before=sha_file(path);predictions=[json.loads(line) for line in path.read_text().splitlines()]
            if len(predictions)!=5000 or len({str(r['question_id']) for r in predictions})!=5000:
                raise ValueError('Require complete unique saved predictions')
            old_scores=[];new_scores=[];differences=[];saved_mismatches=0
            for prediction in predictions:
                row=by_id[str(prediction['question_id'])]
                local=float(score_prediction(prediction['prediction'],row,'textvqa'))
                doc={'question_id':row['question_id'],'answers':list(row['answers'])}
                external=float(upstream_score(doc,[prediction['prediction']])['exact_match'])
                delta=external-local
                saved_mismatches+=int(abs(local-float(prediction['score']))>1e-12)
                old_scores.append(local);new_scores.append(external);differences.append(delta)
                writer.writerow([condition,row['question_id'],prediction['score'],local,external,delta])
            if before!=sha_file(path):raise ValueError('Saved predictions changed during scoring')
            cells.append({'condition':condition,'questions':5000,'saved_predictions_sha256':before,
                'rarevlm_score_percent':100*statistics.mean(old_scores),'upstream_score_percent':100*statistics.mean(new_scores),
                'different_scores_above_1e_minus_12':sum(abs(x)>1e-12 for x in differences),
                'max_absolute_score_delta':max(map(abs,differences)),'saved_score_mismatches':saved_mismatches})
    examples=[]
    for matches in range(1,5):
        references=['cat']*matches+['dog']*(10-matches)
        examples.append({'matching_references':matches,'score':upstream_score({'question_id':matches,'answers':references},['cat'])['exact_match']})
    result={'status':'completed','completed_at':now(),'upstream_sources_sha256':{str(p.relative_to(root)):sha_file(p) for p in [processor_path,score_path]},
        'project_scorer_sha256':sha_file(SOURCE_ROOT/'vlm_deep/third_party/vqa_eval_metric.py'),
        'data_binding':binding,'cells':cells,'leave_one_out_examples':examples,
        'scope':'actual unmodified pinned scorer function/class AST nodes,stdlib dependencies; saved outputs only,zero generation;not a full lmms-eval framework or input-equivalence run',
        'input_differences':{'lmms_current_qwen_default_max_pixels':1605632,'rarevlm_max_pixels':802816,
            'lmms_default_min_pixels':200704,'rarevlm_min_pixels':50176,
            'question_format':'lmms textvqa capitalizes question; task OCR configurable; RareVLM original question/frozen suffix,no reference OCR',
            'pixel_budget_scope':'declared upper/lower bounds,not asserting every image has doubled visual tokens'}}
    write_json(report,result);return result
