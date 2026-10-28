"""Use the established last-dimension log-softmax kernel for per-token NLL."""
import argparse
from pathlib import Path
import shutil
import sys

import run_literal_token_focus_v1 as study
from vlm_deep.common import sha_file, write_json
from vlm_grpo_iteration_v1.common import read

ATTEMPT1 = study.ROOT
ROOT = ATTEMPT1.parent/'vlm_literal_token_focus_v2'
PROTOCOL = study.PROJECT_ROOT/'docs/VLM_LITERAL_TOKEN_FOCUS_V2_NUMERICAL_CORRECTION.md'
study.ROOT = ROOT
study.core.ROOT = ROOT
study.PROTOCOL = PROTOCOL
study.RESULT = study.PROJECT_ROOT/'docs/VLM_LITERAL_TOKEN_FOCUS_V2_RESULTS.md'
study.PAGE = study.PROJECT_ROOT/'docs/learning/最难词元损失_数值修正与32步结果.html'


def reductions(logits, labels):
    import torch
    import torch.nn.functional as F
    targets=labels[:,-logits.shape[1]:][:,1:];valid=targets.ne(-100);counts=valid.sum(1)
    if not torch.equal(counts,labels.ne(-100).sum(1)) or not bool(counts.gt(0).all()):
        raise ValueError('Loss readout lost assistant tokens')
    logps=F.log_softmax(logits[:,:-1].float(),dim=-1).gather(-1,targets.masked_fill(~valid,0).unsqueeze(-1)).squeeze(-1)
    nll=-logps.masked_fill(~valid,0);means=nll.sum(1)/counts
    if logits.is_cuda and logits.requires_grad and not (ROOT/'REPRODUCED_REDUCTION_DIFF.json').exists():
        from vlm_sampled_signal_v1.trajectory import completion_logps
        with torch.no_grad():
            legacy=F.cross_entropy(logits[:,:-1].float().transpose(1,2),targets,reduction='none',ignore_index=-100).sum(1)/counts
            old,old_counts=completion_logps(logits,labels)
            write_json(ROOT/'REPRODUCED_REDUCTION_DIFF.json',{'scope':'Same first actual training logits;no extra model forward or optimizer update',
                'legacy_nd_ce_minus_established_max_abs':float((legacy+old/old_counts).abs().max()),
                'corrected_lastdim_minus_established_max_abs':float((means+old/old_counts).abs().max()),
                'legacy_per_qa':legacy.tolist(),'corrected_per_qa':means.tolist(),'counts':counts.tolist()})
    return means,nll.masked_fill(~valid,-float('inf')).max(1).values,counts


study.reductions=reductions
original_render=study.render


def render(result=None):
    original_render(result)
    text=study.RESULT.read_text().replace('(VLM_LITERAL_TOKEN_FOCUS_V1_PROTOCOL.md)', '(VLM_LITERAL_TOKEN_FOCUS_V2_NUMERICAL_CORRECTION.md)').replace('(learning/最难词元损失_平均概率与自由生成.html)', '(learning/最难词元损失_数值修正与32步结果.html)')
    study.RESULT.write_text(text)
    page=study.PAGE.read_text();note='<p>V1候选在首个optimizer.step前停止；本页训练结果属于V2数值修正版。实际逐词元实现见 <a href="../../tools/run_literal_token_focus_v2.py">V2 log_softmax/gather源码</a>，数据、32步预算与科学目标不变。</p>'
    study.PAGE.write_text(page.replace('<h1>平均loss下降，为什么一个词仍然拼错？</h1>', '<h1>平均loss下降，为什么一个词仍然拼错？</h1>'+note))


study.render=render


def prepare():
    # V1 had no optimizer step: retain its original frozen namespace and logs.
    if (ATTEMPT1/'runs/max32/train/FIRST_UPDATE.json').exists() or (ATTEMPT1/'runs/max32/train/LATEST.json').exists():
        raise ValueError('V1 no-update condition differs')
    study.prepare()
    plan=read(ROOT/'PLAN.json')
    for name in ('DATA.json','CPU_INPUTS.json','SCHEDULE.json'):
        if sha_file(ROOT/name)!=sha_file(ATTEMPT1/name):raise ValueError('Scientific inputs changed')
    for path in [Path(__file__),ATTEMPT1/'PLAN.json',ATTEMPT1/'diagnostic/COMPLETE.json',ATTEMPT1/'diagnostic/TOKENS.json',
                 study.PROJECT_ROOT/'docs/VLM_LITERAL_TOKEN_FOCUS_V1_PROTOCOL.md',ATTEMPT1/'runs/mean32/INPUT_LOCK.json']:
        plan['files'][str(path)]=sha_file(path)
    plan['numerical_correction']='Established last-dimension log_softmax/gather;unchanged math,threshold,data,loss,budget'
    plan['control_predictions_source']=str(ATTEMPT1/'runs/mean32')
    write_json(ROOT/'PLAN.json',plan)
    for name in ('COMPLETE.json','TOKENS.json'):
        write_json(ROOT/'diagnostic'/name,read(ATTEMPT1/'diagnostic'/name))
    study.render()


def collect_control():
    source=ATTEMPT1/'runs/mean32'
    complete=read(source/'COMPLETE.json')
    if complete['status']!='completed' or complete['final_lora_sha256']!=read(ROOT/'PLAN.json')['control_lora_sha256']:
        raise ValueError('Original control not complete with matched32 weights')
    receipt={'status':'completed','source':str(source),'source_plan_sha256':sha_file(ATTEMPT1/'PLAN.json'),
             'source_input_lock_sha256':sha_file(source/'INPUT_LOCK.json'),'prediction_files':{},'optimizer_updates_new':0,'generations_new':0}
    for name in study.NAMES:
        path=source/name/'PREDICTIONS.json';dest=ROOT/'runs/mean32'/name/'PREDICTIONS.json';dest.parent.mkdir(parents=True,exist_ok=True)
        if len(read(path))!=len(read(ROOT/'DATA.json')['evaluations'][name]):raise ValueError('Incomplete control cohort')
        shutil.copyfile(path,dest);receipt['prediction_files'][str(path)]=sha_file(path)
    write_json(ROOT/'runs/mean32/CONTROL_REUSE.json',receipt);write_json(ROOT/'runs/mean32/COMPLETE.json',complete)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--worker',action='store_true');args=parser.parse_args()
    if args.prepare:prepare();return
    if args.worker:study.condition('max32');return
    from research_runtime.queue import run_queue
    study.core.verify();entry=str(Path(__file__).resolve());identity={'plan_sha256':sha_file(ROOT/'PLAN.json')}
    run_queue(ROOT/'candidate_queue',[{'key':'max32','command':[sys.executable,'-u',entry,'--worker'],'completion_marker':str(ROOT/'runs/max32/COMPLETE.json')}],identity)
    collect_control();result=study.analyze();result['numerical_correction']=read(ROOT/'REPRODUCED_REDUCTION_DIFF.json')
    result['failed_attempt1_optimizer_updates']=0;write_json(ROOT/'ANALYSIS.json',result);write_json(ROOT/'COMPLETION.json',result);print('TOKEN_FOCUS_V2_COMPLETE',result,flush=True)


if __name__=='__main__':main()
