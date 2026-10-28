"""Observe saved training-only checkpoints; no optimizer, no new model selection."""
import argparse
from pathlib import Path
import sys
import time
from rarevlm_paths import PROJECT_ROOT,SOURCE_ROOT
from vlm_deep.common import sha_file,write_json
from vlm_grpo_iteration_v1.common import actor,read

ROOT=PROJECT_ROOT.parents[1]/'outputs/vlm_grpo_learning_curve_v1'
OLD=ROOT.parent/'vlm_grpo_iteration_v1'
STEPS=(1,8,16,24)


def prepare():
    if (ROOT/'PLAN.json').exists():raise FileExistsError('Already prepared')
    old=read(OLD/'PLAN.json');data=read(OLD/'DATA.json')
    write_json(ROOT/'DATA.json',{'rows':data['hard_eval'],'images':{q['image_id']:data['images'][q['image_id']] for q in data['hard_eval']}})
    paths=[Path(__file__).resolve(),PROJECT_ROOT/'docs/VLM_GRPO_LEARNING_CURVE_V1_PROTOCOL.md',ROOT/'DATA.json',OLD/'PLAN.json']
    points={}
    for role in ('ce','ce_grpo'):
        points[role]={}
        for step in STEPS:
            matches=list((OLD/'runs'/role/'train').glob(f'checkpoint_{step:06d}_*'))
            if len(matches)!=1:raise ValueError('Ambiguous checkpoint point')
            points[role][str(step)]=str(matches[0]/'adapter')
            paths.extend(matches[0]/'adapter'/name for name in ('adapter_model.safetensors','adapter_config.json'))
        for step,source in [('0',OLD/'initial/hard/PREDICTIONS.json'),('32',OLD/'runs'/role/'hard/PREDICTIONS.json')]:
            write_json(ROOT/role/f'step{step}'/'PREDICTIONS.json',read(source));paths.append(source)
    paths+=[SOURCE_ROOT/f for f in ('vlm_grpo_iteration_v1/common.py','vlm_fp32_adapter/model.py','vlm_fp32_readout_v1/model.py',
        'vlm_runtime_metadata_v1/model.py','vlm_human_sft/evaluate.py','vlm_probe/evaluate.py','vlm_deep/common.py',
        'vlm_deep/evaluate.py','vlm_attention_scope_repair_v1/generation.py','research_runtime/queue.py')]
    write_json(ROOT/'PLAN.json',{'status':'prepared','points':points,'files':{str(p):sha_file(p) for p in paths},
        'images':{im['path']:im['sha256'] for im in read(ROOT/'DATA.json')['images'].values()},'new_generation_qa':256,'training_updates':0})
    print('Prepared eight intermediate checkpoint observations;Final reused',flush=True)


def worker(role):
    import torch
    from vlm_fp32_adapter.model import restore_fp32_adapter
    from vlm_human_sft.evaluate import lora_hash
    from vlm_probe.evaluate import evaluate_batch
    from vlm_deep.evaluate import score_prediction
    from vlm_attention_scope_repair_v1.generation import raw_greedy_config
    plan=read(ROOT/'PLAN.json')
    for path,digest in {**plan['files'],**plan['images']}.items():
        if sha_file(path)!=digest:raise ValueError('Frozen curve source changed: '+path)
    model,processor,_=actor();data=read(ROOT/'DATA.json');started=time.monotonic()
    model.eval();model.generation_config=raw_greedy_config(model.generation_config,32)
    with torch.inference_mode():
        for step in STEPS:
            path=ROOT/role/f'step{step}'/'PREDICTIONS.json'
            restore_fp32_adapter(model,plan['points'][role][str(step)]);model.eval()
            identity=lora_hash(model);rows=read(path) if path.exists() else []
            if len(rows)%4 or [q['question_id'] for q in rows]!=[q['question_id'] for q in data['rows'][:len(rows)]]:raise ValueError('Curve resume grouping changed')
            for start in range(len(rows),32,4):
                chunk=data['rows'][start:start+4]
                answers,_=evaluate_batch(model,processor,chunk,data['images'],ROOT,next(model.parameters()).device,1024,processor.tokenizer.convert_tokens_to_ids('<|image_pad|>'),32)
                for q,a in zip(chunk,answers):
                    if a['input_tokens']!=q['expected_prefix_tokens'] or a['image_grid_thw']!=q['expected_grid']:raise ValueError('Curve frame changed')
                    rows.append({'question_id':q['question_id'],'target':q['target'],'image_id':q['image_id'],**a,'vqa_score':score_prediction(a['prediction'],q,'textvqa')})
                write_json(path,rows);write_json(ROOT/role/'PROGRESS.json',{'status':'observing','step':step,'completed_qa_this_step':len(rows),'active_seconds':time.monotonic()-started,'training_updates':0})
                if time.monotonic()-started>300:raise TimeoutError('Curve read-only active cap')
            if identity!=lora_hash(model):raise ValueError('Read-only curve changed adapter')
            write_json(ROOT/role/f'step{step}'/'IDENTITY.json',{'adapter':plan['points'][role][str(step)],'lora_sha256':identity})
    write_json(ROOT/role/'COMPLETE.json',{'status':'completed','new_generation_qa':128,'training_updates':0,'active_seconds':time.monotonic()-started})


def analyze():
    data=read(ROOT/'DATA.json');result={'status':'completed','new_generation_qa':256,'training_updates':0,'roles':{}}
    for role in ('ce','ce_grpo'):
        points={step:read(ROOT/role/f'step{step}'/'PREDICTIONS.json') for step in (0,1,8,16,24,32)}
        series={str(step):{'full':sum(q['vqa_score']>=.999999 for q in rows),'vqa_percent':100*sum(q['vqa_score'] for q in rows)/32} for step,rows in points.items()}
        cases=[]
        for i,q in enumerate(data['rows']):
            success=[step for step,rows in points.items() if rows[i]['vqa_score']>=.999999]
            cases.append({'question_id':q['question_id'],'target':q['target'],'full_score_observed_at':success,
                'ever_full':bool(success),'final_full':32 in success,'observed_full_then_final_not_full':bool(success) and 32 not in success,
                'predictions':{str(step):rows[i]['prediction'] for step,rows in points.items()}})
        result['roles'][role]={'points':series,'never_full_at_observed_points':sum(not q['ever_full'] for q in cases),
            'ever_full_then_final_not_full':sum(q['observed_full_then_final_not_full'] for q in cases),'cases':cases}
    write_json(ROOT/'ANALYSIS.json',result)
    text='# 已存训练检查点纠错曲线完成（0更新）\n\n'
    for role,v in result['roles'].items():
        text+=f'## {role}\n\n|步数|训练满分/32|训练VQA|\n|---|---:|---:|\n'
        text+=''.join(f'|{step}|{scores["full"]}|{scores["vqa_percent"]:.6f}|\n' for step,scores in v['points'].items())
        text+=f'\n离散观察点从未满分{v["never_full_at_observed_points"]}题；曾满分而Final不满分{v["ever_full_then_final_not_full"]}题。\n\n'
    text+='0/32复用旧预测、1/8/16/24各新生成，0优化器。不能由离散观察断言未观察步骤全过程，不据此选择checkpoint或回写旧Final/512失败；这是训练学习机制，非泛化提升。\n'
    (PROJECT_ROOT/'docs/VLM_GRPO_LEARNING_CURVE_V1_RESULTS.md').write_text(text)
    write_json(ROOT/'COMPLETION.json',result);print({r:{k:v for k,v in d.items() if k!='cases'} for r,d in result['roles'].items()},flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--role',choices=['ce','ce_grpo']);a=p.parse_args()
    if a.prepare:prepare()
    elif a.role:worker(a.role)
    else:
        from research_runtime.queue import run_queue
        run_queue(ROOT/'queue',[{'key':r,'command':[sys.executable,'-u',str(Path(__file__).resolve()),'--role',r],
            'completion_marker':str(ROOT/r/'COMPLETE.json')} for r in ('ce','ce_grpo')],{'plan_sha256':sha_file(ROOT/'PLAN.json')})
        analyze()


if __name__=='__main__':main()
