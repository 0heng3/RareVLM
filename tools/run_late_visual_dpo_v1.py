"""Freeze early vision LoRA; reuse the completed matched DPO training loop."""
import argparse
import importlib
import json
from pathlib import Path
import re
import sys

from rarevlm_paths import PROJECT_ROOT
from vlm_deep.common import sha_file, write_json
from vlm_grpo_iteration_v1.common import read

BASE = PROJECT_ROOT.parents[1]/'outputs'
PARENT = BASE/'vlm_visual_preference_v1'
ROOT = BASE/'vlm_late_visual_dpo_v1'
PROTOCOL = PROJECT_ROOT/'docs/VLM_LATE_VISUAL_DPO_V1_PROTOCOL.md'
PARAMETERS = 614400
INTERNAL = ('native_hard','literal_validation','seen_validation','retention')


def selected(name):
    match = re.search(r'(?:^|\.)visual\.blocks\.(\d+)\.attn\.qkv\.lora_[AB](?:\.default)?\.weight$', name)
    return bool(match and 22 <= int(match[1]) <= 31)


def cpu_check():
    import torch
    from safetensors.torch import load_file
    c=read(PARENT/'CONFIG.json');state=load_file(str(Path(c['initial_adapter'])/'adapter_model.safetensors'))
    chosen={n:t for n,t in state.items() if selected(n)}
    assert len(state)==204 and len(chosen)==20 and sum(t.numel() for t in chosen.values())==PARAMETERS
    assert all(t.dtype==torch.float32 for t in state.values())
    assert not selected('base_model.model.visual.blocks.21.attn.qkv.lora_B.weight')
    assert selected('base_model.model.visual.blocks.22.attn.qkv.lora_B.default.weight')
    assert selected('base_model.model.visual.blocks.31.attn.qkv.lora_A.weight')
    assert not selected('base_model.model.visual.blocks.32.attn.qkv.lora_A.weight')
    assert not selected('base_model.model.model.layers.31.self_attn.q_proj.lora_A.weight')
    early={n:t for n,t in state.items() if '.visual.' in n and not selected(n)}
    assert len(early)==40 and sum(t.numel() for t in early.values())==1228800
    assert all(torch.count_nonzero(t)==0 for n,t in state.items() if '.visual.' in n and '.lora_B.' in n)
    data=read(PARENT/'data/MANIFEST.json');rows={q['question_id']:q for q in data['rows']}
    updates=read(PARENT/'data/SCHEDULE.json')['updates']
    counts={'common':0,'preference':0}
    for update in updates:
        assert len(update['micros'])==4
        for ids in update['micros']:
            assert len(ids)==4
            for qid in ids:counts[rows[qid]['study_source']]+=1
    assert len(updates)==32 and len(rows)==384 and counts=={'common':256,'preference':256}
    ref=read(PARENT/'reference/COMPLETE.json')
    assert sha_file(PARENT/'reference/CACHE.json')==ref['cache_sha256']
    assert read(PARENT/'reference/CACHE.json')['reference_lora_sha256']==read(PARENT/'STUDY_INPUT_LOCK.json')['initial_lora_sha256']
    return {'status':'passed','selected_factors':20,'selected_parameters':PARAMETERS,
            'frozen_early_visual_factors':40,'stored_factors':204,'stored_parameters':3686400,
            'unique_training_qa':384,'chosen_exposures':512,'negative_sequences':256,
            'updates':32,'new_reference_forwards':0,'initial_all_visual_B_zero':True,
            'scope':'Scope boundaries, real adapter inventory, exact old exposure schedule/reference; no repeated model/mask/save audit.'}


def prepare():
    if (ROOT/'STUDY_INPUT_LOCK.json').exists():raise FileExistsError('Study already frozen')
    if read(PARENT/'COMPLETION.json')['status']!='completed':raise ValueError('Require complete matched control')
    write_json(ROOT/'CPU_CHECK.json',cpu_check())
    for name in ('MANIFEST.json','CPU_INPUTS.json','SCHEDULE.json'):
        target=ROOT/'data'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((PARENT/'data'/name).read_bytes())
    c={**read(PARENT/'CONFIG.json'),'output_root':str(ROOT),'roles':['ce_dpo'],
       'algorithm':'late_blocks22_31_CE_plus_reference_DPO','expected_trainable':PARAMETERS,
       'scope_budget':PARAMETERS,'selected_visual_blocks':list(range(22,32)),
       'dpo_weight':{'ce_dpo':1.},'automatic_main_evaluation':True,
       'complete_task_evaluation_unconditional_on_training_fit':True}
    write_json(ROOT/'CONFIG.json',c)
    receipt={**read(PARENT/'DATA_RECEIPT.json'),'optimized_parameters':PARAMETERS,
             'matched_control_source':str(PARENT),'same_384_training_qa_and_512_exposures':True}
    write_json(ROOT/'DATA_RECEIPT.json',receipt)
    sources=[Path(__file__),PROTOCOL,ROOT/'CONFIG.json',ROOT/'CPU_CHECK.json',ROOT/'DATA_RECEIPT.json',
             PARENT/'STUDY_INPUT_LOCK.json',PARENT/'COMPLETION.json',PARENT/'CONFIG.json',
             PARENT/'runs/ce_dpo/train/TRAIN_COMPLETE.json',BASE/'vlm_native_scope_v1/CONFIG.json']
    reuse={}
    for name in INTERNAL+('transfer_dev',):
        source=PARENT/'baseline'/name/'PREDICTIONS.json';dest=ROOT/'baseline'/name/'PREDICTIONS.json'
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(source.read_bytes())
        control=PARENT/'runs/ce_dpo'/('transfer' if name=='transfer_dev' else 'internal')/name/'PREDICTIONS.json'
        reuse[name]={'baseline':str(source),'baseline_sha256':sha_file(source),'control':str(control),'control_sha256':sha_file(control),'qa':len(read(source))}
        sources += [source,dest,control]
    for part,names in [('a',('native_hard','literal_validation','retention')),('b',('seen_validation',))]:
        for name in names:
            dest=ROOT/('baseline_'+part)/name/'PREDICTIONS.json';dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/'baseline'/name/'PREDICTIONS.json').read_bytes())
    write_json(ROOT/'baseline_TRANSFER_COMPLETE.json',{'status':'completed','new_generations':0,'reused':True})
    for name in ('CACHE.json','COMPLETE.json'):
        dest=ROOT/'reference'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((PARENT/'reference'/name).read_bytes());sources += [PARENT/'reference'/name,dest]
    write_json(ROOT/'CONTROL_REUSE.json',{'status':'completed','cohorts':reuse,'new_baseline_or_control_generations':0,'new_control_updates':0,'new_reference_forwards':0,'not_independent_replication':True})
    sources += [ROOT/'CONTROL_REUSE.json']+[ROOT/'data'/n for n in ('MANIFEST.json','CPU_INPUTS.json','SCHEDULE.json')]
    sources += [PROJECT_ROOT/'tools'/n for n in ['run_visual_preference_v1.py','run_native_scope_v1.py','run_native_correction_sft_dpo_v2.py']]
    sources += [PROJECT_ROOT/'src'/n for n in ['vlm_native_scope_v1/model.py','vlm_batch_stable_v1/model.py','vlm_literal_preference_v1/common.py','vlm_alignment_signal/loss.py','vlm_object_train/train.py','research_runtime/queue.py']]
    control=read(PARENT/'runs/ce_dpo/train/TRAIN_COMPLETE.json')
    for directory in [Path(c['initial_adapter']),Path(control['adapter'])]:sources += [directory/n for n in ('adapter_config.json','adapter_model.safetensors')]
    write_json(ROOT/'STUDY_INPUT_LOCK.json',{'status':'frozen','files':{str(f):sha_file(f) for f in sources},
               'initial_lora_sha256':read(PARENT/'STUDY_INPUT_LOCK.json')['initial_lora_sha256'],
               'optimized_parameters':PARAMETERS,'new_updates':32,'new_generations':2577,
               'new_reference_forwards':0,'automatic_more_training':False})
    render()


def verify():
    for f,digest in read(ROOT/'STUDY_INPUT_LOCK.json')['files'].items():
        if sha_file(f)!=digest:raise ValueError('Frozen scope study input changed: '+f)


def load_late(adapter,role,training=False):
    from vlm_native_scope_v1.model import load
    model,processor,binding=load(adapter,role,training)
    if training:
        for name,param in model.named_parameters():param.requires_grad_(selected(name))
        actual=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
        if len(actual)!=20 or sum(p.numel() for n,p in actual)!=PARAMETERS:raise ValueError('Actual late scope differs')
    return model,processor,{**binding,'optimized_scope':'visual_blocks22_31','optimized_parameters':PARAMETERS,'frozen_early_visual_blocks':list(range(2,22))}


def gradient_check(model,role):
    import torch
    chosen=[];b=[];a=[]
    for name,param in model.named_parameters():
        if param.requires_grad!=selected(name):raise ValueError('Actual scope drift: '+name)
        if not param.requires_grad:
            if param.grad is not None:raise ValueError('Frozen parameter got gradient: '+name)
            continue
        if param.dtype!=torch.float32 or param.grad is None or not bool(torch.isfinite(param.grad).all()):raise ValueError('Invalid selected gradient: '+name)
        chosen.append(param)
        (b if '.lora_B.' in name else a).append(float(param.grad.norm()))
    if len(chosen)!=20 or sum(p.numel() for p in chosen)!=PARAMETERS or len(a)!=10 or len(b)!=10:raise ValueError('Late gradient inventory')
    if any(v!=0 for v in a) or not all(v>0 for v in b):raise ValueError('Initial A/B gradient condition')
    if not model.get_base_model().visual.gradient_checkpointing:raise ValueError('Actual visual checkpointing disabled')
    return {'selected_factors':20,'selected_parameters':PARAMETERS,'frozen_parameters_no_gradient':True,
            'visual_A_initial_gradients_zero':True,'visual_B_nonzero_finite':10,
            'visual_B_gradient_min':min(b),'visual_B_gradient_max':max(b),'actual_visual_checkpointing':True}


def train():
    verify();trainer=importlib.import_module('run_visual_preference_v1');scope=importlib.import_module('run_native_scope_v1')
    trainer.ROOT=ROOT;trainer.PARAMETERS=PARAMETERS;trainer.load=load_late;scope.gradient_check=gradient_check
    trainer.train('ce_dpo')


def evaluate(phase):
    from vlm_native_scope_v1.model import load
    from vlm_human_sft.evaluate import lora_hash
    verify();final=read(ROOT/'runs/ce_dpo/train/TRAIN_COMPLETE.json')
    model,processor,_=load(final['adapter'],'visual',False)
    if lora_hash(model)!=final['final_lora_sha256']:raise ValueError('Wrong fixed Final')
    evaluator=importlib.import_module('run_native_correction_sft_dpo_v2');evaluator.ROOT=ROOT
    names=INTERNAL if phase=='internal' else ('transfer_dev',)
    evaluator.evaluate(model,processor,ROOT/'runs/ce_dpo'/phase,names,read(ROOT/'data/MANIFEST.json'))
    write_json(ROOT/'runs/ce_dpo'/(phase+'_COMPLETE.json'),{'status':'completed','cohorts':list(names),'new_generations':960 if phase=='internal' else 1617})


def analyze():
    from vlm_grpo_iteration_v1.study import paired
    verify();data=read(ROOT/'data/MANIFEST.json');rec={r:{} for r in ('start','full_visual_dpo','ce_dpo')}
    for name in INTERNAL+('transfer_dev',):
        rec['start'][name]=read(ROOT/'baseline'/name/'PREDICTIONS.json')
        rec['full_visual_dpo'][name]=read(PARENT/'runs/ce_dpo'/('transfer' if name=='transfer_dev' else 'internal')/name/'PREDICTIONS.json')
        rec['ce_dpo'][name]=read(ROOT/'runs/ce_dpo'/('transfer' if name=='transfer_dev' else 'internal')/name/'PREDICTIONS.json')
        expected=[(q['question_id'],q['inference_input_sha256']) for q in data['evaluations'][name]]
        if any([(q['question_id'],q['input_sha256']) for q in rec[r][name]]!=expected for r in rec):raise ValueError('Evaluation input frame differs')
    for r,records in rec.items():
        for name in ('singleword','multiword','rare_1_5'):
            ids={q['question_id'] for q in data['evaluations']['seen_validation'] if (1<=q['answer_frequency']<=5 if name=='rare_1_5' else q['stratum']==name)}
            records[name]=[q for q in records['seen_validation'] if q['question_id'] in ids]
        ids={q['question_id'] for q in data['evaluations']['transfer_dev'] if 1<=q['answer_frequency']<=5}
        records['full_rare_1_5']=[q for q in records['transfer_dev'] if q['question_id'] in ids]
    def norm(rows,name):
        return [{**q,'image_id':q['filename_family'],'vqa_score':float(q['iiit_alnum_exact'])} for q in rows] if name=='literal_validation' else rows
    metrics={r:{name:100*sum(q['vqa_score'] for q in norm(rows,name))/len(rows) for name,rows in records.items()} for r,records in rec.items()}
    comparisons={r:{name:paired(norm(rec[r][name],name),norm(rows,name)) for name,rows in rec['ce_dpo'].items()} for r in ('start','full_visual_dpo')}
    protections=('literal_validation','seen_validation','singleword','multiword','rare_1_5','retention','full_rare_1_5')
    checks={'task_vs_'+r:comparisons[r]['transfer_dev']['delta_pp']>=.5 and comparisons[r]['transfer_dev']['ci95'][0]>0 for r in comparisons}
    checks.update({'point_protections_vs_'+r:all(comparisons[r][n]['delta_pp']>=-.5 for n in protections) for r in comparisons})
    signal=all(checks.values())
    result={'status':'completed','phase':'pilot_complete_positive_pending_confirmation' if signal else 'pilot_complete_negative',
            'metrics':metrics,'vs_start':{'ce_dpo':comparisons['start']},'candidate_vs_full_dpo':comparisons['full_visual_dpo'],
            'screen_checks':{'ce_dpo':checks},'screen_passed':{'ce_dpo':signal},'native_positive_signal':{'ce_dpo':signal},
            'new_updates':32,'new_generations':2577,'new_reference_forwards':0,'goal_complete':False,
            'automatic_more_training':False,'scope':'Single-seed historical internal matched layer-scope/parameter-budget intervention; not independent replication or external confirmation.'}
    for name in ('SCREEN_ANALYSIS.json','TRANSFER_ANALYSIS.json','COMPLETION.json'):write_json(ROOT/name,result)
    write_json(ROOT/'DECISION.json',{'status':'completed','decision':'internal_signal_pending_independent_confirmation' if signal else 'closed_no_expansion','new_training_admitted':False,'goal_complete':False})
    render(result)


def render(result=None):
    import html,os
    e=lambda s:html.escape(str(s));data=read(ROOT/'data/MANIFEST.json');cards=[]
    cases=[('21816','native_hard'),('11435','transfer_dev')]
    for qid,name in cases:
        q=next(q for q in data['evaluations'][name] if q['question_id']==qid)
        im=data['images'][q['image_id']]['path'];predictions=[]
        for role,label in [('start','共同起点'),('full_visual_dpo','全层DPO强对照'),('ce_dpo','本轮后10层DPO')]:
            f=ROOT/'baseline'/name/'PREDICTIONS.json' if role=='start' else (PARENT if role=='full_visual_dpo' else ROOT)/'runs/ce_dpo'/('transfer' if name=='transfer_dev' else 'internal')/name/'PREDICTIONS.json'
            rows=read(f) if f.exists() else [];a=next((a for a in rows if a['question_id']==qid),None)
            predictions.append(f'{label}：{e(a["prediction"])}（{a["vqa_score"]:.1f}）' if a else label+'：待评测')
        cards.append(f'<article><img src="{e(os.path.relpath(im,ROOT))}"><p>{e(q["question"])}</p><p>'+ '<br>'.join(predictions)+'</p><p>'+('训练诊断原图，不能代表迁移。' if name=='native_hard' else '历史internal病例，不回填训练。')+'</p></article>')
    conclusion='Final待评测，尚无新能力结果。' if result is None else ('内部单seed信号通过，仍待独立确认；不自动扩种。' if result['native_positive_signal']['ce_dpo'] else '完整任务/保持联合门槛未通过，停止该配置。')
    table=''
    if result:
        table='<table><tr><th>模型</th><th>完整1617 VQA%</th><th>完整rare342%</th></tr>'
        for role,label in [('start','共同起点'),('full_visual_dpo','全层DPO'),('ce_dpo','后10层DPO')]:table+=f'<tr><td>{label}</td><td>{result["metrics"][role]["transfer_dev"]:.6f}</td><td>{result["metrics"][role]["full_rare_1_5"]:.6f}</td></tr>'
        table+='</table>'
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>后10层视觉DPO</title><style>body{font:16px/1.8 system-ui;max-width:1200px;margin:30px auto;padding:20px;color:#173044;background:#f4f6f8}.cards{display:grid;grid-template-columns:1fr 1fr;gap:20px}article{background:white;padding:20px}img{width:100%;height:380px;object-fit:contain}pre{padding:20px;background:#e8eff3;white-space:pre-wrap}td,th{padding:10px;border:1px solid #bbcbd5}@media(max-width:700px){.cards{grid-template-columns:1fr}}</style><h1>冻结早层，只更新视觉后10层</h1><p>'+conclusion+'</p><pre>图像 → 视觉blocks0–21冻结 → blocks22–31的qkv LoRA更新 → 冻结merger → 冻结语言模型 → 回答\nW_eff = W_base + (alpha/r) × B @ A\n原控制：30视觉块 / 1843200可训练参数\n本轮：10视觉块 / 614400可训练参数（存储和初始权重不变）\nloss = chosen CE + DPO，beta=.1；同384独立QA、512 chosen曝光、32步。</pre><p>假设是减少共享早层的更新干扰；也可能因容量减少而更难纠错。层范围与参数数同时改变，不能当等参数纯位置因果。训练拟合不阻断完整1617评测，损失下降不算能力提升。</p>'+table+'<div class="cards">'+''.join(cards)+'</div><p>真实原图等比缩放与保存回答；不是注意力，待测指标不填0。</p><p><a href="'+e(os.path.relpath(PROTOCOL,ROOT))+'">冻结协议</a> · <a href="'+e(os.path.relpath(Path(__file__),ROOT))+'">真实训练绑定源码</a> · <a href="../rarevlm_monitor/状态看板.html">运行观察台</a></p></html>'
    (ROOT/'后10层视觉适配.html').write_text(page)
    status='# 后10层视觉DPO当前状态\n\n'+conclusion+'\n\n同原128纠错/384独立QA/512曝光/32步，只将可训练范围缩到视觉blocks22–31，共614400参数。完整1617及内部960在Final后并行评测，不以训练拟合阻断。\n\n[冻结协议](VLM_LATE_VISUAL_DPO_V1_PROTOCOL.md) · [真实代码](../tools/run_late_visual_dpo_v1.py) · [图解与真实病例](../../../outputs/vlm_late_visual_dpo_v1/后10层视觉适配.html)\n'
    if result:status+='\n```json\n'+json.dumps({'metrics':result['metrics'],'checks':result['screen_checks']},ensure_ascii=False,indent=2)+'\n```\n'
    (PROJECT_ROOT/'docs/VLM_LATE_VISUAL_DPO_V1_STATUS.md').write_text(status)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--worker',choices=('train','internal','transfer'));args=parser.parse_args()
    if args.prepare:prepare();return
    if args.worker:
        if args.worker=='train':train()
        else:evaluate(args.worker)
        return
    from research_runtime.queue import run_queue
    verify();identity={'study_sha256':sha_file(ROOT/'STUDY_INPUT_LOCK.json')};entry=str(Path(__file__).resolve())
    run_queue(ROOT/'training_queue',[{'key':'ce_dpo','command':[sys.executable,'-u',entry,'--worker','train'],'completion_marker':str(ROOT/'runs/ce_dpo/train/TRAIN_COMPLETE.json')}],identity)
    run_queue(ROOT/'transfer_queue',[{'key':phase,'command':[sys.executable,'-u',entry,'--worker',phase],'completion_marker':str(ROOT/'runs/ce_dpo'/(phase+'_COMPLETE.json'))} for phase in ('transfer','internal')],identity)
    analyze()


if __name__=='__main__':main()
