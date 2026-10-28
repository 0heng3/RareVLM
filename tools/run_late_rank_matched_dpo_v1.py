"""A single equal-trainable-budget late-vision intervention, not a rank sweep."""
import argparse
import copy
import importlib
import json
import math
from pathlib import Path
import sys
import time

from rarevlm_paths import PROJECT_ROOT
from vlm_deep.common import sha_file,write_json
from vlm_grpo_iteration_v1.common import read
import run_late_visual_dpo_v1 as late

BASE=PROJECT_ROOT.parents[1]/'outputs'
ROOT=BASE/'vlm_late_rank_matched_dpo_v1'
PARENT=BASE/'vlm_visual_preference_v1'
LOW=BASE/'vlm_late_visual_dpo_v1'
PROTOCOL=PROJECT_ROOT/'docs/VLM_LATE_RANK_MATCHED_DPO_V1_PROTOCOL.md'
PARAMETERS=1843200


def build_initialization():
    import torch
    from safetensors.torch import load_file,save_file
    from peft import LoraConfig,get_peft_model
    from transformers import AutoConfig,Qwen2_5_VLForConditionalGeneration
    from vlm_human_sft.evaluate import lora_hash
    original=Path(read(PARENT/'CONFIG.json')['initial_adapter']);old=load_file(str(original/'adapter_model.safetensors'))
    state={n:t.clone() for n,t in old.items()};cfg=copy.deepcopy(read(original/'adapter_config.json'))
    for block in range(22,32):
        module=f'visual.blocks.{block}.attn.qkv';cfg['rank_pattern'][module]=36;cfg['alpha_pattern'][module]=72
        a=next(n for n in state if f'.{module}.lora_A.' in n);b=a.replace('.lora_A.','.lora_B.')
        assert old[a].shape==(12,1280) and old[b].shape==(3840,12)
        extra=torch.empty((24,1280),dtype=torch.float32)
        torch.nn.init.kaiming_uniform_(extra,a=math.sqrt(5),generator=torch.Generator().manual_seed(42+block))
        state[a]=torch.cat([old[a],extra],0);state[b]=torch.zeros((3840,36),dtype=torch.float32)
    cfg.update(inference_mode=False,lora_dropout=0.)
    dest=ROOT/'initial_adapter';dest.mkdir(parents=True,exist_ok=False)
    write_json(dest/'adapter_config.json',cfg);save_file(state,str(dest/'adapter_model.safetensors'),metadata={'format':'pt'})
    base=read(read(PARENT/'CONFIG.json')['baseline_config']);config=AutoConfig.from_pretrained(base['model_root'],local_files_only=True)
    with torch.device('meta'):model=get_peft_model(Qwen2_5_VLForConditionalGeneration(config),LoraConfig.from_pretrained(str(dest)))
    found=set()
    for name,param in list(model.named_parameters()):
        if 'lora_' not in name:continue
        key=name.replace('.default.','.')
        if key not in state or param.shape!=state[key].shape:raise ValueError('Actual PEFT rank shape mismatch: '+name)
        path,leaf=name.rsplit('.',1);model.get_submodule(path).register_parameter(leaf,torch.nn.Parameter(state[key].clone()));found.add(key)
    if found!=set(state) or len(state)!=204 or sum(t.numel() for t in state.values())!=4915200:raise ValueError('Initializer inventory mismatch')
    if sum(t.numel() for n,t in state.items() if late.selected(n))!=PARAMETERS:raise ValueError('Trainable budget mismatch')
    for name,module in model.named_modules():
        if hasattr(module,'lora_A') and len(module.lora_A):
            expected=36 if any(f'.visual.blocks.{i}.attn.qkv' in name for i in range(22,32)) else 12 if '.visual.' in name else 8
            if module.r['default']!=expected or module.scaling['default']!=2.:raise ValueError('Actual PEFT rank/scaling mismatch')
    for name,tensor in old.items():
        if not late.selected(name):assert torch.equal(tensor,state[name])
        elif '.lora_A.' in name:assert torch.equal(tensor,state[name][:12])
        else:assert torch.count_nonzero(state[name])==0
    assert all(torch.count_nonzero(t)==0 for n,t in state.items() if '.visual.' in n and '.lora_B.' in n)
    digest=lora_hash(model);model.save_pretrained(ROOT/'cpu_roundtrip',safe_serialization=True)
    saved=load_file(str(ROOT/'cpu_roundtrip/adapter_model.safetensors'))
    assert set(saved)==set(state) and all(torch.equal(v,saved[n]) for n,v in state.items())
    result={'status':'passed','adapter':str(dest),'lora_sha256':digest,'stored_factors':204,'stored_parameters':4915200,
            'selected_factors':20,'selected_parameters':PARAMETERS,'rank':36,'alpha':72,'scale':2.,
            'old_language_and_early_values_exact':True,'old_late_A_first12_rows_exact':True,'all_visual_B_zero':True,
            'actual_peft_injection_and_save_roundtrip':True,'effective_initialization_equivalence':'Same language factors/scale; all visual BA zero. Actual fixed8 GPU parity still required.',
            'source_adapter':str(original),'source_adapter_sha256':sha_file(original/'adapter_model.safetensors')}
    write_json(ROOT/'INITIALIZATION.json',result);return result


def prepare():
    if (ROOT/'STUDY_INPUT_LOCK.json').exists():raise FileExistsError('Study already frozen')
    if read(LOW/'COMPLETION.json')['phase']!='pilot_complete_negative':raise ValueError('Require closed lower-budget result')
    init=build_initialization()
    for name in ('MANIFEST.json','CPU_INPUTS.json','SCHEDULE.json'):
        dest=ROOT/'data'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((LOW/'data'/name).read_bytes())
    c={**read(LOW/'CONFIG.json'),'output_root':str(ROOT),'initial_adapter':init['adapter'],
       'expected_trainable':PARAMETERS,'scope_budget':PARAMETERS,'shared_composite_parameters':4915200,
       'algorithm':'late_blocks22_31_rank36_CE_plus_DPO','selected_rank':36,'selected_alpha':72}
    write_json(ROOT/'CONFIG.json',c)
    write_json(ROOT/'DATA_RECEIPT.json',{**read(LOW/'DATA_RECEIPT.json'),'optimized_parameters':PARAMETERS,'rank':36,'data_source':str(LOW),'no_new_data':True})
    paths=[Path(__file__),PROTOCOL,ROOT/'INITIALIZATION.json',ROOT/'CONFIG.json',ROOT/'DATA_RECEIPT.json',
           LOW/'COMPLETION.json',LOW/'STUDY_INPUT_LOCK.json',LOW/'CPU_CHECK.json',PARENT/'STUDY_INPUT_LOCK.json',
           BASE/'vlm_native_scope_v1/CONFIG.json']
    for name in ('MANIFEST.json','CPU_INPUTS.json','SCHEDULE.json'):paths += [ROOT/'data'/name,LOW/'data'/name]
    cohorts=late.INTERNAL+('transfer_dev',);reuse={}
    for name in cohorts:
        source=LOW/'baseline'/name/'PREDICTIONS.json';dest=ROOT/'baseline'/name/'PREDICTIONS.json'
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(source.read_bytes())
        controls={r:root/'runs/ce_dpo'/('transfer' if name=='transfer_dev' else 'internal')/name/'PREDICTIONS.json' for r,root in [('full_visual_dpo',PARENT),('late12',LOW)]}
        reuse[name]={'baseline':str(source),'baseline_sha256':sha_file(source),'controls':{r:{'path':str(f),'sha256':sha_file(f)} for r,f in controls.items()}}
        paths += [source,dest,*controls.values()]
    for part,names in [('a',('native_hard','literal_validation','retention')),('b',('seen_validation',))]:
        for name in names:
            dest=ROOT/('baseline_'+part)/name/'PREDICTIONS.json';dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/'baseline'/name/'PREDICTIONS.json').read_bytes())
    write_json(ROOT/'baseline_TRANSFER_COMPLETE.json',{'status':'completed','new_generations':0,'reused':True})
    for name in ('CACHE.json','COMPLETE.json'):
        source=PARENT/'reference'/name;dest=ROOT/'reference'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(source.read_bytes());paths += [source,dest]
    write_json(ROOT/'CONTROL_REUSE.json',{'status':'completed','cohorts':reuse,'new_control_updates':0,'new_control_generations':0,'new_reference_forwards':0,
               'original_reference_identity_preserved':True,'new_initialization_lora_sha256':init['lora_sha256'],
               'reuse_basis':init['effective_initialization_equivalence'],'not_independent_replication':True})
    paths += [ROOT/'CONTROL_REUSE.json']
    paths += [PROJECT_ROOT/'tools'/n for n in ('run_late_visual_dpo_v1.py','run_visual_preference_v1.py','run_native_scope_v1.py','run_native_correction_sft_dpo_v2.py')]
    paths += [PROJECT_ROOT/'src'/n for n in ('vlm_batch_stable_v1/model.py','vlm_native_scope_v1/model.py','vlm_literal_preference_v1/common.py','vlm_alignment_signal/loss.py','research_runtime/queue.py')]
    paths += [Path(init['adapter'])/n for n in ('adapter_config.json','adapter_model.safetensors')]
    write_json(ROOT/'STUDY_INPUT_LOCK.json',{'status':'frozen','files':{str(f):sha_file(f) for f in paths},
               'initial_lora_sha256':init['lora_sha256'],'optimized_parameters':PARAMETERS,'new_updates':32,
               'new_generations':2585,'new_reference_forwards':0,'automatic_more_training':False})
    render()


def verify():
    for f,digest in read(ROOT/'STUDY_INPUT_LOCK.json')['files'].items():
        if sha_file(f)!=digest:raise ValueError('Frozen input changed: '+f)


def load_new(adapter,role,training=False):
    import torch
    from vlm_batch_stable_v1.model import load
    from vlm_probe.evaluate import processor_for
    from vlm_human_sft.evaluate import lora_hash
    from vlm_object_train.train import rng_state,restore_rng
    base=read(read(PARENT/'CONFIG.json')['baseline_config'])
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model,binding=load(base['model_root'],adapter,torch.device('cuda:0'),policy='imagewise_fp32_language',training=training,dropout=0.)
    processor=processor_for(Path(base['model_root']),read(base['reference_config']),1024)
    if training:
        for name,param in model.named_parameters():param.requires_grad_(late.selected(name))
        params=[p for p in model.parameters() if p.requires_grad]
        if len(params)!=20 or sum(p.numel() for p in params)!=PARAMETERS:raise ValueError('Wrong actual rank36 budget')
        if not (ROOT/'INITIALIZATION_GPU_PARITY.json').exists():
            from vlm_fit_positive_control_v1.study import generated
            before=lora_hash(model);saved_rng=rng_state();begun=time.monotonic();model.eval()
            if before!=read(ROOT/'INITIALIZATION.json')['lora_sha256']:raise ValueError('New initializer tensor identity differs')
            data=read(ROOT/'data/MANIFEST.json');rows=data['evaluations']['native_hard'][:8]
            predictions=generated(model,processor,rows,data);expected=read(ROOT/'baseline/native_hard/PREDICTIONS.json')[:8]
            if any((a['question_id'],a['completion_ids'],a['input_sha256'])!=(b['question_id'],b['completion_ids'],b['input_sha256']) for a,b in zip(predictions,expected)):raise ValueError('New rank36 zero-output initialization parity failed')
            if len(predictions)!=8 or lora_hash(model)!=before or time.monotonic()-begun>300 or torch.cuda.max_memory_allocated()/2**30>65:raise ValueError('Initializer parity count/identity/resource')
            restore_rng(saved_rng)
            write_json(ROOT/'INITIALIZATION_GPU_PARITY.json',{'status':'passed','new_generations':8,'optimizer_updates':0,
                       'actual_completion_ids_and_prefixes_equal':True,'lora_hash_unchanged':True,'actual_lora_sha256':before,
                       'active_seconds':time.monotonic()-begun,'predictions':predictions,'scope':'Fixed8 only; not all128/1617 parity or ability gain.'})
        model.train();model.get_base_model().visual.train(True);model.config.use_cache=False
    return model,processor,{**binding,'optimized_scope':'visual_blocks22_31_rank36','optimized_parameters':PARAMETERS}


def worker(kind):
    verify();late.ROOT=ROOT;late.PARAMETERS=PARAMETERS
    if kind=='train':late.load_late=load_new;late.train()
    else:late.evaluate(kind)


def analyze():
    from vlm_grpo_iteration_v1.study import paired
    verify();data=read(ROOT/'data/MANIFEST.json');records={r:{} for r in ('start','full_visual_dpo','late12','ce_dpo')}
    for name in late.INTERNAL+('transfer_dev',):
        phase='transfer' if name=='transfer_dev' else 'internal'
        paths={'start':ROOT/'baseline'/name/'PREDICTIONS.json','full_visual_dpo':PARENT/'runs/ce_dpo'/phase/name/'PREDICTIONS.json',
               'late12':LOW/'runs/ce_dpo'/phase/name/'PREDICTIONS.json','ce_dpo':ROOT/'runs/ce_dpo'/phase/name/'PREDICTIONS.json'}
        frame=[(q['question_id'],q['inference_input_sha256']) for q in data['evaluations'][name]]
        for role,f in paths.items():
            rows=read(f)
            if [(q['question_id'],q['input_sha256']) for q in rows]!=frame:raise ValueError('Evaluation frame changed')
            records[role][name]=rows
    for role,rec in records.items():
        for name in ('singleword','multiword','rare_1_5'):
            ids={q['question_id'] for q in data['evaluations']['seen_validation'] if (1<=q['answer_frequency']<=5 if name=='rare_1_5' else q['stratum']==name)}
            rec[name]=[q for q in rec['seen_validation'] if q['question_id'] in ids]
        ids={q['question_id'] for q in data['evaluations']['transfer_dev'] if 1<=q['answer_frequency']<=5}
        rec['full_rare_1_5']=[q for q in rec['transfer_dev'] if q['question_id'] in ids]
    def norm(rows,name):return [{**q,'image_id':q['filename_family'],'vqa_score':float(q['iiit_alnum_exact'])} for q in rows] if name=='literal_validation' else rows
    metrics={r:{n:100*sum(q['vqa_score'] for q in norm(rows,n))/len(rows) for n,rows in rec.items()} for r,rec in records.items()}
    comparisons={r:{n:paired(norm(records[r][n],n),norm(rows,n)) for n,rows in records['ce_dpo'].items()} for r in ('start','full_visual_dpo','late12')}
    protect=('literal_validation','seen_validation','singleword','multiword','rare_1_5','retention','full_rare_1_5')
    checks={'task_vs_'+r:comparisons[r]['transfer_dev']['delta_pp']>=.5 and comparisons[r]['transfer_dev']['ci95'][0]>0 for r in ('start','full_visual_dpo')}
    checks.update({'point_protections_vs_'+r:all(comparisons[r][n]['delta_pp']>=-.5 for n in protect) for r in ('start','full_visual_dpo')})
    signal=all(checks.values());result={'status':'completed','phase':'pilot_complete_positive_pending_confirmation' if signal else 'pilot_complete_negative',
        'metrics':metrics,'vs_start':{'ce_dpo':comparisons['start']},'candidate_vs_full_dpo':comparisons['full_visual_dpo'],
        'candidate_vs_late12_secondary':comparisons['late12'],'screen_checks':{'ce_dpo':checks},'screen_passed':{'ce_dpo':signal},
        'native_positive_signal':{'ce_dpo':signal},'new_updates':32,'new_generations':2585,'new_reference_forwards':0,
        'goal_complete':False,'automatic_more_training':False,'scope':'Fixed single-seed historical internal, equal trainable count but different rank/scope/geometry; no external or seed-stability claim.'}
    for name in ('SCREEN_ANALYSIS.json','TRANSFER_ANALYSIS.json','COMPLETION.json'):write_json(ROOT/name,result)
    write_json(ROOT/'DECISION.json',{'status':'completed','decision':'internal_signal_pending_independent_confirmation' if signal else 'closed_no_further_late_rank_sweep','new_training_admitted':False,'goal_complete':False})
    render(result)


def render(result=None):
    import html,os
    e=lambda s:html.escape(str(s));data=read(ROOT/'data/MANIFEST.json');cards=[]
    for qid,name in [('21816','native_hard'),('11435','transfer_dev')]:
        q=next(q for q in data['evaluations'][name] if q['question_id']==qid);lines=[]
        for role,label,root in [('start','共同起点',ROOT/'baseline'),('full_visual_dpo','30层r12',PARENT/'runs/ce_dpo'),('late12','10层r12',LOW/'runs/ce_dpo'),('ce_dpo','本轮10层r36',ROOT/'runs/ce_dpo')]:
            f=root/name/'PREDICTIONS.json' if role=='start' else root/('transfer' if name=='transfer_dev' else 'internal')/name/'PREDICTIONS.json'
            rows=read(f) if f.exists() else [];a=next((a for a in rows if a['question_id']==qid),None)
            lines.append(label+'：'+(e(a['prediction'])+f'（{a["vqa_score"]:.1f}）' if a else '待评测'))
        image=data['images'][q['image_id']]['path'];cards.append('<article><img src="'+e(os.path.relpath(image,ROOT))+'"><p>'+e(q['question'])+'</p><p>'+'<br>'.join(lines)+'</p></article>')
    conclusion='初始化/训练与完整评测待完成，尚无新能力结果。' if result is None else '内部主门槛通过，仍待独立确认。' if result['native_positive_signal']['ce_dpo'] else '未通过主门槛，停止后层补参路线，不再扫描rank。'
    table=''
    if result:
        table='<table><tr><th>模型</th><th>完整1617 VQA%</th><th>rare342%</th></tr>'+''.join(f'<tr><td>{e(r)}</td><td>{m["transfer_dev"]:.6f}</td><td>{m["full_rare_1_5"]:.6f}</td></tr>' for r,m in result['metrics'].items())+'</table>'
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>后10层等训练参数DPO</title><style>body{font:16px/1.8 system-ui;max-width:1200px;margin:30px auto;padding:20px;color:#173044;background:#f4f6f8}.cards{display:grid;grid-template-columns:1fr 1fr;gap:20px}article{background:white;padding:20px}img{width:100%;height:380px;object-fit:contain}pre{background:#e8eff3;padding:20px;white-space:pre-wrap}td,th{padding:10px;border:1px solid #ccd5dd}@media(max-width:700px){.cards{grid-template-columns:1fr}}</style><h1>后10层：恢复与强控制相同的训练参数预算</h1><p>'+conclusion+'</p><pre>30层 × r12 = 10层 × r36 → 都训练1843200参数\n每块 A:[36,1280]，B:[3840,36]；alpha72/r36=2\nA前12行原样保留，另24行固定随机初始化；B全0 → 初始有效更新仍0\n相同384QA/512曝光/32步/CE+DPO，前20个原适配块与语言冻结。</pre><p>上一10层r12训练满分26/128（原30层43），完整1617八升七降净−.1分。现在恢复训练预算，检验其与位置干预的混杂；不由单次失败断言唯一病因。rank/参数化几何也变，等训练参数不等于同存储/FLOPs或纯位置因果。</p>'+table+'<div class="cards">'+''.join(cards)+'</div><p>原图等比缩放与实际保存回答；训练/历史internal病例，不回填标签或冒充注意力。</p><p><a href="'+e(os.path.relpath(PROTOCOL,ROOT))+'">冻结协议</a> · <a href="'+e(os.path.relpath(Path(__file__),ROOT))+'">真实源码</a> · <a href="../rarevlm_monitor/状态看板.html">观察台</a></p></html>'
    (ROOT/'后10层等参数DPO.html').write_text(page)
    (PROJECT_ROOT/'docs/VLM_LATE_RANK_MATCHED_DPO_V1_STATUS.md').write_text('# 后10层等训练参数DPO\n\n'+conclusion+'\n\n[协议](VLM_LATE_RANK_MATCHED_DPO_V1_PROTOCOL.md) · [矩阵形状、源码与真实病例](../../../outputs/vlm_late_rank_matched_dpo_v1/后10层等参数DPO.html)\n'+('\n```json\n'+json.dumps({'metrics':result['metrics'],'checks':result['screen_checks']},ensure_ascii=False,indent=2)+'\n```\n' if result else ''))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--worker',choices=('train','internal','transfer'));args=parser.parse_args()
    if args.prepare:prepare();return
    if args.worker:worker(args.worker);return
    from research_runtime.queue import run_queue
    verify();entry=str(Path(__file__).resolve());identity={'study_sha256':sha_file(ROOT/'STUDY_INPUT_LOCK.json')}
    run_queue(ROOT/'training_queue',[{'key':'ce_dpo','command':[sys.executable,'-u',entry,'--worker','train'],'completion_marker':str(ROOT/'runs/ce_dpo/train/TRAIN_COMPLETE.json')}],identity)
    run_queue(ROOT/'transfer_queue',[{'key':phase,'command':[sys.executable,'-u',entry,'--worker',phase],'completion_marker':str(ROOT/'runs/ce_dpo'/(phase+'_COMPLETE.json'))} for phase in ('transfer','internal')],identity)
    analyze()


if __name__=='__main__':main()
