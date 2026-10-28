"""New official supervision, matched-parameter scopes, fixed Final64/test3000."""
import argparse
import html
import os
from pathlib import Path
import re
import sys
import time
from urllib.parse import quote

from rarevlm_paths import PROJECT_ROOT
from vlm_deep.common import sha_file,write_json
from vlm_grpo_iteration_v1.common import actor,read
from vlm_iiit_scope_v1.initialization import ROOT
from vlm_iiit_scope_v1 import data as dataset
from vlm_iiit_scope_v1.train import train
from run_literal_tile_v1 import generate as literal_generate
from vlm_fit_positive_control_v1.study import generated
import run_literal_case_sft_v1 as core

core.ROOT=ROOT
PROTOCOL=PROJECT_ROOT/'docs/VLM_IIIT_SCOPE_V1_PROTOCOL.md'
RESULT=PROJECT_ROOT/'docs/VLM_IIIT_SCOPE_V1_RESULTS.md'
PAGE=PROJECT_ROOT/'docs/learning/场景文字新数据_语言与视觉LoRA对照.html'
INTERNAL=('hard','literal_validation','seen_validation','retention')
OCR={'hard','literal_validation','official_test'}


def alphanumeric(text):
    return re.sub('[^a-z0-9]','',dataset.canonical(text))


def score(q,name):
    return float(q['iiit_alnum_exact']) if name in OCR else q['vqa_score']


def evaluate(model,processor,data,directory,names,data_sha256,shard=None):
    import torch
    from vlm_human_sft.evaluate import lora_hash
    from vlm_object_train.train import lock_json
    model.eval();model.config.use_cache=True
    before=lora_hash(model);lock_json(directory/'INPUT_LOCK.json',{'plan_sha256':sha_file(ROOT/'PLAN.json'),'data_sha256':data_sha256,'model_lora_sha256':before,'shard':shard})
    started=time.monotonic();cap=2400 if 'official_test' in names else 1500
    for name in names:
        rows=data['evaluations'][name];groups=[rows[i:i+4] for i in range(0,len(rows),4)]
        if shard is not None:groups=groups[shard::2]
        ordered=[q for group in groups for q in group];path=directory/name/'PREDICTIONS.json';preds=read(path) if path.exists() else []
        if len(preds)>len(ordered) or (len(preds)%4 and len(preds)!=len(ordered)) or [q['question_id'] for q in preds]!=[q['question_id'] for q in ordered[:len(preds)]]:
            raise ValueError('Whole-batch evaluation resume differs')
        for group in groups[len(preds)//4:]:
            answers=literal_generate(model,processor,group,data) if name in OCR else generated(model,processor,group,data)
            if name in OCR:
                for q,a in zip(group,answers):
                    a['iiit_alnum_exact']=alphanumeric(a['prediction'])==alphanumeric(q['target'])
                    a['filename_family']=q['filename_family'];a['sft_exact_answer_frequency']=q['sft_exact_answer_frequency']
            preds.extend(answers);tmp=path.with_suffix('.tmp');write_json(tmp,preds);tmp.replace(path)
            write_json(directory/'EVAL_PROGRESS.json',{'status':'evaluating','cohort':name,'completed_qa':len(preds),'expected_qa':len(ordered),'active_seconds':time.monotonic()-started})
            if time.monotonic()-started>cap:raise TimeoutError('Evaluation active cap')
            if torch.cuda.max_memory_allocated()/2**30>65:raise RuntimeError('Evaluation memory cap65GiB')
    if lora_hash(model)!=before:raise ValueError('Evaluation changed adapter')


def baseline(shard):
    from vlm_human_sft.evaluate import lora_hash
    plan=core.verify();initial=read(ROOT/'INITIALIZATION.json');model,processor,_=actor(adapter=initial['adapter'])
    if lora_hash(model)!=plan['initial_lora_sha256']:raise ValueError('Wrong common baseline')
    directory=ROOT/'baseline_shards'/str(shard);evaluate(model,processor,read(ROOT/'DATA.json'),directory,INTERNAL,sha_file(ROOT/'DATA.json'),shard)
    write_json(directory/'COMPLETE.json',{'status':'completed','shard':shard,'generations':480,'optimizer_updates':0})


def merge_baseline():
    data=read(ROOT/'DATA.json')
    for name in INTERNAL:
        rows=[q for shard in (0,1) for q in read(ROOT/'baseline_shards'/str(shard)/name/'PREDICTIONS.json')]
        byid={q['question_id']:q for q in rows};ordered=data['evaluations'][name]
        if len(rows)!=len(ordered) or set(byid)!={q['question_id'] for q in ordered}:raise ValueError('Incomplete baseline shards')
        write_json(ROOT/'baseline'/name/'PREDICTIONS.json',[byid[q['question_id']] for q in ordered])
    write_json(ROOT/'baseline/COMPLETE.json',{'status':'completed','generations':960,'optimizer_updates':0})
    render()


def condition(role):
    from vlm_human_sft.evaluate import lora_hash
    plan=core.verify();model,processor,result=train(role,plan)
    if lora_hash(model)!=result['final_lora_sha256']:raise ValueError('Wrong Final64')
    evaluate(model,processor,read(ROOT/'DATA.json'),ROOT/'runs'/role,INTERNAL,sha_file(ROOT/'DATA.json'))
    write_json(ROOT/'runs'/role/'COMPLETE.json',{'status':'completed','role':role,'updates':64,'final_lora_sha256':result['final_lora_sha256']})


def test(role):
    from vlm_human_sft.evaluate import lora_hash
    plan=core.verify();binding=read(ROOT/'TEST_PLAN.json')
    if sha_file(ROOT/'TEST_DATA.json')!=binding['data_sha256']:raise ValueError('Official test metadata changed')
    for path,digest in binding['test_images_sha256'].items():
        if sha_file(path)!=digest:raise ValueError('Official test input changed')
    if role=='initial':adapter=read(ROOT/'INITIALIZATION.json')['adapter'];expected=plan['initial_lora_sha256']
    else:
        result=read(ROOT/'runs'/role/'train/TRAIN_COMPLETE.json');adapter=result['adapter'];expected=binding['all_models_frozen_before_test_open'][role]
    model,processor,_=actor(adapter=adapter)
    if lora_hash(model)!=expected:raise ValueError('Test model differs from frozen configuration')
    evaluate(model,processor,read(ROOT/'TEST_DATA.json'),ROOT/'official_test'/role,('official_test',),binding['data_sha256'])
    write_json(ROOT/'official_test'/role/'COMPLETE.json',{'status':'completed','role':role,'generations':3000,'model_lora_sha256':expected})


def analyze(final=False):
    from vlm_grpo_iteration_v1.study import paired
    data=read(ROOT/'DATA.json');roles=['initial']+read(ROOT/'PLAN.json')['roles'];records={};names=list(INTERNAL)+['supported_validation']
    for role in roles:
        directory=ROOT/'baseline' if role=='initial' else ROOT/'runs'/role
        records[role]={n:read(directory/n/'PREDICTIONS.json') for n in INTERNAL}
        records[role]['supported_validation']=[q for q in records[role]['literal_validation'] if q['question_id'] in set(data['supported_validation_qids'])]
        if final:records[role]['official_test']=read(ROOT/'official_test'/role/'official_test/PREDICTIONS.json')
    if final:names+=['official_test']
    ocr=OCR|{'supported_validation'}
    def value(q,n):return float(q['iiit_alnum_exact']) if n in ocr else q['vqa_score']
    metrics={r:{n:{'qa':len(rows),'score_percent':100*sum(value(q,n) for q in rows)/len(rows),'full':sum(value(q,n)>=.999999 for q in rows)} for n,rows in v.items()} for r,v in records.items()}
    def normalize(rows,n):return [{**q,'image_id':q['filename_family'] if n in ocr else q['image_id'],'vqa_score':value(q,n)} for q in rows]
    def pair(a,b,n):return paired(normalize(records[a][n],n),normalize(records[b][n],n))
    changes={r:{n:pair('initial',r,n) for n in names} for r in roles[1:]}
    matched={'coverage':{n:pair('sparse_language','dense_language',n) for n in names},'scope':{n:pair('dense_language','dense_visual',n) for n in names}}
    slices={};native_meta={q['question_id']:q for q in data['evaluations']['seen_validation']}
    for role in roles[1:]:
        slices[role]={}
        for group in ['singleword','multiword','rare_1_5']:
            ids={qid for qid,q in native_meta.items() if (q['stratum']==group if group!='rare_1_5' else q.get('answer_frequency',100)<=5)}
            slices[role][group]=paired([q for q in records['initial']['seen_validation'] if q['question_id'] in ids],[q for q in records[role]['seen_validation'] if q['question_id'] in ids])
    result={'status':'completed','phase':'official_test_complete' if final else 'internal_complete_test_pending','metrics':metrics,'vs_initial':changes,'matched_comparisons':matched,'native_slices':slices,
        'new_optimizer_updates':192,'new_training_qa_exposures':4608,'goal_complete':False,'official_test_used_for_configuration':False,
        'scope':'Fixed single-seed coverage/scope contrast;official3000 test first opened after all Final64.Conditional family/image-cluster intervals;multiple endpoints exploratory,not seed-stable or pretraining-unseen evidence.'}
    if final:
        result['local_ocr_gain_and_point_protection']={r:changes[r]['official_test']['delta_pp']>=.5 and changes[r]['official_test']['ci95'][0]>0 and
            changes[r]['seen_validation']['delta_pp']>=-.5 and changes[r]['retention']['delta_pp']>=-.5 and all(v['delta_pp']>=-.5 for v in slices[r].values()) for r in roles[1:]}
        result['native_statistical_noninferiority']={r:changes[r]['seen_validation']['ci95'][0]>-.5 for r in roles[1:]}
        result['joint_positive_signal']={r:result['local_ocr_gain_and_point_protection'][r] and changes[r]['seen_validation']['delta_pp']>=.5 and changes[r]['seen_validation']['ci95'][0]>0 for r in roles[1:]}
        result['coverage_advantage']=matched['coverage']['official_test']['delta_pp']>=.5 and matched['coverage']['official_test']['ci95'][0]>0
        result['visual_scope_advantage']=matched['scope']['official_test']['delta_pp']>=.5 and matched['scope']['official_test']['ci95'][0]>0
        testmeta={q['question_id']:q for q in read(ROOT/'TEST_DATA.json')['evaluations']['official_test']};result['test_frequency_slices']={}
        for group in ['zero','rare_1_5','more_than5','extra_train_exact_target_unseen']:
            ids={qid for qid,q in testmeta.items() if (not q['extra_train_exact_target_seen'] if group=='extra_train_exact_target_unseen' else q['sft_exact_answer_frequency']==0 if group=='zero' else 1<=q['sft_exact_answer_frequency']<=5 if group=='rare_1_5' else q['sft_exact_answer_frequency']>5)}
            if ids:result['test_frequency_slices'][group]={r:paired(normalize([q for q in records['initial']['official_test'] if q['question_id'] in ids],'official_test'),normalize([q for q in records[r]['official_test'] if q['question_id'] in ids],'official_test')) for r in roles[1:]}
    write_json(ROOT/('ANALYSIS.json' if final else 'INTERNAL_ANALYSIS.json'),result)
    if final:write_json(PROJECT_ROOT/'results/vlm_iiit_scope_v1_metrics.json',result)
    render(result);return result


def render(result=None):
    data=read(ROOT/'DATA.json');plan=read(ROOT/'PLAN.json');meta={q['question_id']:q for q in data['evaluations']['hard']};link=lambda p:quote(os.path.relpath(p,PAGE.parent),safe='/._-')
    state='数据/同参数初始化已准备；新质量结果尚待实际训练与完整测试。';table=''
    baseline=ROOT/'baseline/COMPLETE.json'
    if baseline.exists():
        rows=read(ROOT/'baseline/literal_validation/PREDICTIONS.json');state=f'内部256字母数字转写初始{sum(q["iiit_alnum_exact"] for q in rows)}/256；此处不是官方test起点。'
    if result:
        state+=f' 当前阶段：{result["phase"]}；项目整体目标尚未完成。'
        test_ready='official_test' in result['metrics']['initial']
        table='|模型|共同训练探针/128|内部256%|支持子集%|native512%|保持64%'+('|官方test3000%|' if test_ready else '|')+'\n|---|---:|---:|---:|---:|---:'+('|---:|' if test_ready else '|')+'\n'
        for role,m in result['metrics'].items():
            table+='|'+role+'|'+str(m['hard']['full'])+'|'+'|'.join(f'{m[n]["score_percent"]:.6f}' for n in ['literal_validation','supported_validation','seen_validation','retention'])+ ('|'+f'{m["official_test"]["score_percent"]:.6f}' if test_ready else '')+'|\n'
        if test_ready:state+=' 新域局部收益/点值保护='+str(result['local_ocr_gain_and_point_protection'])+'；联合正向信号='+str(result['joint_positive_signal'])+'。'
    RESULT.write_text('# 新官方监督：数据覆盖与视觉/语言适配\n\n'+state+'\n\n'+table+'\n[协议](VLM_IIIT_SCOPE_V1_PROTOCOL.md) · [真实图像与算法源码](learning/场景文字新数据_语言与视觉LoRA对照.html)\n')
    cards=[]
    for qid in plan['preview_qids']:
        q=meta[qid];answers=[]
        for role in ['initial']+plan['roles']:
            directory=ROOT/'baseline' if role=='initial' else ROOT/'runs'/role;path=directory/'hard/PREDICTIONS.json'
            if path.exists():
                a=next((a for a in read(path) if a['question_id']==qid),None)
                if a:answers.append(role+': '+html.escape(a['prediction'])+' / '+str(a['iiit_alnum_exact']))
        cards.append('<article><h2>'+html.escape(q['target'])+'</h2><img src="'+link(data['images'][q['image_id']]['path'])+'" alt="实际词图输入"><p>原尺寸 '+str(q['original_crop_size'])+' → 输入尺寸 '+str(q['actual_input_size'])+'；原SFT整答案曝光 '+str(q['sft_exact_answer_frequency'])+'。</p><p>'+'；'.join(answers)+'</p></article>')
    PAGE.write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>新数据与视觉LoRA</title><style>body{max-width:1080px;margin:30px auto;padding:0 18px;font:17px/1.7 system-ui}article{border:1px solid #cbd5e1;padding:16px;margin:18px 0}img{max-width:100%;width:480px}pre{white-space:pre-wrap;background:#f1f5f9;padding:16px}</style><h1>从反复纠错，转向数据覆盖和视觉表征的对照</h1><p>'+html.escape(state)+'</p><pre>sparse_language: 128不同词×8曝光；语言q/v rank8\ndense_language:  512不同词×2曝光；语言q/v rank8\ndense_visual:    同512词×2曝光；视觉blocks2–31 qkv rank12\n每组：64更新，1024 OCR+512 native QA，更新1843200参数\nL = .5 mean(OCR逐QA token均值CE) + .5 mean(native逐QA token均值CE)\nΔW = (alpha/r) B A；初始化B=0，前向增量为0\n视觉A[12,1280]、B[3840,12]，30模块共1843200参数</pre><p>覆盖对照保持分层槽位、native回放与更新预算；scope对照保持512图，改更新范围。视觉侧更靠近图像特征，语言侧改变答案条件生成，但必须看结果才能判断哪一侧更有效。等参数不等于等FLOPs或纯位置因果。</p><p>IIIT任务按官方alphanumeric口径：BT-Pusher→btpusher、$2.00→200，保留原始输出及严格字面exact。它不是价格数值理解或TextOCR完整字面任务；native VQA规则不改。全部官方test3000仅在三个Final64保存后打开，固定四模型统一测试，不挑checkpoint。公开预训练重叠未知，filename-family只是保守分组。</p><p>内部256全部保留，213助手支持子集另报；审核不等于专家真值。下面8图在输出前固定，为三组共同训练探针，不当泛化。页面缩放不增加原生细节。未实际运行完成的指标不填写，不自动长训。</p><p><a href="'+link(PROTOCOL)+'">协议/判定</a> · <a href="'+link(Path(__file__).resolve())+'">运行与分析源码</a> · <a href="'+link(PROJECT_ROOT/'src/vlm_iiit_scope_v1/train.py')+'">训练与梯度scope</a> · <a href="'+link(ROOT/'AUDIT_REVIEW.json')+'">预测前审核</a></p>'+''.join(cards)+'<pre>'+html.escape(table)+'</pre></html>')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--worker');args=parser.parse_args()
    if args.prepare:dataset.prepare();render();return
    if args.worker:
        if args.worker.startswith('baseline:'):baseline(int(args.worker.split(':')[1]))
        elif args.worker.startswith('test:'):test(args.worker.split(':')[1])
        else:condition(args.worker)
        return
    from research_runtime.queue import run_queue
    plan=core.verify();identity={'plan_sha256':sha_file(ROOT/'PLAN.json')};entry=str(Path(__file__).resolve())
    job=lambda key,arg,marker:{'key':key,'command':[sys.executable,'-u',entry,'--worker',arg],'completion_marker':str(marker)}
    run_queue(ROOT/'baseline_queue',[job('baseline_'+str(i),'baseline:'+str(i),ROOT/'baseline_shards'/str(i)/'COMPLETE.json') for i in (0,1)],identity);merge_baseline()
    run_queue(ROOT/'training_queue',[job(r,r,ROOT/'runs'/r/'COMPLETE.json') for r in plan['roles']],identity);analyze()
    dataset.prepare_test()
    run_queue(ROOT/'test_queue',[job('test_'+r,'test:'+r,ROOT/'official_test'/r/'COMPLETE.json') for r in ['initial']+plan['roles']],identity)
    result=analyze(True);write_json(ROOT/'COMPLETION.json',result);print('IIIT_SCOPE_COMPLETE',result,flush=True)


if __name__=='__main__':main()
