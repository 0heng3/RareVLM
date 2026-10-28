"""Matched native-pixel versus normalized-region SFT; reuse the frozen trainer."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys

from rarevlm_paths import PROJECT_ROOT, SOURCE_ROOT
from vlm_deep.common import NORMALIZER, read_manifest, sha_file, write_json

ROOT=PROJECT_ROOT.parents[1]/'outputs/vlm_region_pixel_sft_v1'
PARENT=PROJECT_ROOT.parents[1]/'outputs/vlm_region_ocr_sft_v1'
COORD=PROJECT_ROOT.parents[1]/'outputs/vlm_region_coordinates_v1'
CONFIG=PROJECT_ROOT/'configs/vlm_region_pixel_sft_v1.json'
ROLES=('normalized','pixel')


def read(path):return json.loads(Path(path).read_text())


def prepare():
    from vlm_region_ocr_sft_v1.study import verify as verify_parent
    verify_parent()
    data=read(ROOT/'DATA_COMPLETE.json')
    if data['cpu_registered_unique_qa']!=4800 or data['new_actual_cpu_encoded_qa']!=3200:
        raise ValueError('New coordinate CPU inputs incomplete')
    if not read(COORD/'COMPLETION.json')['input_evidence_gate_passed']:
        raise ValueError('Native-pixel input evidence gate failed')
    rows={q['source_question_id']:q for q in read_manifest(ROOT/'data/PIXEL_TRAIN.json')['splits']['train_all'] if q['study_source']=='region_aux'}
    for q in read(COORD/'DATA.json')['variants']['pixel_dimensions']:
        if any(rows[q['question_id']][key]!=q[key] for key in ('question','image_id','target')):
            raise ValueError('Fixed probe differs from new pixel training task')
    files=[CONFIG,PROJECT_ROOT/'docs/VLM_REGION_PIXEL_SFT_V1_PROTOCOL.md',
        PROJECT_ROOT/'tools/prepare_region_pixel_sft_v1.py',ROOT/'DATA_COMPLETE.json',
        COORD/'DATA.json',COORD/'COMPLETION.json',COORD/'PLAN.json',
        SOURCE_ROOT/'vlm_region_ocr_sft_v1/train.py',SOURCE_ROOT/'vlm_region_coordinates_v1/data.py']
    files+=list((SOURCE_ROOT/'vlm_region_pixel_sft_v1').glob('*.py'))+list((ROOT/'data').glob('*'))
    for task in ('text','human'):
        files += [PARENT/'initial'/task/name for name in ('predictions.jsonl','MODEL_IDENTITY.json','GENERATION_POLICY.json')]
    parent_lock=read(PARENT/'PRELAUNCH_INPUT_LOCK.json')
    lock={'files':{str(p):sha_file(p) for p in sorted(set(files))},
        'parent_files':parent_lock['files'],'base_weight_files':parent_lock['base_weight_files'],
        'packages':parent_lock['packages'],
        'initial_lora_sha256':read(PARENT/'STUDY_INPUT_LOCK.json')['initial_lora_sha256'],
        'fresh_control':'Normalized coordinates plus identical actual image dimensions',
        'candidate':'Actual processed pixel coordinates plus image dimensions',
        'loss':'Same complete ordinary assistant CE, including termination tokens'}
    if (ROOT/'STUDY_INPUT_LOCK.json').exists():raise FileExistsError('Study already frozen')
    write_json(ROOT/'STUDY_INPUT_LOCK.json',lock)


def verify():
    from vlm_object_train.train import package_lock
    lock=read(ROOT/'STUDY_INPUT_LOCK.json')
    for path,value in {**lock['parent_files'],**lock['files']}.items():
        if sha_file(path)!=value:raise ValueError('Frozen input changed: '+path)
    if package_lock()!=lock['packages']:raise ValueError('Package versions changed')
    for path,value in lock['base_weight_files'].items():
        s=Path(path).stat()
        if {'bytes':s.st_size,'mtime_ns':s.st_mtime_ns}!=value:raise ValueError('Base storage changed')


def probe(model,processor,role):
    import torch
    from vlm_probe.evaluate import evaluate_batch
    from vlm_attention_scope_repair_v1.generation import raw_greedy_config
    data=read(COORD/'DATA.json');rows=data['variants']['pixel_dimensions']
    directory=ROOT/'runs'/role/'pixel_train_probe';directory.mkdir(parents=True,exist_ok=True)
    path=directory/'PREDICTIONS.json';predictions=read(path) if path.exists() else []
    if len(predictions)%4 or [q['question_id'] for q in predictions]!=[q['question_id'] for q in rows[:len(predictions)]]:
        raise ValueError('Fixed probe cache grouping changed')
    model.eval();model.config.use_cache=True
    model.generation_config=raw_greedy_config(model.generation_config,32)
    with torch.inference_mode():
        for start in range(len(predictions),64,4):
            chunk=rows[start:start+4]
            answers,_=evaluate_batch(model,processor,chunk,data['images'],PARENT/'data',
                next(model.parameters()).device,1024,processor.tokenizer.convert_tokens_to_ids('<|image_pad|>'),32)
            for q,a in zip(chunk,answers):
                if a['image_grid_thw']!=q['expected_grid_thw']:raise ValueError('Actual probe frame changed')
                predictions.append({'question_id':q['question_id'],'image_id':q['image_id'],'target':q['target'],**a,
                    'raw_exact':a['prediction']==q['target'],
                    'casefold_exact':a['prediction'].casefold()==q['target'].casefold()})
            write_json(path,predictions)
    write_json(directory/'COMPLETE.json',{'status':'completed','qa':64,
        'scope':'Same native pixel prompt for both Final models; training-only, initial41/64 casefold',
        'raw_exact':sum(q['raw_exact'] for q in predictions),
        'casefold_exact':sum(q['casefold_exact'] for q in predictions)})


def worker(role):
    from vlm_region_ocr_sft_v1.train import train
    from vlm_region_ocr_sft_v1.study import eval_config
    from vlm_attention_scope_repair_v1.generation import evaluate
    from vlm_human_sft.evaluate import lora_hash
    verify()
    model,processor,result=train(read(CONFIG),role,ROOT/'STUDY_INPUT_LOCK.json')
    if lora_hash(model)!=result['final_lora_sha256']:raise ValueError('Final identity changed')
    for task in ('text','human'):evaluate(model,processor,eval_config('screen'),task,ROOT/'runs'/role/task)
    probe(model,processor,role)
    if lora_hash(model)!=result['final_lora_sha256']:raise ValueError('Evaluation changed adapter')
    write_json(ROOT/'runs'/role/'COMPLETE.json',{'status':'completed','updates':200,
        'training_active_seconds':result['active_seconds']})


def analyze():
    from vlm_sft_resolution.experiment import paired_interval
    c=read(CONFIG);records={}
    for role in ('initial',)+ROLES:
        directory=PARENT/'initial' if role=='initial' else ROOT/'runs'/role
        records[role]={task:[json.loads(line) for line in (directory/task/'predictions.jsonl').read_text().splitlines() if line.strip()] for task in ('text','human')}
    gold={task:read_manifest(PARENT/f'data/SCREEN_{task.upper()}.json')['splits']['dev'] for task in ('text','human')}
    for tasks in records.values():
        for task in gold:
            if [(q['question_id'],q['image_id']) for q in tasks[task]]!=[(q['question_id'],q['image_id']) for q in gold[task]]:
                raise ValueError('Unpaired screen predictions')
    frequency=Counter(NORMALIZER(q['target']) for q in read_manifest(c['source_manifest'])['splits']['train_all'])
    components={p[k]:p['component_id'] for p in read(PARENT/'data/SCREEN_PAIRS.json') for k in ('positive_question_id','negative_question_id')}
    comparisons={};criteria={};changed={}
    for before in ('initial','normalized'):
        comparisons[before]={};changed[before]={}
        for task in ('text','human'):
            changed[before][task]=[{'question_id':a['question_id'],'image_id':a['image_id'],
                'before':a['prediction'],'after':b['prediction'],'before_score':a['primary_score'],'after_score':b['primary_score']}
                for a,b in zip(records[before][task],records['pixel'][task]) if a['prediction']!=b['prediction']]
            masks={'overall':[True]*len(gold[task])}
            if task=='text':
                masks.update(rare_1_5=[1<=frequency[NORMALIZER(q['target'])]<=5 for q in gold[task]],
                    frequent_gt20=[frequency[NORMALIZER(q['target'])]>20 for q in gold[task]])
            else:masks.update({label:[q['target']==label for q in gold[task]] for label in ('yes','no')})
            for name,mask in masks.items():
                def transform(role):
                    return [{'question_id':q['question_id'],'image_id':components[q['question_id']] if task=='human' else q['image_id'],
                        'score':q['primary_score']} for q,keep in zip(records[role][task],mask) if keep]
                left,right=transform(before),transform('pixel')
                value=paired_interval(left,right,c['bootstrap_replicates'],c['bootstrap_seed'])
                value.update(corrected_qa=sum(b['score']>a['score'] for a,b in zip(left,right)),
                    broken_qa=sum(b['score']<a['score'] for a,b in zip(left,right)),
                    net_score_points=sum(b['score']-a['score'] for a,b in zip(left,right)))
                key=task+'_'+name;comparisons[before][key]=value
                gain=task=='text' and name in ('overall','rare_1_5')
                criteria[before+'_'+key]=value['delta_pp']>=(c['minimum_gain_pp'] if gain else -c['protection_margin_pp'])
                if task=='text' and name=='overall':criteria[before+'_positive_text_ci']=value['image_cluster_95_ci_pp'][0]>0
    result={'status':'completed','stage':'screen','passed':all(criteria.values()),'criteria':criteria,'comparisons':comparisons,
        'scores_percent':{role:{task:100*sum(q['primary_score'] for q in rows)/len(rows) for task,rows in tasks.items()} for role,tasks in records.items()},
        'training_probe':{role:read(ROOT/'runs'/role/'pixel_train_probe/COMPLETE.json') for role in ROLES},
        'scope':'Single seed42; historically exposed internal screen; input gain is separate from weight capability',
        'next':'Stop if failed; independent confirmation and replication required if passed; no automatic more training'}
    write_json(ROOT/'CHANGED_CASES.json',changed)
    write_json(ROOT/'SCREEN_ANALYSIS.json',result);write_json(ROOT/'COMPLETION.json',result)
    lines=['# 实际像素坐标SFT结果','',f'联合筛选通过：{result["passed"]}。','',
        '|模型|Text512|Human512|','|---|---:|---:|']
    lines += [f'|{role}|{value["text"]:.6f}|{value["human"]:.6f}|' for role,value in result['scores_percent'].items()]
    lines += ['',result['scope'],'','完整配对区间、rare、保持指标和升降见对应输出目录。']
    (PROJECT_ROOT/'docs/VLM_REGION_PIXEL_SFT_V1_RESULTS.md').write_text('\n'.join(lines)+'\n')
    page=PROJECT_ROOT/'docs/learning/区域坐标修复_训练与验证.html'
    if page.exists():
        block='<h2>新匹配训练最终结果</h2><table><tr><th>模型</th><th>Text512</th><th>Human512</th></tr>'+''.join(
            f'<tr><td>{role}</td><td>{v["text"]:.6f}</td><td>{v["human"]:.6f}</td></tr>' for role,v in result['scores_percent'].items())+'</table><p>联合门槛通过：'+str(result['passed'])+'；单seed探索，仍需按协议确认。</p>'
        page.write_text(re.sub(r'<!--MODEL_RESULT_START-->.*?<!--MODEL_RESULT_END-->',
            '<!--MODEL_RESULT_START-->'+block+'<!--MODEL_RESULT_END-->',page.read_text(),flags=re.S))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--role',choices=ROLES)
    args=parser.parse_args()
    if args.prepare:prepare();return
    if args.role:
        try:worker(args.role)
        except Exception as exc:
            write_json(ROOT/'runs'/args.role/'ERROR.json',{'status':'failed','error':repr(exc),'other_tasks_untouched':True})
            raise
        return
    verify()
    from research_runtime.queue import run_queue
    run_queue(ROOT/'queue',[{'key':role,'command':[sys.executable,'-u','-m',__package__+'.study','--role',role],
        'completion_marker':str(ROOT/'runs'/role/'COMPLETE.json')} for role in ROLES],
        {'study_sha256':sha_file(ROOT/'STUDY_INPUT_LOCK.json')})
    analyze()


if __name__=='__main__':main()
