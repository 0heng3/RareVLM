"""Fresh native-QA correction supervision candidates; no optimizer."""
import argparse
from collections import Counter
import html
import os
from pathlib import Path
import sys
import time

from rarevlm_paths import PROJECT_ROOT
from vlm_deep.common import NORMALIZER, content_sha, encode, sha_file, write_json
from vlm_grpo_iteration_v1.common import read
from vlm_native_scope_v1.model import ROOT as INITIAL_STUDY, load

BASE = PROJECT_ROOT.parents[1] / 'outputs'
ROOT = BASE / 'vlm_native_diversity_mining_v1'
PROTOCOL = PROJECT_ROOT / 'docs/VLM_NATIVE_DIVERSITY_MINING_V1_PROTOCOL.md'
SOURCE = BASE / 'vlm_visual_preference_v1'
MANIFEST = BASE / 'vlm_deep_v2/prepared/manifest.json'
COUNTS = {'multiword': 224, 'singleword': 2848}


def walk(value):
    if isinstance(value, dict):
        if 'question_id' in value and 'image_id' in value:
            yield value
        else:
            for child in value.values():
                yield from walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk(child)


def prepare():
    import torch
    from vlm_probe.evaluate import inference_row, processor_for
    if (ROOT / 'PLAN.json').exists():
        raise FileExistsError('Mining already frozen')
    exclusions = [BASE / name for name in ['vlm_iiit_scope_v1/DATA.json', 'vlm_replay_diversity_v1/DATA.json',
        'vlm_error_supervision_mining_v1/DATA.json', 'vlm_native_ocr_correction_v1/revision2/DATA.json',
        'vlm_native_correction_mining_v2/DATA.json', 'vlm_native_correction_supplement_v2/DATA.json',
        'vlm_visual_preference_v1/data/MANIFEST.json']]
    config = read(SOURCE / 'CONFIG.json'); adapter = Path(config['initial_adapter'])
    inputs = [Path(__file__), PROTOCOL, MANIFEST, SOURCE / 'CONFIG.json', SOURCE / 'COMPLETION.json',
        SOURCE / 'STUDY_INPUT_LOCK.json', INITIAL_STUDY / 'INITIALIZATION.json',
        PROJECT_ROOT / 'src/vlm_native_scope_v1/model.py', PROJECT_ROOT / 'src/vlm_batch_stable_v1/model.py',
        adapter / 'adapter_config.json', adapter / 'adapter_model.safetensors', *exclusions]
    if not (ROOT / 'SELECTION.json').exists():
        manifest = read(MANIFEST); blocked = {q['image_id'] for fold in ('dev', 'final') for q in manifest['splits'][fold]}
        for path in exclusions:
            blocked.update(q.get('source_image_id', q['image_id']) for q in walk(read(path)))
        pixels = {manifest['images'][i]['pixel_sha256'] for i in blocked if i in manifest['images']}
        prior_targets = {NORMALIZER(q['target']) for q in read(exclusions[-1])['rows']}
        frequencies = Counter(NORMALIZER(q['target']) for q in manifest['splits']['train_all'])
        groups = {name: [] for name in COUNTS}
        for q in manifest['splits']['train_all']:
            target = NORMALIZER(q['target'])
            if q['image_id'] in blocked or manifest['images'][q['image_id']]['pixel_sha256'] in pixels:
                continue
            if not target or target in prior_targets or target in {'yes','no','unknown','unanswerable','unreadable'} or len(q['target']) > 64 or len(q['target'].split()) > 6:
                continue
            votes = Counter(NORMALIZER(a) for a in q['answers'])[target]
            if votes < 8:
                continue
            group = 'singleword' if len(q['target'].split()) == 1 else 'multiword'
            groups[group].append({**q, 'stratum': group, 'answer_frequency': frequencies[target], 'target_consensus_votes': votes})
        selected = []; used = set(); targets = Counter()
        for group, count in COUNTS.items():
            chosen = []
            for q in sorted(groups[group], key=lambda q: content_sha(['native_diversity_mining_v1',q['question_id']])):
                pixel = manifest['images'][q['image_id']]['pixel_sha256']; target = NORMALIZER(q['target'])
                if pixel in used or targets[target] >= 2:
                    continue
                chosen.append(q); used.add(pixel); targets[target] += 1
                if len(chosen) == count:
                    break
            if len(chosen) != count:
                raise ValueError('Insufficient fixed quota; no relaxation: ' + group)
            selected += chosen
        images = {q['image_id']: {**manifest['images'][q['image_id']], 'path': str((MANIFEST.parent / manifest['images'][q['image_id']]['path']).resolve())} for q in selected}
        write_json(ROOT / 'SELECTION.json', {'rows': selected, 'images': images, 'counts': COUNTS, 'blocked_images': len(blocked), 'excluded_training_targets': len(prior_targets), 'eligible_rows': {k:len(v) for k,v in groups.items()}})
        write_json(ROOT / 'SELECTION_LOCK.json', {str(p):sha_file(p) for p in inputs + [ROOT / 'SELECTION.json']})
    for path,digest in read(ROOT / 'SELECTION_LOCK.json').items():
        if sha_file(path) != digest:
            raise ValueError('Selection source changed: ' + path)
    selection = read(ROOT / 'SELECTION.json'); torch.set_num_threads(2)
    base = read(config['baseline_config']); processor = processor_for(Path(base['model_root']),read(base['reference_config']),1024)
    rows = read(ROOT / 'CPU_ROWS_PARTIAL.json') if (ROOT / 'CPU_ROWS_PARTIAL.json').exists() else []
    if [q['question_id'] for q in rows] != [q['question_id'] for q in selection['rows'][:len(rows)]]:
        raise ValueError('CPU resume order changed')
    started = time.monotonic()
    for index in range(len(rows), len(selection['rows'])):
        q = dict(selection['rows'][index]); sample = encode(processor,inference_row(q),Path(selection['images'][q['image_id']]['path']),False)
        if sample['input_ids'].shape[1] + 32 > 2048:
            raise ValueError('Prefix exceeds registered sequence budget')
        q.update(expected_prefix_tokens=int(sample['input_ids'].shape[1]),expected_grid=sample['image_grid_thw'][0].tolist(),inference_input_sha256=content_sha(sample['input_ids'].tolist())); rows.append(q)
        if len(rows)%64 == 0 or len(rows) == len(selection['rows']):
            write_json(ROOT / 'CPU_ROWS_PARTIAL.json',rows); write_json(ROOT / 'CPU_PROGRESS.json',{'status':'preparing_inputs','completed':len(rows),'expected':3072,'active_seconds_this_process':time.monotonic()-started,'optimizer_updates':0})
    write_json(ROOT / 'DATA.json',{'rows':rows,'images':selection['images']})
    files = {**read(ROOT / 'SELECTION_LOCK.json'),str(ROOT / 'SELECTION_LOCK.json'):sha_file(ROOT / 'SELECTION_LOCK.json'),str(ROOT / 'DATA.json'):sha_file(ROOT / 'DATA.json')}
    write_json(ROOT / 'PLAN.json',{'status':'frozen','files':files,'qa':3072,'counts':COUNTS,'shards':2,'qa_per_shard':1536,
        'initial_adapter':str(adapter),'initial_lora_sha256':read(INITIAL_STUDY / 'INITIALIZATION.json')['lora_sha256'],
        'images':{v['path']:v['sha256'] for v in selection['images'].values()},'new_optimizer_updates':0,'new_generations':3072,
        'automatic_training':False,'minimum_new_reviewed_errors':64,'minimum_new_unique_targets':64,'review_cap':256,'cap_seconds_per_shard':1800})
    print('DIVERSITY_MINING_PREPARED',{'qa':3072,'counts':COUNTS,'new_optimizer_updates':0},flush=True)


def verify():
    plan = read(ROOT / 'PLAN.json')
    for path,digest in plan['files'].items():
        if sha_file(path) != digest:
            raise ValueError('Frozen mining input changed: ' + path)
    return plan


def worker(shard):
    import torch
    from vlm_fit_positive_control_v1.study import generated
    from vlm_human_sft.evaluate import lora_hash
    plan = verify(); data = read(ROOT / 'DATA.json'); groups = [data['rows'][i:i+4] for j,i in enumerate(range(0,3072,4)) if j%2 == shard]
    ordered = [q for group in groups for q in group]
    for q in ordered:
        path = data['images'][q['image_id']]['path']
        if sha_file(path) != plan['images'][path]:
            raise ValueError('New selected image changed')
    model,processor,binding = load(plan['initial_adapter'],'visual',False); model.eval(); model.requires_grad_(False); model.config.use_cache = True; before = lora_hash(model)
    if before != plan['initial_lora_sha256']:
        raise ValueError('Wrong shared composite initializer')
    directory = ROOT / f'shard_{shard}'; path = directory / 'PREDICTIONS.json'; predictions = read(path) if path.exists() else []
    if len(predictions)>1536 or len(predictions)%4 or [q['question_id'] for q in predictions] != [q['question_id'] for q in ordered[:len(predictions)]]:
        raise ValueError('Mining resume differs')
    started = time.monotonic(); torch.cuda.reset_peak_memory_stats()
    for group in groups[len(predictions)//4:]:
        answers = generated(model,processor,group,data)
        if any(a['input_sha256'] != q['inference_input_sha256'] for a,q in zip(answers,group)):
            raise ValueError('Actual mining prefix changed')
        predictions += answers; temporary = path.with_suffix('.tmp'); write_json(temporary,predictions); temporary.replace(path)
        write_json(directory / 'PROGRESS.json',{'status':'mining','completed':len(predictions),'expected':1536,'active_seconds':time.monotonic()-started,'optimizer_updates':0,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30})
        if time.monotonic()-started>1800 or torch.cuda.max_memory_allocated()/2**30>65:
            raise RuntimeError('Mining budget exceeded')
    if lora_hash(model) != before:
        raise ValueError('Read-only mining changed weights')
    write_json(directory / 'COMPLETE.json',{'status':'completed','qa':1536,'optimizer_updates':0,'active_seconds':time.monotonic()-started,'model_hash_unchanged':True,'binding':binding})


def analyze():
    data = read(ROOT / 'DATA.json'); predictions = {q['question_id']:q for s in (0,1) for q in read(ROOT / f'shard_{s}/PREDICTIONS.json')}
    if len(predictions) != 3072:
        raise ValueError('Incomplete mining predictions')
    errors = []; counts = Counter()
    for q in sorted(data['rows'],key=lambda q:content_sha(['native_diversity_mining_v1:errors',q['question_id']])):
        p = predictions[q['question_id']]; target = NORMALIZER(q['target'])
        if not p['terminated'] or p['vqa_score']>.3+1e-8 or target in NORMALIZER(p['prediction']) or counts[target]>=2:
            continue
        errors.append({**q,'initial_prediction':p}); counts[target] += 1
    write_json(ROOT / 'ERROR_CANDIDATES.json',{'rows':errors,'images':data['images']})
    cards = []
    for index,q in enumerate(errors[:256]):
        path = Path(data['images'][q['image_id']]['path']); e = lambda x:html.escape(str(x))
        cards.append(f'<article><h3>#{index} QA {e(q["question_id"])}</h3><img src="{e(os.path.relpath(path,ROOT))}"><p>{e(q["question"])}</p><p>GT: {e(q["target"])}<br>Pred: {e(q["initial_prediction"]["prediction"])}</p></article>')
    (ROOT / '候选原图审核.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>新纠错候选，尚未准入训练</title><style>body{font:16px/1.7 system-ui;margin:30px}.grid{display:grid;grid-template-columns:repeat(2,1fr);gap:20px}img{width:100%;height:430px;object-fit:contain}article{border:1px solid #ccd5dd;padding:16px}</style><h1>原图审核：低VQA不等于语义错误</h1><p>仅候选；需助手非专家原图复核，语义等价/题义歧义/证据不足不准入，不改GT或分数。</p><div class="grid">'+''.join(cards)+'</div></html>')
    write_json(ROOT / 'COMPLETION.json',{'status':'completed','new_optimizer_updates':0,'new_generations':3072,'mined_qa':3072,'candidate_errors':len(errors),'candidate_unique_targets':len(counts),'strata':dict(Counter(q['stratum'] for q in errors)),'training_admitted':False,'visual_review_completed':False,'goal_complete':False,'automatic_training':False})


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare',action='store_true'); parser.add_argument('--prepare-run',action='store_true'); parser.add_argument('--worker',type=int,choices=[0,1]); args = parser.parse_args()
    if args.prepare or (args.prepare_run and not (ROOT / 'PLAN.json').exists()):
        prepare()
        if args.prepare:
            return
    if args.worker is not None:
        worker(args.worker); return
    from research_runtime.queue import run_queue
    verify(); entry = str(Path(__file__).resolve()); run_queue(ROOT / 'queue',[{'key':str(s),'command':[sys.executable,'-u',entry,'--worker',str(s)],'completion_marker':str(ROOT / f'shard_{s}/COMPLETE.json')} for s in (0,1)],{'plan_sha256':sha_file(ROOT / 'PLAN.json')}); analyze()


if __name__ == '__main__':
    main()
