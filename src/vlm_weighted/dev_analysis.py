"""CPU-only paired image-cluster analysis and predeclared internal-dev pilot gate."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from rarevlm_paths import PROJECT_ROOT, SOURCE_ROOT
import statistics
import sys

sys.dont_write_bytecode = True
from vlm_a100.evaluate import validate_prediction
from vlm_deep.common import NORMALIZER, training_target
from vlm_formal.evaluate import question_prompt_sha
from vlm_probe.evaluate import content_sha, lock_file, now, private_path, read_json, read_manifest, sha_file, write_json
from .evaluate import CONDITIONS, DEFAULT_CONFIG, dev_data, model_key, source_hashes

REPO = PROJECT_ROOT
LABELS = ('unseen_0', 'rare_1_5', 'medium_6_20', 'frequent_gt20')
EXPECTED_COUNTS = dict(zip(LABELS, (687, 342, 171, 417)))


def bucket(frequency):
    return LABELS[0] if frequency == 0 else LABELS[1] if frequency <= 5 else LABELS[2] if frequency <= 20 else LABELS[3]


def prepare_slices(asset_root, output_root, config_path):
    path = asset_root / 'prepared/manifest.json'
    manifest = read_manifest(path)
    frequency = Counter()
    for row in manifest['splits']['train_all']:
        if row['target'] != training_target(row['answers']):
            raise ValueError('actual train target differs from frozen raw modal rule')
        frequency[NORMALIZER(row['target'])] += 1
    if len(manifest['splits']['train_all']) != 32985 or sum(frequency.values()) != 32985:
        raise ValueError('full target exposure differs')
    features = []
    for row in manifest['splits']['dev']:
        if row['target'] != training_target(row['answers']):
            raise ValueError('dev raw modal target differs')
        target = NORMALIZER(row['target'])
        features.append({'question_id': row['question_id'], 'image_id': row['image_id'],
            'question_sha256': content_sha(row['question']),
            'image_sha256': manifest['images'][row['image_id']]['sha256'],
            'sft_target_normalized': target, 'train_all_sft_target_qa_frequency': frequency[target],
            'sft_target_answer_frequency': bucket(frequency[target])})
    counts = Counter(row['sft_target_answer_frequency'] for row in features)
    if dict(counts) != EXPECTED_COUNTS:
        raise ValueError(f'fixed dev exposure bins differ:{dict(counts)}')
    lock = {'schema_version': 'rarevlm.weighted_sft.dev_slices.v1',
        'purpose': 'predeclared internal-dev pilot gate; no official benchmark feedback',
        'manifest_path': str(path), 'manifest_file_sha256': sha_file(path),
        'manifest_sha256': manifest['manifest_sha256'],
        'normalizer_source_sha256': sha_file(SOURCE_ROOT / 'vlm_deep/third_party/vqa_eval_metric.py'),
        'training_target_source_sha256': sha_file(SOURCE_ROOT / 'vlm_deep/common.py'),
        'definition': 'Normalize each verified actual train_all raw-stripped modal target (lexical ties), increment exactly once per QA including normalized empty targets. Dev lookup uses normalized verified actual raw-modal target.',
        'train_all_questions': 32985, 'dev_questions': 1617, 'dev_images': 1024,
        'slice_counts': dict(counts), 'features': features,
        'target_frequency_sha256': content_sha(dict(sorted(frequency.items()))),
        'limitations': ['supervised answer-text exposure proxy; not semantic class/pretraining frequency',
            'frequency groups differ in ambiguity,length,image difficulty; conditional exploratory CI only']}
    lock['slice_lock_sha256'] = content_sha(lock)
    lock_file(output_root / 'SLICE_LOCK.json', lock)
    return lock


def read_cell(root, condition, seed, references, images, config_path):
    directory = root / 'runs' / condition / f'seed{seed}' / 'dev'
    evaluation_path, lock_path, prediction_path = (directory / name for name in ('EVALUATION.json', 'INPUT_LOCK.json', 'predictions.jsonl'))
    if not evaluation_path.exists():
        return {'complete': False, 'condition': condition, 'seed': seed, 'path': str(directory)}
    evaluation, lock = read_json(evaluation_path), read_json(lock_path)
    if (lock.get('condition') != condition or lock.get('seed') != seed or lock.get('split') != 'dev'
        or lock.get('weighted_config_sha256') != sha_file(config_path)
        or lock.get('model_key') != model_key(condition, seed)
        or lock.get('source_sha256') != source_hashes()):
        raise ValueError('new dev lock condition/config/source differs')
    expected = {'numerical_path': 'original_bf16', 'precision': 'bf16', 'quantization': 'original',
        'vision_precision': 'bf16', 'visual_token_budget': 1024, 'batch_size': 4,
        'min_pixels': 50176, 'max_pixels': 802816, 'max_new_tokens': 32, 'do_sample': False,
        'attention_implementation': 'sdpa', 'input_truncation': False,
        'tf32_matmul': False, 'tf32_cudnn': False, 'n':1617}
    for key, value in expected.items():
        if lock.get(key) != value:
            raise ValueError(f'dev lock fixed contract differs:{key}')
    train_path = Path(lock['training_binding']['path'])
    train = read_json(train_path)
    if sha_file(train_path) != lock['training_binding']['sha256'] or train['condition'] != condition or train['seed'] != seed:
        raise ValueError('completed training identity changed')
    adapter = train_path.parent / 'adapter'
    files = {p.name: sha_file(p) for p in sorted(adapter.iterdir()) if p.is_file()}
    if files != lock['adapter_files_sha256'] or files != train['adapter_files_sha256']:
        raise ValueError('completed adapter changed')
    data = prediction_path.read_bytes()
    if not data.endswith(b'\n'):
        raise ValueError('completed prediction has interrupted tail')
    records = [json.loads(line) for line in data.splitlines()]
    if len(records) != len(references):
        raise ValueError('completed dev count differs')
    for index, (row, reference) in enumerate(zip(records, references)):
        validate_prediction(row, reference, images, 'textvqa')
        expected_row = {'condition':condition,'seed':seed,'task':'textvqa','model_key':model_key(condition, seed),
            'input_lock_sha256':content_sha(lock),'source_binding_sha256':content_sha(lock['source_sha256']),
            'question_prompt_sha256':question_prompt_sha(reference),'visual_token_budget':1024,
            'numerical_path':'original_bf16','precision':'bf16','quantization':'original',
            'model_revision':lock['model_revision'],'adapter_files_sha256':files,
            'batch_start':(index//4)*4,'batch_size':min(4,len(references)-(index//4)*4)}
        for key,value in expected_row.items():
            if row.get(key) != value:
                raise ValueError(f'prediction binding differs:{key}')
        grid = row['image_grid_thw']
        visual = math.prod(grid)//4
        if (len(grid)!=3 or grid[0]!=1 or math.prod(grid)%4 or visual!=row['visual_tokens']
            or visual!=row['counted_image_tokens'] or visual>1024 or row['input_truncated']):
            raise ValueError('actual input token contract differs')
    score = 100*statistics.mean(row['score'] for row in records)
    if (evaluation.get('status')!='completed' or evaluation['predictions_sha256']!=sha_file(prediction_path)
        or evaluation['input_lock_sha256']!=content_sha(lock)
        or not math.isclose(score,evaluation['score_percent'],abs_tol=1e-10,rel_tol=0)):
        raise ValueError('saved score/prediction hash differs')
    return {'complete':True,'condition':condition,'seed':seed,'rows':records,'lock':lock,
        'evaluation':evaluation,'training':train,'score_percent':score,
        'provenance':{'predictions_file':str(prediction_path),'predictions_sha256':sha_file(prediction_path),
            'evaluation_file':str(evaluation_path),'evaluation_sha256':sha_file(evaluation_path),
            'input_lock_sha256':content_sha(lock),'training_result_sha256':sha_file(train_path)}}


def paired_interval(before, after, features, label, replicates, seed):
    import numpy as np
    by_image = defaultdict(lambda: [0.0,0])
    pairs = []
    for left,right,feature in zip(before,after,features):
        if left['question_id']!=right['question_id'] or left['image_id']!=right['image_id']:
            raise ValueError('unpaired question/image')
        if label != 'overall' and feature['sft_target_answer_frequency'] != label:
            continue
        delta = right['score']-left['score']
        by_image[left['image_id']][0] += delta
        by_image[left['image_id']][1] += 1
        pairs.append((left['score'],right['score']))
    if not pairs:
        raise ValueError('predeclared gate slice is empty')
    matrix = np.asarray(list(by_image.values()),dtype=float)
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates,dtype=float)
    # Bootstrap images containing this slice, retaining all qualifying same-image QA.
    for index in range(replicates):
        sampled = matrix[rng.integers(0,len(matrix),size=len(matrix))].sum(axis=0)
        estimates[index]=100*sampled[0]/sampled[1]
    return {'slice':label,'uniform_score_percent':100*statistics.mean(p[0] for p in pairs),
        'frequency_score_percent':100*statistics.mean(p[1] for p in pairs),
        'delta_pp':100*statistics.mean(p[1]-p[0] for p in pairs),
        'questions':len(pairs),'image_clusters':len(matrix),
        'image_cluster_bootstrap_95_pp':np.quantile(estimates,[.025,.975]).tolist(),
        'bootstrap_replicates':replicates,'bootstrap_seed':seed,
        'uncertainty':'fixed trained checkpoint pair; image clusters; exploratory pointwise unadjusted CI; not equivalence or training-seed uncertainty'}


def gate(contrasts, config):
    declared = config.get('study', {}).get('dev_gate',config.get('dev_gate',{}))
    thresholds = {'rare_min_delta_pp':.5,'overall_min_delta_pp':-.5,'medium_min_delta_pp':-.5}
    for key,value in thresholds.items():
        if key in declared and declared[key]!=value:
            raise ValueError(f'declared pilot gate differs:{key}')
    lookup = {row['slice']:row for row in contrasts}
    checks = {'rare':lookup['rare_1_5']['delta_pp']>=.5,
        'overall':lookup['overall']['delta_pp']>=-.5,
        'medium':lookup['medium_6_20']['delta_pp']>=-.5}
    passed=all(checks.values())
    return {'passed':passed,'checks':checks,'thresholds':thresholds,
        'decision':'recommend same-protocol paired replication seeds0/200 before any frozen formal evaluation' if passed else 'stop candidate expansion; diagnose failures without official-val retuning',
        'automatic_new_training':False,'automatic_official_evaluation':False,
        'interpretation':'single-seed exploratory advancement gate on point estimates; neither significance nor success claim'}


def analyze(asset_root, output_root, config_path, seed=42):
    config=read_json(config_path)
    slices=prepare_slices(asset_root,output_root,config_path)
    references,images,_,_=dev_data(asset_root)
    cells={condition:read_cell(output_root,condition,seed,references,images,config_path) for condition in CONDITIONS}
    result={'schema_version':'rarevlm.weighted_sft.dev_analysis.v1','updated_at':now(),'seed':seed,
        'complete':all(c['complete'] for c in cells.values()),'completed_cells':sum(c['complete'] for c in cells.values()),
        'expected_cells':2,'config_sha256':sha_file(config_path),'analysis_source_sha256':sha_file(Path(__file__)),
        'slice_lock_sha256':slices['slice_lock_sha256'],'expected_predictions':3234,
        'scope':'new controlled full-data single-seed internal-dev comparison; no official-val or automatic expansion'}
    if not result['complete']:
        result.update(status='waiting_dev_predictions',pending=[c for c in cells.values() if not c['complete']])
        write_json(output_root/'analysis/ANALYSIS.json',result)
        return result
    before,after=cells['uniform'],cells['frequency']
    for key in ('weighted_config_sha256','reference_config_sha256','source_sha256','packages','model_revision',
        'model_provenance_sha256','processor_and_metadata_sha256','actual_parameter_dtypes','manifest_sha256',
        'batch_size','visual_token_budget','question_order_sha256','question_prompt_order_sha256'):
        if before['lock'].get(key)!=after['lock'].get(key):
            raise ValueError(f'paired dev common contract differs:{key}')
    for left,right in zip(before['rows'],after['rows']):
        for key in ('question_id','image_id','question_sha256','question_prompt_sha256','image_sha256','image_grid_thw',
            'visual_tokens','input_tokens','padded_input_tokens','batch_start','batch_size','input_truncated'):
            if left[key]!=right[key]:
                raise ValueError(f'paired encoded input differs:{key}')
    # Fair loss contrast requires identical initialization, order and input contract.
    required=('initial_trainable_tensor_sha256','initialization_contract_sha256',
        'global_question_order_sha256','frozen_parameter_sha256_before','input_tokenizer_processor_contract','weight_lock_sha256')
    for key in required:
        if key not in before['training'] or key not in after['training']:
            raise ValueError(f'missing controlled training binding:{key}')
    for key in ('initial_trainable_tensor_sha256','initialization_contract_sha256',
        'global_question_order_sha256','frozen_parameter_sha256_before','input_tokenizer_processor_contract','planned_updates',
        'actual_questions_seen','completed_updates','effective_batch','trainable_parameters','weight_lock_sha256'):
        if key in before['training'] or key in after['training']:
            if before['training'].get(key)!=after['training'].get(key):
                raise ValueError(f'controlled training binding differs:{key}')
    evaluation=config.get('evaluation',{})
    replicates=evaluation.get('bootstrap_replicates',2000)
    bootstrap_seed=evaluation.get('bootstrap_seed',20261011)
    if replicates!=2000:
        raise ValueError('predeclared paired bootstrap requires2000replicates')
    contrasts=[paired_interval(before['rows'],after['rows'],slices['features'],label,replicates,bootstrap_seed+index)
        for index,label in enumerate(('overall',*LABELS))]
    result.update(status='complete',validated_predictions=3234,contrasts=contrasts,
        gate=gate(contrasts,config),source_data={key:cell['provenance'] for key,cell in cells.items()},
        cells=[{'condition':key,'seed':seed,'score_percent':cell['score_percent'],
            'resources':cell['evaluation']['resources'],'input_statistics':cell['evaluation']['input_statistics'],
            'training_hardware':cell['training'].get('hardware')} for key,cell in cells.items()])
    write_json(output_root/'analysis/ANALYSIS.json',result)
    lines=['# 温和频次加权SFT：seed42内部dev探索','',
        '| 切片 | QA | 普通CE | 频次加权 | 差值pp | 图像配对95%区间 |',
        '|---|---:|---:|---:|---:|---|']
    for row in contrasts:
        ci=row['image_cluster_bootstrap_95_pp']
        lines.append(f"| {row['slice']} | {row['questions']} | {row['uniform_score_percent']:.3f} | {row['frequency_score_percent']:.3f} | {row['delta_pp']:+.3f} | [{ci[0]:.3f}, {ci[1]:.3f}] |")
    lines+=['',f"探索门控：{'通过' if result['gate']['passed'] else '未通过'}。{result['gate']['decision']}",
        '', '门控依点值而非显著性；区间条件于固定checkpoint，未作多重校正。频次为实际训练target文本曝光代理。',
        '本轮未自动扩展种子或正式集，不更新简历；新研究结果不自动代表用户掌握或个人贡献。']
    (output_root/'analysis/findings.md').write_text('\n'.join(lines)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset-root',type=Path,required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    parser.add_argument('--seed',type=int,default=42,choices=(0,42,200))
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    for name in ('asset_root','output_root','config'):
        setattr(args,name,private_path(getattr(args,name)))
    if args.prepare_only:
        slices=prepare_slices(args.asset_root,args.output_root,args.config)
        result={'prepared':True,'slice_lock_sha256':slices['slice_lock_sha256'],'slice_counts':slices['slice_counts'],
            'analysis_source_sha256':sha_file(Path(__file__))}
        write_json(args.output_root/'ANALYSIS_READY.json',result)
    else:
        result=analyze(args.asset_root,args.output_root,args.config,args.seed)
    print(json.dumps({key:result[key] for key in ('complete','prepared','completed_cells','gate','slice_counts') if key in result},ensure_ascii=False))


if __name__=='__main__':
    main()
