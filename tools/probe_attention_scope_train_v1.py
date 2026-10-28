"""One fixed, train-only 80-QA diagnostic of three completed checkpoints; zero updates."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import sys

PACKAGE = Path(__file__).resolve().parents[1]
OUTPUTS = PACKAGE.parents[1] / 'outputs'
STUDY = OUTPUTS / 'vlm_attention_scope_v1'
ROOT = STUDY / 'failure_analysis_v1/fixed_train_probe_v1'


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def prepare():
    from vlm_deep.common import content_sha
    from vlm_attention_scope_v1.study import verify as verify_parent
    verify_parent()
    if (ROOT / 'INPUT_LOCK.json').exists():
        verify(); return
    source_path = STUDY / 'data/TRAINING_MANIFEST.json'
    source = read(source_path)
    human = [q for q in source['splits']['train_all'] if q['study_source'] == 'human']
    initial_path = OUTPUTS / 'vlm_complementary_rank_v1/initial/train/human/predictions.jsonl'
    initial = {str(q['question_id']): q for q in rows(initial_path)}
    wrong = [q for q in human if initial[str(q['question_id'])]['primary_score'] == 0]
    if len(wrong) != 40:
        raise ValueError('The predeclared historical training-error count changed')
    counts = Counter(q['target'] for q in wrong)
    anchors = []
    salt = 'attention_scope_train_probe_v1_correct_anchors'
    for label in ('yes', 'no'):
        candidates = [q for q in human if q['target'] == label and initial[str(q['question_id'])]['primary_score'] == 1]
        candidates.sort(key=lambda q: hashlib.sha256((salt + str(q['question_id'])).encode()).hexdigest())
        anchors.extend(candidates[:40 - counts[label]])
    error_ids = {str(q['question_id']) for q in wrong}
    selected = [{**q, 'probe_group': 'historically_wrong' if str(q['question_id']) in error_ids else 'correct_anchor'}
                for q in wrong + anchors]
    selected.sort(key=lambda q: hashlib.sha256(('fixed_probe_order_v1' + str(q['question_id'])).encode()).hexdigest())
    if len(selected) != 80 or Counter(q['target'] for q in selected) != {'yes': 40, 'no': 40}:
        raise ValueError('Probe QA count or balance differs')
    manifest = {'schema_version': source['schema_version'],
        'purpose': 'Historical training errors plus fixed correct anchors; train-only diagnostic, not capability evaluation',
        'images': {q['image_id']: source['images'][q['image_id']] for q in selected},
        'splits': {'train_all': [], 'train_10k': [], 'dev': selected, 'final': []}}
    manifest['manifest_sha256'] = content_sha(manifest)
    write(ROOT / 'TRAIN_EVAL.json', manifest)
    config = read(PACKAGE / 'configs/vlm_attention_scope_v1.json')
    models = {'initial': {'adapter': config['initial_adapter'],
                         'lora_sha256': read(STUDY / 'initial/human/MODEL_IDENTITY.json')['lora_sha256']}}
    for role in ('qv', 'qkvo'):
        t = read(STUDY / 'runs' / role / 'train/TRAIN_COMPLETE.json')
        models[role] = {'adapter': t['adapter'], 'lora_sha256': t['final_lora_sha256']}
    protocol = {'models': models, 'baseline_config': config['baseline_config'],
        'manifest': str(ROOT / 'TRAIN_EVAL.json'), 'selection': {'historical_errors': 40, 'anchors': 40,
        'historical_error_labels': dict(counts), 'anchor_salt': salt, 'overall_label_balance': {'yes': 40, 'no': 40}},
        'generation': {'batch': 4, 'max_new_tokens': 32, 'visual_budget': 1024},
        'readout': 'prompt-only dense-head canonical and pre-fixed alias groups; no decision calibration',
        'scope': 'Single fixed train cohort, not overall training accuracy or heldout gain. Fresh initial uses same 80-QA batches.',
        'training_updates': 0, 'confirmation_used': False}
    write(ROOT / 'PROTOCOL.json', protocol)
    files = [Path(__file__), source_path, initial_path, ROOT / 'TRAIN_EVAL.json', ROOT / 'PROTOCOL.json',
             Path(config['baseline_config']), PACKAGE / 'src/vlm_fixed_error_readout/readout.py',
             PACKAGE / 'tools/run_fixed_error_readout_v1.py', PACKAGE / 'src/vlm_runtime_metadata_v1/model.py',
             PACKAGE / 'src/vlm_human_sft/evaluate.py', PACKAGE / 'src/vlm_deep/common.py',
             PACKAGE / 'src/vlm_fp32_adapter/model.py', PACKAGE / 'src/vlm_probe/evaluate.py']
    for model in models.values():
        files.extend(Path(model['adapter']) / name for name in ('adapter_model.safetensors', 'adapter_config.json'))
    write(ROOT / 'INPUT_LOCK.json', {'files': {str(f): sha(f) for f in files},
                                    'new_model_updates': 0, 'no_hyperparameter_or_sample_sweep': True})


def verify():
    from vlm_attention_scope_v1.study import verify as verify_parent
    verify_parent()
    for path, expected in read(ROOT / 'INPUT_LOCK.json')['files'].items():
        if sha(path) != expected:
            raise ValueError('Frozen diagnostic source changed: ' + path)


def worker(job):
    import torch
    from vlm_deep.common import collate, encode, to_device
    from vlm_fp32_adapter.model import restore_fp32_adapter
    from vlm_runtime_metadata_v1.model import load
    from vlm_human_sft.evaluate import evaluate, lora_hash
    from vlm_probe.evaluate import inference_row, processor_for
    from vlm_fixed_error_readout.readout import forward_first_readout
    from run_fixed_error_readout_v1 import token_aliases
    verify()
    p = read(ROOT / 'PROTOCOL.json');base = read(p['baseline_config'])
    roles = ('initial',) if job == 'initial' else ('qv', 'qkvo')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32 = False;torch.backends.cudnn.allow_tf32 = False
    model, _ = load(base['model_root'], p['models'][roles[0]]['adapter'], torch.device('cuda:0'), training=False)
    processor = processor_for(Path(base['model_root']), read(base['reference_config']), 1024)
    aliases = token_aliases(processor.tokenizer);yes,no = [aliases['groups'][label] for label in ('yes','no')]
    m = read(p['manifest']);qa=m['splits']['dev'];images=m['images']
    for index, role in enumerate(roles):
        if index:
            restore_fp32_adapter(model, p['models'][role]['adapter'])
        if lora_hash(model) != p['models'][role]['lora_sha256']:
            raise ValueError('Diagnostic loaded a different adapter')
        ec = {**base, 'initial_adapter': p['models']['initial']['adapter'], 'human_manifest': p['manifest'],
              'expected_eval_questions': {'human': 80, 'text': 512}}
        evaluate(model, processor, ec, 'human', ROOT / 'runs' / role / 'generation')
        values = []
        model.eval()
        with torch.inference_mode():
            for start in range(0,80,4):
                chunk = qa[start:start+4]
                encoded=[encode(processor,inference_row(q),Path(images[q['image_id']]['path']),False) for q in chunk]
                batch=to_device(collate(encoded,processor.tokenizer.pad_token_id),torch.device('cuda:0'))
                with torch.autocast('cuda',enabled=False):
                    v=forward_first_readout(model,batch,yes['ids'],no['ids'],yes['canonical'],no['canonical'],slice_head=False)
                values.extend({'question_id':q['question_id'],**value} for q,value in zip(chunk,v))
        write(ROOT / 'runs' / role / 'READOUT.json', values)
        if lora_hash(model) != p['models'][role]['lora_sha256']:
            raise ValueError('Read-only evaluation altered adapter')
        write(ROOT / 'runs' / role / 'COMPLETE.json', {'status':'completed','qa':80,'lora_unchanged':True,'training_updates':0})


def analyze():
    m=read(ROOT/'TRAIN_EVAL.json');qa=m['splits']['dev'];results={}
    for role in ('initial','qv','qkvo'):
        pred={str(q['question_id']):q for q in rows(ROOT/'runs'/role/'generation/predictions.jsonl')}
        logits={str(q['question_id']):q for q in read(ROOT/'runs'/role/'READOUT.json')}
        result={}
        for group,selected in [('all80',qa),('historically_wrong',[q for q in qa if q['probe_group']=='historically_wrong']),
            ('correct_anchor',[q for q in qa if q['probe_group']=='correct_anchor'])]:
            for label in ('all','yes','no'):
                selected_label=[q for q in selected if label=='all' or q['target']==label]
                result[group+'_'+label]={'qa':len(selected_label),
                    'generation_correct':sum(pred[str(q['question_id'])]['primary_score'] for q in selected_label),
                    'mean_canonical_first_nll':statistics.mean(-logits[str(q['question_id'])][f'canonical_{q["target"]}_log_probability'] for q in selected_label),
                    'mean_alias_first_nll':statistics.mean(-logits[str(q['question_id'])][f'{q["target"]}_alias_log_probability'] for q in selected_label)}
        results[role]=result
    out={'status':'completed','results':results,'training_updates':0,'scope':'Fixed balanced80 training diagnostic; not overall training accuracy or heldout gain'}
    write(ROOT/'ANALYSIS.json',out);write(ROOT/'COMPLETION.json',out)
    print(json.dumps(out,ensure_ascii=False,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare',action='store_true');parser.add_argument('--worker',choices=('initial','finals'));a=parser.parse_args()
    if a.prepare:prepare();return
    if a.worker:worker(a.worker);return
    verify()
    from research_runtime.queue import run_queue
    run_queue(ROOT/'queue',[{'key':job,'command':[sys.executable,'-u',str(Path(__file__).resolve()),'--worker',job],
        'completion_marker':str(ROOT/'runs'/('initial' if job=='initial' else 'qkvo')/'COMPLETE.json')} for job in ('initial','finals')],
        {'input_lock_sha256':sha(ROOT/'INPUT_LOCK.json'),'training_updates':0})
    analyze()


if __name__=='__main__':main()
