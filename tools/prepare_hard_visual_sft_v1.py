"""Freeze image folds, mine training errors, and establish a separate screen baseline."""
import json
from pathlib import Path
import sys

from rarevlm_paths import PROJECT_ROOT
from vlm_deep.common import content_sha, sha_file, write_json

ROOT = Path('/opt/data/private/longtail_vlm/rarevlm_pilot/outputs')
OUT = ROOT/'vlm_hard_visual_sft_v1'


def read(path):
    return json.loads(Path(path).read_text())


def freeze(path, document):
    if path.exists() and read(path) != document:
        raise ValueError(f'Frozen content changed: {path}')
    write_json(path, document)


def fold(image_id):
    bucket = int(content_sha(['hard_visual_sft_v1_image_fold', image_id])[:12], 16) % 10
    return 'screen' if bucket == 0 else 'confirm' if bucket == 1 else 'train'


def eval_manifest(rows, images, provenance):
    document = {'schema_version': 'rarevlm.vlm_data.v2', 'purpose': provenance,
                'images': {row['image_id']: images[row['image_id']] for row in rows},
                'splits': {'train_all': [], 'train_10k': [], 'dev': rows, 'final': []}}
    document['manifest_sha256'] = content_sha(document)
    return document


def main():
    from research_runtime.queue import run_queue
    OUT.mkdir(parents=True, exist_ok=True)
    source_path = ROOT/'vlm_human_vqa_v1/MANIFEST.json'
    source = read(source_path)
    rows = {name: [row for row in source['splits']['train_all'] if fold(row['image_id']) == name]
            for name in ('train', 'screen', 'confirm')}
    image_sets = {name: {row['image_id'] for row in values} for name, values in rows.items()}
    assert not (image_sets['train'] & image_sets['screen'] or image_sets['train'] & image_sets['confirm']
                or image_sets['screen'] & image_sets['confirm'])
    original = read(PROJECT_ROOT/'configs/vlm_human_sft_baseline_v1.json')
    split = ((len(rows['train'])+7)//8)*4
    shards = {'mine0': rows['train'][:split], 'mine1': rows['train'][split:], 'screen': rows['screen']}
    plan = {'source_sha256': sha_file(source_path), 'initial_adapter': original['initial_adapter'],
            'image_fold_rule': 'sha256 fixed salt modulo10: screen0/confirm1/train2..9',
            'question_ids': {name: [row['question_id'] for row in values] for name, values in rows.items()},
            'counts': {name: {'qa': len(values), 'images': len(image_sets[name])} for name, values in rows.items()},
            'prior_exposure': 'source was used in the prior completed study and a small readout diagnostic; new runs start from the original SFT42, with continuation image folds disjoint',
            'confirm_generation_allowed_now': False, 'training_updates': 0,
            'planned_research': 'class-balanced hard-example SFT vs class-balanced uniform SFT; identical replay and optimization; <=200 updates/branch before gate'}
    freeze(OUT/'DATA_PLAN.json', plan)
    jobs = []
    for name, selected in shards.items():
        manifest_path = OUT/'data'/f'{name.upper()}_EVAL.json'
        freeze(manifest_path, eval_manifest(selected, source['images'], 'training-only error mining' if name != 'screen' else 'continuation-heldout development screen'))
        config = {**original, 'output_root': str(OUT/'mining'/name),
                  'human_manifest': str(manifest_path), 'expected_eval_questions': {'human': len(selected), 'text': 1617}}
        path = OUT/'data'/f'{name.upper()}_EVAL_CONFIG.json'; freeze(path, config)
        jobs.append({'key': name, 'command': [sys.executable, '-u', '-m', 'vlm_human_sft.evaluate',
                     '--config', str(path), '--task', 'human'],
                     'completion_marker': str(OUT/'mining'/name/'initial/human/COMPLETE.json')})
    freeze(OUT/'data/CONFIRM_EVAL.json', eval_manifest(rows['confirm'], source['images'], 'not generated before candidate/screen decision is frozen'))
    # Preserve each original four-question batch to reuse the exact raw-FP32 initial predictions.
    text_path = Path(original['text_manifest']); text = read(text_path); dev = text['splits']['dev']
    starts = list(range(0, len(dev)-3, 4))
    starts = sorted(sorted(starts, key=lambda start: content_sha(['hard_visual_text_screen_v1', dev[start]['question_id']]))[:128])
    selected = [row for start in starts for row in dev[start:start+4]]
    freeze(OUT/'data/TEXT_SCREEN_EVAL.json', eval_manifest(selected, text['images'], 'fixed complete original batches, no outcome selection'))
    initial_path = ROOT/'vlm_human_sft_repair_v1/initial/text/predictions.jsonl'
    predictions = list(map(json.loads, initial_path.read_text().splitlines()))
    chosen = [row for start in starts for row in predictions[start:start+4]]
    freeze(OUT/'data/INITIAL_TEXT_SCREEN.json', {'source_sha256': sha_file(initial_path),
           'original_batch_starts': starts, 'records': chosen,
           'scope': 'derived original predictions, same model/profile and complete four-QA groups; no new generation'})
    print(json.dumps(plan['counts']), flush=True)
    run_queue(OUT/'mining_queue', jobs, {'plan_sha256': sha_file(OUT/'DATA_PLAN.json'),
        'tool_sha256': sha_file(Path(__file__)), 'optimizer_updates': 0, 'confirmation_not_generated': True})
    write_json(OUT/'MINING_COMPLETE.json', {'status': 'completed', 'optimizer_updates': 0, 'confirm_generated': False})


if __name__ == '__main__':
    main()
