"""Prepare a separate train-only semantic-admission view; never train a model."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from vlm_semantic_admission_v1 import POLICY, build_training_view

PACKAGE = Path(__file__).resolve().parents[1]
OUTPUTS = PACKAGE.parents[1] / 'outputs'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode()).hexdigest()


def write_once(path, value):
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if path.exists():
        if path.read_text() != text:
            raise ValueError(f'Refusing to change an existing artifact: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-train', type=Path,
                        default=OUTPUTS / 'vlm_complementary_rank_v1/data/TRAIN_EVAL.json')
    parser.add_argument('--prior-review', type=Path,
                        default=OUTPUTS / 'vlm_complementary_rank_v1/data_preparation/revision2/DATA_REVIEW.json')
    parser.add_argument('--visual-review', type=Path,
                        default=OUTPUTS / 'vlm_deep_pipeline_diagnosis_v1/data/VISUAL_REVIEW_COHORT.json')
    parser.add_argument('--output', type=Path, default=OUTPUTS / 'vlm_semantic_admission_v1')
    args = parser.parse_args()
    root = Path('/opt/data/private')
    for path in [args.source_train, args.prior_review, args.visual_review, args.output]:
        if not path.resolve().is_relative_to(root):
            raise ValueError('All files must remain under /opt/data/private')
    # Freeze policy before inspecting/applying the actual candidate population.
    write_once(args.output / 'ADMISSION_POLICY.json', POLICY)
    source = json.loads(args.source_train.read_text())
    if source.get('manifest_sha256') != digest({k: v for k, v in source.items() if k != 'manifest_sha256'}):
        raise ValueError('Source training export identity mismatch')
    prior = json.loads(args.prior_review.read_text())
    raw_visual = json.loads(args.visual_review.read_text())
    additional = [{
        'question_id': row['question_id'], 'pair_id': row['pair_id'], 'source_split': 'train',
        'target': row['target'], 'classification': row['visual_review']['classification'],
        'evidence': row['visual_review']['observation'],
        'source_review': {'path': str(args.visual_review.resolve()), 'sha256': sha(args.visual_review)},
        'source_selection_limitation': POLICY['review_selection_limitation'],
        'decision_origin': 'previous assistant original-image review, not expert relabeling',
    } for row in raw_visual]
    write_once(args.output / 'ADDITIONAL_TRAINING_REVIEW.json', additional)
    view, decisions, qa_admissions, summary = build_training_view(source, prior, additional)
    view['manifest_sha256'] = digest(view)
    refs = {key: {'path': str(path.resolve()), 'sha256': sha(path)} for key, path in {
        'source_train_only_export': args.source_train, 'pretraining_review': args.prior_review,
        'additional_review_source': args.visual_review,
        'policy': args.output / 'ADMISSION_POLICY.json',
        'admission_code': PACKAGE / 'src/vlm_semantic_admission_v1/admission.py',
        'preparation_code': Path(__file__),
    }.items()}
    summary['inputs'] = refs
    summary['known_pretraining_vetoes'] = [decision for decision in decisions
                                          if any(e.get('review_id') in ('T02', 'T08')
                                                 for e in decision['review_evidence'])]
    summary['confirm_boundary_note'] = (
        'This preparation tool reads only the independent training export and training review rows. '
        'Earlier manual schema inspection displayed some confirm pair metadata from a combined legacy '
        'manifest; no confirm prediction, score, or example was used in admission or model computation.')
    source_rows = source['splits']['dev'] if source['splits'].get('dev') else source['splits']['train_all']
    rows_by_id = {row['question_id']: row for row in source_rows}
    source_positions = {row['question_id']: index for index, row in enumerate(source_rows)}
    support = []
    for admission in qa_admissions:
        if admission['admitted'] and admission['status'] == 'reviewed_original_label_supported':
            row = rows_by_id[admission['question_id']]
            support.append({'question_id': row['question_id'], 'pair_id': row['complementary_pair_id'],
                            'target': row['target'], 'question': row['question'],
                            'image_id': row['image_id'], 'image_path': source['images'][row['image_id']]['path'],
                            'source_training_export': refs['source_train_only_export'],
                            'source_row_index': source_positions[row['question_id']],
                            'note': 'Supports original GT; no new annotation or model evaluation.'})
    write_once(args.output / 'TRAINING_MANIFEST.json', view)
    write_once(args.output / 'PAIR_ADMISSION.json', decisions)
    write_once(args.output / 'QA_ADMISSION.json', qa_admissions)
    write_once(args.output / 'REVIEW_SUPPORTED_TRAINING_QA.json', support)
    write_once(args.output / 'PREPARATION_RESULT.json', summary)
    print(json.dumps({key: value for key, value in summary.items()
                      if key not in ('inputs', 'known_pretraining_vetoes')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
