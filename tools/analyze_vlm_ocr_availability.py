"""Post-generation OCR availability diagnostic; never selects model inputs.

The reference oracle may output only the saved baseline prediction or one
nonempty normalized recognized OCR line. It is not deployable and is not an
upper bound for arbitrary OCR reasoning, substring extraction, or fusion.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


sys.dont_write_bytecode = True
BASE = Path('/opt/data/private/longtail_vlm/rarevlm_pilot/outputs')
REPO = Path(__file__).resolve().parents[1]
EPS = 1e-12


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_predictions(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    result = {str(row['question_id']): row for row in rows}
    if len(result) != len(rows):
        raise ValueError('Duplicate saved prediction IDs')
    return result


def candidate_maximum(items, field, references, normalizer):
    """Exact existing TextVQA LOO score, except empty OCR candidates excluded."""
    best = dict(score=0.0, matching_references=0, text=None, detector_index=None)
    empty_normalized = []
    for item in items:
        text = str(item[field])
        normalized = normalizer(text)
        if not normalized:
            empty_normalized.append(dict(text=text, detector_index=item.get('detector_index')))
            continue
        matches = sum(normalized == answer for answer in references)
        score = sum(min(sum(normalized == answer for j, answer in enumerate(references)
                            if j != i) / 3, 1) for i in range(10)) / 10
        if (score, matches) > (best['score'], best['matching_references']):
            best = dict(score=score, matching_references=matches, text=text,
                        detector_index=item.get('detector_index'))
    return dict(best=best, candidate_lines=len(items),
                empty_normalized_excluded=empty_normalized,
                nonempty_normalized_lines=len(items) - len(empty_normalized))


def summarize(rows, candidate_name):
    baseline_sum = sum(row['baseline_score'] for row in rows)
    oracle_sum = sum(row['restricted_oracles'][candidate_name] for row in rows)
    return dict(questions=len(rows),
                baseline_saved_consensus_percent=100 * baseline_sum / len(rows),
                restricted_oracle_consensus_percent=100 * oracle_sum / len(rows),
                restricted_oracle_gain_pp=100 * (oracle_sum - baseline_sum) / len(rows),
                improvable_questions=sum(row['restricted_oracles'][candidate_name] >
                                         row['baseline_score'] + EPS for row in rows),
                any_positive_ocr_line_questions=sum(row['candidates'][candidate_name]['best']['score'] > EPS
                                                    for row in rows),
                oracle_reaches_full_score_questions=sum(row['restricted_oracles'][candidate_name] >= 1 - EPS
                                                       for row in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=BASE / 'vlm_ocr_compact_v1/diagnostics/OCR_AVAILABILITY.json')
    args = parser.parse_args()
    paths = {
        'manifest': BASE / 'vlm_deep_v2/prepared/manifest.json',
        'raw_ocr': BASE / 'vlm_ocr_context_v1/recognizer/OCR_RAW.json',
        'v1_context': BASE / 'vlm_ocr_context_v1/OCR_CONTEXT_LOCK.json',
        'compact_context': BASE / 'vlm_ocr_compact_v1/OCR_CONTEXT_LOCK.json',
        'normalizer_source': REPO / 'src/vlm_deep/third_party/vqa_eval_metric.py',
        'existing_consensus_source': REPO / 'src/vlm_deep/common.py',
        'diagnostic_source': Path(__file__).resolve(),
    }
    for condition in ('baseline', 'empty_template', 'predicted_ocr', 'shuffled_ocr'):
        paths['predictions_' + condition] = (BASE / 'vlm_ocr_context_v1/runs' / condition /
                                            'dev/predictions.jsonl')
    before = {key: file_sha(path) for key, path in paths.items()}
    module_spec = importlib.util.spec_from_file_location('availability_vqa_metric', paths['normalizer_source'])
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    normalizer = module.EvalAIAnswerProcessor()
    manifest = load_json(paths['manifest'])
    references = {str(row['question_id']): row for row in manifest['splits']['dev']}
    predictions = {condition: read_predictions(paths['predictions_' + condition])
                   for condition in ('baseline', 'empty_template', 'predicted_ocr', 'shuffled_ocr')}
    if len(references) != 1617 or any(set(rows) != set(references) for rows in predictions.values()):
        raise ValueError('Expected four complete 1617-question dev predictions')
    raw, v1, compact = (load_json(paths[key]) for key in ('raw_ocr', 'v1_context', 'compact_context'))
    if not all(obj.get('complete') for obj in (raw, v1, compact)):
        raise ValueError('OCR raw/context records incomplete')
    if v1['ocr_raw_file_sha256'] != before['raw_ocr'] or compact['ocr_raw_file_sha256'] != before['raw_ocr']:
        raise ValueError('Context locks use different raw OCR')
    agreement = Counter()
    results = []
    for qid, refrow in sorted(references.items(), key=lambda pair: int(pair[0])):
        image_id = refrow['image_id']
        if any(rows[qid]['image_id'] != image_id for rows in predictions.values()):
            raise ValueError('Prediction image ID mismatch')
        answers = [normalizer(answer) for answer in refrow['answers']]
        if len(answers) != 10:
            raise ValueError('TextVQA requires 10 references')
        correct = predictions['predicted_ocr'][qid]['prediction']
        wrong = predictions['shuffled_ocr'][qid]['prediction']
        correct_norm, wrong_norm = normalizer(correct), normalizer(wrong)
        agreement['raw_exact'] += correct == wrong
        agreement['normalized_exact_including_empty'] += correct_norm == wrong_norm
        agreement['normalized_nonempty_exact'] += bool(correct_norm) and correct_norm == wrong_norm
        agreement['both_normalized_empty'] += not correct_norm and not wrong_norm
        eligible = [item for item in raw['images'][image_id]['items'] if float(item['rec_score']) >= 0.8]
        v1_items = v1['images'][image_id]['context_items']
        compact_items = compact['images'][image_id]['context_items']
        old_indices = {item['detector_index'] for item in v1_items}
        added_items = [item for item in compact_items if item['detector_index'] not in old_indices]
        candidates = {
            'raw_rec_ge_0_8': candidate_maximum(eligible, 'rec_text', answers, normalizer),
            'v1_retained': candidate_maximum(v1_items, 'input_text', answers, normalizer),
            'compact_retained': candidate_maximum(compact_items, 'input_text', answers, normalizer),
        }
        baseline = predictions['baseline'][qid]
        base_score = float(baseline['score'])
        if not 0 <= base_score <= 1:
            raise ValueError('Invalid saved baseline score')
        results.append(dict(question_id=qid, image_id=image_id, baseline_score=base_score,
                            baseline_prediction_normalized_empty=not bool(normalizer(baseline['prediction'])),
                            reference_normalized_empty_count=sum(not answer for answer in answers),
                            candidates=candidates,
                            restricted_oracles={name: max(base_score, value['best']['score'])
                                                for name, value in candidates.items()},
                            compact_added_lines=candidate_maximum(added_items, 'input_text', answers, normalizer)))
    unresolved = [row for row in results if row['baseline_score'] < 1 - EPS]
    if len(unresolved) != 323:
        raise ValueError('Expected original 323 baseline unresolved questions')
    n = len(results)
    agreement_report = dict(questions=n, conditions=['predicted_ocr', 'shuffled_ocr'],
                            raw_comparison='Literal stored prediction string equality; no stripping.',
                            normalized_comparison='Existing EvalAIAnswerProcessor; nonempty match separately reported.',
                            counts=dict(agreement),
                            fractions={key: value / n for key, value in agreement.items()},
                            interpretation='High agreement does not prove equivalence, negligible causal contribution, or quality preservation.')
    compact_added_report = dict(
        original_unresolved_questions=len(unresolved),
        questions_with_any_added_compact_line=sum(row['compact_added_lines']['candidate_lines'] > 0 for row in unresolved),
        questions_added_line_has_positive_reference_support=sum(row['compact_added_lines']['best']['score'] > EPS
                                                                for row in unresolved),
        questions_added_line_exceeds_v1_max_loo_score=sum(row['compact_added_lines']['best']['score'] >
                                                        row['candidates']['v1_retained']['best']['score'] + EPS
                                                        for row in unresolved),
        questions_added_line_exceeds_v1_max_reference_votes=sum(row['compact_added_lines']['best']['matching_references'] >
                                                              row['candidates']['v1_retained']['best']['matching_references']
                                                              for row in unresolved),
        questions_compact_restricted_oracle_exceeds_v1=sum(row['restricted_oracles']['compact_retained'] >
                                                          row['restricted_oracles']['v1_retained'] + EPS
                                                          for row in unresolved),
        note='Added means detector_index retained by compact but absent from v1; positive support differs from improvement beyond old best line or saved baseline.')
    after = {key: file_sha(path) for key, path in paths.items()}
    if before != after:
        raise ValueError('Read-only diagnostic input/source changed during analysis')
    report = dict(
        schema_version='rarevlm.ocr_availability.posthoc.v1',
        created_at=datetime.now(timezone.utc).isoformat(),
        scope='Post-generation CPU-only diagnosis on old complete dev predictions and already-frozen OCR locks.',
        source_files={key: dict(path=str(paths[key]), sha256=sha) for key, sha in before.items()},
        contracts=dict(question_count=n, image_count=len({r['image_id'] for r in results}),
                       complete_old_predictions={key: len(value) for key, value in predictions.items()},
                       source_and_input_hashes_unchanged=True, references_used_only_in_postgeneration_analysis=True,
                       selector_config_prepare_writes=False, new_gpu_calls=0, new_recognition_calls=0),
        answer_agreement=agreement_report,
        restricted_oracle_definition=dict(
            options='Saved baseline output OR exactly one recognized OCR content line; maximize reference-based LOO score after generation.',
            text_fields=dict(raw='rec_text with rec_score>=0.8', v1='retained context_items.input_text',
                             compact='retained context_items.input_text'),
            coordinate_and_prompt_wrappers_excluded=True,
            deployment='Not deployable: candidate selection uses dev human references.',
            upper_bound_scope='Upper bound only for this restricted baseline-or-single-OCR-line output set; not all OCR-based strategies.',
            excluded_capabilities=['Substring extraction', 'Concatenating multiple OCR lines',
                                   'Reasoning or paraphrasing from OCR', 'Model output recombination'],
            empty_normalized_policy='Ignore OCR candidates normalized to empty; never credit equality of empty OCR and empty reference strings.',
            baseline_policy='Keep the original saved official baseline score; do not rewrite historical predictions/scorer or claim a new deployment score.'),
        summaries={name: dict(all_dev=summarize(results, name),
                              original_unresolved=summarize(unresolved, name)) for name in candidates},
        compact_added_reference_support=compact_added_report,
        normalizer_boundary=dict(
            implementation=str(paths['normalizer_source']),
            articles_removed=['a', 'an', 'the'],
            examples={text: normalizer(text) for text in ('a', 'an', 'the', 'the a', 'A', 'zero', 'one')},
            interpretation='A standalone answer letter A also becomes empty. This is a normalizer boundary, not an OCR recognition failure. Empty OCR candidates get no diagnostic reference credit.',
            baseline_empty_normalized_questions=sum(row['baseline_prediction_normalized_empty'] for row in results),
            baseline_empty_normalized_positive_saved_score_questions=sum(row['baseline_prediction_normalized_empty'] and
                                                                        row['baseline_score'] > EPS for row in results),
            reference_empty_normalized_questions=sum(row['reference_normalized_empty_count'] > 0 for row in results),
            excluded_empty_normalized_line_occurrences_per_qa={name: sum(len(row['candidates'][name]['empty_normalized_excluded'])
                                                                        for row in results) for name in candidates}),
        per_question=results,
        limitations=['References used post hoc; no selector/config/prepare changes permitted from this diagnostic.',
                     'Already-exposed internal dev and a single saved checkpoint; no new generalization or statistical-equivalence claim.',
                     'Line reference availability does not establish the VLM consumed the text correctly or that compact v2 improves model quality.',
                     'No bootstrap inference added; this is deterministic finite-sample candidate availability.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(output=str(args.output), answer_agreement=agreement_report,
                          summaries=report['summaries'], compact_added_reference_support=compact_added_report,
                          normalizer_boundary=report['normalizer_boundary']), ensure_ascii=False))


if __name__ == '__main__':
    main()
