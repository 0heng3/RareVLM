"""A known uncertain training example vetoes its whole complementary pair.

Global review pass status never overrides sample-level uncertainty. Remaining
unreviewed examples retain only the original official human-consensus claim.
This module has no dependency on predictions, model scores, or heldout rows.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import re


POLICY = {
    'schema_version': 'rarevlm.semantic_admission.v1',
    'scope': 'training_only; preserve original question, target, image and fold',
    'minimum_official_answer_votes': 9,
    'required_answer_count': 10,
    'required_label_source': 'official_vqa_v2_train2014_human_answers',
    'pair_veto': 'either QA uncertain/rejected or failed prior pair review excludes both QAs',
    'pool_status_cannot_override_sample_veto': True,
    'question_rules': [
        {'id': 'subjective_predominance', 'pattern': r'\bpredominance\b',
         'reason': 'No explicit area threshold or silver/gray/white category boundary.'},
        {'id': 'appropriateness_right_ball', 'pattern': r'\bright ball\b',
         'reason': 'Ball appropriateness depends on inferred game/rules.'},
        {'id': 'intent_or_unobserved_trajectory', 'pattern': r'\btrying\b|\bwent right to her\b',
         'reason': 'Intent and preceding ball trajectory are not directly established by one frame.'},
    ],
    'additional_review_veto_classifications': [
        'insufficient_fine_detail', 'label_plausible_detail_limited',
        'question_semantics_require_context', 'small_overlapping_people_count_uncertain',
        'subjective_category_and_proportion',
    ],
    'additional_review_supported_classifications': [
        'label_visually_supported', 'label_supported_by_unpacking_evidence',
    ],
    'unreviewed_status': 'official_human_consensus_not_individually_verified',
    'exclusion_does_not_mean_proven_mislabel': True,
    'historical_scores_unchanged': True,
    'review_selection_limitation': (
        'The existing eight-image review came from training residual errors, not a blind random '
        'sample. Its uncertainty veto is training-only; it does not estimate overall noise or '
        'change historical evaluation denominators. No prediction/score is an admission input.'),
}


def extract_training_rows(manifest):
    """Accept one nonempty train-only view, including legacy dev-slot train exports."""
    splits = manifest.get('splits', {})
    nonempty = [name for name, rows in splits.items() if rows]
    if len(nonempty) != 1 or nonempty[0] not in ('train_all', 'dev'):
        raise ValueError('Expected one train-only export; heldout or multiple nonempty folds rejected')
    for name, pairs in manifest.get('complementary_pairs', {}).items():
        if name != 'train' and pairs:
            raise ValueError('Heldout pair records are forbidden in the training-only input')
    rows = splits[nonempty[0]]
    if any(row.get('source_split') != 'train' for row in rows):
        raise ValueError('Every input QA must explicitly originate in the training fold')
    if len({row['question_id'] for row in rows}) != len(rows):
        raise ValueError('Duplicate training question IDs')
    if set(manifest['images']) != {row['image_id'] for row in rows}:
        raise ValueError('Training-only image metadata must exactly match the training QA population')
    return rows


def build_training_view(manifest, prior_review, additional_review, policy=POLICY):
    rows = extract_training_rows(manifest)
    grouped = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[row['complementary_pair_id']].append((index, row))
    priors = defaultdict(list)
    for review in prior_review.get('rows', []):
        if review.get('split') == 'train':
            priors[review['pair_id']].append(review)
    additional = defaultdict(list)
    for review in additional_review:
        if review.get('source_split') != 'train':
            raise ValueError('Additional reviews must be training-only')
        additional[review['pair_id']].append(review)
    if set(priors) - set(grouped) or set(additional) - set(grouped):
        raise ValueError('A training review refers to a pair outside the source training population')

    decisions, qa_admissions, accepted_pairs, keep = [], [], [], set()
    source_qids = {row['question_id'] for row in rows}
    for reviews in additional.values():
        if any(review['question_id'] not in source_qids for review in reviews):
            raise ValueError('Additional review references an unknown training QA')
    for pair_id, members in grouped.items():
        pair_rows = [row for _, row in members]
        if (len(pair_rows) != 2 or {row['target'] for row in pair_rows} != {'yes', 'no'}
                or len({row['question'] for row in pair_rows}) != 1):
            raise ValueError(f'Expected same-question original yes/no pair: {pair_id}')
        reasons, evidence, pair_reviewed, individually_supported = [], [], False, set()
        for row in pair_rows:
            target = row['target']
            votes = sum(answer == target for answer in row.get('answers', []))
            if (row.get('label_source') != policy['required_label_source']
                    or len(row.get('answers', [])) != policy['required_answer_count']
                    or votes < policy['minimum_official_answer_votes']
                    or row.get('answer_vote_count') != votes):
                reasons.append('official_human_consensus_below_required_or_inconsistent')
                evidence.append({'kind': 'official_annotation', 'question_id': row['question_id'],
                                 'target': target, 'counted_target_votes': votes,
                                 'declared_target_votes': row.get('answer_vote_count')})
        for review in priors[pair_id]:
            answers = review.get('answers', {})
            targets = review.get('original_targets', {})
            known = bool(answers) and set(answers) == set(targets)
            explicit_unknown = not known or any(value not in ('yes', 'no') for value in answers.values())
            matches = known and all(value == targets[key] for key, value in answers.items())
            if review.get('pair_passed') is not True:
                reasons.append('pretraining_sample_review_not_passed')
            if explicit_unknown:
                reasons.append('pretraining_sample_review_unknown')
            elif not matches:
                reasons.append('pretraining_sample_review_disagrees_with_original_target')
            evidence.append({'kind': 'pretraining_review', **deepcopy(review)})
            if review.get('pair_passed') is True and not explicit_unknown and matches:
                pair_reviewed = True
        for review in additional[pair_id]:
            qid = review['question_id']
            row = next((row for row in pair_rows if row['question_id'] == qid), None)
            if row is None or review['target'] != row['target']:
                raise ValueError('Additional review QA/pair/target identity differs from training source')
            classification = review['classification']
            if classification in policy['additional_review_veto_classifications']:
                reasons.append('additional_training_review_uncertain')
            elif classification in policy['additional_review_supported_classifications']:
                individually_supported.add(qid)
            else:
                raise ValueError(f'Unknown additional review classification: {classification}')
            evidence.append({'kind': 'additional_training_review', **deepcopy(review)})
        for rule in policy['question_rules']:
            if re.search(rule['pattern'], pair_rows[0]['question'], flags=re.IGNORECASE):
                reasons.append(f"question_rule:{rule['id']}")
                evidence.append({'kind': 'question_rule', **deepcopy(rule)})
        admitted = not reasons
        reasons = list(dict.fromkeys(reasons))
        decisions.append({'pair_id': pair_id, 'question': pair_rows[0]['question'],
                          'question_ids': [row['question_id'] for row in pair_rows],
                          'admitted': admitted, 'exclusion_reasons': reasons,
                          'review_evidence': evidence,
                          'original_targets_unchanged': True})
        for row in pair_rows:
            reviewed = pair_reviewed or row['question_id'] in individually_supported
            qa_admissions.append({'question_id': row['question_id'], 'pair_id': pair_id,
                                  'admitted': admitted, 'target': row['target'],
                                  'status': ('reviewed_original_label_supported' if reviewed else
                                             policy['unreviewed_status']),
                                  'exclusion_reasons': reasons})
        if admitted:
            keep.update(row['question_id'] for row in pair_rows)
            accepted_pairs.append({'pair_id': pair_id, 'question': pair_rows[0]['question'],
                                   'positive_question_id': next(row['question_id'] for row in pair_rows
                                                                if row['target'] == 'yes'),
                                   'negative_question_id': next(row['question_id'] for row in pair_rows
                                                                if row['target'] == 'no'),
                                   'source_training_row_indices': [index for index, _ in members]})
    kept_rows = [deepcopy(row) for row in rows if row['question_id'] in keep]
    image_ids = {row['image_id'] for row in kept_rows}
    view = {'schema_version': manifest['schema_version'],
            'purpose': 'Independent training-only semantic admission v1; original labels unchanged',
            'training_allowed': True,
            'images': {key: deepcopy(value) for key, value in manifest['images'].items() if key in image_ids},
            'splits': {'train_all': kept_rows, 'train_10k': [], 'dev': [], 'final': []},
            'complementary_pairs': {'train': accepted_pairs},
            'semantic_admission': {'policy': deepcopy(policy), 'not_a_full_individual_image_audit': True,
                                   'known_uncertainty_veto_applied': True}}
    summary = {'source_pairs': len(grouped), 'source_qa': len(rows),
               'admitted_pairs': len(accepted_pairs), 'admitted_qa': len(kept_rows),
               'excluded_pairs': len(grouped) - len(accepted_pairs),
               'exclusion_reason_pair_counts': dict(Counter(reason for decision in decisions
                                                              for reason in decision['exclusion_reasons'])),
               'admitted_qa_review_status_counts': dict(Counter(item['status'] for item in qa_admissions
                                                                 if item['admitted'])),
               'new_model_forwards': 0, 'training_updates': 0, 'label_edits': 0,
               'original_images_reused': True, 'heldout_evaluation_executed': False}
    return view, decisions, qa_admissions, summary
