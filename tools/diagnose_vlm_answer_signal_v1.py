"""Describe saved answer likelihoods and ranking logit derivatives; no model IO.

The conditional two-token loss below is diagnostic, not the applied vocabulary CE.
Its derivatives must not be called shared-parameter gradients.
"""

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics


OUTPUTS = Path('/opt/data/private/longtail_vlm/rarevlm_pilot/outputs')
READOUT = OUTPUTS / 'vlm_fixed_error_readout_v1'
STUDY = OUTPUTS / 'vlm_complementary_rank_v1'
DEST = OUTPUTS / 'vlm_deep_pipeline_diagnosis_v1' / 'answer_signal'


def read(path):
    return json.loads(path.read_text())


def rows(path):
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def describe(values):
    return ({'n': len(values), 'mean': statistics.mean(values),
             'median': statistics.median(values), 'min': min(values), 'max': max(values)}
            if values else {'n': 0})


def softplus(x):
    return max(x, 0) + math.log1p(math.exp(-abs(x)))


def sigmoid(x):
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1 + ex)


def main():
    sources = [READOUT / 'data/COHORT.json', READOUT / 'PAIR_DIAGNOSES.jsonl']
    cohort = read(sources[0])
    manifest_path = Path(cohort['manifests']['train_selected']['path'])
    sources.append(manifest_path)
    manifest = read(manifest_path)
    qa = {q['question_id']: q for q in manifest['splits']['dev']}
    pair_predictions = {
        (r['role'], r['pair_id']): r for r in rows(sources[1])
        if r['task'] == 'train_selected'}
    readouts = {}
    for role in ('initial', 'ce', 'ce_rank'):
        path = READOUT / 'runs' / role / 'train_selected/readout.jsonl'
        sources.append(path)
        readouts[role] = {r['question_id']: r for r in rows(path)}
        if set(readouts[role]) != set(qa):
            raise ValueError('Saved readouts and selected training questions differ')

    per_qa = []
    groups = defaultdict(list)
    pair_counts = Counter()
    for pair in cohort['pairs']['train_selected']:
        if pair['anchor_group'] != 'initial_wrong':
            continue
        pid = pair['pair_id']
        initial = pair_predictions['initial', pid]['generation']
        repaired = pair_predictions['ce_rank', pid]['generation']['both_correct']
        pair_group = 'rank_repaired_pair' if repaired else 'rank_unrepaired_pair'
        pair_counts[pair_group] += 1
        for side, qid in [('positive', pair['positive_question_id']),
                          ('negative', pair['negative_question_id'])]:
            target = qa[qid]['target']
            before_correct = initial[f'{side}_primary_score'] == 1
            final_correct = pair_predictions['ce_rank', pid]['generation'][f'{side}_primary_score'] == 1
            item = {'pair_id': pid, 'question_id': qid, 'target': target,
                    'pair_group': pair_group,
                    'initial_generation_correct': before_correct,
                    'rank_final_generation_correct': final_correct, 'models': {}}
            for role in ('initial', 'ce', 'ce_rank'):
                r = readouts[role][qid]
                item['models'][role] = {
                    'target_first_token_nll': -r[f'canonical_{target}_log_probability'],
                    'target_first_token_probability': math.exp(r[f'canonical_{target}_log_probability']),
                    'canonical_yes_no_probability_mass':
                        math.exp(r['canonical_yes_log_probability']) + math.exp(r['canonical_no_log_probability']),
                    'target_signed_canonical_margin': r['canonical_margin'] * (1 if target == 'yes' else -1),
                    'actual_generation': pair_predictions[role, pid]['generation'][f'{side}_prediction'],
                    'actual_generation_correct': pair_predictions[role, pid]['generation'][f'{side}_primary_score'] == 1,
                }
            item['rank_nll_delta'] = (item['models']['ce_rank']['target_first_token_nll']
                                      - item['models']['initial']['target_first_token_nll'])
            per_qa.append(item)
            groups['all_90_qa_in_initial_wrong_pairs'].append(item)
            groups[pair_group].append(item)
            if not before_correct:
                groups['initial_wrong_qa'].append(item)
                groups['initial_wrong_qa_rank_corrected' if final_correct else
                       'initial_wrong_qa_rank_still_wrong'].append(item)
                groups[f'initial_wrong_target_{target}'].append(item)
            else:
                groups['initial_correct_qa_within_initial_wrong_pairs'].append(item)

    summary = {}
    for group, values in groups.items():
        summary[group] = {'qa': len(values), 'by_model': {}}
        for role in ('initial', 'ce', 'ce_rank'):
            rv = [v['models'][role] for v in values]
            summary[group]['by_model'][role] = {
                'correct': sum(v['actual_generation_correct'] for v in rv),
                **{field: describe([v[field] for v in rv]) for field in (
                    'target_first_token_nll', 'target_first_token_probability',
                    'canonical_yes_no_probability_mass', 'target_signed_canonical_margin')}}
        summary[group]['rank_minus_initial_target_nll'] = describe([v['rank_nll_delta'] for v in values])
        summary[group]['rank_nll_decreased'] = sum(v['rank_nll_delta'] < 0 for v in values)

    trace_path = STUDY / 'runs/ce_rank/train/INPUT_TRACES.jsonl'
    sources.append(trace_path)
    human = [r for r in rows(trace_path) if r['source'] == 'human']
    derivative_rows = []
    for r in human:
        signed = r['first_yes_minus_no_margin'] * (1 if r['target'] == 'yes' else -1)
        gap = r['pair_margin_gap']
        rank_derivative = (0.1 / 4) * sigmoid(-gap)
        item = {
            'update': r['update'], 'question_id': r['question_id'], 'pair_id': r['pair_id'],
            'target_signed_margin': signed, 'gap': gap,
            'conditional_binary_nll': softplus(-signed),
            'rank_derivative_magnitude_per_canonical_target_logit': rank_derivative,
            'nonpositive_target_margin': signed <= 0,
        }
        if signed <= 0:
            # Actual full-vocabulary p(target) <= 1/2 when another canonical
            # answer has at least as high a logit. With 16 QA and 3 targets,
            # |d L_CE / d z_target| >= (1/16)*(1/3)*(1/2) = 1/96.
            assert r['assistant_tokens'] == 3
            item['applied_ce_target_logit_derivative_lower_bound'] = 1 / 96
            item['rank_over_ce_target_logit_derivative_upper_bound'] = rank_derivative * 96
        derivative_rows.append(item)
    relative_errors = [r for r in derivative_rows if r['nonpositive_target_margin']]
    ordered_errors = [r for r in relative_errors if r['gap'] > 0]
    derivative_summary = {
        'scope': 'online teacher-forcing logit derivatives; not generated errors or parameter gradients',
        'human_qa_exposures': len(human),
        'nonpositive_target_margin_qa_exposures': len(relative_errors),
        'ordered_pair_nonpositive_margin_qa_exposures': len(ordered_errors),
        'ordered_error_rank_over_ce_upper_bound': describe([
            r['rank_over_ce_target_logit_derivative_upper_bound'] for r in ordered_errors]),
        'ordered_error_rank_at_most_10_percent_ce_count': sum(
            r['rank_over_ce_target_logit_derivative_upper_bound'] <= 0.1 for r in ordered_errors),
        'ordered_error_rank_at_most_1_percent_ce_count': sum(
            r['rank_over_ce_target_logit_derivative_upper_bound'] <= 0.01 for r in ordered_errors),
        'center_derivative_of_ranking_loss': 0,
        'formulas': {
            'applied_ce': 'sum_QA(mean_3_assistant_tokens(CE))/16',
            'rank': '0.1*sum_4_pairs(softplus(-gap))/4',
            'rank_target_logit_derivative_magnitude': '0.025*sigmoid(-gap)',
            'ce_target_logit_derivative_lower_bound_for_nonpositive_margin': '1/96',
            'ratio_upper_bound': '2.4*sigmoid(-gap)'},
    }
    result = {
        'status': 'completed_cpu_saved_records_only', 'model_loads': 0, 'optimizer_updates': 0,
        'scope': 'Exploratory selected training cohort; not overall training accuracy or causal ablation',
        'pair_counts': dict(pair_counts), 'answer_signal_groups': summary,
        'ranking_logit_derivatives': derivative_summary,
        'limitations': [
            'Prefix-only dense readout differs from teacher-forcing tail-head GEMM shape; near-boundary rounding can differ.',
            'Full-vocabulary first-target NLL is saved-readout NLL, not the three-token online training CE.',
            'No EOS/newline per-token CE is saved: their actual loss or parameter-gradient share cannot be recovered.',
            'Ranking/CE logit derivative ratio bounds do not measure shared-parameter gradient norms or interference.',
            'This diagnosis does not establish that removing format tokens, changing loss or increasing updates improves held-out performance.'],
        'sources': [{'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in sources],
    }
    DEST.mkdir(parents=True, exist_ok=True)
    for name, values in [('QA_ANSWER_SIGNAL.jsonl', per_qa), ('RANK_LOGIT_DERIVATIVES.jsonl', derivative_rows)]:
        (DEST / name).write_text(''.join(json.dumps(x, ensure_ascii=False, allow_nan=False) + '\n' for x in values))
    (DEST / 'ANALYSIS.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'pair_counts': result['pair_counts'], 'ranking': derivative_summary,
                      'answer_groups': {k: {'qa': v['qa'], 'correct_initial_rank': [v['by_model'][r]['correct'] for r in ('initial', 'ce_rank')],
                      'mean_first_nll_initial_rank': [v['by_model'][r]['target_first_token_nll']['mean'] for r in ('initial', 'ce_rank')],
                      'nll_decreased': v['rank_nll_decreased']} for k, v in summary.items()}},
                     ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
