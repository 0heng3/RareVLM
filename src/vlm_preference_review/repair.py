"""Re-mine all four existing responses, with a frozen precision-first rule set."""
from collections import Counter
import re
from vlm_deep.common import NORMALIZER, consensus
from vlm_preference_data.quality import compact, quality_check

ALIASES = [frozenset({'coca cola', 'coke'})]

def literal_question(question):
    q = question.lower().strip()
    return bool(re.search(r'\b(?:what|which) (?:is |are )?(?:the )?(?:number|numbers|letters|letter|words|word|text|year|date|percentage|percent|title|name|brand|company|author|label)\b', q)
                or re.search(r'\bwhat (?:does|do) .+ (?:say|read)\b', q)
                or re.search(r'\bwho (?:wrote|is the author|is the manufacturer)\b', q)
                or re.search(r'\bhow many (?:ml|milliliters|pints|ounces|hours)\b', q))

def pair_reasons(row, chosen, rejected):
    result = quality_check(row, chosen, rejected)
    reasons = list(result['reasons'])
    if result['chosen_normalized_votes'] < 8: reasons.append('chosen_fewer_than8_reference_votes')
    if not literal_question(row['question']): reasons.append('outside_literal_reading_question_policy')
    r = compact(rejected)
    # Both directions, including partial numeric strings; deliberately sacrifices recall.
    if r and any(len(compact(a)) >= 2 and len(r) >= 2 and
                 (r in compact(a) or compact(a) in r) for a in row['answers']):
        reasons.append('bidirectional_reference_span_overlap')
    normalized = {NORMALIZER(chosen), NORMALIZER(rejected)}
    if any(normalized <= group for group in ALIASES): reasons.append('known_semantic_alias')
    return sorted(set(reasons))

def repair_pair(row, record):
    texts = [s['prediction'] for s in record['samples']]
    scores = [consensus(a, row['answers']) for a in texts]
    if scores != record['rewards']: raise ValueError('Frozen score differs')
    votes = Counter(NORMALIZER(a) for a in row['answers'])
    good = sorted([i for i,a in enumerate(texts) if a.strip() and scores[i] >= .9],
                  key=lambda i: (-votes[NORMALIZER(texts[i])], -scores[i], i))
    bad = [i for i,a in enumerate(texts) if a.strip() and scores[i] == 0]
    for c in good:
        for r in bad:
            if not pair_reasons(row, texts[c], texts[r]):
                return {'question_id': row['question_id'], 'image_id': row['image_id'],
                        'question': row['question'], 'chosen': texts[c], 'rejected': texts[r],
                        'chosen_index': c, 'rejected_index': r, 'chosen_score': scores[c], 'rejected_score': scores[r],
                        'chosen_normalized_votes': votes[NORMALIZER(texts[c])],
                        'label_source': 'ten-reference rule-derived weak preference;not direct human preferences',
                        'semantic_human_preference_verified': False,
                        'source_prediction_lock_sha256': record['input_lock_sha256']}
    return None
