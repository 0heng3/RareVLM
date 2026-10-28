"""Screen suspected lexical/annotation artifacts;never change benchmark scoring."""
from collections import Counter
import re
from vlm_deep.common import NORMALIZER

def compact(value):return re.sub(r'[\W_]+','',NORMALIZER(value),flags=re.UNICODE)

def quality_check(row,chosen,rejected):
    refs=[NORMALIZER(x) for x in row['answers']];c=NORMALIZER(chosen);r=NORMALIZER(rejected)
    votes=Counter(refs);reasons=[]
    if votes[c]<6:reasons.append('chosen_has_fewer_than6_normalized_human_votes')
    if votes['unanswerable']>=3 or c=='unanswerable':reasons.append('ambiguous_or_abstention_annotation')
    core_r=compact(rejected)
    if core_r and any(core_r==compact(x) for x in row['answers'] if compact(x)):
        reasons.append('punctuation_spacing_equivalent_to_a_reference')
    # Conservative: containing a reference is excluded even in a negated/wrong span.
    # This favors precision and can remove genuinely wrong answers;not a semantic judge.
    if any(len(compact(x))>=2 and compact(x) in core_r for x in row['answers']):
        reasons.append('rejected_contains_reference_core_possible_span_or_format_artifact')
    return {'candidate_admitted':not reasons,'reasons':reasons,'chosen_normalized_votes':votes[c],
        'semantic_human_preference_verified':False,
        'scope':'conservative training-data screen only;missed synonyms and false rejections remain;official metric unchanged'}
