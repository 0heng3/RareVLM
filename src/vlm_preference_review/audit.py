import hashlib
import math

CATEGORIES = {'clear_visual_preference', 'semantic_alias', 'answer_scope_or_partial',
              'question_or_entity_ambiguity', 'unresolved_visual_detail'}

def fixed_sample(pairs, prefix, n=64, exclude=()):
    excluded = set(exclude)
    population = [x for x in pairs if x['question_id'] not in excluded]
    if len({x['question_id'] for x in population}) != len(population):
        raise ValueError('Duplicate question in audit population')
    if len(population) < n: raise ValueError('Insufficient independent audit questions')
    return sorted(population, key=lambda x: hashlib.sha256((prefix + x['question_id']).encode()).hexdigest())[:n]

def wilson(success, n):
    if n <= 0 or not 0 <= success <= n: raise ValueError('Invalid binomial counts')
    z = 1.959963984540054; p = success/n; denominator = 1 + z*z/n
    center = (p + z*z/(2*n))/denominator
    radius = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n))/denominator
    return [center-radius, center+radius]

def audit_gate(records, plan):
    if len(records) != plan['sample_size']: raise ValueError('Incomplete visual audit')
    if [x['question_id'] for x in records] != plan['selected_question_ids']:
        raise ValueError('Audit selection/order changed')
    if any(x['category'] not in CATEGORIES or not x.get('note') for x in records):
        raise ValueError('Every audit item needs an explicit category and evidence note')
    clear = sum(x['category'] == 'clear_visual_preference' for x in records)
    interval = wilson(clear, len(records))
    return {'n': len(records), 'clear': clear, 'not_clear': len(records)-clear,
            'clear_fraction': clear/len(records), 'wilson95': interval,
            'gate_passed': clear/len(records) >= plan['clear_fraction_min'] and interval[0] >= plan['wilson95_lower_min'],
            'limitations': 'Nominal interval for assistant clear-preference judgments;not independent human truth or a full-population error rate'}
