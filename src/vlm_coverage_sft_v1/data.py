from collections import Counter,defaultdict
import hashlib
import random


def cell(row):
    return (row['review_error_type'],row['frequency_slice'],row['stratum'])


def order(namespace,qid):
    return hashlib.sha256((namespace+':'+qid).encode()).hexdigest()


def schedule(hard,replay):
    if len(hard)!=128 or len(replay)!=512:raise ValueError('Require128 hard and512 replay')
    groups=defaultdict(list)
    for q in hard:groups[cell(q)].append(q)
    if len(groups)>32:raise ValueError('Too many occupied strata for32 controls')
    quota={key:1 for key in groups}
    while sum(quota.values())<32:
        choices=[k for k,v in groups.items() if quota[k]<len(v)]
        key=max(choices,key=lambda k:(len(groups[k])/4-quota[k],k))
        quota[key]+=1
    subset={k:sorted(v,key=lambda q:order('coverage_sft_v1:subset',q['question_id']))[:quota[k]] for k,v in groups.items()}
    hard_stream=list(hard)*4;replay_stream=list(replay)
    rng=random.Random(42);rng.shuffle(hard_stream);rng.shuffle(replay_stream)
    cursor=Counter();control=[]
    for q in hard_stream:
        key=cell(q);control.append(subset[key][cursor[key]%len(subset[key])]);cursor[key]+=1
    result={}
    for role,stream in [('repeat32',control),('coverage128',hard_stream)]:
        updates=[]
        for step in range(64):
            h=stream[step*8:step*8+8];r=replay_stream[step*8:step*8+8]
            updates.append({'micros':[[q['question_id'] for q in r[:4]],[q['question_id'] for q in h[:4]],
                [q['question_id'] for q in r[4:]],[q['question_id'] for q in h[4:]]]})
        result[role]=updates
    return {'roles':result,'repeat_subset':[q['question_id'] for key in sorted(subset) for q in subset[key]],
        'strata':[{'key':list(k),'coverage_questions':len(groups[k]),'repeat_questions':quota[k]} for k in sorted(groups)]}
