from collections import Counter,defaultdict
from rarevlm_paths import PROJECT_ROOT
from vlm_grpo_iteration_v1.common import actor,chosen_batch,chosen_sample,read
from vlm_deep.common import content_sha

PARENT=PROJECT_ROOT.parents[1]/'outputs/vlm_reference_replay_v1'
ROOT=PARENT.parent/'vlm_replay_diversity_v1'
MINING=PARENT.parent/'vlm_error_coverage_mining_v1'
PROTOCOL=PROJECT_ROOT/'docs/VLM_REPLAY_DIVERSITY_V1_PROTOCOL.md'
RESULT=PROJECT_ROOT/'docs/VLM_REPLAY_DIVERSITY_V1_RESULTS.md'
PAGE=PROJECT_ROOT/'docs/learning/回放多样性_重复512与覆盖1024.html'
ROLES=('repeat512','unique1024')


def cell(row):return row['frequency_slice'],row['stratum']


def order(row):return content_sha(['replay_diversity_v1',row['question_id']])


def source_coefficient(source,beta):
    # The source loss masses are invariant to doubling replay microbatches.
    return .5/8 if source=='hard' else (.5+beta)/16


def verify():
    from vlm_deep.common import sha_file
    plan=read(ROOT/'PLAN.json')
    for path,digest in {**plan['files'],**plan['images']}.items():
        if sha_file(path)!=digest:raise ValueError('Frozen source/input changed: '+path)
    return plan


def schedules(original,old_replay,new_rows):
    buckets=defaultdict(list)
    for q in new_rows:buckets[cell(q)].append(q)
    for values in buckets.values():values.sort(key=order)
    used=Counter();mapping={}
    for q in old_replay:
        c=cell(q);mapping[q['question_id']]=buckets[c][used[c]]['question_id'];used[c]+=1
    result={r:[] for r in ROLES}
    for block in original:
        for role in ROLES:
            micros=[]
            for i,ids in enumerate(block['micros']):
                micros.append(ids)
                if i in (0,2):micros.append([mapping[q] for q in ids] if role=='unique1024' else ids)
            result[role].append({'micros':micros})
    return result,mapping
