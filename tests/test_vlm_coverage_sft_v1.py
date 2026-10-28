from collections import Counter
from vlm_coverage_sft_v1.data import cell,schedule


def test_coverage_changes_unique_errors_without_changing_slots_or_replay():
    hard=[{'question_id':f'h{i}','review_error_type':'spelling' if i<127 else 'scale','frequency_slice':'rare','stratum':'single'} for i in range(128)]
    replay=[{'question_id':f'r{i}'} for i in range(512)]
    result=schedule(hard,replay);lookup={q['question_id']:q for q in hard}
    exposures={}
    for role,updates in result['roles'].items():
        hard_ids=[qid for u in updates for m in (1,3) for qid in u['micros'][m]]
        replay_ids=[qid for u in updates for m in (0,2) for qid in u['micros'][m]]
        assert len(updates)==64 and len(hard_ids)==len(replay_ids)==512
        assert Counter(replay_ids)==Counter(q['question_id'] for q in replay)
        exposures[role]=hard_ids
    assert len(set(exposures['repeat32']))==32 and len(set(exposures['coverage128']))==128
    assert set(Counter(exposures['coverage128']).values())=={4}
    assert all(cell(lookup[a])==cell(lookup[b]) for a,b in zip(exposures['repeat32'],exposures['coverage128']))
    assert result==schedule(hard,replay)
