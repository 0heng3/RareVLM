"""One serialization compatibility rule, without changing an original object."""
from copy import deepcopy
import hashlib
import json


def content_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def legacy_anchor_view(anchor,old_schedule,new_schedule,plan_schedule,train):
    """Derive only a historically absent digest from the locked complete schedule."""
    expected_keys={'object_question_ids_by_update','replay_question_ids_by_update','learning_rates_by_update',
        'training_microbatch_questions','object_microbatches_by_update','replay_microbatches_by_update'}
    if set(old_schedule)!=expected_keys or old_schedule!=new_schedule or new_schedule!=plan_schedule:
        raise ValueError('complete old/new/PLAN schedules differ')
    digest=content_sha(old_schedule)
    if anchor.get('schedule_sha256')!=digest or train.get('schedule_sha256')!=digest:
        raise ValueError('locked result to full schedule content chain differs')
    if old_schedule['training_microbatch_questions']!=4:
        raise ValueError('original microbatch width differs')
    for key in ('object_question_ids_by_update','replay_question_ids_by_update','learning_rates_by_update',
                'object_microbatches_by_update','replay_microbatches_by_update'):
        if len(old_schedule[key])!=181:raise ValueError('complete181 schedule required')
    for source,count,total in [('object',16,2888),('replay',32,5792)]:
        updates=old_schedule[source+'_question_ids_by_update'];micro=old_schedule[source+'_microbatches_by_update']
        if sum(map(len,updates))!=total:raise ValueError('complete source exposure count differs')
        for index,(questions,batches) in enumerate(zip(updates,micro)):
            expected=8 if source=='object' and index==180 else count
            if (len(questions)!=expected or any(not isinstance(q,str) for q in questions)
                or batches!=[questions[i:i+4] for i in range(0,len(questions),4)]):
                raise ValueError('actual QA grouping or four-question boundaries differ')
        flat=[q for group in updates for q in group]
        if len(set(flat))!=total:raise ValueError('source QA stream must be unique')
        order_key=source+'_order_sha256';micro_key=source+'_microbatch_order_sha256'
        if anchor.get(order_key)!=content_sha(updates) or train.get(order_key)!=content_sha(updates):
            raise ValueError('locked source QA order digest differs')
        if source=='object' and (anchor.get(micro_key)!=content_sha(micro) or train.get(micro_key)!=content_sha(micro)):
            raise ValueError('locked object microbatch order differs')
    derived=content_sha(old_schedule['replay_microbatches_by_update'])
    if train.get('replay_microbatch_order_sha256')!=derived:
        raise ValueError('new replay microbatch digest differs from actual schedule')
    if 'replay_microbatch_order_sha256' in anchor and anchor['replay_microbatch_order_sha256']!=derived:
        raise ValueError('declared historical replay microbatch digest differs; not a missing-field case')
    view=deepcopy(anchor);missing='replay_microbatch_order_sha256' not in view
    view['replay_microbatch_order_sha256']=derived
    if {k:v for k,v in view.items() if k!='replay_microbatch_order_sha256'}!={k:v for k,v in anchor.items() if k!='replay_microbatch_order_sha256'}:
        raise ValueError('compatibility changed another historical field')
    return view,{'scope':'only missing old serialized replay microbatch digest; in-memory view',
        'historical_field_was_absent':missing,'derived_replay_microbatch_order_sha256':derived,
        'complete_schedule_content_sha256':digest,'schedule_updates':181,'microbatch_questions':4,
        'old_schedule_originally_bound_by':'locked old TRAIN_RESULT.schedule_sha256 content chain; no original direct file lock claimed',
        'all_other_anchor_fields_unchanged':True}
