"""Fixed-initial greedy-answer CE rehearsal; independent prospective protocol."""

SCHEMA = "rarevlm.object_behavior_replay.train.v1"
CANDIDATE = "teacher_ce32"
CONTROL = "gt_ce32"
CONDITIONS = (CANDIDATE, CONTROL)
