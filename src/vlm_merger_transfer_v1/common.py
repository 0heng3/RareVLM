from rarevlm_paths import PROJECT_ROOT
from vlm_grpo_iteration_v1.common import chosen_sample, chosen_batch, read

ROOT = PROJECT_ROOT.parents[1] / 'outputs/vlm_merger_transfer_v1'
OLD = ROOT.parent / 'vlm_coverage_sft_v2'
ROLES = ('initial', 'candidate', 'qv_control')


def actor(training=False, adapter=None):
    from vlm_grpo_iteration_v1.common import actor as original
    return original(training=training, adapter=adapter or ROOT/'initial_adapter')
