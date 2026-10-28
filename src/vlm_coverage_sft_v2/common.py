from rarevlm_paths import PROJECT_ROOT
from vlm_grpo_iteration_v1.common import chosen_sample, chosen_batch, actor, read

ROOT = PROJECT_ROOT.parents[1] / 'outputs/vlm_coverage_sft_v2'
OLD = ROOT.parent / 'vlm_coverage_sft_v1'
FIT = ROOT.parent / 'vlm_fit_positive_control_v1'
ROLES = ('repeat32', 'coverage128')
