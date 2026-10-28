from rarevlm_paths import PROJECT_ROOT
from vlm_grpo_iteration_v1.common import actor, chosen_batch, read

ROOT=PROJECT_ROOT.parents[1]/'outputs/vlm_reference_replay_v1'
TRANSFER=ROOT.parent/'vlm_merger_transfer_v1'
DATA_SOURCE=ROOT.parent/'vlm_length_replay_v1'
ROLES=('plain_transfer','soft_kl','hard_replay')


def cache_path(step,micro):return ROOT/'teacher_cache'/f'{step:03d}_{micro}.pt'
