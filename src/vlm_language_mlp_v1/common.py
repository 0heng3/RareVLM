from rarevlm_paths import PROJECT_ROOT
from vlm_grpo_iteration_v1.common import actor as original_actor,chosen_batch,read

PARENT=PROJECT_ROOT.parents[1]/'outputs/vlm_reference_replay_v1'
ROOT=PARENT.parent/'vlm_language_mlp_v1'
DIVERSITY=PARENT.parent/'vlm_replay_diversity_v1'
MERGER=PARENT.parent/'vlm_merger_transfer_v1'
PROTOCOL=PROJECT_ROOT/'docs/VLM_LANGUAGE_MLP_V1_PROTOCOL.md'
RESULT=PROJECT_ROOT/'docs/VLM_LANGUAGE_MLP_V1_RESULTS.md'
PAGE=PROJECT_ROOT/'docs/learning/语言MLP适配_容量与迁移.html'
NEW_PARAMETERS=2820096
TOTAL_PARAMETERS=4663296


def actor(training=False,adapter=None):
    return original_actor(training=training,adapter=adapter or ROOT/'initial_adapter')


def is_mlp(name):
    return '.mlp.' in name and 'lora_' in name and '.visual.' not in name
