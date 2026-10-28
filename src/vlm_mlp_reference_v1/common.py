from rarevlm_paths import PROJECT_ROOT
from vlm_language_mlp_v1.common import actor as original_actor,chosen_batch,read,is_mlp
from vlm_mlp_only_v1.common import frozen_qv_hash

PARENT=PROJECT_ROOT.parents[1]/'outputs/vlm_reference_replay_v1'
SCOPE=PARENT.parent/'vlm_mlp_only_v1'
JOINT=PARENT.parent/'vlm_language_mlp_v1'
DIVERSITY=PARENT.parent/'vlm_replay_diversity_v1'
ROOT=PARENT.parent/'vlm_mlp_reference_v1'
PROTOCOL=PROJECT_ROOT/'docs/VLM_MLP_REFERENCE_V1_PROTOCOL.md'
RESULT=PROJECT_ROOT/'docs/VLM_MLP_REFERENCE_V1_RESULTS.md'
PAGE=PROJECT_ROOT/'docs/learning/MLP输出分布保持_KL与硬回放.html'


def actor(training=False,adapter=None):
    model,processor,binding=original_actor(training=training,adapter=adapter)
    if training:
        for name,p in model.named_parameters():
            if 'lora_' in name:p.requires_grad_(is_mlp(name))
        active=[p for p in model.parameters() if p.requires_grad]
        if len(active)!=216 or sum(p.numel() for p in active)!=2820096:raise ValueError('MLP-only update scope differs')
        model._frozen_qv_initial_sha256=frozen_qv_hash(model)
        if model._frozen_qv_initial_sha256!=read(PARENT/'PLAN.json')['initial_lora_sha256']:raise ValueError('Original q/v values differ')
    return model,processor,binding
