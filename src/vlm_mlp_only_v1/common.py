from rarevlm_paths import PROJECT_ROOT
from vlm_language_mlp_v1.common import PARENT,DIVERSITY,actor as joint_actor,chosen_batch,read,is_mlp

JOINT=PARENT.parent/'vlm_language_mlp_v1'
ROOT=PARENT.parent/'vlm_mlp_only_v1'
TOTAL_PARAMETERS=2820096
PROTOCOL=PROJECT_ROOT/'docs/VLM_MLP_ONLY_V1_PROTOCOL.md'
RESULT=PROJECT_ROOT/'docs/VLM_MLP_ONLY_V1_RESULTS.md'
PAGE=PROJECT_ROOT/'docs/learning/冻结注意力适配_只训练MLP.html'


def frozen_qv_hash(model):
    from vlm_object_train.train import tensor_hash
    return tensor_hash((n,p) for n,p in model.named_parameters() if 'lora_' in n and not is_mlp(n))


def actor(training=False,adapter=None):
    from vlm_deep.common import write_json
    model,processor,binding=joint_actor(training=training,adapter=adapter)
    if training:
        for n,p in model.named_parameters():
            if 'lora_' in n:p.requires_grad_(is_mlp(n))
        active=[p for p in model.parameters() if p.requires_grad]
        qv=[p for n,p in model.named_parameters() if 'lora_' in n and not is_mlp(n)]
        if len(active)!=216 or sum(p.numel() for p in active)!=TOTAL_PARAMETERS or len(qv)!=144 or sum(p.numel() for p in qv)!=1843200:
            raise ValueError('Actual MLP-only update scope differs')
        model._frozen_qv_initial_sha256=frozen_qv_hash(model)
        if model._frozen_qv_initial_sha256!=read(PARENT/'PLAN.json')['initial_lora_sha256']:
            raise ValueError('Frozen q/v must exactly preserve original SFT values')
        write_json(ROOT/'runs/candidate/train/RUNTIME_SCOPE_REVIEW.json',{'status':'passed','trainable_factors':216,'trainable_parameters':TOTAL_PARAMETERS,
                   'frozen_qv_factors':144,'frozen_qv_parameters':1843200,'frozen_qv_sha256':model._frozen_qv_initial_sha256,
                   'new_optimizer_updates':0,'forward_graph_adapter_parameters':4663296})
    return model,processor,binding
