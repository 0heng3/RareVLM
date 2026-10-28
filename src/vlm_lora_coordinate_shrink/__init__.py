"""Fixed factor-coordinate export from an explicitly separate unshrunk producer."""
CONFIG_SCHEMA='rarevlm.lora_coordinate_shrink.v1'
SCHEMA='rarevlm.lora_coordinate_shrink.producer_train.v1'
CANDIDATE='lora_half32'
CONTROL='teacher_kl32'
CONDITIONS=(CANDIDATE,)
