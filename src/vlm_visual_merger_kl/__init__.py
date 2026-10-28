"""Composite frozen language SFT plus trainable visual merger; independent study."""
SCHEMA=CONFIG_SCHEMA="rarevlm.visual_merger_kl.train.v1"
CANDIDATE="merger_kl32"
CONTROL="teacher_kl32"
CONDITIONS=(CANDIDATE,)
