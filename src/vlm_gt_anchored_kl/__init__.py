"""Matched GT-prefix supervised anchoring and independent full-reference KL."""
SCHEMA="rarevlm.gt_anchored_kl.train.v1"
CANDIDATE="gt_ce_kl32"
CONTROL="gt_kl32"
CONDITIONS=(CANDIDATE,CONTROL)
