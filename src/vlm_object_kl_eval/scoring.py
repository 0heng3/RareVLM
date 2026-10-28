"""Use the unchanged pinned official parser; strict output audits stay separate."""
from vlm_object_eval.scoring import diagnostics, first_yesno, metrics, official_prediction

__all__ = ["diagnostics", "first_yesno", "metrics", "official_prediction"]
