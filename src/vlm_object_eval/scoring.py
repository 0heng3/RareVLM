"""Pinned official binary scoring plus explicitly separate first-token audits."""
from __future__ import annotations

import re

from vlm_posttrain_pope.scoring import metrics as official_metrics
from vlm_posttrain_pope.scoring import official_prediction, validity_diagnostic

PROMPT_SUFFIX = "\nAnswer yes or no."
_FIRST = re.compile(r"^\s*(yes|no)(?=$|[\s.,!?;:])", re.IGNORECASE | re.ASCII)


def first_yesno(text):
    """First lexical answer must be yes/no; never search later in an answer."""
    if not isinstance(text, str):
        raise TypeError("prediction must be a string")
    match = _FIRST.match(text)
    return match.group(1).lower() if match else None


def diagnostics(text):
    value = first_yesno(text)
    return {**validity_diagnostic(text), "first_yesno_value": value,
        "first_yesno_invalid": value is None,
        "first_yesno_official_disagreement": value is not None and value != official_prediction(text)}


def metrics(records):
    rows = list(records)
    result = official_metrics(rows)
    for name, selected in [("overall", rows), *[(key, [r for r in rows if r["category"] == key])
            for key in sorted(result["by_category"])]]:
        summary = result["overall"] if name == "overall" else result["by_category"][name]
        invalid = sum(first_yesno(r["prediction"]) is None for r in selected)
        correct = sum(first_yesno(r["prediction"]) == r["label"] for r in selected)
        summary.update(first_yesno_invalid_count=invalid,
            first_yesno_invalid_rate=invalid / len(selected) if selected else None,
            first_yesno_accuracy=correct / len(selected) if selected else None)
    result["first_yesno_contract"] = {"pattern": _FIRST.pattern,
        "invalid_accuracy_denominator": "all questions; invalid first answer scores zero",
        "primary_metric": "unchanged pinned official parser, including all invalid answers"}
    return result
