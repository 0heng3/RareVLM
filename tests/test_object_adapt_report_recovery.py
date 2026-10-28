"""Minimal standalone CPU contracts for the explicit metadata-only recovery."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[1] / "tools/recover_object_adapt_report.py"
spec = importlib.util.spec_from_file_location("object_report_recovery_contract", path)
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)


def rows():
    return [{"question_id": "q", "image_id": "i", "score": .6, "input_ids": [151655, 1],
        "labels": [-100, 1], "pixel_values": {"shape": [4, 1176], "sha256": "a" * 64},
        "image_grid_thw": [1, 2, 2], "prediction": "yes"}]


@pytest.mark.parametrize("alias", ["missing", None, ""])
def test_only_empty_text_category_aliases_are_removed(alias):
    value = rows()
    if alias != "missing":
        value[0]["category"] = alias
    original = deepcopy(value)
    normalized, proof = recovery.normalize_category(value, "text")
    assert "category" not in normalized[0]
    assert value == original
    assert proof["noncategory_fields_sha256_before"] == proof["noncategory_fields_sha256_after"]
    assert proof["score_fields_sha256_before"] == proof["score_fields_sha256_after"]


@pytest.mark.parametrize("category", ["random", "bird", 0, False])
def test_nonempty_text_category_is_a_hard_failure(category):
    value = rows()
    value[0]["category"] = category
    with pytest.raises(ValueError, match="not an allowed"):
        recovery.normalize_category(value, "text")


@pytest.mark.parametrize("task", ["object", "pope"])
def test_binary_task_rows_are_returned_entirely_unchanged(task):
    value = rows()
    value[0].update(category="random", label="yes")
    original = deepcopy(value)
    normalized, proof = recovery.normalize_category(value, task)
    assert normalized is value
    assert value == original
    assert proof["normalized_categories"] == 0


def test_no_noncategory_value_score_order_or_source_row_is_changed():
    value = rows() + [{**deepcopy(rows()[0]), "question_id": "q2", "score": 1 / 3, "category": None}]
    before = deepcopy(value)
    normalized, proof = recovery.normalize_category(value, "text")
    for raw, actual in zip(before, normalized):
        assert actual == {key: item for key, item in raw.items() if key != "category"}
    assert value == before
    assert proof["scores_exactly_equal"] is True
    assert proof["noncategory_fields_exactly_equal"] is True
