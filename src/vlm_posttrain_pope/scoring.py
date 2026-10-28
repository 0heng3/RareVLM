"""Pinned POPE scoring, with separate format and denominator diagnostics.

All rates are fractions in [0, 1], not percentages. Every supplied record is
counted, including empty or invalid answers. No quality/progression gate is
defined here. Specificity and strict answer validity are diagnostics only.
"""
from collections import defaultdict
from collections.abc import Iterable, Mapping
import re
from typing import Any


OFFICIAL_COMMIT = "08d957b917e5a378a2f99d35b6293c536a66298b"
OFFICIAL_EVALUATE_SHA256 = "f0371dcec5f1bdbde017d1a02f5a9a04c53cb561a2874f3fb8f0718ce3d87036"
PARSER_CONTRACT = {
    "upstream": "https://github.com/AoiDragon/POPE",
    "commit": OFFICIAL_COMMIT,
    "evaluate_py_sha256": OFFICIAL_EVALUATE_SHA256,
    "first_sentence": "text.split('.')[0] if '.' occurs; ASCII full stop only",
    "commas": "remove all ASCII commas in the retained first sentence",
    "tokenization": "split(' '), without strip, lower, or generic whitespace splitting",
    "negative_tokens": ["No", "no", "not"],
    "otherwise": "yes, including empty and malformed answers",
}
STRICT_VALID_BINARY_DEFINITION = (
    "After Python str.strip(), the complete answer must be ASCII yes or no "
    "(case insensitive), optionally followed immediately by exactly one ASCII "
    "period, comma, exclamation mark, or question mark. No other text, internal "
    "whitespace, or punctuation is allowed. This diagnostic never changes the "
    "official prediction or denominator."
)
_STRICT_BINARY = re.compile(r"(yes|no)[.,!?]?", flags=re.IGNORECASE | re.ASCII)


def official_prediction(text: str) -> str:
    """Use the exact answer parser from the pinned official evaluate.py.

    In particular, ``NO``, ``No!``, and ``''`` become ``'yes'``. These behaviors
    are preserved rather than corrected by the format diagnostic.
    """
    if not isinstance(text, str):
        raise TypeError("POPE prediction must be a string")
    if text.find('.') != -1:
        text = text.split('.')[0]
    text = text.replace(',', '')
    words = text.split(' ')
    if 'No' in words or 'not' in words or 'no' in words:
        return 'no'
    return 'yes'


def validity_diagnostic(text: str) -> dict[str, Any]:
    """Describe complete-answer binary formatting, independently of scoring."""
    if not isinstance(text, str):
        raise TypeError("POPE prediction must be a string")
    stripped = text.strip()
    match = _STRICT_BINARY.fullmatch(stripped)
    strict_value = match.group(1).lower() if match else None
    official_value = official_prediction(text)
    return {
        "strict_valid_binary": match is not None,
        "strict_binary_value": strict_value,
        "reason": "valid" if match else ("empty" if not stripped else "not_complete_binary_answer"),
        "official_prediction": official_value,
        "strict_official_disagreement": bool(match and strict_value != official_value),
    }


def _summarize(rows: list[tuple[str, str, bool]]) -> dict[str, Any]:
    n = len(rows)
    tp = sum(pred == 'yes' and label == 'yes' for pred, label, _ in rows)
    fp = sum(pred == 'yes' and label == 'no' for pred, label, _ in rows)
    tn = sum(pred == 'no' and label == 'no' for pred, label, _ in rows)
    fn = sum(pred == 'no' and label == 'yes' for pred, label, _ in rows)
    invalid = sum(not valid for _, _, valid in rows)
    denominators = {
        "accuracy": n,
        "precision": tp + fp,
        "recall": tp + fn,
        "f1": 2 * tp + fp + fn,
        "yes_rate": n,
        "invalid_rate": n,
        "specificity": tn + fp,
    }
    numerators = {
        "accuracy": tp + tn, "precision": tp, "recall": tp, "f1": 2 * tp,
        "yes_rate": tp + fp, "invalid_rate": invalid, "specificity": tn,
    }
    rates = {key: numerators[key] / denominator if denominator else None
        for key, denominator in denominators.items()}
    # The official script evaluates precision, recall, then 2*p*r/(p+r).
    # A count-defined F1 of zero is retained even when that script would raise.
    official_would_raise = (
        n == 0 or tp + fp == 0 or tp + fn == 0 or tp == 0
    )
    return {
        "n": n, "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "n_yes": tp + fp, "n_invalid": invalid,
        "n_label_yes": tp + fn, "n_label_no": tn + fp,
        **rates,
        "zero_denominator": {key: value == 0 for key, value in denominators.items()},
        "undefined_metrics": [key for key, value in rates.items() if value is None],
        "official_formula_would_raise": official_would_raise,
    }


def metrics(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Score raw ``prediction`` text and yes/no ``label`` per ``category``.

    Labels must be exactly ``'yes'`` or ``'no'`` and category must be a nonempty
    string. A saved ``official_prediction``, if supplied, is checked against a
    fresh parse of the raw answer; it cannot silently override that answer.

    F1 is 2*TP/(2*TP+FP+FN). It is zero when errors exist but TP=0, and None
    when that count denominator is zero. Other undefined ratios are None.
    ``official_formula_would_raise`` explicitly identifies the original
    script's zero-division cases, including precision=recall=0. No rows are
    discarded to make any ratio defined.
    """
    overall: list[tuple[str, str, bool]] = []
    categories: dict[str, list[tuple[str, str, bool]]] = defaultdict(list)
    for index, record in enumerate(records):
        label = record['label']
        category = record['category']
        if label not in ('yes', 'no'):
            raise ValueError(f"record {index}: label must be exactly 'yes' or 'no'")
        if not isinstance(category, str) or not category.strip():
            raise ValueError(f"record {index}: category must be a nonempty string")
        diagnostic = validity_diagnostic(record['prediction'])
        prediction = diagnostic['official_prediction']
        if 'official_prediction' in record and record['official_prediction'] != prediction:
            raise ValueError(f"record {index}: saved official_prediction disagrees with raw prediction")
        row = (prediction, label, diagnostic['strict_valid_binary'])
        overall.append(row)
        categories[category].append(row)
    return {
        "overall": _summarize(overall),
        "by_category": {category: _summarize(categories[category]) for category in sorted(categories)},
        "parser_contract": {**PARSER_CONTRACT, "negative_tokens": list(PARSER_CONTRACT['negative_tokens'])},
        "validity_contract": {
            "strict_valid_binary": STRICT_VALID_BINARY_DEFINITION,
            "invalid_rate_denominator": "all supplied records, without exclusion or remapping",
            "diagnostic_only": ["strict_valid_binary", "invalid_rate", "specificity"],
            "rates_unit": "fraction_0_1",
            "f1": "2*TP/(2*TP+FP+FN); zero if errors exist, None if denominator is zero",
            "undefined_ratio": "None plus explicit zero_denominator flags; no epsilon smoothing",
        },
    }
