"""One precisely scoped identity projection for raw export-verification bindings.

Original locks remain byte-authoritative. This module never mutates a lock,
loads model weights, or treats a missing timing field as an already-stable view.
"""
from __future__ import annotations

from copy import deepcopy
import math
from numbers import Real
from typing import Any

_VOLATILE_KEY = 'verify_CPU_wall_seconds'
_VOLATILE_PATH = 'export_verification.' + _VOLATILE_KEY


def stable_binding_view(binding: dict[str, Any]) -> dict[str, Any]:
    """Copy a raw binding and remove only its validated engineering-time leaf.

    Both inputs to a comparison must be raw bindings containing this leaf. A
    stable view is a comparison value, never a replacement for a saved lock.
    """
    if type(binding) is not dict or type(binding.get('export_verification')) is not dict:
        raise ValueError('raw binding requires dictionary export_verification')
    verification = binding['export_verification']
    if _VOLATILE_KEY not in verification:
        raise ValueError('raw binding missing ' + _VOLATILE_PATH)
    value = verification[_VOLATILE_KEY]
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(_VOLATILE_PATH + ' must be a finite nonnegative real, not bool')
    try:
        valid = math.isfinite(value) and value >= 0
    except (TypeError, ValueError, OverflowError):
        valid = False
    if not valid:
        raise ValueError(_VOLATILE_PATH + ' must be a finite nonnegative real, not bool')
    view = deepcopy(binding)
    del view['export_verification'][_VOLATILE_KEY]
    return view


def _exact_equal(left: Any, right: Any) -> bool:
    """Preserve structure and types: scientific True/1 and 1/1.0 differ."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        if {(type(key), key) for key in left} != {(type(key), key) for key in right}:
            return False
        return all(_exact_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(_exact_equal(a, b) for a, b in zip(left, right))
    return bool(left == right)


def binding_identity_equal(saved: dict[str, Any], fresh: dict[str, Any]) -> bool:
    """Compare all raw scientific identity except the sole permitted time leaf.

    Invalid telemetry raises ValueError, rather than becoming a normal mismatch
    or silently accepting a stripped view. All other differences return False.
    """
    saved_view = stable_binding_view(saved)
    fresh_view = stable_binding_view(fresh)
    return _exact_equal(saved_view, fresh_view)
