"""Resolve moved source files and audit historical hash bindings.

Migration is a recorded previous->current chain, not a claim that relocated,
layout-adjusted source has the same bytes as the executed historical source.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MIGRATION_PATH = ROOT / "metadata/ARCHITECTURE_MIGRATION.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def project_file(name):
    """Find an existing project path, including pre-src source references."""
    path = Path(name)
    if path.is_absolute():
        if path.exists():
            return path
        try:
            relative = path.relative_to(ROOT)
        except ValueError:
            return path
    else:
        relative = path
    current = ROOT / relative
    if current.exists():
        return current
    moved = ROOT / "src" / relative
    return moved if moved.exists() else current


def migration_entries():
    if not MIGRATION_PATH.exists():
        return {}
    payload = json.loads(MIGRATION_PATH.read_text())
    if payload.get("schema_version") != "rarevlm.architecture_migration.v1":
        raise ValueError("unknown architecture migration schema")
    entries = {}
    destinations = set()
    for item in payload["entries"]:
        for key in ("old_path", "new_path"):
            value = Path(item[key])
            if value.is_absolute() or ".." in value.parts:
                raise ValueError(f"unsafe migration path: {item[key]}")
            if not (ROOT / value).resolve().is_relative_to(ROOT):
                raise ValueError(f"migration path outside repository: {value}")
        old, new = item["old_path"], item["new_path"]
        if old in entries or new in destinations:
            raise ValueError(f"duplicate migration path: {old} -> {new}")
        previous, current = item["previous_sha256"], item["current_sha256"]
        if any(len(value) != 64 or any(c not in "0123456789abcdef" for c in value)
               for value in (previous, current)):
            raise ValueError(f"invalid migration digest: {old}")
        if item["changed_for_layout"] is not (previous != current):
            raise ValueError(f"migration change flag differs from hashes: {old}")
        entries[old] = item
        destinations.add(new)
    return entries


def verify_source(name, expected, *, allow_current=False):
    """Verify an original digest and the current bytes through the migration.

With allow_current=True a newly generated lock can bind current source bytes;
the return value explicitly distinguishes it from a historical binding.
"""
    given = Path(name)
    if given.is_absolute():
        try:
            relative = given.relative_to(ROOT)
        except ValueError as error:
            raise ValueError("source location outside RareVLM package") from error
    else:
        relative = given
    if ".." in relative.parts:
        raise ValueError(f"unsafe source path: {name}")
    item = migration_entries().get(relative.as_posix())
    path = ROOT / item["new_path"] if item else project_file(relative)
    if not path.resolve().is_relative_to(ROOT):
        raise ValueError(f"source location outside RareVLM package: {path}")
    actual = sha(path)
    if item:
        if actual != item["current_sha256"]:
            raise ValueError(f"current source differs from migration: {name}")
        historical = expected == item["previous_sha256"]
        if not historical and not (allow_current and expected == actual):
            raise ValueError(f"historical source differs from migration baseline: {name}")
        return {"path": str(path), "bound_sha256": expected, "current_sha256": actual,
                "binding": "historical_migration" if historical else "current",
                "changed_for_layout": item["changed_for_layout"],
                "migration_path": str(MIGRATION_PATH), "migration_sha256": sha(MIGRATION_PATH)}
    if actual != expected:
        raise ValueError(f"source differs from frozen binding: {name}")
    return {"path": str(path), "bound_sha256": expected, "current_sha256": actual,
            "binding": "direct", "changed_for_layout": False}
