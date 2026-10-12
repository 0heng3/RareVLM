"""Separate repository assets from installed sources after the ``src`` move.

Research provenance keeps its historical logical source names (for example,
``vlm_deep/common.py``); resolve those names with :func:`source_path`. Configs,
tools, data, and artifact defaults belong to :func:`project_path` instead.
"""
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SOURCE_ROOT.parent


def project_path(value: str | Path) -> Path:
    """Resolve a repository asset without altering an explicit absolute path."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def source_path(value: str | Path) -> Path:
    """Resolve a logical source name without rewriting recorded hashes."""
    path = Path(value)
    return path if path.is_absolute() else SOURCE_ROOT / path
