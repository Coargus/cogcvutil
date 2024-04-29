"""Path utility."""

from __future__ import annotations

from pathlib import Path


def validate_path(path: str) -> Path:
    """Validate path."""
    if isinstance(path, str):
        return Path(path)
    else:
        return path
