r"""Sentinel values used across metriclab."""

from __future__ import annotations

__all__ = ["NOT_SET"]


class _NotSet:
    """Sentinel class to indicate a value is not set."""

    def __repr__(self) -> str:
        return "<NotSet>"

    def __bool__(self) -> bool:
        return False


NOT_SET = _NotSet()
