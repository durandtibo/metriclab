r"""Sentinel values used across metriclab.

:data:`NOT_SET` is a singleton used as a default argument value when
``None`` is a valid user-supplied argument and must be distinguished
from "not provided".  It evaluates to ``False`` in boolean contexts and
has the string representation ``<NotSet>``.

Example:
    ```pycon
    >>> from metriclab.utils.sentinel import NOT_SET
    >>> bool(NOT_SET)
    False
    >>> repr(NOT_SET)
    '<NotSet>'

    ```
"""

from __future__ import annotations

__all__ = ["NOT_SET"]


class _NotSet:
    """Sentinel class to indicate a value is not set."""

    def __repr__(self) -> str:
        return "<NotSet>"

    def __bool__(self) -> bool:
        return False


NOT_SET = _NotSet()
