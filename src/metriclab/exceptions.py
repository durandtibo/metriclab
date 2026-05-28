r"""Define the base exceptions."""

from __future__ import annotations

__all__ = ["EmptyMetricError"]


class EmptyMetricError(Exception):
    r"""Raised when an empty metric is evaluated.

    This exception is raised by metric functions when there are no
    valid predictions to evaluate and ``raise_empty=True`` (the
    default).

    Example:
        ```pycon
        >>> from metriclab.exceptions import EmptyMetricError
        >>> from metriclab.functional import accuracy
        >>> try:
        ...     accuracy(y_true=[], y_pred=[])
        ... except EmptyMetricError as e:
        ...     print(type(e).__name__)
        EmptyMetricError

        ```
    """
