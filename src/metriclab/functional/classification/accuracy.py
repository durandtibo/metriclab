r"""Compute classification accuracy from array-like inputs.

This module powers :func:`metriclab.functional.accuracy` and supports
NumPy arrays, and Python sequences.
"""

from __future__ import annotations

__all__ = ["accuracy"]

from typing import TYPE_CHECKING, Any

from metriclab.exceptions import EmptyMetricError
from metriclab.results import AccuracyResult
from metriclab.utils.array import multi_contains_value, preprocess_1d, to_numpy_1d
from metriclab.utils.sentinel import NOT_SET

if TYPE_CHECKING:
    from metriclab.typing import ArrayLike
    from metriclab.utils.missing import MissingPolicy


def accuracy(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    missing_policy: MissingPolicy = "propagate",
    missing_values: Any = NOT_SET,
    raise_empty: bool = True,
) -> AccuracyResult:
    r"""Compute classification accuracy from ground-truth and predicted labels.

    Args:
        y_true: The ground truth target labels.
        y_pred: The predicted labels.
        missing_policy: The policy for handling values equal to
            ``missing_values``. ``"omit"`` removes rows where either
            input contains the missing value before computing the metric.
            ``"propagate"`` keeps the rows but returns a result with
            ``num_correct_predictions=nan`` when missing values are
            detected. ``"raise"`` raises ``ValueError`` instead.
        missing_values: The value to treat as missing. If not set,
            missing value handling is disabled regardless of
            ``missing_policy``.
        raise_empty: If ``True``, raises ``EmptyMetricError`` when
            there are no valid predictions. If ``False``, returns an
            ``AccuracyResult`` with ``num_predictions=0``.

    Returns:
        An :class:`~metriclab.results.AccuracyResult`. When
            ``missing_policy="propagate"`` finds missing values, the
            result keeps ``num_predictions`` and sets
            ``num_correct_predictions`` to ``nan``.

    Raises:
        EmptyMetricError: if there are no valid predictions and
            ``raise_empty`` is ``True``.
        ValueError: if ``missing_policy`` is invalid.
        ValueError: if ``y_true`` or ``y_pred`` contains
            ``missing_values`` and ``missing_policy`` is ``'raise'``.

    Example:
        >>> import numpy as np
        >>> from metriclab.functional import accuracy
        >>> accuracy(y_true=[1, 0, 0, 1, 1], y_pred=[1, 0, 1, 1, 1])
        AccuracyResult(num_correct_predictions=4, num_predictions=5)
        >>> accuracy(
        ...     y_true=np.array([1.0, 0.0, 0.0, 1.0, float("nan")]),
        ...     y_pred=np.array([1.0, 0.0, 0.0, 1.0, 1.0]),
        ...     missing_policy="omit",
        ...     missing_values=float("nan"),
        ... )
        AccuracyResult(num_correct_predictions=4, num_predictions=4)
        >>> accuracy(y_true=[], y_pred=[], raise_empty=False)
        AccuracyResult(num_correct_predictions=nan, num_predictions=0)
    """
    y_true, y_pred = preprocess_1d(
        arrays=[to_numpy_1d(y_true), to_numpy_1d(y_pred)],
        missing_policy=missing_policy,
        missing_values=missing_values,
    )

    num_predictions = y_true.size
    if num_predictions == 0:
        if raise_empty:
            msg = (
                "Cannot compute accuracy because 'y_true' and 'y_pred' are empty. "
                "Use 'raise_empty=False' to return a result with 'num_predictions=0' instead."
            )
            raise EmptyMetricError(msg)
        return AccuracyResult(
            num_correct_predictions=float("nan"),
            num_predictions=0,
        )

    # When missing_policy is 'propagate', check for missing values in
    # the arrays. When 'omit', rows have already been dropped so this
    # check always returns False. When 'raise', preprocess_1d already
    # raised if missing values were present.
    has_missing = (
        missing_values is not NOT_SET
        and missing_policy == "propagate"
        and multi_contains_value([y_true, y_pred], value=missing_values)
    )
    if has_missing:
        return AccuracyResult(
            num_correct_predictions=float("nan"),
            num_predictions=num_predictions,
        )

    return AccuracyResult(
        num_correct_predictions=int((y_true == y_pred).sum()),
        num_predictions=num_predictions,
    )
