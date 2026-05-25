r"""Compute classification accuracy from array-like inputs.

This module powers :func:`metriclab.functional.balanced_accuracy` and
supports NumPy arrays, and Python sequences.
"""

from __future__ import annotations

__all__ = ["balanced_accuracy"]

from typing import TYPE_CHECKING, Any

from sklearn.metrics import balanced_accuracy_score

from metriclab.exceptions import EmptyMetricError
from metriclab.results import BalancedAccuracyResult
from metriclab.utils.array import multi_contains_value, preprocess_1d, to_numpy_1d
from metriclab.utils.sentinel import NOT_SET

if TYPE_CHECKING:
    from metriclab.typing import ArrayLike
    from metriclab.utils.missing import MissingPolicy


def balanced_accuracy(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    missing_policy: MissingPolicy = "propagate",
    missing_values: Any = NOT_SET,
    raise_empty: bool = True,
) -> BalancedAccuracyResult:
    r"""Compute balanced accuracy from ground-truth and predicted labels.

    Args:
        y_true: The ground truth target labels.
        y_pred: The predicted labels.
        missing_policy: The policy for handling values equal to
            ``missing_values``. ``"omit"`` removes rows where either
            input contains the missing value before computing the metric.
            ``"propagate"`` keeps the rows but returns a result with
            ``balanced_accuracy=nan`` when missing values are detected.
            ``"raise"`` raises ``ValueError`` instead.
        missing_values: The value to treat as missing. If not set,
            missing value handling is disabled regardless of
            ``missing_policy``.
        raise_empty: If ``True``, raises ``EmptyMetricError`` when
            there are no valid predictions. If ``False``, returns an
            ``BalancedAccuracyResult`` with ``num_predictions=0``.

    Returns:
        A :class:`~metriclab.results.BalancedAccuracyResult`. When
            ``missing_policy="propagate"`` finds missing values, the
            result keeps ``num_predictions`` and sets
            ``balanced_accuracy`` to ``nan``.

    Raises:
        EmptyMetricError: if there are no valid predictions and
            ``raise_empty`` is ``True``.
        ValueError: if ``missing_policy`` is invalid.
        ValueError: if ``y_true`` or ``y_pred`` contains
            ``missing_values`` and ``missing_policy`` is ``'raise'``.

    Example:
        >>> import numpy as np
        >>> from metriclab.functional import balanced_accuracy
        >>> balanced_accuracy(y_true=[1, 0, 0, 1, 1], y_pred=[1, 0, 1, 1, 1])
        BalancedAccuracyResult(balanced_accuracy=0.75, num_predictions=5)
        >>> balanced_accuracy(
        ...     y_true=np.array([1.0, 0.0, 0.0, 1.0, float("nan")]),
        ...     y_pred=np.array([1.0, 0.0, 0.0, 1.0, 1.0]),
        ...     missing_values=float("nan"),
        ... )
        BalancedAccuracyResult(balanced_accuracy=nan, num_predictions=5)
        >>> balanced_accuracy(y_true=[], y_pred=[], raise_empty=False)
        BalancedAccuracyResult(balanced_accuracy=nan, num_predictions=0)
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
        return BalancedAccuracyResult(
            balanced_accuracy=float("nan"),
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
        return BalancedAccuracyResult(
            balanced_accuracy=float("nan"),
            num_predictions=num_predictions,
        )

    return BalancedAccuracyResult(
        balanced_accuracy=float(balanced_accuracy_score(y_true=y_true, y_pred=y_pred)),
        num_predictions=num_predictions,
    )
