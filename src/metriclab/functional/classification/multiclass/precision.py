r"""Compute multiclass classification precision from array-like inputs.

This module powers :func:`metriclab.functional.multiclass_precision` and
supports NumPy arrays, and Python sequences.
"""

from __future__ import annotations

__all__ = ["multiclass_precision"]

from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.metrics import confusion_matrix

from metriclab.exceptions import EmptyMetricError
from metriclab.results import MulticlassPrecisionResult
from metriclab.utils.array import (
    multi_contains_value,
    multi_count_unique_non_missing,
    preprocess_1d,
    to_numpy_1d,
)
from metriclab.utils.sentinel import NOT_SET
from metriclab.utils.undefined import (
    UndefinedPolicy,
    resolve_fill_value,
    validate_undefined_policy,
)

if TYPE_CHECKING:
    from metriclab.typing import ArrayLike
    from metriclab.utils.missing import MissingPolicy


def multiclass_precision(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    missing_policy: MissingPolicy = "propagate",
    missing_values: Any = NOT_SET,
    raise_empty: bool = True,
    undefined_policy: UndefinedPolicy = "warn",
) -> MulticlassPrecisionResult:
    r"""Compute the multiclass precision score.

    Args:
        y_true: The ground truth target labels.
        y_pred: The predicted labels.
        missing_policy: The policy for handling missing values.
            Valid values are ``'omit'``, ``'propagate'``, or
            ``'raise'``. ``'omit'`` removes rows where any array
            equals ``missing_values`` before computing the metric.
            ``'propagate'`` keeps the rows but returns
            ``precision=nan`` when missing values are detected.
            ``'raise'`` raises a ``ValueError`` if any array contains
            ``missing_values``.
        missing_values: The value to treat as missing. If not set,
            missing value handling is disabled regardless of
            ``missing_policy``.
        raise_empty: If ``True``, raises ``EmptyMetricError`` when
            there are no valid predictions. If ``False``, returns a
            ``MulticlassPrecisionResult`` with ``num_predictions=0``
            and ``precision=nan``.
        undefined_policy: The value to substitute when precision is
            undefined for a class because no samples were predicted as
            that class (i.e. ``TP + FP == 0``). Accepted values:

            - ``0.0``: substitute zero (matches sklearn default).
            - ``1.0``: substitute one (optimistic; use when absent
              classes should not penalise the score).
            - ``float('nan')``: substitute NaN, which then propagates
              into macro/weighted/micro averages.
            - ``"nan"``: string alias for ``float('nan')``.
            - ``"warn"`` *(default)*: substitutes ``0.0`` and emits a
              ``UserWarning`` listing the affected classes.

    Returns:
        A ``MulticlassPrecisionResult``. When missing values are present
            and ``missing_policy='propagate'``, the result keeps
            ``num_predictions`` and sets ``precision`` to ``nan``.

    Raises:
        EmptyMetricError: if there are no valid predictions and
            ``raise_empty`` is ``True``.
        ValueError: if ``missing_policy`` is invalid.
        ValueError: if ``y_true`` or ``y_pred`` contains
            ``missing_values`` and ``missing_policy`` is ``'raise'``.
        ValueError: if ``undefined_policy`` is not one of the
            accepted values.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from metriclab.functional import multiclass_precision
        >>> # with numpy arrays
        >>> multiclass_precision(
        ...     y_true=np.array([1, 0, 0, 1, 1, 2]),
        ...     y_pred=np.array([1, 0, 0, 1, 1, 2]),
        ... )
        MulticlassPrecisionResult(macro_precision=1.0, micro_precision=1.0, weighted_precision=1.0, per_class_precision=array([1., 1., 1.]), support=array([2, 3, 1]), num_predictions=6)
        >>> # with lists
        >>> multiclass_precision(y_true=[1, 0, 0, 1, 1, 2], y_pred=[1, 0, 0, 1, 1, 2])
        MulticlassPrecisionResult(macro_precision=1.0, micro_precision=1.0, weighted_precision=1.0, per_class_precision=array([1., 1., 1.]), support=array([2, 3, 1]), num_predictions=6)
        >>> # with string labels
        >>> multiclass_precision(
        ...     y_true=["cat", "dog", "cat", "dog", "dog", "bear"],
        ...     y_pred=["cat", "dog", "cat", "dog", "dog", "bear"],
        ... )
        MulticlassPrecisionResult(macro_precision=1.0, micro_precision=1.0, weighted_precision=1.0, per_class_precision=array([1., 1., 1.]), support=array([1, 2, 3]), num_predictions=6)
        >>> # with missing values and missing_policy='propagate' (default)
        >>> multiclass_precision(
        ...     y_true=np.array([1.0, 0.0, 0.0, 1.0, float("nan")]),
        ...     y_pred=np.array([1.0, 0.0, 0.0, 1.0, 1.0]),
        ...     missing_values=float("nan"),
        ... )
        MulticlassPrecisionResult(macro_precision=nan, micro_precision=nan, weighted_precision=nan, per_class_precision=array([nan, nan]), support=array([nan, nan]), num_predictions=5)
        >>> # with missing values and missing_policy='omit'
        >>> multiclass_precision(
        ...     y_true=np.array([1.0, 0.0, 0.0, 1.0, float("nan")]),
        ...     y_pred=np.array([1.0, 0.0, 0.0, 1.0, 1.0]),
        ...     missing_policy="omit",
        ...     missing_values=float("nan"),
        ... )
        MulticlassPrecisionResult(macro_precision=1.0, micro_precision=1.0, weighted_precision=1.0, per_class_precision=array([1., 1.]), support=array([2, 2]), num_predictions=4)
        >>> # silence the undefined_policy warning explicitly
        >>> multiclass_precision(
        ...     y_true=[0, 0, 1, 2],
        ...     y_pred=[0, 0, 1, 1],
        ...     undefined_policy=0.0,
        ... )
        MulticlassPrecisionResult(macro_precision=0.5, micro_precision=0.75, weighted_precision=0.625, per_class_precision=array([1. , 0.5, 0. ]), support=array([2, 1, 1]), num_predictions=4)
        >>> # propagate NaN for undefined classes
        >>> multiclass_precision(
        ...     y_true=[0, 0, 1, 2],
        ...     y_pred=[0, 0, 1, 1],
        ...     undefined_policy=float("nan"),
        ... )
        MulticlassPrecisionResult(macro_precision=nan, micro_precision=0.75, weighted_precision=nan, per_class_precision=array([1. , 0.5, nan]), support=array([2, 1, 1]), num_predictions=4)

        ```
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
                "Cannot compute precision because 'y_true' and 'y_pred' are empty. "
                "Use 'raise_empty=False' to return a result with 'num_predictions=0' instead."
            )
            raise EmptyMetricError(msg)
        return compute_multiclass_precision(
            y_true=y_true,
            y_pred=y_pred,
            undefined_policy=undefined_policy,
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
        n_classes = multi_count_unique_non_missing([y_true, y_pred], missing_values=missing_values)
        return MulticlassPrecisionResult(
            macro_precision=float("nan"),
            micro_precision=float("nan"),
            per_class_precision=np.array([float("nan")] * n_classes, dtype=np.float64),
            support=np.array([float("nan")] * n_classes, dtype=np.float64),
            weighted_precision=float("nan"),
            num_predictions=y_true.size,
        )

    return compute_multiclass_precision(
        y_true=y_true,
        y_pred=y_pred,
        undefined_policy=undefined_policy,
    )


def compute_multiclass_precision(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    undefined_policy: UndefinedPolicy = "warn",
) -> MulticlassPrecisionResult:
    r"""Compute multiclass precision metrics from true and predicted
    labels.

    This function calculates per-class precision scores alongside macro, micro,
    and weighted averages. It mimics the behaviour of
    `sklearn.metrics.precision_score` but aggregates all averages into a
    single, structured result object and gracefully handles zero-prediction
    edge cases.

    The precision metric evaluates the quality of positive predictions using
    the formula:
    $$Precision = \frac{TP}{TP + FP}$$

    Args:
        y_true: Ground truth (correct) target values. Shape: ``(n_samples,)``.
        y_pred: Estimated targets as returned by a classifier. Shape: ``(n_samples,)``.
        undefined_policy: Substitution value when precision is undefined
            for a class because no samples were predicted as that class
            (i.e. ``TP + FP == 0``). See :func:`multiclass_precision` for
            the full list of accepted values.

    Returns:
        A ``MulticlassPrecisionResult`` containing:
            - ``macro_precision``: unweighted mean of per-class precision.
            - ``micro_precision``: global precision computed across all classes.
            - ``weighted_precision``: precision weighted by per-class support.
            - ``per_class_precision``: array of precision for each individual class.
            - ``support``: number of true instances for each class.
            - ``num_predictions``: total number of samples evaluated.

    Raises:
        ValueError: if ``undefined_policy`` is not one of the
            accepted values.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from metriclab.functional.classification.multiclass.precision import (
        ...     compute_multiclass_precision,
        ... )
        >>> y_true = np.array([0, 1, 2, 2, 1, 2])
        >>> y_pred = np.array([0, 1, 2, 2, 1, 2])
        >>> result = compute_multiclass_precision(y_true, y_pred)
        >>> result.macro_precision
        1.0
        >>> result
        MulticlassPrecisionResult(macro_precision=1.0, micro_precision=1.0, weighted_precision=1.0, per_class_precision=array([1., 1., 1.]), support=array([1, 2, 3]), num_predictions=6)

        ```
    """
    validate_undefined_policy(undefined_policy)

    if y_true.size == 0:
        return MulticlassPrecisionResult(
            macro_precision=float("nan"),
            micro_precision=float("nan"),
            per_class_precision=np.array([], dtype=np.float64),
            support=np.array([], dtype=np.int64),
            weighted_precision=float("nan"),
            num_predictions=0,
        )

    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
    np.unique(np.concatenate([y_true, y_pred]))

    true_positives = np.diag(cm)
    positive_predictions = cm.sum(axis=0)  # TP + FP per class
    actual_positives = cm.sum(axis=1)  # TP + FN per class

    undefined_mask = positive_predictions == 0

    fill = resolve_fill_value(undefined_mask=undefined_mask, undefined_policy=undefined_policy)
    per_class = np.where(
        ~undefined_mask,
        true_positives / np.where(undefined_mask, 1, positive_predictions),
        fill,
    )

    # Macro: unweighted mean. NaN fill propagates naturally; 0.0/1.0
    # fill affects the average as documented.
    macro = float(np.mean(per_class))

    # Weighted: weighted by support. NaN fill propagates naturally;
    # 0.0/1.0 fill is weighted correctly by actual_positives.
    total_actual = actual_positives.sum()
    weighted = float(np.average(per_class, weights=actual_positives)) if total_actual > 0 else 0.0

    # Micro: global TP / (global TP + global FP).
    total_tp = true_positives.sum()
    total_pp = positive_predictions.sum()
    micro = float(total_tp / total_pp if total_pp > 0 else 0.0)

    return MulticlassPrecisionResult(
        macro_precision=macro,
        micro_precision=micro,
        per_class_precision=per_class,
        support=actual_positives,
        weighted_precision=weighted,
        num_predictions=y_true.size,
    )
