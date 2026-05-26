r"""Compute multiclass classification precision from array-like inputs.

This module powers :func:`metriclab.functional.multiclass_precision` and supports
NumPy arrays, and Python sequences.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix

from metriclab.results import MulticlassPrecisionResult


def compute_multiclass_precision(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> MulticlassPrecisionResult:
    r"""Compute multiclass precision metrics from true and predicted
    labels.

    This function calculates per-class precision scores alongside macro, micro,
    and weighted averages. It mimics the behavior of
    `sklearn.metrics.precision_score` but aggregates all averages into a
    single, structured result object and gracefully handles zero-prediction
    edge cases.

    The precision metric evaluates the quality of positive predictions using
    the formula:
    $$Precision = \frac{TP}{TP + FP}$$

    Args:
        y_true: Ground truth (correct) target values. Shape: `(n_samples,)`.
        y_pred: Estimated targets as returned by a classifier. Shape: `(n_samples,)`.

    Returns:
        A MulticlassPrecisionResult instance container housing:
            - macro_precision: Unweighted mean of per-class precision.
            - micro_precision: Global precision across all classes.
            - weighted_precision: Precision weighted by class support.
            - per_class_precision: Array of precisions for each individual class.
            - support: Number of true instances for each class.
            - num_predictions: Total number of samples evaluated.

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
    # Coerce to numpy arrays gracefully to guarantee .size works,
    # and handle empty array edge cases seamlessly.
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.size == 0:
        return MulticlassPrecisionResult(
            macro_precision=0.0,
            micro_precision=0.0,
            per_class_precision=np.array([], dtype=np.float64),
            support=np.array([], dtype=np.int64),
            weighted_precision=0.0,
            num_predictions=0,
        )

    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)

    # per-class: TP[i] / (TP[i] + FP[i])
    true_positives = np.diag(cm)  # TP per class
    positive_predictions = cm.sum(axis=0)  # TP + FP per class (predicted positives)
    actual_positives = cm.sum(axis=1)  # TP + FN per class (actual positives)

    with np.errstate(divide="ignore", invalid="ignore"):
        per_class = np.where(
            positive_predictions > 0,
            true_positives / positive_predictions,
            0.0,
        )

    # macro: unweighted mean over classes
    macro = float(per_class.mean())

    # weighted: weighted by actual positives (support) per class
    total_actual = actual_positives.sum()
    weighted = float(np.average(per_class, weights=actual_positives) if total_actual > 0 else 0.0)

    # micro: global TP / (global TP + global FP)
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
