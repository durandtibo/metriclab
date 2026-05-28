from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import precision_recall_fscore_support, precision_score

from metriclab.functional import multiclass_precision
from metriclab.results import MulticlassPrecisionResult
from tests.conftest import ignore_single_label_warning

##############################################
#   Tests aligning with scikit-learn metric  #
##############################################


@ignore_single_label_warning
@pytest.mark.parametrize(
    ("y_true", "y_pred"),
    [
        pytest.param(
            np.array([0, 1, 2, 0, 1, 2]),
            np.array([0, 2, 1, 0, 0, 1]),
            id="3-classes-scrambled",
        ),
        pytest.param(
            np.array([0, 1, 2, 3, 0, 1, 2, 3]),
            np.array([0, 1, 2, 3, 3, 2, 1, 0]),
            id="4-classes-partial-errors",
        ),
        pytest.param(
            np.array([0, 0, 1, 1, 2, 2]),
            np.array([0, 0, 0, 0, 2, 2]),
            id="3-classes-completely-missing-predictions-for-class-1",
        ),
        pytest.param(
            np.array([1, 1, 1, 2, 2, 2]),
            np.array([1, 1, 1, 2, 2, 2]),
            id="skewed-support-perfect-predictions",
        ),
        pytest.param(
            np.array([0, 0, 0, 0]),
            np.array([0, 0, 0, 0]),
            id="single-class-universe-perfect",
        ),
        pytest.param(
            np.array([0, 1, 2, 3]),
            np.array([1, 0, 3, 2]),
            id="completely-swapped-labels-4-classes",
        ),
        pytest.param(
            np.array([0, 0, 1, 1, 2, 2, 3, 3]),
            np.array([0, 0, 1, 2, 2, 2, 3, 0]),
            id="balanced-support-mixed-precision-4-classes",
        ),
        pytest.param(
            np.array([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5]),
            np.array([0, 1, 2, 3, 4, 5, 5, 4, 3, 2, 1, 0]),
            id="6-classes-high-cardinality-mirror-errors",
        ),
        pytest.param(
            np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]),
            np.array([1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 0]),
            id="3-classes-cyclical-shift-errors",
        ),
        pytest.param(
            np.array([0, 1, 0, 1, 0, 1]),
            np.array([2, 2, 2, 2, 2, 2]),
            id="predicted-classes-completely-outside-true-labels-universe",
        ),
        pytest.param(
            np.array([5, 5, 12, 12, 7, 7]),
            np.array([5, 12, 12, 7, 7, 5]),
            id="non-sequential-sparse-integer-labels",
        ),
        pytest.param(
            np.array([0, 0, 0, 0]),
            np.array([1, 1, 1, 1]),
            id="different-labels",
        ),
    ],
)
@pytest.mark.parametrize("undefined_policy", [0.0, 1.0])
def test_multiclass_precision_against_sklearn(
    y_true: np.ndarray, y_pred: np.ndarray, undefined_policy: float
) -> None:
    # 1. Dynamically compute the baseline averages using sklearn
    macro_p = precision_score(y_true, y_pred, average="macro", zero_division=undefined_policy)
    micro_p = precision_score(y_true, y_pred, average="micro", zero_division=undefined_policy)
    weighted_p = precision_score(y_true, y_pred, average="weighted", zero_division=undefined_policy)
    per_class_p, _, _, expected_support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=undefined_policy
    )

    expected_result = MulticlassPrecisionResult(
        macro_precision=float(macro_p),
        micro_precision=float(micro_p),
        weighted_precision=float(weighted_p),
        per_class_precision=per_class_p,
        support=expected_support.astype(int),
        num_predictions=y_true.size,
    )

    # 2. Assert full structural and mathematical equality
    result = multiclass_precision(y_true=y_true, y_pred=y_pred, undefined_policy=undefined_policy)
    assert result.allclose(expected_result, equal_nan=True)
