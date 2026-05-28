from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import precision_score

from metriclab.functional import binary_precision
from metriclab.results import BinaryPrecisionResult
from metriclab.utils.array import count_values

##############################################
#   Tests aligning with scikit-learn metric  #
##############################################


@pytest.mark.parametrize(
    ("y_true", "y_pred", "pos_label"),
    [
        pytest.param(
            np.array([1, 0, 0, 1, 1]),
            np.array([1, 0, 0, 1, 1]),
            1,
            id="perfect-predictions",
        ),
        pytest.param(
            np.array([1, 0, 0, 1, 1]),
            np.array([1, 0, 1, 1, 1]),
            1,
            id="one-false-positive",
        ),
        pytest.param(
            np.array([1, 0, 0, 1, 1]),
            np.array([0, 0, 0, 0, 0]),
            1,
            id="no-positive-predictions",
        ),
        pytest.param(
            np.array([1, 1, 1, 1]),
            np.array([1, 1, 1, 1]),
            1,
            id="single-class-all-positive",
        ),
        pytest.param(
            np.array([0, 0, 0, 0]),
            np.array([0, 0, 0, 0]),
            1,
            id="single-class-all-negative",
        ),
        pytest.param(
            np.array([1, 0, 1, 0, 1, 0]),
            np.array([0, 1, 0, 1, 0, 1]),
            1,
            id="all-predictions-wrong",
        ),
        pytest.param(
            np.array([1, 0, 0, 1, 1, 0, 0, 1]),
            np.array([1, 1, 0, 1, 0, 0, 1, 1]),
            1,
            id="mixed-errors",
        ),
        pytest.param(
            np.array([0, 1, 0, 1, 0]),
            np.array([1, 1, 1, 1, 1]),
            1,
            id="all-predicted-positive",
        ),
        pytest.param(
            np.array([1, 0, 1, 0]),
            np.array([0, 0, 0, 0]),
            1,
            id="all-predicted-negative",
        ),
        pytest.param(
            np.array(["cat", "dog", "cat", "dog", "dog"]),
            np.array(["cat", "dog", "dog", "dog", "dog"]),
            "dog",
            id="string-labels-dog-positive",
        ),
        pytest.param(
            np.array(["cat", "dog", "cat", "dog", "dog"]),
            np.array(["cat", "dog", "dog", "dog", "dog"]),
            "cat",
            id="string-labels-cat-positive",
        ),
        pytest.param(
            np.array([2, 5, 2, 5, 5]),
            np.array([2, 5, 5, 5, 5]),
            5,
            id="non-binary-integer-labels",
        ),
        pytest.param(
            np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0]),
            np.array([1, 0, 1, 0, 1, 0, 0, 1, 0, 1]),
            1,
            id="balanced-mixed-errors",
        ),
        pytest.param(
            np.array([1, 1, 1, 0, 0]),
            np.array([1, 0, 0, 0, 0]),
            1,
            id="skewed-positive-support",
        ),
    ],
)
def test_binary_precision_against_sklearn(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pos_label: object,
) -> None:
    # 1. Compute the baseline using sklearn directly.
    expected_result = BinaryPrecisionResult(
        precision=float(
            precision_score(
                y_true, y_pred, pos_label=pos_label, average="binary", zero_division=0.0
            )
        ),
        num_predictions=y_true.size,
        num_positive_predictions=count_values(y_pred, value=pos_label),
    )

    # 2. Assert full structural and mathematical equality.
    result = binary_precision(
        y_true=y_true,
        y_pred=y_pred,
        pos_label=pos_label,
    )
    assert result.allclose(expected_result)
