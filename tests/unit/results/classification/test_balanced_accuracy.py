from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import pytest
from coola.equality import objects_are_allclose

from metriclab.results import BalancedAccuracyResult

############################################
#     Tests for BalancedAccuracyResult     #
############################################


# --- instantiation ---


def test_balanced_accuracy_result_instantiation() -> None:
    m = BalancedAccuracyResult(balanced_accuracy=0.7, num_predictions=10)
    assert m.balanced_accuracy == 0.7
    assert m.num_predictions == 10


def test_balanced_accuracy_result_zero_predictions() -> None:
    m = BalancedAccuracyResult(balanced_accuracy=0.0, num_predictions=0)
    assert m.num_predictions == 0


def test_balanced_accuracy_result_nan_balanced_accuracy() -> None:
    m = BalancedAccuracyResult(balanced_accuracy=float("nan"), num_predictions=10)
    assert math.isnan(m.balanced_accuracy)


def test_balanced_accuracy_result_frozen() -> None:
    m = BalancedAccuracyResult(balanced_accuracy=0.7, num_predictions=10)
    with pytest.raises(FrozenInstanceError, match=r"cannot assign to field 'num_predictions'"):
        m.num_predictions = 11


# --- validation ---


def test_balanced_accuracy_result_negative_num_predictions_raises() -> None:
    with pytest.raises(ValueError, match=r"num_predictions must be >= 0"):
        BalancedAccuracyResult(balanced_accuracy=0.7, num_predictions=-1)


def test_balanced_accuracy_result_negative_balanced_accuracy_raises() -> None:
    with pytest.raises(ValueError, match=r"balanced_accuracy must be >= 0"):
        BalancedAccuracyResult(balanced_accuracy=-0.1, num_predictions=10)


def test_balanced_accuracy_result_nan_balanced_accuracy_does_not_raise() -> None:
    BalancedAccuracyResult(balanced_accuracy=float("nan"), num_predictions=10)


def test_balanced_accuracy_result_zero_balanced_accuracy_does_not_raise() -> None:
    BalancedAccuracyResult(balanced_accuracy=0.0, num_predictions=10)


def test_balanced_accuracy_result_one_balanced_accuracy_does_not_raise() -> None:
    BalancedAccuracyResult(balanced_accuracy=1.0, num_predictions=10)


# --- equal ---


def test_balanced_accuracy_result_equal_true() -> None:
    assert BalancedAccuracyResult(balanced_accuracy=0.7, num_predictions=10).equal(
        BalancedAccuracyResult(balanced_accuracy=0.7, num_predictions=10)
    )


def test_balanced_accuracy_result_equal_false_different_balanced_accuracy() -> None:
    assert not BalancedAccuracyResult(balanced_accuracy=0.7, num_predictions=10).equal(
        BalancedAccuracyResult(balanced_accuracy=0.8, num_predictions=10)
    )


def test_balanced_accuracy_result_equal_false_different_num_predictions() -> None:
    assert not BalancedAccuracyResult(balanced_accuracy=0.7, num_predictions=10).equal(
        BalancedAccuracyResult(balanced_accuracy=0.7, num_predictions=5)
    )


def test_balanced_accuracy_result_equal_wrong_type() -> None:
    assert not BalancedAccuracyResult(balanced_accuracy=0.7, num_predictions=10).equal(
        "not a result"
    )


def test_balanced_accuracy_result_equal_nan_false_by_default() -> None:
    assert not BalancedAccuracyResult(balanced_accuracy=float("nan"), num_predictions=10).equal(
        BalancedAccuracyResult(balanced_accuracy=float("nan"), num_predictions=10)
    )


def test_balanced_accuracy_result_equal_nan_true_with_equal_nan() -> None:
    assert BalancedAccuracyResult(balanced_accuracy=float("nan"), num_predictions=10).equal(
        BalancedAccuracyResult(balanced_accuracy=float("nan"), num_predictions=10),
        equal_nan=True,
    )


# --- allclose ---


def test_balanced_accuracy_result_allclose_true() -> None:
    assert BalancedAccuracyResult(balanced_accuracy=0.7, num_predictions=10).allclose(
        BalancedAccuracyResult(balanced_accuracy=0.7, num_predictions=10)
    )


def test_balanced_accuracy_result_allclose_within_tolerance() -> None:
    assert BalancedAccuracyResult(balanced_accuracy=0.7, num_predictions=10).allclose(
        BalancedAccuracyResult(balanced_accuracy=0.7 + 1e-7, num_predictions=10),
        rtol=1e-5,
        atol=1e-6,
    )


def test_balanced_accuracy_result_allclose_outside_tolerance() -> None:
    assert not BalancedAccuracyResult(balanced_accuracy=0.7, num_predictions=10).allclose(
        BalancedAccuracyResult(balanced_accuracy=0.8, num_predictions=10),
        rtol=1e-5,
        atol=1e-8,
    )


def test_balanced_accuracy_result_allclose_false_different_num_predictions() -> None:
    assert not BalancedAccuracyResult(balanced_accuracy=0.7, num_predictions=10).allclose(
        BalancedAccuracyResult(balanced_accuracy=0.7, num_predictions=5)
    )


def test_balanced_accuracy_result_allclose_wrong_type() -> None:
    assert not BalancedAccuracyResult(balanced_accuracy=0.7, num_predictions=10).allclose(
        "not a result"
    )


def test_balanced_accuracy_result_allclose_nan_false_by_default() -> None:
    assert not BalancedAccuracyResult(balanced_accuracy=float("nan"), num_predictions=10).allclose(
        BalancedAccuracyResult(balanced_accuracy=float("nan"), num_predictions=10)
    )


def test_balanced_accuracy_result_allclose_nan_true_with_equal_nan() -> None:
    assert BalancedAccuracyResult(balanced_accuracy=float("nan"), num_predictions=10).allclose(
        BalancedAccuracyResult(balanced_accuracy=float("nan"), num_predictions=10),
        equal_nan=True,
    )


# --- to_dict ---


@pytest.mark.parametrize(
    ("balanced_accuracy", "num_predictions", "prefix", "suffix", "expected"),
    [
        pytest.param(
            0.7,
            10,
            "",
            "",
            {"balanced_accuracy": 0.7, "num_predictions": 10},
            id="standard",
        ),
        pytest.param(
            0.7,
            10,
            "val_",
            "",
            {"val_balanced_accuracy": 0.7, "val_num_predictions": 10},
            id="prefix-only",
        ),
        pytest.param(
            0.7,
            10,
            "",
            "_epoch1",
            {"balanced_accuracy_epoch1": 0.7, "num_predictions_epoch1": 10},
            id="suffix-only",
        ),
        pytest.param(
            0.7,
            10,
            "val_",
            "_epoch1",
            {"val_balanced_accuracy_epoch1": 0.7, "val_num_predictions_epoch1": 10},
            id="prefix-and-suffix",
        ),
        pytest.param(
            1.0,
            10,
            "",
            "",
            {"balanced_accuracy": 1.0, "num_predictions": 10},
            id="perfect",
        ),
        pytest.param(
            0.0,
            10,
            "",
            "",
            {"balanced_accuracy": 0.0, "num_predictions": 10},
            id="zero",
        ),
        pytest.param(
            0.0,
            0,
            "",
            "",
            {"balanced_accuracy": 0.0, "num_predictions": 0},
            id="zero-predictions",
        ),
    ],
)
def test_balanced_accuracy_result_to_dict(
    balanced_accuracy: float,
    num_predictions: int,
    prefix: str,
    suffix: str,
    expected: dict,
) -> None:
    assert objects_are_allclose(
        BalancedAccuracyResult(
            balanced_accuracy=balanced_accuracy, num_predictions=num_predictions
        ).to_dict(prefix=prefix, suffix=suffix),
        expected,
    )


def test_balanced_accuracy_result_to_dict_nan() -> None:
    assert objects_are_allclose(
        BalancedAccuracyResult(balanced_accuracy=float("nan"), num_predictions=10).to_dict(),
        {"balanced_accuracy": float("nan"), "num_predictions": 10},
        equal_nan=True,
    )


# --- to_display ---


@pytest.mark.parametrize(
    ("balanced_accuracy", "num_predictions", "expected"),
    [
        pytest.param(
            0.7,
            10,
            "Balanced Accuracy [██████████████░░░░░░]  0.7000",
            id="standard",
        ),
        pytest.param(
            1.0,
            10,
            "Balanced Accuracy [████████████████████]  1.0000",
            id="perfect",
        ),
        pytest.param(
            0.0,
            10,
            "Balanced Accuracy [░░░░░░░░░░░░░░░░░░░░]  0.0000",
            id="zero",
        ),
        pytest.param(
            0.5,
            10,
            "Balanced Accuracy [██████████░░░░░░░░░░]  0.5000",
            id="half",
        ),
        pytest.param(
            0.0,
            0,
            "BalancedAccuracyResult: no predictions",
            id="zero-predictions",
        ),
        pytest.param(
            float("nan"),
            10,
            "Balanced Accuracy [????????????????????]  nan",
            id="nan",
        ),
    ],
)
def test_balanced_accuracy_result_to_display(
    balanced_accuracy: float, num_predictions: int, expected: str
) -> None:
    assert (
        BalancedAccuracyResult(
            balanced_accuracy=balanced_accuracy, num_predictions=num_predictions
        ).to_display()
        == expected
    )


def test_balanced_accuracy_result_to_display_returns_str() -> None:
    assert isinstance(
        BalancedAccuracyResult(balanced_accuracy=0.7, num_predictions=10).to_display(), str
    )


# --- repr / str ---


def test_balanced_accuracy_result_repr() -> None:
    assert (
        repr(BalancedAccuracyResult(balanced_accuracy=0.7, num_predictions=10))
        == "BalancedAccuracyResult(balanced_accuracy=0.7, num_predictions=10)"
    )


def test_balanced_accuracy_result_str() -> None:
    assert (
        str(BalancedAccuracyResult(balanced_accuracy=0.7, num_predictions=10))
        == "BalancedAccuracyResult(balanced_accuracy=0.7, num_predictions=10)"
    )
