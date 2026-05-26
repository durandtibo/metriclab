from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import pytest
from coola.equality import objects_are_allclose

from metriclab.results import BinaryPrecisionResult

############################################
#     Tests for BinaryPrecisionResult      #
############################################


# --- instantiation ---


def test_binary_precision_result_instantiation() -> None:
    m = BinaryPrecisionResult(precision=0.75, num_predictions=10)
    assert m.precision == 0.75
    assert m.num_predictions == 10


def test_binary_precision_result_zero_predictions() -> None:
    m = BinaryPrecisionResult(precision=0.0, num_predictions=0)
    assert m.precision == 0.0
    assert m.num_predictions == 0


def test_binary_precision_result_nan_precision() -> None:
    m = BinaryPrecisionResult(precision=float("nan"), num_predictions=10)
    assert math.isnan(m.precision)


def test_binary_precision_result_frozen() -> None:
    m = BinaryPrecisionResult(precision=0.75, num_predictions=10)
    with pytest.raises(FrozenInstanceError):
        m.precision = 0.5  # type: ignore[misc]


def test_binary_precision_result_perfect() -> None:
    m = BinaryPrecisionResult(precision=1.0, num_predictions=10)
    assert m.precision == 1.0


def test_binary_precision_result_zero_precision() -> None:
    m = BinaryPrecisionResult(precision=0.0, num_predictions=10)
    assert m.precision == 0.0


# --- validation ---


def test_binary_precision_result_negative_num_predictions_raises() -> None:
    with pytest.raises(ValueError, match="num_predictions must be >= 0"):
        BinaryPrecisionResult(precision=0.75, num_predictions=-1)


def test_binary_precision_result_negative_precision_raises() -> None:
    with pytest.raises(ValueError, match="precision must be >= 0"):
        BinaryPrecisionResult(precision=-0.1, num_predictions=10)


def test_binary_precision_result_nan_precision_does_not_raise() -> None:
    BinaryPrecisionResult(precision=float("nan"), num_predictions=10)


def test_binary_precision_result_zero_precision_does_not_raise() -> None:
    BinaryPrecisionResult(precision=0.0, num_predictions=10)


def test_binary_precision_result_one_precision_does_not_raise() -> None:
    BinaryPrecisionResult(precision=1.0, num_predictions=10)


def test_binary_precision_result_zero_num_predictions_does_not_raise() -> None:
    BinaryPrecisionResult(precision=0.0, num_predictions=0)


# --- equal ---


def test_binary_precision_result_equal_true() -> None:
    assert BinaryPrecisionResult(precision=0.75, num_predictions=10).equal(
        BinaryPrecisionResult(precision=0.75, num_predictions=10)
    )


def test_binary_precision_result_equal_false_different_precision() -> None:
    assert not BinaryPrecisionResult(precision=0.75, num_predictions=10).equal(
        BinaryPrecisionResult(precision=0.8, num_predictions=10)
    )


def test_binary_precision_result_equal_false_different_num_predictions() -> None:
    assert not BinaryPrecisionResult(precision=0.75, num_predictions=10).equal(
        BinaryPrecisionResult(precision=0.75, num_predictions=5)
    )


def test_binary_precision_result_equal_wrong_type() -> None:
    assert not BinaryPrecisionResult(precision=0.75, num_predictions=10).equal("not a result")


def test_binary_precision_result_equal_nan_false_by_default() -> None:
    assert not BinaryPrecisionResult(precision=float("nan"), num_predictions=10).equal(
        BinaryPrecisionResult(precision=float("nan"), num_predictions=10)
    )


def test_binary_precision_result_equal_nan_true_with_equal_nan() -> None:
    assert BinaryPrecisionResult(precision=float("nan"), num_predictions=10).equal(
        BinaryPrecisionResult(precision=float("nan"), num_predictions=10),
        equal_nan=True,
    )


# --- allclose ---


def test_binary_precision_result_allclose_true() -> None:
    assert BinaryPrecisionResult(precision=0.75, num_predictions=10).allclose(
        BinaryPrecisionResult(precision=0.75, num_predictions=10)
    )


def test_binary_precision_result_allclose_within_tolerance() -> None:
    assert BinaryPrecisionResult(precision=0.75, num_predictions=10).allclose(
        BinaryPrecisionResult(precision=0.75 + 1e-7, num_predictions=10),
        rtol=1e-5,
        atol=1e-6,
    )


def test_binary_precision_result_allclose_outside_tolerance() -> None:
    assert not BinaryPrecisionResult(precision=0.75, num_predictions=10).allclose(
        BinaryPrecisionResult(precision=0.8, num_predictions=10),
        rtol=1e-5,
        atol=1e-8,
    )


def test_binary_precision_result_allclose_false_different_num_predictions() -> None:
    assert not BinaryPrecisionResult(precision=0.75, num_predictions=10).allclose(
        BinaryPrecisionResult(precision=0.75, num_predictions=5)
    )


def test_binary_precision_result_allclose_wrong_type() -> None:
    assert not BinaryPrecisionResult(precision=0.75, num_predictions=10).allclose("not a result")


def test_binary_precision_result_allclose_nan_false_by_default() -> None:
    assert not BinaryPrecisionResult(precision=float("nan"), num_predictions=10).allclose(
        BinaryPrecisionResult(precision=float("nan"), num_predictions=10)
    )


def test_binary_precision_result_allclose_nan_true_with_equal_nan() -> None:
    assert BinaryPrecisionResult(precision=float("nan"), num_predictions=10).allclose(
        BinaryPrecisionResult(precision=float("nan"), num_predictions=10),
        equal_nan=True,
    )


# --- to_dict ---


@pytest.mark.parametrize(
    ("precision", "num_predictions", "prefix", "suffix", "expected"),
    [
        pytest.param(
            0.75,
            10,
            "",
            "",
            {"precision": 0.75, "num_predictions": 10},
            id="standard",
        ),
        pytest.param(
            0.75,
            10,
            "val_",
            "",
            {"val_precision": 0.75, "val_num_predictions": 10},
            id="prefix-only",
        ),
        pytest.param(
            0.75,
            10,
            "",
            "_epoch1",
            {"precision_epoch1": 0.75, "num_predictions_epoch1": 10},
            id="suffix-only",
        ),
        pytest.param(
            0.75,
            10,
            "val_",
            "_epoch1",
            {"val_precision_epoch1": 0.75, "val_num_predictions_epoch1": 10},
            id="prefix-and-suffix",
        ),
        pytest.param(
            1.0,
            10,
            "",
            "",
            {"precision": 1.0, "num_predictions": 10},
            id="perfect",
        ),
        pytest.param(
            0.0,
            10,
            "",
            "",
            {"precision": 0.0, "num_predictions": 10},
            id="zero",
        ),
        pytest.param(
            0.0,
            0,
            "",
            "",
            {"precision": 0.0, "num_predictions": 0},
            id="zero-predictions",
        ),
    ],
)
def test_binary_precision_result_to_dict(
    precision: float,
    num_predictions: int,
    prefix: str,
    suffix: str,
    expected: dict,
) -> None:
    assert objects_are_allclose(
        BinaryPrecisionResult(precision=precision, num_predictions=num_predictions).to_dict(
            prefix=prefix, suffix=suffix
        ),
        expected,
    )


def test_binary_precision_result_to_dict_nan() -> None:
    assert objects_are_allclose(
        BinaryPrecisionResult(precision=float("nan"), num_predictions=10).to_dict(),
        {"precision": float("nan"), "num_predictions": 10},
        equal_nan=True,
    )


# --- to_display ---


@pytest.mark.parametrize(
    ("precision", "num_predictions", "expected"),
    [
        pytest.param(
            0.75,
            10,
            "Precision [███████████████░░░░░]  0.7500",
            id="standard",
        ),
        pytest.param(
            1.0,
            10,
            "Precision [████████████████████]  1.0000",
            id="perfect",
        ),
        pytest.param(
            0.0,
            10,
            "Precision [░░░░░░░░░░░░░░░░░░░░]  0.0000",
            id="zero",
        ),
        pytest.param(
            0.5,
            10,
            "Precision [██████████░░░░░░░░░░]  0.5000",
            id="half",
        ),
        pytest.param(
            0.0,
            0,
            "BinaryPrecisionResult: no predictions",
            id="zero-predictions",
        ),
        pytest.param(
            float("nan"),
            10,
            "Precision [????????????????????]  nan",
            id="nan",
        ),
    ],
)
def test_binary_precision_result_to_display(
    precision: float, num_predictions: int, expected: str
) -> None:
    assert (
        BinaryPrecisionResult(precision=precision, num_predictions=num_predictions).to_display()
        == expected
    )


def test_binary_precision_result_to_display_returns_str() -> None:
    assert isinstance(BinaryPrecisionResult(precision=0.75, num_predictions=10).to_display(), str)


# --- repr / str ---


def test_binary_precision_result_repr() -> None:
    assert (
        repr(BinaryPrecisionResult(precision=0.75, num_predictions=10))
        == "BinaryPrecisionResult(precision=0.75, num_predictions=10)"
    )


def test_binary_precision_result_str() -> None:
    assert (
        str(BinaryPrecisionResult(precision=0.75, num_predictions=10))
        == "BinaryPrecisionResult(precision=0.75, num_predictions=10)"
    )
