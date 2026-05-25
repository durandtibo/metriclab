from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import pytest
from coola.equality import objects_are_allclose

from metriclab.results import PrecisionResult

#####################################
#     Tests for PrecisionResult     #
#####################################


# --- instantiation ---


def test_precision_result_instantiation() -> None:
    m = PrecisionResult(num_true_positives=3, num_positive_predictions=4)
    assert m.num_true_positives == 3
    assert m.num_positive_predictions == 4


def test_precision_result_zero_predictions() -> None:
    m = PrecisionResult(num_true_positives=0, num_positive_predictions=0)
    assert m.num_true_positives == 0
    assert m.num_positive_predictions == 0


def test_precision_result_nan_true_positives() -> None:
    m = PrecisionResult(num_true_positives=float("nan"), num_positive_predictions=4)
    assert math.isnan(m.num_true_positives)


def test_precision_result_frozen() -> None:
    m = PrecisionResult(num_true_positives=3, num_positive_predictions=4)
    with pytest.raises(FrozenInstanceError):
        m.num_true_positives = 2  # type: ignore[misc]


def test_precision_result_perfect() -> None:
    m = PrecisionResult(num_true_positives=5, num_positive_predictions=5)
    assert m.precision == 1.0


def test_precision_result_zero_true_positives() -> None:
    m = PrecisionResult(num_true_positives=0, num_positive_predictions=5)
    assert m.precision == 0.0


# --- validation ---


def test_precision_result_negative_num_true_positives_raises() -> None:
    with pytest.raises(ValueError, match="num_true_positives must be >= 0"):
        PrecisionResult(num_true_positives=-1, num_positive_predictions=4)


def test_precision_result_negative_num_positive_predictions_raises() -> None:
    with pytest.raises(ValueError, match="num_positive_predictions must be >= 0"):
        PrecisionResult(num_true_positives=3, num_positive_predictions=-1)


def test_precision_result_num_true_positives_exceeds_num_positive_predictions_raises() -> None:
    with pytest.raises(ValueError, match="cannot exceed num_positive_predictions"):
        PrecisionResult(num_true_positives=5, num_positive_predictions=4)


def test_precision_result_nan_true_positives_does_not_raise() -> None:
    PrecisionResult(num_true_positives=float("nan"), num_positive_predictions=4)


def test_precision_result_zero_true_positives_does_not_raise() -> None:
    PrecisionResult(num_true_positives=0, num_positive_predictions=4)


def test_precision_result_num_true_positives_equals_num_positive_predictions_does_not_raise() -> (
    None
):
    PrecisionResult(num_true_positives=4, num_positive_predictions=4)


def test_precision_result_all_zero_does_not_raise() -> None:
    PrecisionResult(num_true_positives=0, num_positive_predictions=0)


# --- precision property ---


@pytest.mark.parametrize(
    ("num_true_positives", "num_positive_predictions", "expected"),
    [
        pytest.param(3, 4, 0.75, id="standard"),
        pytest.param(5, 5, 1.0, id="perfect"),
        pytest.param(0, 5, 0.0, id="zero"),
        pytest.param(1, 2, 0.5, id="half"),
        pytest.param(1, 4, 0.25, id="quarter"),
    ],
)
def test_precision_result_precision(
    num_true_positives: int,
    num_positive_predictions: int,
    expected: float,
) -> None:
    assert (
        PrecisionResult(
            num_true_positives=num_true_positives,
            num_positive_predictions=num_positive_predictions,
        ).precision
        == expected
    )


def test_precision_result_precision_zero_denominator_returns_nan() -> None:
    assert math.isnan(PrecisionResult(num_true_positives=0, num_positive_predictions=0).precision)


def test_precision_result_precision_nan_true_positives_returns_nan() -> None:
    assert math.isnan(
        PrecisionResult(num_true_positives=float("nan"), num_positive_predictions=4).precision
    )


# --- equal ---


def test_precision_result_equal_true() -> None:
    assert PrecisionResult(num_true_positives=3, num_positive_predictions=4).equal(
        PrecisionResult(num_true_positives=3, num_positive_predictions=4)
    )


def test_precision_result_equal_false_different_num_true_positives() -> None:
    assert not PrecisionResult(num_true_positives=3, num_positive_predictions=4).equal(
        PrecisionResult(num_true_positives=2, num_positive_predictions=4)
    )


def test_precision_result_equal_false_different_num_positive_predictions() -> None:
    assert not PrecisionResult(num_true_positives=3, num_positive_predictions=4).equal(
        PrecisionResult(num_true_positives=3, num_positive_predictions=5)
    )


def test_precision_result_equal_wrong_type() -> None:
    assert not PrecisionResult(num_true_positives=3, num_positive_predictions=4).equal(
        "not a result"
    )


def test_precision_result_equal_nan_false_by_default() -> None:
    assert not PrecisionResult(num_true_positives=float("nan"), num_positive_predictions=4).equal(
        PrecisionResult(num_true_positives=float("nan"), num_positive_predictions=4)
    )


def test_precision_result_equal_nan_true_with_equal_nan() -> None:
    assert PrecisionResult(num_true_positives=float("nan"), num_positive_predictions=4).equal(
        PrecisionResult(num_true_positives=float("nan"), num_positive_predictions=4),
        equal_nan=True,
    )


# --- allclose ---


def test_precision_result_allclose_true() -> None:
    assert PrecisionResult(num_true_positives=3, num_positive_predictions=4).allclose(
        PrecisionResult(num_true_positives=3, num_positive_predictions=4)
    )


def test_precision_result_allclose_within_tolerance() -> None:
    assert PrecisionResult(num_true_positives=3.0, num_positive_predictions=4).allclose(
        PrecisionResult(num_true_positives=3 + 1e-7, num_positive_predictions=4),
        rtol=1e-5,
        atol=1e-6,
    )


def test_precision_result_allclose_outside_tolerance() -> None:
    assert not PrecisionResult(num_true_positives=3, num_positive_predictions=4).allclose(
        PrecisionResult(num_true_positives=2, num_positive_predictions=4),
        rtol=1e-5,
        atol=1e-8,
    )


def test_precision_result_allclose_false_different_num_positive_predictions() -> None:
    assert not PrecisionResult(num_true_positives=3, num_positive_predictions=4).allclose(
        PrecisionResult(num_true_positives=3, num_positive_predictions=5)
    )


def test_precision_result_allclose_wrong_type() -> None:
    assert not PrecisionResult(num_true_positives=3, num_positive_predictions=4).allclose(
        "not a result"
    )


def test_precision_result_allclose_nan_false_by_default() -> None:
    assert not PrecisionResult(
        num_true_positives=float("nan"), num_positive_predictions=4
    ).allclose(PrecisionResult(num_true_positives=float("nan"), num_positive_predictions=4))


def test_precision_result_allclose_nan_true_with_equal_nan() -> None:
    assert PrecisionResult(num_true_positives=float("nan"), num_positive_predictions=4).allclose(
        PrecisionResult(num_true_positives=float("nan"), num_positive_predictions=4),
        equal_nan=True,
    )


# --- from_precision ---


@pytest.mark.parametrize(
    ("precision", "num_positive_predictions", "expected_tp"),
    [
        pytest.param(0.75, 4, 3, id="standard"),
        pytest.param(1.0, 5, 5, id="perfect"),
        pytest.param(0.0, 5, 0, id="zero"),
        pytest.param(0.5, 4, 2, id="half"),
        pytest.param(0.5, 2, 1, id="half-small"),
    ],
)
def test_precision_result_from_precision(
    precision: float, num_positive_predictions: int, expected_tp: int
) -> None:
    m = PrecisionResult.from_precision(
        precision=precision, num_positive_predictions=num_positive_predictions
    )
    assert m.num_true_positives == expected_tp
    assert m.num_positive_predictions == num_positive_predictions


def test_precision_result_from_precision_nan_precision() -> None:
    m = PrecisionResult.from_precision(precision=float("nan"), num_positive_predictions=4)
    assert math.isnan(m.num_true_positives)
    assert m.num_positive_predictions == 4


def test_precision_result_from_precision_zero_denominator() -> None:
    m = PrecisionResult.from_precision(precision=0.0, num_positive_predictions=0)
    assert m.num_true_positives == 0
    assert m.num_positive_predictions == 0


def test_precision_result_from_precision_returns_instance() -> None:
    assert isinstance(
        PrecisionResult.from_precision(precision=0.75, num_positive_predictions=4),
        PrecisionResult,
    )


def test_precision_result_from_precision_roundtrip() -> None:
    # precision computed from counts should match original
    m = PrecisionResult(num_true_positives=3, num_positive_predictions=4)
    m2 = PrecisionResult.from_precision(
        precision=m.precision, num_positive_predictions=m.num_positive_predictions
    )
    assert m.equal(m2)


# --- to_dict ---


@pytest.mark.parametrize(
    ("num_true_positives", "num_positive_predictions", "prefix", "suffix", "expected"),
    [
        pytest.param(
            3,
            4,
            "",
            "",
            {"precision": 0.75, "num_true_positives": 3, "num_positive_predictions": 4},
            id="standard",
        ),
        pytest.param(
            3,
            4,
            "val_",
            "",
            {"val_precision": 0.75, "val_num_true_positives": 3, "val_num_positive_predictions": 4},
            id="prefix-only",
        ),
        pytest.param(
            3,
            4,
            "",
            "_epoch1",
            {
                "precision_epoch1": 0.75,
                "num_true_positives_epoch1": 3,
                "num_positive_predictions_epoch1": 4,
            },
            id="suffix-only",
        ),
        pytest.param(
            3,
            4,
            "val_",
            "_epoch1",
            {
                "val_precision_epoch1": 0.75,
                "val_num_true_positives_epoch1": 3,
                "val_num_positive_predictions_epoch1": 4,
            },
            id="prefix-and-suffix",
        ),
        pytest.param(
            5,
            5,
            "",
            "",
            {"precision": 1.0, "num_true_positives": 5, "num_positive_predictions": 5},
            id="perfect",
        ),
        pytest.param(
            0,
            5,
            "",
            "",
            {"precision": 0.0, "num_true_positives": 0, "num_positive_predictions": 5},
            id="zero",
        ),
        pytest.param(
            0,
            0,
            "",
            "",
            {"precision": float("nan"), "num_true_positives": 0, "num_positive_predictions": 0},
            id="zero-predictions",
        ),
    ],
)
def test_precision_result_to_dict(
    num_true_positives: int,
    num_positive_predictions: int,
    prefix: str,
    suffix: str,
    expected: dict,
) -> None:
    assert objects_are_allclose(
        PrecisionResult(
            num_true_positives=num_true_positives,
            num_positive_predictions=num_positive_predictions,
        ).to_dict(prefix=prefix, suffix=suffix),
        expected,
        equal_nan=True,
    )


def test_precision_result_to_dict_nan_true_positives() -> None:
    assert objects_are_allclose(
        PrecisionResult(num_true_positives=float("nan"), num_positive_predictions=4).to_dict(),
        {
            "precision": float("nan"),
            "num_true_positives": float("nan"),
            "num_positive_predictions": 4,
        },
        equal_nan=True,
    )


# --- to_display ---


@pytest.mark.parametrize(
    ("num_true_positives", "num_positive_predictions", "expected"),
    [
        pytest.param(
            3,
            4,
            "Precision [███████████████░░░░░]  0.7500  (3/4)",
            id="standard",
        ),
        pytest.param(
            5,
            5,
            "Precision [████████████████████]  1.0000  (5/5)",
            id="perfect",
        ),
        pytest.param(
            0,
            5,
            "Precision [░░░░░░░░░░░░░░░░░░░░]  0.0000  (0/5)",
            id="zero",
        ),
        pytest.param(
            2,
            4,
            "Precision [██████████░░░░░░░░░░]  0.5000  (2/4)",
            id="half",
        ),
        pytest.param(
            0,
            0,
            "PrecisionResult: no predictions",
            id="zero-predictions",
        ),
        pytest.param(
            float("nan"),
            4,
            "Precision [????????????????????]  nan  (?/4)",
            id="nan-true-positives",
        ),
    ],
)
def test_precision_result_to_display(
    num_true_positives: float,
    num_positive_predictions: int,
    expected: str,
) -> None:
    assert (
        PrecisionResult(
            num_true_positives=num_true_positives,
            num_positive_predictions=num_positive_predictions,
        ).to_display()
        == expected
    )


def test_precision_result_to_display_returns_str() -> None:
    assert isinstance(
        PrecisionResult(num_true_positives=3, num_positive_predictions=4).to_display(),
        str,
    )


def test_precision_result_to_display_large_numbers() -> None:
    assert PrecisionResult(num_true_positives=1125, num_positive_predictions=1500).to_display() == (
        "Precision [███████████████░░░░░]  0.7500  (1,125/1,500)"
    )
