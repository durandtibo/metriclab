from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import pytest
from coola.equality import objects_are_allclose

from metriclab.results import PrecisionResult
from metriclab.results.classification.precision import compute_precision

#####################################
#     Tests for PrecisionResult     #
#####################################


# --- instantiation ---


def test_precision_result_instantiation() -> None:
    m = PrecisionResult(true_positives=3, false_positives=1)
    assert m.true_positives == 3
    assert m.false_positives == 1


def test_precision_result_zero_predictions() -> None:
    m = PrecisionResult(true_positives=0, false_positives=0)
    assert m.true_positives == 0
    assert m.false_positives == 0


def test_precision_result_nan_true_positives() -> None:
    m = PrecisionResult(true_positives=float("nan"), false_positives=1)
    assert math.isnan(m.true_positives)


def test_precision_result_nan_false_positives() -> None:
    m = PrecisionResult(true_positives=3, false_positives=float("nan"))
    assert math.isnan(m.false_positives)


def test_precision_result_frozen() -> None:
    m = PrecisionResult(true_positives=3, false_positives=1)
    with pytest.raises(FrozenInstanceError, match="cannot assign to field 'true_positives'"):
        m.true_positives = 2


def test_precision_result_perfect() -> None:
    m = PrecisionResult(true_positives=5, false_positives=0)
    assert m.precision == 1.0


def test_precision_result_zero_true_positives() -> None:
    m = PrecisionResult(true_positives=0, false_positives=5)
    assert m.precision == 0.0


# --- validation ---


def test_precision_result_negative_true_positives_raises() -> None:
    with pytest.raises(ValueError, match="true_positives must be >= 0"):
        PrecisionResult(true_positives=-1, false_positives=1)


def test_precision_result_negative_false_positives_raises() -> None:
    with pytest.raises(ValueError, match="false_positives must be >= 0"):
        PrecisionResult(true_positives=3, false_positives=-1)


def test_precision_result_nan_true_positives_does_not_raise() -> None:
    PrecisionResult(true_positives=float("nan"), false_positives=1)


def test_precision_result_nan_false_positives_does_not_raise() -> None:
    PrecisionResult(true_positives=3, false_positives=float("nan"))


def test_precision_result_zero_true_positives_does_not_raise() -> None:
    PrecisionResult(true_positives=0, false_positives=5)


def test_precision_result_zero_false_positives_does_not_raise() -> None:
    PrecisionResult(true_positives=5, false_positives=0)


def test_precision_result_all_zero_does_not_raise() -> None:
    PrecisionResult(true_positives=0, false_positives=0)


# --- num_positive_predictions property ---


@pytest.mark.parametrize(
    ("true_positives", "false_positives", "expected"),
    [
        pytest.param(3, 1, 4, id="standard"),
        pytest.param(5, 0, 5, id="no-false-positives"),
        pytest.param(0, 5, 5, id="no-true-positives"),
        pytest.param(0, 0, 0, id="all-zero"),
    ],
)
def test_precision_result_num_positive_predictions(
    true_positives: int, false_positives: int, expected: int
) -> None:
    assert (
        PrecisionResult(
            true_positives=true_positives, false_positives=false_positives
        ).num_positive_predictions
        == expected
    )


def test_precision_result_num_positive_predictions_nan_true_positives() -> None:
    assert math.isnan(
        PrecisionResult(true_positives=float("nan"), false_positives=1).num_positive_predictions
    )


def test_precision_result_num_positive_predictions_nan_false_positives() -> None:
    assert math.isnan(
        PrecisionResult(true_positives=3, false_positives=float("nan")).num_positive_predictions
    )


# --- precision property ---


@pytest.mark.parametrize(
    ("true_positives", "false_positives", "expected"),
    [
        pytest.param(3, 1, 0.75, id="standard"),
        pytest.param(5, 0, 1.0, id="perfect"),
        pytest.param(0, 5, 0.0, id="zero"),
        pytest.param(1, 1, 0.5, id="half"),
        pytest.param(1, 3, 0.25, id="quarter"),
    ],
)
def test_precision_result_precision(
    true_positives: int, false_positives: int, expected: float
) -> None:
    assert (
        PrecisionResult(true_positives=true_positives, false_positives=false_positives).precision
        == expected
    )


def test_precision_result_precision_zero_denominator_returns_zero() -> None:
    assert PrecisionResult(true_positives=0, false_positives=0).precision == 0.0


def test_precision_result_precision_nan_true_positives_returns_nan() -> None:
    assert math.isnan(PrecisionResult(true_positives=float("nan"), false_positives=1).precision)


def test_precision_result_precision_nan_false_positives_returns_nan() -> None:
    assert math.isnan(PrecisionResult(true_positives=3, false_positives=float("nan")).precision)


# --- equal ---


def test_precision_result_equal_true() -> None:
    assert PrecisionResult(true_positives=3, false_positives=1).equal(
        PrecisionResult(true_positives=3, false_positives=1)
    )


def test_precision_result_equal_false_different_true_positives() -> None:
    assert not PrecisionResult(true_positives=3, false_positives=1).equal(
        PrecisionResult(true_positives=2, false_positives=1)
    )


def test_precision_result_equal_false_different_false_positives() -> None:
    assert not PrecisionResult(true_positives=3, false_positives=1).equal(
        PrecisionResult(true_positives=3, false_positives=2)
    )


def test_precision_result_equal_wrong_type() -> None:
    assert not PrecisionResult(true_positives=3, false_positives=1).equal("not a result")


def test_precision_result_equal_nan_false_by_default() -> None:
    assert not PrecisionResult(true_positives=float("nan"), false_positives=1).equal(
        PrecisionResult(true_positives=float("nan"), false_positives=1)
    )


def test_precision_result_equal_nan_true_with_equal_nan() -> None:
    assert PrecisionResult(true_positives=float("nan"), false_positives=1).equal(
        PrecisionResult(true_positives=float("nan"), false_positives=1),
        equal_nan=True,
    )


# --- allclose ---


def test_precision_result_allclose_true() -> None:
    assert PrecisionResult(true_positives=3, false_positives=1).allclose(
        PrecisionResult(true_positives=3, false_positives=1)
    )


def test_precision_result_allclose_within_tolerance() -> None:
    assert PrecisionResult(true_positives=3.0, false_positives=1).allclose(
        PrecisionResult(true_positives=3 + 1e-7, false_positives=1),
        rtol=1e-5,
        atol=1e-6,
    )


def test_precision_result_allclose_outside_tolerance() -> None:
    assert not PrecisionResult(true_positives=3, false_positives=1).allclose(
        PrecisionResult(true_positives=2, false_positives=1),
        rtol=1e-5,
        atol=1e-8,
    )


def test_precision_result_allclose_false_different_false_positives() -> None:
    assert not PrecisionResult(true_positives=3, false_positives=1).allclose(
        PrecisionResult(true_positives=3, false_positives=2)
    )


def test_precision_result_allclose_wrong_type() -> None:
    assert not PrecisionResult(true_positives=3, false_positives=1).allclose("not a result")


def test_precision_result_allclose_nan_false_by_default() -> None:
    assert not PrecisionResult(true_positives=float("nan"), false_positives=1).allclose(
        PrecisionResult(true_positives=float("nan"), false_positives=1)
    )


def test_precision_result_allclose_nan_true_with_equal_nan() -> None:
    assert PrecisionResult(true_positives=float("nan"), false_positives=1).allclose(
        PrecisionResult(true_positives=float("nan"), false_positives=1),
        equal_nan=True,
    )


# --- from_precision ---


@pytest.mark.parametrize(
    ("precision", "num_positive_predictions", "expected_tp", "expected_fp"),
    [
        pytest.param(0.75, 4, 3, 1, id="standard"),
        pytest.param(1.0, 5, 5, 0, id="perfect"),
        pytest.param(0.0, 5, 0, 5, id="zero"),
        pytest.param(0.5, 4, 2, 2, id="half"),
        pytest.param(0.5, 2, 1, 1, id="half-small"),
    ],
)
def test_precision_result_from_precision(
    precision: float,
    num_positive_predictions: int,
    expected_tp: int,
    expected_fp: int,
) -> None:
    m = PrecisionResult.from_precision(
        precision=precision, num_positive_predictions=num_positive_predictions
    )
    assert m.equal(PrecisionResult(true_positives=expected_tp, false_positives=expected_fp))
    assert m.num_positive_predictions == num_positive_predictions


def test_precision_result_from_precision_nan_precision() -> None:
    m = PrecisionResult.from_precision(precision=float("nan"), num_positive_predictions=4)
    assert m.equal(
        PrecisionResult(true_positives=float("nan"), false_positives=float("nan")), equal_nan=True
    )


def test_precision_result_from_precision_nan_num_positive_predictions() -> None:
    m = PrecisionResult.from_precision(precision=0.75, num_positive_predictions=float("nan"))
    assert m.equal(
        PrecisionResult(true_positives=float("nan"), false_positives=float("nan")), equal_nan=True
    )


def test_precision_result_from_precision_zero_denominator() -> None:
    m = PrecisionResult.from_precision(precision=0.0, num_positive_predictions=0)
    assert m.equal(PrecisionResult(true_positives=0, false_positives=0))
    assert m.num_positive_predictions == 0


def test_precision_result_from_precision_roundtrip() -> None:
    m = PrecisionResult(true_positives=3, false_positives=1)
    m2 = PrecisionResult.from_precision(
        precision=m.precision, num_positive_predictions=m.num_positive_predictions
    )
    assert m.equal(m2)


# --- to_dict ---


@pytest.mark.parametrize(
    ("true_positives", "false_positives", "prefix", "suffix", "expected"),
    [
        pytest.param(
            3,
            1,
            "",
            "",
            {
                "precision": 0.75,
                "true_positives": 3,
                "false_positives": 1,
                "num_positive_predictions": 4,
            },
            id="standard",
        ),
        pytest.param(
            3,
            1,
            "val_",
            "",
            {
                "val_precision": 0.75,
                "val_true_positives": 3,
                "val_false_positives": 1,
                "val_num_positive_predictions": 4,
            },
            id="prefix-only",
        ),
        pytest.param(
            3,
            1,
            "",
            "_epoch1",
            {
                "precision_epoch1": 0.75,
                "true_positives_epoch1": 3,
                "false_positives_epoch1": 1,
                "num_positive_predictions_epoch1": 4,
            },
            id="suffix-only",
        ),
        pytest.param(
            3,
            1,
            "val_",
            "_epoch1",
            {
                "val_precision_epoch1": 0.75,
                "val_true_positives_epoch1": 3,
                "val_false_positives_epoch1": 1,
                "val_num_positive_predictions_epoch1": 4,
            },
            id="prefix-and-suffix",
        ),
        pytest.param(
            5,
            0,
            "",
            "",
            {
                "precision": 1.0,
                "true_positives": 5,
                "false_positives": 0,
                "num_positive_predictions": 5,
            },
            id="perfect",
        ),
        pytest.param(
            0,
            5,
            "",
            "",
            {
                "precision": 0.0,
                "true_positives": 0,
                "false_positives": 5,
                "num_positive_predictions": 5,
            },
            id="zero",
        ),
        pytest.param(
            0,
            0,
            "",
            "",
            {
                "precision": 0.0,
                "true_positives": 0,
                "false_positives": 0,
                "num_positive_predictions": 0,
            },
            id="zero-predictions",
        ),
    ],
)
def test_precision_result_to_dict(
    true_positives: int,
    false_positives: int,
    prefix: str,
    suffix: str,
    expected: dict,
) -> None:
    assert objects_are_allclose(
        PrecisionResult(
            true_positives=true_positives,
            false_positives=false_positives,
        ).to_dict(prefix=prefix, suffix=suffix),
        expected,
        equal_nan=True,
    )


def test_precision_result_to_dict_nan_true_positives() -> None:
    assert objects_are_allclose(
        PrecisionResult(true_positives=float("nan"), false_positives=1).to_dict(),
        {
            "precision": float("nan"),
            "true_positives": float("nan"),
            "false_positives": 1,
            "num_positive_predictions": float("nan"),
        },
        equal_nan=True,
    )


def test_precision_result_to_dict_nan_false_positives() -> None:
    assert objects_are_allclose(
        PrecisionResult(true_positives=3, false_positives=float("nan")).to_dict(),
        {
            "precision": float("nan"),
            "true_positives": 3,
            "false_positives": float("nan"),
            "num_positive_predictions": float("nan"),
        },
        equal_nan=True,
    )


# --- to_display ---


@pytest.mark.parametrize(
    ("true_positives", "false_positives", "expected"),
    [
        pytest.param(
            3,
            1,
            "Precision [███████████████░░░░░]  0.7500  (3/4)",
            id="standard",
        ),
        pytest.param(
            5,
            0,
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
            2,
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
            1,
            "Precision [????????????????????]  nan  (?/?)",
            id="nan-true-positives",
        ),
        pytest.param(
            3,
            float("nan"),
            "Precision [????????????????????]  nan  (3/?)",
            id="nan-false-positives",
        ),
    ],
)
def test_precision_result_to_display(
    true_positives: float,
    false_positives: float,
    expected: str,
) -> None:
    assert (
        PrecisionResult(
            true_positives=true_positives,
            false_positives=false_positives,
        ).to_display()
        == expected
    )


def test_precision_result_to_display_returns_str() -> None:
    assert isinstance(PrecisionResult(true_positives=3, false_positives=1).to_display(), str)


def test_precision_result_to_display_large_numbers() -> None:
    assert (
        PrecisionResult(true_positives=1125, false_positives=375).to_display()
        == "Precision [███████████████░░░░░]  0.7500  (1,125/1,500)"
    )


# --- repr / str ---


def test_precision_result_repr() -> None:
    assert (
        repr(PrecisionResult(true_positives=3, false_positives=1))
        == "PrecisionResult(true_positives=3, false_positives=1)"
    )


def test_precision_result_str() -> None:
    assert (
        str(PrecisionResult(true_positives=3, false_positives=1))
        == "PrecisionResult(true_positives=3, false_positives=1)"
    )


########################################
#     Tests for compute_precision      #
########################################


@pytest.mark.parametrize(
    ("true_positives", "false_positives", "expected"),
    [
        pytest.param(3, 1, 0.75, id="standard"),
        pytest.param(5, 0, 1.0, id="no-false-positives"),
        pytest.param(0, 5, 0.0, id="no-true-positives"),
        pytest.param(1, 1, 0.5, id="equal-tp-fp"),
    ],
)
def test_compute_precision(
    true_positives: float,
    false_positives: float,
    expected: float,
) -> None:
    assert compute_precision(true_positives, false_positives) == expected


def test_compute_precision_zero_denominator() -> None:
    assert compute_precision(true_positives=0, false_positives=0) == 0.0


@pytest.mark.parametrize(
    ("true_positives", "false_positives"),
    [
        pytest.param(float("nan"), 1, id="nan-tp"),
        pytest.param(3, float("nan"), id="nan-fp"),
        pytest.param(float("nan"), float("nan"), id="nan-all"),
    ],
)
def test_compute_precision_nan(
    true_positives: float,
    false_positives: float,
) -> None:
    assert math.isnan(compute_precision(true_positives, false_positives))
