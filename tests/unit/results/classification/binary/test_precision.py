from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import pytest
from coola.equality import objects_are_allclose

from metriclab.results import BinaryPrecisionResult
from metriclab.results.classification.binary.precision import compute_precision

############################################
#     Tests for BinaryPrecisionResult      #
############################################


# --- instantiation ---


def test_binary_precision_result_instantiation() -> None:
    m = BinaryPrecisionResult(precision=0.75, num_predictions=10, num_positive_predictions=4)
    assert m.precision == 0.75
    assert m.num_predictions == 10
    assert m.num_positive_predictions == 4


def test_binary_precision_result_zero_predictions() -> None:
    m = BinaryPrecisionResult(precision=0.0, num_predictions=0, num_positive_predictions=0)
    assert m.precision == 0.0
    assert m.num_predictions == 0
    assert m.num_positive_predictions == 0


def test_binary_precision_result_nan_precision() -> None:
    m = BinaryPrecisionResult(
        precision=float("nan"), num_predictions=10, num_positive_predictions=4
    )
    assert math.isnan(m.precision)


def test_binary_precision_result_frozen() -> None:
    m = BinaryPrecisionResult(precision=0.75, num_predictions=10, num_positive_predictions=4)
    with pytest.raises(FrozenInstanceError, match=r"cannot assign to field 'precision'"):
        m.precision = 0.5


def test_binary_precision_result_perfect() -> None:
    m = BinaryPrecisionResult(precision=1.0, num_predictions=10, num_positive_predictions=10)
    assert m.precision == 1.0


def test_binary_precision_result_zero_precision() -> None:
    m = BinaryPrecisionResult(precision=0.0, num_predictions=10, num_positive_predictions=4)
    assert m.precision == 0.0


# --- validation ---


def test_binary_precision_result_negative_num_predictions_raises() -> None:
    with pytest.raises(ValueError, match=r"'num_predictions' must be >= 0"):
        BinaryPrecisionResult(precision=0.75, num_predictions=-1, num_positive_predictions=0)


def test_binary_precision_result_negative_num_positive_predictions_raises() -> None:
    with pytest.raises(ValueError, match=r"'num_positive_predictions' must be >= 0"):
        BinaryPrecisionResult(precision=0.75, num_predictions=10, num_positive_predictions=-1)


def test_binary_precision_result_num_positive_predictions_exceeds_num_predictions_raises() -> None:
    with pytest.raises(
        ValueError,
        match=r"'num_positive_predictions' \(6\) must be <= 'num_predictions' \(5\)",
    ):
        BinaryPrecisionResult(precision=0.75, num_predictions=5, num_positive_predictions=6)


def test_binary_precision_result_negative_precision_raises() -> None:
    with pytest.raises(ValueError, match=r"'precision' must be >= 0"):
        BinaryPrecisionResult(precision=-0.1, num_predictions=10, num_positive_predictions=4)


def test_binary_precision_result_nan_precision_does_not_raise() -> None:
    BinaryPrecisionResult(precision=float("nan"), num_predictions=10, num_positive_predictions=4)


def test_binary_precision_result_zero_precision_does_not_raise() -> None:
    BinaryPrecisionResult(precision=0.0, num_predictions=10, num_positive_predictions=4)


def test_binary_precision_result_one_precision_does_not_raise() -> None:
    BinaryPrecisionResult(precision=1.0, num_predictions=10, num_positive_predictions=10)


def test_binary_precision_result_zero_num_predictions_does_not_raise() -> None:
    BinaryPrecisionResult(precision=0.0, num_predictions=0, num_positive_predictions=0)


def test_binary_precision_result_num_positive_predictions_equal_num_predictions_does_not_raise() -> (
    None
):
    BinaryPrecisionResult(precision=1.0, num_predictions=5, num_positive_predictions=5)


# --- equal ---


def test_binary_precision_result_equal_true() -> None:
    assert BinaryPrecisionResult(
        precision=0.75, num_predictions=10, num_positive_predictions=4
    ).equal(BinaryPrecisionResult(precision=0.75, num_predictions=10, num_positive_predictions=4))


def test_binary_precision_result_equal_false_different_precision() -> None:
    assert not BinaryPrecisionResult(
        precision=0.75, num_predictions=10, num_positive_predictions=4
    ).equal(BinaryPrecisionResult(precision=0.8, num_predictions=10, num_positive_predictions=4))


def test_binary_precision_result_equal_false_different_num_predictions() -> None:
    assert not BinaryPrecisionResult(
        precision=0.75, num_predictions=10, num_positive_predictions=4
    ).equal(BinaryPrecisionResult(precision=0.75, num_predictions=5, num_positive_predictions=4))


def test_binary_precision_result_equal_false_different_num_positive_predictions() -> None:
    assert not BinaryPrecisionResult(
        precision=0.75, num_predictions=10, num_positive_predictions=4
    ).equal(BinaryPrecisionResult(precision=0.75, num_predictions=10, num_positive_predictions=3))


def test_binary_precision_result_equal_wrong_type() -> None:
    assert not BinaryPrecisionResult(
        precision=0.75, num_predictions=10, num_positive_predictions=4
    ).equal("not a result")


def test_binary_precision_result_equal_nan_false_by_default() -> None:
    assert not BinaryPrecisionResult(
        precision=float("nan"), num_predictions=10, num_positive_predictions=4
    ).equal(
        BinaryPrecisionResult(
            precision=float("nan"), num_predictions=10, num_positive_predictions=4
        )
    )


def test_binary_precision_result_equal_nan_true_with_equal_nan() -> None:
    assert BinaryPrecisionResult(
        precision=float("nan"), num_predictions=10, num_positive_predictions=4
    ).equal(
        BinaryPrecisionResult(
            precision=float("nan"), num_predictions=10, num_positive_predictions=4
        ),
        equal_nan=True,
    )


# --- allclose ---


def test_binary_precision_result_allclose_true() -> None:
    assert BinaryPrecisionResult(
        precision=0.75, num_predictions=10, num_positive_predictions=4
    ).allclose(
        BinaryPrecisionResult(precision=0.75, num_predictions=10, num_positive_predictions=4)
    )


def test_binary_precision_result_allclose_within_tolerance() -> None:
    assert BinaryPrecisionResult(
        precision=0.75, num_predictions=10, num_positive_predictions=4
    ).allclose(
        BinaryPrecisionResult(
            precision=0.75 + 1e-7, num_predictions=10, num_positive_predictions=4
        ),
        rtol=1e-5,
        atol=1e-6,
    )


def test_binary_precision_result_allclose_outside_tolerance() -> None:
    assert not BinaryPrecisionResult(
        precision=0.75, num_predictions=10, num_positive_predictions=4
    ).allclose(
        BinaryPrecisionResult(precision=0.8, num_predictions=10, num_positive_predictions=4),
        rtol=1e-5,
        atol=1e-8,
    )


def test_binary_precision_result_allclose_false_different_num_predictions() -> None:
    assert not BinaryPrecisionResult(
        precision=0.75, num_predictions=10, num_positive_predictions=4
    ).allclose(BinaryPrecisionResult(precision=0.75, num_predictions=5, num_positive_predictions=4))


def test_binary_precision_result_allclose_false_different_num_positive_predictions() -> None:
    assert not BinaryPrecisionResult(
        precision=0.75, num_predictions=10, num_positive_predictions=4
    ).allclose(
        BinaryPrecisionResult(precision=0.75, num_predictions=10, num_positive_predictions=3)
    )


def test_binary_precision_result_allclose_wrong_type() -> None:
    assert not BinaryPrecisionResult(
        precision=0.75, num_predictions=10, num_positive_predictions=4
    ).allclose("not a result")


def test_binary_precision_result_allclose_nan_false_by_default() -> None:
    assert not BinaryPrecisionResult(
        precision=float("nan"), num_predictions=10, num_positive_predictions=4
    ).allclose(
        BinaryPrecisionResult(
            precision=float("nan"), num_predictions=10, num_positive_predictions=4
        )
    )


def test_binary_precision_result_allclose_nan_true_with_equal_nan() -> None:
    assert BinaryPrecisionResult(
        precision=float("nan"), num_predictions=10, num_positive_predictions=4
    ).allclose(
        BinaryPrecisionResult(
            precision=float("nan"), num_predictions=10, num_positive_predictions=4
        ),
        equal_nan=True,
    )


# --- to_dict ---


@pytest.mark.parametrize(
    ("precision", "num_predictions", "num_positive_predictions", "prefix", "suffix", "expected"),
    [
        pytest.param(
            0.75,
            10,
            4,
            "",
            "",
            {"precision": 0.75, "num_predictions": 10, "num_positive_predictions": 4},
            id="standard",
        ),
        pytest.param(
            0.75,
            10,
            4,
            "val_",
            "",
            {"val_precision": 0.75, "val_num_predictions": 10, "val_num_positive_predictions": 4},
            id="prefix-only",
        ),
        pytest.param(
            0.75,
            10,
            4,
            "",
            "_epoch1",
            {
                "precision_epoch1": 0.75,
                "num_predictions_epoch1": 10,
                "num_positive_predictions_epoch1": 4,
            },
            id="suffix-only",
        ),
        pytest.param(
            0.75,
            10,
            4,
            "val_",
            "_epoch1",
            {
                "val_precision_epoch1": 0.75,
                "val_num_predictions_epoch1": 10,
                "val_num_positive_predictions_epoch1": 4,
            },
            id="prefix-and-suffix",
        ),
        pytest.param(
            1.0,
            10,
            10,
            "",
            "",
            {"precision": 1.0, "num_predictions": 10, "num_positive_predictions": 10},
            id="perfect",
        ),
        pytest.param(
            0.0,
            10,
            4,
            "",
            "",
            {"precision": 0.0, "num_predictions": 10, "num_positive_predictions": 4},
            id="zero",
        ),
        pytest.param(
            0.0,
            0,
            0,
            "",
            "",
            {"precision": 0.0, "num_predictions": 0, "num_positive_predictions": 0},
            id="zero-predictions",
        ),
    ],
)
def test_binary_precision_result_to_dict(
    precision: float,
    num_predictions: int,
    num_positive_predictions: int,
    prefix: str,
    suffix: str,
    expected: dict,
) -> None:
    assert objects_are_allclose(
        BinaryPrecisionResult(
            precision=precision,
            num_predictions=num_predictions,
            num_positive_predictions=num_positive_predictions,
        ).to_dict(prefix=prefix, suffix=suffix),
        expected,
    )


def test_binary_precision_result_to_dict_nan() -> None:
    assert objects_are_allclose(
        BinaryPrecisionResult(
            precision=float("nan"), num_predictions=10, num_positive_predictions=4
        ).to_dict(),
        {"precision": float("nan"), "num_predictions": 10, "num_positive_predictions": 4},
        equal_nan=True,
    )


# --- to_display ---


@pytest.mark.parametrize(
    ("precision", "num_predictions", "num_positive_predictions", "expected"),
    [
        pytest.param(
            0.75,
            10,
            4,
            "Precision [███████████████░░░░░]  0.7500  (3/4)  [n=10]",
            id="standard",
        ),
        pytest.param(
            1.0,
            10,
            10,
            "Precision [████████████████████]  1.0000  (10/10)  [n=10]",
            id="perfect",
        ),
        pytest.param(
            0.0,
            10,
            4,
            "Precision [░░░░░░░░░░░░░░░░░░░░]  0.0000  (0/4)  [n=10]",
            id="zero-precision",
        ),
        pytest.param(
            0.5,
            10,
            4,
            "Precision [██████████░░░░░░░░░░]  0.5000  (2/4)  [n=10]",
            id="half",
        ),
        pytest.param(
            0.0,
            0,
            0,
            "BinaryPrecisionResult: no predictions",
            id="zero-predictions",
        ),
        pytest.param(
            float("nan"),
            10,
            4,
            "Precision [????????????????????]  nan  [n=10]",
            id="nan",
        ),
    ],
)
def test_binary_precision_result_to_display(
    precision: float,
    num_predictions: int,
    num_positive_predictions: int,
    expected: str,
) -> None:
    assert (
        BinaryPrecisionResult(
            precision=precision,
            num_predictions=num_predictions,
            num_positive_predictions=num_positive_predictions,
        ).to_display()
        == expected
    )


def test_binary_precision_result_to_display_returns_str() -> None:
    assert isinstance(
        BinaryPrecisionResult(
            precision=0.75, num_predictions=10, num_positive_predictions=4
        ).to_display(),
        str,
    )


# --- repr / str ---


def test_binary_precision_result_repr() -> None:
    assert (
        repr(BinaryPrecisionResult(precision=0.75, num_predictions=10, num_positive_predictions=4))
        == "BinaryPrecisionResult(precision=0.75, num_predictions=10, num_positive_predictions=4)"
    )


def test_binary_precision_result_str() -> None:
    assert (
        str(BinaryPrecisionResult(precision=0.75, num_predictions=10, num_positive_predictions=4))
        == "BinaryPrecisionResult(precision=0.75, num_predictions=10, num_positive_predictions=4)"
    )


@pytest.mark.parametrize(
    ("true_positives", "false_positives", "num_predictions", "expected"),
    [
        pytest.param(
            3,
            1,
            10,
            BinaryPrecisionResult(precision=0.75, num_predictions=10, num_positive_predictions=4),
            id="standard",
        ),
        pytest.param(
            4,
            0,
            10,
            BinaryPrecisionResult(precision=1.0, num_predictions=10, num_positive_predictions=4),
            id="perfect-no-false-positives",
        ),
        pytest.param(
            0,
            4,
            10,
            BinaryPrecisionResult(precision=0.0, num_predictions=10, num_positive_predictions=4),
            id="zero-true-positives",
        ),
        pytest.param(
            0,
            0,
            10,
            BinaryPrecisionResult(
                precision=float("nan"), num_predictions=10, num_positive_predictions=0
            ),
            id="no-positive-predictions",
        ),
        pytest.param(
            0,
            0,
            0,
            BinaryPrecisionResult(
                precision=float("nan"), num_predictions=0, num_positive_predictions=0
            ),
            id="no-predictions-at-all",
        ),
        pytest.param(
            float("nan"),
            1,
            10,
            BinaryPrecisionResult(
                precision=float("nan"), num_predictions=10, num_positive_predictions=float("nan")
            ),
            id="tp-nan",
        ),
        pytest.param(
            1,
            float("nan"),
            10,
            BinaryPrecisionResult(
                precision=float("nan"), num_predictions=10, num_positive_predictions=float("nan")
            ),
            id="fn-nan",
        ),
        pytest.param(
            float("nan"),
            float("nan"),
            10,
            BinaryPrecisionResult(
                precision=float("nan"), num_predictions=10, num_positive_predictions=float("nan")
            ),
            id="both-nan",
        ),
        pytest.param(
            1,
            1,
            4,
            BinaryPrecisionResult(precision=0.5, num_predictions=4, num_positive_predictions=2),
            id="half-precision",
        ),
        pytest.param(
            10,
            10,
            100,
            BinaryPrecisionResult(precision=0.5, num_predictions=100, num_positive_predictions=20),
            id="large-counts",
        ),
    ],
)
def test_binary_precision_result_from_tp_fp(
    true_positives: float,
    false_positives: float,
    num_predictions: int,
    expected: BinaryPrecisionResult,
) -> None:
    assert BinaryPrecisionResult.from_tp_fp(
        true_positives=true_positives,
        false_positives=false_positives,
        num_predictions=num_predictions,
    ).allclose(expected, equal_nan=True)


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
