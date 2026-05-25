from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import pytest
from coola.equality import objects_are_allclose

from metriclab.results import RecallResult
from metriclab.results.classification.recall import compute_recall

##################################
#    Tests for RecallResult      #
##################################


# --- instantiation ---


def test_recall_result_instantiation() -> None:
    m = RecallResult(num_true_positives=3, num_actual_positives=5)
    assert m.num_true_positives == 3
    assert m.num_actual_positives == 5


def test_recall_result_zero_predictions() -> None:
    m = RecallResult(num_true_positives=0, num_actual_positives=0)
    assert m.num_true_positives == 0
    assert m.num_actual_positives == 0


def test_recall_result_nan_true_positives() -> None:
    m = RecallResult(num_true_positives=float("nan"), num_actual_positives=5)
    assert math.isnan(m.num_true_positives)


def test_recall_result_frozen() -> None:
    m = RecallResult(num_true_positives=3, num_actual_positives=5)
    with pytest.raises(FrozenInstanceError):
        m.num_true_positives = 2  # type: ignore[misc]


def test_recall_result_perfect() -> None:
    m = RecallResult(num_true_positives=5, num_actual_positives=5)
    assert m.recall == 1.0


def test_recall_result_zero_true_positives() -> None:
    m = RecallResult(num_true_positives=0, num_actual_positives=5)
    assert m.recall == 0.0


# --- validation ---


def test_recall_result_negative_num_true_positives_raises() -> None:
    with pytest.raises(ValueError, match="num_true_positives must be >= 0"):
        RecallResult(num_true_positives=-1, num_actual_positives=5)


def test_recall_result_negative_num_actual_positives_raises() -> None:
    with pytest.raises(ValueError, match="num_actual_positives must be >= 0"):
        RecallResult(num_true_positives=3, num_actual_positives=-1)


def test_recall_result_num_true_positives_exceeds_num_actual_positives_raises() -> None:
    with pytest.raises(ValueError, match="cannot exceed num_actual_positives"):
        RecallResult(num_true_positives=6, num_actual_positives=5)


def test_recall_result_nan_true_positives_does_not_raise() -> None:
    RecallResult(num_true_positives=float("nan"), num_actual_positives=5)


def test_recall_result_zero_true_positives_does_not_raise() -> None:
    RecallResult(num_true_positives=0, num_actual_positives=5)


def test_recall_result_num_true_positives_equals_num_actual_positives_does_not_raise() -> None:
    RecallResult(num_true_positives=5, num_actual_positives=5)


def test_recall_result_all_zero_does_not_raise() -> None:
    RecallResult(num_true_positives=0, num_actual_positives=0)


# --- recall property ---


@pytest.mark.parametrize(
    ("num_true_positives", "num_actual_positives", "expected"),
    [
        pytest.param(3, 5, 0.6, id="standard"),
        pytest.param(5, 5, 1.0, id="perfect"),
        pytest.param(0, 5, 0.0, id="zero"),
        pytest.param(1, 2, 0.5, id="half"),
        pytest.param(1, 4, 0.25, id="quarter"),
    ],
)
def test_recall_result_recall(
    num_true_positives: int,
    num_actual_positives: int,
    expected: float,
) -> None:
    assert (
        RecallResult(
            num_true_positives=num_true_positives,
            num_actual_positives=num_actual_positives,
        ).recall
        == expected
    )


def test_recall_result_recall_zero_denominator_returns_nan() -> None:
    assert math.isnan(RecallResult(num_true_positives=0, num_actual_positives=0).recall)


def test_recall_result_recall_nan_true_positives_returns_nan() -> None:
    assert math.isnan(RecallResult(num_true_positives=float("nan"), num_actual_positives=5).recall)


# --- equal ---


def test_recall_result_equal_true() -> None:
    assert RecallResult(num_true_positives=3, num_actual_positives=5).equal(
        RecallResult(num_true_positives=3, num_actual_positives=5)
    )


def test_recall_result_equal_false_different_num_true_positives() -> None:
    assert not RecallResult(num_true_positives=3, num_actual_positives=5).equal(
        RecallResult(num_true_positives=2, num_actual_positives=5)
    )


def test_recall_result_equal_false_different_num_actual_positives() -> None:
    assert not RecallResult(num_true_positives=3, num_actual_positives=5).equal(
        RecallResult(num_true_positives=3, num_actual_positives=4)
    )


def test_recall_result_equal_wrong_type() -> None:
    assert not RecallResult(num_true_positives=3, num_actual_positives=5).equal("not a result")


def test_recall_result_equal_nan_false_by_default() -> None:
    assert not RecallResult(num_true_positives=float("nan"), num_actual_positives=5).equal(
        RecallResult(num_true_positives=float("nan"), num_actual_positives=5)
    )


def test_recall_result_equal_nan_true_with_equal_nan() -> None:
    assert RecallResult(num_true_positives=float("nan"), num_actual_positives=5).equal(
        RecallResult(num_true_positives=float("nan"), num_actual_positives=5),
        equal_nan=True,
    )


# --- allclose ---


def test_recall_result_allclose_true() -> None:
    assert RecallResult(num_true_positives=3, num_actual_positives=5).allclose(
        RecallResult(num_true_positives=3, num_actual_positives=5)
    )


def test_recall_result_allclose_within_tolerance() -> None:
    assert RecallResult(num_true_positives=3.0, num_actual_positives=5).allclose(
        RecallResult(num_true_positives=3 + 1e-7, num_actual_positives=5),
        rtol=1e-5,
        atol=1e-6,
    )


def test_recall_result_allclose_outside_tolerance() -> None:
    assert not RecallResult(num_true_positives=3, num_actual_positives=5).allclose(
        RecallResult(num_true_positives=2, num_actual_positives=5),
        rtol=1e-5,
        atol=1e-8,
    )


def test_recall_result_allclose_false_different_num_actual_positives() -> None:
    assert not RecallResult(num_true_positives=3, num_actual_positives=5).allclose(
        RecallResult(num_true_positives=3, num_actual_positives=4)
    )


def test_recall_result_allclose_wrong_type() -> None:
    assert not RecallResult(num_true_positives=3, num_actual_positives=5).allclose("not a result")


def test_recall_result_allclose_nan_false_by_default() -> None:
    assert not RecallResult(num_true_positives=float("nan"), num_actual_positives=5).allclose(
        RecallResult(num_true_positives=float("nan"), num_actual_positives=5)
    )


def test_recall_result_allclose_nan_true_with_equal_nan() -> None:
    assert RecallResult(num_true_positives=float("nan"), num_actual_positives=5).allclose(
        RecallResult(num_true_positives=float("nan"), num_actual_positives=5),
        equal_nan=True,
    )


# --- from_recall ---


@pytest.mark.parametrize(
    ("recall", "num_actual_positives", "expected_tp"),
    [
        pytest.param(0.6, 5, 3, id="standard"),
        pytest.param(1.0, 5, 5, id="perfect"),
        pytest.param(0.0, 5, 0, id="zero"),
        pytest.param(0.5, 4, 2, id="half"),
        pytest.param(0.5, 2, 1, id="half-small"),
    ],
)
def test_recall_result_from_recall(
    recall: float, num_actual_positives: int, expected_tp: int
) -> None:
    m = RecallResult.from_recall(recall=recall, num_actual_positives=num_actual_positives)
    assert m.num_true_positives == expected_tp
    assert m.num_actual_positives == num_actual_positives


def test_recall_result_from_recall_nan_recall() -> None:
    m = RecallResult.from_recall(recall=float("nan"), num_actual_positives=5)
    assert math.isnan(m.num_true_positives)
    assert m.num_actual_positives == 5


def test_recall_result_from_recall_zero_denominator() -> None:
    m = RecallResult.from_recall(recall=0.0, num_actual_positives=0)
    assert m.num_true_positives == 0
    assert m.num_actual_positives == 0


def test_recall_result_from_recall_returns_instance() -> None:
    assert isinstance(
        RecallResult.from_recall(recall=0.6, num_actual_positives=5),
        RecallResult,
    )


def test_recall_result_from_recall_roundtrip() -> None:
    m = RecallResult(num_true_positives=3, num_actual_positives=5)
    m2 = RecallResult.from_recall(recall=m.recall, num_actual_positives=m.num_actual_positives)
    assert m.equal(m2)


# --- to_dict ---


@pytest.mark.parametrize(
    ("num_true_positives", "num_actual_positives", "prefix", "suffix", "expected"),
    [
        pytest.param(
            3,
            5,
            "",
            "",
            {"recall": 0.6, "num_true_positives": 3, "num_actual_positives": 5},
            id="standard",
        ),
        pytest.param(
            3,
            5,
            "val_",
            "",
            {"val_recall": 0.6, "val_num_true_positives": 3, "val_num_actual_positives": 5},
            id="prefix-only",
        ),
        pytest.param(
            3,
            5,
            "",
            "_epoch1",
            {
                "recall_epoch1": 0.6,
                "num_true_positives_epoch1": 3,
                "num_actual_positives_epoch1": 5,
            },
            id="suffix-only",
        ),
        pytest.param(
            3,
            5,
            "val_",
            "_epoch1",
            {
                "val_recall_epoch1": 0.6,
                "val_num_true_positives_epoch1": 3,
                "val_num_actual_positives_epoch1": 5,
            },
            id="prefix-and-suffix",
        ),
        pytest.param(
            5,
            5,
            "",
            "",
            {"recall": 1.0, "num_true_positives": 5, "num_actual_positives": 5},
            id="perfect",
        ),
        pytest.param(
            0,
            5,
            "",
            "",
            {"recall": 0.0, "num_true_positives": 0, "num_actual_positives": 5},
            id="zero",
        ),
        pytest.param(
            0,
            0,
            "",
            "",
            {"recall": float("nan"), "num_true_positives": 0, "num_actual_positives": 0},
            id="zero-predictions",
        ),
    ],
)
def test_recall_result_to_dict(
    num_true_positives: int,
    num_actual_positives: int,
    prefix: str,
    suffix: str,
    expected: dict,
) -> None:
    assert objects_are_allclose(
        RecallResult(
            num_true_positives=num_true_positives,
            num_actual_positives=num_actual_positives,
        ).to_dict(prefix=prefix, suffix=suffix),
        expected,
        equal_nan=True,
    )


def test_recall_result_to_dict_nan_true_positives() -> None:
    assert objects_are_allclose(
        RecallResult(num_true_positives=float("nan"), num_actual_positives=5).to_dict(),
        {"recall": float("nan"), "num_true_positives": float("nan"), "num_actual_positives": 5},
        equal_nan=True,
    )


# --- to_display ---


@pytest.mark.parametrize(
    ("num_true_positives", "num_actual_positives", "expected"),
    [
        pytest.param(
            3,
            5,
            "Recall [████████████░░░░░░░░]  0.6000  (3/5)",
            id="standard",
        ),
        pytest.param(
            5,
            5,
            "Recall [████████████████████]  1.0000  (5/5)",
            id="perfect",
        ),
        pytest.param(
            0,
            5,
            "Recall [░░░░░░░░░░░░░░░░░░░░]  0.0000  (0/5)",
            id="zero",
        ),
        pytest.param(
            2,
            4,
            "Recall [██████████░░░░░░░░░░]  0.5000  (2/4)",
            id="half",
        ),
        pytest.param(
            0,
            0,
            "RecallResult: no predictions",
            id="zero-predictions",
        ),
        pytest.param(
            float("nan"),
            5,
            "Recall [????????????????????]  nan  (?/5)",
            id="nan-true-positives",
        ),
    ],
)
def test_recall_result_to_display(
    num_true_positives: float,
    num_actual_positives: int,
    expected: str,
) -> None:
    assert (
        RecallResult(
            num_true_positives=num_true_positives,
            num_actual_positives=num_actual_positives,
        ).to_display()
        == expected
    )


def test_recall_result_to_display_returns_str() -> None:
    assert isinstance(
        RecallResult(num_true_positives=3, num_actual_positives=5).to_display(),
        str,
    )


def test_recall_result_to_display_large_numbers() -> None:
    assert RecallResult(num_true_positives=1500, num_actual_positives=2500).to_display() == (
        "Recall [████████████░░░░░░░░]  0.6000  (1,500/2,500)"
    )


# --- repr / str ---


def test_recall_result_repr() -> None:
    assert (
        repr(RecallResult(num_true_positives=3, num_actual_positives=5))
        == "RecallResult(num_true_positives=3, num_actual_positives=5)"
    )


def test_recall_result_str() -> None:
    assert (
        str(RecallResult(num_true_positives=3, num_actual_positives=5))
        == "RecallResult(num_true_positives=3, num_actual_positives=5)"
    )


####################################
#     Tests for compute_recall     #
####################################


@pytest.mark.parametrize(
    ("true_positives", "false_negatives", "expected"),
    [
        pytest.param(3, 2, 0.6, id="standard"),
        pytest.param(5, 0, 1.0, id="no-false-negatives"),
        pytest.param(0, 5, 0.0, id="no-true-positives"),
        pytest.param(1, 1, 0.5, id="equal-tp-fn"),
    ],
)
def test_compute_recall(
    true_positives: float,
    false_negatives: float,
    expected: float,
) -> None:
    assert compute_recall(true_positives, false_negatives) == expected


def test_compute_recall_zero_denominator() -> None:
    assert compute_recall(true_positives=0, false_negatives=0) == 0.0


@pytest.mark.parametrize(
    ("true_positives", "false_negatives"),
    [
        pytest.param(float("nan"), 2, id="nan-tp"),
        pytest.param(3, float("nan"), id="nan-fn"),
        pytest.param(float("nan"), float("nan"), id="nan-all"),
    ],
)
def test_compute_recall_nan(
    true_positives: float,
    false_negatives: float,
) -> None:
    assert math.isnan(compute_recall(true_positives, false_negatives))
