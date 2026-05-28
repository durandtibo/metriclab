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
    m = RecallResult(true_positives=3, false_negatives=2)
    assert m.true_positives == 3
    assert m.false_negatives == 2


def test_recall_result_zero_predictions() -> None:
    m = RecallResult(true_positives=0, false_negatives=0)
    assert m.true_positives == 0
    assert m.false_negatives == 0


def test_recall_result_nan_true_positives() -> None:
    m = RecallResult(true_positives=float("nan"), false_negatives=2)
    assert math.isnan(m.true_positives)


def test_recall_result_nan_false_negatives() -> None:
    m = RecallResult(true_positives=3, false_negatives=float("nan"))
    assert math.isnan(m.false_negatives)


def test_recall_result_frozen() -> None:
    m = RecallResult(true_positives=3, false_negatives=2)
    with pytest.raises(FrozenInstanceError):
        m.true_positives = 2


def test_recall_result_perfect() -> None:
    m = RecallResult(true_positives=5, false_negatives=0)
    assert m.recall == 1.0


def test_recall_result_zero_true_positives() -> None:
    m = RecallResult(true_positives=0, false_negatives=5)
    assert m.recall == 0.0


# --- validation ---


def test_recall_result_negative_true_positives_raises() -> None:
    with pytest.raises(ValueError, match=r"true_positives must be >= 0"):
        RecallResult(true_positives=-1, false_negatives=2)


def test_recall_result_negative_false_negatives_raises() -> None:
    with pytest.raises(ValueError, match=r"false_negatives must be >= 0"):
        RecallResult(true_positives=3, false_negatives=-1)


def test_recall_result_nan_true_positives_does_not_raise() -> None:
    RecallResult(true_positives=float("nan"), false_negatives=2)


def test_recall_result_nan_false_negatives_does_not_raise() -> None:
    RecallResult(true_positives=3, false_negatives=float("nan"))


def test_recall_result_zero_true_positives_does_not_raise() -> None:
    RecallResult(true_positives=0, false_negatives=5)


def test_recall_result_zero_false_negatives_does_not_raise() -> None:
    RecallResult(true_positives=5, false_negatives=0)


def test_recall_result_all_zero_does_not_raise() -> None:
    RecallResult(true_positives=0, false_negatives=0)


# --- num_actual_positives property ---


@pytest.mark.parametrize(
    ("true_positives", "false_negatives", "expected"),
    [
        pytest.param(3, 2, 5, id="standard"),
        pytest.param(5, 0, 5, id="no-false-negatives"),
        pytest.param(0, 5, 5, id="no-true-positives"),
        pytest.param(0, 0, 0, id="all-zero"),
    ],
)
def test_recall_result_num_actual_positives(
    true_positives: int, false_negatives: int, expected: int
) -> None:
    assert (
        RecallResult(
            true_positives=true_positives, false_negatives=false_negatives
        ).num_actual_positives
        == expected
    )


def test_recall_result_num_actual_positives_nan_true_positives() -> None:
    assert math.isnan(
        RecallResult(true_positives=float("nan"), false_negatives=2).num_actual_positives
    )


def test_recall_result_num_actual_positives_nan_false_negatives() -> None:
    assert math.isnan(
        RecallResult(true_positives=3, false_negatives=float("nan")).num_actual_positives
    )


# --- recall property ---


@pytest.mark.parametrize(
    ("true_positives", "false_negatives", "expected"),
    [
        pytest.param(3, 2, 0.6, id="standard"),
        pytest.param(5, 0, 1.0, id="perfect"),
        pytest.param(0, 5, 0.0, id="zero"),
        pytest.param(1, 1, 0.5, id="half"),
        pytest.param(1, 3, 0.25, id="quarter"),
    ],
)
def test_recall_result_recall(true_positives: int, false_negatives: int, expected: float) -> None:
    assert (
        RecallResult(true_positives=true_positives, false_negatives=false_negatives).recall
        == expected
    )


def test_recall_result_recall_zero_denominator_returns_zero() -> None:
    assert RecallResult(true_positives=0, false_negatives=0).recall == 0.0


def test_recall_result_recall_nan_true_positives_returns_nan() -> None:
    assert math.isnan(RecallResult(true_positives=float("nan"), false_negatives=2).recall)


def test_recall_result_recall_nan_false_negatives_returns_nan() -> None:
    assert math.isnan(RecallResult(true_positives=3, false_negatives=float("nan")).recall)


# --- equal ---


def test_recall_result_equal_true() -> None:
    assert RecallResult(true_positives=3, false_negatives=2).equal(
        RecallResult(true_positives=3, false_negatives=2)
    )


def test_recall_result_equal_false_different_true_positives() -> None:
    assert not RecallResult(true_positives=3, false_negatives=2).equal(
        RecallResult(true_positives=2, false_negatives=2)
    )


def test_recall_result_equal_false_different_false_negatives() -> None:
    assert not RecallResult(true_positives=3, false_negatives=2).equal(
        RecallResult(true_positives=3, false_negatives=3)
    )


def test_recall_result_equal_wrong_type() -> None:
    assert not RecallResult(true_positives=3, false_negatives=2).equal("not a result")


def test_recall_result_equal_nan_false_by_default() -> None:
    assert not RecallResult(true_positives=float("nan"), false_negatives=2).equal(
        RecallResult(true_positives=float("nan"), false_negatives=2)
    )


def test_recall_result_equal_nan_true_with_equal_nan() -> None:
    assert RecallResult(true_positives=float("nan"), false_negatives=2).equal(
        RecallResult(true_positives=float("nan"), false_negatives=2),
        equal_nan=True,
    )


# --- allclose ---


def test_recall_result_allclose_true() -> None:
    assert RecallResult(true_positives=3, false_negatives=2).allclose(
        RecallResult(true_positives=3, false_negatives=2)
    )


def test_recall_result_allclose_within_tolerance() -> None:
    assert RecallResult(true_positives=3.0, false_negatives=2).allclose(
        RecallResult(true_positives=3 + 1e-7, false_negatives=2),
        rtol=1e-5,
        atol=1e-6,
    )


def test_recall_result_allclose_outside_tolerance() -> None:
    assert not RecallResult(true_positives=3, false_negatives=2).allclose(
        RecallResult(true_positives=2, false_negatives=2),
        rtol=1e-5,
        atol=1e-8,
    )


def test_recall_result_allclose_false_different_false_negatives() -> None:
    assert not RecallResult(true_positives=3, false_negatives=2).allclose(
        RecallResult(true_positives=3, false_negatives=3)
    )


def test_recall_result_allclose_wrong_type() -> None:
    assert not RecallResult(true_positives=3, false_negatives=2).allclose("not a result")


def test_recall_result_allclose_nan_false_by_default() -> None:
    assert not RecallResult(true_positives=float("nan"), false_negatives=2).allclose(
        RecallResult(true_positives=float("nan"), false_negatives=2)
    )


def test_recall_result_allclose_nan_true_with_equal_nan() -> None:
    assert RecallResult(true_positives=float("nan"), false_negatives=2).allclose(
        RecallResult(true_positives=float("nan"), false_negatives=2),
        equal_nan=True,
    )


# --- from_recall ---


@pytest.mark.parametrize(
    ("recall", "num_actual_positives", "expected_tp", "expected_fn"),
    [
        pytest.param(0.6, 5, 3, 2, id="standard"),
        pytest.param(1.0, 5, 5, 0, id="perfect"),
        pytest.param(0.0, 5, 0, 5, id="zero"),
        pytest.param(0.5, 4, 2, 2, id="half"),
        pytest.param(0.5, 2, 1, 1, id="half-small"),
    ],
)
def test_recall_result_from_recall(
    recall: float,
    num_actual_positives: int,
    expected_tp: int,
    expected_fn: int,
) -> None:
    m = RecallResult.from_recall(recall=recall, num_actual_positives=num_actual_positives)
    assert m.equal(
        RecallResult(true_positives=expected_tp, false_negatives=expected_fn), equal_nan=True
    )
    assert m.num_actual_positives == num_actual_positives


def test_recall_result_from_recall_nan_recall() -> None:
    m = RecallResult.from_recall(recall=float("nan"), num_actual_positives=5)
    assert m.equal(
        RecallResult(true_positives=float("nan"), false_negatives=float("nan")), equal_nan=True
    )


def test_recall_result_from_recall_nan_num_actual_positives() -> None:
    m = RecallResult.from_recall(recall=0.6, num_actual_positives=float("nan"))
    assert m.equal(
        RecallResult(true_positives=float("nan"), false_negatives=float("nan")), equal_nan=True
    )


def test_recall_result_from_recall_zero_denominator() -> None:
    m = RecallResult.from_recall(recall=0.0, num_actual_positives=0)
    assert m.equal(RecallResult(true_positives=0, false_negatives=0))


def test_recall_result_from_recall_roundtrip() -> None:
    m = RecallResult(true_positives=3, false_negatives=2)
    m2 = RecallResult.from_recall(recall=m.recall, num_actual_positives=m.num_actual_positives)
    assert m.equal(m2)


# --- to_dict ---


@pytest.mark.parametrize(
    ("true_positives", "false_negatives", "prefix", "suffix", "expected"),
    [
        pytest.param(
            3,
            2,
            "",
            "",
            {"recall": 0.6, "true_positives": 3, "false_negatives": 2, "num_actual_positives": 5},
            id="standard",
        ),
        pytest.param(
            3,
            2,
            "val_",
            "",
            {
                "val_recall": 0.6,
                "val_true_positives": 3,
                "val_false_negatives": 2,
                "val_num_actual_positives": 5,
            },
            id="prefix-only",
        ),
        pytest.param(
            3,
            2,
            "",
            "_epoch1",
            {
                "recall_epoch1": 0.6,
                "true_positives_epoch1": 3,
                "false_negatives_epoch1": 2,
                "num_actual_positives_epoch1": 5,
            },
            id="suffix-only",
        ),
        pytest.param(
            3,
            2,
            "val_",
            "_epoch1",
            {
                "val_recall_epoch1": 0.6,
                "val_true_positives_epoch1": 3,
                "val_false_negatives_epoch1": 2,
                "val_num_actual_positives_epoch1": 5,
            },
            id="prefix-and-suffix",
        ),
        pytest.param(
            5,
            0,
            "",
            "",
            {"recall": 1.0, "true_positives": 5, "false_negatives": 0, "num_actual_positives": 5},
            id="perfect",
        ),
        pytest.param(
            0,
            5,
            "",
            "",
            {"recall": 0.0, "true_positives": 0, "false_negatives": 5, "num_actual_positives": 5},
            id="zero",
        ),
        pytest.param(
            0,
            0,
            "",
            "",
            {"recall": 0.0, "true_positives": 0, "false_negatives": 0, "num_actual_positives": 0},
            id="zero-predictions",
        ),
    ],
)
def test_recall_result_to_dict(
    true_positives: int,
    false_negatives: int,
    prefix: str,
    suffix: str,
    expected: dict,
) -> None:
    assert objects_are_allclose(
        RecallResult(
            true_positives=true_positives,
            false_negatives=false_negatives,
        ).to_dict(prefix=prefix, suffix=suffix),
        expected,
        equal_nan=True,
    )


def test_recall_result_to_dict_nan_true_positives() -> None:
    assert objects_are_allclose(
        RecallResult(true_positives=float("nan"), false_negatives=2).to_dict(),
        {
            "recall": float("nan"),
            "true_positives": float("nan"),
            "false_negatives": 2,
            "num_actual_positives": float("nan"),
        },
        equal_nan=True,
    )


def test_recall_result_to_dict_nan_false_negatives() -> None:
    assert objects_are_allclose(
        RecallResult(true_positives=3, false_negatives=float("nan")).to_dict(),
        {
            "recall": float("nan"),
            "true_positives": 3,
            "false_negatives": float("nan"),
            "num_actual_positives": float("nan"),
        },
        equal_nan=True,
    )


# --- to_display ---


@pytest.mark.parametrize(
    ("true_positives", "false_negatives", "expected"),
    [
        pytest.param(
            3,
            2,
            "Recall [████████████░░░░░░░░]  0.6000  (3/5)",
            id="standard",
        ),
        pytest.param(
            5,
            0,
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
            2,
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
            2,
            "Recall [????????????????????]  nan  (?/?)",
            id="nan-true-positives",
        ),
        pytest.param(
            3,
            float("nan"),
            "Recall [????????????????????]  nan  (3/?)",
            id="nan-false-negatives",
        ),
    ],
)
def test_recall_result_to_display(
    true_positives: float,
    false_negatives: float,
    expected: str,
) -> None:
    assert (
        RecallResult(
            true_positives=true_positives,
            false_negatives=false_negatives,
        ).to_display()
        == expected
    )


def test_recall_result_to_display_returns_str() -> None:
    assert isinstance(RecallResult(true_positives=3, false_negatives=2).to_display(), str)


def test_recall_result_to_display_large_numbers() -> None:
    assert (
        RecallResult(true_positives=1500, false_negatives=1000).to_display()
        == "Recall [████████████░░░░░░░░]  0.6000  (1,500/2,500)"
    )


# --- repr / str ---


def test_recall_result_repr() -> None:
    assert (
        repr(RecallResult(true_positives=3, false_negatives=2))
        == "RecallResult(true_positives=3, false_negatives=2)"
    )


def test_recall_result_str() -> None:
    assert (
        str(RecallResult(true_positives=3, false_negatives=2))
        == "RecallResult(true_positives=3, false_negatives=2)"
    )


######################################
#     Tests for compute_recall       #
######################################


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
