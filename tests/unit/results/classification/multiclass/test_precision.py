from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from coola.equality import objects_are_allclose, objects_are_equal

from metriclab.results import MulticlassPrecisionResult

###################################################
#     Tests for MulticlassPrecisionResult         #
###################################################


# --- fixtures ---


@pytest.fixture
def result() -> MulticlassPrecisionResult:
    return MulticlassPrecisionResult(
        macro_precision=0.7,
        micro_precision=0.72,
        weighted_precision=0.71,
        per_class_precision=np.array([0.8, 0.6, 0.7]),
        support=np.array([100, 50, 150]),
        num_predictions=300,
    )


# --- instantiation ---


def test_multiclass_precision_result_instantiation(
    result: MulticlassPrecisionResult,
) -> None:
    assert result.macro_precision == 0.7
    assert result.micro_precision == 0.72
    assert result.weighted_precision == 0.71
    assert objects_are_equal(result.per_class_precision, np.array([0.8, 0.6, 0.7]))
    assert objects_are_equal(result.support, np.array([100, 50, 150]))
    assert result.num_predictions == 300


def test_multiclass_precision_result_zero_predictions() -> None:
    m = MulticlassPrecisionResult(
        macro_precision=0.0,
        micro_precision=0.0,
        weighted_precision=0.0,
        per_class_precision=np.array([0.0, 0.0]),
        support=np.array([0, 0]),
        num_predictions=0,
    )
    assert m.num_predictions == 0


def test_multiclass_precision_result_nan_precisions() -> None:
    m = MulticlassPrecisionResult(
        macro_precision=float("nan"),
        micro_precision=float("nan"),
        weighted_precision=float("nan"),
        per_class_precision=np.array([float("nan"), float("nan")]),
        support=np.array([0, 0]),
        num_predictions=10,
    )
    assert math.isnan(m.macro_precision)
    assert math.isnan(m.micro_precision)
    assert math.isnan(m.weighted_precision)


def test_multiclass_precision_result_frozen(result: MulticlassPrecisionResult) -> None:
    with pytest.raises(FrozenInstanceError, match=r"cannot assign to field 'macro_precision'"):
        result.macro_precision = 0.5


def test_multiclass_precision_result_single_class() -> None:
    m = MulticlassPrecisionResult(
        macro_precision=1.0,
        micro_precision=1.0,
        weighted_precision=1.0,
        per_class_precision=np.array([1.0]),
        support=np.array([10]),
        num_predictions=10,
    )
    assert m.num_predictions == 10


# --- validation ---


def test_multiclass_precision_result_negative_num_predictions_raises() -> None:
    with pytest.raises(ValueError, match=r"num_predictions must be >= 0"):
        MulticlassPrecisionResult(
            macro_precision=0.7,
            micro_precision=0.72,
            weighted_precision=0.71,
            per_class_precision=np.array([0.8, 0.6, 0.7]),
            support=np.array([100, 50, 150]),
            num_predictions=-1,
        )


@pytest.mark.parametrize(
    ("macro", "micro", "weighted", "match"),
    [
        pytest.param(-0.1, 0.72, 0.71, "macro_precision must be >= 0", id="negative-macro"),
        pytest.param(0.7, -0.1, 0.71, "micro_precision must be >= 0", id="negative-micro"),
        pytest.param(0.7, 0.72, -0.1, "weighted_precision must be >= 0", id="negative-weighted"),
    ],
)
def test_multiclass_precision_result_negative_precision_raises(
    macro: float, micro: float, weighted: float, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        MulticlassPrecisionResult(
            macro_precision=macro,
            micro_precision=micro,
            weighted_precision=weighted,
            per_class_precision=np.array([0.8, 0.6, 0.7]),
            support=np.array([100, 50, 150]),
            num_predictions=300,
        )


def test_multiclass_precision_result_mismatched_shapes_raises() -> None:
    with pytest.raises(
        ValueError, match=r"per_class_precision and support must have the same shape"
    ):
        MulticlassPrecisionResult(
            macro_precision=0.7,
            micro_precision=0.72,
            weighted_precision=0.71,
            per_class_precision=np.array([0.8, 0.6, 0.7]),
            support=np.array([100, 50]),
            num_predictions=300,
        )


def test_multiclass_precision_result_nan_does_not_raise() -> None:
    MulticlassPrecisionResult(
        macro_precision=float("nan"),
        micro_precision=float("nan"),
        weighted_precision=float("nan"),
        per_class_precision=np.array([float("nan"), float("nan")]),
        support=np.array([0, 0]),
        num_predictions=10,
    )


def test_multiclass_precision_result_zero_precisions_do_not_raise() -> None:
    MulticlassPrecisionResult(
        macro_precision=0.0,
        micro_precision=0.0,
        weighted_precision=0.0,
        per_class_precision=np.array([0.0, 0.0]),
        support=np.array([10, 10]),
        num_predictions=20,
    )


# --- equal ---


def test_multiclass_precision_result_equal_true(
    result: MulticlassPrecisionResult,
) -> None:
    assert result.equal(
        MulticlassPrecisionResult(
            macro_precision=0.7,
            micro_precision=0.72,
            weighted_precision=0.71,
            per_class_precision=np.array([0.8, 0.6, 0.7]),
            support=np.array([100, 50, 150]),
            num_predictions=300,
        )
    )


@pytest.mark.parametrize(
    ("macro", "micro", "weighted", "per_class", "sup", "n"),
    [
        pytest.param(0.9, 0.72, 0.71, [0.8, 0.6, 0.7], [100, 50, 150], 300, id="diff-macro"),
        pytest.param(0.7, 0.9, 0.71, [0.8, 0.6, 0.7], [100, 50, 150], 300, id="diff-micro"),
        pytest.param(0.7, 0.72, 0.9, [0.8, 0.6, 0.7], [100, 50, 150], 300, id="diff-weighted"),
        pytest.param(0.7, 0.72, 0.71, [0.9, 0.6, 0.7], [100, 50, 150], 300, id="diff-per-class"),
        pytest.param(0.7, 0.72, 0.71, [0.8, 0.6, 0.7], [200, 50, 150], 300, id="diff-support"),
        pytest.param(0.7, 0.72, 0.71, [0.8, 0.6, 0.7], [100, 50, 150], 200, id="diff-n"),
    ],
)
def test_multiclass_precision_result_equal_false(
    result: MulticlassPrecisionResult,
    macro: float,
    micro: float,
    weighted: float,
    per_class: list,
    sup: list,
    n: int,
) -> None:
    assert not result.equal(
        MulticlassPrecisionResult(
            macro_precision=macro,
            micro_precision=micro,
            weighted_precision=weighted,
            per_class_precision=np.array(per_class),
            support=np.array(sup),
            num_predictions=n,
        )
    )


def test_multiclass_precision_result_equal_wrong_type(
    result: MulticlassPrecisionResult,
) -> None:
    assert not result.equal("not a result")


def test_multiclass_precision_result_equal_nan_false_by_default() -> None:
    assert not MulticlassPrecisionResult(
        macro_precision=float("nan"),
        micro_precision=float("nan"),
        weighted_precision=float("nan"),
        per_class_precision=np.array([float("nan")]),
        support=np.array([0]),
        num_predictions=10,
    ).equal(
        MulticlassPrecisionResult(
            macro_precision=float("nan"),
            micro_precision=float("nan"),
            weighted_precision=float("nan"),
            per_class_precision=np.array([float("nan")]),
            support=np.array([0]),
            num_predictions=10,
        )
    )


def test_multiclass_precision_result_equal_nan_with_equal_nan() -> None:
    assert MulticlassPrecisionResult(
        macro_precision=float("nan"),
        micro_precision=float("nan"),
        weighted_precision=float("nan"),
        per_class_precision=np.array([float("nan")]),
        support=np.array([0]),
        num_predictions=10,
    ).equal(
        MulticlassPrecisionResult(
            macro_precision=float("nan"),
            micro_precision=float("nan"),
            weighted_precision=float("nan"),
            per_class_precision=np.array([float("nan")]),
            support=np.array([0]),
            num_predictions=10,
        ),
        equal_nan=True,
    )


# --- allclose ---


def test_multiclass_precision_result_allclose_true(
    result: MulticlassPrecisionResult,
) -> None:
    assert result.allclose(
        MulticlassPrecisionResult(
            macro_precision=0.7,
            micro_precision=0.72,
            weighted_precision=0.71,
            per_class_precision=np.array([0.8, 0.6, 0.7]),
            support=np.array([100, 50, 150]),
            num_predictions=300,
        )
    )


def test_multiclass_precision_result_allclose_within_tolerance(
    result: MulticlassPrecisionResult,
) -> None:
    assert result.allclose(
        MulticlassPrecisionResult(
            macro_precision=0.7 + 1e-7,
            micro_precision=0.72 + 1e-7,
            weighted_precision=0.71 + 1e-7,
            per_class_precision=np.array([0.8 + 1e-7, 0.6 + 1e-7, 0.7 + 1e-7]),
            support=np.array([100, 50, 150]),
            num_predictions=300,
        ),
        rtol=1e-5,
        atol=1e-6,
    )


def test_multiclass_precision_result_allclose_outside_tolerance(
    result: MulticlassPrecisionResult,
) -> None:
    assert not result.allclose(
        MulticlassPrecisionResult(
            macro_precision=0.9,
            micro_precision=0.72,
            weighted_precision=0.71,
            per_class_precision=np.array([0.8, 0.6, 0.7]),
            support=np.array([100, 50, 150]),
            num_predictions=300,
        ),
        rtol=1e-5,
        atol=1e-8,
    )


def test_multiclass_precision_result_allclose_wrong_type(
    result: MulticlassPrecisionResult,
) -> None:
    assert not result.allclose("not a result")


def test_multiclass_precision_result_allclose_nan_false_by_default() -> None:
    m = MulticlassPrecisionResult(
        macro_precision=float("nan"),
        micro_precision=float("nan"),
        weighted_precision=float("nan"),
        per_class_precision=np.array([float("nan")]),
        support=np.array([0]),
        num_predictions=10,
    )
    assert not m.allclose(m)


def test_multiclass_precision_result_allclose_nan_with_equal_nan() -> None:
    m = MulticlassPrecisionResult(
        macro_precision=float("nan"),
        micro_precision=float("nan"),
        weighted_precision=float("nan"),
        per_class_precision=np.array([float("nan")]),
        support=np.array([0]),
        num_predictions=10,
    )
    assert m.allclose(m, equal_nan=True)


# --- to_dict ---


@pytest.mark.parametrize(
    ("prefix", "suffix"),
    [
        pytest.param("", "", id="no-prefix-suffix"),
        pytest.param("val_", "", id="prefix-only"),
        pytest.param("", "_epoch1", id="suffix-only"),
        pytest.param("val_", "_epoch1", id="prefix-and-suffix"),
    ],
)
def test_multiclass_precision_result_to_dict(
    result: MulticlassPrecisionResult, prefix: str, suffix: str
) -> None:
    assert objects_are_allclose(
        result.to_dict(prefix=prefix, suffix=suffix),
        {
            f"{prefix}macro_precision{suffix}": 0.7,
            f"{prefix}micro_precision{suffix}": 0.72,
            f"{prefix}weighted_precision{suffix}": 0.71,
            f"{prefix}per_class_precision{suffix}": [0.8, 0.6, 0.7],
            f"{prefix}support{suffix}": [100, 50, 150],
            f"{prefix}num_predictions{suffix}": 300,
        },
    )


def test_multiclass_precision_result_to_dict_per_class_is_list(
    result: MulticlassPrecisionResult,
) -> None:
    d = result.to_dict()
    assert isinstance(d["per_class_precision"], list)
    assert isinstance(d["support"], list)


def test_multiclass_precision_result_to_dict_nan() -> None:
    m = MulticlassPrecisionResult(
        macro_precision=float("nan"),
        micro_precision=float("nan"),
        weighted_precision=float("nan"),
        per_class_precision=np.array([float("nan"), float("nan")]),
        support=np.array([0, 0]),
        num_predictions=10,
    )
    assert objects_are_allclose(
        m.to_dict(),
        {
            "macro_precision": float("nan"),
            "micro_precision": float("nan"),
            "weighted_precision": float("nan"),
            "per_class_precision": [float("nan"), float("nan")],
            "support": [0, 0],
            "num_predictions": 10,
        },
        equal_nan=True,
    )


# --- to_display ---


def test_multiclass_precision_result_to_display_zero_predictions() -> None:
    m = MulticlassPrecisionResult(
        macro_precision=0.0,
        micro_precision=0.0,
        weighted_precision=0.0,
        per_class_precision=np.array([0.0, 0.0]),
        support=np.array([0, 0]),
        num_predictions=0,
    )
    assert m.to_display() == "MulticlassPrecisionResult: no predictions"


def test_multiclass_precision_result_to_display_returns_str(
    result: MulticlassPrecisionResult,
) -> None:
    assert isinstance(result.to_display(), str)


def test_multiclass_precision_result_to_display_full(
    result: MulticlassPrecisionResult,
) -> None:
    assert result.to_display() == (
        "Precision (n=300)\n"
        "----------------------------------------------------\n"
        "Macro       [██████████████░░░░░░]  0.7000\n"
        "Micro       [██████████████░░░░░░]  0.7200\n"
        "Weighted    [██████████████░░░░░░]  0.7100\n"
        "\nPer class:\n"
        "  class 0   [████████████████░░░░]  0.8000  (n=100)\n"
        "  class 1   [████████████░░░░░░░░]  0.6000  (n=50)\n"
        "  class 2   [██████████████░░░░░░]  0.7000  (n=150)"
    )


def test_multiclass_precision_result_to_display_nan() -> None:
    m = MulticlassPrecisionResult(
        macro_precision=float("nan"),
        micro_precision=float("nan"),
        weighted_precision=float("nan"),
        per_class_precision=np.array([float("nan"), float("nan")]),
        support=np.array([0, 0]),
        num_predictions=10,
    )
    assert m.to_display() == (
        "Precision (n=10)\n"
        "----------------------------------------------------\n"
        "Macro       [????????????????????]  nan\n"
        "Micro       [????????????????????]  nan\n"
        "Weighted    [????????????????????]  nan\n"
        "\nPer class:\n"
        "  class 0   [????????????????????]  nan  (n=0)\n"
        "  class 1   [????????????????????]  nan  (n=0)"
    )


def test_multiclass_precision_result_to_display_perfect() -> None:
    m = MulticlassPrecisionResult(
        macro_precision=1.0,
        micro_precision=1.0,
        weighted_precision=1.0,
        per_class_precision=np.array([1.0, 1.0]),
        support=np.array([50, 50]),
        num_predictions=100,
    )
    assert m.to_display() == (
        "Precision (n=100)\n"
        "----------------------------------------------------\n"
        "Macro       [████████████████████]  1.0000\n"
        "Micro       [████████████████████]  1.0000\n"
        "Weighted    [████████████████████]  1.0000\n"
        "\nPer class:\n"
        "  class 0   [████████████████████]  1.0000  (n=50)\n"
        "  class 1   [████████████████████]  1.0000  (n=50)"
    )


# --- repr / str ---


def test_multiclass_precision_result_repr(result: MulticlassPrecisionResult) -> None:
    assert repr(result) == (
        "MulticlassPrecisionResult("
        "macro_precision=0.7, "
        "micro_precision=0.72, "
        "weighted_precision=0.71, "
        "per_class_precision=array([0.8, 0.6, 0.7]), "
        "support=array([100,  50, 150]), "
        "num_predictions=300)"
    )


def test_multiclass_precision_result_str(result: MulticlassPrecisionResult) -> None:
    assert str(result) == (
        "MulticlassPrecisionResult("
        "macro_precision=0.7, "
        "micro_precision=0.72, "
        "weighted_precision=0.71, "
        "per_class_precision=array([0.8, 0.6, 0.7]), "
        "support=array([100,  50, 150]), "
        "num_predictions=300)"
    )
