from typing import Any

import numpy as np
import pytest
from coola.equality import objects_are_equal

from metriclab.utils.array import preprocess_1d
from metriclab.utils.missing import MissingPolicy

MISSING_POLICIES = ["omit", "propagate", "raise"]

##################################
#   Tests for preprocess_1d      #
##################################


# --- empty input ---


def test_preprocess_1d_empty_input() -> None:
    assert preprocess_1d([]) == []


@pytest.mark.parametrize("missing_policy", MISSING_POLICIES)
def test_preprocess_1d_empty_input_all_policies(missing_policy: MissingPolicy) -> None:
    assert preprocess_1d([], missing_policy=missing_policy) == []


# --- missing_policy='propagate' (default) ---


def test_preprocess_1d_propagate_no_missing_values_set() -> None:
    arrays = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])]
    result = preprocess_1d(arrays)
    assert objects_are_equal(result, arrays)


def test_preprocess_1d_propagate_returns_list() -> None:
    arrays = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])]
    assert isinstance(preprocess_1d(arrays), list)


def test_preprocess_1d_propagate_with_nan_unchanged() -> None:
    arrays = [np.array([1.0, float("nan"), 3.0]), np.array([4.0, 5.0, 6.0])]
    result = preprocess_1d(arrays, missing_policy="propagate", missing_values=float("nan"))
    assert objects_are_equal(result, arrays, equal_nan=True)


def test_preprocess_1d_propagate_with_none_unchanged() -> None:
    arrays = [np.array([1, None, 3], dtype=object), np.array([4, 5, 6], dtype=object)]
    result = preprocess_1d(arrays, missing_policy="propagate", missing_values=None)
    assert objects_are_equal(result, arrays)


def test_preprocess_1d_propagate_missing_values_not_set_unchanged() -> None:
    arrays = [np.array([1.0, float("nan"), 3.0])]
    result = preprocess_1d(arrays, missing_policy="propagate")
    assert objects_are_equal(result, arrays, equal_nan=True)


# --- missing_policy='omit' ---


@pytest.mark.parametrize(
    ("arrays", "missing_values", "expected"),
    [
        pytest.param(
            [np.array([1.0, float("nan"), 3.0]), np.array([4.0, 5.0, 6.0])],
            float("nan"),
            [np.array([1.0, 3.0]), np.array([4.0, 6.0])],
            id="float-nan-in-first",
        ),
        pytest.param(
            [np.array([1.0, 2.0, 3.0]), np.array([4.0, float("nan"), 6.0])],
            float("nan"),
            [np.array([1.0, 3.0]), np.array([4.0, 6.0])],
            id="float-nan-in-second",
        ),
        pytest.param(
            [np.array([1.0, float("nan"), 3.0]), np.array([float("nan"), 5.0, 6.0])],
            float("nan"),
            [np.array([3.0]), np.array([6.0])],
            id="float-nan-in-both",
        ),
        pytest.param(
            [np.array([float("nan"), float("nan")]), np.array([float("nan"), float("nan")])],
            float("nan"),
            [np.array([]), np.array([])],
            id="float-all-nan",
        ),
        pytest.param(
            [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])],
            float("nan"),
            [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])],
            id="float-nan-absent",
        ),
        pytest.param(
            [np.array([1, None, 3], dtype=object), np.array([4, 5, 6], dtype=object)],
            None,
            [np.array([1, 3], dtype=object), np.array([4, 6], dtype=object)],
            id="object-none-in-first",
        ),
        pytest.param(
            [np.array([1, 2, 3], dtype=object), np.array([4, None, 6], dtype=object)],
            None,
            [np.array([1, 3], dtype=object), np.array([4, 6], dtype=object)],
            id="object-none-in-second",
        ),
        pytest.param(
            [np.array([None, 2, 3], dtype=object), np.array([4, None, 6], dtype=object)],
            None,
            [np.array([3], dtype=object), np.array([6], dtype=object)],
            id="object-none-in-both",
        ),
        pytest.param(
            [np.array([1, 2, 3], dtype=object), np.array([4, 5, 6], dtype=object)],
            None,
            [np.array([1, 2, 3], dtype=object), np.array([4, 5, 6], dtype=object)],
            id="object-none-absent",
        ),
        pytest.param(
            [np.array([1.0, float("inf"), 3.0]), np.array([4.0, 5.0, 6.0])],
            float("inf"),
            [np.array([1.0, 3.0]), np.array([4.0, 6.0])],
            id="float-pos-inf-in-first",
        ),
        pytest.param(
            [np.array([1.0, float("-inf"), 3.0]), np.array([4.0, 5.0, 6.0])],
            float("-inf"),
            [np.array([1.0, 3.0]), np.array([4.0, 6.0])],
            id="float-neg-inf-in-first",
        ),
        pytest.param(
            [np.array([1, 2, 3]), np.array([4, 5, 6])],
            99,
            [np.array([1, 2, 3]), np.array([4, 5, 6])],
            id="int-absent",
        ),
        pytest.param(
            [np.array([1, 99, 3]), np.array([4, 5, 6])],
            99,
            [np.array([1, 3]), np.array([4, 6])],
            id="int-present-in-first",
        ),
        pytest.param(
            [np.array(["cat", "dog", "cat"]), np.array(["bird", "dog", "fish"])],
            "dog",
            [np.array(["cat", "cat"]), np.array(["bird", "fish"])],
            id="str-present",
        ),
        pytest.param(
            [np.array(["cat", "bird", "cat"]), np.array(["bird", "fish", "fox"])],
            "dog",
            [np.array(["cat", "bird", "cat"]), np.array(["bird", "fish", "fox"])],
            id="str-absent",
        ),
    ],
)
def test_preprocess_1d_omit(
    arrays: list[np.ndarray],
    missing_values: Any,
    expected: list[np.ndarray],
) -> None:
    assert objects_are_equal(
        preprocess_1d(arrays, missing_policy="omit", missing_values=missing_values),
        expected,
    )


def test_preprocess_1d_omit_missing_values_not_set() -> None:
    # missing_values=NOT_SET — nothing should be removed
    arrays = [np.array([1.0, float("nan"), 3.0])]
    result = preprocess_1d(arrays, missing_policy="omit")
    assert objects_are_equal(result, arrays, equal_nan=True)


def test_preprocess_1d_omit_returns_list() -> None:
    arrays = [np.array([1.0, float("nan"), 3.0])]
    assert isinstance(
        preprocess_1d(arrays, missing_policy="omit", missing_values=float("nan")), list
    )


def test_preprocess_1d_omit_output_length_matches_input() -> None:
    arrays = [np.array([1.0, float("nan"), 3.0]), np.array([4.0, 5.0, 6.0])]
    result = preprocess_1d(arrays, missing_policy="omit", missing_values=float("nan"))
    assert len(result) == len(arrays)


def test_preprocess_1d_omit_three_arrays() -> None:
    assert objects_are_equal(
        preprocess_1d(
            [
                np.array([1.0, float("nan"), 3.0]),
                np.array([4.0, 5.0, float("nan")]),
                np.array([7.0, 8.0, 9.0]),
            ],
            missing_policy="omit",
            missing_values=float("nan"),
        ),
        [np.array([1.0]), np.array([4.0]), np.array([7.0])],
    )


# --- missing_policy='raise' ---


def test_preprocess_1d_raise_no_missing_returns_arrays() -> None:
    arrays = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])]
    result = preprocess_1d(arrays, missing_policy="raise", missing_values=float("nan"))
    assert objects_are_equal(result, arrays)


def test_preprocess_1d_raise_with_nan_raises() -> None:
    with pytest.raises(ValueError, match=r"arrays contain at least one missing value"):
        preprocess_1d(
            [np.array([1.0, float("nan"), 3.0])],
            missing_policy="raise",
            missing_values=float("nan"),
        )


def test_preprocess_1d_raise_with_none_raises() -> None:
    with pytest.raises(ValueError, match=r"arrays contain at least one missing value"):
        preprocess_1d(
            [np.array([1, None, 3], dtype=object)],
            missing_policy="raise",
            missing_values=None,
        )


def test_preprocess_1d_raise_with_inf_raises() -> None:
    with pytest.raises(ValueError, match=r"arrays contain at least one missing value"):
        preprocess_1d(
            [np.array([1.0, float("inf"), 3.0])],
            missing_policy="raise",
            missing_values=float("inf"),
        )


def test_preprocess_1d_raise_with_int_raises() -> None:
    with pytest.raises(ValueError, match=r"arrays contain at least one missing value"):
        preprocess_1d(
            [np.array([1, 99, 3])],
            missing_policy="raise",
            missing_values=99,
        )


def test_preprocess_1d_raise_missing_values_not_set_does_not_raise() -> None:
    arrays = [np.array([1.0, float("nan"), 3.0])]
    result = preprocess_1d(arrays, missing_policy="raise")
    assert objects_are_equal(result, arrays, equal_nan=True)


def test_preprocess_1d_raise_error_message_contains_value() -> None:
    with pytest.raises(ValueError, match=r"arrays contain at least one missing value \(nan\)"):
        preprocess_1d(
            [np.array([1.0, float("nan"), 3.0])],
            missing_policy="raise",
            missing_values=float("nan"),
        )


# --- invalid missing_policy ---


def test_preprocess_1d_invalid_missing_policy_raises() -> None:
    with pytest.raises(
        ValueError,
        match=r"Incorrect 'missing_policy': invalid. The valid values are: 'omit', 'propagate', 'raise'",
    ):
        preprocess_1d([np.array([1, 2, 3])], missing_policy="invalid")


# --- validation ---


def test_preprocess_1d_different_shapes_raises() -> None:
    with pytest.raises(ValueError, match=r"arrays have different shapes: \[\(3,\), \(2,\)\]"):
        preprocess_1d([np.array([1, 2, 3]), np.array([4, 5])])


def test_preprocess_1d_2d_array_raises() -> None:
    with pytest.raises(ValueError, match=r"expected 1D array, got shape \(2, 2\)"):
        preprocess_1d([np.array([[1, 2], [3, 4]])])


# --- single array ---


@pytest.mark.parametrize("missing_policy", MISSING_POLICIES)
def test_preprocess_1d_single_array_no_missing(missing_policy: MissingPolicy) -> None:
    arrays = [np.array([1.0, 2.0, 3.0])]
    result = preprocess_1d(arrays, missing_policy=missing_policy)
    assert objects_are_equal(result, arrays)


def test_preprocess_1d_single_array_omit_nan() -> None:
    assert objects_are_equal(
        preprocess_1d(
            [np.array([1.0, float("nan"), 3.0])],
            missing_policy="omit",
            missing_values=float("nan"),
        ),
        [np.array([1.0, 3.0])],
    )


# --- edge cases ---


def test_preprocess_1d_empty_arrays_omit() -> None:
    assert objects_are_equal(
        preprocess_1d(
            [np.array([], dtype=float), np.array([], dtype=float)],
            missing_policy="omit",
            missing_values=float("nan"),
        ),
        [np.array([], dtype=float), np.array([], dtype=float)],
    )


def test_preprocess_1d_single_element_omit_match() -> None:
    assert objects_are_equal(
        preprocess_1d(
            [np.array([float("nan")]), np.array([1.0])],
            missing_policy="omit",
            missing_values=float("nan"),
        ),
        [np.array([]), np.array([])],
    )


def test_preprocess_1d_single_element_omit_no_match() -> None:
    assert objects_are_equal(
        preprocess_1d(
            [np.array([1.0]), np.array([2.0])],
            missing_policy="omit",
            missing_values=float("nan"),
        ),
        [np.array([1.0]), np.array([2.0])],
    )


@pytest.mark.parametrize("missing_policy", MISSING_POLICIES)
def test_preprocess_1d_output_is_list(missing_policy: MissingPolicy) -> None:
    arrays = [np.array([1.0, 2.0, 3.0])]
    assert isinstance(preprocess_1d(arrays, missing_policy=missing_policy), list)
