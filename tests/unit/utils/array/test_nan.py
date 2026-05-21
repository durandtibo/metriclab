from __future__ import annotations

import numpy as np
import pytest

from metriclab.utils.array import (
    NAN_POLICIES,
    NanPolicy,
    check_nan_policy,
    contains_nan,
    validate_nan_policy,
)

#########################################
#     Tests for validate_nan_policy     #
#########################################


@pytest.mark.parametrize("nan_policy", NAN_POLICIES)
def test_validate_nan_policy_valid(nan_policy: str) -> None:
    validate_nan_policy(nan_policy)  # should not raise


@pytest.mark.parametrize(
    "nan_policy",
    [
        pytest.param("invalid", id="invalid-string"),
        pytest.param("", id="empty-string"),
        pytest.param("OMIT", id="uppercase"),
        pytest.param("Propagate", id="mixed-case"),
        pytest.param("drop", id="similar-but-wrong"),
    ],
)
def test_validate_nan_policy_invalid_raises(nan_policy: str) -> None:
    with pytest.raises(ValueError, match="Incorrect 'nan_policy'"):
        validate_nan_policy(nan_policy)


def test_validate_nan_policy_error_message_contains_valid_values() -> None:
    with pytest.raises(ValueError, match="'omit', 'propagate', 'raise'"):
        validate_nan_policy("invalid")


##################################
#     Tests for contains_nan     #
##################################


# --- float arrays ---


@pytest.mark.parametrize(
    ("arr", "expected"),
    [
        pytest.param(np.array([1.0, 2.0, 3.0]), False, id="float-no-nan"),
        pytest.param(np.array([1.0, float("nan"), 3.0]), True, id="float-nan-middle"),
        pytest.param(np.array([float("nan"), 2.0, 3.0]), True, id="float-nan-start"),
        pytest.param(np.array([1.0, 2.0, float("nan")]), True, id="float-nan-end"),
        pytest.param(np.array([float("nan"), float("nan")]), True, id="float-all-nan"),
        pytest.param(np.array([float("nan")]), True, id="float-single-nan"),
        pytest.param(np.array([1.0]), False, id="float-single-no-nan"),
        pytest.param(np.array([], dtype=float), False, id="float-empty"),
    ],
)
def test_contains_nan_float_array(arr: np.ndarray, expected: bool) -> None:
    assert contains_nan(arr) == expected


# --- int arrays ---


@pytest.mark.parametrize(
    ("arr", "expected"),
    [
        pytest.param(np.array([1, 2, 3]), False, id="int-no-nan"),
        pytest.param(np.array([1, 2, 3], dtype=np.int8), False, id="int8-no-nan"),
        pytest.param(np.array([1, 2, 3], dtype=np.int16), False, id="int16-no-nan"),
        pytest.param(np.array([1, 2, 3], dtype=np.int32), False, id="int32-no-nan"),
        pytest.param(np.array([1, 2, 3], dtype=np.int64), False, id="int64-no-nan"),
        pytest.param(np.array([], dtype=int), False, id="int-empty"),
    ],
)
def test_contains_nan_int_array(arr: np.ndarray, expected: bool) -> None:
    assert contains_nan(arr) == expected


# --- object arrays ---


@pytest.mark.parametrize(
    ("arr", "expected"),
    [
        pytest.param(np.array([1, 2, 3], dtype=object), False, id="object-no-nan"),
        pytest.param(np.array([1, float("nan"), 3], dtype=object), True, id="object-nan-middle"),
        pytest.param(np.array([float("nan"), 2, 3], dtype=object), True, id="object-nan-start"),
        pytest.param(np.array([1, 2, float("nan")], dtype=object), True, id="object-nan-end"),
        pytest.param(
            np.array([float("nan"), float("nan")], dtype=object), True, id="object-all-nan"
        ),
        pytest.param(np.array([None, 2, 3], dtype=object), False, id="object-none-not-nan"),
        pytest.param(np.array([None, float("nan")], dtype=object), True, id="object-none-and-nan"),
        pytest.param(np.array([], dtype=object), False, id="object-empty"),
    ],
)
def test_contains_nan_object_array(arr: np.ndarray, expected: bool) -> None:
    assert contains_nan(arr) == expected


# --- non-numeric dtypes (TypeError fallback) ---


@pytest.mark.parametrize(
    "arr",
    [
        pytest.param(np.array(["a", "b", "c"]), id="str"),
        pytest.param(np.array(["2021-01-01", "2021-01-02"], dtype="datetime64"), id="datetime64"),
        pytest.param(np.array([True, False, True]), id="bool"),
    ],
)
def test_contains_nan_non_numeric_array(arr: np.ndarray) -> None:
    assert not contains_nan(arr)


# --- nan not confused with other special values ---


def test_contains_nan_inf_is_not_nan() -> None:
    assert not contains_nan(np.array([float("inf"), float("-inf")]))


def test_contains_nan_none_is_not_nan() -> None:
    assert not contains_nan(np.array([None, None], dtype=object))


######################################
#     Tests for check_nan_policy     #
######################################


# --- no NaN — all policies return False ---


@pytest.mark.parametrize("nan_policy", NAN_POLICIES)
def test_check_nan_policy_no_nan(nan_policy: NanPolicy) -> None:
    assert not check_nan_policy(np.array([1, 2, 3]), nan_policy=nan_policy)


# --- missing_policy='propagate' ---


def test_check_nan_policy_propagate_returns_true() -> None:
    assert check_nan_policy(np.array([1.0, float("nan"), 3.0]), nan_policy="propagate")


def test_check_nan_policy_propagate_does_not_raise() -> None:
    result = check_nan_policy(np.array([1.0, float("nan"), 3.0]), nan_policy="propagate")
    assert result is True


# --- nan_policy='omit' ---


def test_check_nan_policy_omit_returns_true() -> None:
    assert check_nan_policy(np.array([1.0, float("nan"), 3.0]), nan_policy="omit")


def test_check_nan_policy_omit_does_not_raise() -> None:
    result = check_nan_policy(np.array([1.0, float("nan"), 3.0]), nan_policy="omit")
    assert result is True


# --- nan_policy='raise' ---


def test_check_nan_policy_raise_no_nan_does_not_raise() -> None:
    assert not check_nan_policy(np.array([1, 2, 3]), nan_policy="raise")


def test_check_nan_policy_raise_with_nan_raises() -> None:
    with pytest.raises(ValueError, match="input contains at least one NaN value"):
        check_nan_policy(np.array([1.0, float("nan"), 3.0]), nan_policy="raise")


def test_check_nan_policy_raise_custom_name() -> None:
    with pytest.raises(ValueError, match="my_array contains at least one NaN value"):
        check_nan_policy(
            np.array([1.0, float("nan"), 3.0]),
            nan_policy="raise",
            name="my_array",
        )


# --- invalid nan_policy ---


def test_check_nan_policy_invalid_raises() -> None:
    with pytest.raises(ValueError, match="Incorrect 'nan_policy'"):
        check_nan_policy(np.array([1, 2, 3]), nan_policy="invalid")


# --- different dtypes ---


@pytest.mark.parametrize(
    ("arr", "expected"),
    [
        pytest.param(np.array([1.0, float("nan"), 3.0], dtype=np.float32), True, id="float32"),
        pytest.param(np.array([1.0, float("nan"), 3.0], dtype=np.float64), True, id="float64"),
        pytest.param(np.array([1, float("nan"), 3], dtype=object), True, id="object"),
        pytest.param(np.array([1, 2, 3], dtype=np.int32), False, id="int32"),
        pytest.param(np.array(["a", "b", "c"]), False, id="str"),
        pytest.param(
            np.array(["2021-01-01", "2021-01-02"], dtype="datetime64"), False, id="datetime64"
        ),
    ],
)
def test_check_nan_policy_dtypes(arr: np.ndarray, expected: bool) -> None:
    assert check_nan_policy(arr) == expected


# --- edge cases ---


def test_check_nan_policy_empty_array() -> None:
    assert not check_nan_policy(np.array([], dtype=float))


def test_check_nan_policy_all_nan() -> None:
    assert check_nan_policy(np.array([float("nan"), float("nan")]))


def test_check_nan_policy_single_nan() -> None:
    assert check_nan_policy(np.array([float("nan")]))


def test_check_nan_policy_single_no_nan() -> None:
    assert not check_nan_policy(np.array([1.0]))
