from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
import pytest

from metriclab.utils.undefined import resolve_fill_value, validate_undefined_policy

##################################################
#      Tests for validate_undefined_policy       #
##################################################


# --- valid values: no exception raised ---


@pytest.mark.parametrize(
    "undefined_policy",
    [
        pytest.param(0.0, id="float-zero"),
        pytest.param(1.0, id="float-one"),
        pytest.param(0.5, id="float-half"),
        pytest.param(-1.0, id="float-negative"),
        pytest.param(float("nan"), id="float-nan"),
        pytest.param(float("inf"), id="float-inf"),
        pytest.param("nan", id="string-nan"),
        pytest.param("raise", id="string-raise"),
        pytest.param("warn", id="string-warn"),
    ],
)
def test_validate_undefined_policy_valid(undefined_policy: Any) -> None:
    validate_undefined_policy(undefined_policy)


# --- invalid values: ValueError raised ---


@pytest.mark.parametrize(
    "undefined_policy",
    [
        pytest.param("omit", id="string-omit"),
        pytest.param("propagate", id="string-propagate"),
        pytest.param("zero", id="string-zero"),
        pytest.param("", id="string-empty"),
        pytest.param(0, id="int-zero"),
        pytest.param(1, id="int-one"),
        pytest.param(None, id="none"),
        pytest.param([0.0], id="list"),
        pytest.param({"nan"}, id="set"),
        pytest.param(("warn",), id="tuple"),
    ],
)
def test_validate_undefined_policy_invalid(undefined_policy: Any) -> None:
    with pytest.raises(ValueError, match=r"Invalid 'undefined_policy'"):
        validate_undefined_policy(undefined_policy)


# --- error message content ---


def test_validate_undefined_policy_error_includes_value() -> None:
    with pytest.raises(ValueError, match=r"'bad_value'"):
        validate_undefined_policy("bad_value")


def test_validate_undefined_policy_error_mentions_float() -> None:
    with pytest.raises(ValueError, match=r"float"):
        validate_undefined_policy(1)


def test_validate_undefined_policy_error_mentions_valid_strings() -> None:
    with pytest.raises(ValueError, match=r"'nan'.*'raise'.*'warn'"):
        validate_undefined_policy("bad_value")


##################################################
#        Tests for resolve_fill_value            #
##################################################


# --- no undefined entries: returns 0.0 without side effects ---


@pytest.mark.parametrize(
    "undefined_policy",
    [
        pytest.param(0.0, id="float-zero"),
        pytest.param(1.0, id="float-one"),
        pytest.param(float("nan"), id="float-nan"),
        pytest.param("nan", id="string-nan"),
        pytest.param("raise", id="string-raise"),
        pytest.param("warn", id="string-warn"),
    ],
)
def test_resolve_fill_value_no_undefined_returns_zero(
    undefined_policy: Any,
) -> None:
    # When no entries are undefined the fill value is irrelevant;
    # 0.0 is returned immediately regardless of undefined_policy.
    fill = resolve_fill_value(
        undefined_mask=np.array([False, False, False]),
        undefined_policy=undefined_policy,
    )
    assert fill == 0.0


# --- undefined_policy="nan" ---


def test_resolve_fill_value_nan_string_returns_nan() -> None:
    fill = resolve_fill_value(
        undefined_mask=np.array([True, False]),
        undefined_policy="nan",
    )
    assert math.isnan(fill)


def test_resolve_fill_value_nan_string_no_warning() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        resolve_fill_value(
            undefined_mask=np.array([True, False]),
            undefined_policy="nan",
        )


# --- undefined_policy="raise" ---


def test_resolve_fill_value_raise_raises_value_error() -> None:
    with pytest.raises(
        ValueError, match=r"The metric is undefined for 1 element\(s\) at indices \[0\]"
    ):
        resolve_fill_value(
            undefined_mask=np.array([True, False]),
            undefined_policy="raise",
        )


def test_resolve_fill_value_raise_no_warning() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        with pytest.raises(
            ValueError, match=r"The metric is undefined for 1 element\(s\) at indices \[0\]"
        ):
            resolve_fill_value(
                undefined_mask=np.array([True, False]),
                undefined_policy="raise",
            )


# --- undefined_policy="warn" ---


def test_resolve_fill_value_warn_emits_user_warning() -> None:
    with pytest.warns(
        UserWarning, match=r"The metric is undefined for 1 element\(s\) at indices \[0\]"
    ):
        fill = resolve_fill_value(
            undefined_mask=np.array([True, False]),
            undefined_policy="warn",
        )
    assert fill == 0.0


# --- undefined_policy=float ---


@pytest.mark.parametrize(
    ("undefined_policy", "expected"),
    [
        pytest.param(0.0, 0.0, id="float-zero"),
        pytest.param(1.0, 1.0, id="float-one"),
        pytest.param(0.5, 0.5, id="float-half"),
        pytest.param(-1.0, -1.0, id="float-negative"),
    ],
)
def test_resolve_fill_value_float_returns_value(undefined_policy: float, expected: float) -> None:
    fill = resolve_fill_value(
        undefined_mask=np.array([True, False]),
        undefined_policy=undefined_policy,
    )
    assert fill == expected


def test_resolve_fill_value_float_nan_returns_nan() -> None:
    fill = resolve_fill_value(
        undefined_mask=np.array([True, False]),
        undefined_policy=float("nan"),
    )
    assert math.isnan(fill)  # NaN identity check: NaN != NaN


def test_resolve_fill_value_float_no_warning() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        resolve_fill_value(
            undefined_mask=np.array([True, False]),
            undefined_policy=0.5,
        )


# --- all entries undefined ---


def test_resolve_fill_value_all_undefined_warn() -> None:
    with pytest.warns(
        UserWarning, match=r"The metric is undefined for 3 element\(s\) at indices \[0, 1, 2\]"
    ):
        fill = resolve_fill_value(
            undefined_mask=np.array([True, True, True]),
            undefined_policy="warn",
        )
    assert fill == 0.0


def test_resolve_fill_value_all_undefined_raise() -> None:
    with pytest.raises(
        ValueError, match=r"The metric is undefined for 3 element\(s\) at indices \[0, 1, 2\]"
    ):
        resolve_fill_value(
            undefined_mask=np.array([True, True, True]),
            undefined_policy="raise",
        )


# --- single entry undefined ---


def test_resolve_fill_value_single_undefined_warn() -> None:
    with pytest.warns(
        UserWarning, match=r"The metric is undefined for 1 element\(s\) at indices \[0\]"
    ):
        fill = resolve_fill_value(
            undefined_mask=np.array([True]),
            undefined_policy="warn",
        )
    assert fill == 0.0


def test_resolve_fill_value_single_undefined_raise() -> None:
    with pytest.raises(
        ValueError, match=r"The metric is undefined for 1 element\(s\) at indices \[0\]"
    ):
        resolve_fill_value(
            undefined_mask=np.array([True]),
            undefined_policy="raise",
        )
