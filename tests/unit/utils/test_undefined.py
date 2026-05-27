from __future__ import annotations

from typing import Any

import pytest

from metriclab.utils.undefined import validate_undefined_policy

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
    with pytest.raises(ValueError, match="'bad_value'"):
        validate_undefined_policy("bad_value")


def test_validate_undefined_policy_error_mentions_float() -> None:
    with pytest.raises(ValueError, match="float"):
        validate_undefined_policy(1)


def test_validate_undefined_policy_error_mentions_valid_strings() -> None:
    with pytest.raises(ValueError, match=r"'nan'.*'raise'.*'warn'"):
        validate_undefined_policy("bad_value")
