from __future__ import annotations

import pytest

from metriclab.utils.format import make_robust_bar

##################################
#   Tests for make_robust_bar    #
##################################


# --- Finite values in [0, 1] ---


@pytest.mark.parametrize(
    ("value", "length", "expected"),
    [
        pytest.param(0.0, 10, "[░░░░░░░░░░]", id="zero-length-10"),
        pytest.param(1.0, 10, "[██████████]", id="one-length-10"),
        pytest.param(0.6, 10, "[██████░░░░]", id="0.6-length-10"),
        pytest.param(0.1, 10, "[█░░░░░░░░░]", id="0.1-length-10"),
        pytest.param(0.5, 10, "[█████░░░░░]", id="0.5-length-10"),
        pytest.param(0.0, 20, "[░░░░░░░░░░░░░░░░░░░░]", id="zero-length-20"),
        pytest.param(1.0, 20, "[████████████████████]", id="one-length-20"),
        pytest.param(0.6, 20, "[████████████░░░░░░░░]", id="0.6-length-20"),
        pytest.param(0.7, 10, "[███████░░░]", id="0.7-length-10"),
        pytest.param(0.8, 10, "[████████░░]", id="0.8-length-10"),
    ],
)
def test_make_robust_bar_finite(value: float, length: int, expected: str) -> None:
    assert make_robust_bar(value, length=length) == expected


# --- NaN and inf ---


@pytest.mark.parametrize(
    ("value", "length", "expected"),
    [
        pytest.param(float("nan"), 10, "[??????????]", id="nan-length-10"),
        pytest.param(float("inf"), 10, "[??????????]", id="inf-length-10"),
        pytest.param(float("-inf"), 10, "[??????????]", id="neg-inf-length-10"),
        pytest.param(float("nan"), 20, "[????????????????????]", id="nan-length-20"),
        pytest.param(float("inf"), 20, "[????????????????????]", id="inf-length-20"),
        pytest.param(float("nan"), 5, "[?????]", id="nan-length-5"),
    ],
)
def test_make_robust_bar_nan_inf(value: float, length: int, expected: str) -> None:
    assert make_robust_bar(value, length=length) == expected


# --- Output format ---


@pytest.mark.parametrize("length", [1, 5, 10, 20, 50])
def test_make_robust_bar_output_length(length: int) -> None:
    # bar body is exactly `length` characters, plus 2 for brackets
    assert len(make_robust_bar(0.5, length=length)) == length + 2


def test_make_robust_bar_starts_with_bracket() -> None:
    assert make_robust_bar(0.5).startswith("[")


def test_make_robust_bar_ends_with_bracket() -> None:
    assert make_robust_bar(0.5).endswith("]")


def test_make_robust_bar_nan_starts_with_bracket() -> None:
    assert make_robust_bar(float("nan")).startswith("[")


def test_make_robust_bar_nan_ends_with_bracket() -> None:
    assert make_robust_bar(float("nan")).endswith("]")


# --- Invalid inputs ---


@pytest.mark.parametrize(
    "length",
    [
        pytest.param(0, id="zero"),
        pytest.param(-1, id="negative"),
        pytest.param(-10, id="large-negative"),
    ],
)
def test_make_robust_bar_invalid_length_raises(length: int) -> None:
    with pytest.raises(ValueError, match="length must be a positive integer"):
        make_robust_bar(0.5, length=length)


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(1.1, id="slightly-above-one"),
        pytest.param(2.0, id="two"),
        pytest.param(-0.1, id="slightly-below-zero"),
        pytest.param(-1.0, id="negative-one"),
    ],
)
def test_make_robust_bar_out_of_range_raises(value: float) -> None:
    with pytest.raises(ValueError, match="value must be in"):
        make_robust_bar(value)


# --- Default length ---


def test_make_robust_bar_default_length() -> None:
    assert len(make_robust_bar(0.5)) == 12  # 10 + 2 brackets
