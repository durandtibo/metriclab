from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
from coola.equality import objects_are_equal

from metriclab.results import BaseResult, Result, ResultDict

##################################
#     Tests for ResultDict       #
##################################


# --- instantiation ---


def test_result_dict_instantiation() -> None:
    r = ResultDict({"train": Result({"loss": 0.5}), "val": Result({"loss": 0.3})})
    assert isinstance(r.results["train"], Result)
    assert isinstance(r.results["val"], Result)


def test_result_dict_empty() -> None:
    r = ResultDict({})
    assert r.results == {}


def test_result_dict_frozen() -> None:
    r = ResultDict({"train": Result({"loss": 0.5})})
    with pytest.raises(FrozenInstanceError, match="cannot assign to field 'results'"):
        r.results = {}


def test_result_dict_is_base_result() -> None:
    assert isinstance(ResultDict({}), BaseResult)


# --- __repr__ ---


def test_result_dict_repr_empty() -> None:
    assert repr(ResultDict({})) == "ResultDict()"


def test_result_dict_repr_single_key() -> None:
    r = ResultDict({"train": Result({"loss": 0.5})})
    assert repr(r) == ("ResultDict(\n  (train): Result(results={'loss': 0.5})\n)")


def test_result_dict_repr_multiple_keys() -> None:
    r = ResultDict({"train": Result({"loss": 0.5}), "val": Result({"loss": 0.3})})
    assert repr(r) == (
        "ResultDict(\n"
        "  (train): Result(results={'loss': 0.5})\n"
        "  (val): Result(results={'loss': 0.3})\n"
        ")"
    )


# --- __str__ ---


def test_result_dict_str_empty() -> None:
    assert str(ResultDict({})) == "ResultDict()"


def test_result_dict_str_single_key() -> None:
    r = ResultDict({"train": Result({"loss": 0.5})})
    assert str(r) == ("ResultDict(\n  (train): Result(results={'loss': 0.5})\n)")


def test_result_dict_str_multiple_keys() -> None:
    r = ResultDict({"train": Result({"loss": 0.5}), "val": Result({"loss": 0.3})})
    assert str(r) == (
        "ResultDict(\n"
        "  (train): Result(results={'loss': 0.5})\n"
        "  (val): Result(results={'loss': 0.3})\n"
        ")"
    )


# --- equal ---


def test_result_dict_equal_true() -> None:
    assert ResultDict({"train": Result({"loss": 0.5})}).equal(
        ResultDict({"train": Result({"loss": 0.5})})
    )


def test_result_dict_equal_empty() -> None:
    assert ResultDict({}).equal(ResultDict({}))


def test_result_dict_equal_multiple_keys() -> None:
    assert ResultDict({"train": Result({"loss": 0.5}), "val": Result({"loss": 0.3})}).equal(
        ResultDict({"train": Result({"loss": 0.5}), "val": Result({"loss": 0.3})})
    )


def test_result_dict_equal_false_different_values() -> None:
    assert not ResultDict({"train": Result({"loss": 0.5})}).equal(
        ResultDict({"train": Result({"loss": 0.9})})
    )


def test_result_dict_equal_false_different_keys() -> None:
    assert not ResultDict({"train": Result({"loss": 0.5})}).equal(
        ResultDict({"val": Result({"loss": 0.5})})
    )


def test_result_dict_equal_false_different_number_of_keys() -> None:
    assert not ResultDict({"train": Result({"loss": 0.5}), "val": Result({"loss": 0.3})}).equal(
        ResultDict({"train": Result({"loss": 0.5})})
    )


def test_result_dict_equal_wrong_type() -> None:
    assert not ResultDict({"train": Result({"loss": 0.5})}).equal("not a result")


def test_result_dict_equal_nan_false_by_default() -> None:
    assert not ResultDict({"train": Result({"loss": float("nan")})}).equal(
        ResultDict({"train": Result({"loss": float("nan")})})
    )


def test_result_dict_equal_nan_true_with_equal_nan() -> None:
    assert ResultDict({"train": Result({"loss": float("nan")})}).equal(
        ResultDict({"train": Result({"loss": float("nan")})}), equal_nan=True
    )


# --- allclose ---


def test_result_dict_allclose_true() -> None:
    assert ResultDict({"train": Result({"loss": 0.5})}).allclose(
        ResultDict({"train": Result({"loss": 0.5})})
    )


def test_result_dict_allclose_within_tolerance() -> None:
    assert ResultDict({"train": Result({"loss": 1.0})}).allclose(
        ResultDict({"train": Result({"loss": 1.0 + 1e-7})}), rtol=1e-5, atol=1e-6
    )


def test_result_dict_allclose_outside_tolerance() -> None:
    assert not ResultDict({"train": Result({"loss": 1.0})}).allclose(
        ResultDict({"train": Result({"loss": 1.1})}), rtol=1e-5, atol=1e-8
    )


def test_result_dict_allclose_empty() -> None:
    assert ResultDict({}).allclose(ResultDict({}))


def test_result_dict_allclose_false_different_values() -> None:
    assert not ResultDict({"train": Result({"loss": 0.5})}).allclose(
        ResultDict({"train": Result({"loss": 0.9})})
    )


def test_result_dict_allclose_false_different_keys() -> None:
    assert not ResultDict({"train": Result({"loss": 0.5})}).allclose(
        ResultDict({"val": Result({"loss": 0.5})})
    )


def test_result_dict_allclose_false_different_number_of_keys() -> None:
    assert not ResultDict({"train": Result({"loss": 0.5}), "val": Result({"loss": 0.3})}).allclose(
        ResultDict({"train": Result({"loss": 0.5})})
    )


def test_result_dict_allclose_wrong_type() -> None:
    assert not ResultDict({"train": Result({"loss": 0.5})}).allclose("not a result")


def test_result_dict_allclose_nan_false_by_default() -> None:
    assert not ResultDict({"train": Result({"loss": float("nan")})}).allclose(
        ResultDict({"train": Result({"loss": float("nan")})})
    )


def test_result_dict_allclose_nan_true_with_equal_nan() -> None:
    assert ResultDict({"train": Result({"loss": float("nan")})}).allclose(
        ResultDict({"train": Result({"loss": float("nan")})}), equal_nan=True
    )


# --- to_dict ---


@pytest.mark.parametrize(
    ("results", "prefix", "suffix", "expected"),
    [
        pytest.param(
            {"train": Result({"loss": 0.5}), "val": Result({"loss": 0.3})},
            "",
            "",
            {"train": {"loss": 0.5}, "val": {"loss": 0.3}},
            id="no-prefix-suffix",
        ),
        pytest.param(
            {"train": Result({"loss": 0.5}), "val": Result({"loss": 0.3})},
            "phase_",
            "",
            {"phase_train": {"loss": 0.5}, "phase_val": {"loss": 0.3}},
            id="prefix-only",
        ),
        pytest.param(
            {"train": Result({"loss": 0.5}), "val": Result({"loss": 0.3})},
            "",
            "_epoch1",
            {"train_epoch1": {"loss": 0.5}, "val_epoch1": {"loss": 0.3}},
            id="suffix-only",
        ),
        pytest.param(
            {"train": Result({"loss": 0.5}), "val": Result({"loss": 0.3})},
            "phase_",
            "_epoch1",
            {"phase_train_epoch1": {"loss": 0.5}, "phase_val_epoch1": {"loss": 0.3}},
            id="prefix-and-suffix",
        ),
        pytest.param(
            {},
            "phase_",
            "_epoch1",
            {},
            id="empty-results",
        ),
        pytest.param(
            {"train": Result({"loss": 0.5})},
            "",
            "",
            {"train": {"loss": 0.5}},
            id="single-key",
        ),
    ],
)
def test_result_dict_to_dict(results: dict, prefix: str, suffix: str, expected: dict) -> None:
    assert objects_are_equal(ResultDict(results).to_dict(prefix=prefix, suffix=suffix), expected)


def test_result_dict_to_dict_multiple_metrics() -> None:
    assert objects_are_equal(
        ResultDict({"train": Result({"loss": 0.5, "accuracy": 0.9})}).to_dict(),
        {"train": {"loss": 0.5, "accuracy": 0.9}},
    )


def test_result_dict_to_dict_nan_value() -> None:
    assert objects_are_equal(
        ResultDict({"train": Result({"loss": float("nan")})}).to_dict(),
        {"train": {"loss": float("nan")}},
        equal_nan=True,
    )


def test_result_dict_to_dict_nested_calls_child_to_dict() -> None:
    result = ResultDict({"train": Result({"loss": 0.5, "acc": 0.9})})
    d = result.to_dict(prefix="p_", suffix="_s")
    assert "p_train_s" in d
    assert objects_are_equal(d["p_train_s"], {"loss": 0.5, "acc": 0.9})


# --- to_display ---


def test_result_dict_to_display_empty() -> None:
    assert ResultDict({}).to_display() == ""


def test_result_dict_to_display_single_key() -> None:
    assert ResultDict({"train": Result({"loss": 0.5})}).to_display() == (
        "=== train ===\n{'loss': 0.5}\n"
    )


def test_result_dict_to_display_multiple_keys() -> None:
    assert ResultDict(
        {"train": Result({"loss": 0.5}), "val": Result({"loss": 0.3})}
    ).to_display() == ("=== train ===\n{'loss': 0.5}\n\n=== val ===\n{'loss': 0.3}\n")


def test_result_dict_to_display_three_keys() -> None:
    assert ResultDict(
        {
            "train": Result({"loss": 0.5}),
            "val": Result({"loss": 0.3}),
            "test": Result({"loss": 0.2}),
        }
    ).to_display() == (
        "=== train ===\n"
        "{'loss': 0.5}\n"
        "\n"
        "=== val ===\n"
        "{'loss': 0.3}\n"
        "\n"
        "=== test ===\n"
        "{'loss': 0.2}\n"
    )


def test_result_dict_to_display_multiple_metrics() -> None:
    assert ResultDict({"train": Result({"loss": 0.5, "accuracy": 0.9})}).to_display() == (
        "=== train ===\n{'loss': 0.5, 'accuracy': 0.9}\n"
    )


def test_result_dict_to_display_returns_str() -> None:
    assert isinstance(ResultDict({}).to_display(), str)
