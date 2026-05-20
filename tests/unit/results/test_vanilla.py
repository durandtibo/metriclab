from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from coola.equality import objects_are_equal

from metriclab.results import Result

##################################
#       Tests for Result         #
##################################


# --- instantiation ---


def test_result_instantiation() -> None:
    r = Result(results={"accuracy": 0.9, "loss": 0.1})
    assert objects_are_equal(r.results, {"accuracy": 0.9, "loss": 0.1})


def test_result_empty_results() -> None:
    r = Result(results={})
    assert objects_are_equal(r.results, {})


def test_result_frozen() -> None:
    r = Result(results={"accuracy": 0.9})
    with pytest.raises(FrozenInstanceError, match="cannot assign to field 'results'"):
        r.results = {"accuracy": 0.5}  # type: ignore[misc]


def test_result_various_value_types() -> None:
    r = Result(
        results={
            "int": 1,
            "float": 0.5,
            "str": "good",
            "list": [1, 2, 3],
            "array": np.array([1, 2, 3]),
            "nan": float("nan"),
            "none": None,
        }
    )
    assert objects_are_equal(r.results["int"], 1)
    assert objects_are_equal(r.results["float"], 0.5)
    assert objects_are_equal(r.results["str"], "good")
    assert objects_are_equal(r.results["list"], [1, 2, 3])
    assert objects_are_equal(r.results["array"], np.array([1, 2, 3]))
    assert objects_are_equal(r.results["nan"], float("nan"), equal_nan=True)
    assert objects_are_equal(r.results["none"], None)


# --- equal ---


def test_result_equal_true() -> None:
    assert Result(results={"accuracy": 0.9}).equal(Result(results={"accuracy": 0.9}))


def test_result_equal_empty() -> None:
    assert Result(results={}).equal(Result(results={}))


def test_result_equal_multiple_keys() -> None:
    assert Result(results={"a": 1, "b": 2}).equal(Result(results={"a": 1, "b": 2}))


def test_result_equal_false_different_values() -> None:
    assert not Result(results={"accuracy": 0.9}).equal(Result(results={"accuracy": 0.8}))


def test_result_equal_false_different_keys() -> None:
    assert not Result(results={"accuracy": 0.9}).equal(Result(results={"loss": 0.9}))


def test_result_equal_false_different_number_of_keys() -> None:
    assert not Result(results={"a": 1, "b": 2}).equal(Result(results={"a": 1}))


def test_result_equal_wrong_type() -> None:
    assert not Result(results={"accuracy": 0.9}).equal("not a result")


def test_result_equal_nan_false_by_default() -> None:
    assert not Result(results={"x": float("nan")}).equal(Result(results={"x": float("nan")}))


def test_result_equal_nan_true_with_equal_nan() -> None:
    assert Result(results={"x": float("nan")}).equal(
        Result(results={"x": float("nan")}), equal_nan=True
    )


def test_result_equal_numpy_arrays() -> None:
    assert Result(results={"x": np.array([1, 2, 3])}).equal(
        Result(results={"x": np.array([1, 2, 3])})
    )


def test_result_equal_numpy_arrays_false() -> None:
    assert not Result(results={"x": np.array([1, 2, 3])}).equal(
        Result(results={"x": np.array([1, 2, 4])})
    )


# --- allclose ---


def test_result_allclose_true() -> None:
    assert Result(results={"accuracy": 0.9}).allclose(Result(results={"accuracy": 0.9}))


def test_result_allclose_within_tolerance() -> None:
    assert Result(results={"x": 1.0}).allclose(
        Result(results={"x": 1.0 + 1e-7}), rtol=1e-5, atol=1e-6
    )


def test_result_allclose_outside_tolerance() -> None:
    assert not Result(results={"x": 1.0}).allclose(Result(results={"x": 1.1}), rtol=1e-5, atol=1e-8)


def test_result_allclose_empty() -> None:
    assert Result(results={}).allclose(Result(results={}))


def test_result_allclose_false_different_values() -> None:
    assert not Result(results={"accuracy": 0.9}).allclose(Result(results={"accuracy": 0.8}))


def test_result_allclose_false_different_keys() -> None:
    assert not Result(results={"accuracy": 0.9}).allclose(Result(results={"loss": 0.9}))


def test_result_allclose_wrong_type() -> None:
    assert not Result(results={"accuracy": 0.9}).allclose("not a result")


def test_result_allclose_nan_false_by_default() -> None:
    assert not Result(results={"x": float("nan")}).allclose(Result(results={"x": float("nan")}))


def test_result_allclose_nan_true_with_equal_nan() -> None:
    assert Result(results={"x": float("nan")}).allclose(
        Result(results={"x": float("nan")}), equal_nan=True
    )


def test_result_allclose_numpy_arrays() -> None:
    assert Result(results={"x": np.array([1.0, 2.0])}).allclose(
        Result(results={"x": np.array([1.0, 2.0])})
    )


# --- from_dict ---

# --- from_dict ---


def test_result_from_dict_standard() -> None:
    assert Result.from_dict({"accuracy": 0.9, "loss": 0.1}).equal(
        Result(results={"accuracy": 0.9, "loss": 0.1})
    )


def test_result_from_dict_empty() -> None:
    assert Result.from_dict({}).equal(Result(results={}))


def test_result_from_dict_single_key() -> None:
    assert Result.from_dict({"accuracy": 0.9}).equal(Result(results={"accuracy": 0.9}))


def test_result_from_dict_preserves_types() -> None:
    r = Result.from_dict({"arr": np.array([1, 2, 3]), "val": 0.5, "name": "model"})
    assert objects_are_equal(r.results["arr"], np.array([1, 2, 3]))
    assert objects_are_equal(r.results["val"], 0.5)
    assert objects_are_equal(r.results["name"], "model")


def test_result_from_dict_nan_value() -> None:
    assert objects_are_equal(
        Result.from_dict({"x": float("nan"), "y": 1.0}).results,
        {"x": float("nan"), "y": 1.0},
        equal_nan=True,
    )


def test_result_from_dict_returns_result_instance() -> None:
    assert isinstance(Result.from_dict({"accuracy": 0.9}), Result)


def test_result_from_dict_to_dict_roundtrip() -> None:
    data = {"accuracy": 0.9, "loss": 0.1}
    assert objects_are_equal(Result.from_dict(data).to_dict(), data)


# --- to_dict ---


@pytest.mark.parametrize(
    ("results", "prefix", "suffix", "expected"),
    [
        pytest.param(
            {"accuracy": 0.9, "loss": 0.1},
            "",
            "",
            {"accuracy": 0.9, "loss": 0.1},
            id="no-prefix-suffix",
        ),
        pytest.param(
            {"accuracy": 0.9, "loss": 0.1},
            "train_",
            "",
            {"train_accuracy": 0.9, "train_loss": 0.1},
            id="prefix-only",
        ),
        pytest.param(
            {"accuracy": 0.9, "loss": 0.1},
            "",
            "_val",
            {"accuracy_val": 0.9, "loss_val": 0.1},
            id="suffix-only",
        ),
        pytest.param(
            {"accuracy": 0.9, "loss": 0.1},
            "train_",
            "_val",
            {"train_accuracy_val": 0.9, "train_loss_val": 0.1},
            id="prefix-and-suffix",
        ),
        pytest.param(
            {},
            "train_",
            "_val",
            {},
            id="empty-results",
        ),
        pytest.param(
            {"x": 1},
            "",
            "",
            {"x": 1},
            id="single-key",
        ),
        pytest.param(
            {"a": np.array([1, 2, 3]), "b": 0.5},
            "",
            "",
            {"a": np.array([1, 2, 3]), "b": 0.5},
            id="numpy-array-value",
        ),
    ],
)
def test_result_to_dict(results: dict, prefix: str, suffix: str, expected: dict) -> None:
    assert objects_are_equal(
        Result(results=results).to_dict(prefix=prefix, suffix=suffix), expected
    )


def test_result_to_dict_nan_value() -> None:
    assert objects_are_equal(
        Result(results={"x": float("nan"), "y": 1.0}).to_dict(),
        {"x": float("nan"), "y": 1.0},
        equal_nan=True,
    )


def test_result_to_dict_none_value() -> None:
    assert objects_are_equal(
        Result(results={"x": None, "y": 1.0}).to_dict(),
        {"x": None, "y": 1.0},
    )


def test_result_to_dict_mixed_types() -> None:
    assert objects_are_equal(
        Result(results={"int": 1, "float": 0.5, "str": "ok", "list": [1, 2]}).to_dict(),
        {"int": 1, "float": 0.5, "str": "ok", "list": [1, 2]},
    )


# --- to_display ---


@pytest.mark.parametrize(
    ("results", "expected"),
    [
        pytest.param(
            {},
            "{}",
            id="empty",
        ),
        pytest.param(
            {"accuracy": 0.9},
            "{'accuracy': 0.9}",
            id="single-key",
        ),
        pytest.param(
            {"accuracy": 0.9, "loss": 0.1},
            "{'accuracy': 0.9, 'loss': 0.1}",
            id="multiple-keys",
        ),
        pytest.param(
            {"name": "model", "score": 1},
            "{'name': 'model', 'score': 1}",
            id="string-value",
        ),
    ],
)
def test_result_to_display(results: dict, expected: str) -> None:
    assert Result(results=results).to_display() == expected


def test_result_to_display_matches_to_dict_str() -> None:
    r = Result(results={"accuracy": 0.9, "loss": 0.1})
    assert r.to_display() == str(r.to_dict())


def test_result_to_display_returns_str() -> None:
    assert isinstance(Result(results={"accuracy": 0.9}).to_display(), str)


def test_result_to_display_nan() -> None:
    # nan renders as float("nan") in str(dict)
    result = Result(results={"x": float("nan")})
    assert result.to_display() == str({"x": float("nan")})
