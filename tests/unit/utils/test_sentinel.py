from typing import Any

from metriclab.utils.sentinel import NOT_SET, _NotSet

##################################
#     Tests for _NotSet          #
##################################


# --- instantiation ---


def test_not_set_is_instance_of_not_set_class() -> None:
    assert isinstance(NOT_SET, _NotSet)


def test_not_set_is_singleton() -> None:
    # NOT_SET should be the single instance — creating another is
    # possible but NOT_SET itself should always be the same object
    assert NOT_SET is NOT_SET


# --- __repr__ ---


def test_not_set_repr() -> None:
    assert repr(NOT_SET) == "<NotSet>"


def test_not_set_repr_returns_str() -> None:
    assert isinstance(repr(NOT_SET), str)


def test_not_set_str() -> None:
    # str() falls back to __repr__ when __str__ is not defined
    assert str(NOT_SET) == "<NotSet>"


# --- __bool__ ---


def test_not_set_is_falsy() -> None:
    assert not NOT_SET


def test_not_set_bool_is_false() -> None:
    assert bool(NOT_SET) is False


def test_not_set_in_if_statement() -> None:
    # Verify it behaves correctly in a boolean context
    result = "set" if NOT_SET else "not set"
    assert result == "not set"


# --- identity and equality ---


def test_not_set_is_not_none() -> None:
    assert NOT_SET is not None


def test_not_set_is_not_false() -> None:
    assert NOT_SET is not False


def test_not_set_is_not_empty_string() -> None:
    assert NOT_SET != ""


def test_not_set_is_not_zero() -> None:
    assert NOT_SET != 0


def test_not_set_identity() -> None:
    # Same object should be identical to itself
    sentinel = NOT_SET
    assert sentinel is NOT_SET


# --- usage as sentinel ---


def test_not_set_as_default_argument() -> None:
    def fn(value: Any = NOT_SET) -> bool:
        return value is NOT_SET

    assert fn()  # no argument — sentinel returned
    assert not fn(None)  # None explicitly passed
    assert not fn(0)  # 0 explicitly passed
    assert not fn(False)  # False explicitly passed


def test_not_set_distinguishable_from_none() -> None:
    def fn(value: Any = NOT_SET) -> str:
        if value is NOT_SET:
            return "not provided"
        if value is None:
            return "explicitly none"
        return "provided"

    assert fn() == "not provided"
    assert fn(None) == "explicitly none"
    assert fn(42) == "provided"


def test_not_set_distinguishable_from_false() -> None:
    def fn(value: Any = NOT_SET) -> str:
        if value is NOT_SET:
            return "not provided"
        return "provided"

    assert fn() == "not provided"
    assert fn(False) == "provided"
