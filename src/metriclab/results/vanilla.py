r"""A generic immutable container for metric values."""

from __future__ import annotations

__all__ = ["Result"]

from dataclasses import dataclass
from typing import Any

from coola.equality import objects_are_allclose, objects_are_equal

from metriclab.results.base import BaseResult


@dataclass(frozen=True)
class Result(BaseResult):
    r"""Implement a simple metric result container.

    Args:
        results: A dictionary containing metric names and values.

    ``to_dict`` returns the same key-value pairs as ``results`` when
    no prefix or suffix is provided; keys are prefixed and suffixed
    otherwise. ``to_display`` renders the exported dictionary with
    :class:`str`.

    Example:
        ```pycon
        >>> from metriclab.results import Result
        >>> result = Result({"accuracy": 0.8, "loss": 0.2})
        >>> result.to_dict(suffix="_metric")
        {'accuracy_metric': 0.8, 'loss_metric': 0.2}
        >>> print(result.to_display())
        {'accuracy': 0.8, 'loss': 0.2}

        ```
    """

    results: dict[str, Any]

    def allclose(
        self,
        other: object,
        *,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        equal_nan: bool = False,
    ) -> bool:
        if type(other) is not type(self):
            return False
        return objects_are_allclose(
            self.results,
            other.results,
            rtol=rtol,
            atol=atol,
            equal_nan=equal_nan,
        )

    def equal(self, other: object, equal_nan: bool = False) -> bool:
        if type(other) is not type(self):
            return False
        return objects_are_equal(self.results, other.results, equal_nan=equal_nan)

    def to_dict(self, prefix: str = "", suffix: str = "") -> dict[str, Any]:
        return {f"{prefix}{k}{suffix}": v for k, v in self.results.items()}

    def to_display(self) -> str:
        return str(self.to_dict())
