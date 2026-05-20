r"""Classification accuracy result implementation."""

from __future__ import annotations

__all__ = ["AccuracyResult", "compute_accuracy"]

import math
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from coola.equality import objects_are_allclose, objects_are_equal
from coola.utils.format import make_bar

from metriclab.results.base import BaseResult

if TYPE_CHECKING:
    from typing import Self


@dataclass(frozen=True)
class AccuracyResult(BaseResult):
    r"""Store aggregated values used to compute classification accuracy.

    Attributes:
        num_correct_predictions: The number of correct predictions.
            Set to NaN if the number of correct predictions is unknown
            because of missing values.
        num_predictions: The number of predictions.

    Example:
        ```pycon
        >>> from metriclab.results import AccuracyResult
        >>> m = AccuracyResult(num_correct_predictions=7, num_predictions=10)
        >>> m
        AccuracyResult(num_correct_predictions=7, num_predictions=10)
        >>> m.to_dict()
        {'accuracy': 0.7, 'num_correct_predictions': 7, 'num_predictions': 10}
        >>> print(m.to_display())
        Accuracy [██████████████░░░░░░]  0.7000  (7/10)

        ```
    """

    num_correct_predictions: int | float
    num_predictions: int

    def __post_init__(self) -> None:
        r"""Validate metric counts after dataclass initialization.

        Raises:
            ValueError: If ``num_predictions`` is negative, if
                ``num_correct_predictions`` is negative, or if
                ``num_correct_predictions`` is greater than
                ``num_predictions``.
        """
        if self.num_predictions < 0:
            msg = f"num_predictions must be >= 0, got {self.num_predictions}"
            raise ValueError(msg)
        if self.num_correct_predictions < 0:
            msg = f"num_correct_predictions must be >= 0, got {self.num_correct_predictions}"
            raise ValueError(msg)
        if self.num_correct_predictions > self.num_predictions:
            msg = (
                f"num_correct_predictions ({self.num_correct_predictions}) "
                f"cannot exceed num_predictions ({self.num_predictions})"
            )
            raise ValueError(msg)

    @property
    def accuracy(self) -> float:
        r"""Return the accuracy ratio.

        Returns:
            The ratio ``num_correct_predictions / num_predictions``.
            Returns ``nan`` when ``num_predictions`` is ``0``.
        """
        return compute_accuracy(
            num_correct_predictions=self.num_correct_predictions,
            num_predictions=self.num_predictions,
        )

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
            asdict(self),
            asdict(other),
            rtol=rtol,
            atol=atol,
            equal_nan=equal_nan,
        )

    def equal(self, other: object, equal_nan: bool = False) -> bool:
        if type(other) is not type(self):
            return False
        return objects_are_equal(asdict(self), asdict(other), equal_nan=equal_nan)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(
            num_correct_predictions=data["num_correct_predictions"],
            num_predictions=data["num_predictions"],
        )

    def to_dict(self, prefix: str = "", suffix: str = "") -> dict[str, int | float]:
        return {
            f"{prefix}accuracy{suffix}": self.accuracy,
            f"{prefix}num_correct_predictions{suffix}": self.num_correct_predictions,
            f"{prefix}num_predictions{suffix}": self.num_predictions,
        }

    def to_display(self) -> str:
        if self.num_predictions == 0:
            return f"{self.__class__.__qualname__}: no predictions"
        if math.isnan(self.num_correct_predictions):
            return f"{self.__class__.__qualname__}: unknown number of correct predictions"
        accuracy = self.accuracy
        return (
            f"Accuracy {make_bar(accuracy, length=20)}  {accuracy:.4f}  "
            f"({self.num_correct_predictions:,}/{self.num_predictions:,})"
        )

    def combine(self, other: AccuracyResult) -> AccuracyResult:
        if not isinstance(other, AccuracyResult):
            msg = f"Cannot combine {self.__class__.__qualname__} with {type(other)}"
            raise TypeError(msg)
        return AccuracyResult(
            num_correct_predictions=self.num_correct_predictions + other.num_correct_predictions,
            num_predictions=self.num_predictions + other.num_predictions,
        )


def compute_accuracy(num_correct_predictions: float, num_predictions: float) -> float:
    r"""Compute the accuracy score.

    Args:
        num_correct_predictions: The number of correct predictions,
            or ``nan``.
        num_predictions: The total number of predictions, or ``nan``.

    Returns:
        The ratio ``num_correct_predictions / num_predictions``.
        Returns ``nan`` when ``num_predictions`` is ``0`` or ``nan``.

    Example:
        ```pycon
        >>> from metriclab.results.classification.accuracy import compute_accuracy
        >>> compute_accuracy(num_correct_predictions=7, num_predictions=10)
        0.7
        >>> compute_accuracy(num_correct_predictions=0, num_predictions=0)
        nan
        >>> compute_accuracy(num_correct_predictions=float("nan"), num_predictions=10)
        nan
        >>> compute_accuracy(num_correct_predictions=7, num_predictions=float("nan"))
        nan

        ```
    """
    if math.isnan(num_predictions) or num_predictions == 0:
        return float("nan")
    return num_correct_predictions / num_predictions
