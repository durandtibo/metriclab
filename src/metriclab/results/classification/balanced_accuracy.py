r"""Balanced accuracy result implementation."""

from __future__ import annotations

__all__ = ["BalancedAccuracyResult"]

import math
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from coola.equality import objects_are_allclose, objects_are_equal

from metriclab.results.base import BaseResult
from metriclab.utils.format import make_robust_bar

if TYPE_CHECKING:
    from typing import Self


@dataclass(frozen=True)
class BalancedAccuracyResult(BaseResult):
    r"""Store aggregated values for balanced classification accuracy.

        Balanced accuracy is the macro-average of per-class recall. Unlike
        standard accuracy, it is robust to class imbalance.

    Attributes:
        balanced_accuracy: The balanced accuracy score, in ``[0, 1]``,
            or ``nan`` if it could not be computed.
        num_predictions: The total number of predictions.

    Raises:
        ValueError: if ``num_predictions`` is negative.
        ValueError: if ``balanced_accuracy`` is negative and not
            ``nan``.

    Example:
        ```pycon
        >>> from metriclab.results import BalancedAccuracyResult
        >>> m = BalancedAccuracyResult(balanced_accuracy=0.7, num_predictions=10)
        >>> m
        BalancedAccuracyResult(balanced_accuracy=0.7, num_predictions=10)
        >>> m.to_dict()
        {'balanced_accuracy': 0.7, 'num_predictions': 10}
        >>> print(m.to_display())
        Balanced Accuracy [██████████████░░░░░░]  0.7000

        ```
    """

    balanced_accuracy: float
    num_predictions: int

    def __post_init__(self) -> None:
        if self.num_predictions < 0:
            msg = f"num_predictions must be >= 0, got {self.num_predictions}"
            raise ValueError(msg)
        if not math.isnan(self.balanced_accuracy) and self.balanced_accuracy < 0:
            msg = f"balanced_accuracy must be >= 0, got {self.balanced_accuracy}"
            raise ValueError(msg)

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
            balanced_accuracy=data["balanced_accuracy"],
            num_predictions=data["num_predictions"],
        )

    def to_dict(self, prefix: str = "", suffix: str = "") -> dict[str, int | float]:
        return {
            f"{prefix}balanced_accuracy{suffix}": self.balanced_accuracy,
            f"{prefix}num_predictions{suffix}": self.num_predictions,
        }

    def to_display(self) -> str:
        if self.num_predictions == 0:
            return f"{self.__class__.__qualname__}: no predictions"
        score = self.balanced_accuracy
        bar = make_robust_bar(score, length=20)
        score_str = "nan" if math.isnan(score) else f"{score:.4f}"
        return f"Balanced Accuracy {bar}  {score_str}"
