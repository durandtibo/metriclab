r"""Precision result implementation."""

from __future__ import annotations

__all__ = ["PrecisionResult", "compute_precision"]

import math
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from coola.equality import objects_are_allclose, objects_are_equal

from metriclab.results.base import BaseResult
from metriclab.utils.format import make_robust_bar

if TYPE_CHECKING:
    from typing import Self


@dataclass(frozen=True)
class PrecisionResult(BaseResult):
    r"""Store aggregated values for classification precision.

    Precision is defined as ``TP / (TP + FP)``, i.e. the fraction of
    positive predictions that are correct. It measures how many of the
    predicted positives are actually positive.

    Attributes:
        num_true_positives: The number of true positives (``TP``).
        num_positive_predictions: The number of predicted positives
            (``TP + FP``).

    Raises:
        ValueError: if ``num_true_positives`` is negative.
        ValueError: if ``num_positive_predictions`` is negative.
        ValueError: if ``num_true_positives`` exceeds
            ``num_positive_predictions``.

    Example:
        >>> from metriclab.results import PrecisionResult
        >>> m = PrecisionResult(num_true_positives=3, num_positive_predictions=4)
        >>> m.precision
        0.75
        >>> print(m.to_display())
        Precision [███████████████░░░░░]  0.7500  (3/4)
    """

    num_true_positives: int | float
    num_positive_predictions: int | float

    def __post_init__(self) -> None:
        for name, value in (
            ("num_true_positives", self.num_true_positives),
            ("num_positive_predictions", self.num_positive_predictions),
        ):
            if not math.isnan(float(value)) and value < 0:
                msg = f"{name} must be >= 0, got {value}"
                raise ValueError(msg)
        if (
            not math.isnan(float(self.num_true_positives))
            and not math.isnan(float(self.num_positive_predictions))
            and self.num_true_positives > self.num_positive_predictions
        ):
            msg = (
                f"num_true_positives ({self.num_true_positives}) cannot exceed "
                f"num_positive_predictions ({self.num_positive_predictions})"
            )
            raise ValueError(msg)

    @property
    def precision(self) -> float:
        r"""Return the precision score.

        Returns:
            ``num_true_positives / num_positive_predictions``.
            Returns ``nan`` when ``num_positive_predictions`` is ``0``
            or either count is ``nan``.
        """
        tp = float(self.num_true_positives)
        pp = float(self.num_positive_predictions)
        if math.isnan(tp) or math.isnan(pp) or pp == 0:
            return float("nan")
        return tp / pp

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

    def to_dict(self, prefix: str = "", suffix: str = "") -> dict[str, int | float]:
        return {
            f"{prefix}precision{suffix}": self.precision,
            f"{prefix}num_true_positives{suffix}": self.num_true_positives,
            f"{prefix}num_positive_predictions{suffix}": self.num_positive_predictions,
        }

    def to_display(self) -> str:
        if self.num_positive_predictions == 0:
            return f"{self.__class__.__qualname__}: no predictions"
        score = self.precision
        bar = make_robust_bar(score, length=20)
        score_str = "nan" if math.isnan(score) else f"{score:.4f}"
        tp_str = (
            "?"
            if math.isnan(float(self.num_true_positives))
            else f"{int(self.num_true_positives):,}"
        )
        return f"Precision {bar}  {score_str}  ({tp_str}/{int(self.num_positive_predictions):,})"

    @classmethod
    def from_precision(
        cls,
        precision: float,
        num_positive_predictions: float,
    ) -> Self:
        r"""Create a ``PrecisionResult`` from a precision score and the
        number of positive predictions.

        The number of true positives is back-computed as
        ``round(precision * num_positive_predictions)``.

        Args:
            precision: The precision score, in ``[0, 1]``, or ``nan``.
            num_positive_predictions: The number of predicted positives
                (``TP + FP``).

        Returns:
            A ``PrecisionResult`` with the computed ``num_true_positives``.

        Example:
            >>> from metriclab.results import PrecisionResult
            >>> m = PrecisionResult.from_precision(precision=0.75, num_positive_predictions=4)
            >>> m.num_true_positives
            3
            >>> m.precision
            0.75
        """
        if math.isnan(precision) or math.isnan(float(num_positive_predictions)):
            return cls(
                num_true_positives=float("nan"),
                num_positive_predictions=num_positive_predictions,
            )
        return cls(
            num_true_positives=round(precision * num_positive_predictions),
            num_positive_predictions=num_positive_predictions,
        )


def compute_precision(
    true_positives: float,
    false_positives: float,
) -> float:
    r"""Compute the precision score.

    Precision measures the proportion of true positives among all
    positive predictions.

    Args:
        true_positives: The number of true positives, or ``nan``.
        false_positives: The number of false positives, or ``nan``.

    Returns:
        The ratio ``true_positives / (true_positives + false_positives)``.
            Returns ``nan`` when either ``true_positives`` or
            ``false_positives`` is ``nan``. Returns ``0.0`` when
            ``true_positives + false_positives`` is ``0``.

    Example:
        >>> from metriclab.results.classification.precision import compute_precision
        >>> compute_precision(true_positives=3, false_positives=1)
        0.75
        >>> compute_precision(true_positives=0, false_positives=0)
        0.0
        >>> compute_precision(true_positives=float("nan"), false_positives=1)
        nan
    """
    if math.isnan(true_positives) or math.isnan(false_positives):
        return float("nan")
    denominator = true_positives + false_positives
    return true_positives / denominator if denominator > 0 else 0.0
