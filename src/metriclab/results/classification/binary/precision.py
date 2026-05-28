r"""Binary precision result implementation."""

from __future__ import annotations

__all__ = ["BinaryPrecisionResult", "compute_precision"]

import math
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from coola.equality import objects_are_allclose, objects_are_equal

from metriclab.results.base import BaseResult
from metriclab.utils.format import make_robust_bar

if TYPE_CHECKING:
    from typing import Self


@dataclass(frozen=True)
class BinaryPrecisionResult(BaseResult):
    r"""Store aggregated values for binary classification precision.

    Precision is defined as ``TP / (TP + FP)``, i.e. the fraction of
    positive predictions that are correct. It measures how many of the
    predicted positives are actually positive.

    Attributes:
        precision: The precision score, in ``[0, 1]``, or ``nan`` if
            it could not be computed.
        num_predictions: The total number of predictions.
        num_positive_predictions: The number of predictions where the
            predicted label is the positive class (i.e. ``TP + FP``).

    Raises:
        ValueError: if ``num_predictions`` is negative.
        ValueError: if ``num_positive_predictions`` is negative.
        ValueError: if ``num_positive_predictions`` is greater than
            ``num_predictions``.
        ValueError: if ``precision`` is negative and not ``nan``.

    Example:
        ```pycon
        >>> from metriclab.results import BinaryPrecisionResult
        >>> m = BinaryPrecisionResult(
        ...     precision=0.75, num_predictions=10, num_positive_predictions=4
        ... )
        >>> m
        BinaryPrecisionResult(precision=0.75, num_predictions=10, num_positive_predictions=4)
        >>> m.precision
        0.75
        >>> m.to_dict()
        {'precision': 0.75, 'num_predictions': 10, 'num_positive_predictions': 4}
        >>> print(m.to_display())
        Precision [███████████████░░░░░]  0.7500  (3/4)  [n=10]

        ```
    """

    precision: float
    num_predictions: int
    num_positive_predictions: int | float

    def __post_init__(self) -> None:
        if self.num_predictions < 0:
            msg = f"'num_predictions' must be >= 0, got {self.num_predictions}"
            raise ValueError(msg)
        if self.num_positive_predictions < 0:
            msg = f"'num_positive_predictions' must be >= 0, got {self.num_positive_predictions}"
            raise ValueError(msg)
        if self.num_positive_predictions > self.num_predictions:
            msg = (
                f"'num_positive_predictions' ({self.num_positive_predictions}) "
                f"must be <= 'num_predictions' ({self.num_predictions})"
            )
            raise ValueError(msg)
        if not math.isnan(self.precision) and self.precision < 0:
            msg = f"'precision' must be >= 0, got {self.precision}"
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

    def to_dict(self, prefix: str = "", suffix: str = "") -> dict[str, int | float]:
        return {
            f"{prefix}precision{suffix}": self.precision,
            f"{prefix}num_predictions{suffix}": self.num_predictions,
            f"{prefix}num_positive_predictions{suffix}": self.num_positive_predictions,
        }

    def to_display(self) -> str:
        if self.num_predictions == 0:
            return f"{self.__class__.__qualname__}: no predictions"
        score = self.precision
        bar = make_robust_bar(score, length=20)
        if math.isnan(score):
            score_str = f"nan  [n={self.num_predictions}]"
        else:
            tp = round(score * self.num_positive_predictions)
            score_str = (
                f"{score:.4f}  ({tp}/{self.num_positive_predictions})  [n={self.num_predictions}]"
            )
        return f"Precision {bar}  {score_str}"

    @classmethod
    def from_tp_fp(
        cls, true_positives: float, false_positives: float, num_predictions: int
    ) -> Self:
        """Instantiate from true positives, false positives, and total
        predictions.

        Precision is computed as ``TP / (TP + FP)``. When
        ``TP + FP == 0`` (no positive predictions), precision is set to
        ``nan``.

        Args:
            true_positives: The number of true positive predictions.
            false_positives: The number of false positive predictions.
            num_predictions: The total number of predictions.

        Returns:
            A ``BinaryPrecisionResult`` with precision and
                ``num_positive_predictions`` derived from the inputs.

        Example:
            ```pycon
            >>> from metriclab.results import BinaryPrecisionResult
            >>> BinaryPrecisionResult.from_tp_fp(
            ...     true_positives=3, false_positives=1, num_predictions=10
            ... )
            BinaryPrecisionResult(precision=0.75, num_predictions=10, num_positive_predictions=4)
            >>> # no positive predictions: precision is nan
            >>> BinaryPrecisionResult.from_tp_fp(
            ...     true_positives=0, false_positives=0, num_predictions=10
            ... )
            BinaryPrecisionResult(precision=nan, num_predictions=10, num_positive_predictions=0)

            ```
        """
        return cls(
            precision=compute_precision(
                true_positives=true_positives, false_positives=false_positives
            ),
            num_predictions=num_predictions,
            num_positive_predictions=true_positives + false_positives,
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
        ```pycon
        >>> from metriclab.results.classification.binary.precision import compute_precision
        >>> compute_precision(true_positives=3, false_positives=1)
        0.75
        >>> compute_precision(true_positives=0, false_positives=0)
        0.0
        >>> compute_precision(true_positives=float("nan"), false_positives=1)
        nan

        ```
    """
    if math.isnan(true_positives) or math.isnan(false_positives):
        return float("nan")
    denominator = true_positives + false_positives
    return true_positives / denominator if denominator > 0 else 0.0
