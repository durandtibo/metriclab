r"""Precision result implementation."""

from __future__ import annotations

__all__ = ["PrecisionResult"]

import math
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from coola.equality import objects_are_allclose, objects_are_equal

from metriclab.results.base import BaseResult
from metriclab.results.classification.binary.precision import compute_precision
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
        true_positives: The number of true positives (``TP``).
        false_positives: The number of false positives (``FP``).

    Raises:
        ValueError: if ``true_positives`` is negative and not ``nan``.
        ValueError: if ``false_positives`` is negative and not ``nan``.

    Example:
        ```pycon
        >>> from metriclab.results import PrecisionResult
        >>> m = PrecisionResult(true_positives=3, false_positives=1)
        >>> m
        PrecisionResult(true_positives=3, false_positives=1)
        >>> m.precision
        0.75
        >>> m.num_positive_predictions
        4
        >>> m.to_dict()
        {'precision': 0.75, 'true_positives': 3, 'false_positives': 1, 'num_positive_predictions': 4}
        >>> print(m.to_display())
        Precision [███████████████░░░░░]  0.7500  (3/4)

        ```
    """

    true_positives: int | float
    false_positives: int | float

    def __post_init__(self) -> None:
        for name, value in (
            ("true_positives", self.true_positives),
            ("false_positives", self.false_positives),
        ):
            if not math.isnan(float(value)) and value < 0:
                msg = f"{name} must be >= 0, got {value}"
                raise ValueError(msg)

    @property
    def num_positive_predictions(self) -> int | float:
        r"""Return the number of positive predictions (``TP + FP``).

        Returns:
            The sum ``true_positives + false_positives``, or ``nan``
                if either count is ``nan``.
        """
        tp = float(self.true_positives)
        fp = float(self.false_positives)
        if math.isnan(tp) or math.isnan(fp):
            return float("nan")
        return self.true_positives + self.false_positives

    @property
    def precision(self) -> float:
        r"""Return the precision score.

        Returns:
            ``true_positives / (true_positives + false_positives)``.
                Returns ``nan`` when either count is ``nan``. Returns
                ``0.0`` when ``true_positives + false_positives`` is ``0``.
        """
        return compute_precision(
            true_positives=float(self.true_positives),
            false_positives=float(self.false_positives),
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

    def to_dict(self, prefix: str = "", suffix: str = "") -> dict[str, int | float]:
        return {
            f"{prefix}precision{suffix}": self.precision,
            f"{prefix}true_positives{suffix}": self.true_positives,
            f"{prefix}false_positives{suffix}": self.false_positives,
            f"{prefix}num_positive_predictions{suffix}": self.num_positive_predictions,
        }

    def to_display(self) -> str:
        if self.num_positive_predictions == 0:
            return f"{self.__class__.__qualname__}: no predictions"
        score = self.precision
        bar = make_robust_bar(score, length=20)
        score_str = "nan" if math.isnan(score) else f"{score:.4f}"
        tp_str = "?" if math.isnan(float(self.true_positives)) else f"{int(self.true_positives):,}"
        pp = self.num_positive_predictions
        pp_str = "?" if math.isnan(float(pp)) else f"{int(pp):,}"
        return f"Precision {bar}  {score_str}  ({tp_str}/{pp_str})"

    @classmethod
    def from_precision(
        cls,
        precision: float,
        num_positive_predictions: float,
    ) -> Self:
        r"""Create a ``PrecisionResult`` from a precision score and the
        number of positive predictions.

        The number of true positives is back-computed as
        ``round(precision * num_positive_predictions)`` and false
        positives as the remainder.

        Args:
            precision: The precision score, in ``[0, 1]``, or ``nan``.
            num_positive_predictions: The number of predicted positives
                (``TP + FP``).

        Returns:
            A ``PrecisionResult`` with computed ``true_positives`` and
                ``false_positives``.

        Example:
            ```pycon
            >>> from metriclab.results import PrecisionResult
            >>> m = PrecisionResult.from_precision(precision=0.75, num_positive_predictions=4)
            >>> m.true_positives
            3
            >>> m.false_positives
            1
            >>> m.precision
            0.75

            ```
        """
        if math.isnan(precision) or math.isnan(float(num_positive_predictions)):
            return cls(
                true_positives=float("nan"),
                false_positives=float("nan"),
            )
        tp = round(precision * num_positive_predictions)
        fp = round(num_positive_predictions) - tp
        return cls(true_positives=tp, false_positives=fp)
