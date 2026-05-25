r"""Recall result implementation."""

from __future__ import annotations

__all__ = ["RecallResult", "compute_recall"]

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from coola.equality import objects_are_allclose, objects_are_equal

from metriclab.results.base import BaseResult
from metriclab.utils.format import make_robust_bar

if TYPE_CHECKING:
    from typing import Self


@dataclass(frozen=True)
class RecallResult(BaseResult):
    r"""Store aggregated values for classification recall.

    Recall is defined as ``TP / (TP + FN)``, i.e. the fraction of
    actual positives that are correctly identified. It measures how
    many of the true positives are captured by the model.

    Attributes:
        num_true_positives: The number of true positives (``TP``).
        num_actual_positives: The number of actual positives
            (``TP + FN``).

    Raises:
        ValueError: if ``num_true_positives`` is negative.
        ValueError: if ``num_actual_positives`` is negative.
        ValueError: if ``num_true_positives`` exceeds
            ``num_actual_positives``.

    Example:
    ```pycon
    >>> from metriclab.results import RecallResult
    >>> m = RecallResult(num_true_positives=3, num_actual_positives=5)
    >>> m
    RecallResult(num_true_positives=3, num_actual_positives=5)
    >>> m.recall
    0.6
    >>> m.to_dict()
    {'recall': 0.6, 'num_true_positives': 3, 'num_actual_positives': 5}
    >>> print(m.to_display())
    Recall [████████████░░░░░░░░]  0.6000  (3/5)

    ```
    """

    num_true_positives: int | float
    num_actual_positives: int | float

    def __post_init__(self) -> None:
        for name, value in (
            ("num_true_positives", self.num_true_positives),
            ("num_actual_positives", self.num_actual_positives),
        ):
            if not math.isnan(float(value)) and value < 0:
                msg = f"{name} must be >= 0, got {value}"
                raise ValueError(msg)
        if (
            not math.isnan(float(self.num_true_positives))
            and not math.isnan(float(self.num_actual_positives))
            and self.num_true_positives > self.num_actual_positives
        ):
            msg = (
                f"num_true_positives ({self.num_true_positives}) cannot exceed "
                f"num_actual_positives ({self.num_actual_positives})"
            )
            raise ValueError(msg)

    @property
    def recall(self) -> float:
        r"""Return the recall score.

        Returns:
            ``num_true_positives / num_actual_positives``.
            Returns ``nan`` when ``num_actual_positives`` is ``0``
            or either count is ``nan``.
        """
        tp = float(self.num_true_positives)
        ap = float(self.num_actual_positives)
        if math.isnan(tp) or math.isnan(ap) or ap == 0:
            return float("nan")
        return tp / ap

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
            {
                "num_true_positives": self.num_true_positives,
                "num_actual_positives": self.num_actual_positives,
            },
            {
                "num_true_positives": other.num_true_positives,
                "num_actual_positives": other.num_actual_positives,
            },
            rtol=rtol,
            atol=atol,
            equal_nan=equal_nan,
        )

    def equal(self, other: object, equal_nan: bool = False) -> bool:
        if type(other) is not type(self):
            return False
        return objects_are_equal(
            {
                "num_true_positives": self.num_true_positives,
                "num_actual_positives": self.num_actual_positives,
            },
            {
                "num_true_positives": other.num_true_positives,
                "num_actual_positives": other.num_actual_positives,
            },
            equal_nan=equal_nan,
        )

    @classmethod
    def from_recall(
        cls,
        recall: float,
        num_actual_positives: float,
    ) -> Self:
        r"""Create a ``RecallResult`` from a recall score and the number
        of actual positives.

        The number of true positives is back-computed as
        ``round(recall * num_actual_positives)``.

        Args:
            recall: The recall score, in ``[0, 1]``, or ``nan``.
            num_actual_positives: The number of actual positives
                (``TP + FN``).

        Returns:
            A ``RecallResult`` with the computed ``num_true_positives``.

        Example:
            ```pycon
            >>> from metriclab.results import RecallResult
            >>> m = RecallResult.from_recall(recall=0.6, num_actual_positives=5)
            >>> m.num_true_positives
            3
            >>> m.recall
            0.6

            ```
        """
        if math.isnan(recall) or math.isnan(float(num_actual_positives)):
            return cls(
                num_true_positives=float("nan"),
                num_actual_positives=num_actual_positives,
            )
        return cls(
            num_true_positives=round(recall * num_actual_positives),
            num_actual_positives=num_actual_positives,
        )

    def to_dict(self, prefix: str = "", suffix: str = "") -> dict[str, int | float]:
        return {
            f"{prefix}recall{suffix}": self.recall,
            f"{prefix}num_true_positives{suffix}": self.num_true_positives,
            f"{prefix}num_actual_positives{suffix}": self.num_actual_positives,
        }

    def to_display(self) -> str:
        if self.num_actual_positives == 0:
            return f"{self.__class__.__qualname__}: no predictions"
        score = self.recall
        bar = make_robust_bar(score, length=20)
        score_str = "nan" if math.isnan(score) else f"{score:.4f}"
        tp_str = (
            "?"
            if math.isnan(float(self.num_true_positives))
            else f"{int(self.num_true_positives):,}"
        )
        return f"Recall {bar}  {score_str}  ({tp_str}/{int(self.num_actual_positives):,})"


def compute_recall(
    true_positives: float,
    false_negatives: float,
) -> float:
    r"""Compute the recall (sensitivity) score.

    Recall measures the proportion of actual positives that are
    correctly identified.

    Args:
        true_positives: The number of true positives, or ``nan``.
        false_negatives: The number of false negatives, or ``nan``.

    Returns:
        The ratio ``true_positives / (true_positives + false_negatives)``.
            Returns ``nan`` when either ``true_positives`` or
            ``false_negatives`` is ``nan``. Returns ``0.0`` when
            ``true_positives + false_negatives`` is ``0``.

    Example:
        ```pycon
        >>> from metriclab.results.classification.recall import compute_recall
        >>> compute_recall(true_positives=3, false_negatives=2)
        0.6
        >>> compute_recall(true_positives=0, false_negatives=0)
        0.0
        >>> compute_recall(true_positives=float("nan"), false_negatives=2)
        nan

        ```
    """
    if math.isnan(true_positives) or math.isnan(false_negatives):
        return float("nan")
    denominator = true_positives + false_negatives
    return true_positives / denominator if denominator > 0 else 0.0
