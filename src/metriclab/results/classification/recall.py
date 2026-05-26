r"""Recall result implementation."""

from __future__ import annotations

__all__ = ["RecallResult", "compute_recall"]

import math
from dataclasses import asdict, dataclass
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
        true_positives: The number of true positives (``TP``).
        false_negatives: The number of false negatives (``FN``).

    Raises:
        ValueError: if ``true_positives`` is negative and not ``nan``.
        ValueError: if ``false_negatives`` is negative and not ``nan``.

    Example:
        ```pycon
        >>> from metriclab.results import RecallResult
        >>> m = RecallResult(true_positives=3, false_negatives=2)
        >>> m
        RecallResult(true_positives=3, false_negatives=2)
        >>> m.recall
        0.6
        >>> m.num_actual_positives
        5
        >>> m.to_dict()
        {'recall': 0.6, 'true_positives': 3, 'false_negatives': 2, 'num_actual_positives': 5}
        >>> print(m.to_display())
        Recall [████████████░░░░░░░░]  0.6000  (3/5)

        ```
    """

    true_positives: int | float
    false_negatives: int | float

    def __post_init__(self) -> None:
        for name, value in (
            ("true_positives", self.true_positives),
            ("false_negatives", self.false_negatives),
        ):
            if not math.isnan(float(value)) and value < 0:
                msg = f"{name} must be >= 0, got {value}"
                raise ValueError(msg)

    @property
    def num_actual_positives(self) -> int | float:
        r"""Return the number of actual positives (``TP + FN``).

        Returns:
            The sum ``true_positives + false_negatives``, or ``nan``
            if either count is ``nan``.
        """
        tp = float(self.true_positives)
        fn = float(self.false_negatives)
        if math.isnan(tp) or math.isnan(fn):
            return float("nan")
        return self.true_positives + self.false_negatives

    @property
    def recall(self) -> float:
        r"""Return the recall score.

        Returns:
            ``true_positives / (true_positives + false_negatives)``.
            Returns ``nan`` when either count is ``nan``. Returns
            ``0.0`` when ``true_positives + false_negatives`` is ``0``.
        """
        return compute_recall(
            true_positives=float(self.true_positives),
            false_negatives=float(self.false_negatives),
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
            f"{prefix}recall{suffix}": self.recall,
            f"{prefix}true_positives{suffix}": self.true_positives,
            f"{prefix}false_negatives{suffix}": self.false_negatives,
            f"{prefix}num_actual_positives{suffix}": self.num_actual_positives,
        }

    def to_display(self) -> str:
        if self.num_actual_positives == 0:
            return f"{self.__class__.__qualname__}: no predictions"
        score = self.recall
        bar = make_robust_bar(score, length=20)
        score_str = "nan" if math.isnan(score) else f"{score:.4f}"
        tp_str = "?" if math.isnan(float(self.true_positives)) else f"{int(self.true_positives):,}"
        ap = self.num_actual_positives
        ap_str = "?" if math.isnan(float(ap)) else f"{int(ap):,}"
        return f"Recall {bar}  {score_str}  ({tp_str}/{ap_str})"

    @classmethod
    def from_recall(
        cls,
        recall: float,
        num_actual_positives: float,
    ) -> Self:
        r"""Create a ``RecallResult`` from a recall score and the number
        of actual positives.

        The number of true positives is back-computed as
        ``round(recall * num_actual_positives)`` and false negatives
        as the remainder.

        Args:
            recall: The recall score, in ``[0, 1]``, or ``nan``.
            num_actual_positives: The number of actual positives
                (``TP + FN``).

        Returns:
            A ``RecallResult`` with computed ``true_positives`` and
                ``false_negatives``.

        Example:
            ```pycon
            >>> from metriclab.results import RecallResult
            >>> m = RecallResult.from_recall(recall=0.6, num_actual_positives=5)
            >>> m.true_positives
            3
            >>> m.false_negatives
            2
            >>> m.recall
            0.6

            ```
        """
        if math.isnan(recall) or math.isnan(float(num_actual_positives)):
            return cls(
                true_positives=float("nan"),
                false_negatives=float("nan"),
            )
        tp = round(recall * num_actual_positives)
        fn = round(num_actual_positives) - tp
        return cls(true_positives=tp, false_negatives=fn)


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
