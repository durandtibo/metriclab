r"""Multiclass precision result implementation."""

from __future__ import annotations

__all__ = ["MulticlassPrecisionResult"]

import math
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from coola.equality import objects_are_allclose, objects_are_equal

from metriclab.results.base import BaseResult
from metriclab.utils.format import make_robust_bar

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True)
class MulticlassPrecisionResult(BaseResult):
    r"""Store aggregated values for multiclass classification precision.

    Stores per-class precision scores alongside macro, micro, and
    weighted averages, consistent with
    :func:`sklearn.metrics.precision_score`.

    Attributes:
        macro_precision: Unweighted mean of per-class precision scores.
        micro_precision: Global precision computed from total
            true/false positives across all classes.
        weighted_precision: Per-class precision weighted by class
            support (number of actual instances per class).
        per_class_precision: A 1D array of per-class precision scores,
            one entry per class.
        support: A 1D array of per-class support counts (number of
            actual instances per class), same length as
            ``per_class_precision``.
        num_predictions: The total number of predictions.

    Raises:
        ValueError: if ``num_predictions`` is negative.
        ValueError: if ``macro_precision`` is negative and not ``nan``.
        ValueError: if ``micro_precision`` is negative and not ``nan``.
        ValueError: if ``weighted_precision`` is negative and not
            ``nan``.
        ValueError: if ``per_class_precision`` and ``support`` have
            different shapes.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from metriclab.results import MulticlassPrecisionResult
        >>> m = MulticlassPrecisionResult(
        ...     macro_precision=0.7,
        ...     micro_precision=0.72,
        ...     weighted_precision=0.71,
        ...     per_class_precision=np.array([0.8, 0.6, 0.7]),
        ...     support=np.array([100, 50, 150]),
        ...     num_predictions=300,
        ... )
        >>> m.macro_precision
        0.7
        >>> m.to_dict()  # doctest: +NORMALIZE_WHITESPACE
        {'macro_precision': 0.7, 'micro_precision': 0.72,
         'weighted_precision': 0.71, 'per_class_precision': [0.8, 0.6, 0.7],
         'support': [100, 50, 150], 'num_predictions': 300}

        ```
    """

    macro_precision: float
    micro_precision: float
    weighted_precision: float
    per_class_precision: np.ndarray
    support: np.ndarray
    num_predictions: int

    def __post_init__(self) -> None:
        if self.num_predictions < 0:
            msg = f"num_predictions must be >= 0, got {self.num_predictions}"
            raise ValueError(msg)
        for name, value in (
            ("macro_precision", self.macro_precision),
            ("micro_precision", self.micro_precision),
            ("weighted_precision", self.weighted_precision),
        ):
            if not math.isnan(value) and value < 0:
                msg = f"{name} must be >= 0, got {value}"
                raise ValueError(msg)
        if self.per_class_precision.shape != self.support.shape:
            msg = (
                f"per_class_precision and support must have the same shape, "
                f"got {self.per_class_precision.shape} and {self.support.shape}"
            )
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
            show_difference=True,
        )

    def equal(self, other: object, equal_nan: bool = False) -> bool:
        if type(other) is not type(self):
            return False
        return objects_are_equal(asdict(self), asdict(other), equal_nan=equal_nan)

    def to_dict(
        self, prefix: str = "", suffix: str = ""
    ) -> dict[str, int | float | list[int | float]]:
        return {
            f"{prefix}macro_precision{suffix}": self.macro_precision,
            f"{prefix}micro_precision{suffix}": self.micro_precision,
            f"{prefix}weighted_precision{suffix}": self.weighted_precision,
            f"{prefix}per_class_precision{suffix}": self.per_class_precision.tolist(),
            f"{prefix}support{suffix}": self.support.tolist(),
            f"{prefix}num_predictions{suffix}": self.num_predictions,
        }

    def to_display(self) -> str:
        if self.num_predictions == 0:
            return f"{self.__class__.__qualname__}: no predictions"
        return (
            f"Precision (n={self.num_predictions:,})\n{'-' * 52}\n"
            + _precision_row("Macro", self.macro_precision)
            + _precision_row("Micro", self.micro_precision)
            + _precision_row("Weighted", self.weighted_precision)
            + "\nPer class:\n"
            + "".join(
                _precision_row(label=f"  class {i}", score=float(p), support=int(s))
                for i, (p, s) in enumerate(zip(self.per_class_precision, self.support, strict=True))
            ).rstrip("\n")
        )


def _precision_row(label: str, score: float, support: int | None = None) -> str:
    r"""Format a single precision row for terminal display.

    Args:
        label: The row label.
        score: The precision score.
        support: Optional support count shown at the end of the row.

    Returns:
        A formatted string row ending with a newline.
    """
    bar = make_robust_bar(score, length=20)
    score_str = "nan" if math.isnan(score) else f"{score:.4f}"
    support_str = f"  (n={support:,})" if support is not None else ""
    return f"{label:<12}{bar}  {score_str}{support_str}\n"
