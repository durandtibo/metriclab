r"""Binary confusion matrix result implementation."""

from __future__ import annotations

__all__ = [
    "BinaryConfusionMatrixResult",
    "check_betas",
    "compute_f_beta_score",
    "compute_precision",
    "compute_recall",
    "compute_specificity",
    "f_beta_label",
]

import math
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from coola.equality import objects_are_allclose, objects_are_equal

from metriclab.results.base import BaseResult
from metriclab.results.classification.accuracy import compute_accuracy
from metriclab.utils.format import make_robust_bar

if TYPE_CHECKING:
    from collections.abc import Sequence

CONFUSION_MATRIX_ATTRS = (
    "true_positives",
    "true_negatives",
    "false_positives",
    "false_negatives",
)


def check_betas(betas: Sequence[float]) -> None:
    r"""Check that all beta values are non-negative.

    Args:
        betas: The beta values to check.
    """
    for beta in betas:
        if beta < 0:
            msg = f"beta values must be >= 0, got {beta}"
            raise ValueError(msg)


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
        >>> from metriclab.results.classification.binary_confmat import compute_precision
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
        >>> from metriclab.results.classification.binary_confmat import compute_recall
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


def compute_specificity(
    true_negatives: float,
    false_positives: float,
) -> float:
    r"""Compute the specificity (true negative rate) score.

    Specificity measures the proportion of actual negatives that are
    correctly identified.

    Args:
        true_negatives: The number of true negatives, or ``nan``.
        false_positives: The number of false positives, or ``nan``.

    Returns:
        The ratio ``true_negatives / (true_negatives + false_positives)``.
        Returns ``nan`` when either ``true_negatives`` or
        ``false_positives`` is ``nan``. Returns ``0.0`` when
        ``true_negatives + false_positives`` is ``0``.

    Example:
        ```pycon
        >>> from metriclab.results.classification.binary_confmat import compute_specificity
        >>> compute_specificity(true_negatives=4, false_positives=1)
        0.8
        >>> compute_specificity(true_negatives=0, false_positives=0)
        0.0
        >>> compute_specificity(true_negatives=float("nan"), false_positives=1)
        nan

        ```
    """
    if math.isnan(true_negatives) or math.isnan(false_positives):
        return float("nan")
    denominator = true_negatives + false_positives
    return true_negatives / denominator if denominator > 0 else 0.0


def compute_f_beta_score(precision: float, recall: float, beta: float) -> float:
    r"""Compute the F-beta score.

    The F-beta score is the weighted harmonic mean of precision and
    recall. ``beta=1`` gives equal weight to precision and recall
    (F1 score), ``beta<1`` weights precision more, and ``beta>1``
    weights recall more.

    Args:
        precision: The precision score.
        recall: The recall score.
        beta: The beta value. Must be non-negative.

    Returns:
        The F-beta score. Returns ``nan`` when either ``precision``
        or ``recall`` is ``nan``. Returns ``0.0`` when both
        ``precision`` and ``recall`` are ``0.0``.

    Raises:
        ValueError: if ``beta`` is negative.

    Example:
        ```pycon
        >>> from metriclab.results.classification.binary_confmat import compute_f_beta_score
        >>> compute_f_beta_score(precision=0.75, recall=0.6, beta=1.0)
        0.6666666666666665
        >>> compute_f_beta_score(precision=0.75, recall=0.6, beta=0.5)
        0.7142857142857143
        >>> compute_f_beta_score(precision=0.75, recall=0.6, beta=2.0)
        0.625
        >>> compute_f_beta_score(precision=0.0, recall=0.0, beta=1.0)
        0.0

        ```
    """
    if beta < 0:
        msg = f"beta must be >= 0, got {beta}"
        raise ValueError(msg)
    if math.isnan(precision) or math.isnan(recall):
        return float("nan")
    beta_sq = beta**2
    denominator = beta_sq * precision + recall
    return (1 + beta_sq) * (precision * recall) / denominator if denominator > 0 else 0.0


def f_beta_label(beta: float, label: str = "F") -> str:
    r"""Return the label for an F-beta score.

    Integer beta values are formatted without a decimal point
    (e.g. ``1.0`` → ``'F1'``), while non-integer beta values
    use ``g`` formatting to strip trailing zeros
    (e.g. ``0.5`` → ``'F0.5'``).

    Args:
        beta: The beta value to format.
        label: The prefix to use for the label. Defaults to ``'F'``.

    Returns:
        A string label of the form ``'{label}{beta}'``.

    Example:
        ```pycon
        >>> from metriclab.results.classification.binary_confmat import f_beta_label
        >>> f_beta_label(1.0)
        'F1'
        >>> f_beta_label(2.0)
        'F2'
        >>> f_beta_label(0.5)
        'F0.5'
        >>> f_beta_label(1.5)
        'F1.5'
        >>> f_beta_label(1.0, label="f")
        'f1'
        >>> f_beta_label(0.5, label="beta")
        'beta0.5'

        ```
    """
    return f"{label}{int(beta)}" if beta == int(beta) else f"{label}{beta:g}"


@dataclass(frozen=True)
class BinaryConfusionMatrixResult(BaseResult):
    r"""Store aggregated values from a binary confusion matrix used to
    compute classification metrics including accuracy, precision,
    recall, specificity, and F-beta scores.

    Use :meth:`from_confusion_matrix` to construct an instance from
    raw confusion matrix counts. Any count set to ``nan`` propagates
    to all derived metrics.

    Attributes:
        true_positives: The number of true positives, or ``nan``.
        true_negatives: The number of true negatives, or ``nan``.
        false_positives: The number of false positives, or ``nan``.
        false_negatives: The number of false negatives, or ``nan``.
        num_predictions: The total number of predictions, or ``nan``.
        num_correct_predictions: The number of correct predictions,
            or ``nan``.
        accuracy: The accuracy score, or ``nan``.
        precision: The precision score, or ``nan``.
        recall: The recall score, or ``nan``.
        specificity: The specificity score, or ``nan``.
        f_beta_scores: A mapping of beta values to F-beta scores.

    Example:
        ```pycon
        >>> from metriclab.results import BinaryConfusionMatrixResult
        >>> m = BinaryConfusionMatrixResult.from_confusion_matrix(
        ...     true_positives=3,
        ...     true_negatives=4,
        ...     false_positives=1,
        ...     false_negatives=2,
        ... )
        >>> m
        BinaryConfusionMatrixResult(true_positives=3, true_negatives=4, false_positives=1, false_negatives=2, num_predictions=10, num_correct_predictions=7, accuracy=0.7, precision=0.75, recall=0.6, specificity=0.8, f_beta_scores={1.0: 0.6666...})
        >>> m.accuracy
        0.7
        >>> m.precision
        0.75
        >>> m.recall
        0.6
        >>> m.specificity
        0.8
        >>> m.f_beta_scores
        {1.0: 0.6666666666666665}
        >>> print(m.to_display())
        Binary Confusion Matrix
        -----------------------
        n=10  TP=3  TN=4  FP=1  FN=2
        Accuracy    [██████████████░░░░░░]  0.7000  (7/10)
        Precision   [███████████████░░░░░]  0.7500  (3/4)
        Recall      [████████████░░░░░░░░]  0.6000  (3/5)
        Specificity [████████████████░░░░]  0.8000  (4/5)
        F1          [█████████████░░░░░░░]  0.6667

        >>> # NaN propagates to derived metrics
        >>> m_nan = BinaryConfusionMatrixResult.from_confusion_matrix(
        ...     true_positives=float("nan"),
        ...     true_negatives=4,
        ...     false_positives=1,
        ...     false_negatives=2,
        ... )
        >>> m_nan.accuracy
        nan

        ```
    """

    true_positives: int | float
    true_negatives: int | float
    false_positives: int | float
    false_negatives: int | float
    num_predictions: int | float
    num_correct_predictions: int | float
    accuracy: float
    precision: float
    recall: float
    specificity: float
    f_beta_scores: dict[float, float]

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
            asdict(self), asdict(other), atol=atol, rtol=rtol, equal_nan=equal_nan
        )

    def equal(self, other: object, equal_nan: bool = False) -> bool:
        if type(other) is not type(self):
            return False
        return objects_are_equal(asdict(self), asdict(other), equal_nan=equal_nan)

    def to_dict(self, prefix: str = "", suffix: str = "") -> dict[str, int | float]:
        out: dict[str, int | float] = {
            f"{prefix}accuracy{suffix}": self.accuracy,
            f"{prefix}precision{suffix}": self.precision,
            f"{prefix}recall{suffix}": self.recall,
            f"{prefix}specificity{suffix}": self.specificity,
        }
        for beta, score in self.f_beta_scores.items():
            out[f"{prefix}{f_beta_label(beta, label='f')}{suffix}"] = score
        out.update(
            {
                f"{prefix}num_correct_predictions{suffix}": self.num_correct_predictions,
                f"{prefix}num_predictions{suffix}": self.num_predictions,
                f"{prefix}true_positives{suffix}": self.true_positives,
                f"{prefix}true_negatives{suffix}": self.true_negatives,
                f"{prefix}false_positives{suffix}": self.false_positives,
                f"{prefix}false_negatives{suffix}": self.false_negatives,
            }
        )
        return out

    def to_display(self) -> str:
        header = "Binary Confusion Matrix"
        separator = "-" * len(header)
        summary = (
            f"n={self.num_predictions:,}  "
            f"TP={self.true_positives:,}  "
            f"TN={self.true_negatives:,}  "
            f"FP={self.false_positives:,}  "
            f"FN={self.false_negatives:,}"
        )

        tp, tn, fp, fn = (
            self.true_positives,
            self.true_negatives,
            self.false_positives,
            self.false_negatives,
        )
        metrics: list[tuple[str, float, tuple[int | float, int | float] | None]] = [
            ("Accuracy", self.accuracy, (self.num_correct_predictions, self.num_predictions)),
            ("Precision", self.precision, (tp, tp + fp)),
            ("Recall", self.recall, (tp, tp + fn)),
            ("Specificity", self.specificity, (tn, tn + fp)),
            *[(f_beta_label(beta), score, None) for beta, score in self.f_beta_scores.items()],
        ]

        metric_lines = []
        for name, value, counts in metrics:
            line = f"{name:<11} {make_robust_bar(value, length=20)}  {value:.4f}"
            if counts is not None:
                numerator, denominator = counts
                line += f"  ({numerator:,}/{denominator:,})"
            metric_lines.append(line)

        metric_text = "\n".join(metric_lines)
        return f"{header}\n{separator}\n{summary}\n{metric_text}"

    def combine(self, other: BinaryConfusionMatrixResult) -> BinaryConfusionMatrixResult:
        if not isinstance(other, BinaryConfusionMatrixResult):
            msg = f"Cannot combine {self.__class__.__qualname__} with {type(other)}"
            raise TypeError(msg)
        return BinaryConfusionMatrixResult.from_confusion_matrix(
            true_positives=self.true_positives + other.true_positives,
            true_negatives=self.true_negatives + other.true_negatives,
            false_positives=self.false_positives + other.false_positives,
            false_negatives=self.false_negatives + other.false_negatives,
            betas=list(self.f_beta_scores.keys()),
        )

    @classmethod
    def from_confusion_matrix(
        cls,
        true_positives: float,
        true_negatives: float,
        false_positives: float,
        false_negatives: float,
        betas: Sequence[float] = (1.0,),
    ) -> BinaryConfusionMatrixResult:
        r"""Create a result from raw confusion matrix counts.

        Any count set to ``nan`` propagates to all derived metrics
        (``num_predictions``, ``num_correct_predictions``,
        ``accuracy``, ``precision``, ``recall``, ``specificity``,
        and all F-beta scores).

        Args:
            true_positives: The number of true positives, or ``nan``.
            true_negatives: The number of true negatives, or ``nan``.
            false_positives: The number of false positives, or ``nan``.
            false_negatives: The number of false negatives, or ``nan``.
            betas: The beta values for F-beta score computation.
                Defaults to ``(1.0,)`` which gives the F1 score.

        Returns:
            A fully populated ``BinaryConfusionMatrixResult``.

        Raises:
            ValueError: if any non-NaN count is negative.
            ValueError: if any beta value is negative.

        Example:
            ```pycon
            >>> from metriclab.results import BinaryConfusionMatrixResult
            >>> m = BinaryConfusionMatrixResult.from_confusion_matrix(
            ...     true_positives=3,
            ...     true_negatives=4,
            ...     false_positives=1,
            ...     false_negatives=2,
            ...     betas=[0.5, 1.0, 2.0],
            ... )
            >>> m.f_beta_scores
            {0.5: 0.7142857142857143, 1.0: 0.6666666666666665, 2.0: 0.625}
            >>> import math
            >>> m_nan = BinaryConfusionMatrixResult.from_confusion_matrix(
            ...     true_positives=float("nan"),
            ...     true_negatives=4,
            ...     false_positives=1,
            ...     false_negatives=2,
            ... )
            >>> math.isnan(m_nan.precision)
            True

            ```
        """
        for name, value in (
            ("true_positives", true_positives),
            ("true_negatives", true_negatives),
            ("false_positives", false_positives),
            ("false_negatives", false_negatives),
        ):
            if not math.isnan(value) and value < 0:
                msg = f"{name} must be >= 0, got {value}"
                raise ValueError(msg)
        check_betas(betas)

        num_predictions = true_positives + true_negatives + false_positives + false_negatives
        num_correct_predictions = true_positives + true_negatives
        precision = compute_precision(
            true_positives=true_positives,
            false_positives=false_positives,
        )
        recall = compute_recall(
            true_positives=true_positives,
            false_negatives=false_negatives,
        )

        return cls(
            true_positives=true_positives,
            true_negatives=true_negatives,
            false_positives=false_positives,
            false_negatives=false_negatives,
            num_predictions=num_predictions,
            num_correct_predictions=num_correct_predictions,
            accuracy=compute_accuracy(
                num_correct_predictions=num_correct_predictions, num_predictions=num_predictions
            ),
            precision=precision,
            recall=recall,
            specificity=compute_specificity(
                true_negatives=true_negatives,
                false_positives=false_positives,
            ),
            f_beta_scores={
                beta: compute_f_beta_score(precision=precision, recall=recall, beta=beta)
                for beta in betas
            },
        )
