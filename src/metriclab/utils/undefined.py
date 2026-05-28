r"""Validation helpers for undefined-value handling policies."""

from __future__ import annotations

__all__ = ["UndefinedPolicy", "validate_undefined_policy"]

from typing import Any, Literal

UndefinedPolicy = float | Literal["nan", "raise", "warn"]


def validate_undefined_policy(undefined_policy: Any) -> None:
    r"""Validate an ``undefined_policy`` value.

    Args:
        undefined_policy: The policy value to validate. Accepted values:

            - ``float``: substitute the undefined entry with this
              value (e.g. ``0.0``, ``1.0``, ``float('nan')``).
            - ``"nan"``: string alias for ``float('nan')``.
            - ``"raise"``: raise an exception when an undefined value
              is encountered.
            - ``"warn"``: emit a ``UserWarning`` and substitute
              ``0.0``.

    Raises:
        ValueError: if ``undefined_policy`` is not one of the accepted
            values.

    Example:
        ```pycon
        >>> from metriclab.utils.undefined import validate_undefined_policy
        >>> validate_undefined_policy("warn")
        >>> validate_undefined_policy(0.0)
        >>> validate_undefined_policy(float("nan"))

        ```
    """
    if isinstance(undefined_policy, float) or (
        isinstance(undefined_policy, str) and undefined_policy in {"nan", "raise", "warn"}
    ):
        return
    msg = (
        f"Invalid 'undefined_policy' value {undefined_policy!r}. "
        "Expected a float (e.g. 0.0, 1.0, float('nan')) or one of the "
        "string aliases 'nan', 'raise', or 'warn'."
    )
    raise ValueError(msg)
