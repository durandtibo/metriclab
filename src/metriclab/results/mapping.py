r"""Implement a result that is a container for a dict of results."""

from __future__ import annotations

__all__ = ["ResultDict"]

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from coola.equality import objects_are_allclose, objects_are_equal
from coola.utils.format import (
    repr_indent,
    repr_mapping,
    str_indent,
    str_mapping,
)

from metriclab.results.base import BaseResult

if TYPE_CHECKING:
    from typing import Self


@dataclass(frozen=True)
class ResultDict(BaseResult):
    r"""Implement a result that is a container for a dict of results.

    Args:
        results: A mapping from string keys to :class:`BaseResult`
            instances. Each key typically identifies a dataset split or
            evaluation phase (e.g. ``"train"``, ``"val"``).

    Note:
        :meth:`to_markdown` renders one top-level bullet per key and
        nests the child result markdown underneath it. Empty mappings
        return ``"_No results available._"``.

    Example:
        ```pycon
        >>> from metriclab.results import Result, ResultDict
        >>> result = ResultDict({"train": Result({"loss": 0.5}), "val": Result({"loss": 0.3})})
        >>> result.to_dict()
        {'train': {'loss': 0.5}, 'val': {'loss': 0.3}}
        >>> print(result.to_display())
        === train ===
        {'loss': 0.5}

        === val ===
        {'loss': 0.3}

        ```
    """

    results: dict[str, BaseResult]

    def __repr__(self) -> str:
        args = repr_indent(repr_mapping(self.results))
        if args:
            args = f"\n  {args}\n"
        return f"{self.__class__.__qualname__}({args})"

    def __str__(self) -> str:
        args = str_indent(str_mapping(self.results))
        if args:
            args = f"\n  {args}\n"
        return f"{self.__class__.__qualname__}({args})"

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
            self.results,
            other.results,
            rtol=rtol,
            atol=atol,
            equal_nan=equal_nan,
            show_difference=True,
        )

    def equal(self, other: object, equal_nan: bool = False) -> bool:
        if type(other) is not type(self):
            return False
        return objects_are_equal(self.results, other.results, equal_nan=equal_nan)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(data)

    def to_dict(self, prefix: str = "", suffix: str = "") -> dict[str, Any]:
        return {f"{prefix}{key}{suffix}": value.to_dict() for key, value in self.results.items()}

    def to_display(self) -> str:
        return "\n".join(
            [f"=== {key} ===\n{value.to_display()}\n" for key, value in self.results.items()]
        )
