from __future__ import annotations

import typing

import numpy as np

from metriclab.typing import ArrayLike

###############################
#     Tests for ArrayLike     #
###############################


def test_array_like_includes_numpy() -> None:
    assert np.ndarray in typing.get_args(ArrayLike)


def test_array_like_includes_list() -> None:
    # list[Any] is a generic alias; check the origin
    origins = {typing.get_origin(t) for t in typing.get_args(ArrayLike)}
    assert list in origins


def test_array_like_includes_tuple() -> None:
    origins = {typing.get_origin(t) for t in typing.get_args(ArrayLike)}
    assert tuple in origins
