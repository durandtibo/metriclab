r"""Helpers to configure and format logging output."""

from __future__ import annotations

__all__ = ["configure_logging", "log_dict_pretty"]

import logging
from typing import Any

from metriclab.utils.imports import is_colorlog_available, is_rich_available

if is_colorlog_available():  # pragma: no cover
    import colorlog
if is_rich_available():  # pragma: no cover
    from rich.console import Console
    from rich.panel import Panel
    from rich.pretty import Pretty

logger: logging.Logger = logging.getLogger(__name__)


def configure_logging(level: int = logging.INFO) -> None:
    r"""Configure the logging module with a colored formatter.

    If the ``colorlog`` package is installed, a colored formatter is
    used. Otherwise, the standard ``logging.basicConfig`` is called.

    Args:
        level: The minimum log level to capture. Defaults to
            ``logging.INFO``.

    Example:
        ```pycon
        >>> import logging
        >>> from metriclab.utils.logging import configure_logging
        >>> configure_logging(level=logging.DEBUG)

        ```
    """
    if not is_colorlog_available():
        logging.basicConfig(level=level)
        return

    handler = colorlog.StreamHandler()
    formatter = colorlog.ColoredFormatter(
        fmt=(
            "%(log_color)s(%(process)d) %(asctime)s [%(levelname)s] %(name)s:%(lineno)s%(reset)s "
            "%(message_log_color)s%(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "bold_yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        },
        secondary_log_colors={
            "message": {
                "DEBUG": "cyan",
                "INFO": "reset",
                "WARNING": "bold_yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            }
        },
    )
    handler.setFormatter(formatter)

    logging.basicConfig(level=level, handlers=[handler])


def log_dict_pretty(
    data: dict[Any, Any], level: int = logging.INFO, title: str | None = None
) -> None:
    r"""Log a dictionary in a pretty format if rich is available.

    If the ``rich`` package is installed, the dictionary is rendered
    using :class:`~rich.pretty.Pretty` and printed to the console via
    :class:`~rich.console.Console`. Otherwise, the dictionary is logged
    with the standard :mod:`logging` module at the specified level.

    Args:
        data: The dictionary to log.
        level: The log level used when ``rich`` is not available.
            Defaults to ``logging.INFO``.
        title: Optional panel title when ``rich`` is available, and a
            text prefix when ``rich`` is unavailable.

    Example:
        ```pycon
        >>> from metriclab.utils.logging import log_dict_pretty
        >>> log_dict_pretty({"accuracy": 0.75}, title="Validation")

        ```
    """
    if is_rich_available():
        console = Console()
        console.print(Panel(Pretty(data), title=title))
    else:
        if title:
            data = f"{title}:\n{data}"
        logger.log(level, data)
