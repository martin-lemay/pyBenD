# SPDX-FileCopyrightText: Copyright 2025 Martin Lemay <martin.lemay@mines-paris.org>
# SPDX-FileContributor: Martin Lemay
import logging
from typing import Optional

from typing_extensions import Self

__doc__ = """
Logging module manages logging tools.

Code was modified from <https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output>
"""

# types redefinition to import logging.* from this module
Logger = logging.Logger  #: logger type


DEBUG: int = logging.DEBUG
INFO: int = logging.INFO
WARNING: int = logging.WARNING
ERROR: int = logging.ERROR
CRITICAL: int = logging.CRITICAL


class CustomLoggerFormatter(logging.Formatter):
    """Custom formatter for the logger.

    To use it:

    .. code-block:: python

        logger = logging.getLogger("Logger name")
        ch = logging.StreamHandler()
        ch.setFormatter(CustomLoggerFormatter())
        logger.addHandler(ch)
    """

    # define color codes
    grey: str = "\x1b[38;20m"
    yellow: str = "\x1b[33;20m"
    red: str = "\x1b[31;20m"
    bold_red: str = "\x1b[31;1m"
    reset: str = "\x1b[0m"

    # define prefix of log messages
    format1: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    format2: str = (
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    )

    #: format for each logger output type
    FORMATS: dict[int, str] = {
        DEBUG: grey + format2 + reset,
        INFO: grey + format1 + reset,
        WARNING: yellow + format1 + reset,
        ERROR: red + format1 + reset,
        CRITICAL: bold_red + format2 + reset,
    }

    def format(self: Self, record: logging.LogRecord) -> str:
        """Return the format according to input record.

        Parameters:
        ----------
            record (logging.LogRecord): record

        Returns:
        ----------
            str: format as a string
        """
        log_fmt: Optional[str] = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def getLogger(title: str) -> Logger:
    """Return the Logger with pre-defined configuration.

    Example:

    .. code-block:: python

        # module import
        import Logger

        # logger instanciation
        logger :Logger.Logger = Logger.getLogger("My application")

        # logger use
        logger.debug("debug message")
        logger.info("info message")
        logger.warning("warning message")
        logger.error("error message")
        logger.critical("critical message")

    Parameters:
        ----------
        title (str): Name of the logger.

    Returns:
        ----------
        Logger: logger
    """
    logger: Logger = logging.getLogger(title)
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setFormatter(CustomLoggerFormatter())
    logger.addHandler(ch)
    return logger


logger = getLogger("pyBenD")
