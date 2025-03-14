# SPDX-FileCopyrightText: Copyright 2025 Martin Lemay <martin.lemay@mines-paris.org>
# SPDX-FileContributor: Martin Lemay
# ruff: noqa: E402 # disable Module level import not at top of file

import os

from pybend.utils.logging import logger

__doc__ = r"""
Defines global parameters such as the number of processors to use and
functions to get and set these parameters.

"""
# number of processors for multiprocessing
NB_PROCS: int = 1


def get_nb_procs() -> int:
    """Get the number of processors.

    Number of processors is the minimum between expected values and the number
    of available processors.

    Returns:
        int: Number of processors.

    """
    procs_count: int = len(os.sched_getaffinity(0))
    if procs_count < NB_PROCS:
        logger.warning(f"Available number of cpu: {procs_count}")
    return min(procs_count, NB_PROCS)


def set_nb_procs(nb: int) -> None:
    """Set the number of desired processors.

    Args:
        nb (int): Number of desired processors.

    """
    global NB_PROCS
    NB_PROCS = nb
