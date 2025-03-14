# SPDX-FileCopyrightText: Copyright 2025 Martin Lemay <martin.lemay@mines-paris.org>
# SPDX-FileContributor: Martin Lemay
# ruff: noqa: E402 # disable Module level import not at top of file

from enum import Enum
from typing import Self

import numpy as np
import numpy.typing as npt

from pybend.model.ClPoint import ClPoint

__doc__ = r"""
Isoline module define Isoline abstract object and children objects including
ChannelCrossSection. Children objects specified the type IsolineType and the
geometry from parametric function.

An Isoline object corresponds to a line defined by an isovalue (such as the
age). ChannelCrossSection object allows to represent channel section geometry
from a reference point and a paramteric shape.

"""


class IsolineType(Enum):
    """Types of isoline."""

    #: channel cross-section
    CHANNEL = "Channel"
    #: undefined
    UNKNOWN = "Unknown"


class Isoline:
    def __init__(
        self: Self, age: int, cl_pt_ref: ClPoint, isoline_type: IsolineType
    ) -> None:
        """Store points of the same age (for instance channel cross-section).

        Args:
            age (int): age of the points
            cl_pt_ref (ClPoint): reference ClPoint
            isoline_type (str): isoline type (currently only 'Channel')
        """
        self.age: int = age
        self.cl_pt_ref: ClPoint = cl_pt_ref
        #: point coordinates according to cl_pt_ref
        self.points: npt.NDArray[np.float64] = np.empty(0)
        # isoline type
        self.isoline_type: IsolineType = isoline_type


class ChannelCrossSection(Isoline):
    def __init__(self: Self, age: int, cl_pt_ref: ClPoint) -> None:
        """Isoline for channel cross-section.

        Args:
            age (int): age of the points
            cl_pt_ref (npt.NDArray[np.float64]): reference ClPoint
        """
        super().__init__(age, cl_pt_ref, IsolineType.CHANNEL)

    def complete_channel_shape(self: Self, nb_pts: int = 11) -> None:
        """Create channel cross-section assuming parabolic shape.

        Args:
            nb_pts (int, optional): Number of points.

                Defaults to 11.
        """
        # to get an odd number
        if nb_pts % 2 == 0:
            nb_pts += 1
        Xparabol: npt.NDArray[np.float64] = np.linspace(-1.0, 1.0, nb_pts)

        Yparabol: npt.NDArray[np.float64] = (
            Xparabol * Xparabol * self.cl_pt_ref.depth_max()
        )
        Xparabol *= self.cl_pt_ref.width() / 2.0
        self.points = np.column_stack((Xparabol, Yparabol))
