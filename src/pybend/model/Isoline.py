# SPDX-FileCopyrightText: Copyright 2025 Martin Lemay <martin.lemay@mines-paris.org>
# SPDX-FileContributor: Martin Lemay
# ruff: noqa: E402 # disable Module level import not at top of file

"""Isoline models.

Defines `Isoline` and derived types such as channel cross-sections. Children
objects specified the type IsolineType and the geometry from parametric
function.

An Isoline object corresponds to a line defined by an isovalue (such as the
age). ChannelCrossSection object allows to represent channel section geometry
from a reference point and a paramteric shape.
"""

from enum import Enum
from typing import Self

import numpy as np
import numpy.typing as npt

from pybend.model.ClPoint import ClPoint


class IsolineType(Enum):
    """Types of isoline."""

    #: channel cross-section
    CHANNEL = "Channel"
    #: undefined
    UNKNOWN = "Unknown"


class Isoline:
    """A polyline associated with a constant isovalue (e.g., age)."""

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
    """A channel cross-section isoline derived from a reference `ClPoint`."""

    def __init__(self: Self, age: int, cl_pt_ref: ClPoint) -> None:
        """Isoline for channel cross-section.

        Args:
            age (int): age of the points
            cl_pt_ref (npt.NDArray[np.float64]): reference ClPoint
        """
        super().__init__(age, cl_pt_ref, IsolineType.CHANNEL)

    def complete_channel_shape(self: Self, nb_pts: int = 11) -> None:
        """Create channel cross-section assuming parabolic shape.

        The number of points is made odd so that the middle point is the
        channel reference point (bottom of the parabola). With a single
        point, only the reference point is created.

        Args:
            nb_pts (int, optional): Number of points (strictly positive).
                Defaults to 11.
        """
        assert nb_pts > 0, "Number of points must be strictly positive."
        # to get an odd number
        if nb_pts % 2 == 0:
            nb_pts += 1
        # np.linspace(-1, 1, 1) returns [-1] (channel bank), so handle the
        # single point case to keep the reference point at the middle
        Xparabol: npt.NDArray[np.float64] = (
            np.linspace(-1.0, 1.0, nb_pts) if nb_pts > 1 else np.zeros(1)
        )

        Yparabol: npt.NDArray[np.float64] = (
            Xparabol * Xparabol * self.cl_pt_ref.depth_max()
        )
        Xparabol *= self.cl_pt_ref.width() / 2.0
        self.points = np.column_stack((Xparabol, Yparabol))
