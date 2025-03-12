# SPDX-FileCopyrightText: Copyright 2025 Martin Lemay <martin.lemay@mines-paris.org>
# SPDX-FileContributor: Martin Lemay
# ruff: noqa: E402 # disable Module level import not at top of file

__doc__ = """
Tests functions for Isoline.py.
"""

import os
import unittest
from typing import Self

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd

from pybend.model.ClPoint import ClPoint
from pybend.model.enumerations import PropertyNames
from pybend.model.Isoline import ChannelCrossSection, Isoline, IsolineType

# output directory for figures
fig_path: str = "tests/.out/"
# create it if absent
if not os.path.exists(fig_path):
    os.makedirs(fig_path)

# inputs
columns = (
    PropertyNames.CURVILINEAR_ABSCISSA.value,
    PropertyNames.CARTESIAN_ABSCISSA.value,
    PropertyNames.CARTESIAN_ORDINATE.value,
    PropertyNames.ELEVATION.value,
    PropertyNames.CURVATURE.value,
    PropertyNames.VELOCITY.value,
    PropertyNames.DEPTH_MEAN.value,
    PropertyNames.DEPTH_MAX.value,
    PropertyNames.WIDTH.value,
    PropertyNames.VELOCITY_PERTURBATION.value,
    "Property1",
    "Property2",
    "Property3",
)
ide1: str = "1"
age1: int = 100
data1: npt.NDArray[np.float64] = np.array(
    [1.2, 152.0, 652.0, 2.2, 0.31, 0.8, 1.1, 4.0, 30.2, 1.3, 2.0, 3.0, 4.0]
)
df: pd.Series = pd.Series(data1, index=columns)
cl_point = ClPoint(ide1, age1, df)

nb_pts = 15

# expected
points_out = [
    [-15.1, 4.0],
    [-12.9429, 2.9388],
    [-10.7857, 2.0408],
    [-8.6286, 1.3061],
    [-6.4714, 0.7347],
    [-4.3143, 0.3265],
    [-2.1571, 0.0816],
    [0.0, 0.0],
    [2.1571, 0.0816],
    [4.3143, 0.3265],
    [6.4714, 0.7347],
    [8.6286, 1.3061],
    [10.7857, 2.0408],
    [12.9429, 2.9388],
    [15.1, 4.0],
]


class TestsIsoline(unittest.TestCase):
    def test_isoline_initialization(self: Self) -> None:
        """Test of Isoline.__init__() function."""
        isoline = Isoline(age1, cl_point, IsolineType.CHANNEL)
        self.assertEqual(isoline.age, age1, "Isoline age was not correctly set.")
        self.assertEqual(
            isoline.cl_pt_ref,
            cl_point,
            "Isoline reference ClPoint was not correctly set.",
        )
        self.assertEqual(
            isoline.isoline_type,
            IsolineType.CHANNEL,
            "Isoline type was not correctly set.",
        )
        self.assertEqual(
            len(isoline.points), 0, "Isoline point list was not correctly set."
        )

    def test_channel_cross_section_initialization(self: Self) -> None:
        """Test of ChannelCrossSection.__init__() function."""
        channel_cs = ChannelCrossSection(age1, cl_point)
        self.assertEqual(
            channel_cs.age, age1, "Channel cross-section age was not correctly set."
        )
        self.assertEqual(
            channel_cs.cl_pt_ref,
            cl_point,
            "Channel cross-section reference ClPoint was not correctly set.",
        )
        self.assertEqual(
            channel_cs.isoline_type,
            IsolineType.CHANNEL,
            "Channel cross-section type was not correctly set.",
        )
        self.assertEqual(
            len(channel_cs.points),
            0,
            "Channel cross-section point list was not correctly set.",
        )

    def test_complete_channel_shape(self: Self) -> None:
        """Test of ChannelCrossSection.complete_channel_shape() function."""
        channel_cs = ChannelCrossSection(age1, cl_point)
        channel_cs.complete_channel_shape(nb_pts)
        self.assertEqual(
            len(channel_cs.points),
            nb_pts,
            "Channel cross-section number of points is wrong.",
        )

        points_obs = [np.round(pt, 4).tolist() for pt in channel_cs.points]

        points = np.array(channel_cs.points)
        plt.figure(dpi=150)
        plt.plot(points[:, 0], points[:, 1], "k-")
        plt.savefig(fig_path + "channel_cross_section.png", dpi=150)

        self.assertSequenceEqual(
            points_obs, points_out, "Channel cross-section list of points is wrong."
        )
