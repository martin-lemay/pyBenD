# SPDX-FileCopyrightText: Copyright 2025 Martin Lemay <martin.lemay@mines-paris.org>
# SPDX-FileContributor: Martin Lemay
# ruff: noqa: E402 # disable Module level import not at top of file

__doc__ = """
Tests functions for Isoline.py.
"""

import os
import unittest
from typing import Optional, Self

import numpy as np
import numpy.typing as npt
import pandas as pd

from pybend.model.ClPoint import ClPoint
from pybend.model.enumerations import PropertyNames
from pybend.model.Isoline import ChannelCrossSection, Isoline
from pybend.model.Section import Section

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
channel_cs = ChannelCrossSection(age1, cl_point)
channel_cs.complete_channel_shape(nb_pts)


ide: str = "Section-1"
bend_id: str = "Bend-1"
pt_start: npt.NDArray[np.float64] = np.array([1.0, 1.0])
pt_stop: npt.NDArray[np.float64] = np.array([10.0, 20.0])
isolines: list[Isoline] = [channel_cs]
same_bend: Optional[list[bool]] = None
flow_dir: npt.NDArray[np.float64] = np.array([1.0, 0.0])


# expected
section_dir_exp: npt.NDArray[np.float64] = pt_start - pt_stop
section_dir_exp /= np.linalg.norm(section_dir_exp)


class TestsSection(unittest.TestCase):
    def test_section_initialization(self: Self) -> None:
        """Test of Isoline.__init__() function."""
        section = Section(
            ide, bend_id, pt_start, pt_stop, isolines, same_bend, flow_dir
        )
        self.assertEqual(section.id, ide, "Id was not correctly set.")
        self.assertEqual(
            section.bend_id, bend_id, "Bend ID was not correctly set."
        )
        self.assertTrue(
            np.array_equal(section.pt_start, pt_start),
            "Start point was not correctly set.",
        )
        self.assertTrue(
            np.array_equal(section.pt_stop, pt_stop),
            "End point was not correctly set.",
        )
        self.assertEqual(
            len(section.isolines),
            len(isolines),
            "Isoline list was not correctly set.",
        )

        self.assertSequenceEqual(
            np.round(section.dir, 4).tolist(),
            np.round(section_dir_exp, 4).tolist(),
            "Section direction was not correctly computed.",
        )


if __name__ == "__main__":
    unittest.main()
