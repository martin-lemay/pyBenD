# SPDX-FileCopyrightText: Copyright 2025 Martin Lemay <martin.lemay@mines-paris.org>
# SPDX-FileContributor: Martin Lemay
# ruff: noqa: E402 # disable Module level import not at top of file

__doc__ = """
Tests functions for Isoline.py.
"""

import os
import unittest
import warnings
from typing import Optional, Self

import numpy as np
import numpy.typing as npt
import pandas as pd

from pybend.model.ClPoint import ClPoint
from pybend.model.enumerations import PropertyNames
from pybend.model.Isoline import ChannelCrossSection, Isoline
from pybend.model.Section import Section, StackingPatternType

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
section_dir_exp: npt.NDArray[np.float64] = pt_stop - pt_start
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


def _make_isoline(
    age: int, x: float, y: float, z: float, width: float = 10.0
) -> ChannelCrossSection:
    """Create a channel cross-section at the given location."""
    cols = (
        PropertyNames.CURVILINEAR_ABSCISSA.value,
        PropertyNames.CARTESIAN_ABSCISSA.value,
        PropertyNames.CARTESIAN_ORDINATE.value,
        PropertyNames.ELEVATION.value,
        PropertyNames.DEPTH_MAX.value,
        PropertyNames.WIDTH.value,
    )
    cl_pt = ClPoint(
        str(age), age, pd.Series([0.0, x, y, z, 2.0, width], index=cols)
    )
    isoline = ChannelCrossSection(age, cl_pt)
    isoline.complete_channel_shape(5)
    return isoline


def _make_section(
    lateral: list[float], vertical: Optional[list[float]] = None
) -> Section:
    """Create a section whose isolines migrate along y (flow along x)."""
    if vertical is None:
        vertical = [0.0] * len(lateral)
    isos: list[Isoline] = [
        _make_isoline(i, 0.0, y, z)
        for i, (y, z) in enumerate(zip(lateral, vertical, strict=True))
    ]
    return Section(
        "S", "B", np.array([0.0, -10.0]), np.array([0.0, 10.0]), isos
    )


class TestsSectionFixes(unittest.TestCase):
    """Tests of Section fixes on synthetic isolines."""

    def test_direction_from_start_to_stop(self: Self) -> None:
        """Section direction goes from start to stop with int inputs."""
        section = Section("S", "B", np.array([0, 0]), np.array([3, 4]), [])
        self.assertSequenceEqual(np.round(section.dir, 4).tolist(), [0.6, 0.8])

    def test_origin_uses_horizontal_distance(self: Self) -> None:
        """Lateral offset ignores aggradation."""
        isos: list[Isoline] = [
            _make_isoline(0, 0.0, 0.0, 0.0),
            _make_isoline(1, 0.0, 5.0, 100.0),
            _make_isoline(2, 0.0, -3.0, 200.0),
        ]
        section = Section(
            "S", "B", np.array([0.0, 0.0]), np.array([0.0, 1.0]), isos
        )
        self.assertListEqual(
            [tuple(map(float, o)) for o in section.isolines_origin],
            [(0.0, 0.0), (5.0, 100.0), (-3.0, 200.0)],
        )

    def test_empty_and_single_isoline(self: Self) -> None:
        """Sections with less than 2 isolines do not raise."""
        section = Section(
            "S", "B", np.array([0.0, 0.0]), np.array([1.0, 0.0]), []
        )
        self.assertListEqual(section.isolines_origin, [])
        self.assertEqual(
            section.get_stacking_pattern_type(1.0),
            StackingPatternType.UNDEFINED,
        )
        section = _make_section([0.0])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.assertEqual(
                section.get_stacking_pattern_type(1.0),
                StackingPatternType.UNDEFINED,
            )

    def test_stacking_pattern_merge_groups(self: Self) -> None:
        """Groups split by an isolated step are merged."""
        # steps: [+1] * 5 + [-1] + [+1] * 5
        lateral = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 40.0]
        lateral += [50.0, 60.0, 70.0, 80.0, 90.0]
        section = _make_section(lateral)
        self.assertEqual(
            section.get_stacking_pattern_type(1.0),
            StackingPatternType.ONE_WAY,
        )
        # steps: [+1] * 5 + [-1] + [+1] * 5 + [-1] * 5
        lateral += [80.0, 70.0, 60.0, 50.0, 40.0]
        section = _make_section(lateral)
        self.assertEqual(
            section.get_stacking_pattern_type(1.0),
            StackingPatternType.TWO_WAYS,
        )

    def test_local_stratigraphic_number_null_aggradation(self: Self) -> None:
        """Null vertical displacement gives inf/NaN without warnings."""
        section = _make_section([0.0, 5.0, 5.0, 2.0, 4.0], [0, 0, 0, 0, 2])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            section.channel_apparent_displacements()
        # steps: (+5, 0), (0, 0), (-3, 0), (+2, +2)
        np.testing.assert_array_equal(
            section.local_disp[:, 2], [np.inf, np.nan, -np.inf, 1.0]
        )
        section = _make_section([0.0, 0.0], [0.0, 0.0])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            section.channel_apparent_displacements()
        self.assertTrue(np.isnan(section.local_disp[0, 2]))

    def test_average_disp_without_aggradation(self: Self) -> None:
        """Belt thickness is defined when Dz is null."""
        section = _make_section([0.0, 10.0, 20.0])
        section.get_stacking_pattern_type(1.0)
        metrics = section._compute_average_disp(10.0, 2.0, True)
        self.assertEqual(metrics[0], 20.0)
        self.assertEqual(metrics[1], 0.0)
        self.assertEqual(metrics[2], 30.0)
        self.assertEqual(metrics[3], 2.0)
        self.assertEqual(metrics[4], 3.0)
        self.assertEqual(metrics[5], 1.0)
        self.assertEqual(metrics[6], 15.0)
        self.assertTrue(np.isnan(metrics[7]))

    def test_write_results(self: Self) -> None:
        """Averaged displacements are written in a file."""
        section = _make_section([0.0, 10.0, 20.0], [0.0, 1.0, 2.0])
        section.get_stacking_pattern_type(1.0)
        filepath: str = fig_path + "section_averaged_disp.csv"
        if os.path.exists(filepath):
            os.remove(filepath)
        section.section_averaged_channel_displacements(
            10.0, 2.0, True, filepath
        )
        with open(filepath) as fin:
            lines = fin.read().splitlines()
        self.assertEqual(len(lines), 2)
        for line in lines:
            values = [float(v) for v in line.split(";")]
            self.assertEqual(len(values), 6)
        self.assertListEqual(
            [float(v) for v in lines[0].split(";")],
            [30.0, 4.0, 3.0, 2.0, 7.5, 2.0],
        )


if __name__ == "__main__":
    unittest.main()
