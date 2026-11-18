# SPDX-FileCopyrightText: Copyright 2025 Martin Lemay <martin.lemay@mines-paris.org>
# SPDX-FileContributor: Martin Lemay
# ruff: noqa: E402 # disable Module level import not at top of file

__doc__ = """
Tests functions for CenterlineCollection class - Run with input data
centerline_Collection_test_data*.csv.
"""

import os
import unittest
import warnings
from typing import Self

import numpy as np
import numpy.typing as npt
import pandas as pd

from pybend.io import (
    CenterlineIOFormat,
    load_centerline_from_file,
)
from pybend.model.Bend import Bend
from pybend.model.Centerline import Centerline
from pybend.model.ClPoint import ClPoint
from pybend.model.enumerations import (
    BendSide,
    MorphometricNames,
    PropertyNames,
)
from pybend.model.Morphometry import Morphometry
from pybend.utils.globalParameters import set_nb_procs

set_nb_procs(1)

# inputs

# output directory for figures
dir_path: str = os.getcwd() + "/tests/data/"
fig_path: str = os.getcwd() + "/tests/.out/"
# create it if absent
if not os.path.exists(fig_path):
    os.makedirs(fig_path)


filepath: str = dir_path + "centerline_flumy2500.csv"
spacing: float = 200  # spacing between channel point (m)
smooth_distance: int = 500  # channel point location smoothing distance (m)
use_fix_nb_points: bool = False
filtering_window: int = 10  # number of points for filtered curvature
sinuo_thres: float = 1.05  # threshold for bends
n = 2  # exponent of curvature distribution function

compute_curvature: bool = True
interpol_props: bool = True
find_bends: bool = True

flow_dir: npt.NDArray[np.float64] = np.array([1.0, 0.0])

nb_procs: int = 3  # number of procs


age, dataset = load_centerline_from_file(
    filepath, kind=CenterlineIOFormat.FLUMY_CSV
)
centerline = Centerline(
    age,
    dataset,
    spacing,
    smooth_distance,
    use_fix_nb_points,
    filtering_window,
    sinuo_thres,
    n,
    compute_curvature=True,
    interpol_props=True,
    find_bends=False,
)
centerline.find_bends(sinuo_thres, 3)

assert centerline.get_nb_bends() == 46, (
    "Number of bends in invalid. Run tests_Centerline.py first."
)

assert centerline.get_nb_valid_bends() == 34, (
    "Number of bends in invalid. Run tests_Centerline.py first."
)


valid_bend_indexes: list[int] = centerline.get_valid_bend_indexes()

# expected results
# names = (
#     "Sinuosity",
#     "Arc_length",
#     "Wavelength",
#     "Sinuosity",
#     "Amplitude",
#     "Extension",
#     "RadiusCurvature",
#     "Asymmetry",
#     "Roundness",
#     "Skewness",
#     "Wavelength_Leopold",
#     "Amplitude_Leopold",
# )
# df1 = pd.DataFrame(expected.T, columns=names)
# df1.to_csv(dir_path + "morphometricResults.csv", index=False)
# df2 = pd.DataFrame(average_metrics_exp.T, columns=names)
# df2.to_csv(dir_path + "averageMorphometricResults.csv", index=False)

metrics_exp = pd.read_csv(dir_path + "morphometricResults.csv")
average_metrics_exp = pd.read_csv(dir_path + "averageMorphometricResults.csv")


class TestsMorphometry(unittest.TestCase):
    def test_initialization(self: Self) -> None:
        """Test of Morphometry initialization."""
        morph: Morphometry = Morphometry(centerline)
        self.assertIsNotNone(morph.centerline)
        self.assertEqual(age, morph.centerline.age)

    def test_compute_bend_arc_length(self: Self) -> None:
        """Test of compute_bend_arc_length method."""
        morph: Morphometry = Morphometry(centerline)
        obs = [
            morph.compute_bend_arc_length(i)
            for i in range(morph.centerline.get_nb_bends())
        ]
        print(obs)
        self.assertSequenceEqual(
            obs,
            metrics_exp["Arc_length"].tolist(),
            "Arc length are not equal.",
        )

    def test_compute_bend_wavelength(self: Self) -> None:
        """Test of compute_bend_wavelength method."""
        morph: Morphometry = Morphometry(centerline)
        obs = [
            morph.compute_bend_wavelength(i)
            for i in range(morph.centerline.get_nb_bends())
        ]
        print(obs)
        self.assertSequenceEqual(
            obs,
            metrics_exp["Wavelength"].tolist(),
            "Wavelength are not equal.",
        )

    def test_compute_bend_sinuosity(self: Self) -> None:
        """Test of compute_bend_sinuosity method."""
        morph: Morphometry = Morphometry(centerline)
        obs = [
            morph.compute_bend_sinuosity(i)
            for i in range(morph.centerline.get_nb_bends())
        ]
        print(obs)
        self.assertSequenceEqual(
            obs, metrics_exp["Sinuosity"].tolist(), "Sinuosity are not equal."
        )

    def test_compute_bend_amplitude(self: Self) -> None:
        """Test of compute_bend_amplitude method."""
        morph: Morphometry = Morphometry(centerline)
        obs = [
            morph.compute_bend_amplitude(i)
            for i in range(morph.centerline.get_nb_bends())
        ]
        print(obs)
        self.assertSequenceEqual(
            obs, metrics_exp["Amplitude"].tolist(), "Amplitude are not equal."
        )

    def test_compute_bend_extension(self: Self) -> None:
        """Test of compute_bend_extension method."""
        morph: Morphometry = Morphometry(centerline)
        obs = [
            morph.compute_bend_extension(i)
            for i in range(morph.centerline.get_nb_bends())
        ]
        print(obs)
        self.assertSequenceEqual(
            obs, metrics_exp["Extension"].tolist(), "Extension are not equal."
        )

    def test_compute_bend_radius(self: Self) -> None:
        """Test of compute_bend_radius method."""
        morph: Morphometry = Morphometry(centerline)
        obs = [
            morph.compute_bend_radius(i)
            for i in range(morph.centerline.get_nb_bends())
        ]
        print(obs)
        self.assertSequenceEqual(
            obs,
            metrics_exp["RadiusCurvature"].tolist(),
            "Radius are not equal.",
        )

    def test_compute_bend_asymmetry_apex(self: Self) -> None:
        """Test of compute_bend_asymmetry_apex method."""
        morph: Morphometry = Morphometry(centerline)
        obs = [
            morph.compute_bend_asymmetry_apex(i)
            for i in range(morph.centerline.get_nb_bends())
        ]
        print(obs)
        self.assertSequenceEqual(
            obs, metrics_exp["Asymmetry"].tolist(), "Asymmetry are not equal."
        )

    def test_compute_bend_roundness(self: Self) -> None:
        """Test of compute_bend_roundness method."""
        morph: Morphometry = Morphometry(centerline)
        obs = [
            morph.compute_bend_roundness(i)
            for i in range(morph.centerline.get_nb_bends())
        ]
        print(obs)
        self.assertSequenceEqual(
            obs, metrics_exp["Roundness"].tolist(), "Roudness are not equal."
        )

    def test_compute_bend_skewness(self: Self) -> None:
        """Test of compute_bend_skewness method."""
        morph: Morphometry = Morphometry(centerline)
        obs = [
            morph.compute_bend_skewness(i)
            for i in range(morph.centerline.get_nb_bends())
        ]
        print(obs)
        self.assertSequenceEqual(
            obs, metrics_exp["Skewness"].tolist(), "Skewness are not equal."
        )

    def test_compute_bend_wavelength_leopold(self: Self) -> None:
        """Test of compute_bend_wavelength_leopold method."""
        morph: Morphometry = Morphometry(centerline)
        obs = [
            morph.compute_bend_wavelength_leopold(i)
            for i in range(1, morph.centerline.get_nb_bends() - 1, 1)
        ]
        self.assertSequenceEqual(
            obs,
            metrics_exp["Wavelength_Leopold"].tolist()[1:-1],
            "Leopold wavelength are not equal.",
        )

    def test_compute_bend_amplitude_leopold(self: Self) -> None:
        """Test of compute_bend_amplitude_leopold method."""
        morph: Morphometry = Morphometry(centerline)
        obs = [
            morph.compute_bend_amplitude_leopold(i)
            for i in range(1, morph.centerline.get_nb_bends() - 1, 1)
        ]
        self.assertSequenceEqual(
            obs,
            metrics_exp["Amplitude_Leopold"].tolist()[1:-1],
            "Leopold amplitude are not equal.",
        )

    def test_compute_bends_morphometry_all(self: Self) -> None:
        """Test of compute_bends_morphometry method."""
        morph: Morphometry = Morphometry(centerline)
        obs = morph.compute_bends_morphometry(valid_bends=False).to_numpy()
        self.assertTrue(np.array_equal(obs, metrics_exp.to_numpy(), True))

    def test_compute_bends_morphometry_valid_bends(self: Self) -> None:
        """Test of compute_bends_morphometry method."""
        morph: Morphometry = Morphometry(centerline)
        obs = morph.compute_bends_morphometry(valid_bends=True).to_numpy()
        self.assertTrue(
            np.array_equal(
                obs, metrics_exp.to_numpy()[valid_bend_indexes], True
            )
        )

    def test_compute_bend_sinuosity_moving_window(self: Self) -> None:
        """Test of compute_bend_sinuosity_moving_window method."""
        morph: Morphometry = Morphometry(centerline)
        obs = [
            morph.compute_bend_sinuosity_moving_window(i, 5000.0)
            for i in range(morph.centerline.get_nb_bends())
        ]
        print(obs)
        self.assertSequenceEqual(
            obs,
            average_metrics_exp["Sinuosity"].tolist(),
            "Sinuosity over moving window are not equal.",
        )

    def test_compute_average_metric_window(self: Self) -> None:
        """Test of compute_average_metric_window method."""
        morph: Morphometry = Morphometry(centerline)
        obs = [
            morph.compute_average_metric_window(i, 5000.0).to_numpy()
            for i in range(morph.centerline.get_nb_bends())
        ]
        # skip 1st column since it is sinuosity
        self.assertTrue(
            np.array_equal(obs, average_metrics_exp.to_numpy()[:, 1:], True)
        )


def _make_sine_centerline(
    nb_bends: int = 7,
    half_wavelength: float = 10.0,
    amplitude: float = 3.0,
    slope: float = -5.0,
    straight_bend_ids: tuple[int, ...] = (),
) -> Centerline:
    """Build a synthetic sine centerline with strongly varying elevation.

    Each bend is a half sine wave between 2 inflection points. Elevation
    varies strongly so that 3D distances differ from horizontal ones.
    """
    nb_pts_bend: int = 10
    nb_pts: int = nb_bends * nb_pts_bend + 1
    x = np.linspace(0.0, nb_bends * half_wavelength, nb_pts)
    k: float = np.pi / half_wavelength
    y = amplitude * np.sin(k * x)
    z = slope * x
    dy = amplitude * k * np.cos(k * x)
    ddy = -amplitude * k * k * np.sin(k * x)
    curv = ddy / (1.0 + dy * dy) ** 1.5
    ds = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    s = np.concatenate(([0.0], np.cumsum(ds)))
    cols = (
        PropertyNames.CURVILINEAR_ABSCISSA.value,
        PropertyNames.CARTESIAN_ABSCISSA.value,
        PropertyNames.CARTESIAN_ORDINATE.value,
        PropertyNames.ELEVATION.value,
        PropertyNames.CURVATURE.value,
        PropertyNames.CURVATURE_FILTERED.value,
    )
    cl = Centerline.__new__(Centerline)
    cl.age = 0
    cl.cl_points = [
        ClPoint(
            str(i),
            0,
            pd.Series([s[i], x[i], y[i], z[i], curv[i], curv[i]], index=cols),
        )
        for i in range(nb_pts)
    ]
    cl.bends = []
    for j in range(nb_bends):
        up = j * nb_pts_bend
        down = up + nb_pts_bend
        side = BendSide.UP if j % 2 == 0 else BendSide.DOWN
        if j in straight_bend_ids:
            side = BendSide.STRAIGHT
        bend = Bend(j, up, down, 0, side)
        bend.index_apex = up + nb_pts_bend // 2
        bend.pt_center = np.array(
            [0.5 * (x[up] + x[down]), 0.0, 1000.0 * (j + 1)]
        )
        cl.bends.append(bend)
    return cl


class TestsMorphometryFixes(unittest.TestCase):
    """Tests on synthetic centerlines for morphometry fixes."""

    def test_planform_metrics_are_horizontal(self: Self) -> None:
        """Planform metrics must ignore elevation."""
        cl = _make_sine_centerline()
        morph = Morphometry(cl)
        self.assertAlmostEqual(morph.compute_bend_wavelength(2), 10.0, 4)
        self.assertAlmostEqual(morph.compute_bend_amplitude(2), 3.0, 4)
        self.assertAlmostEqual(morph.compute_bend_extension(2), 3.0, 4)
        self.assertAlmostEqual(morph.compute_bend_max_amplitude(2), 3.0, 4)
        self.assertAlmostEqual(morph.compute_bend_max_extension(2), 5.0, 4)
        self.assertAlmostEqual(morph.compute_bend_maturity_index(2), 0.3, 4)
        self.assertAlmostEqual(
            morph.compute_bend_wavelength_leopold(2), 20.0, 4
        )
        self.assertAlmostEqual(morph.compute_bend_amplitude_leopold(2), 6.0, 4)
        sinuo_exp = round(
            morph.compute_bend_arc_length(2) / 10.0,
            4,
        )
        self.assertAlmostEqual(morph.compute_bend_sinuosity(2), sinuo_exp, 3)

    def test_leopold_first_last_bends_are_nan(self: Self) -> None:
        """Leopold metrics are NaN for first and last bends."""
        cl = _make_sine_centerline()
        morph = Morphometry(cl)
        last: int = cl.get_nb_bends() - 1
        for bend_id in (0, last):
            self.assertTrue(
                np.isnan(morph.compute_bend_wavelength_leopold(bend_id))
            )
            self.assertTrue(
                np.isnan(morph.compute_bend_amplitude_leopold(bend_id))
            )

    def test_no_apex_returns_nan(self: Self) -> None:
        """Apex-based metrics are NaN when index_apex is -1 or None."""
        for no_apex in (-1, None):
            cl = _make_sine_centerline()
            cl.bends[2].index_apex = no_apex  # type: ignore[assignment]
            morph = Morphometry(cl)
            self.assertTrue(np.isnan(morph.compute_bend_amplitude(2)))
            self.assertTrue(np.isnan(morph.compute_bend_extension(2)))
            self.assertTrue(np.isnan(morph.compute_bend_radius(2)))
            self.assertTrue(np.isnan(morph.compute_bend_asymmetry_apex(2)))
            self.assertTrue(np.isnan(morph.compute_bend_amplitude_leopold(2)))
            self.assertTrue(np.isnan(morph.compute_bend_wavelength_leopold(1)))
            self.assertTrue(np.isnan(morph.compute_bend_wavelength_leopold(3)))

    def test_straight_bend_roundness_skewness_nan(self: Self) -> None:
        """Roundness and skewness are NaN for null curvature or 2-pt bends."""
        cl = _make_sine_centerline()
        for cl_pt in cl.cl_points[
            cl.bends[2].index_inflex_up : cl.bends[2].index_inflex_down + 1
        ]:
            cl_pt.set_property(PropertyNames.CURVATURE_FILTERED.value, 0.0)
        # 2-point bend
        cl.bends[4].index_inflex_down = cl.bends[4].index_inflex_up + 1
        morph = Morphometry(cl)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.assertTrue(np.isnan(morph.compute_bend_roundness(2)))
            self.assertTrue(np.isnan(morph.compute_bend_skewness(2)))
            self.assertTrue(np.isnan(morph.compute_bend_skewness(4)))
        # regular bend still defined
        self.assertFalse(np.isnan(morph.compute_bend_roundness(3)))
        self.assertFalse(np.isnan(morph.compute_bend_skewness(3)))

    def test_compute_bends_morphometry_index(self: Self) -> None:
        """Rows of compute_bends_morphometry are indexed by bend ids."""
        cl = _make_sine_centerline(straight_bend_ids=(1, 4))
        morph = Morphometry(cl)
        valid = morph.compute_bends_morphometry(valid_bends=True)
        self.assertEqual(valid.index.name, "Bend_id")
        self.assertListEqual(valid.index.tolist(), [0, 2, 3, 5, 6])
        full = morph.compute_bends_morphometry(valid_bends=False)
        self.assertListEqual(full.index.tolist(), list(range(7)))
        self.assertTrue(
            np.array_equal(
                valid.to_numpy(), full.loc[[0, 2, 3, 5, 6]].to_numpy(), True
            )
        )

    def test_window_centred_on_bend(self: Self) -> None:
        """Moving window is centred on the bend middle abscissa."""
        cl = _make_sine_centerline()
        morph = Morphometry(cl)
        bend_length: float = morph.compute_bend_arc_length(3)
        self.assertEqual(
            morph._get_window_end_indexes(3, 0.5 * bend_length), (3, 3)
        )
        self.assertEqual(
            morph._get_window_end_indexes(3, 2.5 * bend_length), (2, 4)
        )
        self.assertEqual(
            morph._get_window_end_indexes(3, 3.5 * bend_length), (1, 5)
        )
        self.assertEqual(
            morph._get_window_end_indexes(0, 2.5 * bend_length), (0, 1)
        )
        self.assertEqual(
            morph._get_window_end_indexes(6, 2.5 * bend_length), (5, 6)
        )
        self.assertEqual(morph._get_window_end_indexes(3, 0.0), (3, 3))
        # sinuosity over the window [2, 4]
        cl_pt_min = cl.cl_points[cl.bends[2].index_inflex_up]
        cl_pt_max = cl.cl_points[cl.bends[4].index_inflex_down]
        sinuo_exp = round(
            (cl_pt_max._s - cl_pt_min._s)
            / np.linalg.norm(cl_pt_max.pt[:2] - cl_pt_min.pt[:2]),
            4,
        )
        self.assertAlmostEqual(
            morph.compute_bend_sinuosity_moving_window(3, 2.5 * bend_length),
            sinuo_exp,
            3,
        )

    def test_average_metric_window_skips_nan(self: Self) -> None:
        """Window average ignores NaN values per metric."""
        cl = _make_sine_centerline()
        morph = Morphometry(cl)
        bend_length: float = morph.compute_bend_arc_length(0)
        # window covers bends 0 and 1; Leopold metrics are NaN for bend 0
        mean = morph.compute_average_metric_window(0, 2.5 * bend_length)
        self.assertAlmostEqual(
            mean[MorphometricNames.WAVELENGTH_LEOPOLD.value], 20.0, 3
        )
        self.assertAlmostEqual(
            mean[MorphometricNames.AMPLITUDE_LEOPOLD.value], 6.0, 3
        )
        self.assertAlmostEqual(
            mean[MorphometricNames.WAVELENGTH.value], 10.0, 3
        )
        # single bend window: Leopold metrics are all NaN
        mean = morph.compute_average_metric_window(0, 0.5 * bend_length)
        self.assertTrue(
            np.isnan(mean[MorphometricNames.WAVELENGTH_LEOPOLD.value])
        )


if __name__ == "__main__":
    unittest.main()
