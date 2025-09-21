# SPDX-FileCopyrightText: Copyright 2025 Martin Lemay <martin.lemay@mines-paris.org>
# SPDX-FileContributor: Martin Lemay
# ruff: noqa: E402 # disable Module level import not at top of file

__doc__ = """
Tests functions for CenterlineCollection class - Run with input data
centerline_Collection_test_data*.csv.
"""

import os
import unittest
from typing import Self

import numpy as np
import numpy.typing as npt
import pandas as pd

from pybend.algorithms.pybend_io import (
    load_centerline_dataset_from_Flumy_csv,
)
from pybend.model.Centerline import Centerline
from pybend.model.Morphometry import Morphometry
from pybend.utils.globalParameters import set_nb_procs

set_nb_procs(1)

# inputs

# output directory for figures
dir_path: str = "tests/data/"
fig_path: str = "tests/.out/"
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


age, dataset = load_centerline_dataset_from_Flumy_csv(filepath)
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

assert (
    centerline.get_nb_bends() == 46
), "Number of bends in invalid. Run tests_Centerline.py first."

assert (
    centerline.get_nb_valid_bends() == 34
), "Number of bends in invalid. Run tests_Centerline.py first."


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

    def test_compute_bend_asymmetry(self: Self) -> None:
        """Test of compute_bend_asymmetry method."""
        morph: Morphometry = Morphometry(centerline)
        obs = [
            morph.compute_bend_asymmetry(i)
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


if __name__ == "__main__":
    unittest.main()
