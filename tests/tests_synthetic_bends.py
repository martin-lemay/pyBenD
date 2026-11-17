# SPDX-FileCopyrightText: Copyright 2025 Martin Lemay <martin.lemay@mines-paris.org>
# SPDX-FileContributor: Martin Lemay
# ruff: noqa: E402 # disable Module level import not at top of file

__doc__ = """
Tests functions for centerline_process_functions.py.
"""
import os
import unittest
from typing import Self

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from pybend.algorithms.synthetic_bends import (
    circular_bend,
    kinoshita_bend,
    mirror,
)

# output directory for figures
fig_path: str = "tests/.out/"
# create it if absent
if not os.path.exists(fig_path):
    os.makedirs(fig_path)


teta_max = 110.0 * np.pi / 180.0
nb_pts: int = 50


class TestsProcessFunctions(unittest.TestCase):
    def test_kinoshita_bend(self: Self) -> None:
        """Test of kinoshita_bend function."""
        bend = kinoshita_bend(nb_pts, 100.0 * np.pi / 180.0, 0.01, 0.03)
        expected = np.array(
            [
                [0.0058, 0.0],
                [0.0036, 0.0567],
                [0.0016, 0.1134],
                [0.0003, 0.1702],
                [0.0, 0.227],
                [0.0011, 0.2837],
                [0.0042, 0.3404],
                [0.0097, 0.3969],
                [0.0182, 0.453],
                [0.0302, 0.5085],
                [0.0464, 0.5629],
                [0.0671, 0.6157],
                [0.0928, 0.6663],
                [0.1237, 0.714],
                [0.1598, 0.7578],
                [0.201, 0.7968],
                [0.2469, 0.8302],
                [0.2969, 0.857],
                [0.3502, 0.8766],
                [0.4057, 0.8884],
                [0.4623, 0.8922],
                [0.5189, 0.8878],
                [0.5744, 0.8756],
                [0.6276, 0.8558],
                [0.6778, 0.8293],
                [0.7242, 0.7967],
                [0.7665, 0.7588],
                [0.8044, 0.7165],
                [0.8378, 0.6707],
                [0.8669, 0.6219],
                [0.892, 0.571],
                [0.9132, 0.5184],
                [0.9311, 0.4645],
                [0.946, 0.4097],
                [0.9583, 0.3543],
                [0.9684, 0.2985],
                [0.9766, 0.2423],
                [0.9833, 0.1859],
                [0.9885, 0.1294],
                [0.9926, 0.0728],
                [0.9957, 0.0161],
                [0.9979, -0.0406],
                [0.9993, -0.0973],
                [1.0, -0.1541],
                [1.0, -0.2109],
                [0.9993, -0.2676],
                [0.9981, -0.3244],
                [0.9964, -0.3811],
                [0.9943, -0.4378],
            ]
        )
        print(np.round(bend, 4).tolist())
        self.assertTrue(np.array_equal(expected, np.round(bend, 4)))

        # visual check
        # plt.figure(dpi=150)
        # plt.plot(bend.T[0], bend.T[1], "k--", label="Kinoshita bend")
        # plt.plot(
        #     expected.T[0],
        #     expected.T[1],
        #     "bo",
        #     markersize=2,
        #     label="Expected path",
        # )
        # plt.legend()
        # plt.axis("equal")
        # plt.savefig(fig_path + "test_kinoshita_bend.png", dpi=150)
        # plt.close()

        # self.assertTrue(False)

    def test_circular_bend(self: Self) -> None:
        """Test of test_circular_bend function."""
        bend = circular_bend(nb_pts, 1.0)
        expected = np.array(
            [
                [-1.0000, 0.0000],
                [-0.9979, 0.0641],
                [-0.9918, 0.1279],
                [-0.9816, 0.1912],
                [-0.9673, 0.2537],
                [-0.9491, 0.3151],
                [-0.9269, 0.3753],
                [-0.9010, 0.4339],
                [-0.8713, 0.4907],
                [-0.8381, 0.5455],
                [-0.8014, 0.5981],
                [-0.7614, 0.6482],
                [-0.7183, 0.6957],
                [-0.6723, 0.7403],
                [-0.6235, 0.7818],
                [-0.5721, 0.8202],
                [-0.5184, 0.8551],
                [-0.4625, 0.8866],
                [-0.4048, 0.9144],
                [-0.3454, 0.9385],
                [-0.2845, 0.9587],
                [-0.2225, 0.9749],
                [-0.1596, 0.9872],
                [-0.0960, 0.9954],
                [-0.0321, 0.9995],
                [0.0321, 0.9995],
                [0.0960, 0.9954],
                [0.1596, 0.9872],
                [0.2225, 0.9749],
                [0.2845, 0.9587],
                [0.3454, 0.9385],
                [0.4048, 0.9144],
                [0.4625, 0.8866],
                [0.5184, 0.8551],
                [0.5721, 0.8202],
                [0.6235, 0.7818],
                [0.6723, 0.7403],
                [0.7183, 0.6957],
                [0.7614, 0.6482],
                [0.8014, 0.5981],
                [0.8381, 0.5455],
                [0.8713, 0.4907],
                [0.9010, 0.4339],
                [0.9269, 0.3753],
                [0.9491, 0.3151],
                [0.9673, 0.2537],
                [0.9816, 0.1912],
                [0.9918, 0.1279],
                [0.9979, 0.0641],
                [1.0000, 0.0000],
            ]
        )
        self.assertTrue(np.array_equal(expected, np.round(bend, 4)))

        # visual check
        plt.figure(dpi=150)
        plt.plot(bend.T[0], bend.T[1], "k--", label="Circular bend")
        plt.plot(
            expected.T[0],
            expected.T[1],
            "bo",
            markersize=2,
            label="Expected path",
        )
        plt.legend()
        plt.axis("equal")
        plt.savefig(fig_path + "test_circular_bend.png", dpi=150)
        plt.close()

    def test_mirror(self: Self) -> None:
        """Test of mirror function."""
        coords = circular_bend(nb_pts, 1.0)
        new_coords = mirror(coords, 10)

        expected: npt.NDArray[np.float64] = np.array(
            [
                [-1.1986, -0.5981],
                [-1.1619, -0.5455],
                [-1.1287, -0.4907],
                [-1.0990, -0.4339],
                [-1.0731, -0.3753],
                [-1.0509, -0.3151],
                [-1.0327, -0.2537],
                [-1.0184, -0.1912],
                [-1.0082, -0.1279],
                [-1.0021, -0.0641],
                [-1.0000, 0.0000],
                [-0.9979, 0.0641],
                [-0.9918, 0.1279],
                [-0.9816, 0.1912],
                [-0.9673, 0.2537],
                [-0.9491, 0.3151],
                [-0.9269, 0.3753],
                [-0.9010, 0.4339],
                [-0.8713, 0.4907],
                [-0.8381, 0.5455],
                [-0.8014, 0.5981],
                [-0.7614, 0.6482],
                [-0.7183, 0.6957],
                [-0.6723, 0.7403],
                [-0.6235, 0.7818],
                [-0.5721, 0.8202],
                [-0.5184, 0.8551],
                [-0.4625, 0.8866],
                [-0.4048, 0.9144],
                [-0.3454, 0.9385],
                [-0.2845, 0.9587],
                [-0.2225, 0.9749],
                [-0.1596, 0.9872],
                [-0.0960, 0.9954],
                [-0.0321, 0.9995],
                [0.0321, 0.9995],
                [0.0960, 0.9954],
                [0.1596, 0.9872],
                [0.2225, 0.9749],
                [0.2845, 0.9587],
                [0.3454, 0.9385],
                [0.4048, 0.9144],
                [0.4625, 0.8866],
                [0.5184, 0.8551],
                [0.5721, 0.8202],
                [0.6235, 0.7818],
                [0.6723, 0.7403],
                [0.7183, 0.6957],
                [0.7614, 0.6482],
                [0.8014, 0.5981],
                [0.8381, 0.5455],
                [0.8713, 0.4907],
                [0.9010, 0.4339],
                [0.9269, 0.3753],
                [0.9491, 0.3151],
                [0.9673, 0.2537],
                [0.9816, 0.1912],
                [0.9918, 0.1279],
                [0.9979, 0.0641],
                [1.0000, 0.0000],
                [1.0021, -0.0641],
                [1.0082, -0.1279],
                [1.0184, -0.1912],
                [1.0327, -0.2537],
                [1.0509, -0.3151],
                [1.0731, -0.3753],
                [1.0990, -0.4339],
                [1.1287, -0.4907],
                [1.1619, -0.5455],
                [1.1986, -0.5981],
            ]
        )
        self.assertTrue(np.array_equal(expected, np.round(new_coords, 4)))

        # visual check
        plt.figure(dpi=150)
        plt.plot(
            new_coords.T[0],
            new_coords.T[1],
            "k--",
            label="Circular bend mirror",
        )
        plt.plot(
            expected.T[0],
            expected.T[1],
            "bo",
            markersize=2,
            label="Expected path",
        )
        plt.legend()
        plt.axis("equal")
        plt.savefig(fig_path + "test_mirror_bend.png", dpi=150)
        plt.close()

    def test_mirror_zero_points(self: Self) -> None:
        """mirror() with 0 point returns a copy of input coordinates."""
        coords = circular_bend(nb_pts, 1.0)
        new_coords = mirror(coords, 0)
        self.assertTrue(np.array_equal(coords, new_coords))
        self.assertIsNot(coords, new_coords)
        with self.assertRaises(ValueError):
            mirror(coords, -1)

    def test_kinoshita_bend_normalisation(self: Self) -> None:
        """Kinoshita x is normalised in [0, 1] and y starts at 0."""
        bend = kinoshita_bend(nb_pts, 100.0 * np.pi / 180.0, 0.01, 0.03)
        self.assertAlmostEqual(float(np.min(bend[:, 0])), 0.0, places=12)
        self.assertAlmostEqual(float(np.max(bend[:, 0])), 1.0, places=12)
        self.assertAlmostEqual(float(bend[0, 1]), 0.0, places=12)
        # aspect ratio is preserved: unit spacing between raw points is
        # scaled by the same factor along x and y
        steps = np.linalg.norm(np.diff(bend, axis=0), axis=1)
        self.assertTrue(np.allclose(steps, steps[0], rtol=1e-6))
        # symmetric bend: inflection points at the same y
        sym = kinoshita_bend(nb_pts, 100.0 * np.pi / 180.0, 0.0, 0.0)
        self.assertLess(abs(sym[-1, 1] - sym[0, 1]), 0.05)


if __name__ == "__main__":
    unittest.main()
