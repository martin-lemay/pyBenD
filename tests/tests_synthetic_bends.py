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
                [0.0058, 0.7272],
                [0.0036, 0.7839],
                [0.0016, 0.8406],
                [0.0003, 0.8974],
                [0.0, 0.9541],
                [0.0011, 1.0109],
                [0.0042, 1.0675],
                [0.0097, 1.124],
                [0.0182, 1.1802],
                [0.0302, 1.2356],
                [0.0464, 1.29],
                [0.0671, 1.3429],
                [0.0928, 1.3935],
                [0.1237, 1.4411],
                [0.1598, 1.4849],
                [0.201, 1.524],
                [0.2469, 1.5573],
                [0.2969, 1.5842],
                [0.3502, 1.6038],
                [0.4057, 1.6156],
                [0.4623, 1.6193],
                [0.5189, 1.615],
                [0.5744, 1.6027],
                [0.6276, 1.583],
                [0.6778, 1.5565],
                [0.7242, 1.5238],
                [0.7665, 1.486],
                [0.8044, 1.4437],
                [0.8378, 1.3978],
                [0.8669, 1.3491],
                [0.892, 1.2982],
                [0.9132, 1.2455],
                [0.9311, 1.1917],
                [0.946, 1.1369],
                [0.9583, 1.0815],
                [0.9684, 1.0256],
                [0.9766, 0.9695],
                [0.9833, 0.9131],
                [0.9885, 0.8566],
                [0.9926, 0.8],
                [0.9957, 0.7433],
                [0.9979, 0.6866],
                [0.9993, 0.6298],
                [1.0, 0.5731],
                [1.0, 0.5163],
                [0.9993, 0.4595],
                [0.9981, 0.4028],
                [0.9964, 0.3461],
                [0.9943, 0.2893],
            ]
        )
        print(np.round(bend, 4).tolist())
        self.assertTrue(np.array_equal(expected, np.round(bend, 4)))

        # visual check
        plt.figure(dpi=150)
        plt.plot(bend.T[0], bend.T[1], "k--", label="Kinoshita bend")
        plt.plot(
            expected.T[0],
            expected.T[1],
            "bo",
            markersize=2,
            label="Expected path",
        )
        plt.legend()
        plt.axis("equal")
        plt.savefig(fig_path + "test_kinoshita_bend.png", dpi=150)
        plt.close()

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


if __name__ == "__main__":
    unittest.main()
