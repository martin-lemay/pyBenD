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
                [-0.1494, 4.9976],
                [-0.1874, 5.9968],
                [-0.2184, 6.9964],
                [-0.2364, 7.9962],
                [-0.2344, 8.9962],
                [-0.2043, 9.9957],
                [-0.1373, 10.9935],
                [-0.0238, 11.987],
                [0.1458, 12.9725],
                [0.3814, 13.9444],
                [0.692, 14.8949],
                [1.0855, 15.8142],
                [1.568, 16.6902],
                [2.1426, 17.5086],
                [2.8092, 18.254],
                [3.5639, 18.9101],
                [4.3984, 19.4611],
                [5.3005, 19.8927],
                [6.2542, 20.1932],
                [7.2411, 20.3545],
                [8.241, 20.3728],
                [9.2333, 20.2489],
                [10.1986, 19.988],
                [11.1199, 19.5991],
                [11.983, 19.0941],
                [12.7776, 18.4869],
                [13.497, 17.7924],
                [14.1383, 17.0251],
                [14.7017, 16.1989],
                [15.1902, 15.3263],
                [15.6084, 14.4179],
                [15.9625, 13.4827],
                [16.2593, 12.5278],
                [16.5059, 11.5587],
                [16.7091, 10.5795],
                [16.8752, 9.5934],
                [17.0099, 8.6026],
                [17.118, 7.6084],
                [17.2035, 6.6121],
                [17.2696, 5.6143],
                [17.3189, 4.6155],
                [17.353, 3.6161],
                [17.3733, 2.6163],
                [17.3809, 1.6163],
                [17.3767, 0.6163],
                [17.3616, -0.3836],
                [17.3371, -1.3833],
                [17.3048, -2.3828],
                [17.2671, -3.3821],
                [17.2272, -4.3813],
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
