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
import pandas as pd  # type: ignore[import-untyped]

from pybend.algorithms.centerline_process_function import (
    clpoints2coords,
    compute_colinear,
    compute_curvature_at_point,
    compute_curvature_at_point_Menger,
    compute_cuvilinear_abscissa,
    distance,
    distance_arrays,
    filter_consecutive_indices,
    find_2_closest_points_mono_proc,
    find_2_closest_points_multi_proc,
    find_inflection_points,
    find_inflection_points_from_peaks,
    perp,
    project_orthogonal,
    resample_path,
    seg_intersect,
)
from pybend.model.ClPoint import ClPoint
from pybend.model.enumerations import PropertyNames
from pybend.utils.globalParameters import (
    get_nb_procs,
    set_nb_procs,
)

# inputs

# output directory for figures
fig_path: str = "tests/.out/"
# create it if absent
if not os.path.exists(fig_path):
    os.makedirs(fig_path)

nb_procs: int = min(5, get_nb_procs())

# points
pts_in: list[tuple[float, float, float]] = [
    (5.2, 6.4, 1.0),
    (4.2, 8.6, 1.2),
    (10.1, 9.5, 1.5),
    (11.5, 10.6, 2.0),
    (12.0, 10.0, 1.6),
    (12.5, 11.1, 1.5),
    (13.1, 8.6, 1.3),
    (13.4, 8.4, 1.1),
    (14.0, 8.0, 1.0),
]

pt1: tuple[float, float] = (0, 0)
pt2: tuple[float, float] = (1, 0)
pt11: npt.NDArray[np.float64] = np.array((0, 0))
pt12: npt.NDArray[np.float64] = np.array((5, 0))
pt21: npt.NDArray[np.float64] = np.array((1, 1))
pt22: npt.NDArray[np.float64] = np.array((1, -2))
k: int = 5
pts1_curv: tuple[npt.NDArray[np.float64], ...] = (
    np.array((0, 0)),
    np.array((1, 0)),
    np.array((1, 0)),
)
pts2_curv: tuple[npt.NDArray[np.float64], ...] = (
    np.array((1, 0)),
    np.array((0, 0)),
    np.array((0, 0)),
)
pts3_curv: tuple[npt.NDArray[np.float64], ...] = (
    np.array((2, 0)),
    np.array((0, 1)),
    np.array((0, -1)),
)

# ClPoint
dataset = pd.DataFrame(
    np.array(pts_in).reshape(len(pts_in), len(pts_in[0])),
    columns=(
        PropertyNames.CARTESIAN_ABSCISSA.value,
        PropertyNames.CARTESIAN_ORDINATE.value,
        PropertyNames.ELEVATION.value,
    ),
)

dataset[PropertyNames.CURVILINEAR_ABSCISSA.value] = np.arange(0, len(pts_in), 1)
cl_pts_in = [ClPoint("0", 0, data) for _, data in dataset.iterrows()]

nb_pts: int = 50
lx: npt.NDArray[np.float64] = np.linspace(0, 10, 10)
ly: npt.NDArray[np.float64] = np.sin(lx)
dataset1 = pd.DataFrame(
    np.concatenate((lx.reshape(lx.size, 1), ly.reshape(ly.size, 1)), axis=1),
    columns=("X", "Y"),
)
x_prop: str = "X"
y_prop: str = "Y"

lx2: npt.NDArray[np.float64] = np.array(
    [
        -0.9906,
        -1.5219,
        -1.8323,
        -1.617,
        -0.556,
        1.4964,
        4.2984,
        7.2383,
        9.677,
        11.2984,
        12.1511,
        12.4633,
        12.4619,
        12.2894,
        11.9982,
        11.5892,
        11.0714,
    ]
)
ly2: npt.NDArray[np.float64] = np.array(
    [
        5.9167,
        8.8691,
        11.8514,
        14.8376,
        17.6293,
        19.7877,
        20.782,
        20.3342,
        18.6225,
        16.1134,
        13.2434,
        10.2619,
        7.2625,
        4.2677,
        1.282,
        -1.6898,
        -4.6447,
    ]
)

# expected results
coords_out: npt.NDArray[np.float64] = np.array(pts_in).reshape(
    len(pts_in), len(pts_in[0])
)
curv_abscissa_out = np.array(
    [0.0, 2.4166, 8.3848, 10.1652, 10.9462, 12.1545, 14.7255, 15.0861, 15.8072]
)
pt_out_colinear: npt.NDArray[np.float64] = np.array((5, 0))
pt_out1: npt.NDArray[np.float64] = np.array((5, 5))
perp_out: npt.NDArray[np.float64] = np.array((-1, 1))
pt_out_intersect: npt.NDArray[np.float64] = np.array((1, 0))
curvs_out: tuple[float, ...] = (0.0, 2**0.5, -(2**0.5))

lx_out: npt.NDArray[np.float64] = np.array(
    (
        0.0,
        0.12,
        0.278,
        0.467,
        0.682,
        0.916,
        1.163,
        1.416,
        1.669,
        1.916,
        2.151,
        2.367,
        2.565,
        2.75,
        2.926,
        3.099,
        3.272,
        3.45,
        3.635,
        3.83,
        4.035,
        4.251,
        4.481,
        4.722,
        4.968,
        5.211,
        5.443,
        5.656,
        5.848,
        6.027,
        6.198,
        6.367,
        6.541,
        6.726,
        6.924,
        7.135,
        7.355,
        7.581,
        7.81,
        8.04,
        8.27,
        8.497,
        8.719,
        8.935,
        9.143,
        9.341,
        9.528,
        9.701,
        9.859,
        10.0,
    )
)
ly_out: npt.NDArray[np.float64] = np.array(
    (
        0.0,
        0.239,
        0.442,
        0.61,
        0.744,
        0.842,
        0.907,
        0.937,
        0.933,
        0.896,
        0.825,
        0.721,
        0.588,
        0.432,
        0.257,
        0.071,
        -0.122,
        -0.316,
        -0.502,
        -0.67,
        -0.811,
        -0.914,
        -0.969,
        -0.972,
        -0.928,
        -0.845,
        -0.732,
        -0.597,
        -0.446,
        -0.282,
        -0.109,
        0.069,
        0.25,
        0.429,
        0.601,
        0.755,
        0.88,
        0.964,
        0.998,
        0.976,
        0.903,
        0.789,
        0.642,
        0.472,
        0.288,
        0.097,
        -0.09,
        -0.265,
        -0.419,
        -0.544,
    )
)

lx2_out: npt.NDArray[np.float64] = np.array(
    [
        -0.9906,
        -1.1698,
        -1.3466,
        -1.5109,
        -1.6526,
        -1.7613,
        -1.8271,
        -1.8394,
        -1.7862,
        -1.6549,
        -1.4334,
        -1.113,
        -0.6868,
        -0.149,
        0.4972,
        1.2436,
        2.081,
        2.991,
        3.9494,
        4.9317,
        5.913,
        6.868,
        7.7721,
        8.6079,
        9.3643,
        10.0306,
        10.6035,
        11.086,
        11.4816,
        11.7966,
        12.0399,
        12.2209,
        12.3483,
        12.4304,
        12.4757,
        12.4909,
        12.4813,
        12.4517,
        12.4059,
        12.3458,
        12.2728,
        12.1881,
        12.0912,
        11.9818,
        11.8596,
        11.725,
        11.5784,
        11.4203,
        11.2511,
        11.0714,
    ]
)
ly2_out: npt.NDArray[np.float64] = np.array(
    [
        5.9167,
        6.8751,
        7.8362,
        8.8004,
        9.7678,
        10.7388,
        11.7137,
        12.6921,
        13.6693,
        14.6392,
        15.5946,
        16.5222,
        17.4057,
        18.2286,
        18.9711,
        19.6116,
        20.1292,
        20.5087,
        20.7385,
        20.8077,
        20.7167,
        20.4731,
        20.0852,
        19.5672,
        18.938,
        18.2167,
        17.4207,
        16.5664,
        15.6699,
        14.7434,
        13.7954,
        12.834,
        11.8649,
        10.891,
        9.9147,
        8.9378,
        7.961,
        6.9845,
        6.0087,
        5.0337,
        4.0596,
        3.0865,
        2.1145,
        1.1438,
        0.1747,
        -0.7928,
        -1.7586,
        -2.7226,
        -3.6846,
        -4.6447,
    ]
)

dataset2 = pd.DataFrame(
    np.concatenate(
        (lx_out.reshape(lx_out.size, 1), ly_out.reshape(ly_out.size, 1)),
        axis=1,
    ),
    columns=("X", "Y"),
)
columns: tuple[str, str, str, str] = ("index1", "index2", "d1", "d2")
res_closest: npt.NDArray[np.float64] = np.array(
    (
        (0, 0, 0.0, 0.0),
        (6, 5, 0.0530, 0.2025),
        (10, 11, 0.0772, 0.1627),
        (16, 17, 0.0920, 0.1713),
        (22, 21, 0.0369, 0.1999),
        (27, 26, 0.1214, 0.1309),
        (33, 32, 0.0808, 0.1767),
        (38, 37, 0.0322, 0.1995),
        (43, 42, 0.0601, 0.2148),
        (49, 49, 0.0, 0.0),
    )
)
result_out = pd.DataFrame(res_closest, columns=columns)

eps: float = 1e-6


class TestsProcessFunctions(unittest.TestCase):
    def test_nb_procs(self: Self) -> None:
        """Test of get_nb_procs() function."""
        set_nb_procs(nb_procs)
        self.assertEqual(
            get_nb_procs(),
            nb_procs,
            "User defined number of procs must be %s." % nb_procs,
        )

    def test_clpoints2coords(self: Self) -> None:
        """Test of clpoints2coords() function."""
        coords = clpoints2coords(cl_pts_in)
        self.assertTrue(np.array_equal(coords, coords_out), "Coordinates are wrong.")

    def test_compute_cuvilinear_abscissa(self: Self) -> None:
        """Test of compute_cuvilinear_abscissa() function."""
        XY = coords_out[:, :2]
        curv_abscissa = compute_cuvilinear_abscissa(XY)
        self.assertAlmostEqual(
            (curv_abscissa - curv_abscissa_out).sum(),
            0.0,
            3,
            "Curvilinear abscissa are wrong.",
        )

    def test_compute_colinear(self: Self) -> None:
        """Test of compute_colinear() function."""
        pt = compute_colinear(pt11, pt21, k)
        self.assertTrue(np.array_equal(pt, pt_out1), "Point coordinate is wrong.")

    def test_distance_arrays(self: Self) -> None:
        """Test of distance_arrays() function."""
        pts1, pts2 = coords_out[:5], coords_out[4:]
        array1: npt.NDArray[np.float64] = distance_arrays(pts1, pts2, prec=4)
        array2: npt.NDArray[np.float64] = np.array(
            (7.7175, 8.6735, 3.1385, 3.043, 2.8914)
        )

        self.assertTrue(
            np.array_equal(array1, array2),
            "Distances from distance_arrays are different.",
        )

    def test_distance(self: Self) -> None:
        """Test of distance() function."""
        pts1, pts2 = coords_out[:5], coords_out[4:]
        d1 = distance_arrays(pts1, pts2, prec=4)
        d2 = [distance(pt1, pt2, prec=4) for pt1, pt2 in zip(pts1, pts2, strict=False)]
        self.assertTrue(np.array_equal(d1, d2), "Distances are different.")

    def test_perp(self: Self) -> None:
        """Test of perp() function."""
        vec_in = pt21 - pt11
        vec = perp(vec_in)
        self.assertTrue(np.array_equal(vec, perp_out))

    def test_seg_intersect(self: Self) -> None:
        """Test of seg_intersect() function."""
        pt = seg_intersect(pt11, pt12, pt21, pt22)
        self.assertTrue(np.array_equal(pt, pt_out_intersect))

    def test_project_orthogonal(self: Self) -> None:
        """Test of project_orthogonal() function."""
        pt_proj = project_orthogonal(pt21, pt11, pt12)
        self.assertAlmostEqual(pt_proj[0], pt_out_intersect[0], 3)
        self.assertAlmostEqual(pt_proj[1], pt_out_intersect[1], 3)

    def test_resample_path0(self: Self) -> None:
        """Test of resample_path() function."""
        lx0 = np.array([0.0, 1.0])
        lx_new, ly_new = resample_path(lx0, lx0, nb_pts)
        self.assertTrue(np.any(np.abs(lx0 - lx_new) < eps))
        self.assertTrue(np.any(np.abs(lx0 - ly_new) < eps))

    def test_resample_path1(self: Self) -> None:
        """Test of resample_path() function."""
        lx_new, ly_new = resample_path(lx, ly, 0)
        self.assertTrue(np.any(np.abs(lx - lx_new) < eps))
        self.assertTrue(np.any(np.abs(ly - ly_new) < eps))

        lx_new, ly_new = resample_path(lx, ly, nb_pts)
        self.assertEqual(lx_new.size, nb_pts)
        self.assertTrue(np.any(np.abs(lx_out - lx_new) < eps))
        self.assertTrue(np.any(np.abs(ly_out - ly_new) < eps))

        # visual check
        plt.figure(dpi=150)
        plt.plot(lx, ly, "k--", label="Initial path")
        plt.plot(lx_new, ly_new, "r-", label="Resample path")
        plt.plot(lx_out, ly_out, "bo", markersize=2, label="Expected points")
        plt.legend()
        plt.axis("equal")
        plt.savefig(fig_path + "test_resample_path.png", dpi=150)
        plt.close()

    def test_resample_path2(self: Self) -> None:
        """Test of resample_path() function."""
        lx_new, ly_new = resample_path(lx2, ly2, nb_pts)
        print(np.round(lx_new, 4).tolist())
        print(np.round(ly_new, 4).tolist())
        self.assertEqual(lx_new.size, nb_pts)
        self.assertTrue(np.any(np.abs(lx2_out - lx_new) < eps))
        self.assertTrue(np.any(np.abs(ly2_out - ly_new) < eps))

        # visual check
        plt.figure(dpi=150)
        plt.plot(lx2_out, ly2_out, "bo", markersize=2, label="Expected points")
        plt.plot(lx_new, ly_new, "ro", markersize=1, label="Resample path")
        plt.plot(lx2, ly2, "kx", markersize=2, label="Initial path")
        plt.legend()
        plt.axis("equal")
        plt.savefig(fig_path + "test_resample_path2.png", dpi=150)
        plt.close()

    def test_find_2_closest_points(self: Self) -> None:
        """Test of find_2_closest_points() function."""
        result = find_2_closest_points_mono_proc(dataset1, dataset2, "X", "Y")
        self.assertSequenceEqual(
            list(result_out["index1"].to_numpy()),
            list(result["index1"].to_numpy()),
            "Monoprocessing test: Index1 is wrong",
        )
        self.assertSequenceEqual(
            list(result_out["index2"].to_numpy()),
            list(result["index2"].to_numpy()),
            "Monoprocessing test: Index2 is wrong",
        )

        self.assertTrue(
            all(np.abs(result_out["d1"] - result["d1"]) < eps),
            "Monoprocessing test: d1 is wrong",
        )
        self.assertTrue(
            all(np.abs(result_out["d2"] - result["d2"]) < eps),
            "Monoprocessing test: d2 is wrong",
        )

        result2 = find_2_closest_points_multi_proc(
            dataset1, dataset2, "X", "Y", nb_procs
        )
        self.assertSequenceEqual(
            list(result_out["index1"].to_numpy()),
            list(result2["index1"].to_numpy()),
            "Multiprocessing test: Index1 is wrong",
        )
        self.assertSequenceEqual(
            list(result_out["index2"].to_numpy()),
            list(result2["index2"].to_numpy()),
            "Multiprocessing test: Index2 is wrong",
        )
        self.assertTrue(
            all(np.abs(result_out["d1"] - result2["d1"]) < eps),
            "Multiprocessing test: d1 is wrong",
        )
        self.assertTrue(
            all(np.abs(result_out["d2"] - result2["d2"]) < eps),
            "Multiprocessing test: d2 is wrong",
        )

        # visual check
        plt.figure(dpi=150)
        plt.plot(dataset1[x_prop], dataset1[y_prop], "ko", label="All points")
        plt.plot(
            dataset2[x_prop],
            dataset2[y_prop],
            "ro",
            markersize=3,
            label="Points to find the closest points",
        )

        lx_closest = [
            dataset2[x_prop][int(row["index1"])] for _, row in result.iterrows()
        ]
        ly_closest = [
            dataset2[y_prop][int(row["index1"])] for _, row in result.iterrows()
        ]
        plt.plot(
            lx_closest,
            ly_closest,
            "bo",
            markersize=1.5,
            label="1st Closest points",
        )

        lx_closest = [
            dataset2[x_prop][int(row["index2"])] for _, row in result.iterrows()
        ]
        ly_closest = [
            dataset2[y_prop][int(row["index2"])] for _, row in result.iterrows()
        ]
        plt.plot(
            lx_closest,
            ly_closest,
            "go",
            markersize=1.5,
            label="2nd Closest points",
        )

        plt.legend()
        plt.axis("equal")
        plt.savefig(fig_path + "test_find_2_closest_points.png", dpi=150)
        plt.close()

    def test_compute_curvature(self: Self) -> None:
        """Test of compute_curvature() function."""
        for pt1, pt2, pt3, curv_out in zip(
            pts1_curv, pts2_curv, pts3_curv, curvs_out, strict=False
        ):
            curv: float = compute_curvature_at_point(pt1, pt2, pt3)
            curv2: float = compute_curvature_at_point_Menger(pt1, pt2, pt3)
            self.assertAlmostEqual(
                curv, curv_out, 5, "Error in compute_curvature function"
            )
            self.assertAlmostEqual(
                curv2,
                abs(curv_out),
                5,
                "Error in compute_curvature_Menger function.",
            )

    def test_find_inflection_points(self: Self) -> None:
        """Test of find_inflection_points function."""
        curv: npt.NDArray[np.float64] = np.sin(np.linspace(1, 100.0, 100) / 10.0)
        obs: list[int] = find_inflection_points(curv, 2).tolist()
        exp: list[int] = [30, 62, 93]
        self.assertSequenceEqual(obs, exp)

    def test_find_inflection_points_from_peaks(self: Self) -> None:
        """Test of find_inflection_points_from_peaks function."""
        curv = np.sin(np.linspace(1, 100, 100) / 10.0)
        obs: list[int] = find_inflection_points_from_peaks(curv, 0.1).tolist()
        exp: list[int] = [30, 62, 93]
        self.assertSequenceEqual(obs, exp)

    def test_filter_consecutive_indices(self: Self) -> None:
        """Test of filter_consecutive_indices() function."""
        input_list: npt.NDArray[np.int64] = np.array(
            [2, 8, 9, 20, 31, 32, 33, 35, 41, 42, 44]
        )
        lag: int = 1
        obs: list[int] = filter_consecutive_indices(input_list, lag).tolist()
        exp: list[int] = [2, 8, 20, 32, 35, 41, 44]
        self.assertSequenceEqual(obs, exp)

        lag = 2
        obs = filter_consecutive_indices(input_list, lag).tolist()
        exp = [2, 8, 20, 32, 42]
        self.assertSequenceEqual(obs, exp)


if __name__ == "__main__":
    unittest.main()
