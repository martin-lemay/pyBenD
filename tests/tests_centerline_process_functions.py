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

from pybend.algorithms.centerline_process_functions import (
    clpoints2coords,
    compute_curvature,
    compute_curvature_at_point,
    compute_curvature_at_point_Menger,
    compute_cuvilinear_abscissa,
    compute_esperance,
    compute_half_angle_variation,
    compute_kurtosis,
    compute_median_curvature_index,
    compute_point_displacements,
    compute_skewness,
    compute_variance,
    filter_consecutive_indices,
    find_2_closest_points_mono_proc,
    find_2_closest_points_multi_proc,
    find_inflection_points,
    find_inflection_points_from_peaks,
    resample_path,
)
from pybend.algorithms.geometry_functions import normal
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

dataset[PropertyNames.CURVILINEAR_ABSCISSA.value] = np.arange(
    0, len(pts_in), 1
)
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

bend1 = np.array(
    [
        [0.0, 0.02460340231050681],
        [0.033448785710398825, 0.08290781035769729],
        [0.068409450343311, 0.1403184120934265],
        [0.1065689593610426, 0.19565445039360774],
        [0.1493985911953874, 0.24746031782354228],
        [0.19789567265751726, 0.2940033984359062],
        [0.2523606008254085, 0.3333962513777778],
        [0.3122889477405898, 0.36383994529371305],
        [0.37643757089508867, 0.3839192732360411],
        [0.44306060237681255, 0.3928409849226347],
        [0.5102384610946183, 0.39052569631773054],
        [0.5761885925163301, 0.3775331819847616],
        [0.6394716208646274, 0.354873005765189],
        [0.6990683470359246, 0.3237851384710679],
        [0.7543596247467657, 0.28556080234424797],
        [0.8050659156469491, 0.24143487655929194],
        [0.8511960546926158, 0.19254484143084352],
        [0.8930301932749155, 0.1399318191615383],
        [0.931135169086971, 0.0845582146048894],
        [0.9663911274243189, 0.0273284787593636],
        [1.0, -0.030883796126824292],
    ]
)
bend2 = np.array(
    [
        [0.244663364568034, 0.6619724536249403],
        [0.1539288407308944, 0.7828095385212653],
        [0.07269784047837094, 0.9102296291670053],
        [0.01584516400822417, 1.0502372292477686],
        [0.0, 1.2005145917470705],
        [0.03805449179076463, 1.3467548411886683],
        [0.13106958566250876, 1.465845341289326],
        [0.2647388450607147, 1.5363214648641172],
        [0.41534070952253643, 1.5487082853930978],
        [0.5607480592248206, 1.5075858152003705],
        [0.6877058851178338, 1.4256342115143326],
        [0.7919833643220967, 1.31626985611003],
        [0.8743963791805451, 1.1896110629524475],
        [0.9366899997849941, 1.0519380455010126],
        [0.9791355008285192, 0.9069113778037982],
        [1.0, 0.7572483310069439],
        [0.9965088656528036, 0.6061782582487568],
        [0.9667667076202336, 0.45802374789635564],
        [0.9117443024095477, 0.31728679529674453],
        [0.8361890992429324, 0.1864213446359777],
        [0.7480276598396666, 0.06369434880971948],
    ]
)

bend3 = np.array(
    [
        [-1.0, 0.0],
        [-0.9876883405951378, 0.15643446504023087],
        [-0.9510565162951536, 0.3090169943749474],
        [-0.8910065241883679, 0.45399049973954675],
        [-0.8090169943749476, 0.5877852522924731],
        [-0.7071067811865477, 0.7071067811865475],
        [-0.5877852522924732, 0.8090169943749475],
        [-0.4539904997395469, 0.8910065241883678],
        [-0.30901699437494756, 0.9510565162951535],
        [-0.15643446504023104, 0.9876883405951378],
        [-1.8369701987210297e-16, 1.0],
        [0.15643446504023067, 0.9876883405951378],
        [0.30901699437494723, 0.9510565162951536],
        [0.45399049973954664, 0.8910065241883679],
        [0.5877852522924729, 0.8090169943749475],
        [0.7071067811865474, 0.7071067811865476],
        [0.8090169943749473, 0.5877852522924732],
        [0.8910065241883678, 0.45399049973954686],
        [0.9510565162951535, 0.3090169943749475],
        [0.9876883405951377, 0.15643446504023098],
        [1.0, 1.2246467991473532e-16],
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
bend2_curvs: npt.NDArray[np.float64] = np.array(
    [
        0.253,
        0.5518,
        1.1866,
        1.8153,
        2.3193,
        2.6341,
        2.7285,
        2.6117,
        2.3305,
        1.9571,
        1.5725,
        1.2496,
        1.0384,
        0.9558,
        0.9819,
        1.065,
        1.1344,
        1.1163,
        0.9519,
        0.5775,
        0.3282,
    ]
)
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
l_half_angle_index_exp = (10, 7, 10, 7, 10)

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
        self.assertTrue(
            np.array_equal(coords, coords_out), "Coordinates are wrong."
        )

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
            dataset2[x_prop][int(row["index1"])]
            for _, row in result.iterrows()
        ]
        ly_closest = [
            dataset2[y_prop][int(row["index1"])]
            for _, row in result.iterrows()
        ]
        plt.plot(
            lx_closest,
            ly_closest,
            "bo",
            markersize=1.5,
            label="1st Closest points",
        )

        lx_closest = [
            dataset2[x_prop][int(row["index2"])]
            for _, row in result.iterrows()
        ]
        ly_closest = [
            dataset2[y_prop][int(row["index2"])]
            for _, row in result.iterrows()
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
        """Test of compute_curvature_at_point function."""
        curvature = compute_curvature(bend2)
        print(np.round(curvature, 4).tolist())
        self.assertSequenceEqual(
            bend2_curvs.tolist(), np.round(curvature, 4).tolist()
        )

    def test_compute_curvature_at_point(self: Self) -> None:
        """Test of compute_curvature_at_point function."""
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
        curv: npt.NDArray[np.float64] = np.sin(
            np.linspace(1, 100.0, 100) / 10.0
        )
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

    def test_compute_half_angle_variation(self: Self) -> None:
        """Test of compute_half_angle_variation function."""
        for k, (bend, exp) in enumerate(
            zip(
                (bend1, bend2, -1.0 * bend1, -1.0 * bend2, bend3),
                l_half_angle_index_exp,
                strict=True,
            ),
        ):
            ln = np.zeros_like(bend)
            for i, pt in enumerate(bend):
                if i == 0:
                    pt0 = pt
                    pt1 = bend[i + 1]
                elif i == bend.shape[0] - 1:
                    pt0 = bend[i - 1]
                    pt1 = pt
                else:
                    pt0 = bend[i - 1]
                    pt1 = bend[i + 1]
                ln[i] = normal(pt0 - pt1)
            ln *= -1.0

            index = compute_half_angle_variation(ln)

            # visual check
            scale_normal = 0.02
            plt.figure(dpi=150)
            plt.plot(bend[:, 0], bend[:, 1], "k-")
            for i, n in enumerate(ln):
                plt.arrow(
                    bend[i, 0],
                    bend[i, 1],
                    n[0] * scale_normal,
                    n[1] * scale_normal,
                    color="k",
                    width=0.005,
                    linewidth=0.1,
                )

            plt.plot(bend[index, 0], bend[index, 1], "ro", markersize=4)

            plt.axis("equal")
            plt.savefig(fig_path + f"bend_{k}.png", dpi=150)

            self.assertEqual(index, exp, f"Error at index {k}")

    def test_compute_median_curvature_index(self: Self) -> None:
        """Test of compute_median_curvature_index function."""
        for bend, i_exp in zip((bend1, bend1, bend2), (9, 9, 7), strict=False):
            curvature = compute_curvature(bend)
            index = compute_median_curvature_index(curvature, 2)
            print(index)
            self.assertEqual(index, i_exp)

    def test_compute_esperance(self: Self) -> None:
        """Test of compute_esperance function."""
        for bend, absc_exp in zip(
            (bend1, bend1, bend2), (0.619, 0.619, 1.152), strict=False
        ):
            curvature = compute_curvature(bend)
            curv_abs = compute_cuvilinear_abscissa(bend)
            absc = round(compute_esperance(curvature, curv_abs, 2), 3)
            print(absc)
            self.assertEqual(absc, absc_exp)

    def test_compute_variance(self: Self) -> None:
        """Test of compute_variance function."""
        for bend, absc_exp in zip(
            (bend1, bend1, bend2), (0.060, 0.060, 0.353), strict=False
        ):
            curvature = compute_curvature(bend)
            curv_abs = compute_cuvilinear_abscissa(bend)
            absc = round(compute_variance(curvature, curv_abs, 2)[0], 3)
            print(absc)
            self.assertEqual(absc, absc_exp)

    def test_compute_skewness(self: Self) -> None:
        """Test of compute_skewness function."""
        for bend, absc_exp in zip(
            (bend1, bend1, bend2), (0.287, 0.287, 1.067), strict=False
        ):
            curvature = compute_curvature(bend)
            curv_abs = compute_cuvilinear_abscissa(bend)
            absc = round(compute_skewness(curvature, curv_abs, 2), 3)
            print(absc)
            self.assertEqual(absc, absc_exp)

    def test_compute_kurtosis(self: Self) -> None:
        """Test of compute_kurtosis function."""
        for bend, absc_exp in zip(
            (bend1, bend1, bend2), (2.549, 2.549, 3.645), strict=False
        ):
            curvature = compute_curvature(bend)
            curv_abs = compute_cuvilinear_abscissa(bend)
            absc = round(compute_kurtosis(curvature, curv_abs, 2), 3)
            print(absc)
            self.assertEqual(absc, absc_exp)

    def test_compute_point_displacements(self: Self) -> None:
        """Test of compute_point_displacements function."""
        dir_trans = np.array((-1.0, -0.5))
        pts_in2 = [np.array(pt) for pt in pts_in]
        (disp_local, disp_tot) = compute_point_displacements(
            pts_in2, dir_trans
        )
        print(np.round(disp_local, 4).tolist())
        print(np.round(disp_tot, 4).tolist())
        disp_local = np.round(disp_local, 4)
        disp_tot = np.round(disp_tot, 4)
        disp_local_exp = [
            [0.0894, 2.415, 0.2, 2.4166],
            [5.6796, -1.8336, 0.3, 5.9682],
            [1.7441, 0.3578, 0.5, 1.7804],
            [0.1789, -0.7603, -0.4, 0.781],
            [0.9391, 0.7603, -0.1, 1.2083],
            [-0.5814, -2.5044, -0.2, 2.571],
            [0.1789, -0.313, -0.2, 0.3606],
            [0.3578, -0.6261, -0.1, 0.7211],
        ]
        disp_tot_exp = [8.5865, -2.5044, 0.0, 8.9443]
        for j in range(disp_local.shape[1]):
            self.assertSequenceEqual(disp_local[j].tolist(), disp_local_exp[j])
        self.assertSequenceEqual(disp_tot.tolist(), disp_tot_exp)


if __name__ == "__main__":
    unittest.main()
