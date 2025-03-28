# SPDX-FileCopyrightText: Copyright 2025 Martin Lemay <martin.lemay@mines-paris.org>
# SPDX-FileContributor: Martin Lemay
# ruff: noqa: E402 # disable Module level import not at top of file

__doc__ = """
Tests functions for centerline_process_functions.py.
"""

import os
import unittest
from typing import Self

import numpy as np
import numpy.typing as npt
import pandas as pd  # type: ignore[import-untyped]

from pybend.algorithms.geometry_functions import (
    compute_colinear,
    distance,
    distance_arrays,
    get_angle_between_vectors,
    get_MP,
    normal,
    orthogonal_distance,
    perp,
    project_orthogonal,
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

pt1: tuple[float, float] = (0.0, 0.0)
pt2: tuple[float, float] = (1.0, 0.0)
pt11: npt.NDArray[np.float64] = np.array((0.0, 0.0))
pt12: npt.NDArray[np.float64] = np.array((5.0, 0.0))
pt21: npt.NDArray[np.float64] = np.array((1.0, 1.0))
pt22: npt.NDArray[np.float64] = np.array((1.0, -2.0))
k: int = 5
pts1_curv: tuple[npt.NDArray[np.float64], ...] = (
    np.array((0.0, 0.0)),
    np.array((1.0, 0.0)),
    np.array((1.0, 0.0)),
)
pts2_curv: tuple[npt.NDArray[np.float64], ...] = (
    np.array((1.0, 0.0)),
    np.array((0.0, 0.0)),
    np.array((0.0, 0.0)),
)
pts3_curv: tuple[npt.NDArray[np.float64], ...] = (
    np.array((2.0, 0.0)),
    np.array((0.0, 1.0)),
    np.array((0.0, -1.0)),
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

# expected results
coords_out: npt.NDArray[np.float64] = np.array(pts_in).reshape(
    len(pts_in), len(pts_in[0])
)
curv_abscissa_out = np.array(
    [0.0, 2.4166, 8.3848, 10.1652, 10.9462, 12.1545, 14.7255, 15.0861, 15.8072]
)
pt_out_colinear: npt.NDArray[np.float64] = np.array((5.0, 0.0))
pt_out1: npt.NDArray[np.float64] = np.array((5.0, 5.0))
perp_out: npt.NDArray[np.float64] = np.array((-1.0, 1.0))
pt_out_intersect: npt.NDArray[np.float64] = np.array((1.0, 0.0))
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

    def test_compute_colinear(self: Self) -> None:
        """Test of compute_colinear() function."""
        pt = compute_colinear(pt11, pt21, k)
        self.assertTrue(
            np.array_equal(pt, pt_out1), "Point coordinate is wrong."
        )

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
        d2 = [
            distance(pt1, pt2, prec=4)
            for pt1, pt2 in zip(pts1, pts2, strict=False)
        ]
        self.assertTrue(np.array_equal(d1, d2), "Distances are different.")

    def test_orthogonal_distance(self: Self) -> None:
        """Test of orthogonal_distance() function."""
        d = orthogonal_distance(pt21, pt11, pt12, 4)
        d_exp = 1.0
        self.assertAlmostEqual(d, d_exp, 4)

    def test_perp(self: Self) -> None:
        """Test of perp() function."""
        vec_in = pt21 - pt11
        vec = perp(vec_in)
        self.assertTrue(np.array_equal(vec, perp_out))

    def test_normal(self: Self) -> None:
        """Test of normal() function."""
        vec_in = pt21 - pt11
        vec = normal(vec_in)
        vec_exp = perp_out / np.linalg.norm(perp_out)
        self.assertTrue(np.array_equal(vec, vec_exp))

    def test_seg_intersect(self: Self) -> None:
        """Test of seg_intersect() function."""
        pt = seg_intersect(pt11, pt12, pt21, pt22)
        self.assertTrue(np.array_equal(pt, pt_out_intersect))

    def test_project_orthogonal(self: Self) -> None:
        """Test of project_orthogonal() function."""
        pt_proj = project_orthogonal(pt21, pt11, pt12)
        self.assertAlmostEqual(pt_proj[0], pt_out_intersect[0], 3)
        self.assertAlmostEqual(pt_proj[1], pt_out_intersect[1], 3)

    def test_get_angle_between_vectors(self: Self) -> None:
        """Test of get_angle_between_vectors function."""
        vec1 = np.array((-1, 1))
        vec2_1 = np.array((1.0, 0.0))
        vec2_2 = np.array((1.0, 1.0))
        vec2_3 = np.array((0.0, 1.0))
        vec2_4 = np.array((-1.0, 1.0))
        vec2_5 = np.array((-1.0, 0.0))
        vec2_6 = np.array((-1.0, -1.0))
        vec2_7 = np.array((0.0, -1.0))
        vec2_8 = np.array((1.0, -1.0))
        vec2_all = (
            vec2_1,
            vec2_2,
            vec2_3,
            vec2_4,
            vec2_5,
            vec2_6,
            vec2_7,
            vec2_8,
        )

        expected = np.round(
            np.array((225.0, 270.0, 315.0, 0.0, 45.0, 90.0, 135.0, 180.0))
            * np.pi
            / 180.0,
            4,
        )
        for vec2, teta_exp in zip(vec2_all, expected, strict=True):
            teta = round(get_angle_between_vectors(vec1, vec2), 4)
            self.assertAlmostEqual(teta, teta_exp)

    def test_get_MP(self: Self) -> None:
        """Test of get_MP function."""
        dir_trans = np.array((1.0, 1.0))
        mat = get_MP(dir_trans)
        mat_exp = np.sqrt(2) / 2.0 * np.array(((1.0, 1.0), (-1.0, 1.0)))
        print(mat)
        print(mat_exp)
        self.assertSequenceEqual(
            np.round(mat.flatten(), 4).tolist(),
            np.round(mat_exp.flatten(), 4).tolist(),
        )


if __name__ == "__main__":
    unittest.main()
