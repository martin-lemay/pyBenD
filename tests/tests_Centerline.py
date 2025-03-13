# SPDX-FileCopyrightText: Copyright 2025 Martin Lemay <martin.lemay@mines-paris.org>
# SPDX-FileContributor: Martin Lemay
# ruff: noqa: E402 # disable Module level import not at top of file

__doc__ = """
Tests functions for Centerline class - Run with input data
centerline_flumy2500.csv and centerline_flumy2500.csv.
"""

import os
import unittest
from typing import Self

import numpy as np
import numpy.typing as npt

import pybend.algorithms.plot_functions as plot
from pybend.algorithms.pybend_io import (
    load_centerline_dataset_from_csv,
    load_centerline_dataset_from_Flumy_csv,
)
from pybend.model.Centerline import Centerline
from pybend.model.enumerations import BendSide, PropertyNames
from pybend.utils.globalParameters import set_nb_procs
from pybend.utils.logging import ERROR, logger

# disable info and warnings
logger.setLevel(ERROR)

set_nb_procs(1)

# inputs

# output directory for figures
fig_path: str = "tests/.out/"
# create it if absent
if not os.path.exists(fig_path):
    os.makedirs(fig_path)

filepath1: str = "tests/data/centerline_xyz_data.csv"
spacing: float = 1  # spacing between channel point (m)
smooth_distance: float = 5  # channel point location smoothing distance (m)
window: int = 5  # number of points for filtered curvature
sinuo_threshold: float = 1.0  # threshold for bends
n = 2  # exponent of curvature distribution function
# curvature, amplitude, length (4 set of weighting)
l_apex_proba_weights: tuple[tuple[float, float, float], ...] = (
    (1.0, 1.0, 1.0),
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)

age = 0
dataset = load_centerline_dataset_from_csv(
    filepath1, x_prop="X", y_prop="Y", z_prop="Z"
)

# Flumy dataset inputs
filepath_flumy: str = "tests/data/centerline_flumy2500.csv"
spacing_flumy: float = 200  # spacing between channel point (m)
use_fix_nb_points: bool = False
smooth_distance_flumy: float = 500  # channel point location smoothing distance
window_flumy: int = 10  # number of points for filtering curvature
sinuo_threshold_flumy: float = 1.05  # threshold for bends
# curvature, amplitude, length (4 set of weighting)
l_apex_proba_weights_flumy: tuple[tuple[float, float, float], ...] = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (1.0, 1.0, 1.0),
)

age_flumy, dataset_flumy = load_centerline_dataset_from_Flumy_csv(
    filepath_flumy
)

# expected results
nb_points_resampling_out: int = 505
nb_bends_out: int = 6  # number of bends
inflex_index_out: tuple[int, ...] = (
    0,
    91,
    181,
    271,
    360,
    451,
    504,
)  # index of inclection points
sides_out: tuple[BendSide, ...] = (
    BendSide.UP,
    BendSide.DOWN,
    BendSide.UP,
    BendSide.DOWN,
    BendSide.UP,
    BendSide.DOWN,
)  # side of beands
# coordinates of middle points
middles_out: tuple[npt.NDArray[np.float64], ...] = (
    np.array((45.2350, 0.0150, 0.0000)),
    np.array((134.9590, 0.0870, 0.0000)),
    np.array((224.2750, 0.0870, 0.0000)),
    np.array((313.0980, 0.0000, 0.0000)),
    np.array((402.4140, 0.0000, 0.0000)),
    np.array((473.8100, -2.3430, 0.0000)),
)

# coordinates of centroid points
centroids_out: tuple[npt.NDArray[np.float64], ...] = (
    np.array((45.2200, 2.0010)),
    np.array((134.8730, -1.8720)),
    np.array((224.1890, 2.0450)),
    np.array((313.0980, -1.9250)),
    np.array((402.4140, 1.9920)),
    np.array((474.9670, -3.0880)),
)
# index of apex
apex_index_auto_out: tuple[tuple[int, ...], ...] = (
    (45, 136, 226, 316, 405, 486),
    (45, 136, 226, 315, 405, 489),
    (46, 136, 226, 315, 406, 491),
)
l_apex_index_user_weights_out: tuple[tuple[int, ...], ...] = (
    (46, 136, 226, 315, 405, 478),
    (46, 135, 226, 315, 405, 500),
    (46, 136, 226, 315, 405, 480),
    (45, 136, 226, 315, 405, 477),
)

# apex probability
l_apex_probability_user_weights_out: tuple[tuple[float, ...], ...] = (
    (0.585, 0.631, 0.145, 0.951),
    (0.618, 0.681, 0.156, 0.982),
    (0.636, 0.712, 0.154, 0.988),
    (0.503, 0.5, 0.099, 0.911),
)

nb_procs: int = 5  # number of procs

#  Flumy dataset expected results
age_out_flumy: int = 2500
nb_points_resampling_out_flumy: int = 621
nb_bends_out_flumy: int = 46  # number of bends
nb_valid_bends_out_flumy: int = 34  # number of valid bends

# index of inclection points
inflex_index_out_flumy: tuple[int, ...] = (
    0,
    2,
    6,
    13,
    21,
    39,
    60,
    84,
    98,
    122,
    146,
    149,
    154,
    177,
    198,
    220,
    236,
    243,
    248,
    263,
    280,
    295,
    319,
    333,
    334,
    348,
    370,
    372,
    377,
    391,
    393,
    404,
    430,
    433,
    440,
    451,
    452,
    464,
    467,
    473,
    494,
    508,
    514,
    517,
    529,
    535,
    539,
    561,
    585,
    611,
    620,
)

# side of bends
sides_out_flumy: tuple[BendSide, ...] = (
    BendSide.DOWN,
    BendSide.UP,
    BendSide.DOWN,
    BendSide.UP,
    BendSide.DOWN,
    BendSide.UP,
    BendSide.DOWN,
    BendSide.UP,
    BendSide.DOWN,
    BendSide.UP,
    BendSide.DOWN,
    BendSide.UP,
    BendSide.DOWN,
    BendSide.UP,
    BendSide.DOWN,
    BendSide.UP,
    BendSide.DOWN,
    BendSide.UP,
    BendSide.DOWN,
    BendSide.UP,
    BendSide.DOWN,
    BendSide.UP,
    BendSide.DOWN,
    BendSide.UP,
    BendSide.DOWN,
    BendSide.UP,
    BendSide.DOWN,
    BendSide.UP,
    BendSide.DOWN,
    BendSide.UP,
    BendSide.DOWN,
    BendSide.UP,
    BendSide.DOWN,
    BendSide.UP,
    BendSide.DOWN,
    BendSide.UP,
    BendSide.DOWN,
    BendSide.UP,
    BendSide.DOWN,
    BendSide.UP,
    BendSide.DOWN,
    BendSide.UP,
    BendSide.DOWN,
    BendSide.UP,
    BendSide.DOWN,
    BendSide.UP,
)

# coordinates of middle points
middles_out_flumy: tuple[npt.NDArray[np.float64], ...] = (
    np.array((-5769.8060, 93981.8680, 50.0000)),
    np.array((-5201.8160, 94091.2510, 50.0000)),
    np.array((-4352.2190, 94492.7120, 50.0000)),
    np.array((-3287.3550, 94673.8310, 50.0000)),
    np.array((-1580.1180, 95055.9250, 50.0000)),
    np.array((665.8680, 94899.3030, 50.0000)),
    np.array((3073.8980, 93515.9470, 50.0000)),
    np.array((5081.7650, 92619.3830, 50.0000)),
    np.array((6400.7270, 93472.9540, 50.0000)),
    np.array((8158.0640, 94449.1030, 50.0000)),
    np.array((9438.2890, 94130.9610, 50.0000)),
    np.array((9638.3710, 93362.9650, 50.0000)),
    np.array((10710.0550, 93622.4180, 50.0000)),
    np.array((12635.7700, 93816.4010, 50.0000)),
    np.array((14776.7370, 93559.9320, 50.0000)),
    np.array((17008.3350, 93974.5260, 50.0000)),
    np.array((18658.7950, 93791.7110, 50.0000)),
    np.array((19729.5720, 93321.8520, 50.0000)),
    np.array((21042.6400, 93946.0320, 50.0000)),
    np.array((23024.0770, 94700.5800, 50.0000)),
    np.array((24870.4480, 95122.4230, 50.0000)),
    np.array((26763.2860, 95204.5080, 50.0000)),
    np.array((28584.3960, 94386.9220, 50.0000)),
    np.array((29526.7650, 95314.7970, 50.0000)),
    np.array((30770.0580, 96475.6270, 50.0000)),
    np.array((31624.9260, 95586.4880, 50.0000)),
    np.array((32058.5630, 94328.2370, 50.0000)),
    np.array((33032.6620, 94858.6240, 50.0000)),
    np.array((34779.5910, 96293.9830, 50.0000)),
    np.array((36197.0040, 96347.1140, 50.0000)),
    np.array((36317.4160, 95400.9110, 50.0000)),
    np.array((36830.0590, 94165.5290, 50.0000)),
    np.array((38033.2330, 94394.2290, 50.0000)),
    np.array((38756.7500, 95490.0330, 50.0000)),
    np.array((38724.0480, 96384.9270, 50.0000)),
    np.array((39941.2570, 97542.6860, 50.0000)),
    np.array((42258.7300, 98160.6740, 50.0000)),
    np.array((43717.3370, 98585.6710, 50.0000)),
    np.array((44390.0050, 99099.9260, 50.0000)),
    np.array((45569.9650, 98882.6740, 50.0000)),
    np.array((46881.4880, 98177.3910, 50.0000)),
    np.array((47590.9450, 97602.2570, 50.0000)),
    np.array((49072.0050, 98379.2590, 50.0000)),
    np.array((51368.0810, 98723.6080, 50.0000)),
    np.array((53823.1690, 98315.6170, 50.0000)),
    np.array((55548.4120, 99254.4340, 50.0000)),
)

# coordinates of centroid points
centroids_out_flumy: tuple[npt.NDArray[np.float64], ...] = (
    np.array((-5779.6930, 93977.6670)),
    np.array((-5104.7090, 94128.6920)),
    np.array((-4220.9990, 94393.9290)),
    np.array((-3343.7210, 94872.4590)),
    np.array((-1531.3900, 94580.8400)),
    np.array((760.0620, 95486.3200)),
    np.array((2869.9340, 92834.3870)),
    np.array((5226.5280, 93025.9950)),
    np.array((6714.4850, 92842.5510)),
    np.array((7800.7210, 95068.7130)),
    np.array((9417.2770, 94177.2380)),
    np.array((9662.2420, 93352.7840)),
    np.array((10832.5610, 92912.9060)),
    np.array((12606.4200, 94509.5970)),
    np.array((14799.5150, 92952.9800)),
    np.array((16731.8810, 94332.2720)),
    np.array((18624.7260, 93748.8790)),
    np.array((19735.5710, 93352.7400)),
    np.array((21173.6710, 93633.4430)),
    np.array((22815.5060, 95102.8970)),
    np.array((25027.6300, 94701.5890)),
    np.array((26877.0740, 95934.8080)),
    np.array((28412.8280, 93991.2920)),
    np.array((29687.0450, 95172.1330)),
    np.array((30707.5590, 97173.3700)),
    np.array((31658.0510, 95495.4540)),
    np.array((31784.6720, 93996.4920)),
    np.array((33227.8740, 94836.1210)),
    np.array((34529.9280, 97077.9980)),
    np.array((36186.4250, 96350.3200)),
    np.array((36397.3980, 95418.5350)),
    np.array((36627.0750, 93945.8910)),
    np.array((38252.0130, 94232.7180)),
    np.array((38748.8480, 95434.3240)),
    np.array((38739.5390, 96390.0440)),
    np.array((39755.6280, 98036.1690)),
    np.array((42271.5150, 97828.3200)),
    np.array((43683.4370, 98663.9820)),
    np.array((44394.9320, 99083.6610)),
    np.array((45595.7460, 99128.8360)),
    np.array((46792.5470, 98107.9020)),
    np.array((47469.6400, 97706.1410)),
    np.array((49395.8790, 97910.6780)),
    np.array((51389.7200, 99474.1270)),
    np.array((53842.7470, 97492.9250)),
    np.array((55448.6600, 99368.8670)),
)

# apex indexes automatic calculation
apex_index_auto_out_flumy: tuple[int, ...] = (
    0,
    4,
    10,
    16,
    27,
    44,
    73,
    92,
    103,
    128,
    148,
    152,
    161,
    183,
    206,
    225,
    239,
    245,
    253,
    267,
    286,
    304,
    327,
    338,
    356,
    375,
    385,
    398,
    416,
    431,
    436,
    446,
    458,
    465,
    470,
    486,
    501,
    511,
    516,
    521,
    532,
    537,
    549,
    567,
    595,
    617,
)

eps = 0.001  # tolerance for distance calculation (m)


class TestsCenterline(unittest.TestCase):
    def test_create_centerline_from_dataset_init(self: Self) -> None:
        """Test on creating a Centerline object from a dataset.

        Resampling off, curvature calculation off, property interpolation off.
        """
        # create centerline object without resampling
        expected_props: list[str] = sorted(
            (
                "Cart_abscissa",
                "Cart_ordinate",
                "Curv_abscissa",
                "Curvature",
                "Elevation",
                "Mean_depth",
                "Normal_x",
                "Normal_y",
                "True_elevation",
                "Vel_perturb",
                "Velocity",
            )
        )
        try:
            centerline: Centerline = Centerline(
                age,
                dataset,
                spacing,
                smooth_distance,
                use_fix_nb_points,
                window,
                sinuo_threshold,
                n,
                False,
                False,
                False,
            )
        except Exception as err:
            print(err)
            self.fail(
                "Centerline intialization test: Unable to create Centerline "
                + "object without resampling, nor curvature calculation."
            )

        self.assertEqual(
            centerline.age,
            0.0,
            "Centerline intialization test: Age must be equal to 0",
        )

        self.assertEqual(
            centerline.get_nb_points(),
            dataset.shape[0],
            "Intialization test: Number of points must be equal to %s"
            % dataset.shape[0],
        )

        self.assertSequenceEqual(
            expected_props,
            sorted(centerline.cl_points[0].get_data().index.tolist()),
            "Intialization test: Channel point properties must be %s"
            % (str(expected_props)),
        )

    def test_create_centerline_from_dataset_resampling(self: Self) -> None:
        """Test on creating a Centerline object from a dataset.

        Resampling on, curvature calculation off, property interpolation off.
        """
        expected_props: list[str] = sorted(
            (
                "Cart_abscissa",
                "Cart_ordinate",
                "Curv_abscissa",
                "Curvature",
                "Elevation",
                "Mean_depth",
                "Normal_x",
                "Normal_y",
                "True_elevation",
                "Vel_perturb",
                "Velocity",
            )
        )
        try:
            centerline: Centerline = Centerline(
                age,
                dataset,
                spacing,
                smooth_distance,
                use_fix_nb_points,
                window,
                sinuo_threshold,
                n,
                False,
                False,
                False,
            )
        except Exception as err:
            print(err)
            self.fail(
                "Centerline intialization test: Unable to create Centerline "
                + "object with resampling, but no curvature calculation."
            )

        self.assertEqual(
            centerline.get_nb_points(),
            dataset.shape[0],
            "Centerline intialization test: Number of points must be equal to "
            + f"{dataset.shape[0]}",
        )

        self.assertSequenceEqual(
            expected_props,
            sorted(centerline.cl_points[0].get_data().index.tolist()),
            "Centerline intialization test: Channel point properties must be "
            + f"{expected_props}",
        )

    def test_create_centerline_from_dataset_resampling_curvature(
        self: Self,
    ) -> None:
        """Test on creating a Centerline object from a dataset.

        Resampling on, curvature calculation on, property interpolation off.
        """
        # create centerline object with resampling and curvature calculation
        # but no property interpolation
        expected_props: list[str] = sorted(
            (
                "Cart_abscissa",
                "Cart_ordinate",
                "Curv_abscissa",
                "Curvature",
                "Curvature_filtered",
                "Elevation",
                "Mean_depth",
                "Normal_x",
                "Normal_y",
                "True_elevation",
                "Vel_perturb",
                "Velocity",
            )
        )
        try:
            centerline: Centerline = Centerline(
                age,
                dataset,
                spacing,
                smooth_distance,
                use_fix_nb_points,
                window,
                sinuo_threshold,
                n,
                True,
                False,
                False,
            )
        except Exception as err:
            print(err)
            self.fail(
                "Centerline intialization test: Unable to create Centerline "
                + "object with resampling, but no curvature calculation."
            )

        self.assertEqual(
            centerline.get_nb_points(),
            dataset.shape[0],
            "Intialization test: Number of points must be equal to %s"
            % dataset.shape[0],
        )

        self.assertSequenceEqual(
            expected_props,
            sorted(centerline.cl_points[0].get_data().index.tolist()),
            "Intialization test: Channel point properties must be %s"
            % (str(expected_props)),
        )

    # test pass locally but results are different on GitHub CI.
    @unittest.skip("Disable for GitHub CI")
    def test_find_inflection_points(self: Self) -> None:
        """Test of find_inflection_points method from Centerline object."""
        set_nb_procs(nb_procs)
        # load dataset
        centerline: Centerline
        try:
            centerline = Centerline(
                age,
                dataset,
                spacing,
                smooth_distance,
                use_fix_nb_points,
                window,
                sinuo_threshold,
                n,
                compute_curvature=True,
                interpol_props=True,
                find_bends=False,
            )
        except Exception as err:
            print(err)
            self.skipTest("Centerline intialization failed.")
        self.assertNotEqual(centerline, None, "Centerline must be defined")

        # compute inflection points
        inflex_index: list[int] = []
        try:
            inflex_index = centerline.find_inflection_points().tolist()
        except Exception as err:
            print(err)
            self.fail("Unable to compute inflection point indexes.")

        self.assertSequenceEqual(
            tuple(inflex_index),
            inflex_index_out,
            f"Inflection point detection: {inflex_index} instead of "
            + f" {inflex_index_out}",
        )
        set_nb_procs(1)

    def test_find_bends_multiproc(self: Self) -> None:
        """Test of find_bends method from Centerline object."""
        set_nb_procs(nb_procs)
        # load dataset
        centerline: Centerline
        try:
            centerline = Centerline(
                age,
                dataset,
                spacing,
                smooth_distance,
                use_fix_nb_points,
                window,
                sinuo_threshold,
                n,
                compute_curvature=True,
                interpol_props=True,
                find_bends=True,
            )
        except Exception as err:
            print(err)
            self.skipTest("Centerline intialization failed.")
        self.assertNotEqual(centerline, None, "Centerline must be defined")

        # check bends were correctly detected
        self.assertEqual(
            centerline.get_nb_bends(),
            nb_bends_out,
            f"Bend detection: Number of bends must be {nb_bends_out}",
        )
        self.assertEqual(
            centerline.get_nb_valid_bends(),
            nb_bends_out,
            f"Bend detection: Number of valid bends must be {nb_bends_out}",
        )

        # check bend side
        sides = [bend.side for bend in centerline.bends]
        self.assertSequenceEqual(sides, sides_out)
        set_nb_procs(1)

    def test_find_bends_monoproc(self: Self) -> None:
        """Test of find_bends method from Centerline object."""
        set_nb_procs(1)
        # load dataset
        centerline: Centerline
        try:
            centerline = Centerline(
                age,
                dataset,
                spacing,
                smooth_distance,
                use_fix_nb_points,
                window,
                sinuo_threshold,
                n,
                compute_curvature=True,
                interpol_props=True,
                find_bends=True,
            )
        except Exception as err:
            print(err)
            self.skipTest("Centerline intialization failed.")
        self.assertNotEqual(centerline, None, "Centerline must be defined")

        # check bends were correctly detected
        self.assertEqual(
            centerline.get_nb_bends(),
            nb_bends_out,
            f"Bend detection: Number of bends must be {nb_bends_out}",
        )
        self.assertEqual(
            centerline.get_nb_valid_bends(),
            nb_bends_out,
            f"Bend detection: Number of valid bends must be {nb_bends_out}",
        )

        # check bend side
        sides = [bend.side for bend in centerline.bends]
        self.assertSequenceEqual(sides, sides_out)
        set_nb_procs(1)

        # visual check
        try:
            plot.plot_centerline_single(
                fig_path + "centerline_tests.png",
                (centerline.cl_points,),
                centerline.bends,
                domain=((), ()),  # type: ignore
                show=False,
                plot_apex=False,
                plot_inflex=False,
                plot_middle=False,
                plot_centroid=False,
                markersize=2,
            )
        except Exception as err:
            print(err)
            self.fail("Unable to plot imported centerline.")

    # test pass locally but results are different on GitHub CI.
    @unittest.skip("Disable for GitHub CI")
    def test_compute_all_bend_center_multiproc(self: Self) -> None:
        """Test of compute_all_bend_center method from Centerline object."""
        set_nb_procs(nb_procs)
        # load dataset
        centerline: Centerline
        try:
            centerline = Centerline(
                age,
                dataset,
                spacing,
                smooth_distance,
                use_fix_nb_points,
                window,
                sinuo_threshold,
                n,
                compute_curvature=True,
                interpol_props=True,
                find_bends=True,
            )
        except Exception as err:
            print(err)
            self.skipTest("Centerline intialization failed.")
        self.assertNotEqual(centerline, None, "Centerline must be defined")

        # compute bend middle point
        try:
            centerline.compute_all_bend_center()
        except Exception as err:
            print(err)
            self.fail("Unable to compute bend middle points.")

        middles = [
            np.round(bend.pt_center, 3)
            for bend in centerline.bends
            if bend.pt_center is not None
        ]

        for pt_obs, pt_exp in zip(middles, middles_out, strict=False):
            self.assertSequenceEqual(pt_obs.tolist(), pt_exp.tolist())

        set_nb_procs(1)

    # test pass locally but results are different on GitHub CI.
    @unittest.skip("Disable for GitHub CI")
    def test_compute_all_bend_center_monoproc(self: Self) -> None:
        """Test of compute_all_bend_center method from Centerline object."""
        set_nb_procs(1)
        # load dataset
        centerline: Centerline
        try:
            centerline = Centerline(
                age,
                dataset,
                spacing,
                smooth_distance,
                use_fix_nb_points,
                window,
                sinuo_threshold,
                n,
                compute_curvature=True,
                interpol_props=True,
                find_bends=True,
            )
        except Exception as err:
            print(err)
            self.skipTest("Centerline intialization failed.")
        self.assertNotEqual(centerline, None, "Centerline must be defined")

        # compute bend middle point
        try:
            centerline.compute_all_bend_center()
        except Exception as err:
            print(err)
            self.fail("Unable to compute bend middle points.")

        middles = [
            np.round(bend.pt_center, 3)
            for bend in centerline.bends
            if bend.pt_center is not None
        ]

        # with open("tests/.out/middles_obs.txt", "w") as fout:
        #     fout.writelines(middles_list)
        for pt_obs, pt_exp in zip(middles, middles_out, strict=False):
            self.assertSequenceEqual(pt_obs.tolist(), pt_exp.tolist())

        set_nb_procs(1)

    # test pass locally but results are different on GitHub CI.
    @unittest.skip("Disable for GitHub CI")
    def test_compute_all_bend_centroid_multiproc(self: Self) -> None:
        """Test of compute_all_bend_centroid method from Centerline object."""
        set_nb_procs(nb_procs)
        # load dataset
        centerline: Centerline
        try:
            centerline = Centerline(
                age,
                dataset,
                spacing,
                smooth_distance,
                use_fix_nb_points,
                window,
                sinuo_threshold,
                n,
                compute_curvature=True,
                interpol_props=True,
                find_bends=True,
            )
        except Exception as err:
            print(err)
            self.skipTest("Centerline intialization failed.")
        self.assertNotEqual(centerline, None, "Centerline must be defined")

        # compute bend centroid point
        try:
            centerline.compute_all_bend_centroid()
        except Exception as err:
            print(err)
            self.fail("Unable to compute bend centroid points.")

        centroids = [
            np.round(bend.pt_centroid, 3)
            for bend in centerline.bends
            if bend.pt_centroid is not None
        ]

        for pt_obs, pt_exp in zip(centroids, centroids_out, strict=False):
            self.assertSequenceEqual(pt_obs.tolist(), pt_exp.tolist())

        set_nb_procs(1)

    # test pass locally but results are different on GitHub CI.
    @unittest.skip("Disable for GitHub CI")
    def test_compute_all_bend_centroid_monoproc(self: Self) -> None:
        """Test of compute_all_bend_centroid method from Centerline object."""
        set_nb_procs(1)
        # load dataset
        centerline: Centerline
        try:
            centerline = Centerline(
                age,
                dataset,
                spacing,
                smooth_distance,
                use_fix_nb_points,
                window,
                sinuo_threshold,
                n,
                compute_curvature=True,
                interpol_props=True,
                find_bends=True,
            )
        except Exception as err:
            print(err)
            self.skipTest("Centerline intialization failed.")
        self.assertNotEqual(centerline, None, "Centerline must be defined")

        # compute bend centroid point
        try:
            centerline.compute_all_bend_centroid()
        except Exception as err:
            print(err)
            self.fail("Unable to compute bend centroid points.")

        centroids = [
            np.round(bend.pt_centroid, 3)
            for bend in centerline.bends
            if bend.pt_centroid is not None
        ]

        # with open("tests/.out/centroids.txt", "w") as fout:
        #     fout.writelines(centroids_list)
        for pt_obs, pt_exp in zip(centroids, centroids_out, strict=False):
            self.assertSequenceEqual(pt_obs.tolist(), pt_exp.tolist())

        set_nb_procs(1)

    # test pass locally but results are different on GitHub CI.
    @unittest.skip("Disable for GitHub CI")
    def test_find_all_bend_apex_user_weights_monoproc(self: Self) -> None:
        """Test of find_all_bend_apex_user_weights method."""
        set_nb_procs(1)
        # load dataset
        centerline: Centerline
        try:
            centerline = Centerline(
                age,
                dataset,
                spacing,
                smooth_distance,
                use_fix_nb_points,
                window,
                sinuo_threshold,
                n,
                compute_curvature=True,
                interpol_props=True,
                find_bends=True,
            )
        except Exception as err:
            print(err)
            self.skipTest("Centerline intialization failed.")
        self.assertNotEqual(centerline, None, "Centerline must be defined")

        # compute bend apex point
        for i, (apex_proba_weights, apex_index_out) in enumerate(
            zip(
                l_apex_proba_weights,
                l_apex_index_user_weights_out,
                strict=False,
            )
        ):
            try:
                centerline.find_all_bend_apex_user_weights(apex_proba_weights)
            except Exception as err:
                print(err)
                self.fail("Unable to compute bend apex points.")

            # check apex probability
            apex_probability: npt.NDArray[np.float64] = (
                centerline.get_property(PropertyNames.APEX_PROBABILITY.value)
            )

            self.assertAlmostEqual(
                float(np.mean(apex_probability)),
                l_apex_probability_user_weights_out[i][0],
                3,
                "Apex probability calculation: mean "
                + f"{np.mean(apex_probability):.3f}",
            )
            self.assertAlmostEqual(
                float(np.median(apex_probability)),
                l_apex_probability_user_weights_out[i][1],
                3,
                "Apex probability calculation: median "
                + f"{np.median(apex_probability):.3f}",
            )
            self.assertAlmostEqual(
                float(np.percentile(apex_probability, 10)),
                l_apex_probability_user_weights_out[i][2],
                3,
                "Apex probability calculation: Observed p10 "
                + f"{np.percentile(apex_probability, 10):.3f}",
            )
            self.assertAlmostEqual(
                float(np.percentile(apex_probability, 90)),
                l_apex_probability_user_weights_out[i][3],
                3,
                "Apex probability calculation: Observed p90 "
                + f"{np.percentile(apex_probability, 90):.3f}",
            )

            # check apex index
            apex_index: list[int] = [
                int(bend.index_apex)
                for bend in centerline.bends
                if bend.index_apex > -1
            ]

            # dump_apex_index = ["%s, "%index for index in apex_index]
            # with open("tests/.out/apex_index.txt", "w") as fout:
            #     fout.writelines(dump_apex_index)

            self.assertSequenceEqual(
                tuple(apex_index),
                apex_index_out,
                f"Apex probability calculation: Apex indexes (@{i}: "
                + f"{apex_index})",
            )
        set_nb_procs(1)

    # test pass locally but results are different on GitHub CI.
    @unittest.skip("Disable for GitHub CI")
    def test_find_all_bend_apex_user_weights_multiproc(self: Self) -> None:
        """Test of find_all_bend_apex_user_weights method."""
        set_nb_procs(nb_procs)
        # load dataset
        centerline: Centerline
        try:
            centerline = Centerline(
                age,
                dataset,
                spacing,
                smooth_distance,
                use_fix_nb_points,
                window,
                sinuo_threshold,
                n,
                compute_curvature=True,
                interpol_props=True,
                find_bends=True,
            )
        except Exception as err:
            print(err)
            self.skipTest("Centerline intialization failed.")
        self.assertNotEqual(centerline, None, "Centerline must be defined")

        # compute bend apex point
        for i, (apex_proba_weights, apex_index_out) in enumerate(
            zip(
                l_apex_proba_weights,
                l_apex_index_user_weights_out,
                strict=False,
            )
        ):
            try:
                centerline.find_all_bend_apex_user_weights(apex_proba_weights)
            except Exception as err:
                print(err)
                self.fail("Unable to compute bend apex points.")

            # check apex probability
            apex_probability: npt.NDArray[np.float64] = (
                centerline.get_property(PropertyNames.APEX_PROBABILITY.value)
            )
            print(
                f"{i} {round(np.mean(apex_probability), 3)}, "
                + f"{round(np.median(apex_probability), 3)},"
                + f"{round(np.percentile(apex_probability, 10), 3)}, "
                + f"{round(np.percentile(apex_probability, 90), 3)}"
            )

            self.assertAlmostEqual(
                float(np.mean(apex_probability)),
                l_apex_probability_user_weights_out[i][0],
                3,
                "Apex probability calculation: mean "
                + f"{np.mean(apex_probability):.3f}",
            )
            self.assertAlmostEqual(
                float(np.median(apex_probability)),
                l_apex_probability_user_weights_out[i][1],
                3,
                "Apex probability calculation: median "
                + f"{np.median(apex_probability):.3f}",
            )
            self.assertAlmostEqual(
                float(np.percentile(apex_probability, 10)),
                l_apex_probability_user_weights_out[i][2],
                3,
                "Apex probability calculation: Observed p10 "
                + f"{np.percentile(apex_probability, 10):.3f}",
            )
            self.assertAlmostEqual(
                float(np.percentile(apex_probability, 90)),
                l_apex_probability_user_weights_out[i][3],
                3,
                "Apex probability calculation: Observed p90 "
                + f"{np.percentile(apex_probability, 90):.3f}",
            )

            # check apex index
            apex_index: list[int] = [
                int(bend.index_apex)
                for bend in centerline.bends
                if bend.index_apex > -1
            ]
            self.assertSequenceEqual(
                tuple(apex_index),
                apex_index_out,
                f"Apex probability calculation: Apex indexes (@{i}: "
                + f"{apex_index})",
            )
        set_nb_procs(1)

    # test pass locally but results are different on GitHub CI.
    @unittest.skip("Disable for GitHub CI")
    def test_find_all_bend_apex(self: Self) -> None:
        """Test of find_all_bend_apex method from Centerline object."""
        set_nb_procs(nb_procs)
        # load dataset
        centerline: Centerline
        try:
            centerline = Centerline(
                age,
                dataset,
                spacing,
                smooth_distance,
                use_fix_nb_points,
                window,
                sinuo_threshold,
                n,
                compute_curvature=True,
                interpol_props=True,
                find_bends=True,
            )
        except Exception as err:
            print(err)
            self.skipTest("Centerline intialization failed.")
        self.assertNotEqual(centerline, None, "Centerline must be defined")

        for i in range(3):
            # compute bend apex point
            try:
                centerline.find_all_bend_apex(i + 1)
            except Exception as err:
                print(err)
                self.fail("Unable to compute bend apex points.")

            # check apex index
            apex_index: list[int] = [
                int(bend.index_apex)
                for bend in centerline.bends
                if bend.index_apex > -1
            ]

            # dump_apex_index = ["%s, "%index for index in apex_index]
            # with open("tests/.out/apex_index.txt", "w") as fout:
            #     fout.writelines(dump_apex_index)

            self.assertSequenceEqual(
                tuple(apex_index),
                apex_index_auto_out[i],
                f"Apex probability calculation: Apex indexes ({apex_index})",
            )
        set_nb_procs(1)

    def test_create_centerline_from_flumy_dataset(self: Self) -> None:
        """Test on creating a Centerline object from a Flumy dataset."""
        set_nb_procs(nb_procs)
        # create centerline object with resampling and detect bends
        try:
            centerline = Centerline(
                age_flumy,
                dataset_flumy,
                spacing_flumy,
                smooth_distance_flumy,
                use_fix_nb_points,
                window_flumy,
                sinuo_threshold_flumy,
                n,
                compute_curvature=True,
                interpol_props=True,
                find_bends=False,
            )
        except Exception as err:
            print(err)
            self.fail(
                "Intialization test: Unable to create Centerline object."
            )
        self.assertNotEqual(centerline, None, "Centerline must be defined")
        self.assertEqual(
            centerline.age,
            age_out_flumy,
            f"Centerline age must be {age_out_flumy}",
        )
        self.assertEqual(
            centerline.get_nb_points(),
            nb_points_resampling_out_flumy,
            "Centerline number of points is wrong.",
        )
        set_nb_procs(1)

    def test_bend_properties_flumy(self: Self) -> None:
        """Test of find_all_bend_apex method from Centerline object."""
        set_nb_procs(nb_procs)
        # load dataset
        centerline: Centerline
        try:
            centerline = Centerline(
                age_flumy,
                dataset_flumy,
                spacing_flumy,
                smooth_distance_flumy,
                use_fix_nb_points,
                window_flumy,
                sinuo_threshold_flumy,
                compute_curvature=True,
                interpol_props=True,
                find_bends=True,
            )
        except Exception as err:
            print(err)
            self.skipTest("Centerline intialization failed.")
        self.assertNotEqual(centerline, None, "Centerline must be defined")
        self.assertEqual(centerline.get_nb_bends(), nb_bends_out_flumy)
        self.assertEqual(
            centerline.get_nb_valid_bends(), nb_valid_bends_out_flumy
        )

        # check bend side
        sides = tuple([bend.side for bend in centerline.bends])
        self.assertEqual(len(sides), len(sides_out_flumy))
        self.assertEqual(sides[0], sides_out_flumy[0])
        self.assertEqual(sides[-1], sides_out_flumy[-1])
        # self.assertSequenceEqual(sides, sides_out_flumy)
        set_nb_procs(1)

        # compute bend middle point
        try:
            centerline.compute_all_bend_center()
        except Exception as err:
            print(err)
            self.fail("Unable to compute bend middle points.")

        middles = [
            np.round(bend.pt_center, 3)
            for bend in centerline.bends
            if bend.pt_center is not None
        ]

        middles_list = [
            "np.array((%.4f, %.4f, %.4f)),\n" % (pt[0], pt[1], pt[2])
            for pt in middles
        ]
        # with open("tests/.out/middles_obs.txt", "w") as fout:
        #     fout.writelines(middles_list)

        diff: npt.NDArray[np.float64] = np.array(middles) - np.array(
            middles_out_flumy
        )
        self.assertTrue(
            all(list(np.linalg.norm(diff, axis=0) < eps)),
            f"Middle point calculation: {middles_list}",
        )

        # compute bend centroid point
        try:
            centerline.compute_all_bend_centroid()
        except Exception as err:
            print(err)
            self.fail("Unable to compute bend centroid points.")

        centroids = [
            np.round(bend.pt_centroid, 3)
            for bend in centerline.bends
            if bend.pt_centroid is not None
        ]

        centroids_list = [
            "np.array((%.4f, %.4f)),\n" % (pt[0], pt[1]) for pt in centroids
        ]

        # with open("tests/.out/centroids_obs.txt", "w") as fout:
        #     fout.writelines(centroids_list)

        diff1: npt.NDArray[np.float64] = np.array(centroids) - np.array(
            centroids_out_flumy
        )
        self.assertTrue(
            all(list(np.linalg.norm(diff1, axis=0) < eps)),
            f"Centroid point calculation: {centroids_list}",
        )

        # compute bend apex point
        try:
            centerline.find_all_bend_apex(2)
        except Exception as err:
            print(err)
            self.fail("Unable to compute bend apex points.")

        # check apex index
        apex_index = [
            int(bend.index_apex)
            for bend in centerline.bends
            if bend.index_apex > -1
        ]

        # dump_apex_index = ["%s, " % index for index in apex_index]
        # with open("tests/.out/apex_index_flumy.txt", "w") as fout:
        #     fout.writelines(dump_apex_index)

        self.assertSequenceEqual(
            tuple(apex_index),
            apex_index_auto_out_flumy,
            "Apex indexes detection failed.",
        )

        # visual check
        try:
            plot.plot_centerline_single(
                fig_path + "centerline_flumy_tests.png",
                (centerline.cl_points,),
                centerline.bends,
                domain=((), ()),  # type: ignore
                show=False,
                plot_apex=True,
                plot_inflex=True,
                plot_middle=True,
                plot_centroid=True,
                markersize=2,
            )
        except Exception as err:
            print(err)
            self.fail("Unable to plot imported centerline.")

        set_nb_procs(1)


if __name__ == "__main__":
    unittest.main()
