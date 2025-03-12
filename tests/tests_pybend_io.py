# SPDX-FileCopyrightText: Copyright 2025 Martin Lemay <martin.lemay@mines-paris.org>
# SPDX-FileContributor: Martin Lemay
# ruff: noqa: E402 # disable Module level import not at top of file

__doc__ = """
Tests functions for pyBenD IO functions.

Run with input data:
  - centerline_xyz_test_data.csv
  - centerline_test_data.csv
  - centerline_collection_xyz_test_data.csv
  - centerline_collection_test_data.csv

"""

import os
import unittest
from typing import Self

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from pybend.algorithms.pybend_io import (
    create_dataset_from_xy,
    dump_centerline_to_csv,
    load_centerline_collection_dataset_from_Flumy_csv,
    load_centerline_dataset_from_csv,
    load_centerline_dataset_from_Flumy_csv,
    load_centerline_dataset_from_kml,
    load_centerline_evolution_from_multiple_xy_csv,
    load_centerline_evolution_from_single_xy_csv,
)
from pybend.model.Centerline import Centerline

# inputs
dir_path: str = "tests/data/"
filepath1: str = dir_path + "centerline_xyz_data.csv"
filepath2: str = dir_path + "centerline_flumy_data.csv"
filepath3: str = dir_path + "centerline_kml.kml"
filepath_cl_collection_flumy: str = dir_path + "centerline_Collection_test_data.csv"
map_file: dict[int, str] = {
    10: dir_path + "centerline_Collection_test_data10.csv",
    40: dir_path + "centerline_Collection_test_data40.csv",
    70: dir_path + "centerline_Collection_test_data70.csv",
}
filepath_cl_collection_xy: str = (
    dir_path + "end_members_from_Ghinassi2014_Expansion.csv"
)

# output directory
out_path: str = "tests/.out/"
# create it if absent
if not os.path.exists(out_path):
    os.makedirs(out_path)

filepath_out: str = out_path + "centerline_out.csv"


class TestsPybendIO(unittest.TestCase):
    def test_import_xyz_centerline1(self: Self) -> None:
        """Test of load_centerline_dataset_from_csv() function."""
        dataset1: pd.DataFrame = load_centerline_dataset_from_csv(
            filepath1, x_prop="X", y_prop="Y", z_prop=""
        )
        expected: list[str] = sorted(
            (
                "Curv_abscissa",
                "Cart_abscissa",
                "Cart_ordinate",
                "Elevation",
                "Z",
                "Curvature",
                "Vel_perturb",
                "Velocity",
                "Mean_depth",
                "True_elevation",
            )
        )

        self.assertTrue(dataset1 is not None)

        self.assertSequenceEqual(
            expected,
            sorted(dataset1.columns.tolist()),
            "Import xyz dataset - no z: columns are not well imported.",
        )

        self.assertIn(
            "Elevation",
            dataset1.columns,
            "Import xyz centerline - no z: Elevation must be in the dataframe.",
        )
        self.assertEqual(
            dataset1["Elevation"].sum(),
            0.0,
            "Import xyz dataset - no z: Default elevation is wrong",
        )
        self.assertSequenceEqual(
            (505, 10),
            dataset1.shape,
            "Import xyz dataset - " + "Number of rows and columns is wrong.",
        )

    def test_import_xyz_centerline2(self: Self) -> None:
        """Test of load_centerline_dataset_from_csv() function."""
        dataset2: pd.DataFrame = load_centerline_dataset_from_csv(
            filepath1, x_prop="X", y_prop="Y", z_prop="Z"
        )
        expected: list[str] = sorted(
            (
                "Curv_abscissa",
                "Cart_abscissa",
                "Cart_ordinate",
                "Elevation",
                "Curvature",
                "Vel_perturb",
                "Velocity",
                "Mean_depth",
                "True_elevation",
            )
        )

        self.assertTrue(dataset2 is not None)

        self.assertSequenceEqual(
            expected,
            sorted(dataset2.columns.tolist()),
            "Import xyz dataset - with z: columns are not well imported.",
        )
        self.assertSequenceEqual(
            (505, 9),
            dataset2.shape,
            "Import xyz dataset - " + "Number of rows and columns is wrong.",
        )

    def test_import_xyz_centerline3(self: Self) -> None:
        """Test of load_centerline_dataset_from_csv() function."""
        dataset3: pd.DataFrame = load_centerline_dataset_from_csv(
            filepath1,
            x_prop="X",
            y_prop="Y",
            z_prop="Z",
            drop_columns=("Mean_depth", "True_elevation"),
        )
        expected: list[str] = sorted(
            (
                "Curv_abscissa",
                "Cart_abscissa",
                "Cart_ordinate",
                "Elevation",
                "Curvature",
                "Vel_perturb",
                "Velocity",
            )
        )

        self.assertTrue(dataset3 is not None)

        self.assertSequenceEqual(
            expected,
            sorted(dataset3.columns.tolist()),
            "Import xyz dataset - with drop columns: columns are not well imported.",
        )
        self.assertSequenceEqual(
            (505, 7),
            dataset3.shape,
            "Import xyz dataset - " + "Number of rows and columns is wrong.",
        )

    def test_import_flumy_centerline(self: Self) -> None:
        """Test of load_centerline_dataset_from_Flumy_csv() function."""
        # right file format
        age, dataset = load_centerline_dataset_from_Flumy_csv(filepath2)
        self.assertEqual(age, 10, "Import Flumy centerline: Age is wrong.")
        expected: list[str] = sorted(
            (
                "Dist_previous",
                "Curv_abscissa",
                "Cart_abscissa",
                "Cart_ordinate",
                "Elevation",
                "Curvature",
                "Vel_perturb",
                "Velocity",
                "Mean_depth",
                "True_elevation",
            )
        )
        self.assertSequenceEqual(
            expected,
            sorted(dataset.columns.tolist()),
            "Import Flumy dataset: columns are not well imported.",
        )
        self.assertSequenceEqual(
            (505, 10),
            dataset.shape,
            "Import Flumy dataset - " + "Number of rows and columns is wrong.",
        )

    def test_create_dataset_from_kml(self: Self) -> None:
        """Test of load_centerline_dataset_from_kml() function."""
        dataset: pd.DataFrame = load_centerline_dataset_from_kml(filepath3)
        self.assertTrue(
            dataset is not None, "Import xml dataset: dataset was not loaded."
        )
        expected: list[str] = sorted(
            ("Curv_abscissa", "Cart_abscissa", "Cart_ordinate", "Elevation")
        )
        self.assertSequenceEqual(
            expected,
            sorted(dataset.columns.tolist()),
            "Import xml dataset - with drop columns: columns are not well imported.",
        )
        self.assertSequenceEqual(
            (505, 4),
            dataset.shape,
            "Import xml dataset - " + "Number of rows and columns is wrong.",
        )

    def test_create_dataset_from_xy(self: Self) -> None:
        """Test of load_centerline_dataset_from_csv() function."""
        dataset: pd.DataFrame = load_centerline_dataset_from_csv(
            filepath1, x_prop="X", y_prop="Y", z_prop="Z"
        )
        if dataset is None:
            self.skipTest(
                "Test test_create_dataset_from_xy was skipped because"
                + "load_centerline_dataset_from_csv failed"
            )

        dataset2: pd.DataFrame = create_dataset_from_xy(
            dataset["Cart_abscissa"].to_numpy(),
            dataset["Cart_ordinate"].to_numpy(),
        )
        self.assertTrue(
            dataset2 is not None,
            "Create dataset from xy: dataset was not correctly created.",
        )
        expected: list[str] = sorted(
            (
                "Curv_abscissa",
                "Cart_abscissa",
                "Cart_ordinate",
                "Elevation",
                "Curvature",
            )
        )
        self.assertSequenceEqual(
            expected,
            sorted(dataset2.columns.tolist()),
            "Create dataset from xy: dataset was not correctly created.",
        )

    def test_export_centerline(self: Self) -> None:
        """Test of load_centerline_dataset_from_csv() function."""
        dataset: pd.DataFrame = load_centerline_dataset_from_csv(
            filepath1, x_prop="X", y_prop="Y", z_prop="Z"
        )
        centerline: Centerline = Centerline(
            0,
            dataset,
            spacing=1,
            smooth_distance=5,
            compute_curvature=True,
            interpol_props=True,
            find_bends=False,
        )

        if centerline is None:
            self.skipTest(
                "Export Centerline: Centerline instance is null. Check Centerline creation tests."
            )

        dump_centerline_to_csv(filepath_out, centerline)
        self.assertTrue(
            os.path.exists(filepath_out),
            "Export Centerline: centerline data was not exported",
        )

        dataset2: pd.DataFrame = load_centerline_dataset_from_csv(
            filepath_out,
            x_prop="Cart_abscissa",
            y_prop="Cart_ordinate",
            z_prop="Elevation",
        )
        if dataset2 is None:
            self.skipTest("Reload exported centerline data failed.")

        expected: list[str] = sorted(
            (
                "Cart_abscissa",
                "Cart_ordinate",
                "Elevation",
                "Curvature",
                "Vel_perturb",
                "Velocity",
                "Mean_depth",
                "True_elevation",
                "Curv_abscissa",
                "Normal_x",
                "Normal_y",
                "Curvature_filtered",
                "Age",
            )
        )

        self.assertSequenceEqual(
            expected,
            sorted(dataset2.columns.tolist()),
            "Export Centerline: dataset was not well reloaded.",
        )

    def test_import_centerline_collection_multi_files(self: Self) -> None:
        """Test of load_centerline_evolution_from_multiple_xy_csv() function."""
        map_dataset: dict[int, pd.DataFrame] = (
            load_centerline_evolution_from_multiple_xy_csv(
                map_file,
                x_prop="Cart_abscissa",
                y_prop="Cart_ordinate",
                z_prop="Elevation",
                sep=";",
            )
        )
        self.assertTrue(
            map_dataset is not None,
            "Import centerline collection - multiple files: Loaded map is undefined",
        )
        self.assertEqual(
            len(map_dataset.keys()),
            3,
            "Import centerline collection - multiple files: Loaded map must have 3 keys.",
        )
        self.assertSequenceEqual(
            sorted(map_dataset.keys()),
            [10, 40, 70],
            "Import centerline collection - multiple files: Loaded ages must be [10, 40, 70]",
        )

        age: int = list(map_dataset.keys())[0]
        expected: list[str] = sorted(
            (
                "Iteration",
                "Dist_previous",
                "Curv_abscissa",
                "Cart_abscissa",
                "Cart_ordinate",
                "Elevation",
                "Curvature",
                "Vel_perturb",
                "Velocity",
                "Mean_depth",
                "True_elevation",
            )
        )
        self.assertSequenceEqual(
            expected,
            sorted(map_dataset[age].columns.tolist()),
            "Import centerline collection - multiple files: columns are not well loaded.",
        )

    def test_import_centerline_collection_single_file_flumy(
        self: Self,
    ) -> None:
        """Test of load_centerline_collection_dataset_from_Flumy_csv() function."""
        map_dataset: dict[int, pd.DataFrame] = (
            load_centerline_collection_dataset_from_Flumy_csv(
                filepath_cl_collection_flumy,
                sep=";",
            )
        )
        self.assertTrue(
            map_dataset is not None,
            "Import centerline collection - single file: Loaded map is undefined",
        )
        self.assertEqual(
            len(map_dataset.keys()),
            7,
            "Import centerline collection - single file: Loaded map must have 7 keys.",
        )
        expected: list[int] = list(np.arange(10, 71, 10).astype(int))
        self.assertSequenceEqual(
            sorted(map_dataset.keys()),
            expected,
            "Import centerline collection - single file: Loaded ages must be %s"
            % str(expected),
        )

        age: int = list(map_dataset.keys())[0]
        expected1: list[str] = sorted(
            (
                "Curv_abscissa",
                "Cart_abscissa",
                "Cart_ordinate",
                "Elevation",
                "Curvature",
                "Vel_perturb",
                "Velocity",
                "Mean_depth",
                "True_elevation",
            )
        )
        self.assertSequenceEqual(
            expected1,
            sorted(map_dataset[age].columns.tolist()),
            "Import centerline collection - single file: columns are not well loaded.",
        )

    def test_import_centerline_collection_single_file_csv(self: Self) -> None:
        """Test of load_centerline_collection_dataset_from_Flumy_csv() function."""
        map_dataset: dict[int, pd.DataFrame] = (
            load_centerline_evolution_from_single_xy_csv(
                filepath_cl_collection_xy,
                "X",
                "Y",
                age_prop="Index",
                sep=";",
            )
        )
        self.assertTrue(
            map_dataset is not None,
            "Import centerline collection - single file: Loaded map is undefined",
        )
        self.assertEqual(
            len(map_dataset.keys()),
            5,
            "Import centerline collection - single file: Loaded map must have 5 keys.",
        )
        expected: list[int] = list(np.arange(1, 6, 1).astype(int))
        self.assertSequenceEqual(
            sorted(map_dataset.keys()),
            expected,
            "Import centerline collection - single file: Loaded ages must be %s"
            % str(expected),
        )

        age: int = list(map_dataset.keys())[0]
        expected1: list[str] = sorted(
            (
                "Curv_abscissa",
                "Cart_abscissa",
                "Cart_ordinate",
                "Elevation",
            )
        )
        self.assertSequenceEqual(
            expected1,
            sorted(map_dataset[age].columns.tolist()),
            "Import centerline collection - single file: columns are not well loaded.",
        )


if __name__ == "__main__":
    unittest.main()
