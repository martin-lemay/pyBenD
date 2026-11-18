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
import tempfile
import unittest
from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from pybend.io.centerline_collection_io import (
    load_centerline_collection_dataset_from_Flumy_csv,
    load_centerline_evolution_from_multiple_kml,
    load_centerline_evolution_from_multiple_xy_csv,
    load_centerline_evolution_from_single_xy_csv,
)
from pybend.io.centerline_io import (
    EARTH_RADIUS,
    create_dataset_from_xy,
    dump_centerline_to_csv,
    load_centerline_dataset_from_csv,
    load_centerline_dataset_from_Flumy_csv,
    load_centerline_dataset_from_kml,
)
from pybend.io.common import normalize_age, resolve_path
from pybend.model.Centerline import Centerline

# inputs
dir_path: str = os.getcwd() + "/tests/data/"
filepath1: str = dir_path + "centerline_xyz_data.csv"
filepath2: str = dir_path + "centerline_flumy_data.csv"
filepath3: str = dir_path + "centerline_kml.kml"
filepath_cl_collection_flumy: str = (
    dir_path + "centerline_Collection_test_data.csv"
)
map_file: dict[int, str] = {
    10: dir_path + "centerline_Collection_test_data10.csv",
    40: dir_path + "centerline_Collection_test_data40.csv",
    70: dir_path + "centerline_Collection_test_data70.csv",
}
filepath_cl_collection_xy: str = (
    dir_path + "end_members_from_Ghinassi2014_Expansion.csv"
)

# output directory
out_path: str = os.getcwd() + "/tests/.out/"
# create it if absent
if not os.path.exists(out_path):
    os.makedirs(out_path)

filepath_out: str = out_path + "centerline_out.csv"


class TestsPybendIO(unittest.TestCase):
    def test_resolve_path_absolute_existing(self: Self) -> None:
        """resolve_path() returns the same absolute path when it exists."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            fpath = tmp_path / "test.csv"
            fpath.write_text("a;b\n1;2\n", encoding="utf-8")

            resolved = resolve_path(
                base_dir=None, raw_url=str(fpath), ctx="test"
            )
            self.assertEqual(fpath, resolved)

    def test_resolve_path_absolute_missing_raises(self: Self) -> None:
        """resolve_path() raises for missing absolute path."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            missing = tmp_path / "missing.csv"
            with self.assertRaises(FileNotFoundError):
                resolve_path(base_dir=None, raw_url=str(missing), ctx="test")

    def test_resolve_path_relative_with_base_dir(self: Self) -> None:
        """resolve_path() resolves relative paths against base_dir."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            rel = Path("sub") / "file.csv"
            full = base / rel
            full.parent.mkdir(parents=True, exist_ok=True)
            full.write_text("a;b\n1;2\n", encoding="utf-8")

            resolved = resolve_path(
                base_dir=base, raw_url=str(rel), ctx="test"
            )
            self.assertEqual(full.resolve(), resolved)

    def test_resolve_path_relative_without_base_dir_uses_cwd(
        self: Self,
    ) -> None:
        """resolve_path() resolves relative paths against the CWD."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            rel = Path("file.csv")
            (tmp_path / rel).write_text("a;b\n1;2\n", encoding="utf-8")

            old_cwd = os.getcwd()
            try:
                os.chdir(str(tmp_path))
                resolved = resolve_path(
                    base_dir=None, raw_url=str(rel), ctx="test"
                )
                self.assertEqual((tmp_path / rel).resolve(), resolved)
            finally:
                os.chdir(old_cwd)

    def test_resolve_path_relative_missing_raises(self: Self) -> None:
        """resolve_path() raises for missing relative path."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            old_cwd = os.getcwd()
            try:
                os.chdir(tmp_dir)
                with self.assertRaises(FileNotFoundError):
                    resolve_path(
                        base_dir=None, raw_url="missing.csv", ctx="test"
                    )
            finally:
                os.chdir(old_cwd)

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
            "Import xyz centerline, no z: Elevation must be in the dataframe.",
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
            "Import xyz dataset - with drop columns: columns are not well "
            + "imported.",
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
            "Import xml dataset - with drop columns: columns are not well "
            + "imported.",
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
                "Export Centerline: Centerline instance is null. Check "
                + "Centerline creation tests."
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
        """Test of load_centerline_evolution_from_multiple_xy_csv function."""
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
            "Import centerline collection - multiple files: Loaded map is "
            + "undefined",
        )
        self.assertEqual(
            len(map_dataset.keys()),
            3,
            "Import centerline collection - multiple files: Loaded map must "
            + " have 3 keys.",
        )
        self.assertSequenceEqual(
            sorted(map_dataset.keys()),
            [10, 40, 70],
            "Import centerline collection - multiple files: Loaded ages must "
            + "be [10, 40, 70]",
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
            "Import centerline collection - multiple files: columns are not "
            + "well loaded.",
        )

    def test_import_centerline_collection_single_file_flumy(
        self: Self,
    ) -> None:
        """Test load_centerline_collection_dataset_from_Flumy_csv function."""
        map_dataset: dict[int, pd.DataFrame] = (
            load_centerline_collection_dataset_from_Flumy_csv(
                filepath_cl_collection_flumy,
                sep=";",
            )
        )
        self.assertTrue(
            map_dataset is not None,
            "Import centerline collection - single file: Loaded map is "
            + "undefined",
        )
        self.assertEqual(
            len(map_dataset.keys()),
            7,
            "Import centerline collection - single file: Loaded map must have "
            + "7 keys.",
        )
        expected: list[int] = list(np.arange(10, 71, 10).astype(int))
        self.assertSequenceEqual(
            sorted(map_dataset.keys()),
            expected,
            "Import centerline collection -single file: Loaded ages must be %s"
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
            "Import centerline collection - single file: columns are not well "
            + "loaded.",
        )

    def test_import_centerline_collection_single_file_csv(self: Self) -> None:
        """Test load_centerline_collection_dataset_from_Flumy_csv function."""
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
            "Import centerline collection - single file: Loaded map is "
            + "undefined",
        )
        self.assertEqual(
            len(map_dataset.keys()),
            5,
            "Import centerline collection - single file: Loaded map must have "
            + "5 keys.",
        )
        expected: list[int] = list(np.arange(1, 6, 1).astype(int))
        self.assertSequenceEqual(
            sorted(map_dataset.keys()),
            expected,
            "Import centerline collection -single file: Loaded ages must be %s"
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
            "Import centerline collection - single file: columns are not well "
            + "loaded.",
        )


_KML_TEMPLATE: str = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document><Placemark><LineString>
<coordinates>
{coords}
</coordinates>
</LineString></Placemark></Document></kml>
"""


def _write_kml(directory: str, name: str, coords: str) -> str:
    """Write a minimal KML file.

    Args:
        directory: Output directory.
        name: File name.
        coords: Text of the coordinates element.

    Returns:
        Path of the written file.
    """
    path = os.path.join(directory, name)
    with open(path, "w", encoding="utf-8") as fout:
        fout.write(_KML_TEMPLATE.format(coords=coords))
    return path


class TestsKmlAndAges(unittest.TestCase):
    """Tests of the KML reader and age normalisation."""

    def test_kml_reference_values(self: Self) -> None:
        """KML reader parses all points of the reference file."""
        dataset = load_centerline_dataset_from_kml(filepath3)
        self.assertAlmostEqual(dataset["Cart_abscissa"].iloc[0], 0.0)
        self.assertAlmostEqual(
            dataset["Cart_abscissa"].iloc[1], 0.984807753012208
        )
        self.assertAlmostEqual(
            dataset["Cart_ordinate"].iloc[-1], -4.678087313719285
        )

    def test_kml_multiline_whitespace_2d(self: Self) -> None:
        """Tuples split over lines/tabs, 2D tuples, several elements."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = _write_kml(tmp_dir, "a.kml", "1.0,2.0\n\t3.5,4.5   \n5e0,6")
            dataset = load_centerline_dataset_from_kml(path)
        self.assertEqual(dataset.shape, (3, 4))
        self.assertTrue(np.allclose(dataset["Cart_abscissa"], [1.0, 3.5, 5.0]))
        self.assertTrue(np.allclose(dataset["Cart_ordinate"], [2.0, 4.5, 6.0]))
        self.assertTrue(np.allclose(dataset["Elevation"], 0.0))

    def test_kml_no_eval(self: Self) -> None:
        """Malicious content is not evaluated."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = _write_kml(
                tmp_dir, "a.kml", "__import__('os').getcwd(),1.0 2.0,3.0"
            )
            with self.assertRaises(ValueError):
                load_centerline_dataset_from_kml(path)

    def test_kml_to_metric(self: Self) -> None:
        """Lon/lat degrees are projected to local meters."""
        lat0 = 45.0
        coords = f"2.0,{lat0},0 2.01,{lat0},0 2.0,{lat0 + 0.01},0"
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = _write_kml(tmp_dir, "a.kml", coords)
            dataset = load_centerline_dataset_from_kml(path, to_metric=True)
            dataset_deg = load_centerline_dataset_from_kml(path)
        dlon = EARTH_RADIUS * np.radians(0.01) * np.cos(np.radians(lat0))
        dlat = EARTH_RADIUS * np.radians(0.01)
        self.assertTrue(
            np.allclose(dataset["Cart_abscissa"], [0.0, dlon, 0.0])
        )
        self.assertTrue(
            np.allclose(dataset["Cart_ordinate"], [0.0, 0.0, dlat])
        )
        self.assertAlmostEqual(dataset["Cart_abscissa"].iloc[1], 786.3, 0)
        self.assertTrue(
            np.allclose(dataset_deg["Cart_abscissa"], [2.0, 2.01, 2.0])
        )

    def test_multiple_kml_directory_join(self: Self) -> None:
        """Directory without trailing separator is joined correctly."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            _write_kml(tmp_dir, "a.kml", "0,0 1,0 2,0")
            _write_kml(tmp_dir, "b.kml", "0,1 1,1 2,1")
            data = load_centerline_evolution_from_multiple_kml(
                tmp_dir, {1: "a.kml", 2: "b.kml"}
            )
        self.assertEqual(sorted(data.keys()), [1, 2])
        self.assertEqual(data[2].shape[0], 3)

    def test_normalize_age(self: Self) -> None:
        """Integer-valued ages are cast to int, others raise."""
        self.assertEqual(normalize_age(10.0), 10)
        self.assertIsInstance(normalize_age(np.float64(10.0)), int)
        self.assertEqual(normalize_age(np.int64(7)), 7)
        with self.assertRaises(ValueError):
            normalize_age(10.5)
        with self.assertRaises(ValueError):
            normalize_age(float("nan"))

    def test_single_csv_float_ages(self: Self) -> None:
        """Float ages are cast to int or rejected when not integer."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "cl.csv")
            pd.DataFrame(
                {
                    "X": [0.0, 1.0, 0.0, 1.0],
                    "Y": [0.0, 0.0, 1.0, 1.0],
                    "Age": [10.0, 10.0, 20.0, 20.0],
                }
            ).to_csv(path, sep=";", index=False)
            data = load_centerline_evolution_from_single_xy_csv(path)
            self.assertEqual(sorted(data.keys()), [10, 20])
            self.assertTrue(all(type(k) is int for k in data))

            pd.DataFrame(
                {
                    "X": [0.0, 1.0, 0.0, 1.0],
                    "Y": [0.0, 0.0, 1.0, 1.0],
                    "Age": [10.5, 10.5, 20.0, 20.0],
                }
            ).to_csv(path, sep=";", index=False)
            with self.assertRaises(ValueError):
                load_centerline_evolution_from_single_xy_csv(path)


if __name__ == "__main__":
    unittest.main()
