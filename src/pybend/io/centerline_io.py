# SPDX-FileCopyrightText: Copyright 2025 Martin Lemay <martin.lemay@mines-paris.org>
# SPDX-FileContributor: Martin Lemay

"""Centerline I/O.

This module contains helpers to load and dump centerline-related datasets.
"""

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd  # type: ignore[import-untyped]
from typing_extensions import deprecated

import pybend.algorithms.centerline_process_functions as cpf
from pybend.io.common import CenterlineIOFormat, resolve_path
from pybend.model.Centerline import Centerline
from pybend.model.enumerations import PropertyNames


def load_centerline_from_file(
    filepath: str,
    kind: CenterlineIOFormat,
    **kwargs: Any,
) -> tuple[int, pd.DataFrame]:
    """Load a centerline from a file.

    Supported formats are: generic CSV, Flumy CSV, and KML.

    Args:
        filepath (str): Path to the file.
        kind (CenterlineIOFormat): Format of the file to load.
        **kwargs (Any): Loader-specific options.

            For ``CenterlineIOFormat.CSV``:
                - ``x_prop`` (str): X column name (default ``"X"``)
                - ``y_prop`` (str): Y column name (default ``"Y"``)
                - ``z_prop`` (str): Z/elevation column name (default ``"Z"``)
                - ``drop_columns`` (tuple[str, ...]): columns to drop (default
                    empty)
                - ``sep`` (str): separator (default ``";"``)

            For ``CenterlineIOFormat.FLUMY_CSV``:
                - ``sep`` (str): separator (default ``";"``)

            For ``CenterlineIOFormat.KML``:
                - ``keyword`` (str): keyword for coordinate line (default
                    ``"coordinates"``)

    Returns:
        tuple[int, pd.DataFrame]: Tuple containing the centerline age (if
        present) and the dataset as a DataFrame.
    """
    age: int = 0
    ext = filepath.split(".")[-1].lower()
    match kind:
        case CenterlineIOFormat.CSV:
            assert ext == "csv", (
                "File extension does not match specified format."
            )
            xprop = kwargs.get("x_prop", "X")
            yprop = kwargs.get("y_prop", "Y")
            zprop = kwargs.get("z_prop", "Z")
            dropcols = kwargs.get("drop_columns", ())
            sep = kwargs.get("sep", ";")
            dataset = load_centerline_dataset_from_csv(
                filepath,
                x_prop=xprop,
                y_prop=yprop,
                z_prop=zprop,
                drop_columns=dropcols,
                sep=sep,
            )
        case CenterlineIOFormat.FLUMY_CSV:
            assert ext == "csv", (
                "File extension does not match specified format."
            )
            sep = kwargs.get("sep", ";")
            age, dataset = load_centerline_dataset_from_Flumy_csv(
                filepath, sep=sep
            )
        case CenterlineIOFormat.KML:
            assert ext == "kml", (
                "File extension does not match specified format."
            )
            keyword = kwargs.get("keyword", "coordinates")
            dataset = load_centerline_dataset_from_kml(
                filepath, keyword=keyword
            )
        case _:
            raise ValueError(f"File format {kind} not supported.")
    return (age, dataset)


def load_centerline_dataset_from_csv(
    filepath: str,
    x_prop: str = "X",
    y_prop: str = "Y",
    z_prop: str = "Z",
    drop_columns: tuple[str, ...] = (),
    sep: str = ";",
) -> pd.DataFrame:
    """Load a dataset from a csv file containing cartesian coordinates.

    Coordinates must consist in at least x and y, and optionally centerline
    elevation and a list of properties

    Args:
        filepath (str): path to the csv file
        x_prop (str, optional): name of the column for x coordinate.

            Defaults to "X".
        y_prop (str, optional): name of the column for y coordinate.

            Defaults to "Y".
        z_prop (str, optional): name of the column for elevation.

            Defaults to "Z".
        drop_columns (tuple[str, ...], optional): list of the names of the
            columns to drop.

            Defaults is empty.
        sep (str, optional): csv separator.

            Defaults to ";".

    Returns:
        pd.DataFrame: DataFrame containing centerline coordinates and
            properties of each channel point

    """
    path = resolve_path(
        base_dir=None, raw_url=filepath, ctx="loading centerline"
    )
    dataset: pd.DataFrame = pd.read_csv(
        str(path), sep=sep, float_precision="round_trip"
    )

    assert x_prop in dataset.columns, (
        "X coordinate column indexes was not found."
    )
    assert y_prop in dataset.columns, (
        "Y coordinate column indexes was not found."
    )

    for col in drop_columns:
        dataset.drop(columns=col, inplace=True)

    dataset.rename(
        columns={
            x_prop: PropertyNames.CARTESIAN_ABSCISSA.value,
            y_prop: PropertyNames.CARTESIAN_ORDINATE.value,
        },
        inplace=True,
        copy=False,
    )
    if z_prop in dataset.columns:
        dataset.rename(
            columns={
                z_prop: PropertyNames.ELEVATION.value,
            },
            inplace=True,
            copy=False,
        )
    else:
        dataset[PropertyNames.ELEVATION.value] = 0.0

    dataset[PropertyNames.CURVILINEAR_ABSCISSA.value] = (
        cpf.compute_cuvilinear_abscissa(
            dataset.loc[
                :,
                (
                    PropertyNames.CARTESIAN_ABSCISSA.value,
                    PropertyNames.CARTESIAN_ORDINATE.value,
                ),
            ].to_numpy()  # type: ignore
        )
    )
    return dataset


def load_centerline_dataset_from_Flumy_csv(
    filepath: str, sep: str = ";"
) -> tuple[int, pd.DataFrame]:
    """Load a dataset from a csv file coming from Flumy simulation.

    Args:
        filepath (str): path to the csv file using Flumy format
        sep (str, optional): csv column delimiter.

            Defaults to ";".

    Returns:
        tuple[int, pd.DataFrame]: tuple containing the age as first component
            and a DataFrame containing centerline point coordinates and
            properties.

    """
    path = resolve_path(
        base_dir=None, raw_url=filepath, ctx="loading centerline"
    )
    data: pd.DataFrame = pd.read_csv(str(path), sep=sep)

    mess: str = (
        " property is missing. Try to use load_dataset_from_csv loader "
        + "instead."
    )

    assert PropertyNames.CARTESIAN_ABSCISSA.value in data.columns, (
        PropertyNames.CARTESIAN_ABSCISSA.value + mess
    )
    assert PropertyNames.CARTESIAN_ABSCISSA.value in data.columns, (
        PropertyNames.CARTESIAN_ABSCISSA.value + mess
    )
    assert PropertyNames.CARTESIAN_ORDINATE.value in data.columns, (
        PropertyNames.CARTESIAN_ORDINATE.value + mess
    )
    assert PropertyNames.ELEVATION.value in data.columns, (
        PropertyNames.ELEVATION.value + mess
    )
    assert PropertyNames.CURVATURE.value in data.columns, (
        PropertyNames.CURVATURE.value + mess
    )
    assert "Iteration" in data.columns, "Iteration" + mess

    assert data["Iteration"].unique().size == 1, (
        "Selected file contains several centerlines. "
        + "Use load_centerline_collection_dataset_from_Flumy_csv instead."
    )

    age = int(data["Iteration"].unique()[0])
    return age, data.drop(columns="Iteration")


def load_centerline_dataset_from_kml(
    filepath: str, keyword: str = "coordinates"
) -> pd.DataFrame:
    """Load a dataset from a kml file containing centerline point coordinates.

    Args:
        filepath (str): path to the kml file
        keyword (str, optional): keyword to search for coordinate line.

            Defaults to "coordinates".

    Returns:
        pd.DataFrame: DataFrame containing centerline point coordinates.

    """
    path = resolve_path(
        base_dir=None, raw_url=filepath, ctx="loading centerline"
    )
    coords_all = []
    with open(str(path), "r") as fin:
        for line in fin:
            if keyword not in line:
                continue
            line_split = line.split(" ")
            for elt in line_split:
                if len(elt) == 0:
                    continue
                if (">" in elt) or ("<" in elt):
                    if "</" in elt:
                        # last pts
                        elt = elt.split("<")[0]
                    else:
                        # 1st point
                        elt = elt.split(">")[-1]

                if "," in elt:
                    coords = elt.split(",")
                    coords_all += [coords]

    columns = (
        PropertyNames.CARTESIAN_ABSCISSA.value,
        PropertyNames.CARTESIAN_ORDINATE.value,
        PropertyNames.ELEVATION.value,
    )
    nb_pts = len(coords_all)
    assert nb_pts > 0, "Point coordinates were not found."

    nb_col = len(coords_all[0])
    data = np.zeros((nb_pts, nb_col))
    for i, coords in enumerate(coords_all):
        coords2 = [eval(val) for val in coords]
        data[i] = coords2

    dataset = pd.DataFrame(data, columns=columns[:nb_col])
    if nb_col == 2:
        dataset[PropertyNames.ELEVATION.value] = 0.0
    dataset[PropertyNames.CURVILINEAR_ABSCISSA.value] = (
        cpf.compute_cuvilinear_abscissa(
            dataset.loc[
                :,
                (
                    PropertyNames.CARTESIAN_ABSCISSA.value,
                    PropertyNames.CARTESIAN_ORDINATE.value,
                ),
            ].to_numpy()  # type: ignore
        )
    )
    return dataset


@deprecated("Use load_centerline_dataset_from_csv instead.")
def create_dataset_from_xy(
    X: npt.NDArray[np.float64], Y: npt.NDArray[np.float64]
) -> pd.DataFrame:
    """Create a dataset from X and Y 1D arrays.

    Args:
        X (npt.NDArray[np.float64]): X coordinates
        Y (npt.NDArray[np.float64]): Y coordinates

    Returns:
        pd.DataFrame: DataFrame with centerline point coordinates and
            properties.

    """
    data = np.zeros((X.size, 5))
    data[:, 1] = X
    data[:, 2] = Y
    data[:, 0] = cpf.compute_cuvilinear_abscissa(data[:, 1:3])
    for i in range(1, data.shape[0] - 1, 1):
        pt1 = data[i - 1, 1:3]
        pt2 = data[i, 1:3]
        pt3 = data[i + 1, 1:3]
        data[i, 4] = cpf.compute_curvature_at_point(pt1, pt2, pt3)

    dataset = pd.DataFrame(
        data,
        columns=(
            PropertyNames.CURVILINEAR_ABSCISSA.value,
            PropertyNames.CARTESIAN_ABSCISSA.value,
            PropertyNames.CARTESIAN_ORDINATE.value,
            PropertyNames.ELEVATION.value,
            PropertyNames.CURVATURE.value,
        ),
    )
    return dataset


def dump_centerline_to_csv(
    filepath: str, centerline: Centerline, sep: str = ";"
) -> None:
    """Write a csv file containing centerline data.

    Args:
        filepath (str): path to write the csv file
        centerline (Centerline): Centerline object to dump
        sep (str, optional): csv separator.

            Defaults to ";".

    """
    p = Path(filepath)
    parent = p.parent
    path = resolve_path(
        base_dir=None, raw_url=str(parent), ctx="saving centerline"
    )
    columns = centerline.cl_points[0].get_data().index.tolist() + [
        PropertyNames.AGE.value
    ]
    nrows = len(centerline.cl_points)
    data = pd.DataFrame(np.zeros((nrows, len(columns))), columns=columns)
    data[PropertyNames.AGE.value] = centerline.age
    for i, cl_pt in enumerate(centerline.cl_points):
        data.loc[i, cl_pt.get_data().index] = cl_pt.get_data()

    data.to_csv(str(path), sep=sep, index=False)
