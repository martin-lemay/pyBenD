# SPDX-FileCopyrightText: Copyright 2025 Martin Lemay <martin.lemay@mines-paris.org>
# SPDX-FileContributor: Martin Lemay
# ruff: noqa: E402 # disable Module level import not at top of file

import numpy as np
import numpy.typing as npt
import pandas as pd  # type: ignore[import-untyped]
from typing_extensions import deprecated

import pybend.algorithms.centerline_process_function as cpf
from pybend.model.Centerline import Centerline
from pybend.model.enumerations import PropertyNames

__doc__ = """
IO methods for pyBenD.
"""


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

    Parameters:
    ----------
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
    -------
        pd.DataFrame: DataFrame containing centerline coordinates and
            properties of each channel point

    """
    dataset: pd.DataFrame = pd.read_csv(
        filepath, sep=sep, float_precision="round_trip"
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

    Parameters:
    ----------
        filepath (str): path to the csv file using Flumy format
        sep (str, optional): csv column delimiter.

            Defaults to ";".

    Returns:
    --------
        tuple[int, pd.DataFrame]: tuple containing the age as first component
            and a DataFrame containing centerline point coordinates and
            properties.

    """
    data: pd.DataFrame = pd.read_csv(filepath, sep=sep)

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

    Parameters:
    ----------
        filepath (str): path to the kml file
        keyword (str, optional): keyword to search for coordinate line.

            Defaults to "coordinates".

    Returns:
    --------
        pd.DataFrame: DataFrame containing centerline point coordinates.

    """
    coords_all = []
    with open(filepath, "r") as fin:
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


def dump_centerline_to_csv(
    filepath: str, centerline: Centerline, sep: str = ";"
) -> None:
    """Write a csv file containing centerline data.

    Parameters:
    ----------
        filepath (str): path to write the csv file
        centerline (Centerline): Centerline object to dump
        sep (str, optional): csv separator.

            Defaults to ";".

    """
    columns = centerline.cl_points[0].get_data().index.tolist() + [
        PropertyNames.AGE.value
    ]
    nrows = len(centerline.cl_points)
    data = pd.DataFrame(np.zeros((nrows, len(columns))), columns=columns)
    data[PropertyNames.AGE.value] = centerline.age
    for i, cl_pt in enumerate(centerline.cl_points):
        data.loc[i, cl_pt.get_data().index] = cl_pt.get_data()

    data.to_csv(filepath, sep=sep, index=False)


def load_centerline_collection_dataset_from_Flumy_csv(
    filepath: str, sep: str = ";"
) -> dict[int, pd.DataFrame]:
    """Load enterline collection dataset from a csv file generated by Flumy.

    Parameters:
    ----------
        filepath (str): path to write the csv file
        sep (str, optional): csv separator.

            Defaults to ";".

    Returns:
    ----------
        dict [int, pd.DataFrame]: dictionary where ages are keys and DataFrame
            with centerline point coordinates and properties are values.
    """
    data: pd.DataFrame = pd.read_csv(filepath, sep=sep)

    mess: str = (
        " property is missing. Try to use load_centerline_evolution_"
        + "from_multiple_xy_csv loader instead."
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

    assert data["Iteration"].unique().size > 1, (
        "Selected file contains a single centerline. Use load_centerline_"
        + "dataset_from_Flumy_csv instead."
    )

    if "Dist_previous" in data.columns:
        data.drop("Dist_previous", axis=1, inplace=True)

    map_dataset: dict[int, pd.DataFrame] = {}
    ages: npt.NDArray[np.int64] = data["Iteration"].unique()
    for age in ages.tolist():
        sub_data = data[data["Iteration"] == age].drop("Iteration", axis=1)
        map_dataset[age] = sub_data.reset_index(drop=True)

    return map_dataset


def load_centerline_evolution_from_single_xy_csv(
    filepath: str,
    x_prop: str = "X",
    y_prop: str = "Y",
    z_prop: str = "Z",
    age_prop: str = "Age",
    drop_columns: tuple[str, ...] = (),
    sep: str = ";",
) -> dict[int, pd.DataFrame]:
    """Load centerline data from multiple files.

    Parameters:
    ----------
        filepath (str): file path
        x_prop (str, optional): name of the column for x coordinate

            Defaults to "X".
        y_prop (str, optional): name of the column for y coordinate

            Defaults to "Y".
        z_prop (str, optional): name of the column for elevation

            Defaults to "Z".
        age_prop (str, optional): name of the column for centerline age

            Defaults to "Age".
        drop_columns (tuple[str,...], optional): list of the names of the
            columns to drop

            Defaults is empty.
        sep (str, optional): separator of the csv files

            Defaults to ";".

    Returns:
    ----------
        dict[int, pd.DataFrame]: dictionary where ages are keys and
            DataFrame with centerline point coordinates and properties are
            values.
    """
    data: pd.DataFrame = pd.read_csv(filepath, sep=sep)

    mess: str = " property is missing. Cannot load the data."
    assert x_prop in data.columns, f"{x_prop}" + mess
    assert y_prop in data.columns, f"{y_prop}" + mess
    assert age_prop in data.columns, f"{age_prop}" + mess

    for col in drop_columns:
        data.drop(columns=col, inplace=True)

    if z_prop in data.columns:
        data.rename(
            columns={
                z_prop: PropertyNames.ELEVATION.value,
            },
            inplace=True,
            copy=False,
        )
    else:
        data[PropertyNames.ELEVATION.value] = 0.0

    map_dataset: dict[int, pd.DataFrame] = {}
    ages: npt.NDArray[np.int64] = data[age_prop].unique()
    for age in ages.tolist():
        sub_data: pd.DataFrame = data[data[age_prop] == age].drop(
            age_prop, axis=1
        )
        sub_data.rename(
            columns={
                x_prop: PropertyNames.CARTESIAN_ABSCISSA.value,
                y_prop: PropertyNames.CARTESIAN_ORDINATE.value,
            },
            inplace=True,
            copy=False,
        )
        sub_data[PropertyNames.CURVILINEAR_ABSCISSA.value] = (
            cpf.compute_cuvilinear_abscissa(
                sub_data.loc[
                    :,
                    (
                        PropertyNames.CARTESIAN_ABSCISSA.value,
                        PropertyNames.CARTESIAN_ORDINATE.value,
                    ),
                ].to_numpy()  # type: ignore
            )
        )
        print(sub_data.columns)
        # sub_data.drop(columns=age_prop, inplace=True)
        map_dataset[age] = sub_data.reset_index(drop=True)
    return map_dataset


def load_centerline_evolution_from_multiple_xy_csv(
    map_file: dict[int, str],
    x_prop: str = "X",
    y_prop: str = "Y",
    z_prop: str = "Z",
    drop_columns: tuple[str, ...] = (),
    sep: str = ";",
) -> dict[int, pd.DataFrame]:
    """Load centerline data from multiple files.

    Parameters:
    ----------
        map_file (dict[int, str]): dictionnary of age and file name in the
            directory.
        x_prop (str, optional): name of the column for x coordinate

            Defaults to "X".
        y_prop (str, optional): name of the column for y coordinate

            Defaults to "Y".
        z_prop (str, optional): name of the column for elevation

            Defaults to "Z".
        drop_columns (tuple[str,...], optional): list of the names of the
            columns to drop

            Defaults is empty.
        sep (str, optional): separator of the csv files

            Defaults to ";".

    Returns:
    ----------
        dict[int, pd.DataFrame]: dictionary where ages are keys and
            DataFrame with centerline point coordinates and properties are
            values.
    """
    assert len(map_file) > 0, "The map of files is empty."

    map_dataset: dict[int, pd.DataFrame] = {}

    for key, filename in map_file.items():
        data: pd.DataFrame = load_centerline_dataset_from_csv(
            filename, x_prop, y_prop, z_prop, drop_columns, sep=sep
        )
        if data is not None:
            map_dataset[key] = data

    return map_dataset


def load_centerline_evolution_from_multiple_kml(
    directory: str, map_file: dict[int, str], keyword: str = "coordinates"
) -> dict[int, pd.DataFrame]:
    """Load centerline data from multiple files.

    Parameters:
    ----------
        directory (str): directory where the kml files are
        map_file (dict[int, str]): dictionnary of age and file name in the
            directory
        keyword (str, optional): keyword to search for coordinate line.

            Defaults to "coordinates".

    Returns:
    --------
        dict[int, pd.DataFrame]: dictionary where ages are keys and DataFrame
            with centerline point coordinates and properties are values.
    """
    assert len(map_file) > 0, "The map of files is empty."

    map_dataset = {}
    for key, filename in map_file.items():
        filepath: str = directory + filename
        data = load_centerline_dataset_from_kml(filepath, keyword)
        if data is not None:
            map_dataset[key] = data

    return map_dataset


@deprecated("Use load_centerline_dataset_from_csv instead.")
def create_dataset_from_xy(
    X: npt.NDArray[np.float64], Y: npt.NDArray[np.float64]
) -> pd.DataFrame:
    """Create a dataset from X and Y 1D arrays.

    Parameters:
    ----------
        X (npt.NDArray[np.float64]): X coordinates
        Y (npt.NDArray[np.float64]): Y coordinates

    Returns:
    --------
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
