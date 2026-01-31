# SPDX-FileCopyrightText: Copyright 2025 Martin Lemay <martin.lemay@mines-paris.org>
# SPDX-FileContributor: Martin Lemay

"""Centerline collection I/O.

This module contains helpers to load centerline-collection related datasets.
"""

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd  # type: ignore[import-untyped]

import pybend.algorithms.centerline_process_functions as cpf
from pybend.io.centerline_io import (
    load_centerline_dataset_from_csv,
    load_centerline_dataset_from_kml,
)
from pybend.io.common import CenterlineIOFormat, resolve_path
from pybend.model.enumerations import PropertyNames


def load_centerline_collection_from_a_file(
    filepath: str,
    kind: CenterlineIOFormat,
    **kwargs: Any,
) -> dict[int, pd.DataFrame]:
    """Load a centerline collection from a single file.

    Supported formats are: generic CSV and Flumy CSV.

    Args:
        filepath (str): Path to the file.
        kind (CenterlineIOFormat): File format.
        **kwargs (Any): Loader-specific options.

            For ``CenterlineIOFormat.CSV``:
                - ``x_prop`` (str): X column name (default ``"X"``)
                - ``y_prop`` (str): Y column name (default ``"Y"``)
                - ``z_prop`` (str): Z/elevation column name (default ``"Z"``)
                - ``age_prop`` (str): age column name (default ``"Age"``)
                - ``drop_columns`` (tuple[str, ...]): columns to drop (default
                    empty)
                - ``sep`` (str): separator (default ``";"``)

            For ``CenterlineIOFormat.FLUMY_CSV``:
                - ``sep`` (str): separator (default ``";"``)

    Returns:
        dict[int, pd.DataFrame]: dictionary where ages are keys and DataFrame
            with centerline point coordinates and properties are values.
    """
    path = resolve_path(
        base_dir=None, raw_url=filepath, ctx="loading centerline collection"
    )
    ext = path.suffix.lower().lstrip(".")
    match kind:
        case CenterlineIOFormat.CSV:
            assert ext == "csv", (
                "File extension does not match specified format."
            )
            x_prop = kwargs.get("x_prop", "X")
            y_prop = kwargs.get("y_prop", "Y")
            z_prop = kwargs.get("z_prop", "Z")
            age_prop = kwargs.get("age_prop", "Age")
            drop_columns = kwargs.get("drop_columns", ())
            sep = kwargs.get("sep", ";")
            dataset = load_centerline_evolution_from_single_xy_csv(
                filepath,
                x_prop=x_prop,
                y_prop=y_prop,
                z_prop=z_prop,
                age_prop=age_prop,
                drop_columns=drop_columns,
                sep=sep,
            )
        case CenterlineIOFormat.FLUMY_CSV:
            assert ext == "csv", (
                "File extension does not match specified format."
            )
            sep = kwargs.get("sep", ";")
            dataset = load_centerline_collection_dataset_from_Flumy_csv(
                filepath, sep=sep
            )
        case _:
            raise ValueError(f"File format {kind} not supported.")
    return dataset


def load_centerline_collection_from_multiple_files(
    map_file: dict[int, str],
    kind: CenterlineIOFormat,
    **kwargs: Any,
) -> dict[int, pd.DataFrame]:
    """Load a centerline collection from multiple files in a directory.

    Supported formats are: generic CSV and KML.

    Args:
        map_file (dict[int, str]): Mapping of ages to file paths.
        kind (CenterlineIOFormat): File format.
        **kwargs (Any): Loader-specific options.

            For ``CenterlineIOFormat.CSV``:
                - ``x_prop`` (str): X column name (default ``"X"``)
                - ``y_prop`` (str): Y column name (default ``"Y"``)
                - ``z_prop`` (str): Z/elevation column name (default ``"Z"``)
                - ``drop_columns`` (tuple[str, ...]): columns to drop (default
                    empty)
                - ``sep`` (str): separator (default ``";"``)

            For ``CenterlineIOFormat.KML``:
                - ``directory`` (str): base directory containing the KML files
                - ``keyword`` (str): keyword for coordinate line (default
                    ``"coordinates"``)

    Returns:
        dict[int, pd.DataFrame]: dictionary where ages are keys and DataFrame
            with centerline point coordinates and properties are values.
    """
    match kind:
        case CenterlineIOFormat.CSV:
            x_prop = kwargs.get("x_prop", "X")
            y_prop = kwargs.get("y_prop", "Y")
            z_prop = kwargs.get("z_prop", "Z")
            drop_columns = kwargs.get("drop_columns", ())
            sep = kwargs.get("sep", ";")
            dataset = load_centerline_evolution_from_multiple_xy_csv(
                map_file,
                x_prop=x_prop,
                y_prop=y_prop,
                z_prop=z_prop,
                drop_columns=drop_columns,
                sep=sep,
            )
        case CenterlineIOFormat.KML:
            directory: str = kwargs.get("directory", "")
            keyword = kwargs.get("keyword", "coordinates")
            dataset = load_centerline_evolution_from_multiple_kml(
                directory, map_file=map_file, keyword=keyword
            )
        case _:
            raise ValueError(f"File format {kind} not supported.")
    return dataset


def load_centerline_collection_dataset_from_Flumy_csv(
    filepath: str, sep: str = ";"
) -> dict[int, pd.DataFrame]:
    """Load enterline collection dataset from a csv file generated by Flumy.

    Args:
        filepath (str): path to write the csv file
        sep (str, optional): csv separator.

            Defaults to ";".

    Returns:
        dict [int, pd.DataFrame]: dictionary where ages are keys and DataFrame
            with centerline point coordinates and properties are values.
    """
    path = resolve_path(
        base_dir=None, raw_url=filepath, ctx="loading centerline collection"
    )
    data: pd.DataFrame = pd.read_csv(str(path), sep=sep)

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

    Args:
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
        dict[int, pd.DataFrame]: dictionary where ages are keys and
            DataFrame with centerline point coordinates and properties are
            values.
    """
    path = resolve_path(
        base_dir=None, raw_url=filepath, ctx="loading centerline collection"
    )
    data: pd.DataFrame = pd.read_csv(str(path), sep=sep)

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

    Args:
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

    Args:
        directory (str): directory where the kml files are
        map_file (dict[int, str]): dictionnary of age and file name in the
            directory
        keyword (str, optional): keyword to search for coordinate line.

            Defaults to "coordinates".

    Returns:
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
