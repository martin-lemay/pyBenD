# SPDX-FileCopyrightText: Copyright 2025 Martin Lemay <martin.lemay@mines-paris.org>
# SPDX-FileContributor: Martin Lemay

"""Centerline I/O.

This module contains helpers to load and dump centerline-related datasets.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd  # type: ignore[import-untyped]
from typing_extensions import deprecated

import pybend.algorithms.centerline_process_functions as cpf
from pybend.io.common import CenterlineIOFormat, read_csv, resolve_path
from pybend.model.Centerline import Centerline
from pybend.model.enumerations import PropertyNames
from pybend.utils.logging import logger


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
                - ``keyword`` (str): tag name of coordinate elements
                  (default ``"coordinates"``)
                - ``to_metric`` (bool): convert lon/lat degrees to local
                  metric coordinates (default ``False``)

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
            to_metric = kwargs.get("to_metric", False)
            dataset = load_centerline_dataset_from_kml(
                filepath, keyword=keyword, to_metric=to_metric
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
    dataset: pd.DataFrame = read_csv(path, sep=sep)

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
    data: pd.DataFrame = read_csv(path, sep=sep)

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


#: Mean Earth radius (m) used for the equirectangular projection of KML
#: coordinates.
EARTH_RADIUS: float = 6371008.8


def _parse_kml_coordinates(text: str) -> list[list[float]]:
    """Parse the text of a KML ``coordinates`` element.

    The text is a whitespace-separated list of ``lon,lat[,alt]`` tuples.

    Args:
        text (str): content of the ``coordinates`` element.

    Returns:
        list[list[float]]: list of parsed tuples.

    Raises:
        ValueError: If a tuple cannot be parsed.
    """
    coords: list[list[float]] = []
    for token in text.split():
        values = [val for val in token.split(",") if len(val) > 0]
        if len(values) < 2 or len(values) > 3:
            raise ValueError(f"Invalid KML coordinate tuple: '{token}'.")
        coords.append([float(val) for val in values])
    return coords


def _lonlat_to_local_metric(
    lon: npt.NDArray[np.float64], lat: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Project lon/lat (degrees) to local metric coordinates.

    Uses an equirectangular projection centred on the first point:
    ``X = R * (lon - lon0) * cos(lat0)`` and ``Y = R * (lat - lat0)``,
    with angles in radians and ``R`` the mean Earth radius.

    Args:
        lon (npt.NDArray[np.float64]): longitudes in degrees.
        lat (npt.NDArray[np.float64]): latitudes in degrees.

    Returns:
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: X and Y
        coordinates in meters relative to the first point.
    """
    lon_rad = np.radians(lon)
    lat_rad = np.radians(lat)
    x = EARTH_RADIUS * (lon_rad - lon_rad[0]) * np.cos(lat_rad[0])
    y = EARTH_RADIUS * (lat_rad - lat_rad[0])
    return x, y


def load_centerline_dataset_from_kml(
    filepath: str, keyword: str = "coordinates", to_metric: bool = False
) -> pd.DataFrame:
    """Load a dataset from a kml file containing centerline point coordinates.

    The file is parsed as XML. Every element whose tag (without namespace)
    is *keyword* is read, in document order, and its text is split into
    ``lon,lat[,alt]`` tuples.

    KML coordinates are geographic coordinates in degrees. If *to_metric*
    is True, they are converted to local metric coordinates (meters) with
    an equirectangular projection centred on the first point (mean Earth
    radius R = 6371008.8 m), which is accurate for centerlines spanning a
    few tens of kilometers. Otherwise, coordinates are kept in degrees and
    a warning is logged since all distances computed afterwards will then
    be expressed in degrees.

    Args:
        filepath (str): path to the kml file
        keyword (str, optional): tag name of the coordinate elements.
            Defaults to "coordinates".
        to_metric (bool, optional): if True, convert lon/lat to local
            metric X/Y coordinates.
            Defaults to False.

    Returns:
        pd.DataFrame: DataFrame containing centerline point coordinates.

    Raises:
        ValueError: If no coordinates are found or if coordinate tuples
            are malformed or have inconsistent dimensions.
    """
    path = resolve_path(
        base_dir=None, raw_url=filepath, ctx="loading centerline"
    )
    root = ET.parse(str(path)).getroot()
    coords_all: list[list[float]] = []
    for elt in root.iter():
        # remove namespace if any: "{namespace}tag" -> "tag" (comments and
        # processing instructions have non-string tags and never match)
        tag = str(elt.tag).rsplit("}", 1)[-1]
        if tag != keyword or elt.text is None:
            continue
        coords_all += _parse_kml_coordinates(elt.text)

    nb_pts = len(coords_all)
    if nb_pts == 0:
        raise ValueError("Point coordinates were not found.")
    nb_col = len(coords_all[0])
    if any(len(coords) != nb_col for coords in coords_all):
        raise ValueError("KML coordinate tuples have inconsistent dimensions.")

    columns = (
        PropertyNames.CARTESIAN_ABSCISSA.value,
        PropertyNames.CARTESIAN_ORDINATE.value,
        PropertyNames.ELEVATION.value,
    )
    data = np.array(coords_all, dtype=np.float64)
    if to_metric:
        data[:, 0], data[:, 1] = _lonlat_to_local_metric(
            data[:, 0], data[:, 1]
        )
    else:
        logger.warning(
            "KML coordinates are loaded in degrees (longitude, latitude): "
            "distances will be expressed in degrees. Use to_metric=True to "
            "convert them to local metric coordinates."
        )

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
    p = Path(filepath).expanduser()
    if p.exists() and p.is_dir():
        raise IsADirectoryError(
            f"dump_centerline_to_csv() expected a file path, "
            f"got directory: {p}"
        )

    # Resolve to an absolute path without requiring the file to pre-exist.
    # (resolve_path() is intended for inputs and enforces existence.)
    out_file = p if p.is_absolute() else (Path.cwd() / p)
    out_file = out_file.resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    columns = centerline.cl_points[0].get_data().index.tolist() + [
        PropertyNames.AGE.value
    ]
    nrows = len(centerline.cl_points)
    data = pd.DataFrame(np.zeros((nrows, len(columns))), columns=columns)
    data[PropertyNames.AGE.value] = centerline.age
    for i, cl_pt in enumerate(centerline.cl_points):
        data.loc[i, cl_pt.get_data().index] = cl_pt.get_data()

    data.to_csv(str(out_file), sep=sep, index=False)
