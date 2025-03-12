# SPDX-FileCopyrightText: Copyright 2025 Martin Lemay <martin.lemay@mines-paris.org>
# SPDX-FileContributor: Martin Lemay
# ruff: noqa: E402 # disable Module level import not at top of file

import functools
from multiprocessing import Pool
from typing import Optional, Self, cast

import numpy as np
import numpy.typing as npt
import pandas as pd  # type: ignore[import-untyped]
from scipy.ndimage import uniform_filter  # type: ignore[import-untyped]
from scipy.signal import (  # type: ignore[import-untyped]
    find_peaks,
    savgol_filter,
)
from shapely.geometry import LineString, Polygon  # type: ignore

import pybend.algorithms.centerline_process_function as cpf
from pybend.model.Bend import Bend
from pybend.model.ClPoint import ClPoint
from pybend.model.enumerations import (
    BendSide,
    FilterName,
    PropertyNames,
)
from pybend.utils.globalParameters import get_nb_procs
from pybend.utils.logging import logger

__doc__ = r"""
Let's consider a channel centerline discretized into successive channel points.
This module defines the Centerline object that stores all channel points.
Centerline object also provides multiple methods to compute centerline
attributes like inflection point locations, meander bends, bend apex,
middle and barycenter point locations.


             x                         x                         x
          .     .                    .    .                   .    .
        .         .                .        .               .        .
       .     b     .              .     b    .             .     b    .
      o      m      o      m     o      m     o     m     o      m     o
                     .     b    .              .    b    .
                      .        .                .       .
                        .    .                    .   .
                           x                        x

*Sinuous channel centerline discretized into a series avec channel points (.)
where bends are defined by upstream and downstream inflection points (o). Bends
contains distinctive points including the apex (x), the middle (m) and the
barucenter (b).*


To use it:

.. code-block:: python

    centerline :Centerline = Centerline()

"""


class Centerline:
    def __init__(
        self: Self,
        age: int,
        dataset: pd.DataFrame,
        spacing: float,
        smooth_distance: float,
        use_fix_nb_points: bool = False,
        curvature_filtering_window: int = 5,
        sinuo_thres: float = 1.05,
        n: float = 2,
        compute_curvature: bool = True,
        interpol_props: bool = True,
        find_bends: bool = True,
    ) -> None:
        """Store channel centerline as a collection of ClPoint.

        Parameters:
        ----------
            age (int): Age of the centerline.
            dataset (pd.DataFrame): DataFrame that contains channel point coordinates
                and properties. Points must be ordered according to flow direction.
            spacing (float): Target distance (m) between channel points after resampling.
                If spacing equals 0, centreline is not resampled
            smooth_distance (float): Smoothing distance (m) for Savitsky-Golay filter
                applied on channel path.
            use_fix_nb_points (bool, optional): If True, the resampled centerline
                will contains exactly spacing points, otherwise, spacing is the targeted
                distance between 2 consecutive points.

                Defaults to False.
            curvature_filtering_window (int, optional): Number of points used for filtering curvature.

                Defaults to 5.
            sinuo_thres (float, optional): Sinuosity threshold used to discriminate valid bends.

                Defaults to 1.05.
            n (float): exponent value for bend apex detection

                Defaults to 2.
            compute_curvature (bool, optional): If True, recompute and filter
                curvature along channel points.

                Defaults to True.
            interpol_props (bool, optional): If True, interpolate channel point
                properties along channel points.

                Defaults to True.
            find_bends (bool, optional): If True, automatically compute curvature
                and interpolate properties and detect meander bends along channel centerline.

                Defaults to True.
        """
        #: age of the enterline
        self.age: int = age
        #: list of Channel points
        self.cl_points: list[ClPoint] = []
        #: list od bends
        self.bends: list[Bend] = []
        self.bends_filtered: list[Bend] = []

        #: indexes of each channel point in the previous centerline
        self.index_cl_pts_prev_centerline: list[int] = []
        #: indexes of each channel point in the next centerline
        self.index_cl_pts_next_centerline: list[list[int]] = []

        # if find bends, automatically compute curvature and interpolate properties
        self._init_centerline(
            dataset,
            age,
            spacing,
            smooth_distance,
            curvature_filtering_window,
            use_fix_nb_points,
            compute_curvature | find_bends,
            interpol_props | find_bends,
        )

        if find_bends and self.find_bends(sinuo_thres, n):
            logger.info("Bends found")

    def _init_centerline(
        self: Self,
        dataset: pd.DataFrame,
        age: int,
        spacing: float,
        smooth_distance: float,
        curvature_filtering_window: int,
        use_fix_nb_points: bool,
        compute_curvature: bool,
        interpol_props: bool,
    ) -> None:
        """Initialize Centerline object.

        Centerline intialization includes:
            - resampling the centerline with a parametric spline function
            - if spacing > 0, smoothing centerline path
            - if interpol_props is True, interpolating centerline properties to
                new points
            - if compute_curvature is True, computing and filtering curvatures
            - fill the list of channel point self.cl_points

        Parameters:
        ----------
            dataset (DataFrame): DataFrame that contains channel point coordinates
                and properties. Points must be ordered according to flow direction.
            age (int): Age of the centerline..
            spacing (float): Target distance (m) between channel points after
                resampling. If spacing equals 0, centreline is not resampled.
            smooth_distance (float): Smoothing distance (m) for Savitsky-Golay
                filter applied on channel path.
            curvature_filtering_window (int): Number of points used for filtering
                curvature.
            use_fix_nb_points (bool): If True, the resampled centerline
                will contains exactly spacing points, otherwise, spacing is the
                targeted distance between 2 consecutive points.
            compute_curvature (bool): If True, compute and smooth
                curvatures.
            interpol_props (bool): if True, interpolate centerline
                properties.

        """
        logger.info(f"Initialize Centerline object {age}")
        logger.info(f"Resample points of centerline {age}")

        cart_abscissa_prop_name: str = PropertyNames.CARTESIAN_ABSCISSA.value
        cart_ordinate_prop_name: str = PropertyNames.CARTESIAN_ORDINATE.value
        # 1. resample the centerline with a parametric spline function
        ls = cpf.compute_cuvilinear_abscissa(
            dataset.loc[
                :, (cart_abscissa_prop_name, cart_ordinate_prop_name)
            ].to_numpy()  # type: ignore
        )
        nb_pts: int = 0  # no resampling if spacing if <= 0
        if spacing > 0:
            if use_fix_nb_points:
                nb_pts = int(spacing)
            else:
                nb_pts = int(ls[-1] / spacing + 1)
        new_points = cpf.resample_path(
            dataset[cart_abscissa_prop_name].to_numpy(),
            dataset[cart_ordinate_prop_name].to_numpy(),
            nb_pts,
        )

        # add normal vector columns
        columns = dataset.columns.tolist() + [
            PropertyNames.NORMAL_X.value,
            PropertyNames.NORMAL_Y.value,
        ]
        dataset_new = pd.DataFrame(
            np.zeros((len(new_points[0]), len(columns))), columns=columns
        )

        # 2. smooth centerline path if spacing > 0
        if spacing:
            logger.info(f"Smooth centerline {age} path")
            window = int(float(smooth_distance / spacing))  # number of points

            if window % 2 == 0:
                window += 1  # to be odd
            if window < 5:
                window = 5
            dataset_new[cart_abscissa_prop_name] = savgol_filter(
                new_points[0], window, polyorder=3, mode="nearest"
            )
            dataset_new[cart_ordinate_prop_name] = savgol_filter(
                new_points[1], window, polyorder=3, mode="nearest"
            )
        else:
            dataset_new[cart_abscissa_prop_name] = dataset[cart_abscissa_prop_name]
            dataset_new[cart_ordinate_prop_name] = dataset[cart_ordinate_prop_name]

        # compute curvilinear abscissa and normals
        self._compute_curvilinear_abscissa(dataset_new)
        self._compute_normal_to_points(dataset_new)

        # 2 bis interpolate centerline properties to new points
        # find the 2 closest points in the old centerline, their distances, and interpolate
        if interpol_props:
            logger.info(f"Interpolate properties to new points (centerline {age})")
            self._interpolate_properties(dataset_new, dataset)
        else:
            logger.warning("Some channel points do not have property values")

        # 3. compute and filtered curvatures
        if compute_curvature:
            logger.info(f"Compute channel point curvature (centerline {age})")
            self._compute_curvature(dataset_new)
            self._compute_filtered_curvature(dataset_new, curvature_filtering_window)
        else:
            logger.warning(
                "Some channel points may have no curvature defined. "
                + "Channel bends cannot be detected without curvature calculation."
            )

        # Create Centerline object as a collection of cl_Points
        for i, row in dataset_new.iterrows():
            ide: str = f"{self.age}-{i}"
            self.cl_points += [ClPoint(ide, age, row)]

        logger.info(f"Centerline object {self.age} initialized")

    def get_nb_points(self: Self) -> int:
        """Get the number of channel points.

        Returns:
        ----------
            int: number of channel points
        """
        return len(self.cl_points)

    def get_nb_bends(self: Self) -> int:
        """Get the number of bends along the centerline.

        Returns:
        ----------
            int: Number of bends

        """
        return len(self.bends)

    def get_nb_valid_bends(self: Self) -> int:
        """Get the number of valid bends along the centerline.

        Returns:
        ----------
            int: Number of valid bends

        """
        return len(self.get_valid_bend_indexes())

    def get_valid_bend_indexes(self: Self) -> list[int]:
        """Get the list of valid bend indexes along the centerline.

        Returns:
        ----------
            list[int]: List of valid bends index

        """
        return [bend.id for bend in self.bends if bend.isvalid]

    def get_property_list(self: Self) -> tuple[str]:
        """Get the list of property name stored on channel points.

        Returns:
        --------
            tuple[str]: tuple of property names.
        """
        return tuple(self.cl_points[0].get_data().index)

    def get_property(self: Self, prop_name: str) -> npt.NDArray[np.float64]:
        """Get the property of channel points along the centerline.

        Parameters:
        ----------
            prop_name (str): Property name

        Returns:
        ----------
            NDArray[float]: Array with property values

        """
        try:
            data: npt.NDArray[np.float64] = np.full(self.get_nb_points(), np.nan)
            for i, cl_pt in enumerate(self.cl_points):
                data[i] = cl_pt.get_property(prop_name)
            return data
        except Exception as err:
            logger.error(str(err))
            return np.full(self.get_nb_points(), np.nan)

    def get_bend_property(
        self: Self, bend_id: int, prop_name: str
    ) -> npt.NDArray[np.float64]:
        """Get the property of channel points along the bend with id bend_id.

        Parameters:
        ----------
            bend_id (int): Bend id.
            prop_name ( str): Property name.

        Returns:
        ----------
            NDArray[float]: Array with property values

        """
        try:
            bend: Bend = self.bends[bend_id]
            data: npt.NDArray[np.float64] = np.full(bend.get_nb_points(), np.nan)
            for i, cl_pt in enumerate(
                self.cl_points[bend.index_inflex_up : bend.index_inflex_down + 1]
            ):
                data[i] = cl_pt.get_property(prop_name)
            return data
        except Exception as err:
            logger.error(str(err))
            return np.full(bend.get_nb_points(), np.nan)

    def get_all_curvature_filtered(self: Self) -> npt.NDArray[np.float64]:
        """Get the array of filtered curvature property along the centerline.

        Returns:
        ----------
            NDArray[float]: Array with smoothed curvature values.

        """
        return self.get_property(PropertyNames.CURVATURE_FILTERED.value)

    def get_bend_curvature_filtered(
        self: Self, bend_id: int
    ) -> npt.NDArray[np.float64]:
        """Get the array of filtered curvature property along the bend if id bend_id.

        Returns:
        ----------
            NDArray[float]: Array with smoothed curvature values.

        """
        return self.get_bend_property(bend_id, PropertyNames.CURVATURE_FILTERED.value)

    def get_all_curvature(self: Self) -> npt.NDArray[np.float64]:
        """Get the array of curvature property along the centerline.

        Returns:
        ----------
            NDArray[float]: Array with curvature values.

        """
        return self.get_property(PropertyNames.CURVATURE.value)

    def get_bend_curvature(self: Self, bend_id: int) -> npt.NDArray[np.float64]:
        """Get the array of curvature property along the bend if id bend_id.

        Returns:
        ----------
            NDArray[float]: Array with curvature values.

        """
        return self.get_bend_property(bend_id, PropertyNames.CURVATURE.value)

    def set_property_point(
        self: Self, cl_pt_index: int, property_name: str, value: float
    ) -> None:
        """Set the property value of input point at index cl_pt_index.

        Parameters:
        ----------
            cl_pt_index (int): Channel point index in self.cl_points.
            property_name (str): Property name.
            value (float): Value of the property at the given channel point.

        """
        self.cl_points[cl_pt_index].set_property(property_name, value)

    def set_property_all_points(
        self: Self, property_name: str, values: npt.NDArray[np.float64]
    ) -> None:
        """Set the property values of all channel points.

        Parameters:
        ----------
            property_name (str): Property name.
            value (NDArray[float]): Array of value of the property.

        """
        if values.size > self.get_nb_points():
            logger.error("The number of values is greater than the number of points.")
            return
        for cl_pt_index, value in enumerate(values):
            self.set_property_point(cl_pt_index, property_name, value)

    def _compute_curvilinear_abscissa(self: Self, dataset: pd.DataFrame) -> None:
        """Compute curvilinear abscissa of channel points along the centerline.

        Parameters:
        ----------
            dataset (DataFrame): DataFrame that contains channel point data.
                Input DataFrame is updated with the computed property.

        """
        dataset[
            PropertyNames.CURVILINEAR_ABSCISSA.value
        ] = cpf.compute_cuvilinear_abscissa(
            dataset.loc[:, (PropertyNames.CARTESIAN_ABSCISSA.value, PropertyNames.CARTESIAN_ORDINATE.value)].to_numpy()  # type: ignore
        )

    def _compute_normal_to_points(self: Self, dataset: pd.DataFrame) -> None:
        """Compute the normal to channel points along the centerline.

        Parameters:
        ----------
            dataset (DataFrame): DataFrame that contains channel point data.
                Input DataFrame is updated with the computed property.

        """
        normal = np.array([0.0, 0.0])
        cart_abscissa_prop_name: str = PropertyNames.CARTESIAN_ABSCISSA.value
        cart_ordinate_prop_name: str = PropertyNames.CARTESIAN_ORDINATE.value
        for i, row in dataset.iterrows():
            i = cast(int, i)
            if i == 0:
                pt_prev = np.array(
                    [row[cart_abscissa_prop_name], row[cart_ordinate_prop_name]]
                )
                pt_next = np.array(
                    [
                        dataset[cart_abscissa_prop_name][i + 1],
                        dataset[cart_ordinate_prop_name][i + 1],
                    ]
                )
            elif i == dataset.shape[0] - 1:
                pt_prev = np.array(
                    [
                        dataset[cart_abscissa_prop_name][i - 1],
                        dataset[cart_ordinate_prop_name][i - 1],
                    ]
                )
                pt_next = np.array(
                    [row[cart_abscissa_prop_name], row[cart_ordinate_prop_name]]
                )
            else:
                pt_prev = np.array(
                    [
                        dataset[cart_abscissa_prop_name][i - 1],
                        dataset[cart_ordinate_prop_name][i - 1],
                    ]
                )
                pt_next = np.array(
                    [
                        dataset[cart_abscissa_prop_name][i + 1],
                        dataset[cart_ordinate_prop_name][i + 1],
                    ]
                )

            normal = cpf.normal(pt_next - pt_prev)
            dataset.loc[i, PropertyNames.NORMAL_X.value] = normal[0]
            dataset.loc[i, PropertyNames.NORMAL_Y.value] = normal[1]

    def _compute_curvature(self: Self, dataset: pd.DataFrame) -> None:
        """Compute the curvature of channel points along the centerline.

        Parameters:
        ----------
            dataset (DataFrame): DataFrame that contains channel point data.
                Input DataFrame is updated with the computed property.

        """
        if PropertyNames.CURVATURE.value not in dataset:
            dataset[PropertyNames.CURVATURE.value] = 0.0

        nb_procs: int = get_nb_procs()

        if nb_procs == 1:
            self._compute_curvature_monoproc(dataset)
        else:
            self._compute_curvature_multiproc(dataset, nb_procs)

    def _compute_filtered_curvature(
        self: Self,
        dataset: pd.DataFrame,
        window: int,
        method: FilterName = FilterName.SAVITSKY,
    ) -> None:
        """Compute the filtered curvature of channel points along the centerline.

        Input DataFrame is updated with the computed property.

        Parameters:
        ----------
            dataset (DataFrame): DataFrame that contains channel point data.
            window (int): Number of points used for filtering curvature.
            method (FilterName): Filter to use, either
                FilterName.UNIFORM or Smooth_filter.SAVITSKY

                Defaults to FilterName.SAVITSKY.
        """
        # apply uniform filter to the curvature to smooth local variations
        match method:
            case FilterName.UNIFORM:
                dataset[PropertyNames.CURVATURE_FILTERED.value] = uniform_filter(
                    dataset[PropertyNames.CURVATURE.value], size=window, mode="nearest"
                )
            # filtered curvature using the Savitzky-Golay filter
            case FilterName.SAVITSKY:
                # window must be odd
                if window % 2 == 0:
                    window += 1  # to be odd
                if window <= 3:
                    logger.warning("Curvature smoothing window cannot be lower than 5.")
                    window = 5
                dataset[PropertyNames.CURVATURE_FILTERED.value] = savgol_filter(
                    dataset[PropertyNames.CURVATURE.value], window, polyorder=2
                )
            case _:
                filternames = " or ".join(list(FilterName))  # type: ignore[unreachable]
                raise TypeError(
                    f"Filter is not managed. Filters are either: {filternames}"
                )

    def _compute_curvature_monoproc(self: Self, dataset: pd.DataFrame) -> None:
        """Compute the curvature of channel points using monoprocessing.

        Parameters:
        ----------
            dataset (DataFrame): DataFrame that contains channel point data.
                Input DataFrame is updated with the computed property.

        """
        cart_abscissa_prop_name: str = PropertyNames.CARTESIAN_ABSCISSA.value
        cart_ordinate_prop_name: str = PropertyNames.CARTESIAN_ORDINATE.value
        for i, row in dataset.iterrows():
            i = cast(int, i)
            if (i > 0) and (i < len(dataset[cart_abscissa_prop_name]) - 1):
                pt1: npt.NDArray[np.float64] = np.array(
                    (
                        dataset[cart_abscissa_prop_name][i - 1],
                        dataset[cart_ordinate_prop_name][i - 1],
                    )
                )
                pt2: npt.NDArray[np.float64] = np.array(
                    (row[cart_abscissa_prop_name], row[cart_ordinate_prop_name])
                )
                pt3: npt.NDArray[np.float64] = np.array(
                    (
                        dataset[cart_abscissa_prop_name][i + 1],
                        dataset[cart_ordinate_prop_name][i + 1],
                    )
                )
                curvature: float = cpf.compute_curvature_at_point(pt1, pt2, pt3)
                dataset.loc[i, PropertyNames.CURVATURE.value] = curvature

    def _compute_curvature_multiproc(
        self: Self, dataset: pd.DataFrame, nb_procs: int
    ) -> None:
        """Compute the curvature of channel points using multirocessing.

        Parameters:
        ----------
            dataset (DataFrame): DataFrame that contains channel point data.
                Input DataFrame is updated with the computed property.

        """
        cart_abscissa_prop_name: str = PropertyNames.CARTESIAN_ABSCISSA.value
        cart_ordinate_prop_name: str = PropertyNames.CARTESIAN_ORDINATE.value
        with Pool(processes=nb_procs) as pool:
            inputs = [
                (
                    np.array(
                        (
                            dataset[cart_abscissa_prop_name][i - 1],
                            dataset[cart_ordinate_prop_name][i - 1],
                        )
                    ),  # type: ignore
                    np.array((row[cart_abscissa_prop_name], row[cart_ordinate_prop_name])),  # type: ignore
                    np.array(
                        (
                            dataset[cart_abscissa_prop_name][i + 1],
                            dataset[cart_ordinate_prop_name][i + 1],
                        )
                    ),
                )  # type: ignore
                for i, row in dataset.iterrows()
                if ((i > 0) and (i < dataset.shape[0] - 1))
            ]  # type: ignore
            outputs = pool.starmap(cpf.compute_curvature_at_point, inputs)  # type: ignore

            for i, curv in enumerate(outputs):
                dataset.loc[i + 1, PropertyNames.CURVATURE.value] = curv  # type: ignore

    def get_bend_index_from_cl_pt_index(self: Self, cl_pt_index: int) -> Optional[int]:
        """Get the index of the bend that contains the channel point at index cl_pt_index.

        Parameters:
        ----------
            cl_pt_index (Optional[int]): Index of the channel point in the list
                    self.cl_points.

        Returns:
        ----------
            int: Index of the bend in the list self.bends, or np.nan if no
                bend was found.

        """
        if cl_pt_index > self.get_nb_points():
            return None
        for i, bend in enumerate(self.bends):
            if (cl_pt_index >= bend.index_inflex_up) & (
                cl_pt_index <= bend.index_inflex_down
            ):
                return i
        return None

    def _interpolate_properties(
        self: Self, dataset_new: pd.DataFrame, dataset: pd.DataFrame
    ) -> bool:
        """Interpolate properties from dataset to dataset_new.

        Parameters:
        ----------
            dataset_new (DataFrame): DataFrame with resampled centerline data
                where to interpolate properties.
            dataset (DataFrame): Original DataFrame from which come properties.

        Returns:
        ----------
            bool: True if interpolation successfully ended.
                Input DataFrame dataset_new is updated with the computed
                properties.

        """
        nb_procs: int = get_nb_procs()
        # 1. find for each point of dataset_new the 2 closests points in dataset1
        try:
            x_prop = PropertyNames.CARTESIAN_ABSCISSA.value
            y_prop = PropertyNames.CARTESIAN_ORDINATE.value
            if nb_procs == 1:
                result = cpf.find_2_closest_points_mono_proc(
                    dataset_new, dataset, x_prop, y_prop
                )
            else:
                result = cpf.find_2_closest_points_multi_proc(
                    dataset_new, dataset, x_prop, y_prop, nb_procs
                )
        except Exception as err:
            logger.error(
                "Error in find_2_closest_points_monoproc function during "
                + "property interpolation calculation: "
            )
            logger.error(err)

            return False

        if result.shape[0] != dataset_new.shape[0]:
            logger.error(
                "Error in find_2_closest_points result during "
                + "property interpolation calculation"
            )
            return False

        # 2. interpolate the properties - compute them into the new point
        # exluded properties - not interpolated
        props_excluded = (
            PropertyNames.CURVILINEAR_ABSCISSA.value,
            PropertyNames.CARTESIAN_ABSCISSA.value,
            PropertyNames.CARTESIAN_ORDINATE.value,
        )
        for i, row in result.iterrows():
            i = cast(int, i)
            self._compute_property_at_point(
                dataset, props_excluded, dataset_new, i, row
            )

        return True

    def _compute_property_at_point(
        self: Self,
        dataset: pd.DataFrame,
        props_excluded: tuple[str, ...],
        dataset_new: pd.DataFrame,
        i: int,
        row: pd.Series,
    ) -> None:
        """Compute interpolated properties at index i in dataset_new from dataset.

        Paramers:
        ---------
            dataset (DataFrame): Original DataFrame from which properties come
                from.

                Input DataFrame dataset_new is updated with the computed properties.
            props_excluded (list[str]): List of properties excluded from the
                interpolation.
            dataset_new (DataFrame): DataFrame to which properties are computed.
            i (int): index in dataset_new.
            row (Series): Series that contains information on original points
                for interpolation.

        """
        j1 = row["index1"]
        j2 = row["index2"]
        d1 = row["d1"]
        d2 = row["d2"]
        props = dataset.columns
        denom = d1 + d2
        if j1 == j2:
            d1 = 0.5
            d2 = 0.5
            denom = 1.0

        for prop in props:
            if prop in props_excluded:
                continue
            if (j1 < dataset.shape[0]) and (j2 < dataset.shape[0]):
                dataset_new.loc[i, prop] = (
                    d1 * dataset[prop][j1] + d2 * dataset[prop][j2]
                ) / denom
            else:
                dataset_new.loc[i, prop] = dataset[prop][j1]

    def find_inflection_points(self: Self) -> npt.NDArray[np.int64]:
        """Find inflection points along the centerline.

        Inflection points are determine such as the smoothed curvature change
        of sign next to it.

        Returns:
        ----------
            NDArray[np.int64]: List of inflection point indexes.

        """
        # find inflection points based on filtered curvatures
        curvature: npt.NDArray[np.float64] = self.get_all_curvature_filtered()
        inflex_pts: npt.NDArray[np.int64] = cpf.find_inflection_points(curvature, 2)

        # add first and last indexes
        if 0 not in inflex_pts:
            inflex_pts = np.append([0], inflex_pts)
        if self.get_nb_points() - 1 not in inflex_pts:
            inflex_pts = np.append(inflex_pts, [self.get_nb_points() - 1])
        return inflex_pts

    def find_bends(self: Self, sinuo_thres: float, n: float) -> bool:
        """Find bends along the centerline.

        Bends are defined as the points between two consecutive inflection points.

        Parameters:
        ----------
            sinuo_thres (float): Sinuosity threshold used to discriminate valid
                bends.
            n (float): exponent value

        Returns:
        ----------
            bool: True if calculation successfully ended.

        """
        logger.info("Find bends")
        assert (
            PropertyNames.CURVATURE_FILTERED.value in self.cl_points[0].get_data().index
        ), (
            "Smoothed curvature is not defined. Bends cannot be computed. "
            + "Set compute_curvature option to True when importing centerline."
        )
        try:
            inflex_pts_index: npt.NDArray[np.int64] = self.find_inflection_points()

            if get_nb_procs() == 1:
                self._create_bends_monoproc(inflex_pts_index, sinuo_thres, n)
            else:
                self._create_bends_multiproc(inflex_pts_index, sinuo_thres, n)

        except Exception as err:
            logger.error("Bends were not detected due to:")
            logger.error(err)

            return False
        return True

    def _create_bend(
        self: Self, bend_id: int, inflex_index_up: int, inflex_index_down: int
    ) -> Bend:
        """Create a Bend object.

        Parameters:
        -----------
            bend_id (int): Id of the bend.
            inflex_index_up (int): Index of upstream inflection point.
            inflex_index_down (int): Index of downstream inflection point.

        Returns:
        ----------
            Bend: Created Bend object.

        """
        return Bend(bend_id, inflex_index_up, inflex_index_down, self.age)

    def _create_bends_monoproc(
        self: Self,
        inflex_pts_index: npt.NDArray[np.int64],
        sinuo_thres: float,
        n: float,
    ) -> None:
        """Create all Bend objects and add them in self.bends using monoprocessing.

        Parameters:
        ----------
            inflex_pts_index (npt.NDArray[np.int64]): List of inflection point
                indexes.
            sinuo_thres (float): Sinuosity threshold used to discriminate valid
                bends.
            n (float): exponent value

        """
        for bend_index, (inflex_index_up, inflex_index_down) in enumerate(
            zip(inflex_pts_index[:-1], inflex_pts_index[1:], strict=False)
        ):
            bend: Bend = self._create_bend(
                bend_index, inflex_index_up, inflex_index_down
            )
            self.bends += [bend]
            side, isvalid, index_apex, pt_middle = self._compute_bend_properties(
                sinuo_thres, n, bend_index
            )
            self._update_bend(bend.id, side, isvalid, index_apex, pt_middle)

    def _create_bends_multiproc(
        self: Self,
        inflex_pts_index: npt.NDArray[np.int64],
        sinuo_thres: float,
        n: float,
    ) -> None:
        """Create all Bend objects add them in self.bends using multiprocessing.

        Parameters:
        ----------
            inflex_pts_index (npt.NDArray[np.int64]): List of inflection point
                indexes.
            sinuo_thres (float): Sinuosity threshold used to discriminate valid
                bends.
            n (float): exponent value

        """
        nb_procs: int = get_nb_procs()
        with Pool(processes=nb_procs) as pool:
            inputs = [
                (bend_index, inflex_index_up, inflex_index_down)
                for bend_index, (
                    inflex_index_up,
                    inflex_index_down,
                ) in enumerate(
                    zip(
                        inflex_pts_index[:-1],
                        inflex_pts_index[1:],
                        strict=False,
                    )
                )
            ]  # type: ignore
            self.bends = pool.starmap(self._create_bend, inputs)  # type: ignore

        assert len(self.bends) > 0, "No bends were found."

        # sort list according to bend index
        self.bends.sort(key=lambda bend: bend.id)

        # compute bend properties
        with Pool(processes=nb_procs) as pool:
            partial_compute_bend_properties = functools.partial(
                self._compute_bend_properties, sinuo_thres, n
            )
            outputs = pool.map(
                partial_compute_bend_properties,  # type: ignore
                [bend.id for bend in self.bends],
            )

        # update bends
        for bend_index, (side, isvalid, index_apex, pt_middle) in enumerate(outputs):
            self._update_bend(bend_index, side, isvalid, index_apex, pt_middle)

    def _update_bend(
        self: Self,
        bend_index: int,
        side: Optional[BendSide] = None,
        valid: Optional[bool] = None,
        index_apex: Optional[int] = None,
        pt_middle: Optional[npt.NDArray[np.float64]] = None,
        pt_centroid: Optional[npt.NDArray[np.float64]] = None,
    ) -> None:
        """Update bend properties (side, validity, apex and middle points).

        Parameters:
        ----------
            bend_index (int): Bend index.
            side (BendSide, optional): bend side

                Defaults to None.
            valid (bool, optional): bend validity

                Defaults to None.
            index_apex (int, optional): bend apex index

                Defaults to None.
            pt_middle (npt.NDArray[np.float64], optional): bend middle point

                Defaults to None.
            pt_centroid (npt.NDArray[np.float64], optional): bend centroid point

                Defaults to None.

        """
        bend: Bend = self.bends[bend_index]
        if side is not None:
            bend.side = side
        if valid is not None:
            bend.isvalid = valid
        if index_apex is not None:
            bend.index_apex = index_apex
        if pt_middle is not None:
            bend.pt_middle = pt_middle
        if pt_centroid is not None:
            bend.pt_centroid = pt_centroid

    def _compute_bend_properties(
        self: Self, sinuo_thres: float, n: float, bend_index: int
    ) -> tuple[BendSide, bool, int, npt.NDArray[np.float64]]:
        """Compute bend properties (side, validity, apex and middle points).

        Parameters:
        ----------
            sinuo_thres (float): Sinuosity threshold used to discriminate valid
                bends.
            n (float): exponent value
            bend_index (int) Bend index.

        Returns:
        -------
            tuple[BendSide, bool, int, npt.NDArray[np.float64]]: tuple containing
                side, isvalid, apex_index, and pt_middle

        """
        side: BendSide = self.get_bend_side(bend_index)
        isvalid: bool = self.check_if_bend_is_valid(sinuo_thres, bend_index)
        index_apex: int = self.find_bend_apex(n, bend_index)
        pt_middle: npt.NDArray[np.float64] = self.compute_bend_middle(bend_index)
        return side, isvalid, index_apex, pt_middle

    # work in progress
    def gather_consecutive_invalid_bends(self: Self, sinuo_thres: float) -> None:
        """Gather consecutive bends when some are unvalid.

        Parameters:
        ---------
            sinuo_thres (float): sinuosity threshold
        """
        new_bends: list[Bend] = []
        for i, bend in enumerate(self.bends):
            bend.id = len(new_bends)  # update bend id
            new_bends += [bend]
            if self.check_if_bend_is_valid(sinuo_thres, i):
                continue

            while i + 1 < len(self.bends) and not self.bends[i + 1].isvalid:
                new_bends[-1] = new_bends[-1] + self.bends[i + 1]
                i += 1
        self.bends_filtered = new_bends
        logger.info("bends filtered")

    # work in progress
    def filter_bends(self: Self) -> bool:
        """Filter bends when some are unvalid.

        Returns:
        ----------
            bool: True when filtering sucessfully ended.
        """
        k = 0
        self.bends_filtered = []
        for i, bend in enumerate(self.bends):
            # if k>0, bend already gathered with bend i-1
            if k > 0:
                k -= 1
                continue

            # if the bend i is valid it is saved
            if bend.isvalid:
                self.bends_filtered += [bend]
            else:
                # look for the next valid bend
                k = 1
                while i + k < len(self.bends) and not self.bends[i + k].isvalid:
                    k += 1

                if i == 0:
                    self.bends_filtered += [self.bends[i]]
                    for j in range(1, k + 1):
                        self.bends_filtered[-1] = (
                            self.bends_filtered[-1] + self.bends[i + j]
                        )
                else:
                    self.bends_filtered[-1] = self.bends_filtered[-1] + self.bends[i]
                    # if the last bend is not valid, add to it to the previous one and continue the loop
                    if i + k == len(self.bends):
                        continue

                # if k is even, or the last bends are not valid
                # add all bends (until the next valid one included) to the last valid bend
                if (k % 2 != 0) or (
                    i + k == len(self.bends) - 1 and self.bends[i + k].isvalid
                ):
                    for j in range(1, k + 1):
                        self.bends_filtered[-1] = (
                            self.bends_filtered[-1] + self.bends[i + j]
                        )

                # if k is odd, means that the 2 consecutive valid bends are not by the same side
                elif k % 2 == 0:
                    # get the middle
                    index_apex_filtered: Optional[int] = self.bends_filtered[
                        -1
                    ].index_apex
                    assert index_apex_filtered is not None, "Apex is undefined"
                    cl_pt_apex0: ClPoint = self.cl_points[index_apex_filtered]
                    index_apex: Optional[int] = self.bends[i + k].index_apex
                    assert index_apex is not None, "Apex is undefined"
                    cl_pt_apex1: ClPoint = self.cl_points[index_apex]
                    best_s = (cl_pt_apex0._s + cl_pt_apex1._s) / 2.0
                    # get the inflection point the closest from the middle
                    ls_inflex = np.array(
                        [
                            best_s
                            - self.cl_points[self.bends[i + j].index_inflex_up]._s
                            for j in range(k)
                        ]
                    )
                    n = ls_inflex.argmin()

                    # gather the last valid bend until those until the middle
                    for j in range(1, n):
                        self.bends_filtered[-1] = (
                            self.bends_filtered[-1] + self.bends[i + j]
                        )
                    # gather the bend in the middle the next ones until the next valid one included
                    self.bends_filtered += [self.bends[i + n]]
                    for j in range(n + 1, k + 1):
                        self.bends_filtered[-1] = (
                            self.bends_filtered[-1] + self.bends[i + j]
                        )

        return True

    def _compute_index_max_curvature(self: Self, bend_index: int) -> int:
        """Compute the index of maximum curvature along a bend.

        Args:
            bend_index (int): bend index

        Returns:
            int: index of max curvature from upstream inflection point

        """
        bend: Bend = self.bends[bend_index]
        # get max curvature
        curvatures: npt.NDArray[np.float64] = np.abs(
            self.get_bend_curvature_filtered(bend.id)
        )

        assert curvatures.size > 0, (
            "Smoothed curvatures were not found, "
            + "index of max curvature was not computed."
        )
        return int(np.argmax(curvatures))

    def _compute_bend_apex_probability_user_weights(
        self: Self,
        bend_index: int,
        curvature_weight: float,
        amplitude_weight: float,
        length_weight: float,
    ) -> npt.NDArray[np.float64]:
        """Compute bend apex probability according to input weights.

        Apex probability corresponds to the probability [0, 1] of a channel point
        being an apex according to its curvature, its distance to the middle
        point of the bend, and its distance to the closest inflection point.
        The result is stored in Bend.apex_probability and as a property of
        channel points.

        Parameters:
        ----------
            bend_index (int): Index of the bend to treat.
            curvature_weight (float): Weight [0, 1] used for curvature property.
            amplitude_weight (float): Weight [0, 1] used for the distance to
                the middle point.
            length_weight (float): Weight [0, 1] used for the distance to
                inflection points.

        Returns:
        ----------
            npt.NDArray[np.float64]: Apex probability of each channel point.

        """
        bend: Bend = self.bends[bend_index]
        # renormalization of weights such the sum equals 1
        tot: float = curvature_weight + amplitude_weight + length_weight
        if tot != 1.0:
            curvature_weight /= tot
            amplitude_weight /= tot
            length_weight /= tot

        apex_probability: npt.NDArray[np.float64] = np.full(
            bend.get_nb_points(), np.nan
        )

        # get max curvature
        curvatures: npt.NDArray[np.float64] = np.abs(
            self.get_bend_curvature_filtered(bend.id)
        )

        assert curvatures.size > 0, (
            "Error during calculation of apex "
            + "probabilities. Filtered curvatures were not found."
        )

        curv_max: float = max(curvatures)
        if curv_max < 1e-6:
            curvatures = np.ones_like(curvatures)
            curv_max = 1.0

        # get max amplitude
        amplitudes: npt.NDArray[np.float64] = np.full(bend.get_nb_points(), np.nan)
        # TODO: to test with BendClPointIndexIter
        for i, pt_index in enumerate(
            range(bend.index_inflex_up, bend.index_inflex_down + 1, 1)
        ):
            amplitudes[i] = abs(
                self._compute_distance_to_bend_middle(bend_index, pt_index)
            )

        ampl_max: float = float(np.max(amplitudes))

        # low sinuosity bends - use the distance perpendicular to inflection point line
        cl_pt_inflex_up: ClPoint = self.cl_points[bend.index_inflex_up]
        cl_pt_inflex_down: ClPoint = self.cl_points[bend.index_inflex_down]
        d_inflex: float = float(
            np.linalg.norm(cl_pt_inflex_down.pt - cl_pt_inflex_up.pt) / 2.0
        )
        if ampl_max <= 1.01 * d_inflex:
            for i, pt_index in enumerate(
                range(bend.index_inflex_up, bend.index_inflex_down + 1, 1)
            ):
                amplitudes[i] = abs(
                    self._compute_distance_orthogonal_to_inflection_segment(
                        bend_index, pt_index
                    )
                )
            ampl_max = float(np.max(amplitudes))

        if ampl_max < 1e-6:
            amplitudes = np.ones_like(amplitudes)
            ampl_max = 1.0

        # length array
        lengths: npt.NDArray[np.float64] = 1 - np.abs(
            (bend.get_nb_points() - 1) / 2.0 - np.arange(bend.get_nb_points())
        ) / ((bend.get_nb_points() - 1) / 2.0)
        length_max: float = float(np.max(np.abs(lengths)))

        if length_max < 1e-6:
            lengths = np.ones_like(lengths)
            length_max = 1

        # apex probability
        apex_probability = (
            curvature_weight * curvatures / curv_max
            + amplitude_weight * amplitudes / ampl_max
            + length_weight * lengths / length_max
        )

        # set apex probability of inflection point to 0
        assert apex_probability is not None, "Apex probability list is undefined"
        apex_probability[0] = 0.0
        apex_probability[-1] = 0.0
        return apex_probability

    def set_bend_apex_probability_user_weights(
        self: Self,
        apex_proba_weights: tuple[float, float, float],
    ) -> None:
        """Set bend apex probability using user-defined weights.

        Parameters:
        ----------
            apex_proba_weights (tuple[float, float, float]): Weights for apex
                probability calculation. Apex probability depends on channel
                point curvature, bend amplitude (m), and the distance from
                inflection points.

        """
        for bend in self.bends:
            bend.index_max_curv = self._compute_index_max_curvature(bend.id)
            bend.apex_probability = self._compute_bend_apex_probability_user_weights(
                bend.id,
                apex_proba_weights[0],
                apex_proba_weights[1],
                apex_proba_weights[2],
            )
            for index, value in enumerate(bend.apex_probability):
                self.set_property_point(
                    bend.index_inflex_up + index,
                    PropertyNames.APEX_PROBABILITY.value,
                    value,
                )

    def check_if_bend_is_valid(self: Self, sinuo_thres: float, bend_index: int) -> bool:
        """Check if a bend is valid.

        Parameters:
        ----------
            sinuo_thres (float): Sinuosity threshold used to discriminate valid
                bends.
            bend_index (int): Index of the bend to treat.

        Returns:
        ----------
            bool: True if the bend is valid, False otherwise.

        """
        bend: Bend = self.bends[bend_index]
        cl_pt_inflex_up: ClPoint = self.cl_points[bend.index_inflex_up]
        cl_pt_inflex_down: ClPoint = self.cl_points[bend.index_inflex_down]

        lentgh: float = abs(cl_pt_inflex_down._s - cl_pt_inflex_up._s)
        d_inflex: float = cpf.distance(cl_pt_inflex_up.pt, cl_pt_inflex_down.pt)
        sinuo: float = 1.0
        if d_inflex > 0.0:
            sinuo = lentgh / d_inflex

        # TODO: refactor to manually set max threshold
        return (sinuo >= sinuo_thres) and (sinuo < 10.0)

    def get_bend_side(self: Self, bend_index: int) -> BendSide:
        """Compute bend side.

        Parameters:
        ----------
            bend_index (int): Index of the bend to treat.

        Returns:
        ----------
            Bend_side: Bend side, either Bend_side.UP or Bend_side.DOWN.

        """
        bend: Bend = self.bends[bend_index]
        curv: float = 0.0
        for cl_pt in self.cl_points[bend.index_inflex_up : bend.index_inflex_down + 1]:
            curv += cl_pt.curvature_filtered()
        return BendSide.UP if curv > 0 else BendSide.DOWN

    # apex from apex probability using automatic calculation
    def find_bend_apex_from_weights(self: Self, bend_index: int) -> int:
        """Find bend apex.

        Parameters:
        ----------
            bend_index (int): Index of the bend to treat.

        Returns:
        ----------
            int: Bend apex index.

        """
        bend: Bend = self.bends[bend_index]
        apex_probability: Optional[npt.NDArray[np.float64]] = bend.apex_probability
        assert apex_probability is not None, "Apex probability is undefined."

        # get maxima
        peak_indexes, _ = find_peaks(apex_probability, height=(0.6, 1.0))

        if len(peak_indexes) == 0:
            return -1

        # if apex_probability get a single global maximum, this is the apex
        apex_index: int = peak_indexes[0]
        # else, apex is the mean index between local maxima
        if len(peak_indexes) > 1:
            apex_index = int(round(np.mean(peak_indexes), 0))

        return bend.index_inflex_up + apex_index

    def find_bend_apex(self: Self, n: float, bend_index: int) -> int:
        """Find bend apex from cummulative curvature function.

        Parameters:
        ----------
            n (float): exponent value

            bend_index (int): Index of the bend to treat.

        Returns:
        ----------
            int: Bend apex index.

        """
        bend: Bend = self.bends[bend_index]
        curvature: npt.NDArray[np.float64] = np.abs(
            self.get_bend_curvature_filtered(bend.id)
        )
        apex_index = cpf.compute_median_curvature_index(curvature, n)
        return bend.index_inflex_up + apex_index

    def find_all_bend_apex_user_weights(
        self: Self, apex_proba_weights: tuple[float, float, float]
    ) -> None:
        """Find bend apex and update Bend for all bends of the centerline.

        Parameters:
        ----------
            apex_proba_weights (tuple[float, float, float]): Weights for apex
                probability calculation. Apex probability depends on channel
                point curvature, bend amplitude (m), and the distance from
                inflection points.

        """
        self.set_bend_apex_probability_user_weights(apex_proba_weights)
        nb_procs: int = get_nb_procs()
        if nb_procs == 1:
            for bend in self.bends:
                bend.index_apex = self.find_bend_apex_from_weights(bend.id)
        else:
            with Pool(processes=nb_procs) as pool:
                outputs = pool.map(
                    self.find_bend_apex_from_weights,  # type: ignore
                    [bend.id for bend in self.bends],
                )

            # update bends
            for bend_index, index_apex in enumerate(outputs):
                self._update_bend(bend_index, index_apex=index_apex)

    def find_all_bend_apex(self: Self, n: float) -> None:
        """Find bend apex and update Bend for all bends of the centerline.

        Parameters:
        ----------
            n (float): exponent value

        """
        nb_procs: int = get_nb_procs()
        if nb_procs == 1:
            for bend in self.bends:
                bend.index_apex = self.find_bend_apex(n, bend.id)
        else:
            partial_find_bend_apex = functools.partial(self.find_bend_apex, n)
            with Pool(processes=nb_procs) as pool:
                outputs = pool.map(
                    partial_find_bend_apex,  # type: ignore
                    [bend.id for bend in self.bends],
                )

            # update bends
            for bend_index, index_apex in enumerate(outputs):
                self._update_bend(bend_index, index_apex=index_apex)

    def compute_all_bend_middle(self: Self) -> None:
        """Compute bend middle point and update Bend for all bends of the centerline."""
        nb_procs: int = get_nb_procs()
        if nb_procs == 1:
            for bend in self.bends:
                self.compute_bend_middle(bend.id)
        else:
            with Pool(processes=nb_procs) as pool:
                outputs = pool.map(
                    self.compute_bend_middle,  # type: ignore
                    [bend.id for bend in self.bends],
                )

            # update bends
            for bend_index, pt_middle in enumerate(outputs):
                self._update_bend(bend_index, pt_middle=pt_middle)

    def compute_bend_middle(self: Self, bend_index: int) -> npt.NDArray[np.float64]:
        """Compute bend middle point and update Bend.

        Parameters:
        ----------
            bend_index (int): Index of the bend to treat.

        Returns:
        -------
            npt.NDArray[np.float64]: middle point coordinates

        """
        bend: Bend = self.bends[bend_index]
        cl_pt_inflex_up: ClPoint = self.cl_points[bend.index_inflex_up]
        cl_pt_inflex_down: ClPoint = self.cl_points[bend.index_inflex_down]
        # compute_colinear deals with 2D points only
        pt_middle: npt.NDArray[np.float64] = cpf.compute_colinear(
            cl_pt_inflex_up.pt, cl_pt_inflex_down.pt, 0.5
        )
        # add z coordinate
        z_middle: float = 0.5 * (cl_pt_inflex_up.pt[2] + cl_pt_inflex_down.pt[2])
        return np.array((pt_middle[0], pt_middle[1], z_middle))

    def compute_all_bend_centroid(self: Self) -> None:
        """Compute bend centroid point and update Bend for all bends of the centerline.

        Bend centroid is the barycenter of the polygon defined by the centerline
        between upstream and downstream inflection points and is closed between
        these points.

        """
        nb_procs: int = get_nb_procs()
        if nb_procs == 1:
            for bend in self.bends:
                bend.pt_centroid = self.compute_bend_centroid(bend.id)
        else:
            with Pool(processes=nb_procs) as pool:
                outputs = pool.map(
                    self.compute_bend_centroid,  # type: ignore
                    [bend.id for bend in self.bends],
                )

            # update bends
            for bend_index, pt_centroid in enumerate(outputs):
                self._update_bend(bend_index, pt_centroid=pt_centroid)

    def compute_bend_centroid(self: Self, bend_index: int) -> npt.NDArray[np.float64]:
        """Compute bend centroid point and update Bend.

        Parameters:
        ----------
            bend_index (int): Index of the bend to treat.

        Returns:
        -------
            npt.NDArray[np.float64]: bend centroid point coordinates

        """
        polygon: Polygon | LineString = self.compute_bend_polygon(bend_index)
        return np.array(polygon.centroid.coords[0])

    def compute_bend_polygon(self: Self, bend_index: int) -> Polygon | LineString:
        """Compute bend polygon.

        Bend polygon defined by the centerline between upstream and downstream
        inflection points and is closed between these points.

        Parameters:
        ----------
            bend_index (int): Index of the bend to treat.

        Returns:
        -------
            Polygon | LineString: bend polygon

        """
        bend: Bend = self.bends[bend_index]
        pts: list[npt.NDArray[np.float64]] = [
            cl_pt.pt
            for cl_pt in self.cl_points[
                bend.index_inflex_up : bend.index_inflex_down + 1
            ]
        ]

        assert len(pts) > 1, "Bend is not valid, polygon cannot be computed."
        if len(pts) > 2:
            return Polygon(pts)

        return LineString(pts)

    def _compute_distance_to_bend_middle(
        self: Self, bend_index: int, cl_pt_index: int
    ) -> float:
        """Compute the distance between a channel point and the middle point of a bend.

        .. WARNING:: Channel point must belongs to the bend.

        Parameters:
        ----------
            bend_index (int): Index of the bend to treat.
            cl_pt_index (int): Index of the channel point.

        Returns:
        ----------
            float: Distance (m) if channel point belongs to the bend.

        """
        bend: Bend = self.bends[bend_index]

        assert (cl_pt_index >= bend.index_inflex_up) & (
            cl_pt_index <= bend.index_inflex_down
        ), (
            "Cannot compute distance to bend middle: "
            + "point does not belong to the given bend."
        )

        if bend.pt_middle is None:
            bend.pt_middle = self.compute_bend_middle(bend.id)

        assert bend.pt_middle is not None, "Bend middle point is undefined"
        cl_pt: ClPoint = self.cl_points[cl_pt_index]
        return cpf.distance(cl_pt.pt, bend.pt_middle)

    def _compute_distance_orthogonal_to_inflection_segment(
        self: Self, bend_index: int, cl_pt_index: int
    ) -> float:
        """Compute the distance between a channel point and the bend basal line.

        Bend basal line is the segment defined by bend inflection points.

        .. WARNING:: Channel point must belongs to the bend.

        Parameters:
        ----------
            bend_index (int): Index of the bend to treat.
            cl_pt_index (int): Index of the channel point.

        Returns:
        ----------
            float: Distance (m) if channel point belongs to the bend.

        """
        bend: Bend = self.bends[bend_index]

        assert (cl_pt_index >= bend.index_inflex_up) & (
            cl_pt_index <= bend.index_inflex_down
        ), (
            "Cannot compute orthogonal amplitude: "
            + "point does not belong to the given bend."
        )

        cl_pt: ClPoint = self.cl_points[cl_pt_index]
        cl_pt_inflex_up: ClPoint = self.cl_points[bend.index_inflex_up]
        cl_pt_inflex_down: ClPoint = self.cl_points[bend.index_inflex_down]

        pt_proj: npt.NDArray[np.float64] = cpf.project_orthogonal(
            cl_pt.pt, cl_pt_inflex_up.pt, cl_pt_inflex_down.pt
        )
        return cpf.distance(pt_proj, cl_pt.pt)
