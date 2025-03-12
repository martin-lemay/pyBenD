# SPDX-FileCopyrightText: Copyright 2025 Martin Lemay <martin.lemay@mines-paris.org>
# SPDX-FileContributor: Martin Lemay
# ruff: noqa: E402 # disable Module level import not at top of file
import functools
from collections import Counter
from concurrent.futures import ProcessPoolExecutor as Pool
from typing import Optional, Self, cast

import dtw  # type: ignore[import-untyped]
import numpy as np
import numpy.typing as npt
import pandas as pd  # type: ignore[import-untyped]
from scipy.signal import savgol_filter  # type: ignore[import-untyped]
from shapely import affinity  # type: ignore
from shapely.geometry import (  # type: ignore
    LineString,
    MultiPoint,
    Point,
    Polygon,
)

import pybend.algorithms.centerline_process_function as cpf
from pybend.model.Bend import Bend, parse_bend_uid
from pybend.model.BendEvolution import BendEvolution
from pybend.model.Centerline import Centerline
from pybend.model.ClPoint import ClPoint
from pybend.model.enumerations import (
    BendConnectionMethod,
    BendSide,
    CreateSectionMethod,
    PropertyNames,
)
from pybend.model.Isoline import ChannelCrossSection, Isoline
from pybend.model.Section import Section
from pybend.utils.globalParameters import get_nb_procs
from pybend.utils.logging import logger

__doc__ = r"""
This module defines CenterlineCollection objet that stores the successive
positions of a sinuous channel from a single channel-belt through time as well
as utilities.

"""


class CenterlineCollection:
    def __init__(
        self: Self,
        map_centerline_data: dict[int, pd.DataFrame],
        spacing: float,
        smooth_distance: float,
        use_fix_nb_points: bool,
        curvature_filtering_window: int = 5,
        sinuo_thres: float = 1.05,
        n: float = 2,
        compute_curvature: bool = True,
        interpol_props: bool = True,
        find_bends: bool = True,
    ) -> None:
        """Store all the successive Centerline objects from a single channel-belt.

        Parameters:
        ----------
            map_centerline_data (dict[int, pd.DataFrame]): dictionnary
              containing for each age a dataframe containing centerline data
            spacing (float): Target distance  (m) between two consecutive
              channel points after resampling, or number of points if
              use_fix_nb_points is True
            smooth_distance (float): smoothing distance for channel point
              location and curvature
            use_fix_nb_points (bool): if True, spacing is the number of points
              of the resampled centerline, otherwise it is the target distance.
            curvature_filtering_window (int, optional): Number of points to
              consider for curvature filtering window.

              Defaults to 5.
            sinuo_thres (float, optional): threshold above which bends are valid.

              Defaults to 1.05.
            n (float): exponent of the curvature distribution function.

                Defaults to 2.
            compute_curvature (bool, optional): if True, recompute channel point
              curvature after resampling.

              Defaults to True.
            interpol_props (bool, optional): if True, interpolate channel point
              properties along channel points after resampling.

              Defaults to True.
            find_bends (bool, optional): if True, automatically compute curvature
              and interpolate properties and detect bends along each centerline.

              Defaults to True.
        """
        #: dictionnary to store Centerline object at each age
        self.centerlines: dict[int, Centerline] = {}
        #: list of BendEvolution objects
        self.bends_evol: list[BendEvolution] = []

        #: True if centerline matching was computed
        self._centerline_matching_computed: bool = False
        #: True if bend tracking was computed
        self.bends_tracking_computed: bool = False
        #: True if sections lines were defined
        self.sections_computed: bool = False
        #: list of section lines
        self.section_lines: list[LineString] = []
        # list of Section objects
        self.sections: list[Section] = []

        #: True if real kinematics were computed
        self.real_kinematics_computed: bool = False
        #: True if apparent kinematics were computed
        self.apparent_kinematics_computed: bool = False

        # 1. create Centerline instances
        if get_nb_procs() < 2:
            self.initialize_monoproc(
                map_centerline_data,
                spacing,
                smooth_distance,
                use_fix_nb_points,
                curvature_filtering_window,
                sinuo_thres,
                n,
                compute_curvature,
                interpol_props,
                find_bends,
            )
        else:
            self.initialize_multiproc(
                map_centerline_data,
                spacing,
                smooth_distance,
                use_fix_nb_points,
                curvature_filtering_window,
                sinuo_thres,
                n,
                compute_curvature,
                interpol_props,
                find_bends,
            )

        logger.info("Centerline_collection instanciated.")

    def initialize_multiproc(
        self: Self,
        map_centerline_data: dict[int, pd.DataFrame],
        spacing: float,
        smooth_distance: float,
        use_fix_nb_points: bool,
        curvature_filtering_window: int,
        sinuo_thres: float,
        n: float,
        compute_curvature: bool,
        interpol_props: bool,
        find_bends: bool,
    ) -> bool:
        """Initialize Centerline_evoution object using multiprocessing.

        Parameters:
        ----------
            map_centerline_data (dict[int, pd.DataFrame]): dictionnary
              containing for each age a dataframe containing centerline data
            spacing (float): Target distance  (m) between two consecutive
              channel points after resampling, or number of points if
              use_fix_nb_points is True
            smooth_distance (float): smoothing distance for channel point
              location and curvature
            use_fix_nb_points (bool): if True, spacing is the number of points
              of the resampled centerline, otherwise it is the target distance.
            curvature_filtering_window (int): Number of points to
              consider for curvature smoothing window.
            sinuo_thres (float): threshold above which bends are valid.
            n (float): exponent of the curvature distribution function.
            compute_curvature (bool): if True, recompute channel point
              curvature after resampling.
            interpol_props (bool): if True, interpolate channel point
              properties along channel points after resampling.
            find_bends (bool): if True, automatically compute curvature
              and interpolate properties and detect bends along each centerline.

        Returns:
        ----------
          bool: True if calculation successfully eneded.

        """
        nb_procs = get_nb_procs()

        # use ProcessPoolExecutor because non deamion process (i.e., a
        # sub-process can invoke itself child processes)
        with Pool(max_workers=nb_procs) as pool:
            inputs: list[int] = list(map_centerline_data.keys())
            partial_create_centerline = functools.partial(
                self._create_centerline,
                map_centerline_data,
                spacing,
                smooth_distance,
                use_fix_nb_points,
                curvature_filtering_window,
                sinuo_thres,
                n,
                compute_curvature,
                interpol_props,
                find_bends,
            )
            outputs = pool.map(partial_create_centerline, inputs)

            centerlines: dict[int, Centerline] = {}
            for age, centerline in zip(inputs, outputs, strict=False):
                assert centerline is not None, (
                    f"Centerline {age} is undefined."
                )
                centerlines[age] = centerline

            self.centerlines = dict(sorted(centerlines.items()))
        return True

    def initialize_monoproc(
        self: Self,
        map_centerline_data: dict[int, pd.DataFrame],
        spacing: float,
        smooth_distance: float,
        use_fix_nb_points: bool,
        curvature_filtering_window: int,
        sinuo_thres: float,
        n: float,
        compute_curvature: bool,
        interpol_props: bool,
        find_bends: bool,
    ) -> bool:
        """Initialize Centerline_evoution object using monoprocessing.

        Parameters:
        ----------
            map_centerline_data (dict[int, pd.DataFrame]): dictionnary
              containing for each age a dataframe containing centerline data
            spacing (float): Target distance  (m) between two consecutive
              channel points after resampling, or number of points if
              use_fix_nb_points is True
            smooth_distance (float): smoothing distance for channel point
              location and curvature
            use_fix_nb_points (bool): if True, spacing is the number of points
              of the resampled centerline, otherwise it is the target distance.
            curvature_filtering_window (int): Number of points to
              consider for curvature smoothing window.
            sinuo_thres (float): threshold above which bends are valid.
            n (float): exponent of the curvature distribution function.
            compute_curvature (bool): if True, recompute channel point
              curvature after resampling.
            interpol_props (bool): if True, interpolate channel point
              properties along channel points after resampling.
            find_bends (bool): if True, automatically compute curvature
              and interpolate properties and detect bends along each centerline.

        Returns:
        ----------
          bool: True if calculation successfully eneded.

        """
        ages: list[int] = list(map_centerline_data.keys())
        for key in ages:
            self.centerlines[key] = self._create_centerline(
                map_centerline_data,
                spacing,
                smooth_distance,
                use_fix_nb_points,
                curvature_filtering_window,
                sinuo_thres,
                n,
                compute_curvature,
                interpol_props,
                find_bends,
                key,
            )
        return True

    def _create_centerline(
        self: Self,
        map_centerline_data: dict[int, pd.DataFrame],
        spacing: float,
        smooth_distance: float,
        use_fix_nb_points: bool,
        curvature_filtering_window: int,
        sinuo_thres: float,
        n: float,
        compute_curvature: bool,
        interpol_props: bool,
        find_bends: bool,
        age: int,
    ) -> Centerline:
        """Create self.centerlines dictionnary.

        Centerline_collection object initialization consists in creating the
        dictionnary self.centerlines containing ages a keys and Centerline object
        as values.

        Parameters:
        ----------
            spacing (float): Target distance  (m) between two consecutive
                channel points after resampling, or number of points if
                use_fix_nb_points is True
            smooth_distance (float): smoothing distance for channel point
                location and curvature
            use_fix_nb_points (bool): if True, spacing is the number of points
                of the resampled centerline, otherwise it is the target distance.
            curvature_filtering_window (int): Number of points to
                consider for curvature smoothing window.
            sinuo_thres (float): threshold above which bends are valid.
            n (float): exponent of the curvature distribution function.
            compute_curvature (bool): if True, recompute channel point
                curvature after resampling.
            interpol_props (bool): if True, interpolate channel point
                properties along channel points after resampling.
            find_bends (bool): if True, automatically compute curvature
                and interpolate properties and detect bends along each centerline.
            age (int): centerline age
            data (pd.DataFrame: centerline properties data
            queue (mp.Queue): queue where to dump created Centerline

        Returns:
        --------
            Centerline: Centerline object

        """
        return Centerline(
            age,
            map_centerline_data[age],
            spacing,
            smooth_distance,
            use_fix_nb_points,
            curvature_filtering_window,
            sinuo_thres,
            n,
            compute_curvature,
            interpol_props,
            find_bends,
        )

    def get_all_ages(self: Self) -> npt.NDArray[np.int64]:
        """Get all centerline ages stored in the Centerline_collection.

        Returns:
        ----------
          NDArray[int]: Array of centerline ages sorted in ascending order.

        """
        return np.sort(list(self.centerlines.keys()))

    def get_centerline_property(
        self: Self, age: int, property_name: str
    ) -> npt.NDArray[np.float64]:
        """Get a property of input centerline channel points.

        Agrs:
            age (int): Age of the Centerline.
            property_name (str): Name of the property.

        Returns:
        ----------
          NDArray[float]: Array containing the values of the property for each channel point.

        """
        if age in self.get_all_ages():
            return self.centerlines[age].get_property(property_name)
        else:
            logger.warning(
                f"Warning: centerline {age} is not in the collection."
            )
            return np.array([])

    def set_centerline_constant_properties(
        self: Self, age: int, property_map: dict[str, float]
    ) -> None:
        """Set properties with constant value to the centerline of input age.

        Parameters:
        ----------
            age (float): Age of the Centerline.
        property_map (dict[str, float]): Dictionnary containings property names
            as keys and property values as values.

        """
        if age in self.get_all_ages():
            for name, value in property_map.items():
                values = value * np.ones(self.centerlines[age].get_nb_points())
                self.set_centerline_properties(age, {name: values})
        else:
            logger.warning(
                f"Warning: centerline {age} is not in the collection."
            )

    def set_centerline_properties(
        self: Self, age: int, property_map: dict[str, npt.NDArray[np.float64]]
    ) -> None:
        """Set properties values to the centerline of input age.

        Parameters:
        ----------
            age (float): Age of the Centerline.
        property_map (dict[str, NDArray[float]]): Dictionnary containings
            property names as keys and array of property values as values.

        """
        if age in self.get_all_ages():
            for name, values in property_map.items():
                self.centerlines[age].set_property_all_points(name, values)
        else:
            logger.warning(
                f"Warning: centerline {age} is not in the collection."
            )

    def get_nb_bends_evol(self: Self) -> int:
        """Get the number of BendEvolution stored in self.bends_evol list.

        Returns:
        ----------
            int: Number of BendEvolution.

        """
        return len(self.bends_evol)

    def get_nb_valid_bends_evol(self: Self) -> int:
        """Get the number of valid BendEvolution stored in self.bends_evol list.

        Returns:
        ----------
            int: Number of valid BendEvolution.

        """
        return len(self.get_valid_bends_evol_id())

    def get_valid_bends_evol_id(self: Self) -> list[int]:
        """Get the ids of valid BendEvolution stored in self.bends_evol list.

        Returns:
        --------
            list[int]: Ids of valid BendEvolution.

        """
        return [
            bend_evol.id for bend_evol in self.bends_evol if bend_evol.isvalid
        ]

    def get_bend_evol_side(self: Self, bend_evol_index: int) -> BendSide:
        """Get BendEvolution side.

        BendEvolution side is the most frequent side of its constitutive bends.

        Parameters:
        ----------
            bend_evol_index (int): BendEvolution index.

        Returns:
        --------
            BendSide: BendEvolution side.

        """
        nb_up: int = 0
        nb_down: int = 0
        for age, bend_indexes in self.bends_evol[
            bend_evol_index
        ].bend_indexes.items():
            bend: Bend = self.centerlines[age].bends[bend_indexes[0]]
            if bend.side == BendSide.UP:
                nb_up += 1
            elif bend.side == BendSide.DOWN:
                nb_down += 1
        return BendSide.UP if nb_up > nb_down else BendSide.DOWN

    def match_centerlines(
        self: Self,
        dmax: float = np.inf,
        distance_weight: float = 0.0,
        vel_perturb_weight: float = 0.0,
        curvature_weight: float = 1.0,
        window: int = 5,
        pattern: str = "asymmetric",
    ) -> None:
        """Public method to compute centerline matching.

        Centerline matching consists in applying Dynamic Time Warping on each pair
        of consecutive Centerline to connect channel points together.

        Parameters:
        ----------
            dmax (float, optional): Maximal allowed distance (m) between
              connected channel point.

              Defaults to np.inf.
            distance_weight (float, optional): Weight [0, 1] on channel point
                distance.

                Defaults to 0.
            vel_perturb_weight (float, optional): Weight [0, 1] on velocity
                perturbation.

                Defaults to 0.
            curvature_weight (float, optional): Weight [0, 1] on curvature.

                Defaults to 1.
            window (int, optional): Number of points for filter.

                Defaults to 5.
            pattern (str, optional): Pattern input of dtw.dtw function.

                Defaults to "asymmetric".

        """
        try:
            nb_procs: int = get_nb_procs()
            if nb_procs == 1:
                self._match_centerlines_monoproc(
                    dmax,
                    distance_weight,
                    vel_perturb_weight,
                    curvature_weight,
                    window,
                    pattern,
                )
            else:
                self._match_centerlines_multiproc(
                    dmax,
                    distance_weight,
                    vel_perturb_weight,
                    curvature_weight,
                    window,
                    pattern,
                    nb_procs,
                )
            self._centerline_matching_computed = True
        except Exception as e:
            logger.critical("Centerline matching was not computed due to")
            logger.critical(e)

    def _match_centerlines_monoproc(
        self: Self,
        dmax: float,
        distance_weight: float,
        vel_perturb_weight: float,
        curvature_weight: float,
        window: int,
        pattern: str,
    ) -> bool:
        """Private method to compute centerline matching using monoprocessing.

        Parameters:
        ----------
            dmax (float): Maximal allowed distance (m) between
              connected channel point.
            distance_weight (float): Weight [0, 1] on channel point
                distance.
            vel_perturb_weight (float): Weight [0, 1] on velocity
                perturbation.
            curvature_weight (float): Weight [0, 1] on curvature.
            window (int): Number of points for filter.
            pattern (str): Pattern input of dtw.dtw function.

        Returns:
        ----------
            bool: True if calculation successfully ended.

        """
        all_iter: npt.NDArray[np.int64] = self.get_all_ages()
        prev_key: int = all_iter[0]
        for key in all_iter[1:]:
            indexes: list[int] = self._apply_centerline_warping(
                dmax,
                distance_weight,
                vel_perturb_weight,
                curvature_weight,
                window,
                pattern,
                (key, prev_key),
            )
            self._set_cl_pts_indexes_in_prev_next_centerlines(
                key, prev_key, indexes, dmax
            )
            prev_key = key
        return True

    def _match_centerlines_multiproc(
        self: Self,
        dmax: float,
        distance_weight: float,
        vel_perturb_weight: float,
        curvature_weight: float,
        window: int,
        pattern: str,
        nb_procs: int,
    ) -> bool:
        """Private method to compute centerline matching using multiprocessing.

        Parameters:
        ----------
            dmax (float): Maximal allowed distance (m) between
              connected channel point.
            distance_weight (float): Weight [0, 1] on channel point
                distance.
            vel_perturb_weight (float): Weight [0, 1] on velocity
                perturbation.
            curvature_weight (float): Weight [0, 1] on curvature.
            window (int): Number of points for smoothing filter.
            pattern (str): Pattern input of dtw.dtw function.
            nb_procs (int): number of processors.

        Returns:
        ----------
            bool: True if calculation successfully ended.

        """
        all_iter: npt.NDArray[np.int64] = self.get_all_ages()
        with Pool(max_workers=nb_procs) as pool:
            inputs = np.column_stack((all_iter[1:], all_iter[0:-1]))
            partial_apply_centerline_warping = functools.partial(
                self._apply_centerline_warping,
                dmax,
                distance_weight,
                vel_perturb_weight,
                curvature_weight,
                window,
                pattern,
            )
            outputs = pool.map(partial_apply_centerline_warping, inputs)  # type: ignore

        for (key, prev_key), indexes in zip(inputs, outputs, strict=False):
            self._set_cl_pts_indexes_in_prev_next_centerlines(
                key, prev_key, indexes, dmax
            )

        return True

    def _apply_centerline_warping(
        self: Self,
        dmax: float,
        distance_weight: float,
        vel_perturb_weight: float,
        curvature_weight: float,
        window: int,
        pattern: str,
        keys: tuple[int, int],
    ) -> list[int]:
        """Private method to compute centerline matching.

        Parameters:
        ----------
            dmax (float): Maximal allowed distance (m) between
              connected channel point.
            distance_weight (float): Weight [0, 1] on channel point
                distance.
            vel_perturb_weight (float): Weight [0, 1] on velocity
                perturbation.
            curvature_weight (float): Weight [0, 1] on curvature.
            window (int): Number of points for filter.
            pattern (str): Pattern input of dtw.dtw function.
            keys (tuple[int, int]): age of the current and previous cenerlines

        Returns:
        ----------
            list[int]: List of tuples containing indexes
                of Centerline of age key and indexes of Centerline of age
                prev_key.

        """
        key: int = keys[0]
        prev_key: int = keys[1]
        lx1: npt.NDArray[np.float64] = self.centerlines[key].get_property(
            PropertyNames.CARTESIAN_ABSCISSA.value
        )
        ly1: npt.NDArray[np.float64] = self.centerlines[key].get_property(
            PropertyNames.CARTESIAN_ORDINATE.value
        )

        lcurv1: npt.NDArray[np.float64] = self.centerlines[
            key
        ].get_all_curvature_filtered()

        vel_perturb_prop_name: str = PropertyNames.VELOCITY_PERTURBATION.value
        lvel_perturb1: npt.NDArray[np.float64] = np.zeros_like(lx1)
        if vel_perturb_prop_name in self.centerlines[key].get_property_list():
            lvel_perturb1 = savgol_filter(
                self.centerlines[key].get_property(vel_perturb_prop_name),
                window_length=window,
                polyorder=2,
            )

        lx0: npt.NDArray[np.float64] = self.centerlines[prev_key].get_property(
            PropertyNames.CARTESIAN_ABSCISSA.value
        )
        ly0: npt.NDArray[np.float64] = self.centerlines[prev_key].get_property(
            PropertyNames.CARTESIAN_ORDINATE.value
        )
        lcurv0: npt.NDArray[np.float64] = self.centerlines[
            prev_key
        ].get_all_curvature_filtered()
        lvel_perturb0: npt.NDArray[np.float64] = np.zeros_like(lx0)
        if vel_perturb_prop_name in self.centerlines[key].get_property_list():
            lvel_perturb0 = savgol_filter(
                self.centerlines[prev_key].get_property(vel_perturb_prop_name),
                window_length=window,
                polyorder=2,
            )

        if (len(lcurv1) == 0) | (len(lcurv0) == 0):
            return []

        distance_matrix_vel_pertub: npt.NDArray[np.float64] = np.zeros(
            (len(lcurv1), len(lcurv0))
        )
        distance_matrix_dist: npt.NDArray[np.float64] = np.zeros_like(
            distance_matrix_vel_pertub
        )
        distance_matrix_curv: npt.NDArray[np.float64] = np.zeros_like(
            distance_matrix_vel_pertub
        )
        for i, (x1, y1, vel_perturb1, curv1) in enumerate(
            zip(lx1, ly1, lvel_perturb1, lcurv1, strict=False)
        ):
            for j, (x0, y0, vel_perturb0, curv0) in enumerate(
                zip(lx0, ly0, lvel_perturb0, lcurv0, strict=False)
            ):
                d: float = cpf.distance((x0, y0), (x1, y1))
                if d > dmax:
                    d = 1e9
                distance_matrix_dist[i, j] = d
                distance_matrix_vel_pertub[i, j] = abs(
                    vel_perturb1 - vel_perturb0
                )
                distance_matrix_curv[i, j] = abs(abs(curv1) - abs(curv0))

        if distance_matrix_dist[distance_matrix_dist != 1e9].max() > 0.0:
            distance_matrix_dist /= distance_matrix_dist[
                distance_matrix_dist != 1e9
            ].max()
        if distance_matrix_curv.max() > 0.0:
            distance_matrix_curv /= distance_matrix_curv.max()
        if distance_matrix_vel_pertub.max() > 0.0:
            distance_matrix_vel_pertub /= distance_matrix_vel_pertub.max()

        distance_matrix: npt.NDArray[np.float64] = (
            vel_perturb_weight * distance_matrix_vel_pertub
            + curvature_weight * distance_matrix_curv
            + distance_weight * distance_matrix_dist
        )

        alignment: dtw.DTW = dtw.dtw(
            distance_matrix, keep_internals=False, step_pattern=pattern
        )
        return dtw.warp(alignment, index_reference=True)

    def _set_cl_pts_indexes_in_prev_next_centerlines(
        self: Self, key: int, prev_key: int, indexes: list[int], dmax: float
    ) -> None:
        """Store the results if centerline matching in each Centerline object.

        The method create the lists index_cl_pts_prev_centerline and
        index_cl_pts_next_centerline of each Centerline object and also
        upate ChannelPoint cl_pt_index_prev and cl_pt_index_next members.

        Parameters:
        ----------
            key (float): Age of the first matched centerline.
            prev_key (float): Age of the second matched centerline.
            indexes (list[int]): List of channel point indexes in the
                Centerline of age prev_key.
            dmax (float): Maximal allowed distance (m) between
                connected channel points.

        """
        self.centerlines[key].index_cl_pts_prev_centerline = self.centerlines[
            key
        ].get_nb_points() * [-1]
        self.centerlines[prev_key].index_cl_pts_next_centerline = [
            [] for _ in range(self.centerlines[prev_key].get_nb_points())
        ]
        for index_key, index_prev_key in enumerate(indexes):
            pt1: npt.NDArray[np.float64] = (
                self.centerlines[key].cl_points[index_key].pt
            )
            pt0: npt.NDArray[np.float64] = (
                self.centerlines[prev_key].cl_points[index_prev_key].pt
            )
            if cpf.distance(pt1, pt0) < dmax:
                self.centerlines[key].index_cl_pts_prev_centerline[
                    index_key
                ] = int(index_prev_key)
                self.centerlines[prev_key].index_cl_pts_next_centerline[
                    index_prev_key
                ] += [index_key]

                # add info into cl_point
                self.centerlines[key].cl_points[
                    index_key
                ].cl_pt_index_prev += [index_prev_key]
                self.centerlines[prev_key].cl_points[
                    index_prev_key
                ].cl_pt_index_next += [index_key]

    def connect_bends(
        self: Self,
        bend_evol_validity: int = 2,
        method: BendConnectionMethod = BendConnectionMethod.APEX,
        dmax: float = np.inf,
        weighting_func_type: str = "uniform",
    ) -> bool:
        """Pulic method to create BendEvolution objects by connecting bends.

        Parameters:
        ----------
            bend_evol_validity (int, optional): Minimum number of bends in the
                BendEvolution to be considered as valid.

                Defaults to 2.
            method (Bend_connection_method, optional): Method to use to
                compute BendEvolution.

                Defaults to Bend_connection_method.MATCHING.
            dmax (float, optional): Maximum allowed distance (m) between 2
                successive apex points.

                Defaults to np.inf.
            weighting_func_type (str, optional): Weighting function type to
                use.

                Defaults to "uniform".

        Returns:
        ----------
            bool: True if the function ends witout errors

        """
        # reset self.bends_evol that will be updated later on
        self.bends_evol = []
        match method:
            case BendConnectionMethod.APEX:
                return self._connect_bends_apex(dmax, bend_evol_validity)
            case BendConnectionMethod.CENTROID:
                return self._connect_bends_centroid(dmax, bend_evol_validity)
            case BendConnectionMethod.MATCHING:
                return self._connect_bends_from_matching(
                    bend_evol_validity, weighting_func_type=weighting_func_type
                )
            case _:
                methods = [str(meth) for meth in list(BendConnectionMethod)]  # type: ignore[unreachable]
                raise TypeError(
                    "Input method is wrong. Methods are: ".join(methods)
                )

    # TODO: refactor with same method as _connect_bends_from_matching
    def _connect_bends_apex(
        self: Self, dmax: float, bend_evol_validity: int
    ) -> bool:
        """Connect bends from the successive centerlines using apex distance.

        Connect bends from the successive centerlines of the collection by searching
        for the closests apex points that belongs to bends of same side

        Parameters:
        ----------
            dmax (float): Maximum allowed distance (m) between 2
                successive apex points.
            bend_evol_validity (int): Minimum number of bends in the
                BendEvolution to be considered as valid.

        Returns:
        ----------
            bool: True if the function ends witout errors

        """
        bends_evol: list[list[Bend]] = []
        prev_key: int = 0
        # connect apexes backward through time
        for i, key in enumerate(self.get_all_ages()[::-1]):
            if i == 0:
                bends_evol += [
                    [bend]
                    for bend in self.centerlines[key].bends
                    if bend.isvalid
                ]
                prev_key = key
                continue

            for _, bend in enumerate(self.centerlines[key].bends):
                if not bend.isvalid:
                    continue

                # look for the closest apex
                dist: npt.NDArray[np.float64] = np.nan * np.zeros(
                    len(bends_evol)
                )
                index: int = -1
                for k, bend_saved in enumerate(bends_evol):
                    # if the last bend_saved was added at the previous key
                    # and is on the same side as bend
                    if (
                        bend_saved[-1].isvalid
                        and bend_saved[-1].age == prev_key
                        and bend_saved[-1].side == bend.side
                    ):
                        # compute the distance between apex points
                        index_apex_prev: Optional[int] = bend_saved[
                            -1
                        ].index_apex
                        if index_apex_prev is None:
                            continue
                        pt1: npt.NDArray[np.float64] = (
                            self.centerlines[prev_key]
                            .cl_points[index_apex_prev]
                            .pt
                        )

                        index_apex_cur: Optional[int] = bend.index_apex
                        if index_apex_cur is None:
                            continue
                        pt2: npt.NDArray[np.float64] = (
                            self.centerlines[key].cl_points[index_apex_cur].pt
                        )
                        dist[k] = cpf.distance(pt1, pt2)

                # take the index of the minimum distance if this distance is lower than dmax
                if np.isfinite(dist).any() and np.nanmin(dist) < dmax:
                    index = int(np.nanargmin(dist))

                # a bend is found
                if index > -1:
                    bends_evol[index] += [bend]
                # no bend found
                else:
                    bends_evol += [[bend]]

            prev_key = key

        for bend_evol_id, bends in enumerate(bends_evol):
            bend_indexes: dict[int, list[int]] = {
                bend.age: [
                    bend.id,
                ]
                for bend in bends
            }
            if len(bend_indexes) > 1:
                self.centerlines[bend.age].bends[
                    bend.id
                ].bend_evol_id = bend_evol_id
                self.bends_evol += [
                    BendEvolution(
                        bend_indexes,
                        i,
                        0,
                        len(bend_indexes) > bend_evol_validity,
                    )
                ]

        self.bends_tracking_computed = len(self.bends_evol) > 0
        return True

    # TODO: refactor with same method as _connect_bends_from_matching
    def _connect_bends_centroid(
        self: Self, dmax: float, bend_evol_validity: int
    ) -> bool:
        """Connect bends from the successive centerlines using apex centroids.

        Connect bends from the successive centerlines of the collection by
        searching for the closests centroids points that belongs to bends of same
        side.

        Parameters:
        ----------
            dmax (float): Maximum allowed distance (m) between 2
                successive apex points.
            bend_evol_validity (int): Minimum number of bends in the
                BendEvolution to be considered as valid.

        Returns:
        ----------
            bool: True if the function ends witout errors

        """
        bends_evol: list[list[Bend]] = []
        prev_key: int = 0
        # connect apexes backward through time
        for i, key in enumerate(self.get_all_ages()[::-1]):
            if i == 0:
                bends_evol += [
                    [bend]
                    for bend in self.centerlines[key].bends
                    if bend.isvalid
                ]
                prev_key = key
                continue

            for _, bend in enumerate(self.centerlines[key].bends):
                if not bend.isvalid:
                    continue

                # look for the closest apex
                dist = np.nan * np.zeros(len(bends_evol))
                index: int = -1
                for k, bend_saved in enumerate(bends_evol):
                    # if the last bend_saved was added at the previous key
                    # and is on the same side as bend
                    if (
                        bend_saved[-1].isvalid
                        and bend_saved[-1].age == prev_key
                        and bend_saved[-1].side == bend.side
                    ):
                        # compute the distance between upstream inflex points (more stable than apex)
                        assert bend_saved[-1].pt_centroid is not None, (
                            "Centroid is undefined"
                        )
                        assert bend.pt_centroid is not None, (
                            "Centroid is undefined"
                        )
                        dist[k] = cpf.distance(
                            bend_saved[-1].pt_centroid, bend.pt_centroid
                        )

                # take the index of the minimum distance if this distance is lower than dmax
                if np.isfinite(dist).any() and np.nanmin(dist) < dmax:
                    index = int(np.nanargmin(dist))

                # a bend is found
                if index > 0:
                    bends_evol[index] += [bend]
                # no bend found, create a new list of bends
                else:
                    bends_evol += [[bend]]

                prev_key = key

        for bends in bends_evol:
            bend_indexes: dict[int, list[int]] = {
                bend.age: [
                    bend.id,
                ]
                for bend in bends
            }
            self.bends_evol += [
                BendEvolution(
                    bend_indexes, i, 0, len(bend_indexes) > bend_evol_validity
                )
            ]

        self.bends_tracking_computed = len(self.bends_evol) > 0
        return True

    def _connect_bends_from_matching(
        self: Self,
        bend_evol_validity: int,
        weighting_func_type: str = "uniform",
    ) -> bool:
        """Connect bends from the successive centerlines using matching results.

        Connect bends from the successive centerlines of the collection by
        searching for the greatest number of connected channel points between
        connected bends.

        Parameters:
        ----------
            bend_evol_validity (int): Minimum number of bends in the
                BendEvolution to be considered as valid.
            weighting_func_type (str, optional): Weighting function type to
                use.

        Returns:
        ----------
            bool: True if the function ends witout errors

        """
        assert self._centerline_matching_computed, (
            "Centerline matching was not "
            + "computed. First use match_centerlines method."
        )
        bend_uids: list[int] = []
        # TODO: compute weights for each point according to weighting function

        # connect bends forward through time
        all_iter: npt.NDArray[np.int64] = self.get_all_ages()
        for i, key in enumerate(all_iter[:-1]):
            next_key: int = all_iter[i + 1]
            for bend_index, bend in enumerate(self.centerlines[key].bends):
                # initialize bend_uids with uids with first centerline bends
                if i == 0:
                    bend_uids += [bend.uid]

                # looks for the bend index of each point of the bend
                counts = self._get_bend_index_connection_counts(
                    bend, key, next_key, True, weighting_func_type
                )
                if len(counts) == 0:
                    continue

                # keep bend index the most present whatever the side
                # TODO: to take into account weigths
                # next_bend_index = Counter(counts).most_common(1)[0][0]
                # if next_bend_index is not None:
                #   self._add_bend_connection(key, next_key, bend_index, next_bend_index)

                # keep all bend index with more than 50% connected cl_points
                for next_bend_index, cnt in Counter(counts).items():
                    if next_bend_index is None:
                        continue
                    nb_pts = (
                        self.centerlines[next_key]
                        .bends[next_bend_index]
                        .get_nb_points()
                    )
                    ratio = float(cnt) / float(nb_pts)
                    if ratio > 0.2:
                        uid: int = self._add_bend_connection(
                            key, next_key, bend_index, next_bend_index
                        )
                        bend_uids += [uid]

        self._create_bend_evolution(bend_evol_validity, bend_uids)
        return True

    def _add_bend_connection(
        self: Self,
        key: int,
        next_key: int,
        cur_bend_index: int,
        next_bend_index: int,
    ) -> int:
        """Register the reciprocal connection of bends of ages key and next_key.

        Parameters:
        ----------
            key (int): Age of centerline.
            next_key (int): Age of successive centerline.
            cur_bend_index (int): index of the bend of age 'key'.
            next_bend_index (int): index of the bend of age 'next_key'.

        Returns:
        ----------
            int: bend unique id

        """
        bend: Bend = self.centerlines[key].bends[cur_bend_index]
        next_bend_uid = self.centerlines[next_key].bends[next_bend_index].uid
        bend.add_bend_connection_next(next_bend_uid)
        self.centerlines[next_key].bends[
            next_bend_index
        ].add_bend_connection_prev(bend.uid)
        return next_bend_uid

    def _get_bend_index_connection_counts(
        self: Self,
        bend: Bend,
        key: int,
        next_key: int,
        next_iter: int,
        weighting_func_type: str,
    ) -> list[int | None]:
        """Get the number of times each connected bends is.

        Parameters:
        ----------
            bend (Bend): Bend object.
            key (int): Age of centerline.
            next_key (int): Next age to consider. It can be lower or greater
                depending on next_iter.
            next_iter (bool): If True, look for next  centerline.
            weighting_func_type (str): Weighting function type to use.

        Returns:
        ----------
            list[int | None]: Bend uid for each connected bends, or None if no
            bend is connected.

        """
        counts: list[int | None] = []
        # does not consider inflection points since they are part of both neighboring bends
        for index in range(
            bend.index_inflex_up + 1, bend.index_inflex_down, 1
        ):
            if next_iter:
                pts_index_next = (
                    self.centerlines[key].cl_points[index].cl_pt_index_next
                )
            else:
                pts_index_next = (
                    self.centerlines[key].cl_points[index].cl_pt_index_prev
                )

            if pts_index_next is None:
                counts += [None]  # type: ignore[unreachable]
                continue

            for pt_index_next in pts_index_next:
                bend_uid_next: Optional[int] = self.centerlines[
                    next_key
                ].get_bend_index_from_cl_pt_index(pt_index_next)
                if bend_uid_next is None:
                    continue
                # TODO: to take into account weigths
                counts += [bend_uid_next]
        return counts

    def _create_bend_evolution(
        self: Self, bend_evol_validity: int, bend_uids: list[int]
    ) -> None:
        """Create the list self.bends_evol with BendEvolution objects.

        Parameters:
        ----------
            bend_evol_validity (int): Minimum number of bends in the
                BendEvolution to be considered as valid.

        """
        bend_evol_all: list[list[int]] = []
        while len(bend_uids) > 0:
            uid: int = bend_uids[0]
            # bend_uids.pop(uid)
            bend_uids.remove(uid)
            to_treat: set[int] = set()
            to_treat.add(uid)
            bend_evol: list[int] = []
            self._update_bend_evolution(to_treat, bend_uids, bend_evol)
            bend_evol_all += [bend_evol]

        self.bends_evol = []
        for i, bend_evol_ids in enumerate(bend_evol_all):
            dico_indexes: dict[int, list[int]] = {}
            for uid in bend_evol_ids:
                age, bend_id = parse_bend_uid(uid)
                dico_indexes[age] = [
                    bend_id,
                ]
            self.bends_evol += [
                BendEvolution(
                    dico_indexes, i, 0, len(bend_evol_ids) > bend_evol_validity
                )
            ]

        self.bends_tracking_computed = True

    def _update_bend_evolution(
        self: Self,
        to_treat: set[int],
        bend_uids: list[int],
        bend_evol: list[int],
    ) -> None:
        """Collect Bend indexes of connected bends.

        The method works recursively from the lists Bend.bend_uid_next and
        Bend.bend_uid_prev.

        Parameters:
        ----------
            to_treat (set[int]): List of remaining Bend uid to treat.
            bend_uids (list[int]): List of all remaining Bend uid to treat.
            bend_evol (list[int]): List of Bend uid that belongs to a same
                BendEvolution.

        """
        if len(to_treat) == 0:
            return

        uid: int = to_treat.pop()
        age, bend_id = parse_bend_uid(uid)
        bend: Bend = self.centerlines[age].bends[bend_id]
        if (bend.bend_uid_next is not None) and (len(bend.bend_uid_next) > 0):
            for bend_next_uid in bend.bend_uid_next:
                if bend_next_uid in bend_uids:
                    to_treat.add(bend_next_uid)
                    # bend_uids.discard(bend_next_uid)
                    bend_uids.remove(bend_next_uid)

        if (bend.bend_uid_prev is not None) and (len(bend.bend_uid_prev) > 0):
            for bend_prev_uid in bend.bend_uid_prev:
                if bend_prev_uid in bend_uids:
                    to_treat.add(bend_prev_uid)
                    # bend_uids.discard(bend_prev_uid)
                    bend_uids.remove(bend_prev_uid)
        bend_evol += [uid]
        self._update_bend_evolution(to_treat, bend_uids, bend_evol)

    def set_section_lines(
        self: Self,
        pts_start: list[tuple[float, float]],
        pts_end: list[tuple[float, float]],
    ) -> None:
        """Manually set section lines across Centerline_collection.

        Parameters:
        ----------
            pts_start (list[tuple[float, float]]): List of 2D coordinates of
                section starting points.
            pts_end : list[tuple[float, float]]): List of 2D coordinates of
                section ending points.

        """
        self.section_lines = []
        for pt_start, pt_end in zip(pts_start, pts_end, strict=False):
            section_line = LineString((pt_start, pt_end))
            self.section_lines += [section_line]

    def create_section_lines(
        self: Self,
        point_name: CreateSectionMethod = CreateSectionMethod.MIDDLE,
    ) -> None:
        """Automatically create section lines across BendEvolution objects.

        Section lines are created

        Parameters:
        ----------
            point_name (Create_section_method, optional) Name of the point to
                use to create the section.

                Defaults to Create_section_method.MIDDLE.

        Raises:
            TypeError: in case of wrong method

        """
        match point_name:
            case CreateSectionMethod.APEX:
                self._create_section_lines_from_neighboring_apex()
            case CreateSectionMethod.MIDDLE:
                self._create_section_lines_from_bend(point_name)
            case CreateSectionMethod.CENTROID:
                self._create_section_lines_from_bend(point_name)
            case _:
                methods = [str(meth) for meth in list(CreateSectionMethod)]  # type: ignore[unreachable]
                raise TypeError(
                    "Unkown method for section line creation. Methods are either: ".join(
                        methods
                    )
                )

    def _create_section_lines_from_bend(
        self: Self, point_name: CreateSectionMethod
    ) -> None:
        """Create section lines across BendEvolution using MIDDLE or CENTROID methods.

        Automatically create section lines across BendEvolution objects such as
        lines goes by the middle or centroid point of the last bend of each
        BendEvolution.

        Parameters:
        ----------
            method (Create_section_method): Name of point to use.
                It can be either Create_section_method.MIDDLE or
                Create_section_method.CENTROID.

        """
        self.section_lines = []
        for i, bend in enumerate(
            self.centerlines[self.get_all_ages()[-1]].bends
        ):
            if (
                not bend.isvalid
                or (i == 0)
                or (
                    i
                    > len(self.centerlines[self.get_all_ages()[-1]].bends) - 2
                )
            ):
                continue
            key = self.get_all_ages()[-1]
            pt_end = None
            if point_name == CreateSectionMethod.MIDDLE:
                pt_end = bend.pt_middle
            elif point_name == CreateSectionMethod.CENTROID:
                pt_end = bend.pt_centroid
            else:
                methods = [
                    str(CreateSectionMethod.MIDDLE),
                    str(CreateSectionMethod.CENTROID),
                ]
                raise TypeError(
                    "Unkown method for section line creation. Methods are either: ".join(
                        methods
                    )
                )
            assert pt_end is not None, (
                "Undefined end point for section line creation"
            )
            assert bend.index_apex > -1, "Bend apex is undefined."
            pt_apex: npt.NDArray[np.float64] = (
                self.centerlines[key].cl_points[bend.index_apex].pt
            )
            section_line = LineString((pt_apex, pt_end))
            self.section_lines += [section_line]

    def _create_section_lines_from_neighboring_apex(self: Self) -> None:
        """Create section lines across BendEvolution using APEX method.

        Automatically create section lines across BendEvolution objects such as
        lines goes by the apex point of the last bend of each BendEvolution.

        """
        self.section_lines = []
        for i, bend in enumerate(
            self.centerlines[self.get_all_ages()[-1]].bends
        ):
            if (
                not bend.isvalid
                or bend.index_apex < 0
                or (i == 0)
                or (
                    i
                    > len(self.centerlines[self.get_all_ages()[-1]].bends) - 2
                )
            ):
                continue

            key = self.get_all_ages()[-1]
            prev_bend = self.centerlines[key].bends[i - 1]
            next_bend = self.centerlines[key].bends[i + 1]

            if prev_bend.isvalid and (prev_bend.index_apex > -1):
                pt0 = self.centerlines[key].cl_points[prev_bend.index_apex].pt
            else:
                k = prev_bend.index_inflex_up + int(
                    (prev_bend.get_nb_points() + 0.5) / 2.0
                )
                pt0 = self.centerlines[key].cl_points[k].pt

            if next_bend.isvalid and (next_bend.index_apex > -1):
                pt1 = self.centerlines[key].cl_points[next_bend.index_apex].pt
            else:
                k = next_bend.index_inflex_up + int(
                    (next_bend.get_nb_points() + 0.5) / 2
                )
                pt1 = self.centerlines[key].cl_points[k].pt

            pt_end: npt.NDArray[np.float64] = (
                np.array(pt0) + np.array(pt1)
            ) / 2.0
            pt_apex: npt.NDArray[np.float64] = (
                self.centerlines[key].cl_points[bend.index_apex].pt
            )
            section_line = LineString((pt_apex, pt_end))
            self.section_lines += [section_line]

    def _create_all_channel_sections(
        self: Self, section_line: LineString
    ) -> tuple[list[Isoline], list[int]]:
        # list of isoline instances to store channel locations
        isolines: list[Isoline] = []
        cl_pt_indexes: list[int] = []

        # research window area defined by the square whose the section is a diagonal
        line2 = affinity.rotate(section_line, 90)  # take the perpendicular
        window = Polygon(
            (
                np.array(section_line.coords)[0],
                np.array(line2.coords)[0],
                np.array(section_line.coords)[1],
                np.array(line2.coords)[1],
            )
        )

        # for each centerline
        for key in self.get_all_ages():
            # for each point of the centerline
            for j, cl_pt in enumerate(self.centerlines[key].cl_points):
                # if the point is inside the window
                if (window.contains(Point(cl_pt.pt))) and (
                    j < len(self.centerlines[key].cl_points) - 2
                ):
                    cl_pt2: ClPoint = self.centerlines[key].cl_points[j + 1]
                    cl_line = LineString([cl_pt.pt, cl_pt2.pt])
                    intersect = section_line.intersection(cl_line)
                    # if the intersection exists
                    if not intersect.is_empty:
                        # interpolate channel points properties to the intersection point
                        d: float = (
                            intersect.distance(Point(cl_pt.pt))
                            / cl_line.length
                        )
                        cl_pt = cl_pt * (1 - d) + cl_pt2 * d

                        channel_section: ChannelCrossSection = (
                            ChannelCrossSection(key, cl_pt)
                        )
                        channel_section.complete_channel_shape(11)
                        isolines += [cast(Isoline, channel_section)]
                        cl_pt_indexes += [j]
        return isolines, cl_pt_indexes

    # done here because may collect centerline points outside bend_evol
    def find_points_on_sections(
        self: Self,
        thres: int = 1,
        flow_dir: npt.NDArray[np.float64] = np.array([1.0, 0]),
        cl_collec_id: int = 0,
    ) -> bool:
        """Collect channels from Centerline_collection that are intersected by section lines.

        This method create the Section objects with intersected channel geometry
        and store them in the list self.sections.

        Parameters:
        ----------
            thres (int, optional): Minimum number of intersected lines to create
                the Section object.

                Defaults to 1.
            flow_dir (NDArray[float], optional): Direction vector of the flow
                used to orientate the Sections.

                Defaults to np.array([1,0]).
            cl_collec_id (int, optional): Centerline_collection id used in the
                Section id.

                Defaults to 0.

        Returns:
        ----------
            bool: True if calculation successfully ended.

        """
        if not self.section_lines:
            logger.error("Please first define section lines")
            return False

        self.sections = []
        # for each bend_evol
        for i, section_line in enumerate(self.section_lines):
            # find all channel points intersecting the section line
            # list of isoline instances to store channel locations
            isolines: list[Isoline]
            cl_pt_indexes: list[int]
            isolines, cl_pt_indexes = self._create_all_channel_sections(
                section_line
            )

            # create the section
            if len(isolines) > thres:
                for isoline, cl_pt_index in zip(
                    isolines, cl_pt_indexes, strict=False
                ):
                    # notify bend that is intersected by the section line
                    bend_index = self._get_bend_index_from_cl_point_index(
                        cl_pt_index, isoline.age
                    )
                    self.centerlines[isoline.age].bends[
                        bend_index
                    ].add_intersected_section_index(i)

                bend_id: str = f"{cl_collec_id}-{bend_index}"
                ide: str = f"{cl_collec_id}-{i}"

                bounds: MultiPoint = section_line.boundary
                pt_start: npt.NDArray[np.float64] = np.array(
                    bounds.geoms[0].coords[0]
                )  # type: ignore
                pt_stop: npt.NDArray[np.float64] = np.array(
                    bounds.geoms[1].coords[0]
                )  # type: ignore
                self.sections += [
                    Section(
                        ide,
                        bend_id,
                        pt_start,
                        pt_stop,
                        isolines,
                        None,
                        flow_dir,
                    )
                ]

        self.sections_computed = True
        return True

    def _get_bend_index_from_cl_point_index(
        self: Self, cl_pt_index: int, age: int
    ) -> int:
        """Get bend index from the index of a channel point of age.

        Parameters:
        ----------
            cl_pt_index (int): Index of the channel point.
            age (float): Age of the Centerline where the channel point belongs to.

        Returns:
        ----------
            int: Index of the Bend the channel point belongs to.

        """
        if cl_pt_index < self.centerlines[age].bends[0].index_inflex_up:
            return 0
        elif cl_pt_index > self.centerlines[age].bends[-1].index_inflex_down:
            return len(self.centerlines[age].bends) - 1

        for bend_index, bend in enumerate(self.centerlines[age].bends):
            if (cl_pt_index >= bend.index_inflex_up) and (
                cl_pt_index < bend.index_inflex_down
            ):
                return bend_index
        return bend_index

    def find_all_bend_middle(
        self: Self, smooth_trajectory: bool = False
    ) -> None:
        """Compute the middle points of all bends of the Centerline_collection.

        Parameters:
        ----------
            smooth_trajectory (bool, optional): If True, middle trajectory
                through BendEvolution is smoothed and stored in the list
                BendEvolution.middle_trajec_smooth.

                Defaults to False.

        """
        for age in self.get_all_ages():
            self.centerlines[age].compute_all_bend_middle()

        if smooth_trajectory:
            assert self.bends_tracking_computed, (
                "Bends were not tracked over "
                + "time. First apply connect_bends."
            )
            if get_nb_procs() == 1:
                for bend_evol_index in range(self.get_nb_bends_evol()):
                    self.bends_evol[
                        bend_evol_index
                    ].middle_trajec_smooth = self._smooth_bend_middle_trajec(
                        bend_evol_index
                    )
            else:
                with Pool(max_workers=get_nb_procs()) as pool:
                    inputs = range(self.get_nb_bends_evol())
                    outputs = pool.map(self._smooth_bend_middle_trajec, inputs)

                    for bend_evol_index, trajec in zip(
                        inputs, outputs, strict=False
                    ):
                        self.bends_evol[
                            bend_evol_index
                        ].middle_trajec_smooth = trajec

    def _smooth_bend_middle_trajec(
        self: Self,
        bend_evol_index: int,
    ) -> list[npt.NDArray[np.float64]]:
        bend_evol: BendEvolution = self.bends_evol[bend_evol_index]
        lx: list[float] = []
        ly: list[float] = []
        for age in bend_evol.get_all_ages():
            bend_id: int = bend_evol.bend_indexes[age][0]
            pt_middle: Optional[npt.NDArray[np.float64]] = (
                self.centerlines[age].bends[bend_id].pt_middle
            )
            assert pt_middle is not None, "Middle point is undefined"
            lx += [pt_middle[0]]
            ly += [pt_middle[1]]

        print(np.round(lx, 4).tolist())
        print(np.round(ly, 4).tolist())
        lx_new, ly_new = cpf.resample_path(
            np.array(lx), np.array(ly), len(bend_evol.get_all_ages()), 0.5
        )
        return [np.array([x, y]) for x, y in zip(lx_new, ly_new, strict=False)]

    def find_all_bend_centroid(
        self: Self, smooth_trajectory: bool = False
    ) -> None:
        """Compute the centroid points of all bends of the Centerline_collection.

        Parameters:
        ----------
            smooth_trajectory (bool, optional): If True, centroid trajectory
                through BendEvolution is smoothed and stored in the list
                BendEvolution.centroid_trajec_smooth.

                Defaults to False.

        """
        for age in self.get_all_ages():
            self.centerlines[age].compute_all_bend_centroid()

        if smooth_trajectory:
            assert self.bends_tracking_computed, (
                "Bends were not tracked over "
                + "time. First apply connect_bends."
            )

            if get_nb_procs() == 1:
                for bend_evol_index in range(self.get_nb_bends_evol()):
                    self.bends_evol[
                        bend_evol_index
                    ].centroid_trajec_smooth = (
                        self._smooth_bend_centroid_trajec(bend_evol_index)
                    )
            else:
                with Pool(max_workers=get_nb_procs()) as pool:
                    inputs = range(self.get_nb_bends_evol())
                    outputs = pool.map(
                        self._smooth_bend_centroid_trajec, inputs
                    )

                    for bend_evol_index, trajec in zip(
                        inputs, outputs, strict=False
                    ):
                        self.bends_evol[
                            bend_evol_index
                        ].centroid_trajec_smooth = trajec

    def _smooth_bend_centroid_trajec(
        self: Self,
        bend_evol_index: int,
    ) -> list[npt.NDArray[np.float64]]:
        bend_evol: BendEvolution = self.bends_evol[bend_evol_index]
        lx: list[float] = []
        ly: list[float] = []

        for age in bend_evol.get_all_ages():
            bend_id: int = bend_evol.bend_indexes[age][0]
            pt_centroid: Optional[npt.NDArray[np.float64]] = (
                self.centerlines[age].bends[bend_id].pt_centroid
            )
            assert pt_centroid is not None, "Centroid point is undefined"
            lx += [pt_centroid[0]]
            ly += [pt_centroid[1]]

        lx_new, ly_new = cpf.resample_path(
            np.array(lx), np.array(ly), len(bend_evol.get_all_ages()), 0.5
        )
        return [np.array([x, y]) for x, y in zip(lx_new, ly_new, strict=False)]

    def find_all_bend_apex_user_weights(
        self: Self,
        apex_proba_weights: tuple[float, float, float],
        smooth_trajectory: bool = False,
    ) -> None:
        """Find the apex of all bends of the Centerline_collection.

        Parameters:
        ----------
            apex_proba_weights (tuple[float, float, float]): Weights for apex
                probability calculation. Apex probability depends on channel
                point curvature, bend amplitude (m), and the distance from
                inflection points.

            smooth_trajectory (bool, optional): If True, middle trajectory
                through BendEvolution is smoothed and stored in the list
                BendEvolution.middle_trajec_smooth.

                Defaults to False.

        """
        for age in self.get_all_ages():
            self.centerlines[age].find_all_bend_apex_user_weights(
                apex_proba_weights
            )

        if smooth_trajectory:
            self._smooth_bend_apex_trajec()
        else:
            for bend_evol_index in range(self.get_nb_bends_evol()):
                self.bends_evol[
                    bend_evol_index
                ].apex_trajec_smooth = self._get_bend_evol_apex_points(
                    bend_evol_index
                )

    def find_all_bend_apex(
        self: Self, n: float, smooth_trajectory: bool = False
    ) -> None:
        """Find the apex of all bends of the Centerline_collection.

        Parameters:
        ----------
            n (float): exponent of the curvature distribution function.

            smooth_trajectory (bool, optional): If True, middle trajectory
                through BendEvolution is smoothed and stored in the list
                BendEvolution.middle_trajec_smooth.

                Defaults to False.

        """
        for age in self.get_all_ages():
            self.centerlines[age].find_all_bend_apex(n)

        if smooth_trajectory:
            self._smooth_bend_apex_trajec()
        else:
            for bend_evol_index in range(self.get_nb_bends_evol()):
                self.bends_evol[
                    bend_evol_index
                ].apex_trajec_smooth = self._get_bend_evol_apex_points(
                    bend_evol_index
                )

    def _smooth_bend_apex_trajec(self: Self) -> None:
        """Smooth apex trajectory of all bend evolutions."""
        assert self.bends_tracking_computed, (
            "Bends were not tracked over " + "time. First apply connect_bends."
        )

        if get_nb_procs() == 1:
            for bend_evol_index in range(self.get_nb_bends_evol()):
                self.bends_evol[
                    bend_evol_index
                ].apex_trajec_smooth = self._do_smooth_bend_apex_trajec(
                    bend_evol_index
                )
        else:
            with Pool(max_workers=get_nb_procs()) as pool:
                inputs = range(self.get_nb_bends_evol())
                outputs = pool.map(self._do_smooth_bend_apex_trajec, inputs)

                for bend_evol_index, trajec in zip(
                    inputs, outputs, strict=False
                ):
                    self.bends_evol[
                        bend_evol_index
                    ].apex_trajec_smooth = trajec

    def _do_smooth_bend_apex_trajec(
        self: Self,
        bend_evol_index: int,
    ) -> list[npt.NDArray[np.float64]]:
        """Smooth apex trajectory of one bend evolution.

        Parameters:
        ----------
            bend_evol_index (int): bend evolution index

        Returns:
        -------
            list[npt.NDArray[np.float64]]: Smooth apex trajectory coordinates.

        """
        bend_evol: BendEvolution = self.bends_evol[bend_evol_index]
        lx: list[float] = []
        ly: list[float] = []
        for age in bend_evol.get_all_ages():
            bend_ids: list[int] = bend_evol.bend_indexes[age]
            index_apex: Optional[int] = (
                self.centerlines[age].bends[bend_ids[0]].index_apex
            )
            if index_apex is None:
                continue
            lx += [self.centerlines[age].cl_points[index_apex].pt[0]]
            ly += [self.centerlines[age].cl_points[index_apex].pt[1]]

        lx_new, ly_new = cpf.resample_path(
            np.array(lx), np.array(ly), len(bend_evol.get_all_ages()), 0.5
        )
        return [np.array([x, y]) for x, y in zip(lx_new, ly_new, strict=False)]

    def _get_bend_evol_apex_points(
        self: Self, bend_evol_index: int
    ) -> list[npt.NDArray[np.float64]]:
        """Get the coordinates of all bend apex of the bend evolution.

        Parameters:
        ----------
            bend_evol_index (int): bend evolution index

        Returns:
        -------
            npt.NDArray[np.float64]: apex coordinates

        """
        bend_evol: BendEvolution = self.bends_evol[bend_evol_index]
        apex_coords: npt.NDArray[np.float64] = np.full(
            (bend_evol.get_all_ages(), 2), np.nan
        )
        for i, age in enumerate(bend_evol.get_all_ages()):
            bend_ids: list[int] = bend_evol.bend_indexes[age]
            index_apex: Optional[int] = (
                self.centerlines[age].bends[bend_ids[0]].index_apex
            )
            if index_apex is None:
                continue
            apex_coords[i] = self.centerlines[age].cl_points[index_apex].pt
        return [
            np.array([x, y])
            for x, y in zip(apex_coords[:, 0], apex_coords[:, 1], strict=False)
        ]
