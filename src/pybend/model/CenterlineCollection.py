# SPDX-FileCopyrightText: Copyright 2025 Martin Lemay <martin.lemay@mines-paris.org>
# SPDX-FileContributor: Martin Lemay
# ruff: noqa: E402 # disable Module level import not at top of file

"""Centerline-collection model.

Defines `CenterlineCollection`, which stores successive channel centerlines
through time for a single channel belt.
"""

import functools
from concurrent.futures import ProcessPoolExecutor as Pool
from typing import TYPE_CHECKING, Optional, Self, cast

import dtw  # type: ignore[import-not-found]
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

import pybend.algorithms.centerline_process_functions as cpf
import pybend.algorithms.geometry_functions as geom
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

if TYPE_CHECKING:
    pass


class CenterlineCollection:
    """A time series of `Centerline` objects representing channel evolution."""

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
        merge_straight_bends: bool = False,
    ) -> None:
        """Store the successive Centerline objects from a single channel-belt.

        Args:
            map_centerline_data (dict[int, pd.DataFrame]): dictionnary
                containing for each age a dataframe containing centerline data
            spacing (float): Target distance  (m) between two consecutive
                channel points after resampling, or number of points if
                use_fix_nb_points is True
            smooth_distance (float): smoothing distance for channel point
                location and curvature
            use_fix_nb_points (bool): if True, spacing is the number of points
                of resampled centerline, otherwise it is the target distance.
            curvature_filtering_window (int, optional): Number of points to
                consider for curvature filtering window.
                Defaults to 5.
            sinuo_thres (float, optional): threshold above which bends are
                valid.
                Defaults to 1.05.
            n (float): exponent of the curvature distribution function.
                Defaults to 2.
            compute_curvature (bool, optional): if True, recompute channel
                point curvature after resampling.
                Defaults to True.
            interpol_props (bool, optional): if True, interpolate channel point
                properties along channel points after resampling.
                Defaults to True.
            find_bends (bool, optional): if True, automatically compute
                curvature and interpolate properties and detect bends along
                each centerline.
                Defaults to True.
            merge_straight_bends (bool, optional): If True, merge
                consecutive STRAIGHT bends into a single bend after
                bend detection in each centerline.
                Defaults to False.
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
                merge_straight_bends,
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
                merge_straight_bends,
            )

        logger.info("CenterlineCollection instanciated.")

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
        merge_straight_bends: bool,
    ) -> bool:
        """Initialize Centerline_evoution object using multiprocessing.

        Args:
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
              and interpolate properties and detect bends along each centerline
            merge_straight_bends (bool): If True, merge consecutive STRAIGHT
              bends into a single bend after bend detection in each centerline.

        Returns:
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
                merge_straight_bends,
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
        merge_straight_bends: bool,
    ) -> bool:
        """Initialize Centerline_evoution object using monoprocessing.

        Args:
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
              and interpolate properties and detect bends along each centerline
            merge_straight_bends (bool): If True, merge consecutive STRAIGHT
              bends into a single bend after bend detection in each centerline.

        Returns:
            bool: True if calculation successfully eneded.

        """
        ages: list[int] = sorted(map_centerline_data.keys())
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
                merge_straight_bends,
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
        merge_straight_bends: bool,
        age: int,
    ) -> Centerline:
        """Create self.centerlines dictionnary.

        CenterlineCollection object initialization consists in creating the
        dictionnary self.centerlines containing ages a keys and Centerline
        object as values.

        Args:
            map_centerline_data (dict[int, pd.DataFrame]): dictionnary of ages
                and dataframe defining channel point location and properties.
            spacing (float): Target distance  (m) between two consecutive
                channel points after resampling, or number of points if
                use_fix_nb_points is True
            smooth_distance (float): smoothing distance for channel point
                location and curvature
            use_fix_nb_points (bool): if True, spacing is the number of points
                of resampled centerline, otherwise it is the target distance.
            curvature_filtering_window (int): Number of points to
                consider for curvature smoothing window.
            sinuo_thres (float): threshold above which bends are valid.
            n (float): exponent of the curvature distribution function.
            compute_curvature (bool): if True, recompute channel point
                curvature after resampling.
            interpol_props (bool): if True, interpolate channel point
                properties along channel points after resampling.
            find_bends (bool): if True, automatically compute curvature and
                interpolate properties and detect bends along each centerline.
            merge_straight_bends (bool): If True, merge consecutive STRAIGHT
                bends into a single bend after bend detection in each
                centerline.
            age (int): centerline age
            data (pd.DataFrame: centerline properties data
            queue (mp.Queue): queue where to dump created Centerline

        Returns:
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
            merge_straight_bends,
        )

    def get_all_ages(self: Self) -> npt.NDArray[np.int64]:
        """Get all centerline ages stored in the CenterlineCollection.

        Returns:
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
            NDArray[float]: Array containing the values of the property for
            each channel point.

        """
        if age in self.get_all_ages():
            centerline = self.centerlines[age]
            if property_name not in centerline.get_property_list():
                logger.warning(
                    f"Warning: property {property_name} is not defined on "
                    + f"centerline {age}."
                )
                return np.full(centerline.get_nb_points(), np.nan)
            return centerline.get_property(property_name)
        else:
            logger.warning(
                f"Warning: centerline {age} is not in the collection."
            )
            return np.array([])

    def set_centerline_constant_properties(
        self: Self, age: int, property_map: dict[str, float]
    ) -> None:
        """Set properties with constant value to the centerline of input age.

        Args:
            age (float): Age of the Centerline.
            property_map (dict[str, float]): Dictionnary containings property
                names as keys and property values as values.

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

        Args:
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
            int: Number of BendEvolution.

        """
        return len(self.bends_evol)

    def get_nb_valid_bends_evol(self: Self) -> int:
        """Get the number of valid BendEvolution stored in bends_evol list.

        Returns:
            int: Number of valid BendEvolution.

        """
        return len(self.get_valid_bends_evol_id())

    def get_valid_bends_evol_id(self: Self) -> list[int]:
        """Get the ids of valid BendEvolution stored in self.bends_evol list.

        Returns:
            list[int]: Ids of valid BendEvolution.

        """
        return [
            bend_evol.id for bend_evol in self.bends_evol if bend_evol.isvalid
        ]

    def get_bend_evol_side(self: Self, bend_evol_index: int) -> BendSide:
        """Get BendEvolution side.

        BendEvolution side is the most frequent side (UP or DOWN) of its
        constitutive bends (the first bend at each age is considered).

        Args:
            bend_evol_index (int): BendEvolution index.

        Returns:
            BendSide: BendEvolution side. STRAIGHT if no bend is UP or DOWN
            (e.g., all bends are STRAIGHT), UNKNOWN if there are as many UP
            bends as DOWN bends.

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
        if nb_up == 0 and nb_down == 0:
            return BendSide.STRAIGHT
        if nb_up == nb_down:
            return BendSide.UNKNOWN
        return BendSide.UP if nb_up > nb_down else BendSide.DOWN

    def match_centerlines(
        self: Self,
        dmax: float = np.inf,
        distance_weight: float = 0.0,
        vel_perturb_weight: float = 0.0,
        curvature_weight: float = 1.0,
        window: int = 5,
        pattern: str = "asymmetric",
    ) -> bool:
        """Public method to compute centerline matching.

        Centerline matching consists in applying Dynamic Time Warping on each
        pair of consecutive Centerline to connect channel points together.

        If the matching fails, the error is logged, the matching is flagged
        as not computed and previously stored matching results are cleared.

        Args:
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

        Returns:
            bool: True if centerline matching was successfully computed,
            False otherwise.
        """
        self._centerline_matching_computed = False
        try:
            self._clear_centerline_matching()
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
        except Exception as e:
            logger.error(f"Centerline matching was not computed due to: {e}")
            self._clear_centerline_matching()
            return False
        self._centerline_matching_computed = True
        return True

    def _clear_centerline_matching(self: Self) -> None:
        """Clear centerline matching results stored in centerlines.

        Reset Centerline.index_cl_pts_prev_centerline,
        Centerline.index_cl_pts_next_centerline and ClPoint.cl_pt_index_prev
        and ClPoint.cl_pt_index_next of all centerlines.
        """
        for centerline in self.centerlines.values():
            centerline.index_cl_pts_prev_centerline = []
            centerline.index_cl_pts_next_centerline = []
            for cl_pt in centerline.cl_points:
                cl_pt.cl_pt_index_prev = []
                cl_pt.cl_pt_index_next = []

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

        Args:
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

        Args:
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

        Args:
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
        if (
            vel_perturb_prop_name
            in self.centerlines[prev_key].get_property_list()
        ):
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
                d: float = geom.distance((x0, y0), (x1, y1))
                if d > dmax:
                    d = 1e9
                distance_matrix_dist[i, j] = d
                distance_matrix_vel_pertub[i, j] = abs(
                    vel_perturb1 - vel_perturb0
                )
                distance_matrix_curv[i, j] = abs(abs(curv1) - abs(curv0))

        # normalize distances by the maximum distance lower than dmax (if
        # any pair of points is closer than dmax)
        valid_dist: npt.NDArray[np.float64] = distance_matrix_dist[
            distance_matrix_dist != 1e9
        ]
        if valid_dist.size > 0 and valid_dist.max() > 0.0:
            distance_matrix_dist /= valid_dist.max()
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

        Args:
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
        # reset channel point connections from a previous matching
        for cl_pt in self.centerlines[key].cl_points:
            cl_pt.cl_pt_index_prev = []
        for cl_pt in self.centerlines[prev_key].cl_points:
            cl_pt.cl_pt_index_next = []
        for index_key, index_prev_key in enumerate(indexes):
            pt1: npt.NDArray[np.float64] = (
                self.centerlines[key].cl_points[index_key].pt
            )
            pt0: npt.NDArray[np.float64] = (
                self.centerlines[prev_key].cl_points[index_prev_key].pt
            )
            if geom.distance(pt1, pt0) < dmax:
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
        norm_width: float = 0.0,
        weights: dict[str, float] | None = None,
        gap_penalty_metric: str = "sinuosity",
        gap_penalty_scale: float = 2.5,
        spatial_tolerance: float = 2.0,
        step_pattern: str = "asymmetric",
        birth_death_threshold: float = 10.0,
        max_apex_distance_factor: float = 10.0,
    ) -> bool:
        """Create BendEvolution objects by connecting bends.

        Args:
            bend_evol_validity (int, optional): A BendEvolution is
                considered valid if it contains strictly more than
                bend_evol_validity bends.
                Defaults to 2.
            method (BendConnectionMethod, optional): Method to use.
                Defaults to BendConnectionMethod.APEX.
            dmax (float, optional): Maximum allowed distance (m)
                between 2 successive apex points.
                Defaults to np.inf.
            weighting_func_type (str, optional): Weighting function
                type. Defaults to "uniform".
            norm_width (float, optional): Channel width for
                normalization. Required for SEQUENCE_ALIGNMENT.
                Defaults to 0.0.
            weights (dict, optional): Cost function weights for
                SEQUENCE_ALIGNMENT. Defaults to None (equal).
            gap_penalty_metric (str, optional): Gap penalty metric
                for SEQUENCE_ALIGNMENT. Defaults to "sinuosity".
            gap_penalty_scale (float, optional): Gap penalty
                multiplier. Defaults to 2.5.
            spatial_tolerance (float, optional): Max distance in
                norm_width units for split/merge detection.
                Defaults to 2.0.
            step_pattern (str, optional): DTW step pattern.
                Defaults to "asymmetric". Only for DTW.
            birth_death_threshold (float, optional): Cost
                above which DTW matches become births/deaths.
                Defaults to 10.0. Only for DTW.
            max_apex_distance_factor (float, optional): If
                apex distance > factor * norm_width, cost is
                inf. Defaults to 10.0. Only for DTW.

        Returns:
            bool: True if the function ends without errors

        Raises:
            NotImplementedError: if the method is SEQUENCE_ALIGNMENT, DTW or
                GENERALIZED_ALIGNMENT, which are not implemented yet.
            TypeError: if the method is unknown.

        """
        # reset self.bends_evol that will be updated later on
        self.bends_evol = []
        self.bends_tracking_computed = False
        for centerline in self.centerlines.values():
            for bend in centerline.bends:
                bend.bend_evol_id = None
        match method:
            case BendConnectionMethod.APEX:
                return self._connect_bends_apex(dmax, bend_evol_validity)
            case BendConnectionMethod.CENTROID:
                return self._connect_bends_centroid(dmax, bend_evol_validity)
            case BendConnectionMethod.MATCHING:
                return self._connect_bends_from_matching(
                    bend_evol_validity,
                    weighting_func_type=weighting_func_type,
                )
            case (
                BendConnectionMethod.SEQUENCE_ALIGNMENT
                | BendConnectionMethod.DTW
                | BendConnectionMethod.GENERALIZED_ALIGNMENT
            ):
                raise NotImplementedError(
                    f"Bend connection method '{method}' is not implemented "
                    + "in CenterlineCollection.connect_bends."
                )
            case _:
                methods = [str(meth) for meth in list(BendConnectionMethod)]  # type: ignore[unreachable]
                raise TypeError(
                    "Input method is wrong. Methods are: " + ", ".join(methods)
                )

    # TODO: refactor with same method as _connect_bends_from_matching
    def _connect_bends_apex(
        self: Self, dmax: float, bend_evol_validity: int
    ) -> bool:
        """Connect bends from the successive centerlines using apex distance.

        Connect bends from the successive centerlines of the collection by
        searching for the closests apex points that belongs to bends of same
        side.

        Args:
            dmax (float): Maximum allowed distance (m) between 2
                successive apex points.
            bend_evol_validity (int): A BendEvolution is considered valid
                if it contains strictly more than bend_evol_validity bends.

        Returns:
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
                    if bend.is_valid
                ]
                prev_key = key
                continue

            for _, bend in enumerate(self.centerlines[key].bends):
                if not bend.is_valid:
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
                        bend_saved[-1].is_valid
                        and bend_saved[-1].age == prev_key
                        and bend_saved[-1].side == bend.side
                    ):
                        # compute the distance between apex points
                        index_apex_prev: Optional[int] = bend_saved[
                            -1
                        ].index_apex
                        if index_apex_prev is None or index_apex_prev < 0:
                            continue
                        pt1: npt.NDArray[np.float64] = (
                            self.centerlines[prev_key]
                            .cl_points[index_apex_prev]
                            .pt
                        )

                        index_apex_cur: Optional[int] = bend.index_apex
                        if index_apex_cur is None or index_apex_cur < 0:
                            continue
                        pt2: npt.NDArray[np.float64] = (
                            self.centerlines[key].cl_points[index_apex_cur].pt
                        )
                        dist[k] = geom.distance(pt1, pt2)

                # take the index of the minimum distance if this distance is
                # lower than dmax
                if np.isfinite(dist).any() and np.nanmin(dist) < dmax:
                    index = int(np.nanargmin(dist))

                # a bend is found
                if index > -1:
                    bends_evol[index] += [bend]
                # no bend found
                else:
                    bends_evol += [[bend]]

            prev_key = key

        self._create_bend_evolutions_from_chains(
            bends_evol, bend_evol_validity
        )
        return True

    def _create_bend_evolutions_from_chains(
        self: Self, bends_evol: list[list[Bend]], bend_evol_validity: int
    ) -> None:
        """Create BendEvolution objects from chains of connected bends.

        Chains made of a single bend are discarded. BendEvolution ids are
        their index in self.bends_evol, and Bend.bend_evol_id is set on every
        bend of each kept chain.

        Args:
            bends_evol (list[list[Bend]]): Chains of connected bends (one bend
                per age).
            bend_evol_validity (int): A BendEvolution is considered valid
                if it contains strictly more than bend_evol_validity bends.

        """
        for bends in bends_evol:
            bend_indexes: dict[int, list[int]] = {
                bend.age: [
                    bend.id,
                ]
                for bend in bends
            }
            if len(bend_indexes) < 2:
                continue
            bend_evol_id: int = len(self.bends_evol)
            for bend in bends:
                self.centerlines[bend.age].bends[
                    bend.id
                ].bend_evol_id = bend_evol_id
            self.bends_evol += [
                BendEvolution(
                    bend_indexes,
                    bend_evol_id,
                    0,
                    len(bend_indexes) > bend_evol_validity,
                )
            ]

        self.bends_tracking_computed = len(self.bends_evol) > 0

    # TODO: refactor with same method as _connect_bends_from_matching
    def _connect_bends_centroid(
        self: Self, dmax: float, bend_evol_validity: int
    ) -> bool:
        """Connect bends from the successive centerlines using apex centroids.

        Connect bends from the successive centerlines of the collection by
        searching for the closests centroids points that belongs to bends of
        same side.

        Args:
            dmax (float): Maximum allowed distance (m) between 2
                successive apex points.
            bend_evol_validity (int): A BendEvolution is considered valid
                if it contains strictly more than bend_evol_validity bends.

        Returns:
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
                    if bend.is_valid
                ]
                prev_key = key
                continue

            for _, bend in enumerate(self.centerlines[key].bends):
                if not bend.is_valid:
                    continue

                # look for the closest apex
                dist = np.nan * np.zeros(len(bends_evol))
                index: int = -1
                for k, bend_saved in enumerate(bends_evol):
                    # if the last bend_saved was added at the previous key
                    # and is on the same side as bend
                    if (
                        bend_saved[-1].is_valid
                        and bend_saved[-1].age == prev_key
                        and bend_saved[-1].side == bend.side
                    ):
                        # compute the distance between upstream inflex points
                        # (more stable than apex)
                        assert bend_saved[-1].pt_centroid is not None, (
                            "Centroid is undefined"
                        )
                        assert bend.pt_centroid is not None, (
                            "Centroid is undefined"
                        )
                        dist[k] = geom.distance(
                            bend_saved[-1].pt_centroid[:2],
                            bend.pt_centroid[:2],
                        )

                # take the index of the minimum distance if this distance is
                # lower than dmax
                if np.isfinite(dist).any() and np.nanmin(dist) < dmax:
                    index = int(np.nanargmin(dist))

                # a bend is found
                if index > -1:
                    bends_evol[index] += [bend]
                # no bend found, create a new list of bends
                else:
                    bends_evol += [[bend]]

            prev_key = key

        self._create_bend_evolutions_from_chains(
            bends_evol, bend_evol_validity
        )
        return True

    def _connect_bends_from_matching(
        self: Self,
        bend_evol_validity: int,
        weighting_func_type: str = "uniform",
    ) -> bool:
        """Connect bends from the successive centerlines using matching result.

        Connect bends from the successive centerlines of the collection by
        searching for the greatest number of connected channel points between
        connected bends. A bend of age t is connected to a bend of age t+1 if
        more than 20% of the points of the bend of age t+1 are connected to
        points of the bend of age t. BendEvolution objects are then the
        connected components (made of at least 2 bends) of the graph of bend
        connections, such that bend splits and merges are kept in the same
        BendEvolution.

        Args:
            bend_evol_validity (int): A BendEvolution is considered valid
                if it contains strictly more than bend_evol_validity bends.
            weighting_func_type (str, optional): Weighting function type to
                use.

        Returns:
            bool: True if the function ends witout errors

        """
        assert self._centerline_matching_computed, (
            "Centerline matching was not "
            + "computed. First use match_centerlines method."
        )
        # TODO: compute weights for each point according to weighting function

        # reset bend connections from a previous call
        for centerline in self.centerlines.values():
            for bend in centerline.bends:
                bend.bend_uid_next = None
                bend.bend_uid_prev = None

        # connect bends forward through time
        all_iter: npt.NDArray[np.int64] = self.get_all_ages()
        for i, key in enumerate(all_iter[:-1]):
            next_key: int = all_iter[i + 1]
            for bend_index, bend in enumerate(self.centerlines[key].bends):
                # looks for the points of the next centerline connected to
                # the points of the bend, grouped by bend index
                connected_pts: dict[int, set[int]] = (
                    self._get_connected_cl_pts_per_bend(bend, key, next_key)
                )

                # keep all next bends with more than 20% of their cl_points
                # connected to the bend
                # TODO: to take into account weigths
                for next_bend_index, pt_indexes in connected_pts.items():
                    nb_pts = (
                        self.centerlines[next_key]
                        .bends[next_bend_index]
                        .get_nb_points()
                    )
                    ratio = float(len(pt_indexes)) / float(nb_pts)
                    if ratio > 0.2:
                        self._add_bend_connection(
                            key, next_key, bend_index, next_bend_index
                        )

        bend_uids: list[int] = [
            bend.uid
            for age in all_iter
            for bend in self.centerlines[age].bends
        ]
        self._create_bend_evolution(bend_evol_validity, bend_uids)
        return True

    def _add_bend_connection(
        self: Self,
        key: int,
        next_key: int,
        cur_bend_index: int,
        next_bend_index: int,
    ) -> int:
        """Register the reciprocal connection of bends of age key and next_key.

        Args:
            key (int): Age of centerline.
            next_key (int): Age of successive centerline.
            cur_bend_index (int): index of the bend of age 'key'.
            next_bend_index (int): index of the bend of age 'next_key'.

        Returns:
            int: bend unique id

        """
        bend: Bend = self.centerlines[key].bends[cur_bend_index]
        next_bend_uid = self.centerlines[next_key].bends[next_bend_index].uid
        bend.add_bend_connection_next(next_bend_uid)
        self.centerlines[next_key].bends[
            next_bend_index
        ].add_bend_connection_prev(bend.uid)
        return next_bend_uid

    def _get_connected_cl_pts_per_bend(
        self: Self,
        bend: Bend,
        key: int,
        next_key: int,
        next_iter: bool = True,
    ) -> dict[int, set[int]]:
        """Get the channel points connected to a bend grouped by bend index.

        Args:
            bend (Bend): Bend object.
            key (int): Age of the centerline the bend belongs to.
            next_key (int): Next age to consider. It can be lower or greater
                depending on next_iter.
            next_iter (bool): If True, look for next centerline, otherwise
                look for previous centerline.
                Defaults to True.

        Returns:
            dict[int, set[int]]: For each bend index of the centerline of age
            next_key, the set of distinct indexes of its channel points
            connected to the bend.

        """
        connected_pts: dict[int, set[int]] = {}
        # does not consider inflection points since they are part of both
        # neighboring bends
        for index in range(
            bend.index_inflex_up + 1, bend.index_inflex_down, 1
        ):
            cl_pt: ClPoint = self.centerlines[key].cl_points[index]
            pts_index_next: Optional[list[int]] = (
                cl_pt.cl_pt_index_next if next_iter else cl_pt.cl_pt_index_prev
            )
            if pts_index_next is None:
                continue

            for pt_index_next in pts_index_next:
                bend_index_next: Optional[int] = self.centerlines[
                    next_key
                ].get_bend_index_from_cl_pt_index(pt_index_next)
                if bend_index_next is None:
                    continue
                connected_pts.setdefault(bend_index_next, set()).add(
                    int(pt_index_next)
                )
        return connected_pts

    def _create_bend_evolution(
        self: Self, bend_evol_validity: int, bend_uids: list[int]
    ) -> None:
        """Create the list self.bends_evol with BendEvolution objects.

        BendEvolution objects are the connected components of the graph of
        bend connections (Bend.bend_uid_next and Bend.bend_uid_prev) that
        contain the input bends. Components made of a single bend are
        discarded. Bend.bend_evol_id is set on every bend of each
        BendEvolution.

        Args:
            bend_evol_validity (int): A BendEvolution is considered valid
                if it contains strictly more than bend_evol_validity bends.
            bend_uids (list[int]): list of bend uids used as seeds of the
                connected components.

        """
        self.bends_evol = []
        visited: set[int] = set()
        for seed_uid in bend_uids:
            if seed_uid in visited:
                continue
            component: list[int] = self._collect_connected_bends(
                seed_uid, visited
            )
            if len(component) < 2:
                continue

            dico_indexes: dict[int, list[int]] = {}
            for uid in component:
                age, bend_id = parse_bend_uid(uid)
                dico_indexes.setdefault(age, []).append(bend_id)
            dico_indexes = {
                age: sorted(set(ids))
                for age, ids in sorted(dico_indexes.items())
            }

            bend_evol_id: int = len(self.bends_evol)
            for age, ids in dico_indexes.items():
                for bend_id in ids:
                    self.centerlines[age].bends[
                        bend_id
                    ].bend_evol_id = bend_evol_id

            self.bends_evol += [
                BendEvolution(
                    dico_indexes,
                    bend_evol_id,
                    0,
                    len(component) > bend_evol_validity,
                )
            ]

        self.bends_tracking_computed = True

    def _collect_connected_bends(
        self: Self, seed_uid: int, visited: set[int]
    ) -> list[int]:
        """Collect the uids of the bends connected to the seed bend.

        Iterative depth-first traversal of the graph of bend connections
        defined by the lists Bend.bend_uid_next and Bend.bend_uid_prev.

        Args:
            seed_uid (int): Uid of the starting bend.
            visited (set[int]): Uids of already visited bends. It is updated
                with the uids of the collected bends.

        Returns:
            list[int]: Uids of the bends of the connected component that
            contains the seed bend, sorted in ascending order.

        """
        component: list[int] = []
        stack: list[int] = [seed_uid]
        visited.add(seed_uid)
        while stack:
            uid: int = stack.pop()
            component.append(uid)
            age, bend_id = parse_bend_uid(uid)
            bend: Bend = self.centerlines[age].bends[bend_id]
            neighbors: list[int] = []
            if bend.bend_uid_next is not None:
                neighbors += bend.bend_uid_next
            if bend.bend_uid_prev is not None:
                neighbors += bend.bend_uid_prev
            for neighbor_uid in neighbors:
                if neighbor_uid not in visited:
                    visited.add(neighbor_uid)
                    stack.append(neighbor_uid)
        return sorted(component)

    def set_section_lines(
        self: Self,
        pts_start: list[tuple[float, float]],
        pts_end: list[tuple[float, float]],
    ) -> None:
        """Manually set section lines across CenterlineCollection.

        Args:
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

        Args:
            point_name (CreateSectionMethod, optional): Name of the point to
                use to create the section.
                Defaults to CreateSectionMethod.MIDDLE.

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
                    "Unknown method for section line creation. Methods are "
                    + "either: "
                    + ", ".join(methods)
                )

    def _create_section_lines_from_bend(
        self: Self, point_name: CreateSectionMethod
    ) -> None:
        """Create section lines across BendEvolution.

        Automatically create section lines across BendEvolution objects such as
        lines goes by the middle or centroid point of the last bend of each
        BendEvolution.

        Args:
            point_name (CreateSectionMethod): Name of point to use.
                It can be either CreateSectionMethod.MIDDLE or
                CreateSectionMethod.CENTROID.

        """
        self.section_lines = []
        for i, bend in enumerate(
            self.centerlines[self.get_all_ages()[-1]].bends
        ):
            if (
                not bend.is_valid
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
                pt_end = bend.pt_center
            elif point_name == CreateSectionMethod.CENTROID:
                pt_end = bend.pt_centroid
            else:
                methods = [
                    str(CreateSectionMethod.MIDDLE),
                    str(CreateSectionMethod.CENTROID),
                ]
                raise TypeError(
                    "Unknown method for section line creation. Methods are "
                    + "either: "
                    + ", ".join(methods)
                )
            assert pt_end is not None, (
                "Undefined end point for section line creation"
            )
            assert (bend.index_apex is not None) and (bend.index_apex > -1), (
                "Bend apex is undefined."
            )
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
                not bend.is_valid
                or bend.index_apex is None
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

            if (
                prev_bend.is_valid
                and (prev_bend.index_apex is not None)
                and (prev_bend.index_apex > -1)
            ):
                pt0 = self.centerlines[key].cl_points[prev_bend.index_apex].pt
            else:
                k = prev_bend.index_inflex_up + int(
                    (prev_bend.get_nb_points() + 0.5) / 2.0
                )
                pt0 = self.centerlines[key].cl_points[k].pt

            if (
                next_bend.is_valid
                and (next_bend.index_apex is not None)
                and (next_bend.index_apex > -1)
            ):
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
        """Create channel cross-sections where centerlines cross the section.

        Each segment between two consecutive channel points of each
        centerline is tested against the section line. When an intersection
        exists, channel point properties are interpolated at the intersection
        point to create a ChannelCrossSection.

        Args:
            section_line (LineString): Section line.

        Returns:
            tuple[list[Isoline], list[int]]: List of channel cross-sections and
            list of the index of the first channel point of the intersected
            segment.

        """
        # list of isoline instances to store channel locations
        isolines: list[Isoline] = []
        cl_pt_indexes: list[int] = []

        # research window area defined by the square whose the section is a
        # diagonal
        line2 = affinity.rotate(section_line, 90)  # take the perpendicular
        window = Polygon(
            (
                np.array(section_line.coords)[0],
                np.array(line2.coords)[0],
                np.array(section_line.coords)[1],
                np.array(line2.coords)[1],
            )
        )
        xmin, ymin, xmax, ymax = window.bounds

        # for each centerline
        for key in self.get_all_ages():
            cl_points: list[ClPoint] = self.centerlines[key].cl_points
            nb_pts: int = len(cl_points)
            if nb_pts < 2:
                continue
            pts: npt.NDArray[np.float64] = np.array(
                [cl_pt.pt[:2] for cl_pt in cl_points], dtype=float
            )
            # fast pre-selection of segments whose bounding box overlaps the
            # window bounding box
            seg_min = np.minimum(pts[:-1], pts[1:])
            seg_max = np.maximum(pts[:-1], pts[1:])
            candidates = np.flatnonzero(
                (seg_max[:, 0] >= xmin)
                & (seg_min[:, 0] <= xmax)
                & (seg_max[:, 1] >= ymin)
                & (seg_min[:, 1] <= ymax)
            )
            # for each segment of the centerline
            for j in candidates:
                cl_pt: ClPoint = cl_points[j]
                cl_pt2: ClPoint = cl_points[j + 1]
                cl_line = LineString([pts[j], pts[j + 1]])
                # if the segment intersects the window
                if (cl_line.length == 0.0) or (not window.intersects(cl_line)):
                    continue
                intersect = section_line.intersection(cl_line)
                # if the intersection exists
                if intersect.is_empty:
                    continue
                # interpolate channel points properties to the
                # intersection point
                d: float = intersect.distance(Point(pts[j])) / cl_line.length
                # intersection at the end point of the segment is handled
                # with the next segment to avoid duplicates
                if (d >= 1.0) and (j < nb_pts - 2):
                    continue
                cl_pt_inter: ClPoint = cl_pt * (1 - d) + cl_pt2 * d

                channel_section: ChannelCrossSection = ChannelCrossSection(
                    key, cl_pt_inter
                )
                channel_section.complete_channel_shape(11)
                isolines += [cast(Isoline, channel_section)]
                cl_pt_indexes += [int(j)]
        return isolines, cl_pt_indexes

    # done here because may collect centerline points outside bend_evol
    def find_points_on_sections(
        self: Self,
        thres: int = 1,
        flow_dir: npt.NDArray[np.float64] = np.array([1.0, 0]),
        cl_collec_id: int = 0,
    ) -> bool:
        """Collect channels that are intersected by sections.

        This method create the Section objects with intersected channel
        geometry and store them in the list self.sections. The bend id of
        each Section is the index of the bend intersected by the section in
        the youngest (i.e., most recent, greatest age) intersected
        centerline, or -1 if this centerline has no bend.

        Args:
            thres (int, optional): A Section object is created only if the
                section line intersects strictly more than thres channels.
                Defaults to 1.
            flow_dir (NDArray[float], optional): Direction vector of the flow
                used to orientate the Sections.
                Defaults to np.array([1,0]).
            cl_collec_id (int, optional): CenterlineCollection id used in the
                Section id.
                Defaults to 0.

        Returns:
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
                # bend index in the youngest intersected centerline
                ref_bend_index: Optional[int] = None
                ref_age: Optional[int] = None
                for isoline, cl_pt_index in zip(
                    isolines, cl_pt_indexes, strict=False
                ):
                    # notify bend that is intersected by the section line
                    bend_index: Optional[int] = (
                        self._get_bend_index_from_cl_point_index(
                            cl_pt_index, isoline.age
                        )
                    )
                    if bend_index is not None:
                        self.centerlines[isoline.age].bends[
                            bend_index
                        ].add_intersected_section_index(i)
                    if (ref_age is None) or (isoline.age > ref_age):
                        ref_age = isoline.age
                        ref_bend_index = bend_index

                bend_id: str = (
                    f"{cl_collec_id}-"
                    + f"{-1 if ref_bend_index is None else ref_bend_index}"
                )
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
    ) -> Optional[int]:
        """Get bend index from the index of a channel point of age.

        Args:
            cl_pt_index (int): Index of the channel point.
            age (float): Age of the Centerline where the channel point
                belongs to.

        Returns:
            int | None: Index of the Bend the channel point belongs to, or
            None if the centerline has no bend.

        """
        bends: list[Bend] = self.centerlines[age].bends
        if len(bends) == 0:
            return None
        if cl_pt_index < bends[0].index_inflex_up:
            return 0
        elif cl_pt_index > bends[-1].index_inflex_down:
            return len(bends) - 1

        for bend_index, bend in enumerate(bends):
            if (cl_pt_index >= bend.index_inflex_up) and (
                cl_pt_index < bend.index_inflex_down
            ):
                return bend_index
        return len(bends) - 1

    def find_all_bend_middle(
        self: Self,
        smooth_trajectory: bool = False,
        smoothing_factor: float = 0.5,
    ) -> None:
        """Compute the middle points of all bends of the CenterlineCollection.

        Args:
            smooth_trajectory (bool, optional): If True, middle trajectory
                through BendEvolution is smoothed and stored in the list
                BendEvolution.middle_trajec_smooth.
                Defaults to False.
            smoothing_factor (float, optional): Smoothing factor s (m²) of the
                spline used to smooth the trajectory, i.e., the upper bound of
                the sum of squared residuals between input and smoothed
                points. 0 means interpolation through all points.
                Defaults to 0.5.

        """
        for age in self.get_all_ages():
            self.centerlines[age].compute_all_bend_center()

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
                        bend_evol_index, smoothing_factor
                    )
            else:
                with Pool(max_workers=get_nb_procs()) as pool:
                    inputs = range(self.get_nb_bends_evol())
                    outputs = pool.map(
                        functools.partial(
                            self._smooth_bend_middle_trajec,
                            smoothing_factor=smoothing_factor,
                        ),
                        inputs,
                    )

                    for bend_evol_index, trajec in zip(
                        inputs, outputs, strict=False
                    ):
                        self.bends_evol[
                            bend_evol_index
                        ].middle_trajec_smooth = trajec

    @staticmethod
    def _smooth_trajectory(
        ages: npt.NDArray[np.int64],
        lx: npt.NDArray[np.float64],
        ly: npt.NDArray[np.float64],
        smoothing_factor: float,
    ) -> list[npt.NDArray[np.float64]]:
        """Smooth a point trajectory through time with a smoothing spline.

        The spline is parametrized by the ages, and evaluated at the input
        ages such as the output point i corresponds to the age i. Points
        with undefined (nan) coordinates are ignored for the smoothing and
        remain undefined in the output. If less than 3 points are defined,
        no smoothing is applied.

        Args:
            ages (npt.NDArray[np.int64]): Strictly increasing ages.
            lx (npt.NDArray[np.float64]): x coordinates at each age.
            ly (npt.NDArray[np.float64]): y coordinates at each age.
            smoothing_factor (float): Smoothing factor s (m²) of the spline,
                i.e., the upper bound of the sum of squared residuals.

        Returns:
            list[npt.NDArray[np.float64]]: Smoothed point coordinates at each
            age.

        """
        u: npt.NDArray[np.float64] = np.asarray(ages, dtype=float)
        x: npt.NDArray[np.float64] = np.asarray(lx, dtype=float)
        y: npt.NDArray[np.float64] = np.asarray(ly, dtype=float)
        coords: npt.NDArray[np.float64] = np.full((u.size, 2), np.nan)
        mask = np.isfinite(x) & np.isfinite(y)
        if np.count_nonzero(mask) < 3:
            coords[mask, 0] = x[mask]
            coords[mask, 1] = y[mask]
        else:
            xy_new: npt.NDArray[np.float64] = cpf.resample_path(
                x[mask], y[mask], 0, smoothing_factor, u=u[mask]
            )
            coords[mask] = xy_new.T
        return [np.array(coord) for coord in coords]

    def _smooth_bend_middle_trajec(
        self: Self,
        bend_evol_index: int,
        smoothing_factor: float = 0.5,
    ) -> list[npt.NDArray[np.float64]]:
        """Smooth middle point trajectory of one bend evolution.

        Args:
            bend_evol_index (int): bend evolution index
            smoothing_factor (float, optional): Smoothing factor s (m²) of the
                spline, i.e., the upper bound of the sum of squared residuals.
                Defaults to 0.5.

        Returns:
            list[npt.NDArray[np.float64]]: Smoothed middle point coordinates
            at each age of the bend evolution.

        """
        bend_evol: BendEvolution = self.bends_evol[bend_evol_index]
        ages: npt.NDArray[np.int64] = bend_evol.get_all_ages()
        lx: list[float] = []
        ly: list[float] = []
        for age in ages:
            bend_id: int = bend_evol.bend_indexes[age][0]
            pt_center: Optional[npt.NDArray[np.float64]] = (
                self.centerlines[age].bends[bend_id].pt_center
            )
            assert pt_center is not None, "Middle point is undefined"
            lx += [pt_center[0]]
            ly += [pt_center[1]]

        return self._smooth_trajectory(
            ages, np.array(lx), np.array(ly), smoothing_factor
        )

    def find_all_bend_centroid(
        self: Self,
        smooth_trajectory: bool = False,
        smoothing_factor: float = 0.5,
    ) -> None:
        """Compute the centroid point of each bend of the CenterlineCollection.

        Args:
            smooth_trajectory (bool, optional): If True, centroid trajectory
                through BendEvolution is smoothed and stored in the list
                BendEvolution.centroid_trajec_smooth.
                Defaults to False.
            smoothing_factor (float, optional): Smoothing factor s (m²) of the
                spline used to smooth the trajectory, i.e., the upper bound of
                the sum of squared residuals between input and smoothed
                points. 0 means interpolation through all points.
                Defaults to 0.5.

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
                        self._smooth_bend_centroid_trajec(
                            bend_evol_index, smoothing_factor
                        )
                    )
            else:
                with Pool(max_workers=get_nb_procs()) as pool:
                    inputs = range(self.get_nb_bends_evol())
                    outputs = pool.map(
                        functools.partial(
                            self._smooth_bend_centroid_trajec,
                            smoothing_factor=smoothing_factor,
                        ),
                        inputs,
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
        smoothing_factor: float = 0.5,
    ) -> list[npt.NDArray[np.float64]]:
        """Smooth centroid trajectory of one bend evolution.

        Args:
            bend_evol_index (int): bend evolution index
            smoothing_factor (float, optional): Smoothing factor s (m²) of the
                spline, i.e., the upper bound of the sum of squared residuals.
                Defaults to 0.5.

        Returns:
            list[npt.NDArray[np.float64]]: Smoothed centroid coordinates at
            each age of the bend evolution.

        """
        bend_evol: BendEvolution = self.bends_evol[bend_evol_index]
        ages: npt.NDArray[np.int64] = bend_evol.get_all_ages()
        lx: list[float] = []
        ly: list[float] = []

        for age in ages:
            bend_id: int = bend_evol.bend_indexes[age][0]
            pt_centroid: Optional[npt.NDArray[np.float64]] = (
                self.centerlines[age].bends[bend_id].pt_centroid
            )
            assert pt_centroid is not None, "Centroid point is undefined"
            lx += [pt_centroid[0]]
            ly += [pt_centroid[1]]

        return self._smooth_trajectory(
            ages, np.array(lx), np.array(ly), smoothing_factor
        )

    def find_all_bend_apex_user_weights(
        self: Self,
        apex_proba_weights: tuple[float, float, float],
        smooth_trajectory: bool = False,
        smoothing_factor: float = 0.5,
    ) -> None:
        """Find the apex of all bends of the CenterlineCollection.

        Args:
            apex_proba_weights (tuple[float, float, float]): Weights for apex
                probability calculation. Apex probability depends on channel
                point curvature, bend amplitude (m), and the distance from
                inflection points.
            smooth_trajectory (bool, optional): If True, apex trajectory
                through BendEvolution is smoothed and stored in the list
                BendEvolution.apex_trajec_smooth. Otherwise, raw apex
                locations are stored in this list.
                Defaults to False.
            smoothing_factor (float, optional): Smoothing factor s (m²) of the
                spline used to smooth the trajectory, i.e., the upper bound of
                the sum of squared residuals between input and smoothed
                points. 0 means interpolation through all points.
                Defaults to 0.5.

        """
        for age in self.get_all_ages():
            self.centerlines[age].find_all_bend_apex_user_weights(
                apex_proba_weights
            )

        if smooth_trajectory:
            self._smooth_bend_apex_trajec(smoothing_factor)
        else:
            for bend_evol_index in range(self.get_nb_bends_evol()):
                self.bends_evol[
                    bend_evol_index
                ].apex_trajec_smooth = self._get_bend_evol_apex_points(
                    bend_evol_index
                )

    def find_all_bend_apex(
        self: Self,
        n: float,
        smooth_trajectory: bool = False,
        smoothing_factor: float = 0.5,
    ) -> None:
        """Find the apex of all bends of the CenterlineCollection.

        Args:
            n (float): exponent of the curvature distribution function.
            smooth_trajectory (bool, optional): If True, apex trajectory
                through BendEvolution is smoothed and stored in the list
                BendEvolution.apex_trajec_smooth. Otherwise, raw apex
                locations are stored in this list.
                Defaults to False.
            smoothing_factor (float, optional): Smoothing factor s (m²) of the
                spline used to smooth the trajectory, i.e., the upper bound of
                the sum of squared residuals between input and smoothed
                points. 0 means interpolation through all points.
                Defaults to 0.5.

        """
        for age in self.get_all_ages():
            self.centerlines[age].find_all_bend_apex(n)

        if smooth_trajectory:
            self._smooth_bend_apex_trajec(smoothing_factor)
        else:
            for bend_evol_index in range(self.get_nb_bends_evol()):
                self.bends_evol[
                    bend_evol_index
                ].apex_trajec_smooth = self._get_bend_evol_apex_points(
                    bend_evol_index
                )

    def _smooth_bend_apex_trajec(
        self: Self, smoothing_factor: float = 0.5
    ) -> None:
        """Smooth apex trajectory of all bend evolutions.

        Args:
            smoothing_factor (float, optional): Smoothing factor s (m²) of the
                spline, i.e., the upper bound of the sum of squared residuals.
                Defaults to 0.5.

        """
        assert self.bends_tracking_computed, (
            "Bends were not tracked over " + "time. First apply connect_bends."
        )

        if get_nb_procs() == 1:
            for bend_evol_index in range(self.get_nb_bends_evol()):
                self.bends_evol[
                    bend_evol_index
                ].apex_trajec_smooth = self._do_smooth_bend_apex_trajec(
                    bend_evol_index, smoothing_factor
                )
        else:
            with Pool(max_workers=get_nb_procs()) as pool:
                inputs = range(self.get_nb_bends_evol())
                outputs = pool.map(
                    functools.partial(
                        self._do_smooth_bend_apex_trajec,
                        smoothing_factor=smoothing_factor,
                    ),
                    inputs,
                )

                for bend_evol_index, trajec in zip(
                    inputs, outputs, strict=False
                ):
                    self.bends_evol[
                        bend_evol_index
                    ].apex_trajec_smooth = trajec

    def _do_smooth_bend_apex_trajec(
        self: Self,
        bend_evol_index: int,
        smoothing_factor: float = 0.5,
    ) -> list[npt.NDArray[np.float64]]:
        """Smooth apex trajectory of one bend evolution.

        Ages where the bend has no apex are ignored for the smoothing and the
        corresponding output coordinates are nan.

        Args:
            bend_evol_index (int): bend evolution index
            smoothing_factor (float, optional): Smoothing factor s (m²) of the
                spline, i.e., the upper bound of the sum of squared residuals.
                Defaults to 0.5.

        Returns:
            list[npt.NDArray[np.float64]]: Smooth apex trajectory coordinates
            at each age of the bend evolution.

        """
        bend_evol: BendEvolution = self.bends_evol[bend_evol_index]
        apex_coords: list[npt.NDArray[np.float64]] = (
            self._get_bend_evol_apex_points(bend_evol_index)
        )
        lx: npt.NDArray[np.float64] = np.array(
            [coord[0] for coord in apex_coords], dtype=float
        )
        ly: npt.NDArray[np.float64] = np.array(
            [coord[1] for coord in apex_coords], dtype=float
        )
        return self._smooth_trajectory(
            bend_evol.get_all_ages(), lx, ly, smoothing_factor
        )

    def _get_bend_evol_apex_points(
        self: Self, bend_evol_index: int
    ) -> list[npt.NDArray[np.float64]]:
        """Get the coordinates of all bend apex of the bend evolution.

        Args:
            bend_evol_index (int): bend evolution index

        Returns:
            list[npt.NDArray[np.float64]]: apex coordinates at each age of the
            bend evolution (nan if the bend has no apex).

        """
        bend_evol: BendEvolution = self.bends_evol[bend_evol_index]
        ages: npt.NDArray[np.int64] = bend_evol.get_all_ages()
        apex_coords: npt.NDArray[np.float64] = np.full((len(ages), 2), np.nan)
        for i, age in enumerate(ages):
            bend_ids: list[int] = bend_evol.bend_indexes[age]
            index_apex: Optional[int] = (
                self.centerlines[age].bends[bend_ids[0]].index_apex
            )
            if index_apex is None or index_apex < 0:
                continue
            apex_coords[i] = self.centerlines[age].cl_points[index_apex].pt[:2]
        return [
            np.array([x, y])
            for x, y in zip(apex_coords[:, 0], apex_coords[:, 1], strict=False)
        ]
