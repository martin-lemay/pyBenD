# SPDX-FileCopyrightText: Copyright 2025 Martin Lemay <martin.lemay@mines-paris.org>
# SPDX-FileContributor: Martin Lemay
# ruff: noqa: E402 # disable Module level import not at top of file

"""Bend morphometry computations.

Bends are defined as the channel path comprised between 2 consecutive
inflection points (o). A Bend contains a maximum curvature point (+) and an
apex (x) whose definition may vary.

Metrics include:

* arc length: curvilinear distance between inflection points
* wavelength (W): euclidean distance between inflection points
* amplitude (Am): orthogonal distance between bend apex and chord
* extension (Ex): distance between bend apex and center
* asymmetry coefficient: A=(Lup-Ldown) / L, where Lup and Ldown are arc length
  distances between bend apex and upstream and downstream inflection point
  respectively, and L is the bend arc length.
* maximum amplitude: maximum orthogonal distance between any bend point
  and the chord
* maximum extension: maximum distance between any bend point and the chord
  center
* maturity index: ratio of maximum amplitude to chord length (wavelength)
* radius of curvature: inverse of bend apex curvature
* roundess: ratio of maximum to mean curvature along the bend

.. code-block:: bash

                                 .   x  .
                             .      /|    .
                          .        / |     +
        Flow            .      Ex /  | Am  .
        -->            .         /   |    .
    Direction         .         /    |   .
                     o --------c------- o
                     <---------------->
                            W


To use it:

.. code-block:: python

    centerline :Centerline
    morphComputer :Morphometry = Morphometry(centerline)
    metrics :pd.DataFrame = morphComputer.compute_bend_morphometry()
"""

from typing import Optional, Self

import numpy as np
import numpy.typing as npt
import pandas as pd  # type: ignore[import-untyped]

import pybend.algorithms.centerline_process_functions as cpf
import pybend.algorithms.geometry_functions as geom
from pybend.model.Bend import Bend
from pybend.model.Centerline import Centerline, ClPoint
from pybend.model.enumerations import MorphometricNames, PropertyNames


class Morphometry:
    """Compute morphometric parameters from a `Centerline`."""

    def __init__(
        self: Self,
        centerline: Centerline,
    ) -> None:
        """Class to compute morphometric parameters from Centerline object.

        Args:
            centerline (Centerline): Centerline object

        """
        self.centerline: Centerline = centerline

        self._bend_metrics: tuple[str, ...] = (
            MorphometricNames.ARC_LENGTH.value,
            MorphometricNames.WAVELENGTH.value,
            MorphometricNames.SINUOSITY.value,
            MorphometricNames.AMPLITUDE.value,
            MorphometricNames.EXTENSION.value,
            MorphometricNames.RADIUS_CURVATURE.value,
            MorphometricNames.ASYMMETRY.value,
            MorphometricNames.ASYMMETRY_CENTROID.value,
            MorphometricNames.ROUNDNESS.value,
            MorphometricNames.SKEWNESS.value,
            MorphometricNames.MAX_AMPLITUDE.value,
            MorphometricNames.MAX_EXTENSION.value,
            MorphometricNames.MATURITY_INDEX.value,
            MorphometricNames.WAVELENGTH_LEOPOLD.value,
            MorphometricNames.AMPLITUDE_LEOPOLD.value,
        )

    @staticmethod
    def _has_valid_apex(bend: Bend) -> bool:
        """Check whether a bend has a valid apex index.

        An apex index equal to None or negative (-1) means "no apex".

        Args:
            bend (Bend): bend to check

        Returns:
            bool: True if the bend apex index is defined and lies between
            bend inflection points.
        """
        if bend.index_apex is None or bend.index_apex < 0:
            return False
        return (
            bend.index_apex >= bend.index_inflex_up
            and bend.index_apex <= bend.index_inflex_down
        )

    def _get_pt_2d(self: Self, index: int) -> npt.NDArray[np.float64]:
        """Get the horizontal coordinates of the channel point at index.

        Planform metrics are computed in the horizontal plane only,
        consistently with the curvilinear abscissa.

        Args:
            index (int): channel point index along the centerline

        Returns:
            npt.NDArray[np.float64]: (x, y) coordinates
        """
        return np.asarray(self.centerline.cl_points[index].pt[:2], dtype=float)

    def compute_bends_morphometry(
        self: Self, valid_bends: bool = True
    ) -> pd.DataFrame:
        """Compute all bend morphometric parameters.

        Args:
            valid_bends (bool): if True, compute morphometry on valid bends
                only
                Defaults to True.

        Returns:
            pd.DataFrame: dataframe with morphometric measurements. The
            DataFrame index (named "Bend_id") contains the id of the bend
            associated to each row.
        """
        n_rows: int = (
            self.centerline.get_nb_valid_bends()
            if valid_bends
            else self.centerline.get_nb_bends()
        )
        bend_ids: list[int] = [
            bend.id
            for bend in self.centerline.bends
            if not (valid_bends and not bend.is_valid)
        ]
        assert len(bend_ids) == n_rows, "Inconsistent number of bends."
        data: pd.DataFrame = pd.DataFrame(
            np.full((n_rows, len(self._bend_metrics)), np.nan),
            columns=self._bend_metrics,
            index=pd.Index(bend_ids, name="Bend_id"),
        )
        for bend in self.centerline.bends:
            if valid_bends and not bend.is_valid:
                continue
            i = bend.id
            data.loc[i, MorphometricNames.ARC_LENGTH.value] = (
                self.compute_bend_arc_length(bend.id)
            )
            data.loc[i, MorphometricNames.WAVELENGTH.value] = (
                self.compute_bend_wavelength(bend.id)
            )
            # sinuosity
            if data.loc[i, MorphometricNames.WAVELENGTH.value] > 0:  # type: ignore
                data.loc[i, MorphometricNames.SINUOSITY.value] = np.round(
                    data.loc[i, MorphometricNames.ARC_LENGTH.value]  # type: ignore
                    / data.loc[i, MorphometricNames.WAVELENGTH.value],  # type: ignore
                    4,
                )  # type: ignore
            data.loc[i, MorphometricNames.AMPLITUDE.value] = (
                self.compute_bend_amplitude(bend.id)
            )
            data.loc[i, MorphometricNames.EXTENSION.value] = (
                self.compute_bend_extension(bend.id)
            )
            data.loc[i, MorphometricNames.RADIUS_CURVATURE.value] = (
                self.compute_bend_radius(bend.id)
            )
            data.loc[i, MorphometricNames.ASYMMETRY.value] = (
                self.compute_bend_asymmetry_apex(bend.id)
            )
            data.loc[i, MorphometricNames.ASYMMETRY_CENTROID.value] = (
                self.compute_bend_asymmetry_centroid(bend.id)
            )
            data.loc[i, MorphometricNames.ROUNDNESS.value] = (
                self.compute_bend_roundness(bend.id)
            )
            data.loc[i, MorphometricNames.SKEWNESS.value] = (
                self.compute_bend_skewness(bend.id)
            )
            data.loc[i, MorphometricNames.MAX_AMPLITUDE.value] = (
                self.compute_bend_max_amplitude(bend.id)
            )
            data.loc[i, MorphometricNames.MAX_EXTENSION.value] = (
                self.compute_bend_max_extension(bend.id)
            )
            data.loc[i, MorphometricNames.MATURITY_INDEX.value] = (
                self.compute_bend_maturity_index(bend.id)
            )
            if (bend.id > 0) and (
                bend.id < self.centerline.get_nb_bends() - 1
            ):
                data.loc[i, MorphometricNames.WAVELENGTH_LEOPOLD.value] = (
                    self.compute_bend_wavelength_leopold(bend.id)
                )
                data.loc[i, MorphometricNames.AMPLITUDE_LEOPOLD.value] = (
                    self.compute_bend_amplitude_leopold(bend.id)
                )
        return data

    def compute_bend_sinuosity(self: Self, bend_id: int) -> float:
        """Compute bend sinuosity.

        Args:
            bend_id (int): bend index

        Returns:
            float: bend sinuosity
        """
        assert (bend_id > -1) and (bend_id < self.centerline.get_nb_bends()), (
            "Bend index is undefined."
        )

        den: float = self.compute_bend_wavelength(bend_id)
        sinuo: float = np.nan
        if den > 0:
            num: float = self.compute_bend_arc_length(bend_id)
            sinuo = num / den
        return round(sinuo, 4)

    def compute_bend_wavelength(self: Self, bend_id: int) -> float:
        """Compute bend wavelength.

        Args:
            bend_id (int): bend index

        Returns:
            float: bend wavelength
        """
        assert (bend_id > -1) and (bend_id < self.centerline.get_nb_bends()), (
            "Bend index is undefined."
        )
        bend: Bend = self.centerline.bends[bend_id]
        pt_inflex_up: npt.NDArray[np.float64] = self._get_pt_2d(
            bend.index_inflex_up
        )
        pt_inflex_down: npt.NDArray[np.float64] = self._get_pt_2d(
            bend.index_inflex_down
        )
        return geom.distance(pt_inflex_up, pt_inflex_down)

    def compute_bend_amplitude(self: Self, bend_id: int) -> float:
        """Compute bend amplitude.

        Args:
            bend_id (int): bend index

        Returns:
            float: bend amplitude
        """
        assert (bend_id > -1) and (bend_id < self.centerline.get_nb_bends()), (
            "Bend index is undefined."
        )
        bend: Bend = self.centerline.bends[bend_id]
        if not self._has_valid_apex(bend):
            return np.nan
        pt_apex: npt.NDArray[np.float64] = self._get_pt_2d(bend.index_apex)
        pt_inflex_up: npt.NDArray[np.float64] = self._get_pt_2d(
            bend.index_inflex_up
        )
        pt_inflex_down: npt.NDArray[np.float64] = self._get_pt_2d(
            bend.index_inflex_down
        )
        return geom.orthogonal_distance(pt_apex, pt_inflex_up, pt_inflex_down)

    def compute_bend_extension(self: Self, bend_id: int) -> float:
        """Compute bend extension.

        Args:
            bend_id (int): bend index

        Returns:
            float: bend extension
        """
        assert (bend_id > -1) and (bend_id < self.centerline.get_nb_bends()), (
            "Bend index is undefined."
        )
        bend: Bend = self.centerline.bends[bend_id]
        if not self._has_valid_apex(bend):
            return np.nan
        pt_apex: npt.NDArray[np.float64] = self._get_pt_2d(bend.index_apex)
        pt_center: Optional[npt.NDArray[np.float64]] = bend.pt_center
        if pt_center is not None:
            return geom.distance(pt_apex, np.asarray(pt_center)[:2])
        return np.nan

    def compute_bend_max_amplitude(self: Self, bend_id: int) -> float:
        """Compute bend maximum amplitude.

        Maximum orthogonal distance from any bend point to the chord
        (line between inflection points).

        Args:
            bend_id (int): bend index

        Returns:
            float: bend maximum amplitude
        """
        assert (bend_id > -1) and (bend_id < self.centerline.get_nb_bends()), (
            "Bend index is undefined."
        )
        bend: Bend = self.centerline.bends[bend_id]
        pt_inflex_up: npt.NDArray[np.float64] = self._get_pt_2d(
            bend.index_inflex_up
        )
        pt_inflex_down: npt.NDArray[np.float64] = self._get_pt_2d(
            bend.index_inflex_down
        )
        max_dist: float = 0.0
        for cl_pt in self.centerline.cl_points[
            bend.index_inflex_up : bend.index_inflex_down + 1
        ]:
            d = geom.orthogonal_distance(
                cl_pt.pt[:2], pt_inflex_up, pt_inflex_down
            )
            if d > max_dist:
                max_dist = d
        return round(max_dist, 4)

    def compute_bend_max_extension(self: Self, bend_id: int) -> float:
        """Compute bend maximum extension.

        Maximum distance from any bend point to the chord center.

        Args:
            bend_id (int): bend index

        Returns:
            float: bend maximum extension
        """
        assert (bend_id > -1) and (bend_id < self.centerline.get_nb_bends()), (
            "Bend index is undefined."
        )
        bend: Bend = self.centerline.bends[bend_id]
        pt_center: Optional[npt.NDArray[np.float64]] = bend.pt_center
        if pt_center is None:
            return np.nan
        max_dist: float = 0.0
        for cl_pt in self.centerline.cl_points[
            bend.index_inflex_up : bend.index_inflex_down + 1
        ]:
            d = geom.distance(cl_pt.pt[:2], np.asarray(pt_center)[:2])
            if d > max_dist:
                max_dist = d
        return round(max_dist, 4)

    def compute_bend_maturity_index(self: Self, bend_id: int) -> float:
        """Compute bend maturity index.

        Ratio of maximum amplitude to chord length (wavelength).

        Args:
            bend_id (int): bend index

        Returns:
            float: bend maturity index
        """
        max_amp: float = self.compute_bend_max_amplitude(bend_id)
        wavelength: float = self.compute_bend_wavelength(bend_id)
        if wavelength > 0:
            return round(max_amp / wavelength, 4)
        return np.nan

    def compute_bend_radius(self: Self, bend_id: int) -> float:
        """Compute bend radius.

        Args:
            bend_id (int): bend index

        Returns:
            float: bend radius
        """
        assert (bend_id > -1) and (bend_id < self.centerline.get_nb_bends()), (
            "Bend index is undefined."
        )
        bend: Bend = self.centerline.bends[bend_id]
        if not self._has_valid_apex(bend):
            return np.nan
        curvature: npt.NDArray[np.float64] = np.abs(
            self.centerline.get_bend_curvature_filtered(bend_id)
        )
        curv: float = float(curvature[bend.index_apex - bend.index_inflex_up])
        if curv > 0:
            return round(1.0 / curv, 4)
        return np.nan

    def compute_bend_arc_length(self: Self, bend_id: int) -> float:
        """Compute bend arc length.

        Args:
            bend_id (int): bend index

        Returns:
            float: bend arc length
        """
        assert (bend_id > -1) and (bend_id < self.centerline.get_nb_bends()), (
            "Bend index is undefined."
        )
        curv_abs: npt.NDArray[np.float64] = self.centerline.get_bend_property(
            bend_id, PropertyNames.CURVILINEAR_ABSCISSA.value
        )
        return float(round(curv_abs[-1] - curv_abs[0], 4))

    def compute_bend_roundness(self: Self, bend_id: int) -> float:
        """Compute bend roundness.

        Roundness coefficient from `Schwenk et al. (2015) <https://doi.org/10.1002/2014JF003252)>`_

        Args:
            bend_id (int): bend index

        Returns:
            float: bend roundness, or NaN if the mean curvature is null
            (straight bend) or undefined.
        """
        assert (bend_id > -1) and (bend_id < self.centerline.get_nb_bends()), (
            "Bend index is undefined."
        )
        curvature: npt.NDArray[np.float64] = np.abs(
            self.centerline.get_bend_curvature_filtered(bend_id)
        )
        if curvature.size == 0 or not np.all(np.isfinite(curvature)):
            return np.nan
        mean_curv: float = float(np.mean(curvature))
        # straight bend: roundness is undefined
        if mean_curv == 0.0:
            return np.nan
        return float(round(np.max(curvature) / mean_curv, 4))

    def compute_bend_asymmetry_apex(self: Self, bend_id: int) -> float:
        """Compute bend asymmetry coefficient.

        Asymmetry coefficient from `Howard and Hemberger (1991) <https://doi.org/10.1016/0169-555X(91)90002-R>`_

        Args:
            bend_id (int): bend index

        Returns:
            float: bend asymmetry coefficient
        """
        assert (bend_id > -1) and (bend_id < self.centerline.get_nb_bends()), (
            "Bend index is undefined."
        )
        bend: Bend = self.centerline.bends[bend_id]
        if not self._has_valid_apex(bend):
            return np.nan
        curv_abs: npt.NDArray[np.float64] = self.centerline.get_bend_property(
            bend_id, PropertyNames.CURVILINEAR_ABSCISSA.value
        )

        arc_length_tot = curv_abs[-1] - curv_abs[0]
        arc_length1 = (
            curv_abs[bend.index_apex - bend.index_inflex_up] - curv_abs[0]
        )
        arc_length2 = (
            curv_abs[-1] - curv_abs[bend.index_apex - bend.index_inflex_up]
        )
        if arc_length_tot > 0:
            return float(
                round((arc_length1 - arc_length2) / arc_length_tot, 4)
            )
        return np.nan

    def compute_bend_asymmetry_centroid(self: Self, bend_id: int) -> float:
        """Compute bend asymmetry from loop-area centroid.

        The centroid of the polygon enclosed between the bend arc
        and the chord is projected onto the chord. The asymmetry
        is ``2 * t_c / L - 1`` where *t_c* is the distance from
        the upstream inflection point to the projection and *L*
        the chord length.

        Args:
            bend_id (int): bend index

        Returns:
            float: centroid asymmetry coefficient
        """
        assert (bend_id > -1) and (bend_id < self.centerline.get_nb_bends()), (
            "Bend index is undefined."
        )
        bend: Bend = self.centerline.bends[bend_id]
        coords: npt.NDArray[np.float64] = np.array(
            [
                cl_pt.pt[:2]
                for cl_pt in self.centerline.cl_points[
                    bend.index_inflex_up : bend.index_inflex_down + 1
                ]
            ]
        )
        if coords.shape[0] < 3:
            return np.nan
        return round(cpf.compute_loop_area_centroid_asymmetry(coords), 4)

    def compute_bend_skewness(self: Self, bend_id: int) -> float:
        """Compute bend skewness coefficient.

        Skewness coefficient corresponds to the Pearson's skewness coefficient
        of the curvature satial distribution.

        Args:
            bend_id (int): bend index

        Returns:
            float: bend skewness coefficient, or NaN if it is undefined
            (bend with less than 3 points, null curvature, or null standard
            deviation).
        """
        assert (bend_id > -1) and (bend_id < self.centerline.get_nb_bends()), (
            "Bend index is undefined."
        )
        bend: Bend = self.centerline.bends[bend_id]

        curv_abs: npt.NDArray[np.float64] = self.centerline.get_bend_property(
            bend_id, PropertyNames.CURVILINEAR_ABSCISSA.value
        )
        curvature = np.abs(
            self.centerline.get_bend_property(
                bend.id, PropertyNames.CURVATURE_FILTERED.value
            )
        )
        if (
            curv_abs.size < 3
            or not np.all(np.isfinite(curv_abs))
            or not np.all(np.isfinite(curvature))
        ):
            return np.nan
        abs_range: float = float(curv_abs.max() - curv_abs.min())
        if abs_range == 0.0 or float(np.sum(curvature)) == 0.0:
            return np.nan
        curv_abs_norm = (curv_abs - curv_abs.min()) / abs_range

        mean_abs = cpf.compute_esperance(curvature, curv_abs_norm, 1)
        std_dev = cpf.compute_variance(curvature, curv_abs_norm, 1)[1]
        # normalized abscissa lies in [0, 1]: use an absolute tolerance
        if not np.isfinite(std_dev) or std_dev < 1e-12:
            return np.nan
        apex_index = cpf.compute_median_curvature_index(curvature, 1)
        skew_pearson = 3 * (mean_abs - curv_abs_norm[apex_index]) / std_dev
        return float(round(skew_pearson, 4))

    def compute_bend_wavelength_leopold(self: Self, bend_id: int) -> float:
        """Compute bend wavelength according to Leopold method.

        Leopold method is described in `Leopold and Wolman (1957) <https://doi.org/10.1130/0016-7606(1960)71[769:RM]2.0.CO;2)>`_

        The first and last bends lack a neighbouring bend, so the Leopold
        wavelength is undefined for them and NaN is returned.

        Args:
            bend_id (int): bend index.

        Returns:
            float: Leopold wavelength, or NaN for first and last bends or
            when neighbouring bends have no apex.
        """
        nb_bends: int = self.centerline.get_nb_bends()
        assert (bend_id > -1) and (bend_id < nb_bends), (
            "Bend index is undefined."
        )
        if (bend_id == 0) or (bend_id == nb_bends - 1):
            return np.nan
        prev_bend: Bend = self.centerline.bends[bend_id - 1]
        next_bend: Bend = self.centerline.bends[bend_id + 1]
        if not self._has_valid_apex(prev_bend) or not self._has_valid_apex(
            next_bend
        ):
            return np.nan
        return geom.distance(
            self._get_pt_2d(prev_bend.index_apex),
            self._get_pt_2d(next_bend.index_apex),
        )

    def compute_bend_amplitude_leopold(self: Self, bend_id: int) -> float:
        """Compute bend ampltiude according to Leopold method.

        Leopold method is described in `Leopold and Wolman (1957) <https://doi.org/10.1130/0016-7606(1960)71[769:RM]2.0.CO;2>`_

        The first and last bends lack a neighbouring bend, so the Leopold
        amplitude is undefined for them and NaN is returned.

        Args:
            bend_id (int): bend index.

        Returns:
            float: Leopold amplitude, or NaN for first and last bends or
            when the bend or its neighbours have no apex.
        """
        nb_bends: int = self.centerline.get_nb_bends()
        assert (bend_id > -1) and (bend_id < nb_bends), (
            "Bend index is undefined."
        )
        if (bend_id == 0) or (bend_id == nb_bends - 1):
            return np.nan

        prev_bend: Bend = self.centerline.bends[bend_id - 1]
        bend: Bend = self.centerline.bends[bend_id]
        next_bend: Bend = self.centerline.bends[bend_id + 1]
        if not (
            self._has_valid_apex(prev_bend)
            and self._has_valid_apex(bend)
            and self._has_valid_apex(next_bend)
        ):
            return np.nan

        return geom.orthogonal_distance(
            self._get_pt_2d(bend.index_apex),
            self._get_pt_2d(prev_bend.index_apex),
            self._get_pt_2d(next_bend.index_apex),
        )

    def compute_bend_sinuosity_moving_window(
        self: Self,
        bend_id: int,
        window_size: float,
    ) -> float:
        """Compute bend sinuosity inside a moving window.

        The window is centred on the bend: it covers the curvilinear range
        [s_mid - window_size / 2, s_mid + window_size / 2] where s_mid is the
        curvilinear abscissa of the middle of the bend (halfway between its
        inflection points). All the bends overlapping this range are
        included, so the sinuosity is computed between the upstream
        inflection point of the first bend and the downstream inflection
        point of the last bend of the window.

        Args:
            bend_id (int): bend index
            window_size (float): total curvilinear length of the window
                centred on the bend [m].

        Returns:
            float: sinuosity over the window.

        """
        # get bends included in the window
        bend_index_min, bend_index_max = self._get_window_end_indexes(
            bend_id, window_size
        )
        cl_ptmin: ClPoint = self.centerline.cl_points[
            self.centerline.bends[bend_index_min].index_inflex_up
        ]
        cl_ptmax: ClPoint = self.centerline.cl_points[
            self.centerline.bends[bend_index_max].index_inflex_down
        ]

        # compute sinuosity (in the horizontal plane)
        arc_length: float = float(round(cl_ptmax._s - cl_ptmin._s, 4))
        cart_length: float = geom.distance(cl_ptmin.pt[:2], cl_ptmax.pt[:2], 4)
        sinuo: float = (
            round(abs(arc_length / cart_length), 4)
            if abs(cart_length) > 0
            else np.nan
        )
        return sinuo

    def compute_average_metric_window(
        self: Self, bend_id: int, window_size: float
    ) -> pd.Series:
        """Compute average morphometrics inside a moving window.

        The window is centred on the bend (see
        compute_bend_sinuosity_moving_window). For each metric, the average
        is computed over the bends of the window ignoring NaN values (i.e.,
        only non-NaN values are counted). A metric that is NaN for all the
        bends of the window is NaN.

        Args:
            bend_id (int): bend index
            window_size (float): total curvilinear length of the window
                centred on the bend [m].

        Returns:
            pd.Series: Series containing average values of all metrics.

        """
        # get bends included in the window
        bend_index_min, bend_index_max = self._get_window_end_indexes(
            bend_id, window_size
        )
        count: int = bend_index_max - bend_index_min + 1
        assert count > 0, "Total number of bends must be strictly positive."

        # compute average metric
        metrics_all: pd.DataFrame = self.compute_bends_morphometry(False)
        window: pd.DataFrame = metrics_all.iloc[
            bend_index_min : bend_index_max + 1
        ].astype(float)
        mean: pd.Series = window.mean(axis=0, skipna=True)
        return mean.round(4)

    def _get_bend_curv_abscissa_range(
        self: Self, bend_index: int
    ) -> tuple[float, float]:
        """Get the curvilinear abscissa of bend inflection points.

        Args:
            bend_index (int): bend index

        Returns:
            tuple[float, float]: curvilinear abscissa of upstream and
            downstream inflection points.
        """
        bend: Bend = self.centerline.bends[bend_index]
        s_up: float = float(self.centerline.cl_points[bend.index_inflex_up]._s)
        s_down: float = float(
            self.centerline.cl_points[bend.index_inflex_down]._s
        )
        return s_up, s_down

    def _get_window_end_indexes(
        self: Self, bend_index: int, window_size: float
    ) -> tuple[int, int]:
        """Compute the indexes of first and last bends included in the window.

        The window is the curvilinear range
        [s_mid - window_size / 2, s_mid + window_size / 2] centred on the
        middle abscissa s_mid of the bend (halfway between its inflection
        points). Returned indexes are those of the first and last bends
        overlapping this range. The current bend is always included.

        Args:
            bend_index (int): current bend index
            window_size (float): total curvilinear length of the window [m].

        Returns:
            tuple[int, int]: indexes of first and last bends of the window.

        """
        nb_bends: int = len(self.centerline.bends)
        assert (bend_index > -1) and (bend_index < nb_bends), (
            "Bend index is undefined."
        )
        s_up, s_down = self._get_bend_curv_abscissa_range(bend_index)
        s_mid: float = 0.5 * (s_up + s_down)
        half_window: float = 0.5 * max(float(window_size), 0.0)
        smin: float = s_mid - half_window
        smax: float = s_mid + half_window

        # upstream bends: overlap if their downstream end is beyond smin
        jmin: int = bend_index
        while jmin > 0 and self._get_bend_curv_abscissa_range(jmin - 1)[1] > (
            smin
        ):
            jmin -= 1

        # downstream bends: overlap if their upstream end is before smax
        jmax: int = bend_index
        while (jmax < nb_bends - 1) and (
            self._get_bend_curv_abscissa_range(jmax + 1)[0] < smax
        ):
            jmax += 1
        return jmin, jmax
