# SPDX-FileCopyrightText: Copyright 2025 Martin Lemay <martin.lemay@mines-paris.org>
# SPDX-FileContributor: Martin Lemay
# ruff: noqa: E402 # disable Module level import not at top of file

from typing import Optional, Self

import numpy as np
import numpy.typing as npt
import pandas as pd  # type: ignore[import-untyped]

import pybend.algorithms.centerline_process_function as cpf
from pybend.model.Bend import Bend
from pybend.model.Centerline import Centerline, ClPoint
from pybend.model.enumerations import MorphometricNames, PropertyNames

__doc__ = r"""
This module defines the Morphometry class that compute meander bend
 morphometric parameters from a Centerline object.

Bends are defined as the channel path comprised between 2 consecutive
 inflection points (o). A Bend contains a maximum curvature point (+) and an
 apex (x) whose definition may vary.

Metrics include:

* arc length: curvilinear distance between inflection points
* wavelength (W): euclidean distance between inflection points
* amplitude (Am): orthogonal distance between bend apex and chord
* extension (Ex): distance between bend apex and center
* asymmetry coefficient: A=(Lup-Ldown) / L, where Lup and Ldown are arc length
* radius of curvature: inverse of bend apex curvature
* roundess: ratio of maximum to mean curvature along the bend distances between
 bend apex and upstream and downstream inflection point respectively, and L
 is the bend arc length.

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


class Morphometry:
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
            MorphometricNames.ROUNDNESS.value,
            MorphometricNames.WAVELENGTH_LEOPOLD.value,
            MorphometricNames.AMPLITUDE_LEOPOLD.value,
        )

    def compute_bends_morphometry(
        self: Self, valid_bends: bool = True
    ) -> pd.DataFrame:
        """Compute all bend morphometric parameters.

        Args:
            valid_bends (bool): if True, compute morphometry on valid bends
                only

                Defaults to True.

        Returns:
            pd.DataFrame: dataframe with morphometric measurements.
        """
        n_rows: int = (
            self.centerline.get_nb_valid_bends()
            if valid_bends
            else self.centerline.get_nb_bends()
        )
        data: pd.DataFrame = pd.DataFrame(
            np.full((n_rows, len(self._bend_metrics)), np.nan),
            columns=self._bend_metrics,
        )
        i = 0
        for bend in self.centerline.bends:
            if valid_bends and not bend.isvalid:
                continue
            data.loc[i, MorphometricNames.ARC_LENGTH.value] = (
                self.compute_bend_arc_length(bend.id)
            )
            data.loc[i, MorphometricNames.WAVELENGTH.value] = (
                self.compute_bend_wavelength(bend.id)
            )
            # sinuosity
            if data.loc[i, MorphometricNames.WAVELENGTH.value] > 0:
                data.loc[i, MorphometricNames.SINUOSITY.value] = np.round(
                    data.loc[i, MorphometricNames.ARC_LENGTH.value]
                    / data.loc[i, MorphometricNames.WAVELENGTH.value],
                    4,
                )
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
                self.compute_bend_asymmetry(bend.id)
            )
            data.loc[i, MorphometricNames.ROUNDNESS.value] = (
                self.compute_bend_roundness(bend.id)
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
            i += 1
        return data

    def compute_bend_sinuosity(self: Self, bend_id: int) -> float:
        """Compute bend sinuosity.

        Args:
            bend_id (int): bend index

        Returns:
            float: bend sinuosity
        """
        assert (bend_id > -1) and bend_id < self.centerline.get_nb_bends(), (
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
        assert (bend_id > -1) and bend_id < self.centerline.get_nb_bends(), (
            "Bend index is undefined."
        )
        bend: Bend = self.centerline.bends[bend_id]
        pt_inflex_up: npt.NDArray[np.float64] = self.centerline.cl_points[
            bend.index_inflex_up
        ].pt
        pt_inflex_down: npt.NDArray[np.float64] = self.centerline.cl_points[
            bend.index_inflex_down
        ].pt
        return cpf.distance(pt_inflex_up, pt_inflex_down)

    def compute_bend_amplitude(self: Self, bend_id: int) -> float:
        """Compute bend amplitude.

        Args:
            bend_id (int): bend index

        Returns:
            float: bend amplitude
        """
        assert (bend_id > -1) and bend_id < self.centerline.get_nb_bends(), (
            "Bend index is undefined."
        )
        bend: Bend = self.centerline.bends[bend_id]
        pt_apex: npt.NDArray[np.float64] = self.centerline.cl_points[
            bend.index_apex
        ].pt
        pt_inflex_up: npt.NDArray[np.float64] = self.centerline.cl_points[
            bend.index_inflex_up
        ].pt
        pt_inflex_down: npt.NDArray[np.float64] = self.centerline.cl_points[
            bend.index_inflex_down
        ].pt
        return cpf.orthogonal_distance(pt_apex, pt_inflex_up, pt_inflex_down)

    def compute_bend_extension(self: Self, bend_id: int) -> float:
        """Compute bend extension.

        Args:
            bend_id (int): bend index

        Returns:
            float: bend extension
        """
        assert (bend_id > -1) and bend_id < self.centerline.get_nb_bends(), (
            "Bend index is undefined."
        )
        bend: Bend = self.centerline.bends[bend_id]
        pt_apex: npt.NDArray[np.float64] = self.centerline.cl_points[
            bend.index_apex
        ].pt
        pt_center: Optional[npt.NDArray[np.float64]] = bend.pt_center
        if pt_center is not None:
            return cpf.distance(pt_apex, pt_center)
        return np.nan

    def compute_bend_radius(self: Self, bend_id: int) -> float:
        """Compute bend radius.

        Args:
            bend_id (int): bend index

        Returns:
            float: bend radius
        """
        assert (bend_id > -1) and bend_id < self.centerline.get_nb_bends(), (
            "Bend index is undefined."
        )
        bend: Bend = self.centerline.bends[bend_id]
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
        assert (bend_id > -1) and bend_id < self.centerline.get_nb_bends(), (
            "Bend index is undefined."
        )
        curv_abs: npt.NDArray[np.float64] = self.centerline.get_bend_property(
            bend_id, PropertyNames.CURVILINEAR_ABSCISSA.value
        )
        return float(round(curv_abs[-1] - curv_abs[0], 4))

    def compute_bend_roundness(self: Self, bend_id: int) -> float:
        """Compute bend roundness.

        Roundness coefficient from `Schwenk et al. (2015)<https://doi.org/10.1002/2014JF003252)>`_

        Args:
            bend_id (int): bend index

        Returns:
            float: bend roundness
        """
        assert (bend_id > -1) and bend_id < self.centerline.get_nb_bends(), (
            "Bend index is undefined."
        )
        curvature: npt.NDArray[np.float64] = np.abs(
            self.centerline.get_bend_curvature_filtered(bend_id)
        )
        return float(round(np.max(curvature) / np.mean(curvature), 4))

    def compute_bend_asymmetry(self: Self, bend_id: int) -> float:
        """Compute bend asymmetry coefficient.

        Asymmetry coefficient from `Howard and Hemberger (1991)<https://doi.org/10.1016/0169-555X(91)90002-R>`_

        Args:
            bend_id (int): bend index

        Returns:
            float: bend asymmetry coefficient
        """
        assert (bend_id > -1) and bend_id < self.centerline.get_nb_bends(), (
            "Bend index is undefined."
        )
        bend: Bend = self.centerline.bends[bend_id]
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

    def compute_bend_wavelength_leopold(self: Self, bend_id: int) -> float:
        """Compute bend wavelength according to Leopold method.

        Leopold method is described in `Leopold and Wolman (1957)<https://doi.org/10.1130/0016-7606(1960)71[769:RM]2.0.CO;2)>`_

        Args:
            bend_id (int): bend index.

        Returns:
            float: Leopold wavelength.
        """
        assert (
            bend_id > 0
        ) and bend_id < self.centerline.get_nb_bends() - 1, (
            "Bend index is undefined."
        )
        prev_bend: Bend = self.centerline.bends[bend_id - 1]
        next_bend: Bend = self.centerline.bends[bend_id + 1]
        clpt_apex_prev = self.centerline.cl_points[prev_bend.index_apex]
        clpt_apex_next = self.centerline.cl_points[next_bend.index_apex]
        return cpf.distance(clpt_apex_prev.pt, clpt_apex_next.pt)

    def compute_bend_amplitude_leopold(self: Self, bend_id: int) -> float:
        """Compute bend ampltiude according to Leopold method.

        Leopold method is described in `Leopold and Wolman (1957)<https://doi.org/10.1130/0016-7606(1960)71[769:RM]2.0.CO;2>`_

        Args:
            bend_id (int): bend index.

        Returns:
            float: Leopold amplitude.
        """
        assert (
            bend_id > 0
        ) and bend_id < self.centerline.get_nb_bends() - 1, (
            "Bend index is undefined."
        )
        prev_bend: Bend = self.centerline.bends[bend_id - 1]
        bend: Bend = self.centerline.bends[bend_id]
        next_bend: Bend = self.centerline.bends[bend_id + 1]

        clpt_apex_prev = self.centerline.cl_points[prev_bend.index_apex]
        clpt_apex = self.centerline.cl_points[bend.index_apex]
        clpt_apex_next = self.centerline.cl_points[next_bend.index_apex]
        return cpf.orthogonal_distance(
            clpt_apex.pt, clpt_apex_prev.pt, clpt_apex_next.pt
        )

    def compute_bend_sinuosity_moving_window(
        self: Self,
        bend_id: int,
        window_size: float,
    ) -> float:
        """Compute bend sinuosity inside a moving window.

        Args:
            bend_id (int): bend index
            window_size (float): curvulinear length of the window [m].

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

        # compute sinuosity
        arc_length: float = float(round(cl_ptmax._s - cl_ptmin._s, 4))
        cart_length: float = cpf.distance(cl_ptmin.pt, cl_ptmax.pt, 4)
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

        Args:
            bend_id (int): bend index
            window_size (float): curvulinear length of the window [m].

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
        mean: pd.Series = pd.Series(
            np.zeros(metrics_all.shape[1]), index=metrics_all.columns
        )
        for i in range(bend_index_min, bend_index_max + 1, 1):
            mean += metrics_all.loc[i]
        return np.round(mean / count, 4)

    def _get_window_end_indexes(
        self: Self, bend_index: int, window_size: float
    ) -> tuple[int, int]:
        """Compute the indexes of first and last bends included in the window.

        Args:
            bend_index (int): current bend index
            window_size (float): curvilinear length of the window [m].

        Returns:
            tuple[int, int]: indexes of first and last bends of the window.

        """
        bend = self.centerline.bends[bend_index]
        cl_ptmin = self.centerline.cl_points[bend.index_inflex_up]
        smin = cl_ptmin._s - window_size
        jmin = bend_index
        while cl_ptmin._s > smin and jmin > 0:
            jmin -= 1
            cl_ptmin = self.centerline.cl_points[
                self.centerline.bends[jmin].index_inflex_up
            ]

        smax = cl_ptmin._s + window_size
        jmax = bend_index
        cl_ptmax = self.centerline.cl_points[bend.index_inflex_down]
        while cl_ptmax._s < smax and jmax < len(self.centerline.bends) - 1:
            jmax += 1
            cl_ptmax = self.centerline.cl_points[
                self.centerline.bends[jmax].index_inflex_down
            ]
        return jmin, jmax
