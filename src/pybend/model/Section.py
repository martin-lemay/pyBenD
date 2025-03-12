# SPDX-FileCopyrightText: Copyright 2025 Martin Lemay <martin.lemay@mines-paris.org>
# SPDX-FileContributor: Martin Lemay
# ruff: noqa: E402 # disable Module level import not at top of file

from enum import Enum
from typing import Optional, Self

import numpy as np
import numpy.typing as npt

import pybend.algorithms.centerline_process_function as cpf
from pybend.model.ClPoint import ClPoint
from pybend.model.Isoline import Isoline


class StackingPatternType(Enum):
    """Channel stacking pattern type.

    Let's suppose a section across a meander bend. The channel moves laterally
    and vertically through time. According to channel basal point trajectory,
    4 stacking pattern types are defined
    (see `Lemay et al., 2024 <https://doi.org/10.1144/SP540-2022-143>`_):

       One Way      | Aggradation then One Way |    Two Ways   |  Multiple Ways

                 .                    .           .                  .
                .                 .                   .                 .
             .                 .                        :                 :
          .                  .                        .                .
      .                     :                      .                 :
    .                       :                    .                       .

    *Channel Stacking pattern types. Each point represents the position of the
    channel basal point. ':' stands for aggradation phase.*

    """

    #: pure aggradation
    AGGRADATION = "Aggradation Only"
    #: one way migration
    ONE_WAY = "One Way"
    #: Aggradation then 1 way migration
    AGGRAD_ONE_WAY = "Aggradation Then One Way"
    #: two ways migration
    TWO_WAYS = "Two Way"
    #: multiple ways migration
    MULTIPLE_WAYS = "Multiple Ways"
    #: undefined
    UNDEFINED = "Undefined"


class Section:
    def __init__(
        self: Self,
        ide: str,
        bend_id: str,
        pt_start: npt.NDArray[np.float64],
        pt_stop: npt.NDArray[np.float64],
        isolines: list[Isoline],
        same_bend: Optional[list[bool]] = None,
        flow_dir: npt.NDArray[np.float64] = np.array([1, 0]),
    ) -> None:
        """Section object to store 2D stratigraphy composed of multiple Isolines.

        ..WARNING: Code implementation in progress...

        Parameters:
        ----------
            ide (str): section id
            bend_id (str): bend id crossed by the section
            pt_start (npt.NDArray[np.float64]): start point coordinates
            pt_stop (npt.NDArray[np.float64]): end point coordinates
            isolines (list[Isoline]): list of Isoline objects
            same_bend (Optional[list[bool]], optional): list of boolean, True
                if isoline at the same index belongs to the same BendEvolution
                object as the first one.

                Defaults to None.
            flow_dir (npt.NDArray[np.float64], optional): flow directin to
                orientate the section.

                Defaults to np.array([1, 0]).

        """
        #: section id
        self.id: str = ide
        #: reference bend id
        self.bend_id: str = bend_id
        #: section start point coordinates
        self.pt_start: npt.NDArray[np.float64] = pt_start
        #: section end point coordinates
        self.pt_stop: npt.NDArray[np.float64] = pt_stop
        #: section direction
        self.dir: npt.NDArray[np.float64] = pt_start - pt_stop
        self.dir /= np.linalg.norm(self.dir)

        #: list of isolines
        self.isolines: list[Isoline] = isolines
        if len(self.isolines) > 0:
            #: list of isoline origin coordinates from first isoline of the list
            self.isolines_origin: list[tuple[float, float]] = self._compute_origin(
                flow_dir
            )

        #: list of boolean, True if isoline at the same index belongs to the
        #: same BendEvolution object as the last one.
        self.same_bend: Optional[list[bool]] = same_bend
        if same_bend is None:
            self.same_bend = [True for _ in range(len(isolines))]

        #: list of displacements betweeen each pair of successive isolines
        self.local_disp: npt.NDArray[np.float64] = np.empty(0)
        #: list of average displacements between first and last isoline of the section
        self.averaged_disp: dict[str, npt.NDArray[np.float64]] = {}
        #: stacking pattern type
        self.stacking_pattern_type: StackingPatternType = StackingPatternType.UNDEFINED

    def _compute_origin(
        self: Self, flow_dir: npt.NDArray[np.float64] = np.array([1, 0])
    ) -> list[tuple[float, float]]:
        """Compute isoline coordinates along the section from reference ClPoint.

        Parameters:
        ----------
            flow_dir (npt.NDArray[np.float64], optional): Flow direction.

                Defaults to np.array([1, 0]).

        Returns:
        ----------
            list[tuple[float, float]]: isoline coordinates along the section
        """
        isolines_origin: list[tuple[float, float]] = []

        # use the orthogonal vector to the flow dir to find the sign
        flow_dir_perp: npt.NDArray[np.float64] = cpf.perp(flow_dir)

        cl_pt_ref: ClPoint = self.isolines[0].cl_pt_ref
        for isoline in self.isolines:
            cl_pt: ClPoint = isoline.cl_pt_ref
            # direction of migration according to the flow direction
            sign: float = 1.0
            # normed apparent mig vector
            vec: npt.NDArray[np.float64] = cl_pt.pt[:2] - cl_pt_ref.pt[:2]
            norm_vec = np.linalg.norm(vec)

            dot: float = 1.0
            if norm_vec > 0:
                vec /= norm_vec
                # dot product
                dot = np.dot(flow_dir_perp, vec)

            if dot < 0:
                sign = -1.0

            d: float = sign * cpf.distance(
                cl_pt_ref.pt, cl_pt.pt
            )  # distance to cl_pt_ref
            dz: float = cl_pt.pt[2] - cl_pt_ref.pt[2]

            isolines_origin += [(d, dz)]
        return isolines_origin

    def get_stacking_pattern_type(
        self: Self,
        mig_threshold: float,
        frac_threshold: float = 0.95,
        begin_threshold: float = 0.1,
    ) -> StackingPatternType:
        """Get the channel stacking pattern of isolines on the section.

        Parameters:
        ----------
            mig_threshold (float): lateral migration threshold (m)
            frac_threshold (float, optional): fraction of the total number of
                isolines.

                Defaults to 0.95.
            begin_threshold (float, optional): fraction of the total number of
                isolines.

                Defaults to 0.1.

        Returns:
        ----------
            StackingPatternType: Stacking pattern type
        """
        mig_steps: list[int] = []
        pt_origin_prev: tuple[float, float] = (0.0, 0.0)
        for i, pt_origin in enumerate(self.isolines_origin):
            if i == 0:
                continue

            mig: float = pt_origin[0] - pt_origin_prev[0]
            if abs(mig) < mig_threshold:
                mig_steps += [0]
            else:
                if mig > 0:
                    mig_steps += [1]
                else:
                    mig_steps += [-1]

            pt_origin_prev = pt_origin

        mig_steps1: npt.NDArray[np.int64] = np.array(mig_steps).astype(int)

        frac_0: float = float(np.sum(mig_steps1 == 0) / mig_steps1.size)
        frac_1: float = float(np.sum(mig_steps1 > 0) / mig_steps1.size)
        frac_2: float = float(np.sum(mig_steps1 < 0) / mig_steps1.size)

        groups: list[int] = []
        types: list[int] = []
        prev_mig_step = 2
        if (frac_1 > frac_threshold) | (frac_2 > frac_threshold):
            self.stacking_pattern_type = StackingPatternType.ONE_WAY
        elif frac_0 > frac_threshold:
            self.stacking_pattern_type = StackingPatternType.AGGRADATION
        elif ((frac_1 + frac_0) > frac_threshold) | (
            (frac_2 + frac_0) > frac_threshold
        ):
            for mig_step in mig_steps1:
                if (frac_1 > frac_2) & (mig_step == -1):
                    continue
                if (frac_1 < frac_2) & (mig_step == 1):
                    continue
                if mig_step == prev_mig_step:
                    groups[-1] += 1
                else:
                    groups += [1]
                    types += [mig_step]
                prev_mig_step = mig_step

            index0: int = 1  # index of the first phase of aggradation
            index1: int = 0  # index of the first phase of migration
            if 0 in types:
                index0 = types.index(0)
            if 1 in types:
                index1 = types.index(1)

            if (index0 == 1) & (groups[index1] > begin_threshold * mig_steps1.size):
                self.stacking_pattern_type = StackingPatternType.ONE_WAY
            elif groups[index0] > begin_threshold * mig_steps1.size:
                self.stacking_pattern_type = StackingPatternType.AGGRAD_ONE_WAY
            else:
                self.stacking_pattern_type = StackingPatternType.ONE_WAY
        else:
            for mig_step in mig_steps1:
                if mig_step == 0:
                    continue
                if mig_step == prev_mig_step:
                    groups[-1] += 1
                else:
                    groups += [1]
                    types += [mig_step]
                prev_mig_step = mig_step

            groups = list(filter(lambda a: a != 1, groups))
            if len(groups) == 1:
                self.stacking_pattern_type = (
                    StackingPatternType.ONE_WAY
                )  # should not happen
            elif len(groups) == 2:
                self.stacking_pattern_type = StackingPatternType.TWO_WAYS
            else:
                self.stacking_pattern_type = StackingPatternType.MULTIPLE_WAYS

        return self.stacking_pattern_type

    def channel_apparent_displacements(
        self: Self,
        norm_hor: float = 1,
        norm_vert: float = 1,
        smooth: bool = False,
    ) -> None:
        """Compute channel apparent displacements along the section.

        Parameters:
        ----------
            norm_hor (float, optional): Normalisation value for horizontal
                dimension.

                Defaults to 1.
            norm_vert (float, optional): Normalisation value for vertical
                dimension.

                Defaults to 1.
            smooth (bool, optional): if True, channel apparent trajectory is
                smoothed.

                Defaults to False.

        """
        l_pt: list[npt.NDArray[np.float64]] = [
            np.array(pt_origin) for pt_origin in self.isolines_origin
        ]
        # smooth isolines loc
        if smooth:
            ages: npt.NDArray[np.float64] = np.array(
                [isoline.age for isoline in self.isolines]
            ).astype(float)
            l_pt = cpf.resample_path(l_pt, ages, 0).T
            # l_pt = cpf.coords2points(
            #     cpf.resample_path(l_pt, ages, 0)#cpf.points2coords(l_pt), ages, 0)
            # )

        self.local_disp = np.full((len(l_pt) - 1, 3), np.nan)
        pt_origin_prev: npt.NDArray[np.float64] = np.array((0.0, 0.0))
        for i, pt_origin in enumerate(l_pt):
            if i == 0:
                continue
            self.local_disp[i - 1, 0] = (
                pt_origin[0] - pt_origin_prev[0]
            )  # lateral displacements
            self.local_disp[i - 1, 1] = (
                pt_origin[1] - pt_origin_prev[1]
            )  # vertical displacements
            pt_origin_prev = pt_origin
        # local stratigraphic number
        self.local_disp[:, 2] = (self.local_disp[:, 0] / self.local_disp[:, 1]) * (
            norm_vert / norm_hor
        )

    def section_averaged_channel_displacements(
        self: Self,
        norm_hor: float = 1.0,
        norm_vert: float = 1.0,
        write_results: bool = False,
        filepath: str = "",
    ) -> None:
        """Compute section-averaged channel displacement.

        Parameters:
        ----------
            norm_hor (float, optional): Normalisation value for horizontal
                dimension.

                Defaults to 1.
            norm_vert (float, optional): Normalisation value for vertical
                dimension.

                Defaults to 1
            write_results (bool, optional): if True, write results in a file.

                Defaults to False.
            filepath (str, optional): Full nae of the file to export the results.

                Defaults to "".
        """
        self.averaged_disp = {}
        self.averaged_disp["full"] = self._compute_average_disp(
            norm_hor, norm_vert, True
        )
        self.averaged_disp["bend"] = self._compute_average_disp(
            norm_hor, norm_vert, False
        )

        if write_results:
            with open(filepath, "a") as fout:
                line_full: str = ";".join(self.averaged_disp["full"][2:])
                line_bend: str = ";".join(self.averaged_disp["bend"][2:])
                fout.write(line_full)
                fout.write(line_bend)

    def _compute_average_disp(
        self: Self, width: float, depth: float, whole_trajec: bool
    ) -> npt.NDArray[np.float64]:
        """Compute section-averaged channel displacement metrics.

        Metrics include
        (see `Jobe et al., 2016<https://doi.org/10.1130/G38158.1>`_):
            * Dx: lateral displacement
            * Dy: vertical displacement
            * Bcb: channel belt with
            * Hcb: channel belt thickness
            * Bcb_norm: normalized channel belt with
            * Hcb_norm: normalized channel belt thickness
            * Bcb_on_Hcb: channel belt aspect ratio
            * Msb: Stratigraphic Mobility number

        Parameters:
        ----------
            width (float): channel width (m)
            depth (float): channel depth (m)
            whole_trajec (bool): if True, compute channel displacements from
                first to last channel of the isoline, otherwise from first to
                last channel of the last migration phase.

        Returns:
        ----------
            npt.NDArray[np.float64]: Averaged displacement metrics including in
                the order: Dx, Dz, Bcb, Hcb, Bcb_norm, Hcb_norm, Bcb_on_Hcb, Msb

        """
        pt_apex: tuple[float, float] = self.isolines_origin[-1]
        pt_ref: tuple[float, float] = self.isolines_origin[0]
        if (not whole_trajec) and (
            (self.stacking_pattern_type is None)
            or (
                (self.stacking_pattern_type is not None)
                and (self.stacking_pattern_type is not StackingPatternType.ONE_WAY)
            )
        ):
            # reference point from which to compute 'bend' averaged displacements
            pt_ref = pt_apex
            dmax: float = 0
            cpt: int = 0
            for pt_origin in self.isolines_origin[::-1]:
                d: float = abs(pt_apex[0] - pt_origin[0])
                if d > (dmax):
                    dmax = d
                    pt_ref = pt_origin
                    cpt = 0
                else:
                    cpt += 1

                if cpt > 3:
                    break

        # Dx, Dz, Bcb, Hcb, Bcb_norm, Hcb_norm, Bcb_on_Hcb, Msb
        metrics: npt.NDArray[np.float64] = np.full(8, np.nan)
        metrics[0] = round(abs(pt_apex[0] - pt_ref[0]), 4)
        metrics[1] = round(abs(pt_apex[1] - pt_ref[1]), 4)
        if metrics[1] != 0:
            metrics[7] = round((metrics[0] / metrics[1]) * (depth / width), 4)
            metrics[3] = metrics[1] + depth  # full channel belt thickness
            metrics[2] = metrics[0] + width  # full channel belt width
            metrics[6] = round(metrics[2] / metrics[3], 4)
        else:
            metrics[3] = np.nan if metrics[1] == 0 else metrics[1] + depth
            metrics[2] = np.nan if metrics[0] == 0 else metrics[0] + width

        metrics[4] = round(metrics[2] / width, 4)
        metrics[5] = round(metrics[3] / depth, 4)
        return metrics
