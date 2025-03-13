# SPDX-FileCopyrightText: Copyright 2025 Martin Lemay <martin.lemay@mines-paris.org>
# SPDX-FileContributor: Martin Lemay
# ruff: noqa: E402 # disable Module level import not at top of file

from typing import Optional

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.markers import MarkerStyle

from pybend.model.Bend import Bend
from pybend.model.BendEvolution import BendEvolution
from pybend.model.Centerline import Centerline
from pybend.model.CenterlineCollection import CenterlineCollection
from pybend.model.ClPoint import ClPoint
from pybend.model.enumerations import BendSide
from pybend.model.Section import Section
from pybend.utils.logging import logger

__doc__ = """
Plot methods.
"""


def plot_centerline_collection(
    filepath: str,
    cl_collec: CenterlineCollection,
    domain: tuple[tuple[float, float], tuple[float, float]],
    nb_cl: int = 999,
    show: bool = False,
    annotate: bool = False,
    plot_apex: bool = True,
    plot_inflex: bool = False,
    plot_middle: bool = False,
    plot_centroid: bool = False,
    annot_text_size: int = 10,
    color_bend: bool = False,
    plot_apex_trajec: bool = False,
    plot_middle_trajec: bool = False,
    plot_centroid_trajec: bool = False,
    plot_normal: bool = False,
    scale_normal: float = 1.0,
    plot_section: bool = False,
    plot_warping: bool = True,
    cmap_name: str = "Blues",
) -> None:
    """Function to plot CenterlineCollection object.

    Args:
        filepath (str): path to export figures if not empty.
        cl_collec (CenterlineCollection): CenterlineCollection object to plot
        domain (tuple[tuple[float, float],tuple[float, float]]): display domain
        nb_cl (int, optional): Number of centerline to show.

            Defaults to 999 (i.e., plot all centerlines).
        show (bool, optional): if True, show the figure.

            Defaults to False.
        annotate (bool, optional): if True, add bend ids.

            Defaults to False.
        plot_apex (bool, optional): if True, plot bend apex.

            Defaults to True.
        plot_inflex (bool, optional): if True, plot inflection points.

            Defaults to False.
        plot_middle (bool, optional): if True, plot bend middle point.

            Defaults to False.
        plot_centroid (bool, optional): if True, plot bend centroid.

            Defaults to False.
        annot_text_size (int, optional): Text size for annotations.

            Defaults to 10.
        color_bend (bool, optional): if True, bends are colored in blue and red
            according to UP and DOWN side respectively.

            Defaults to False.
        plot_apex_trajec (bool, optional): if True, plot apex trajectory.

            Defaults to False.
        plot_middle_trajec (bool, optional): if True, plot middle trajectory.

            Defaults to False.
        plot_centroid_trajec (bool, optional): if True, plot bend centroid
            trajectory.

            Defaults to False.
        plot_normal (bool, optional): if True, plot normal vector of channel
            points.

            Defaults to False.
        scale_normal (float, optional): Scale for normal vectors.

            Defaults to 1.0.
        plot_section (bool, optional): if True, plot section lines.

            Defaults to False.
        plot_warping (bool, optional): if True, plot channel point trajectory.

            Defaults to True.
        cmap_name (str, optional): Name of the color map to use.

            Defaults to "Blues".

    """
    all_ages: npt.NDArray[np.int64] = cl_collec.get_all_ages()

    # get the centerlines to plot
    keys: npt.NDArray[np.int64] = _get_keys_to_plot(all_ages, nb_cl)

    # get color map
    cmap: colors.Colormap = plt.colormaps[cmap_name]
    cmap_norm: colors.Normalize = colors.Normalize(vmin=keys[0], vmax=keys[-1])

    # create plot
    fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
    for key in keys:
        cl: Centerline = cl_collec.centerlines[key]
        cl_color: tuple[float, float, float, float] = cmap(cmap_norm(key))

        # plot the last centerline bigger markers
        markersize: float = 1.0
        if key == np.max(all_ages):
            markersize = 2.0

        plot_bends(
            ax,
            (cl.cl_points,),
            cl.bends,
            domain=domain,
            annotate=annotate,
            plot_apex=plot_apex,
            plot_inflex=plot_inflex,
            plot_middle=plot_middle,
            plot_centroid=plot_centroid,
            plot_normal=plot_normal,
            scale_normal=scale_normal,
            annot_text_size=annot_text_size,
            color_bend=color_bend,
            alpha=1,
            linewidth=1,
            markersize=markersize,
            cl_color=cl_color,
        )

    for bend_evol in cl_collec.bends_evol:
        _plot_bend_evol_trajectories(
            ax,
            bend_evol,
            plot_apex_trajec,
            plot_middle_trajec,
            plot_centroid_trajec,
        )

    if plot_section and cl_collec.section_lines:
        for section_line in cl_collec.section_lines:
            coords: npt.NDArray[np.float64] = np.array(section_line.coords)
            if (coords.shape[0] == 2) and (coords.shape[1] == 2):
                ax.plot(
                    (coords[0][0], coords[1][0]),
                    (coords[0][1], coords[1][1]),
                    "k-",
                    linewidth=1,
                )

    if plot_warping:
        _plot_warping(ax, cl_collec)

    _update_plot_properties(filepath, domain, show)


def plot_centerline_single(
    filepath: str,
    cl_points: tuple[list[ClPoint]],
    bends: list[Bend],
    domain: tuple[tuple[float, float], tuple[float, float]],
    show: bool = False,
    annotate: bool = False,
    plot_apex: bool = True,
    plot_inflex: bool = False,
    plot_middle: bool = False,
    plot_centroid: bool = False,
    plot_pt_start: bool = False,
    plot_apex_proba: bool = False,
    plot_normal: bool = False,
    scale_normal: float = 1.0,
    annot_text_size: float = 10,
    color_bend: bool = True,
    linewidth: float = 1,
    markersize: float = 2,
    ax0: Optional[Axes] = None,
) -> None:
    """Plot a single centerline.

    Args:
        filepath (str): path to export figures if not empty.
        cl_points (tuple[list[ClPoint]]): list of ClPoint objects.
        bends (list[Bend]): list of Bend objects to plot
        domain (tuple[tuple[float, float],tuple[float, float]]): display domain
        show (bool, optional): if True, show the figure.

            Defaults to False.
        annotate (bool, optional): if True, add bend ids.

            Defaults to False.
        plot_apex (bool, optional): if True, plot bend apex.

            Defaults to True.
        plot_inflex (bool, optional): if True, plot inflection points.

            Defaults to False.
        plot_middle (bool, optional): if True, plot bend middle point.

            Defaults to False.
        plot_centroid (bool, optional): if True, plot bend centroid.

            Defaults to False.
        plot_pt_start (bool, optional): if True, plot centerline starting point

            Defaults to False.
        plot_apex_proba (bool, optional): If True, color channel points with
            apex probability property values.

            Defaults to False.
        plot_normal (bool, optional): if True, plot normal vector of channel
            points.

            Defaults to False.
        scale_normal (float, optional): Scale for normal vectors.

            Defaults to 1.0.
        annot_text_size (float, optional): Text size for annotations.

            Defaults to 10.
        color_bend (bool, optional): if True, bends are colored in blue and red
            according to UP and DOWN side respectively.

            Defaults to True.
        linewidth (float, optional): Line width.

            Defaults to 1.
        markersize (float, optional): Marker size.

            Defaults to 2.
        ax0 (Optional[Axes], optional): Axes where to plot.

            Defaults to None.

    """
    ax: Axes
    if ax0 is None:
        fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
    else:
        ax = ax0

    plot_bends(
        ax,
        cl_points,
        bends,
        domain=domain,
        annotate=annotate,
        plot_apex=plot_apex,
        plot_inflex=plot_inflex,
        plot_middle=plot_middle,
        plot_centroid=plot_centroid,
        plot_apex_proba=plot_apex_proba,
        plot_normal=plot_normal,
        scale_normal=scale_normal,
        annot_text_size=annot_text_size,
        color_bend=color_bend,
        linewidth=linewidth,
        markersize=markersize,
        alpha=1,
        cl_color=None,
    )

    if plot_pt_start:
        index_start = bends[0].index_inflex_up
        pt_start = cl_points[0][index_start].pt
        ax.plot(pt_start[0], pt_start[1], "kx", markersize=10)

    # update plot properties if no ax was provided
    if ax0 is None:
        _update_plot_properties(filepath, domain, show)


def plot_bend_evol(
    ax: Axes,
    cl_collec: tuple[CenterlineCollection],
    bend_evol: BendEvolution,
    nb_cl: int = 999,
    domain: tuple[tuple[float, float], tuple[float, float]] = ((), ()),  # type: ignore
    annotate: bool = False,
    plot_apex: bool = True,
    plot_inflex: bool = False,
    plot_middle: bool = False,
    plot_centroid: bool = False,
    plot_centroid_trajec: bool = False,
    plot_apex_trajec: bool = False,
    plot_middle_trajec: bool = False,
    plot_section: bool = False,
    plot_warping: bool = False,
    color_bend: bool = False,
    markersize: float = 2,
    cmap_name: str = "Blues",
) -> None:
    """Plot BendEvolution object.

    Args:
        ax (Axes): Axes where to plot.
        cl_collec (tuple[CenterlineCollection]): CenterlineCollection object
        bend_evol (BendEvolution): BendEvolution object to plot.
        nb_cl (int, optional): Number of centerline to plot.

            Defaults to 999 (i.e., plot all centerlines).
        domain (tuple[tuple[float, float],tuple[float, float]]): display domain
        annotate (bool, optional): if True, add bend ids.

            Defaults to False.
        plot_apex (bool, optional): if True, plot bend apex.

            Defaults to True.
        plot_inflex (bool, optional): if True, plot inflection points.

            Defaults to False.
        plot_middle (bool, optional): if True, plot bend middle point.

            Defaults to False.
        plot_centroid (bool, optional): if True, plot bend centroid.

            Defaults to False.
        plot_centroid_trajec (bool, optional): if True, plot bend centroid
            trajectory.

            Defaults to False.
        plot_apex_trajec (bool, optional): if True, plot bend apex trajectory.

            Defaults to False.
        plot_middle_trajec (bool, optional): if True, plot bend middle point
            trajectory.

            Defaults to False.
        plot_section (bool, optional): if True, plot section lines.

            Defaults to False.

        plot_warping (bool, optional): if True, plot channel point
            trajectories.

            Defaults to False.
        annot_text_size (float, optional): Text size for annotations.

            Defaults to 10.
        color_bend (bool, optional): if True, bends are colored in blue and red
            according to UP and DOWN side respectively.

            Defaults to True.
        linewidth (float, optional): Line width.

            Defaults to 1.
        markersize (float, optional): Marker size.
        cmap_name (str, optional): Name of the color map to use.

            Defaults to "Blues".

    """
    bend_evol_all_iter: npt.NDArray[np.int64] = bend_evol.get_all_ages()
    # get the centerlines to plot
    keys: npt.NDArray[np.int64] = _get_keys_to_plot(bend_evol_all_iter, nb_cl)

    # get color map
    cmap: colors.Colormap = plt.colormaps[cmap_name]
    cmap_norm = colors.Normalize(vmin=keys[0], vmax=keys[-1])
    # list of upstream and downstream bend indexes for plot_warping if needed
    indexes: dict[int, tuple[int, int]] = {}
    for age, bend_indexes in bend_evol.bend_indexes.items():
        bends: list[Bend] = [
            cl_collec[0].centerlines[age].bends[bend_index]
            for bend_index in bend_indexes
        ]
        indexes[age] = (bends[0].index_inflex_up, bends[-1].index_inflex_down)

        if age not in keys:
            continue

        cl_color: tuple[float, float, float, float] = cmap(cmap_norm(age))
        markersize0: float = markersize
        if age == np.max(keys):
            markersize0 = 2.0 * markersize

        plot_bends(
            ax,
            (cl_collec[0].centerlines[age].cl_points,),
            bends,
            domain=domain,
            annotate=annotate,
            plot_apex=plot_apex,
            plot_inflex=plot_inflex,
            plot_middle=plot_middle,
            plot_centroid=plot_centroid,
            annot_text_size=10,
            color_bend=color_bend,
            alpha=1,
            linewidth=2,
            markersize=markersize0,
            cl_color=cl_color,
        )

    _plot_bend_evol_trajectories(
        ax,
        bend_evol,
        plot_apex_trajec,
        plot_middle_trajec,
        plot_centroid_trajec,
    )

    if plot_warping:
        _plot_warping(ax, cl_collec[0], indexes)

    if plot_section and (len(cl_collec[0].sections) > 0):
        section_indexes: set[int] = set()
        for age, bend_indexes in bend_evol.bend_indexes.items():
            for bend_index in bend_indexes:
                section_index: Optional[list[int]] = (
                    cl_collec[0]
                    .centerlines[age]
                    .bends[bend_index]
                    .intersected_section_indexes
                )
                if section_index is not None:
                    section_indexes.update(set(section_index))

        for index in section_indexes:
            section: Section = cl_collec[0].sections[index]
            X, Y = [], []
            for isoline in section.isolines:
                i = len(isoline.points) // 2  # centerline point
                pt = isoline.points[i]
                X += [pt[0]]
                Y += [pt[1]]
                ax.plot(X, Y, "k-", linewidth=1)


def plot_bends(
    ax: Axes,
    cl_points: tuple[list[ClPoint]],
    bends: list[Bend],
    domain: tuple[tuple[float, float], tuple[float, float]] = ((), ()),  # type: ignore
    annotate: bool = False,
    plot_apex: bool = True,
    plot_inflex: bool = False,
    plot_middle: bool = False,
    plot_centroid: bool = False,
    plot_normal: bool = False,
    scale_normal: float = 1.0,
    annot_text_size: float = 10,
    color_bend: bool = False,
    alpha: float = 1,
    linewidth: float = 1,
    markersize: float = 2,
    cl_color: Optional[str | tuple[float, float, float, float]] = None,
    plot_apex_proba: bool = False,
    plot_property: bool = False,
    property_name: str = "",
    rotate: bool = False,
) -> None:
    """Plot Bend objects.

    ax (Axes): Axes where to plot.
    cl_points (tuple[list[ClPoint]]): list of ClPoint objects
    bends (list[Bend]): list of Bend objects to plot.
    domain (tuple[tuple[float, float], tuple[float, float]]): display domain
    annotate (bool, optional): if True, add bend ids.

        Defaults to False.
    plot_apex (bool, optional): if True, plot bend apex.

        Defaults to True.
    plot_inflex (bool, optional): if True, plot inflection points.

        Defaults to False.
    plot_middle (bool, optional): if True, plot bend middle point.

        Defaults to False.
    plot_centroid (bool, optional): if True, plot bend centroid.

        Defaults to False.
    plot_normal (bool, optional): if True, plot normal vector of channel
        points.

        Defaults to False.
    scale_normal (float, optional): Scale for normal vectors.

        Defaults to 1.0.
    annot_text_size (float, optional): Text size for annotations.

        Defaults to 10.
    color_bend (bool, optional): if True, bends are colored in blue and red
        according to UP and DOWN side respectively.

        Defaults to True.
    alpha (float, optional): Transparency.

        Defaults to 1.0.
    linewidth (float, optional): Line width.

        Defaults to 1.
    markersize (float, optional): Marker size.
    cl_color (Optional[tuple[Any]]): Centerline color. If plot_bend is set to
        True, centerline color is overwrite.

        Defaults to None.
    plot_apex_proba (bool, optional): If True, color channel points with
        apex probability property values.

        Defaults to False.
    plot_property (bool, optional): If True, color channel points with
        input property values.

        Defaults to False.

    property_name (str, optional): If plot_property is True, name if the
        property to plot.

        Defaults to "".

    rotate (bool, optional): if True, rotate bend such as inflection points
        are aligned along horizontal axis.

        Defaults to False.

    """
    color: str | tuple[float, float, float, float] = "k"
    if cl_color is not None:
        color = cl_color
    index: int = 0
    for i, bend in enumerate(bends):
        coords: npt.NDArray[np.float64] = np.full(
            (bend.get_nb_points(), 2), np.nan
        )
        for i, cl_pt in enumerate(
            cl_points[0][bend.index_inflex_up : bend.index_inflex_down + 1]
        ):
            coords[i, 0] = cl_pt.pt[0]
            coords[i, 1] = cl_pt.pt[1]

        if rotate:
            # compute rotation angle
            vec_inflex: npt.NDArray[np.float64] = (
                cl_points[0][bend.index_inflex_down].pt
                - cl_points[0][bend.index_inflex_up].pt
            )[:2]
            vec_inflex /= np.linalg.norm(vec_inflex)
            cos = np.dot(vec_inflex, np.array([1.0, 0.0]))
            # sin=cos(pi/2-a)
            vec_inflex2: npt.NDArray[np.float64] = np.array(
                [vec_inflex[1], -vec_inflex[0]]
            )
            sin = np.dot(vec_inflex2, np.array([1.0, 0.0]))
            # rotate x,y coordinates
            rot: npt.NDArray[np.float64] = np.array([[cos, -sin], [sin, cos]])
            coords = np.dot(coords, rot)
            coords -= (coords[0] + coords[1]) / 2.0

        if color_bend:
            color = "r"
            if bend.side == BendSide.UP:
                color = "b"

        ax.plot(
            coords[:, 0],
            coords[:, 1],
            linestyle="-",
            linewidth=linewidth,
            color=color,
            alpha=alpha,
        )

        if plot_inflex:
            ax.plot(
                coords[0, 0],
                coords[0, 1],
                marker="o",
                markerfacecolor="green",
                markeredgecolor="k",
                markersize=markersize,
            )

            if i == len(bends) - 1:
                ax.plot(
                    coords[-1, 0],
                    coords[-1, 1],
                    marker="o",
                    markerfacecolor="green",
                    markeredgecolor="k",
                    markersize=markersize,
                )

        if plot_apex and bend.isvalid and bend.index_apex > -1:
            index = bend.index_apex - bend.index_inflex_up
            ax.plot(
                coords[index, 0],
                coords[index, 1],
                marker="d",
                markeredgecolor="k",
                markerfacecolor="r",
                markersize=1.5 * markersize,
            )

        if plot_middle and bend.isvalid and bend.pt_center is not None:
            pt_center: npt.NDArray[np.float64] = bend.pt_center
            if rotate:
                pt_center = (coords[-1] + coords[0]) / 2.0
            ax.plot(
                pt_center[0],
                pt_center[1],
                marker="o",
                color="k",
                markersize=0.8 * markersize,
            )

        if plot_centroid and bend.isvalid and bend.pt_centroid is not None:
            pt_centroid: npt.NDArray[np.float64] = bend.pt_centroid
            if rotate:
                pt_centroid = np.mean(coords, axis=0)
            ax.plot(
                pt_centroid[0],
                pt_centroid[1],
                marker="o",
                markeredgecolor="k",
                markerfacecolor="orange",
                markersize=0.8 * markersize,
            )

        if (
            plot_apex_proba
            and bend.isvalid
            and bend.apex_probability is not None
        ):
            ax.scatter(
                coords[:, 0],
                coords[:, 1],
                marker=MarkerStyle("o"),
                c=bend.apex_probability,
                cmap="jet",
            )

        if plot_normal:
            for i, cl_pt in enumerate(
                cl_points[0][bend.index_inflex_up : bend.index_inflex_down + 1]
            ):
                plt.arrow(
                    coords[i, 0],
                    coords[i, 1],
                    cl_pt.get_property("Normal_x") * scale_normal,
                    cl_pt.get_property("Normal_y") * scale_normal,
                    color="k",
                    width=4,
                    linewidth=1,
                )

        if plot_property:
            prop = np.array(
                [
                    cl_pt.get_property(property_name)
                    for cl_pt in cl_points[0][
                        bend.index_inflex_up : bend.index_inflex_down + 1
                    ]
                ]
            )

            vmax = np.max(np.abs(prop))
            ax.scatter(
                coords[:, 0],
                coords[:, 1],
                marker=MarkerStyle("o"),
                c=prop,
                cmap="seismic",
                vmin=-1 * vmax,
                vmax=vmax,
            )

        if annotate and bend.index_apex > -1:
            index = bend.index_apex - bend.index_inflex_up
            pt_apex = coords[index]
            if (
                len(domain[0]) > 0
                and pt_apex[0] > domain[0][0]
                and pt_apex[0] < domain[0][1]
                and pt_apex[1] > domain[1][0]
                and pt_apex[1] < domain[1][1]
            ):
                plt.text(
                    pt_apex[0],
                    pt_apex[1],
                    str(bend.bend_evol_id),
                    size=10,
                )
            else:
                plt.text(
                    pt_apex[0],
                    pt_apex[1],
                    str(bend.bend_evol_id),
                    size=annot_text_size,
                    horizontalalignment="center",
                )


def plot_section(
    section: Section,
    ax: Axes,
    norm_hor: float = 1,
    norm_vert: float = 1,
    color_same_bend: bool = True,
    cmap_name: str = "",
) -> None:
    """Plot Section object.

    Args:
        section (Section): Section object to plot
        ax (Axes): Axes where to plot
        norm_hor (float, optional): Horizontal normalization.

            Defaults to 1.
        norm_vert (float, optional): Vertical normalization.

            Defaults to 1.
        color_same_bend (bool, optional): If True, use a color for isolines
            that belongs to the last bend and another for the other isolines.

            Defaults to None.
        cmap_name (str, optional): name of the color map.

            Defaults to "".

    """
    # get color map
    colors_norm: Optional[colors.Normalize] = None
    cmap: Optional[colors.Colormap] = None
    if not color_same_bend:
        colors_norm = colors.Normalize(vmin=0, vmax=len(section.isolines))
        cmap = (
            colors.Colormap(cmap_name)
            if len(cmap_name) > 0
            else colors.Colormap("Blues")
        )

    for i, isoline in enumerate(section.isolines):
        # coordinates to plot
        origin: tuple[float, float] = section.isolines_origin[i]
        # print(isoline.points)
        # print(origin)
        coords = np.array(isoline.points).T
        # print(coords[0])
        lx: npt.NDArray[np.float64] = (origin[0] + coords[0]) / norm_hor
        ly: npt.NDArray[np.float64] = (origin[1] + coords[1]) / norm_vert
        # print(ly)
        # print()
        color: str | tuple[float, float, float, float]
        if not color_same_bend:
            color = cmap(colors_norm(i))
        else:
            color = "b"
            if isoline.cl_pt_ref.get_property("Curvature") < 0:
                color = "r"

        ax.fill(
            lx,
            ly,
            linestyle="--",
            edgecolor=color,
            fill=True,
            facecolor="w",
        )


def plot_versus_curvilinear(
    work_dir: str,
    abscissa: npt.NDArray[np.float64],
    curves1: list[npt.NDArray[np.float64]],
    labels1: list[str],
    curves2: list[npt.NDArray[np.float64]],
    labels2: list[str],
    show: bool = False,
) -> None:
    """Plot 2 set of properties against abscissa.

    Args:
        work_dir (str): file name to save the figure if not empty.
        abscissa (npt.NDArray[np.float64]): abscissa values
        curves1 (list[npt.NDArray[np.float64]]): first set of curves
        labels1 (list[str]): labels of first set of curves
        curves2 (list[npt.NDArray[np.float64]]): second set of curves
        labels2 (list[str]): labels of second set of curves
        show (bool, optional): if True, show the figure.

            Defaults to False.

    """
    colors1: list[str] = [
        "b",
        "g",
        "m",
        "c",
        "lawngreen",
        "purple",
        "dodgerblue",
    ]
    colors2: list[str] = [
        "r",
        "orange",
        "y",
        "chocolate",
        "gold",
        "coral",
        "hotpink",
    ]

    assert len(curves1) == len(labels1), (
        "The number of Curves and labels from first set is different"
    )
    assert len(curves2) == len(labels2), (
        "The number of Curves and labels from second set is different"
    )
    assert len(curves1) > len(colors1), (
        "Too many curves to plot from first set."
    )
    assert len(curves2) > len(colors2), (
        "Too many curves to plot from second set."
    )

    _, ax1 = plt.subplots()

    color: str = "k"
    for k, curve in enumerate(curves1):
        color = colors1[k]
        ax1.plot(abscissa, curve, color, label=labels1[k])
        if k == 0:
            ax1.set_ylabel(labels1[0], color="k")
            for tl in ax1.get_yticklabels():
                tl.set_color("k")

    ax1.set_xlabel("Curvilinear abscissa (m)")
    if len(curves2) > 0:
        ax2 = ax1.twinx()
        for k, curve in enumerate(curves2):
            color = colors2[k]
            ax2.plot(abscissa, curve, color, label=labels2[k])
            if k == 0:
                ax2.set_ylabel(labels2[0], color="r")
                for tl in ax2.get_yticklabels():
                    tl.set_color("r")

    plt.ylim(-0.1, 0.1)
    plt.tight_layout()
    if len(work_dir) > 0:
        plt.savefig(work_dir + "props_versus_curv_abscissa.png", dpi=300)

    if show:
        plt.show()

    plt.close("all")
    plt.close("all")


def _get_keys_to_plot(
    all_keys: npt.NDArray[np.int64], nb_cl: int
) -> npt.NDArray[np.int64]:
    """Get keys to plot according to the number of centerlines.

    Args:
        all_keys (npt.NDArray[np.int64]): All centerline ages
        nb_cl (int): Number of centerlines

    Returns:
        npt.NDArray[np.int64]: list of centerline ages to plot

    """
    # get the centerlines to plot
    keys: npt.NDArray[np.int64]
    if nb_cl > all_keys.size:
        keys = all_keys
    else:
        ite = np.linspace(np.min(all_keys), np.max(all_keys), nb_cl)
        keys = np.empty_like(ite).astype(np.int64)
        for i, it in enumerate(ite):
            diff = np.abs(all_keys - it)
            keys[i] = all_keys[diff == np.min(diff)]
    return keys


def _plot_bend_evol_trajectories(
    ax: Axes,
    bend_evol: BendEvolution,
    plot_apex_trajec: bool,
    plot_middle_trajec: bool,
    plot_centroid_trajec: bool,
) -> None:
    """Plot BendEvolution characteristic point trajectories if needed.

    Args:
        ax (Axes): Axes where to plot
        bend_evol (BendEvolution): BendEvolution object
        plot_apex_trajec (bool, optional): if True, plot bend apex trajectory.
        plot_middle_trajec (bool, optional): if True, plot bend middle point
            trajectory.
        plot_centroid_trajec (bool, optional): if True, plot bend centroid
            trajectory.

    """
    if plot_apex_trajec and (len(bend_evol.apex_trajec_smooth) > 0):
        coords0: npt.NDArray[np.float64] = np.array(
            bend_evol.apex_trajec_smooth
        )  # cpf.points2coords(bend_evol.apex_trajec_smooth)
        ax.plot(coords0[:, 0], coords0[:, 1], "r-", linewidth=1)

    if plot_middle_trajec and (len(bend_evol.middle_trajec_smooth) > 0):
        coords1: npt.NDArray[np.float64] = np.array(
            bend_evol.middle_trajec_smooth
        )  # cpf.points2coords(bend_evol.middle_trajec_smooth)
        ax.plot(coords1[:, 0], coords1[:, 1], "b-", linewidth=1)

    if plot_centroid_trajec and (len(bend_evol.centroid_trajec_smooth) > 0):
        coords2: npt.NDArray[np.float64] = np.array(
            bend_evol.centroid_trajec_smooth
        )  # cpf.points2coords(bend_evol.centroid_trajec_smooth)
        ax.plot(coords2[:, 0], coords2[:, 1], "-", color="orange", linewidth=1)


def _plot_warping(
    ax: Axes,
    cl_collec: CenterlineCollection,
    indexes: dict[int, tuple[int, int]] = {},  # noqa: B006
) -> None:
    """Plot centerline warping.

    Args:
        ax (Axes): Axes where to plot
        cl_collec (CenterlineCollection): CenterlineCollection object
        indexes (dict[int, tuple[int, int]]): dictionnary containing a list of
            indexes of centerline points to plot for each ages

            Defaults to empty dictionnary.

    """
    try:
        all_ages: npt.NDArray[np.int64] = cl_collec.get_all_ages()
        for i, key2 in enumerate(all_ages[:-1]):
            key1: int = all_ages[i + 1]

            ctls1: Centerline = cl_collec.centerlines[key1]
            ctls2: Centerline = cl_collec.centerlines[key2]

            if len(ctls1.index_cl_pts_prev_centerline) == 0:
                continue

            index_start, index_stop = 0, ctls1.get_nb_points()
            if len(indexes) > 0:
                index_start, index_stop = indexes[key1]

            warp_x, warp_y = [], []
            for index1, index2 in enumerate(
                ctls1.index_cl_pts_prev_centerline[index_start:index_stop]
            ):
                if index2 < 0:
                    continue

                pt1: npt.NDArray[np.float64] = ctls1.cl_points[
                    int(index_start + index1)
                ].pt
                pt2: npt.NDArray[np.float64] = ctls2.cl_points[int(index2)].pt
                warp_x += [[pt1[0], pt2[0]]]
                warp_y += [[pt1[1], pt2[1]]]

            for x, y in zip(warp_x, warp_y, strict=False):
                ax.plot(x, y, "k-", linewidth=0.25)
    except Exception as e:
        logger.error("Centerline warping was not plotted due to:")
        logger.error(str(e))


def _update_plot_properties(
    filepath: str,
    domain: tuple[tuple[float, float], tuple[float, float]],
    show: bool = False,
) -> None:
    """Update plot properties.

    Args:
        filepath (str): directory where to export figure if not empty
        domain (tuple[tuple[float, float], tuple[float, float]]): plot limits
            ((xmin, xmax), (ymin, ymax))
        show (bool, optional): if True, show figure.

            Defaults to False.

    """
    if (len(domain) == 0) or (len(domain[0]) == 0) or (len(domain[1]) == 0):
        plt.axis("equal")  # type: ignore[unreachable]
    elif len(domain[0]) > 0:
        plt.xlim(domain[0])  # type: ignore[unreachable]
    elif len(domain[1]) > 0:  # type: ignore[unreachable]
        plt.ylim(domain[1])  # type: ignore[unreachable]

    plt.grid(True, which="both", axis="both")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")

    if len(filepath) > 0:
        plt.savefig(filepath, dpi=300)

    if show:
        plt.show()

    plt.close("all")
