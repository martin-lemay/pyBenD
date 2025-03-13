# SPDX-FileCopyrightText: Copyright 2025 Martin Lemay <martin.lemay@mines-paris.org>
# SPDX-FileContributor: Martin Lemay
# ruff: noqa: E402 # disable Module level import not at top of file

import functools
import math
from multiprocessing import Pool
from typing import Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd  # type: ignore[import-untyped]
from scipy.interpolate import splev, splprep  # type: ignore[import-untyped]
from scipy.signal import find_peaks  # type: ignore[import-untyped]

from pybend.model.ClPoint import ClPoint
from pybend.utils.logging import logger

__doc__ = """
Usefull methods.
"""


def clpoints2coords(cl_pts: list[ClPoint]) -> npt.NDArray[np.float64]:
    """Transform a list of ClPoint into a numpy array (1 point per row).

    Args:
        cl_pts (list[ClPoint]): List of ClPoint.

    Returns:
        NDArray[float]: Array of point coordinates.

    """
    return np.array([cl_pt.pt for cl_pt in cl_pts])


def compute_cuvilinear_abscissa(
    XY: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Compute curvilinear abscissa from cartesian XY coordinates.

    Args:
        XY (NDArray[float]): 2D array with XY coordinates.

    Returns:
        NDArray[float]: Array of curvilinear abscissa values.

    """
    ds = distance_arrays(XY[:-1], XY[1:], 4)
    return np.append([0], np.cumsum(ds))


def compute_colinear(
    pt1: npt.NDArray[np.float64] | Sequence[float],
    pt2: npt.NDArray[np.float64] | Sequence[float],
    k: float,
) -> npt.NDArray[np.float64]:
    """Return a point which is k times (pt2-pt1) from pt1.

    Args:
        pt1 (npt.NDArray[np.float64] | Sequence[float]): Coordinates of
            the first point.
        pt2 (npt.NDArray[np.float64] | Sequence[float]):  Coordinates of
            the second point.
        k (float): Factor

    Returns:
        NDArray[float]: Array with computed coordinates.

    """
    pt1Array: npt.NDArray[np.float64] = np.array(pt1)
    pt2Array: npt.NDArray[np.float64] = np.array(pt2)
    return pt1Array + k * (pt2Array - pt1Array)


def distance_arrays(
    pts1: npt.NDArray[np.float64], pts2: npt.NDArray[np.float64], prec: int = 4
) -> npt.NDArray[np.float64]:
    """Compute the distance between points.

    Args:
        pts1 (NDArray[float]): 2D array with coordinates of the first points,
            1 point per row.
        pts2 (NDArray[float]): 2D array with coordinates of the second points,
            1 point per row.
        prec (int, optional): Precision to round distances (i.e., number of
            decimals)

            Defaults to 4.

    Returns:
        NDArray[float]: 1D array with computed distances between each pair of
            points.
    """
    assert pts1.size == pts2.size, "Point arrays must have the same size."
    return np.round(np.linalg.norm(pts2 - pts1, axis=1), prec)


def distance(
    pt1: npt.NDArray[np.float64] | Sequence[float],
    pt2: npt.NDArray[np.float64] | Sequence[float],
    prec: int = 4,
) -> float:
    """Distance between pt1 and pt2.

    Args:
        pt1 (npt.NDArray[np.float64] | Sequence[float]): Coordinates of
            the first point.
        pt2 (npt.NDArray[np.float64] | Sequence[float]): Coordinates of
            the second point.
        prec (int): Precision to round distances (i.e., number of decimals)

    Returns:
        float: Distance between the 2 points.

    """
    pt1Array: npt.NDArray[np.float64] = np.array(pt1)
    pt2Array: npt.NDArray[np.float64] = np.array(pt2)

    dim: int = min(pt1Array.size, pt2Array.size)
    d: float = float(
        np.linalg.norm(pt2Array[:dim] - pt1Array[:dim]).astype(float)
    )
    return round(d, prec)


# TODO: add unit test
def orthogonal_distance(
    pt: npt.NDArray[np.float64],
    seg_pt1: npt.NDArray[np.float64],
    seg_pt2: npt.NDArray[np.float64],
    prec: int = 4,
) -> float:
    """Orthogonal distance between pt and its projection on segment.

    Args:
        pt (npt.NDArray[np.float64] | Sequence[float]): Coordinates of
            the point.
        seg_pt1 (npt.NDArray[np.float64] | Sequence[float]): Coordinates of
            the first point of the segment.
        seg_pt2 (npt.NDArray[np.float64] | Sequence[float]): Coordinates of
            the second point of the segment.
        prec (int): Precision to round distances (i.e., number of decimals)

    Returns:
        float: Distance between the 2 points.

    """
    pt_proj: npt.NDArray[np.float64] = project_orthogonal(pt, seg_pt1, seg_pt2)
    return distance(pt_proj, pt, prec)


def perp(vec: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Compute the orthogonal vector to input.

    Args:
        vec (NDArray[float]): Coordinates of the vector.

    Returns:
        NDArray[float]: Coordinates of the orthogonal vector.

    """
    vec_new: npt.NDArray[np.float64] = np.empty_like(vec)
    vec_new[0], vec_new[1] = -vec[1], vec[0]
    return vec_new


def normal(vec: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Compute the normalized orthogonal vector to nput.

    Args:
        vec (NDArray[float]): Coordinates of the vector.

    Returns:
        NDArray[float]: Coordinates of the normalized orthogonal vector.

    """
    normal_vec = perp(vec)
    normal_vec /= np.linalg.norm(normal_vec)
    return normal_vec


def seg_intersect(
    pt11: npt.NDArray[np.float64],
    pt12: npt.NDArray[np.float64],
    pt21: npt.NDArray[np.float64],
    pt22: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Compute the intersection point to the segments (pt11,pt12), (pt21,pt22).

    Args:
        pt11 (NDArray[float]): Coordinates of the 1st point of the first line
        pt12 (NDArray[float]): Coordinates of the 2nd point of the first line
        pt21 (NDArray[float]): Coordinates of the 1st point of the second line
        pt22 (NDArray[float]): Coordinates of the 2nd second of the second line

    Returns:
        NDArray[float]: Coordinates of the intersection point.

    """
    da: npt.NDArray[np.float64] = pt12 - pt11
    db: npt.NDArray[np.float64] = pt22 - pt21
    dp: npt.NDArray[np.float64] = pt11 - pt21
    dap: npt.NDArray[np.float64] = perp(da)
    denom: float = np.dot(dap, db).astype(float)
    num: float = np.dot(dap, dp).astype(float)
    assert denom != 0, "No intersection between the two lines."
    return (num / denom) * db + pt21


def project_orthogonal(
    pt: npt.NDArray[np.float64],
    pt1: npt.NDArray[np.float64],
    pt2: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Compute the point, image of pt projected on the vector vec=(pt2-pt1).

    Args:
        pt (NDArray[float]): Coordinates of the point to project.
        pt1 (NDArray[float]): Coordinates of the first point of the line.
        pt2 (NDArray[float]): Coordinates of the second point of the line.

    Returns:
        NDArray[float]: Coordinates of the projected point.

    """
    vec: npt.NDArray[np.float64] = pt2 - pt1
    d: float = np.linalg.norm(vec).astype(float)
    k: float = 0.0
    if d > 1e-6:
        k = np.dot(vec, pt - pt1) / d**2
    return compute_colinear(pt1, pt2, k)


# not used anymore
def project_point(
    pt_new0: npt.NDArray[np.float64],
    pt_new1: npt.NDArray[np.float64],
    pt_new2: npt.NDArray[np.float64],
    pt0: npt.NDArray[np.float64],
    pt1: npt.NDArray[np.float64],
    pt2: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], int]:
    """Compute the coordinates of the projected point of pt_new1.

    Args:
        pt_new0 (NDArray[float]): Coordinates of the first point of the line.
        pt_new1 (NDArray[float]): Coordinates of the point to project.
        pt_new2 (NDArray[float]): Coordinates of the second point of the line.
        pt0 (NDArray[float]): Coordinates of the first point of the line
            (pt0, pt1).
        pt1 (NDArray[float]): Coordinates of the common point of the 2 lines
            (pt0, pt1) and (pt2, pt1).
        pt2 (NDArray[float]): Coordinates of the first point of the line
            (pt2, pt1).

    Returns:
        tuple[npt.NDArray[np.float64], int]: tuple containing the coordinates
            of the projected point and an int if the closest point is before
            (-1) or after (+1).

    """
    pt_proj: npt.NDArray[np.float64] = np.copy(pt1)
    j2: int = 0

    try:
        # vector along which to project pt_new1
        pt_new12: npt.NDArray[np.float64] = pt_new2 - pt_new0
        pt_new12 = pt_new1 + perp(pt_new12)

        # projection onto the segment pt0, pt1
        pt_proj0: npt.NDArray[np.float64] = seg_intersect(
            pt_new1, pt_new12, pt0, pt1
        )
        # projection onto the segment pt2, pt1
        pt_proj2: npt.NDArray[np.float64] = seg_intersect(
            pt_new1, pt_new12, pt2, pt1
        )

        # keep the closest pojected point when they exist
        if (pt_proj0 is None) & (pt_proj2 is None):
            raise AssertionError("Both projected points are undefined.")

        if pt_proj0 is None:
            j2 = -1  # type: ignore[unreachable]
            pt_proj = pt_proj2
        elif pt_proj2 is None:
            j2 = 1  # type: ignore[unreachable]
            pt_proj = pt_proj0
        else:
            d: float = distance(pt_new1, pt_proj0) - distance(
                pt_new1, pt_proj2
            )
            if d < 0:
                j2 = -1
                pt_proj = pt_proj0
            else:
                j2 = 1
                pt_proj = pt_proj2

        # if (pt_proj0 is not None) and (pt_proj2 is not None):
        #     d: float = distance(pt_new1, pt_proj0) - distance(
        #         pt_new1, pt_proj2
        #     )
        #     if d < 0:
        #         j2 = -1
        #         pt_proj = pt_proj0
        #     else:
        #         j2 = 1
        #         pt_proj = pt_proj2
        # elif pt_proj2 is not None:
        #     j2 = -1
        #     pt_proj = pt_proj0
        # elif pt_proj0 is not None:
        #     j2 = 1
        #     pt_proj = pt_proj2
        # else:
        #     raise AssertionError("No projection")
    except AssertionError:
        logger.error(
            "Error when projecting the point to the former centerline"
        )
    return pt_proj, j2


def resample_path(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    nb_pts: int = 0,
    s: float = 0,
) -> npt.NDArray[np.float64]:
    """Resample coordinates with nb_pts points according to spline function.

    Args:
        x (NDArray[float]): x coordinates
        y (NDArray[float]): y coordinates
        nb_pts (int, optional): Number of points to return. If nb_pts equals 0,
            return (x,y) points without resampling.

            Defaults to 0.
        s (float, optional): smoothing parameter of B-spline interpolation

            Defaults to 0

    Returns:
        NDArray[float] | tuple[NDArray[float], NDArray[float]]: Coordinates
            of the new points.

    """
    assert x.size == y.size, "x and y must have the same size."
    if x.size < 3:
        logger.warning("Too few number of points. No resampling is applied.")
        return np.column_stack((x, y))

    k = min(x.size - 1, 3)
    tck, u = splprep([x, y], s=s, k=k)
    if nb_pts:
        u = np.linspace(0.0, 1.0, nb_pts)
    return splev(u, tck)


def find_2_closest_points_multi_proc(
    dataset1: pd.DataFrame,
    dataset2: pd.DataFrame,
    x_prop: str = "X",
    y_prop: str = "Y",
    nb_procs: int = 1,
) -> pd.DataFrame:
    """Find the 2 closest points from dataset1 in dataset2 using multiproc.

    Args:
        dataset1 (DataFrame): DataFrame containing x,y coordinates
        dataset2 (DataFrame): DataFrame containing x,y coordinates where to
            find the closest points.
        x_prop (str, optional): Column name of x coordinate.

            Defaults to "X".
        y_prop (str, optional): Column name of y coordinate.

            Defaults to "Y".
        nb_procs (int, optional): Number of processor to use.

            Defaults to 1.

    Returns:
        DataFrame: DataFrame of size (dataset1.shape[0], 4) where columns are:
            - index1: index of the closest (first) point in dataset2
            - index2: index of the second closest point in dataset2
            - d1: distance to the closest point
            - d2: distance to the second closest point

    """
    columns: tuple[str, str, str, str] = ("index1", "index2", "d1", "d2")
    result: pd.DataFrame = pd.DataFrame(
        np.nan * np.ones((dataset1.shape[0], 4)), columns=columns
    )

    with Pool(processes=nb_procs) as pool:
        partial_find_2_closest_points = functools.partial(
            find_2_closest_points, dataset2, x_prop, y_prop, 0
        )
        inputs = [
            np.array((row[x_prop], row[y_prop]))
            for _, row in dataset1.iterrows()
        ]
        outputs = pool.map(partial_find_2_closest_points, inputs)
        for i, (j1, j2, d1, d2) in enumerate(outputs):
            result.loc[i, "index1"] = j1  # type: ignore
            result.loc[i, "index2"] = j2  # type: ignore
            result.loc[i, "d1"] = d1  # type: ignore
            result.loc[i, "d2"] = d2  # type: ignore

    return result


def find_2_closest_points_mono_proc(
    dataset1: pd.DataFrame,
    dataset2: pd.DataFrame,
    x_prop: str = "X",
    y_prop: str = "Y",
) -> pd.DataFrame:
    """Find the 2 closest points from dataset1 in dataset2 using monoproc.

    Args:
        dataset1 (DataFrame): DataFrame containing x,y coordinates
        dataset2 (DataFrame): DataFrame containing x,y coordinates where to
            find the closest points.
        x_prop (str, optional): Column name of x coordinate.

            Defaults to "X".
        y_prop (str, optional): Column name of y coordinate.

            Defaults to "Y".

    Returns:
        DataFrame: DataFrame of size (dataset1.shape[0], 4) where columns are:
            - index1: index of the closest (first) point in dataset2
            - index2: index of the second closest point in dataset2
            - d1: distance to the closest point
            - d2: distance to the second closest point

    """
    columns: tuple[str, str, str, str] = ("index1", "index2", "d1", "d2")
    result: pd.DataFrame = pd.DataFrame(
        np.nan * np.ones((dataset1.shape[0], 4)), columns=columns
    )
    # 1. find the closest point in dataset
    j1: int = 0  # index of closest point from pt_new in dataset2
    for i, row_new in dataset1.iterrows():
        pt_new: npt.NDArray[np.float64] = np.array(
            (row_new[x_prop], row_new[y_prop])
        )
        (j1, j2, d1, d2) = find_2_closest_points(
            dataset2, x_prop, y_prop, j1, pt_new
        )
        result.loc[i, "index1"] = j1  # type: ignore
        result.loc[i, "index2"] = j2  # type: ignore
        result.loc[i, "d1"] = d1  # type: ignore
        result.loc[i, "d2"] = d2  # type: ignore
    return result


def find_2_closest_points(
    dataset2: pd.DataFrame,
    x_prop: str,
    y_prop: str,
    j1: int,
    pt_new: npt.NDArray[np.float64],
) -> tuple[int, int, float, float]:
    """Find the 2 closest points from dataset1 in dataset2.

    Args:
        dataset2 (DataFrame): DataFrame containing x,y coordinates where to
            find the closest points.
        x_prop (str, optional): Column name of x coordinate.

            Defaults to "X".
        y_prop (str, optional): Column name of y coordinate.

            Defaults to "Y".
        j1 (int): index of previous found point for optimization.
        pt_new (NDArray[float]): Reference points from which to compute the
            distances.

    Returns:
        tuple[int, int, float, float]: tuple containing the following values:
            - index1: index of the closest (first) point in dataset2
            - index2: index of the second closest point in dataset2
            - d1: distance to the closest point
            - d2: distance to the second closest point

    """
    d1: float = np.inf  # minimum distance
    d_prev: float = np.inf
    for j, row in dataset2.iterrows():
        # optimization and prevent to find a point ahead of another point
        # already found
        if j < j1:  # type: ignore
            continue

        pt0: npt.NDArray[np.float64] = np.array((row[x_prop], row[y_prop]))
        d: float = distance(pt0, pt_new)

        # optimization: stop when distance between new and old points increase
        if d > d_prev:
            break

        if d < d1:
            d1 = d
            j1 = j  # type: ignore

        d_prev = d

    # 2. find the second neighbor point (the one before or after the closest)
    # case where points are superposed
    pt_next: npt.NDArray[np.float64] = np.empty(0).astype(float)
    pt_prev: npt.NDArray[np.float64] = np.empty(0).astype(float)
    d_next: float = 0.0
    if d1 < 1e-6:
        j2 = j1
        d2 = 0.0
    # case where j1 == 0 or j1 == dataset2.shape[0]-1
    elif j1 == 0:
        pt_next = np.array(
            (dataset2[x_prop][j1 + 1], dataset2[y_prop][j1 + 1])
        )
        d2 = distance(pt_next, pt_new)
        j2 = j1 + 1
    elif j1 == dataset2.shape[0] - 1:
        pt_prev = np.array(
            (dataset2[x_prop][j1 - 1], dataset2[y_prop][j1 - 1])
        )
        d2 = distance(pt_prev, pt_new)
        j2 = j1 - 1
    else:
        pt_prev = np.array(
            (dataset2[x_prop][j1 - 1], dataset2[y_prop][j1 - 1])
        )
        pt_next = np.array(
            (dataset2[x_prop][j1 + 1], dataset2[y_prop][j1 + 1])
        )
        d_prev = distance(pt_prev, pt_new)
        d_next = distance(pt_next, pt_new)
        if d_prev < d_next:
            d2 = d_prev
            j2 = j1 - 1
        else:
            d2 = d_next
            j2 = j1 + 1
    return j1, j2, d1, d2


def find_inflection_points(
    curvature: npt.NDArray[np.float64], lag: int
) -> npt.NDArray[np.int64]:
    r"""Find inflection points from curvature array.

    Inflection points are determine such as the curvature change of sign. A
    given point at index i is an inflection point if the following condition
    is met: $sign(C_{i-1}+C_{i}) != sign(C_{i}+C_{i+1})$.

    Args:
        curvature(NDArray[np.float64]): List of inflection point indexes.
        lag (int): number of points between two consecutive inflection points

    Returns:
        npt.NDArray[np.int64]: inflection point indices

    """
    assert np.all(np.isfinite(curvature)), (
        "No data value in smoothed curvature. First compute smoothed "
        + "curvature before inflection points"
    )

    # duplicate first and last values for calculation
    curvature1: npt.NDArray[np.float64] = np.zeros(curvature.size + 2)
    curvature1[1:-1] = curvature
    curvature1[0], curvature1[-1] = curvature1[1], curvature1[-2]

    # get C'_{i}=C_{i-1}+C_{i}
    curvature2: npt.NDArray[np.float64] = np.zeros_like(curvature1)
    curvature2[:-1] = curvature1[:-1] + curvature1[1:]
    curvature2[-1] = curvature2[-2]
    # sign(C'_{i}) != sign(C'_{i+1})
    inflex_pts: npt.NDArray[np.int64] = np.argwhere(
        (curvature2[:-1] * curvature2[1:]) <= 0
    ).flatten("A")
    return filter_consecutive_indices(inflex_pts, lag)


def find_inflection_points_from_peaks(
    curvature: npt.NDArray[np.float64], curv_threshold: float = 0.1
) -> npt.NDArray[np.int64]:
    r"""Find inflection points from curvature array.

    Inflection points are determine such as the opposite of absolute values
    of curvature reach local maxima using scipy.signal.find_peaks function.

    Args:
        curvature(NDArray[np.float64]): curvature of each point.
        curv_threshold (float): curvature threshold for peak detection.

            Defaults to 0.001.

    Returns:
        npt.NDArray[np.int64]: inflection point indices

    """
    assert np.all(np.isfinite(curvature)), (
        "No data value in smoothed curvature. First compute smoothed "
        + "curvature before inflection points"
    )

    # take the opposite of the absolute values of curvature for
    # 0 curvature being a peak
    curv1 = np.abs(curvature)
    # normalize curvature values between 0 and 1
    curv1 /= -1 * np.max(curv1)
    # use threshold to avoid division by 0
    curv1[curv1 > -1e-6] = -1e-6

    # normalizes curvature must be < 0.001
    peak_indexes, _ = find_peaks(curv1, height=(-1.0 * curv_threshold, 0))
    return peak_indexes


def filter_consecutive_indices(
    values: npt.NDArray[np.int64], lag: int
) -> npt.NDArray[np.int64]:
    """Filter consecutive indices.

    Args:
        values (npt.NDArray[np.int64]): indices to filter
        lag (int): lag between 2 consecutive indices

    Returns:
        npt.NDArray[np.int64]: filtered indices
    """
    # Compute differences between consecutive values
    diffs: npt.NDArray[np.int64] = np.diff(values)

    # find indices where differences > lag
    non_consecutive_indices0: npt.NDArray[np.int64] = (
        np.where(diffs > lag)[0] + 1
    )

    # Add first and last index of values list
    non_consecutive_indices: list[int] = (
        [0] + non_consecutive_indices0.tolist() + [len(values)]
    )

    # get filtered values
    filtered_values: list[int] = []
    for i in range(len(non_consecutive_indices) - 1):
        start: int = non_consecutive_indices[i]
        end: int = non_consecutive_indices[i + 1]
        # >2 consecutive values
        if end - start > 2:
            # keep the middle value (the one before if even number of elements)
            middle_index = start + int(end - start - 0.5) // 2
            filtered_values.append(values[middle_index])
        # 2 consecutive values
        else:
            filtered_values.append(values[start])
    return np.array(sorted(set(filtered_values)))


# TODO: add unit test
def compute_curvature(XY: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Compute curvature of an ensemble of points.

    Args:
        XY (npt.NDArray[np.float64]): 2D array with XY coordinates.

    Returns:
        npt.NDArray[np.float64]: curvature at each points

    """
    assert XY.shape[1] == 2, "XY coordinates must 2D array with 2 columns."
    curv: npt.NDArray[np.float64] = np.full(XY.shape[0], np.nan)
    for i in range(1, XY.shape[0] - 1, 1):
        curv[i] = compute_curvature_at_point(XY[i - 1], XY[i], XY[i + 1])
    # copy curvature from neighbors at end points
    curv[0], curv[-1] = curv[1], curv[-2]
    return curv


def compute_curvature_at_point(
    pt1: npt.NDArray[np.float64],
    pt2: npt.NDArray[np.float64],
    pt3: npt.NDArray[np.float64],
) -> float:
    """Compute curvature from 3 points.

    Args:
        pt1 (NDArray[float]): Coordinates of the first point.
        pt2 (NDArray[float]): Coordinates of the second point.
        pt3 (NDArray[float]): Coordinates of the third point.

    Returns:
        float: Value of the curvature.

    """
    x1, y1 = pt1
    x2, y2 = pt2
    x3, y3 = pt3

    ds12: float = distance(pt1, pt2)
    ds23: float = distance(pt2, pt3)
    ds13: float = ds12 + ds23

    dxds: float = (x3 - x1) / (ds13)
    dyds: float = (y3 - y1) / (ds13)

    d2xds2: float = (ds12 * (x3 - x2) - ds23 * (x2 - x1)) / (
        ds12 * ds23 * ds13
    )
    d2yds2: float = (ds12 * (y3 - y2) - ds23 * (y2 - y1)) / (
        ds12 * ds23 * ds13
    )
    return -(dxds * d2yds2 - dyds * d2xds2) / pow(dxds**2 + dyds**2, 3.0 / 2.0)


# not used
def compute_curvature_at_point_Menger(
    pt1: npt.NDArray[np.float64],
    pt2: npt.NDArray[np.float64],
    pt3: npt.NDArray[np.float64],
) -> float:
    """Compute curvature from 3 points according to Menger formula.

    Args:
        pt1 (NDArray[float]): Coordinates of the first point.
        pt2 (NDArray[float]): Coordinates of the second point.
        pt3 (NDArray[float]): Coordinates of the third point.

    Returns:
        float: Absolute value of the curvature.

    """
    # Calculating length of all three sides
    len_side_1: float = math.dist(pt1, pt2)
    len_side_2: float = math.dist(pt2, pt3)
    len_side_3: float = math.dist(pt1, pt3)

    # sp is semi-perimeter
    sp: float = (len_side_1 + len_side_2 + len_side_3) / 2

    # Calculating area using Herons formula
    area: float = np.sqrt(
        sp * (sp - len_side_1) * (sp - len_side_2) * (sp - len_side_3)
    )

    # Calculating curvature using Menger curvature formula
    return (4 * area) / (len_side_1 * len_side_2 * len_side_3)


# not used # TODO: create unit test
def compute_curvature_at_point_flumy(
    pt1: npt.NDArray[np.float64],
    pt2: npt.NDArray[np.float64],
    pt3: npt.NDArray[np.float64],
) -> float:
    """Compute the curvature according to the formula used in Flumy.

    Args:
        pt1 (NDArray[float]): Coordinates of the first point.
        pt2 (NDArray[float]): Coordinates of the second point.
        pt3 (NDArray[float]): Coordinates of the third point.

    Returns:
        float: Value of the curvature.

    """
    x1, y1 = pt1
    x2, y2 = pt2
    x3, y3 = pt3

    M1M2: tuple[float, float] = (x2 - x1, y2 - y1)
    M2M3: tuple[float, float] = (x3 - x2, y3 - y2)

    M1sq: float = x1 * x1 + y1 * y1
    M2sq: float = x2 * x2 + y2 * y2
    M3sq: float = x3 * x3 + y3 * y3

    curvature: float = 0.0

    # Compute curvature -- begin
    det: float = M1M2[0] * M2M3[1] - M1M2[1] * M2M3[0]

    # avoid a division by 0 when the distance between the 2 last points is very
    # small
    if abs(det) > 1e-6:
        a: float = M2sq - M3sq
        xc: float = -a * y1
        yc: float = a * x1
        a = M3sq - M1sq
        xc -= a * y2
        yc += a * x2
        a = M1sq - M2sq
        xc -= a * y3
        yc += a * x3
        xc *= 0.5
        xc /= det
        yc *= 0.5
        yc /= det
        n: tuple[float, float] = (x2 - xc, y2 - yc)
        curvature = 1.0 / math.sqrt(n[0] * n[0] + n[1] * n[1])
        # Clockwise orientation for curvature
        if det > 0:
            curvature *= -1.0
    return curvature


# TODO: add unit test
def compute_median_curvature_index(
    curvature: npt.NDArray[np.float64], n: float
) -> int:
    """Get median abscissa using curvature distribution as weighting function.

    Args:
        curvature (npt.NDArray[np.float64]): curvature array
        n (float): exponent value

    Returns:
        int: index of median curvature

    """
    curvature1: npt.NDArray[np.float64] = np.abs(curvature)
    cumsum: npt.NDArray[np.float64] = np.cumsum(curvature1**n) / np.sum(
        curvature1**n
    )
    return np.argwhere(cumsum > 0.5).flatten("A")[0]


def compute_esperance(
    curvature: npt.NDArray[np.float64],
    curv_abscissa: npt.NDArray[np.float64],
    n: float,
) -> float:
    """Get average abscissa using curvature distribution as weighting function.

    Args:
        curvature (npt.NDArray[np.float64]): curvature distrution function
        curv_abscissa (npt.NDArray[np.float64]): curvilinear abscissa
        n (float): exponent

    Returns:
        float: average abscissa.
    """
    curvature1: npt.NDArray[np.float64] = np.abs(curvature)
    mean: float = float(
        np.sum(curv_abscissa * curvature1**n) / np.sum(curvature1**n)
    )
    return mean


def compute_variance(
    curvature: npt.NDArray[np.float64],
    curv_abscissa: npt.NDArray[np.float64],
    n: float,
) -> tuple[float, float]:
    """Get variance abscissa from curvature distribution as weighting function.

    Args:
        curvature (npt.NDArray[np.float64]): curvature distrution function
        curv_abscissa (npt.NDArray[np.float64]): curvilinear abscissa
        n (float): exponent

    Returns:
        tuple[float, float]: tuple containing the variance and std deviation.
    """
    mean = compute_esperance(curvature, curv_abscissa, n)
    abs2 = (curv_abscissa - mean) ** 2
    var = compute_esperance(curvature, abs2, n)
    return var, float(np.sqrt(var))


def compute_skewness(
    curvature: npt.NDArray[np.float64],
    curv_abscissa: npt.NDArray[np.float64],
    n: float,
) -> float:
    """Compute Pearson's skewness coeff of curvature distribution function.

    Args:
        curvature (npt.NDArray[np.float64]): curvature distrution function
        curv_abscissa (npt.NDArray[np.float64]): curvilinear abscissa
        n (float): exponent

    Returns:
        float: skewness coefficient.
    """
    mean = compute_esperance(curvature, curv_abscissa, n)
    var, std_dev = compute_variance(curvature, curv_abscissa, n)
    abs2 = ((curv_abscissa - mean) / std_dev) ** 3
    return float(compute_esperance(curvature, abs2, n))


def compute_kurtosis(
    curvature: npt.NDArray[np.float64],
    curv_abscissa: npt.NDArray[np.float64],
    n: float,
) -> float:
    """Compute the kurtosis coefficient of curvature distribution function.

    Args:
        curvature (npt.NDArray[np.float64]): curvature distrution function
        curv_abscissa (npt.NDArray[np.float64]): curvilinear abscissa
        n (float): exponent

    Returns:
        float: kurtosis coefficient.
    """
    mean = compute_esperance(curvature, curv_abscissa, n)
    var, std_dev = compute_variance(curvature, curv_abscissa, n)
    abs2 = ((curv_abscissa - mean) / std_dev) ** 4
    return float(compute_esperance(curvature, abs2, n))


# TODO: create unit test
def get_MP(
    dir_trans: npt.NDArray[np.float64] = np.array((1.0, 0.0)),
    ref: npt.NDArray[np.float64] = np.array((1.0, 0.0)),
) -> npt.NDArray[np.float64]:
    """Get the rotation 2D matrix between ref and dir_trans.

    Args:
        dir_trans (NDArray[float], optional): Direction.

            Defaults to np.array((1., 0.)).
        ref (NDArray[float], optional): Reference direction.

            Defaults to np.array((1., 0.)).

    Returns:
        NDArray[float]: Array corresponding to rotation 2D matrix.

    """
    dir_trans_norm: npt.NDArray[np.float64] = dir_trans / np.linalg.norm(
        dir_trans
    )
    ref_norm: npt.NDArray[np.float64] = ref / np.linalg.norm(ref)
    if np.dot(dir_trans_norm, ref) < 0.0:
        dir_trans_norm *= -1.0

    cos: float = float(np.dot(dir_trans_norm, ref_norm))
    teta: float = float(np.arccos(cos))
    det: float = float(np.linalg.norm((dir_trans_norm, ref_norm)))
    if det < 0:
        teta = np.pi - teta
    sin: float = np.sin(teta)
    return np.array([[cos, sin], [-sin, cos]])


# TODO: create unit test
def compute_point_displacements(
    l_pt: list[npt.NDArray[np.float64]],
    dir_trans: npt.NDArray[np.float64] = np.array((1.0, 0.0)),
    ref: npt.NDArray[np.float64] = np.array((1.0, 0.0)),
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute the displacements of a serie of points.

    Args:
        l_pt (list[NDArray[float]]: List of point coordinates.
        dir_trans (NDArray[float], optional): Direction.

            Defaults to np.array((1., 0.)).
        ref (NDArray[float], optional): Reference direction.

            Defaults to np.array((1., 0.)).

    Returns:
        tuple[NDArray[float], NDArray[float]]: tuple containing:
            - Displacement in-between each successive points of the serie.
            - Displacement between first and last points of the serie.

    """
    # compute change-of-basis matrix
    MP: npt.NDArray[np.float64] = get_MP(dir_trans, ref)

    # compute displacement
    # dX, dY, dZ, dMig
    local_disp: npt.NDArray[np.float64] = np.nan * np.zeros((len(l_pt) - 1, 4))
    # deltaX, deltaY, deltaZ, deltaMig
    whole_disp: npt.NDArray[np.float64] = np.nan * np.zeros(4)

    # compute incremental displacements
    pt1: npt.NDArray[np.float64] = l_pt[0]
    disp: npt.NDArray[np.float64]
    disp2: npt.NDArray[np.float64]
    for i, pt2 in enumerate(l_pt):
        if i > 0:
            disp = pt2 - pt1
            disp2 = np.dot(MP, disp)

            local_disp[i - 1, 0] = disp2[0]
            local_disp[i - 1, 1] = disp2[1]
            if len(pt1) > 2:
                local_disp[i - 1, 2] = pt2[2] - pt1[2]
            else:
                local_disp[i - 1, 2] = 0
            local_disp[i - 1, 3] = np.linalg.norm(disp2)

        pt1 = pt2

    # compute global displacements
    pt00: npt.NDArray[np.float64] = l_pt[0]
    pt01: npt.NDArray[np.float64] = l_pt[-1]

    disp = pt01 - pt00
    disp2 = np.dot(MP, disp)
    whole_disp[0] = disp2[0]
    whole_disp[1] = disp2[1]

    if len(pt01) > 2:
        whole_disp[2] = pt01[2] - pt00[2]
    else:
        whole_disp[2] = 0
    whole_disp[3] = np.linalg.norm(disp2)
    return local_disp, whole_disp


def barycenter(l_val: list[float], l_pond: list[float]) -> float:
    """Compute the weighted average of values in l_val.

    Args:
        l_val (list[float]): List of values to compute the mean.
        l_pond (list[float]): List of weights for each value of l_val.

    Returns:
        float: weighted mean.

    """
    assert len(l_val) == len(l_pond), (
        "The length of the lists of values and weighting coefficients "
        + "must be the same to compute the barycenter"
    )

    mean: float = 0.0
    for val, pond in zip(l_val, l_pond, strict=False):
        mean += val * pond
    return mean / sum(l_pond)


def sort_key(labels: list[str], reverse: bool = False) -> list[str]:
    """Sort the labels.

    Args:
        labels (list[str]): List of labels that can be cast to int/float values
        reverse (bool, optional): if True, sorting is descending.

            Defaults to False.

    Returns:
        labels2 (list[str]): List of sorted labels.

    """
    labels_int = [eval(val) for val in labels]
    labels_int.sort(reverse=reverse)
    labels2 = [str(val) for val in labels_int]
    return labels2


def get_keys_from_to(
    all_keys: list[str],
    key_min: int = 0,
    key_max: int = 999999,
    sort_reverse: bool = False,
) -> list[str]:
    """Extract keys from key_min to key_max from the list all_keys.

    Args:
        all_keys (list[str]): List of keys that can be cast to int values.
        key_min (int, optional): Minimum key.

            Defaults to 0.
        key_max (int, optional): Maximum key.

            Defaults to 999999.
        sort_reverse (bool, optional): If True, sorting is descending.

            Defaults to False.

    Returns:
        list: List of extracted keys.

    """
    lkeys: list[str] = []
    for key in all_keys:
        if int(key) <= int(key_max) and int(key) >= int(key_min):
            lkeys += [key]

    if len(lkeys) > 1:
        lkeys = sort_key(lkeys, sort_reverse)

    return [str(key) for key in lkeys]
