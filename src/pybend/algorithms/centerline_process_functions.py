# SPDX-FileCopyrightText: Copyright 2025 Martin Lemay <martin.lemay@mines-paris.org>
# SPDX-FileContributor: Martin Lemay
# ruff: noqa: E402 # disable Module level import not at top of file

"""Algorithms for centerline processing."""

import math

import numpy as np
import numpy.typing as npt
import pandas as pd  # type: ignore[import-untyped]
from scipy.interpolate import splev, splprep  # type: ignore[import-untyped]
from scipy.signal import find_peaks  # type: ignore[import-untyped]
from shapely.geometry import Polygon  # type: ignore[import-untyped]
from shapely.validation import make_valid  # type: ignore[import-untyped]

import pybend.algorithms.geometry_functions as geom
from pybend.model.ClPoint import ClPoint
from pybend.model.enumerations import BendSide
from pybend.utils.logging import logger


def clpoints2coords(cl_pts: list[ClPoint]) -> npt.NDArray[np.float64]:
    """Transform a list of ClPoint into a numpy array (1 point per row).

    Args:
        cl_pts (list[ClPoint]): List of ClPoint.

    Returns:
        NDArray[float]: Array of point coordinates.

    """
    return np.array([cl_pt.pt for cl_pt in cl_pts])


def resample_path(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    nb_pts: int = 0,
    s: float | None = 0,
    u: npt.NDArray[np.float64] | None = None,
) -> npt.NDArray[np.float64]:
    """Resample coordinates with nb_pts points according to spline function.

    Args:
        x (NDArray[float]): x coordinates
        y (NDArray[float]): y coordinates
        nb_pts (int, optional): Number of points to return.
            If nb_pts equals 0, return (x,y) points without resampling.
            Defaults to 0.
        s (float | None, optional): Smoothing parameter of B-spline
            interpolation. 0 means interpolation through all points, None
            uses scipy default (m - sqrt(2m), m being the number of points).
            Defaults to 0.
        u (NDArray[float] | None, optional): Strictly increasing parameter
            values of the points (e.g., ages). If None, the normalized
            cumulative distance along the path is used.
            Defaults to None.

    Returns:
        NDArray[float]: Array of shape (2, N) with x coordinates in the first
        row and y coordinates in the second row.

    """
    assert x.size == y.size, "x and y must have the same size."
    if x.size < 3:
        logger.warning("Too few number of points. No resampling is applied.")
        return np.vstack((x, y)).astype(float)

    k = min(x.size - 1, 3)
    tck, u_out = splprep([x, y], u=u, s=s, k=k)
    if nb_pts:
        u_out = np.linspace(u_out[0], u_out[-1], nb_pts)
    return np.asarray(splev(u_out, tck), dtype=float)


def compute_cuvilinear_abscissa(
    XY: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Compute curvilinear abscissa from cartesian XY coordinates.

    Args:
        XY (NDArray[float]): 2D array with XY coordinates.

    Returns:
        NDArray[float]: Array of curvilinear abscissa values.

    """
    ds = geom.distance_arrays(XY[:-1], XY[1:], 4)
    return np.append([0], np.cumsum(ds))


def find_2_closest_points_multi_proc(
    dataset1: pd.DataFrame,
    dataset2: pd.DataFrame,
    x_prop: str = "X",
    y_prop: str = "Y",
    nb_procs: int = 1,
) -> pd.DataFrame:
    """Find the 2 closest points from dataset1 in dataset2.

    Points of dataset1 are tracked sequentially along dataset2 (the search
    for a point starts from the match of the previous point), so this
    sequential search cannot be split between processes. The function
    therefore gives the same result as find_2_closest_points_mono_proc,
    whose search is vectorized. The nb_procs argument is kept for backward
    compatibility.

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
    return find_2_closest_points_mono_proc(dataset1, dataset2, x_prop, y_prop)


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
    values: npt.NDArray[np.float64] = np.full((dataset1.shape[0], 4), np.nan)
    pts_new: npt.NDArray[np.float64] = dataset1[[x_prop, y_prop]].to_numpy(
        dtype=float
    )
    labels2 = dataset2.index
    # j1 is the position of the closest point in dataset2 of the previous
    # point, from which the search starts
    j1: int = 0
    for i, pt_new in enumerate(pts_new):
        (j1, j2, d1, d2) = find_2_closest_points(
            dataset2, x_prop, y_prop, j1, pt_new
        )
        # return dataset2 labels (identical to positions for a RangeIndex)
        values[i] = (labels2[j1], labels2[j2], d1, d2)
    return pd.DataFrame(values, columns=columns)


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
        j1 (int): position of previous found point. The search starts from
            this point and walks forward along dataset2 while the distance
            decreases (local minimum), which tracks points along the path.
        pt_new (NDArray[float]): Reference points from which to compute the
            distances.

    Returns:
        tuple[int, int, float, float]: tuple containing the following values
            (indices are positions in dataset2, not labels):
            - index1: index of the closest (first) point in dataset2
            - index2: index of the second closest point in dataset2
            - d1: distance to the closest point
            - d2: distance to the second closest point

    """
    xy: npt.NDArray[np.float64] = dataset2[[x_prop, y_prop]].to_numpy(
        dtype=float
    )
    nb_pts: int = xy.shape[0]
    pt_ref: npt.NDArray[np.float64] = np.asarray(pt_new, dtype=float)[:2]
    start: int = min(max(int(j1), 0), nb_pts - 1)

    # 1. walk forward from j1 while the distance decreases; distances are
    # computed by windows for efficiency
    d1: float = np.inf
    window: int = 64
    pos: int = start
    found: bool = False
    while pos < nb_pts and not found:
        stop: int = min(pos + window, nb_pts)
        d = np.round(np.linalg.norm(xy[pos:stop] - pt_ref, axis=1), 4)
        for k, dk in enumerate(d):
            if dk > d1:  # distance increases: local minimum reached
                found = True
                break
            if dk < d1:
                d1 = float(dk)
                j1 = pos + k
        pos = stop
    j1 = int(j1) if np.isfinite(d1) else start

    # 2. find the second neighbor point (the one before or after the closest)
    j2: int
    d2: float
    if d1 < 1e-6 or nb_pts == 1:  # superposed points
        j2 = j1
        d2 = 0.0 if d1 < 1e-6 else d1
    elif j1 == 0:
        j2 = 1
        d2 = geom.distance(xy[j2], pt_ref)
    elif j1 == nb_pts - 1:
        j2 = j1 - 1
        d2 = geom.distance(xy[j2], pt_ref)
    else:
        d_prev: float = geom.distance(xy[j1 - 1], pt_ref)
        d_next: float = geom.distance(xy[j1 + 1], pt_ref)
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
    is met: :math:`sign(C_{i-1}+C_{i}) != sign(C_{i}+C_{i+1})`.

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
    # the last tested index is beyond the last point
    inflex_pts = inflex_pts[inflex_pts < curvature.size]
    return filter_consecutive_indices(inflex_pts, lag)


def find_inflection_points_from_peaks(
    curvature: npt.NDArray[np.float64], curv_threshold: float = 0.1
) -> npt.NDArray[np.int64]:
    r"""Find inflection points from curvature array.

    Inflection points are determine such as the opposite of absolute values
    of curvature reach local maxima using scipy.signal.find_peaks function.
    Only minima of absolute curvature where the curvature changes of sign
    are kept, so that a local decrease of curvature within a bend is not
    taken as an inflection point.

    Args:
        curvature(NDArray[np.float64]): curvature of each point.
        curv_threshold (float): threshold on normalized absolute curvature
            (between 0 and 1) for peak detection.
            Defaults to 0.1.

    Returns:
        npt.NDArray[np.int64]: inflection point indices

    """
    assert np.all(np.isfinite(curvature)), (
        "No data value in smoothed curvature. First compute smoothed "
        + "curvature before inflection points"
    )

    curv_abs: npt.NDArray[np.float64] = np.abs(curvature)
    curv_max: float = float(np.max(curv_abs)) if curv_abs.size > 0 else 0.0
    if curv_max == 0.0:
        # straight path: no inflection point
        return np.empty(0, dtype=np.int64)

    # take the opposite of the normalized absolute values of curvature for
    # 0 curvature being a peak (values between -1 and 0)
    curv1 = -curv_abs / curv_max
    # clip values close to 0 so that near-zero plateaus are flat
    curv1[curv1 > -1e-6] = -1e-6

    # normalized absolute curvature must be < curv_threshold
    peak_indexes, _ = find_peaks(curv1, height=(-1.0 * curv_threshold, 0))

    # keep peaks where curvature changes of sign: compare the sign of the
    # closest non-null curvature values before and after the peak
    non_null: npt.NDArray[np.int64] = np.flatnonzero(
        curv_abs > 1e-6 * curv_max
    )
    inflex_pts: list[int] = []
    for index in peak_indexes:
        pos_before: int = int(np.searchsorted(non_null, index, "left")) - 1
        pos_after: int = int(np.searchsorted(non_null, index, "right"))
        if pos_before < 0 or pos_after >= non_null.size:
            continue
        if np.sign(curvature[non_null[pos_before]]) != np.sign(
            curvature[non_null[pos_after]]
        ):
            inflex_pts.append(int(index))
    return np.array(inflex_pts, dtype=np.int64)


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
    if len(values) == 0:
        return np.empty(0, dtype=np.int64)

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


def compute_curvature(XY: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Compute curvature of an ensemble of points.

    Code was modified from https://github.com/zsylvester/curvaturepy/blob/master/cline_analysis.py

    Args:
        XY (npt.NDArray[np.float64]): 2D array with XY coordinates.

    Returns:
        npt.NDArray[np.float64]: curvature at each points

    """
    assert XY.shape[1] == 2, "XY coordinates must 2D array with 2 columns."
    dx = np.gradient(XY[:, 0])
    dy = np.gradient(XY[:, 1])  # first derivatives
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)  # second derivatives
    curvature = (dx * ddy - dy * ddx) / ((dx**2 + dy**2) ** 1.5)  # curvature
    return -1.0 * curvature


def compute_curvature_at_point(
    pt1: npt.NDArray[np.float64],
    pt2: npt.NDArray[np.float64],
    pt3: npt.NDArray[np.float64],
) -> float:
    r"""Compute the signed curvature from 3 points.

    The curvature is the inverse of the radius of the circle passing through
    the 3 points (signed Menger curvature):
    :math:`C = -2 (P_2-P_1) \times (P_3-P_2) / (|P_1P_2||P_2P_3||P_1P_3|)`.
    Curvature is positive when the path turns clockwise (to the right), as
    in compute_curvature. Curvature is 0 if 2 points are superposed.

    Args:
        pt1 (NDArray[float]): Coordinates of the first point.
        pt2 (NDArray[float]): Coordinates of the second point.
        pt3 (NDArray[float]): Coordinates of the third point.

    Returns:
        float: Value of the curvature.

    """
    p1 = np.asarray(pt1, dtype=float)[:2]
    p2 = np.asarray(pt2, dtype=float)[:2]
    p3 = np.asarray(pt3, dtype=float)[:2]
    v12 = p2 - p1
    v23 = p3 - p2
    denom: float = float(
        np.linalg.norm(v12) * np.linalg.norm(v23) * np.linalg.norm(p3 - p1)
    )
    if denom < 1e-12:
        return 0.0
    cross: float = float(v12[0] * v23[1] - v12[1] * v23[0])
    return -2.0 * cross / denom


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
    denom: float = len_side_1 * len_side_2 * len_side_3
    if denom < 1e-12:
        return 0.0

    # triangle area from the cross product (Heron's formula may give
    # negative values under the square root for nearly collinear points)
    area: float = 0.5 * abs(
        (pt2[0] - pt1[0]) * (pt3[1] - pt1[1])
        - (pt2[1] - pt1[1]) * (pt3[0] - pt1[0])
    )

    # Calculating curvature using Menger curvature formula
    return float((4 * area) / denom)


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


def compute_half_angle_variation(
    normal: npt.NDArray[np.float64], shift: int = 0
) -> int:
    """Get the index of half path angle variation.

    The rotation of normal vectors is accumulated (unwrapped) from the first
    to the last point, so that bends turning by more than pi are handled.
    The apex is the first point where the rotation reaches half of the total
    rotation, or pi/2 if the total rotation is greater than pi.

    Args:
        normal (npt.NDArray[np.float64]): normal vectors to channel path,
            array of shape (N, 2)
        shift (int): number of points excluded at each end of the path to
            compute angle deviation

    Returns:
        int: index of the apex

    """
    normal_arr: npt.NDArray[np.float64] = np.asarray(normal, dtype=float)
    assert normal_arr.ndim == 2 and normal_arr.shape[1] == 2, (
        "Normal vectors must be an array of shape (N, 2)."
    )
    normal2 = normal_arr[shift : normal_arr.shape[0] - shift]
    if normal2.shape[0] < 2:
        logger.warning("Too few points to compute half angle variation.")
        return shift

    # signed angle between consecutive normal vectors in [-pi, pi]
    vec1 = normal2[:-1]
    vec2 = normal2[1:]
    cross = vec1[:, 0] * vec2[:, 1] - vec1[:, 1] * vec2[:, 0]
    dot = vec1[:, 0] * vec2[:, 0] + vec1[:, 1] * vec2[:, 1]
    # cumulative angle variation from the first normal vector
    teta: npt.NDArray[np.float64] = np.concatenate(
        ([0.0], np.cumsum(np.arctan2(cross, dot)))
    )
    # use positive angles whatever the rotation direction
    if teta[-1] < 0.0:
        teta = -teta

    # get the total angle variation from first to last point
    d_teta: float = float(teta[-1])
    # compute angle at apex
    teta_apex: float = d_teta / 2.0 if d_teta <= np.pi else np.pi / 2.0
    # find the index of the apex (always exists since teta[-1] >= teta_apex)
    index = int(np.flatnonzero(teta >= teta_apex)[0])
    return index + shift


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
    curvature1: npt.NDArray[np.float64] = np.abs(curvature) ** n
    total: float = float(np.sum(curvature1))
    if not np.isfinite(total) or total == 0.0:
        logger.warning("Null curvature: median index set to middle point.")
        return int(curvature1.size // 2)
    cumsum: npt.NDArray[np.float64] = np.cumsum(curvature1) / total
    return int(np.argwhere(cumsum > 0.5).flatten("A")[0])


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
    """Compute Fisher-Pearson's skewness coeff of curvature distribution.

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
        float: kurtosis coefficient (not excess kurtosis, i.e., 3 for a
        normal distribution).
    """
    mean = compute_esperance(curvature, curv_abscissa, n)
    var, std_dev = compute_variance(curvature, curv_abscissa, n)
    abs2 = ((curv_abscissa - mean) / std_dev) ** 4
    return float(compute_esperance(curvature, abs2, n))


def compute_loop_area_centroid_asymmetry(
    bend_coords: npt.NDArray[np.float64],
) -> float:
    r"""Compute the loop-area centroid asymmetry of a bend.

    The asymmetry is based on the centroid of the region enclosed
    between the bend arc and the chord joining the inflection
    points. The centroid (barycenter) of the bend polygon is
    projected orthogonally onto the chord, and :math:`t_c` is the
    Cartesian distance from the upstream inflection point to that
    projection:

    .. math::

        \alpha = \frac{2\,t_c}{L} - 1

    where *L* is the chord length and :math:`t_c` is signed (negative if
    the projection lies upstream of the upstream inflection point). The
    result ranges from -1 (fully upstream-skewed) to +1 (fully
    downstream-skewed) when the centroid projects within the chord; 0 means
    symmetric. Values beyond [-1, 1] occur for overhanging bends. If the
    bend crosses the chord, the polygon is made valid first and the centroid
    is computed from all enclosed areas.

    Args:
        bend_coords: Point coordinates of the bend (N×2 array),
            ordered from upstream to downstream inflection point.

    Returns:
        The centroid asymmetry coefficient, or *NaN* if the
        polygon is degenerate.

    """
    p_up = bend_coords[0]
    p_down = bend_coords[-1]
    l_chord = float(geom.distance(p_up, p_down))
    if l_chord == 0.0:
        return float("nan")

    # polygon enclosed between the bend arc and the chord
    poly = Polygon(bend_coords[:, :2])
    if not poly.is_valid:
        # self-intersecting polygon when the bend crosses the chord
        poly = make_valid(poly)
    if poly.area == 0.0 or poly.centroid.is_empty:
        return float("nan")

    # signed abscissa of the centroid projected onto the chord
    centroid = np.array(poly.centroid.coords[0])
    chord = np.asarray(p_down, dtype=float)[:2] - np.asarray(p_up, float)[:2]
    t_c = (
        float(np.dot(centroid - np.asarray(p_up, float)[:2], chord)) / l_chord
    )
    asc = 2.0 * t_c / l_chord - 1.0
    return asc


def compute_point_displacements(
    l_pt: list[npt.NDArray[np.float64]],
    dir_trans: npt.NDArray[np.float64] = np.array((1.0, 0.0)),
    ref: npt.NDArray[np.float64] = np.array((1.0, 0.0)),
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute the displacements of a serie of points.

    Args:
        l_pt (list[NDArray[float]]): List of point coordinates.
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
    MP: npt.NDArray[np.float64] = geom.get_MP(dir_trans, ref)

    # compute displacement
    # dX, dY, dZ, dMig
    local_disp: npt.NDArray[np.float64] = np.nan * np.zeros((len(l_pt) - 1, 4))
    # deltaX, deltaY, deltaZ, deltaMig
    whole_disp: npt.NDArray[np.float64] = np.nan * np.zeros(4)

    # compute incremental displacements
    pt1: npt.NDArray[np.float64] = np.array(l_pt[0])
    disp: npt.NDArray[np.float64]
    disp2: npt.NDArray[np.float64]
    for i, pt2 in enumerate(l_pt):
        if i > 0:
            disp = (np.array(pt2) - np.array(pt1))[:2]
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
    pt00: npt.NDArray[np.float64] = np.array(l_pt[0])
    pt01: npt.NDArray[np.float64] = np.array(l_pt[-1])

    disp = (pt01 - pt00)[:2]
    disp2 = np.dot(MP, disp)
    whole_disp[0] = disp2[0]
    whole_disp[1] = disp2[1]

    if len(pt01) > 2:
        whole_disp[2] = pt01[2] - pt00[2]
    else:
        whole_disp[2] = 0
    whole_disp[3] = np.linalg.norm(disp2)
    return local_disp, whole_disp


def sort_key(labels: list[str], reverse: bool = False) -> list[str]:
    """Sort the labels.

    Args:
        labels (list[str]): List of labels that can be cast to int/float values
        reverse (bool, optional): if True, sorting is descending.
            Defaults to False.

    Returns:
        labels2 (list[str]): List of sorted labels.

    """

    def to_number(val: str) -> int | float:
        try:
            return int(val)
        except ValueError:
            return float(val)

    labels_int = [to_number(val) for val in labels]
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
        if float(key_min) <= float(key) <= float(key_max):
            lkeys += [key]

    if len(lkeys) > 1:
        lkeys = sort_key(lkeys, sort_reverse)

    return [str(key) for key in lkeys]


def compute_bend_side_from_curvature(
    curvature: npt.NDArray[np.float64],
    sinuosity: float,
    sinuo_thres: float,
) -> BendSide:
    """Determine bend side from a curvature series.

    Args:
        curvature: Filtered curvature values over the bend span.
        sinuosity: Precomputed sinuosity (arc length / wavelength).
        sinuo_thres: Sinuosity threshold below which the bend
            is considered straight.

    Returns:
        BendSide: UP if curvature sum > 0, DOWN otherwise,
            or STRAIGHT if sinuosity is below the threshold.
    """
    if sinuosity < sinuo_thres:
        return BendSide.STRAIGHT
    curv_sum: float = float(np.sum(curvature))
    return BendSide.UP if curv_sum > 0 else BendSide.DOWN


def compute_bend_apex_from_curvature(
    curvature: npt.NDArray[np.float64],
    n: float,
) -> int:
    """Find apex relative index within a curvature series.

    Args:
        curvature: Absolute curvature values over the bend span.
        n: Exponent for the median curvature computation.

    Returns:
        Relative index (0-based within the series) of the apex.
    """
    return compute_median_curvature_index(np.abs(curvature), n)
