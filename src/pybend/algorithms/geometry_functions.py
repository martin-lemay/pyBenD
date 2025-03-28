# SPDX-FileCopyrightText: Copyright 2025 Martin Lemay <martin.lemay@mines-paris.org>
# SPDX-FileContributor: Martin Lemay
# ruff: noqa: E402 # disable Module level import not at top of file

from typing import Sequence

import numpy as np
import numpy.typing as npt

__doc__ = """
Usefull geometry functions.
"""


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
        vec (2DArray[float]): Coordinates of the vector.

    Returns:
        2DArray[float]: Coordinates of the orthogonal vector.

    """
    vec_new: npt.NDArray[np.float64] = np.empty_like(vec)
    vec_new[0], vec_new[1] = -vec[1], vec[0]
    return vec_new


# TODO: add unit test
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


def get_angle_between_vectors(
    vec1: npt.NDArray[np.float64], vec2: npt.NDArray[np.float64]
) -> float:
    """Get the oriented angle between two vectors.

    Args:
        vec1 (npt.NDArray[np.float64]): first vector
        vec2 (npt.NDArray[np.float64]): second vector

    Returns:
        float: angle between the two vectors.

    """
    vec1_norm: npt.NDArray[np.float64] = vec1 / np.linalg.norm(vec1)
    vec2_norm: npt.NDArray[np.float64] = vec2 / np.linalg.norm(vec2)
    dot: float = np.dot(vec1_norm, vec2_norm)
    det: float = np.linalg.det((vec1_norm, vec2_norm))
    # round to prevent from numerical approximations
    teta: float = np.arccos(round(dot, 6))
    if det < 0:
        teta = 2.0 * np.pi - teta
    return teta


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
