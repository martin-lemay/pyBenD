# SPDX-FileCopyrightText: Copyright 2025 Martin Lemay <martin.lemay@mines-paris.org>
# SPDX-FileContributor: Martin Lemay
# ruff: noqa: E402 # disable Module level import not at top of file
"""Useful geometry functions."""

from typing import Sequence

import numpy as np
import numpy.typing as npt


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


def orthogonal_distance(
    pt: npt.NDArray[np.float64],
    seg_pt1: npt.NDArray[np.float64],
    seg_pt2: npt.NDArray[np.float64],
    prec: int = 4,
) -> float:
    """Orthogonal distance between pt and its projection on a line.

    The projection is made on the infinite line passing through seg_pt1 and
    seg_pt2, not on the segment itself.

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


def normal(vec: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Compute the normalized orthogonal vector to nput.

    Args:
        vec (NDArray[float]): Coordinates of the vector.

    Returns:
        NDArray[float]: Coordinates of the normalized orthogonal vector.

    """
    normal_vec = perp(np.asarray(vec, dtype=float))
    normal_vec /= np.linalg.norm(normal_vec)
    return normal_vec


def seg_intersect(
    pt11: npt.NDArray[np.float64],
    pt12: npt.NDArray[np.float64],
    pt21: npt.NDArray[np.float64],
    pt22: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Compute the intersection point of lines (pt11,pt12) and (pt21,pt22).

    Lines are infinite: the intersection point may lie outside the segments.

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
    tol: float = 1e-12 * float(np.linalg.norm(da) * np.linalg.norm(db))
    assert abs(denom) > tol, "No intersection between the two lines."
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
        float: counterclockwise angle from vec1 to vec2 in [0, 2*pi).

    """
    dot: float = float(vec1[0] * vec2[0] + vec1[1] * vec2[1])
    det: float = float(vec1[0] * vec2[1] - vec1[1] * vec2[0])
    # arctan2 is accurate for all angles, unlike arccos near 0 and pi
    teta: float = float(np.arctan2(det, dot))
    if teta < 0.0:
        teta += 2.0 * np.pi
    if teta >= 2.0 * np.pi:  # -0. or tiny negative values rounded to 2*pi
        teta = 0.0
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


def get_MP(
    dir_trans: npt.NDArray[np.float64] = np.array((1.0, 0.0)),
    ref: npt.NDArray[np.float64] = np.array((1.0, 0.0)),
) -> npt.NDArray[np.float64]:
    """Get the rotation 2D matrix between ref and dir_trans.

    The returned matrix rotates dir_trans onto ref. dir_trans is treated as
    an axis: if it points away from ref (negative dot product), its opposite
    is used.

    Args:
        dir_trans (NDArray[float], optional): Direction.
            Defaults to np.array((1., 0.)).
        ref (NDArray[float], optional): Reference direction.
            Defaults to np.array((1., 0.)).

    Returns:
        NDArray[float]: Array corresponding to rotation 2D matrix.

    """
    dir_trans_arr: npt.NDArray[np.float64] = np.asarray(dir_trans, float)
    ref_arr: npt.NDArray[np.float64] = np.asarray(ref, float)
    dir_trans_norm: npt.NDArray[np.float64] = dir_trans_arr / np.linalg.norm(
        dir_trans_arr
    )
    ref_norm: npt.NDArray[np.float64] = ref_arr / np.linalg.norm(ref_arr)

    cos: float = float(np.dot(dir_trans_norm, ref_norm))
    if cos < 0.0:
        dir_trans_norm = -dir_trans_norm
        cos = -cos
    # signed sine of the angle from ref to dir_trans (2D cross product)
    sin: float = float(
        ref_norm[0] * dir_trans_norm[1] - ref_norm[1] * dir_trans_norm[0]
    )
    return np.array([[cos, sin], [-sin, cos]])
