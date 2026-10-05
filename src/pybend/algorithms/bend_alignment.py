# SPDX-FileCopyrightText: Copyright 2025 Martin Lemay <martin.lemay@mines-paris.org>
# SPDX-FileContributor: Martin Lemay
# ruff: noqa: E402 # disable Module level import not at top of file

"""Bend alignment dataclasses and cost/penalty functions.

This module provides data structures for representing bend-to-bend
alignments across successive centerlines, as well as functions for
computing substitution costs and gap penalties used in the
Needleman-Wunsch alignment algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

import pybend.algorithms.geometry_functions as geom
from pybend.model.Bend import Bend
from pybend.model.Centerline import Centerline
from pybend.model.Morphometry import Morphometry

#: Default weights for substitution cost components.
_DEFAULT_WEIGHTS: dict[str, float] = {
    "apex_distance": 1.0,
    "amplitude": 1.0,
    "wavelength": 1.0,
    "curvature": 1.0,
    "asymmetry": 1.0,
    "sinuosity": 1.0,
}


@dataclass
class BendAlignment:
    """Result of aligning bends between two centerlines.

    Attributes:
        matches: Pairs of matched bend ids (bend_id_t1, bend_id_t2).
        deletions: Bend ids at T1 with no match at T2.
        insertions: Bend ids at T2 with no match at T1.
        cost_matrix: The dynamic-programming cost matrix.
        total_cost: Total alignment cost.
    """

    matches: list[tuple[int, int]]
    deletions: list[int]
    insertions: list[int]
    cost_matrix: npt.NDArray[np.float64]
    total_cost: float


@dataclass
class SplitMergeResult:
    """Split and merge events detected from an alignment.

    Attributes:
        splits: Each entry is (bend_uid_t1, [bend_uids_t2])
            where one bend at T1 maps to multiple bends at T2.
        merges: Each entry is ([bend_uids_t1], bend_uid_t2)
            where multiple bends at T1 map to one bend at T2.
    """

    splits: list[tuple[int, list[int]]]
    merges: list[tuple[list[int], int]]


def _compute_substitution_cost(
    bend_i: Bend,
    bend_j: Bend,
    cl_i: Centerline,
    cl_j: Centerline,
    morpho_i: Morphometry,
    morpho_j: Morphometry,
    norm_width: float,
    weights: dict[str, float] | None = None,
) -> float:
    """Compute substitution cost between two bends.

    The cost is a weighted sum of normalised differences of
    morphometric properties.

    Args:
        bend_i: Bend from the first centerline.
        bend_j: Bend from the second centerline.
        cl_i: First centerline.
        cl_j: Second centerline.
        morpho_i: Morphometry computer for the first centerline.
        morpho_j: Morphometry computer for the second centerline.
        norm_width: Normalisation width (e.g. channel width).
        weights: Optional weight dict. Keys are ``"apex_distance"``,
            ``"amplitude"``, ``"wavelength"``, ``"curvature"``,
            ``"asymmetry"``, ``"sinuosity"``.  Defaults to 1.0 each.

    Returns:
        Non-negative substitution cost.
    """
    w = weights if weights is not None else _DEFAULT_WEIGHTS

    # Apex distance / norm_width
    pt_apex_i = cl_i.cl_points[bend_i.index_apex].pt
    pt_apex_j = cl_j.cl_points[bend_j.index_apex].pt
    apex_dist = geom.distance(pt_apex_i, pt_apex_j) / norm_width

    # Amplitude difference / norm_width
    amp_i = morpho_i.compute_bend_amplitude(bend_i.id)
    amp_j = morpho_j.compute_bend_amplitude(bend_j.id)
    amp_diff = abs(amp_i - amp_j) / norm_width

    # Wavelength difference / norm_width
    wl_i = morpho_i.compute_bend_wavelength(bend_i.id)
    wl_j = morpho_j.compute_bend_wavelength(bend_j.id)
    wl_diff = abs(wl_i - wl_j) / norm_width

    # Average curvature difference * norm_width
    # Use signed mean curvature for side sensitivity.
    curv_i = cl_i.get_bend_curvature_filtered(bend_i.id)
    curv_j = cl_j.get_bend_curvature_filtered(bend_j.id)
    mean_curv_i = float(np.mean(curv_i))
    mean_curv_j = float(np.mean(curv_j))
    curv_diff = abs(mean_curv_i - mean_curv_j) * norm_width

    # Asymmetry difference (dimensionless)
    asym_i = morpho_i.compute_bend_asymmetry_apex(bend_i.id)
    asym_j = morpho_j.compute_bend_asymmetry_apex(bend_j.id)
    asym_diff = abs(asym_i - asym_j)

    # Sinuosity difference (dimensionless)
    sin_i = morpho_i.compute_bend_sinuosity(bend_i.id)
    sin_j = morpho_j.compute_bend_sinuosity(bend_j.id)
    sin_diff = abs(sin_i - sin_j)

    cost = (
        w.get("apex_distance", 0.0) * apex_dist
        + w.get("amplitude", 0.0) * amp_diff
        + w.get("wavelength", 0.0) * wl_diff
        + w.get("curvature", 0.0) * curv_diff
        + w.get("asymmetry", 0.0) * asym_diff
        + w.get("sinuosity", 0.0) * sin_diff
    )
    return float(cost)


def _compute_gap_penalty(
    bend: Bend,
    cl: Centerline,
    morpho: Morphometry,
    gap_penalty_metric: str,
    gap_penalty_scale: float,
) -> float:
    """Compute gap penalty for inserting/deleting a bend.

    Args:
        bend: The bend being inserted or deleted.
        cl: Centerline the bend belongs to.
        morpho: Morphometry computer for the centerline.
        gap_penalty_metric: Metric to use. Either ``"sinuosity"``
            or ``"curvature"``.
        gap_penalty_scale: Multiplicative scale factor.

    Returns:
        Non-negative gap penalty.

    Raises:
        ValueError: If *gap_penalty_metric* is not recognised.
    """
    if gap_penalty_metric == "sinuosity":
        sinuosity = morpho.compute_bend_sinuosity(bend.id)
        return gap_penalty_scale * sinuosity
    if gap_penalty_metric == "curvature":
        curv = cl.get_bend_curvature_filtered(bend.id)
        return gap_penalty_scale * float(np.mean(np.abs(curv)))
    raise ValueError(
        f"Unknown gap penalty metric: '{gap_penalty_metric}'. "
        "Expected 'sinuosity' or 'curvature'."
    )


def align_bend_sequences(
    cl_t1: Centerline,
    cl_t2: Centerline,
    norm_width: float,
    weights: dict[str, float] | None = None,
    gap_penalty_metric: str = "sinuosity",
    gap_penalty_scale: float = 1.0,
) -> BendAlignment:
    """Align two bend sequences using Needleman-Wunsch.

    Performs global sequence alignment via dynamic programming to
    find the optimal correspondence between bends at two time steps.

    Args:
        cl_t1: First centerline.
        cl_t2: Second centerline.
        norm_width: Normalisation width (e.g. channel width).
        weights: Optional weight dict for substitution cost.
        gap_penalty_metric: Metric for gap penalties
            (``"sinuosity"`` or ``"curvature"``).
        gap_penalty_scale: Scale factor for gap penalties.

    Returns:
        A ``BendAlignment`` with matches, deletions, insertions,
        the full cost matrix, and total alignment cost.
    """
    # 1. Build morphometry instances and compute metrics.
    morpho_t1 = Morphometry(cl_t1)
    morpho_t1.compute_bends_morphometry(valid_bends=False)
    morpho_t2 = Morphometry(cl_t2)
    morpho_t2.compute_bends_morphometry(valid_bends=False)

    bends_t1 = cl_t1.bends
    bends_t2 = cl_t2.bends

    n = len(bends_t1)
    m = len(bends_t2)

    # 2. Initialise DP matrix.
    dp = np.zeros((n + 1, m + 1), dtype=np.float64)

    # 3. Fill first column (deletions) and first row (insertions).
    for i in range(1, n + 1):
        dp[i, 0] = dp[i - 1, 0] + _compute_gap_penalty(
            bends_t1[i - 1],
            cl_t1,
            morpho_t1,
            gap_penalty_metric,
            gap_penalty_scale,
        )
    for j in range(1, m + 1):
        dp[0, j] = dp[0, j - 1] + _compute_gap_penalty(
            bends_t2[j - 1],
            cl_t2,
            morpho_t2,
            gap_penalty_metric,
            gap_penalty_scale,
        )

    # 4. Fill the DP matrix.
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub_cost = dp[i - 1, j - 1] + _compute_substitution_cost(
                bends_t1[i - 1],
                bends_t2[j - 1],
                cl_t1,
                cl_t2,
                morpho_t1,
                morpho_t2,
                norm_width,
                weights,
            )
            del_cost = dp[i - 1, j] + _compute_gap_penalty(
                bends_t1[i - 1],
                cl_t1,
                morpho_t1,
                gap_penalty_metric,
                gap_penalty_scale,
            )
            ins_cost = dp[i, j - 1] + _compute_gap_penalty(
                bends_t2[j - 1],
                cl_t2,
                morpho_t2,
                gap_penalty_metric,
                gap_penalty_scale,
            )
            dp[i, j] = min(sub_cost, del_cost, ins_cost)

    # 5. Traceback.
    matches: list[tuple[int, int]] = []
    deletions: list[int] = []
    insertions: list[int] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            sub_cost = dp[i - 1, j - 1] + _compute_substitution_cost(
                bends_t1[i - 1],
                bends_t2[j - 1],
                cl_t1,
                cl_t2,
                morpho_t1,
                morpho_t2,
                norm_width,
                weights,
            )
            if np.isclose(dp[i, j], sub_cost):
                matches.append((bends_t1[i - 1].id, bends_t2[j - 1].id))
                i -= 1
                j -= 1
                continue
        if i > 0:
            del_cost = dp[i - 1, j] + _compute_gap_penalty(
                bends_t1[i - 1],
                cl_t1,
                morpho_t1,
                gap_penalty_metric,
                gap_penalty_scale,
            )
            if np.isclose(dp[i, j], del_cost):
                deletions.append(bends_t1[i - 1].id)
                i -= 1
                continue
        if j > 0:
            insertions.append(bends_t2[j - 1].id)
            j -= 1

    matches.reverse()
    deletions.reverse()
    insertions.reverse()

    # 6. Return result.
    return BendAlignment(
        matches=matches,
        deletions=deletions,
        insertions=insertions,
        cost_matrix=dp,
        total_cost=float(dp[n, m]),
    )


def _get_bend_centroid(
    bend: Bend,
    cl: Centerline,
) -> npt.NDArray[np.float64]:
    """Return the centroid of a bend.

    Uses ``bend.pt_centroid`` if available, otherwise falls back
    to the apex point coordinates from the centerline.

    Args:
        bend: The bend whose centroid is needed.
        cl: Centerline the bend belongs to.

    Returns:
        2-D coordinate array of the centroid.
    """
    if bend.pt_centroid is not None:
        return bend.pt_centroid
    return cl.cl_points[bend.index_apex].pt


def detect_splits_and_merges(
    alignment: BendAlignment,
    cl_t1: Centerline,
    cl_t2: Centerline,
    norm_width: float,
    spatial_tolerance: float = 5.0,
) -> SplitMergeResult:
    """Detect split (1->N) and merge (N->1) events.

    A **split** occurs when a single deleted bend at T1 has two or
    more inserted bends at T2 within the spatial tolerance.  A
    **merge** is the reverse: a single inserted bend at T2 has two
    or more deleted bends at T1 within the tolerance.

    Args:
        alignment: Result of ``align_bend_sequences``.
        cl_t1: First centerline.
        cl_t2: Second centerline.
        norm_width: Normalisation width (e.g. channel width).
        spatial_tolerance: Distance threshold expressed as a
            multiple of *norm_width*.  Defaults to ``2.0``.

    Returns:
        A ``SplitMergeResult`` with detected splits and merges.
    """
    threshold = spatial_tolerance * norm_width

    bends_t1 = cl_t1.bends
    bends_t2 = cl_t2.bends

    # Build look-ups: bend id -> Bend object.
    t1_by_id = {b.id: b for b in bends_t1}
    t2_by_id = {b.id: b for b in bends_t2}

    deleted = [t1_by_id[bid] for bid in alignment.deletions]
    inserted = [t2_by_id[bid] for bid in alignment.insertions]

    # Track claimed ids to avoid double-counting.
    claimed_insertions: set[int] = set()
    claimed_deletions: set[int] = set()

    # --- Split detection (1 deleted -> N inserted) ---
    splits: list[tuple[int, list[int]]] = []
    for d_bend in deleted:
        d_pt = _get_bend_centroid(d_bend, cl_t1)
        candidates: list[int] = []
        for i_bend in inserted:
            i_pt = _get_bend_centroid(i_bend, cl_t2)
            if geom.distance(d_pt, i_pt) <= threshold:
                candidates.append(i_bend.uid)
        if len(candidates) >= 2:
            splits.append((d_bend.uid, candidates))
            claimed_deletions.add(d_bend.id)
            for i_bend in inserted:
                if i_bend.uid in candidates:
                    claimed_insertions.add(i_bend.id)

    # --- Merge detection (N deleted -> 1 inserted) ---
    merges: list[tuple[list[int], int]] = []
    for i_bend in inserted:
        if i_bend.id in claimed_insertions:
            continue
        i_pt = _get_bend_centroid(i_bend, cl_t2)
        merge_candidates: list[int] = []
        for d_bend in deleted:
            if d_bend.id in claimed_deletions:
                continue
            d_pt = _get_bend_centroid(d_bend, cl_t1)
            if geom.distance(d_pt, i_pt) <= threshold:
                merge_candidates.append(d_bend.uid)
        if len(merge_candidates) >= 2:
            merges.append((merge_candidates, i_bend.uid))
            claimed_insertions.add(i_bend.id)
            for d_bend in deleted:
                if d_bend.uid in merge_candidates:
                    claimed_deletions.add(d_bend.id)

    return SplitMergeResult(splits=splits, merges=merges)
