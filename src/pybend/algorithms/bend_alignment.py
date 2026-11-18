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

from collections.abc import Callable
from dataclasses import dataclass, field

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

#: Penalty used in place of a metric difference that cannot be computed
#: (NaN or infinite, e.g. a bend without apex). It is large compared to
#: typical normalised differences (order 1) and gap penalties, so that a
#: pair of bends with an undefined metric is in practice never matched,
#: while keeping the dynamic-programming matrix finite.
NAN_METRIC_PENALTY: float = 1.0e3

#: Default multiplicative scale of the gap penalties.
DEFAULT_GAP_PENALTY_SCALE: float = 2.5


@dataclass
class BendAlignment:
    """Result of aligning bends between two centerlines.

    Attributes:
        matches: Pairs of matched bend ids (bend_id_t1, bend_id_t2).
        deletions: Bend ids at T1 with no match at T2.
        insertions: Bend ids at T2 with no match at T1.
        cost_matrix: The dynamic-programming cost matrix.
        total_cost: Total alignment cost.
        match_costs: Substitution cost of each matched pair, in the
            same order as *matches*. Empty if not computed.
    """

    matches: list[tuple[int, int]]
    deletions: list[int]
    insertions: list[int]
    cost_matrix: npt.NDArray[np.float64]
    total_cost: float
    match_costs: list[float] = field(default_factory=list)


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


def _has_apex(bend: Bend) -> bool:
    """Return True if the bend has a defined apex.

    Args:
        bend: Bend to check.

    Returns:
        True if ``bend.index_apex`` is neither None nor negative.
    """
    return bend.index_apex is not None and bend.index_apex >= 0


def _apex_distance(
    bend_i: Bend,
    bend_j: Bend,
    cl_i: Centerline,
    cl_j: Centerline,
) -> float:
    """Distance between the apex points of two bends.

    Args:
        bend_i: Bend from the first centerline.
        bend_j: Bend from the second centerline.
        cl_i: First centerline.
        cl_j: Second centerline.

    Returns:
        Apex distance, or NaN if one of the bends has no apex.
    """
    if not (_has_apex(bend_i) and _has_apex(bend_j)):
        return np.nan
    pt_apex_i = cl_i.cl_points[bend_i.index_apex].pt
    pt_apex_j = cl_j.cl_points[bend_j.index_apex].pt
    return float(geom.distance(pt_apex_i, pt_apex_j))


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

    The cost is a weighted sum of dimensionless differences of
    morphometric properties:

    - ``apex_distance``: apex-to-apex distance / *norm_width*,
    - ``amplitude``: amplitude difference / *norm_width*,
    - ``wavelength``: wavelength difference / *norm_width*,
    - ``curvature``: mean signed curvature difference * *norm_width*,
    - ``asymmetry``: apex asymmetry difference,
    - ``sinuosity``: sinuosity difference.

    Terms whose weight is 0 are not computed. A difference that cannot
    be computed (NaN or infinite, e.g. apex distance when a bend has no
    apex) is replaced by ``NAN_METRIC_PENALTY`` so that the cost stays
    finite.

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
            ``"asymmetry"``, ``"sinuosity"``. Missing keys default to
            1.0 (partial dicts are merged with the defaults).

    Returns:
        Non-negative, finite substitution cost.
    """
    w = {**_DEFAULT_WEIGHTS, **(weights or {})}

    def _curv_diff() -> float:
        # Use signed mean curvature for side sensitivity.
        curv_i = cl_i.get_bend_curvature_filtered(bend_i.id)
        curv_j = cl_j.get_bend_curvature_filtered(bend_j.id)
        return abs(float(np.mean(curv_i)) - float(np.mean(curv_j))) * (
            norm_width
        )

    terms: dict[str, Callable[[], float]] = {
        "apex_distance": lambda: (
            _apex_distance(bend_i, bend_j, cl_i, cl_j) / norm_width
        ),
        "amplitude": lambda: (
            abs(
                morpho_i.compute_bend_amplitude(bend_i.id)
                - morpho_j.compute_bend_amplitude(bend_j.id)
            )
            / norm_width
        ),
        "wavelength": lambda: (
            abs(
                morpho_i.compute_bend_wavelength(bend_i.id)
                - morpho_j.compute_bend_wavelength(bend_j.id)
            )
            / norm_width
        ),
        "curvature": _curv_diff,
        "asymmetry": lambda: abs(
            morpho_i.compute_bend_asymmetry_apex(bend_i.id)
            - morpho_j.compute_bend_asymmetry_apex(bend_j.id)
        ),
        "sinuosity": lambda: abs(
            morpho_i.compute_bend_sinuosity(bend_i.id)
            - morpho_j.compute_bend_sinuosity(bend_j.id)
        ),
    }

    cost: float = 0.0
    for key, func in terms.items():
        weight = float(w.get(key, 0.0))
        if weight == 0.0:
            continue
        diff = float(func())
        if not np.isfinite(diff):
            diff = NAN_METRIC_PENALTY
        cost += weight * diff
    return float(cost)


def _compute_gap_penalty(
    bend: Bend,
    cl: Centerline,
    morpho: Morphometry,
    gap_penalty_metric: str,
    gap_penalty_scale: float,
    norm_width: float = 1.0,
) -> float:
    """Compute gap penalty for inserting/deleting a bend.

    The penalty is dimensionless, like the substitution cost:

    - ``"sinuosity"``: ``gap_penalty_scale * sinuosity``,
    - ``"curvature"``: ``gap_penalty_scale * mean(|curvature|) *
      norm_width``.

    If the metric cannot be computed (NaN or infinite), a metric value
    of 1.0 is used (i.e. the penalty equals *gap_penalty_scale*).

    Args:
        bend: The bend being inserted or deleted.
        cl: Centerline the bend belongs to.
        morpho: Morphometry computer for the centerline.
        gap_penalty_metric: Metric to use. Either ``"sinuosity"``
            or ``"curvature"``.
        gap_penalty_scale: Multiplicative scale factor.
        norm_width: Normalisation width (e.g. channel width) used to
            make the curvature metric dimensionless. Callers should
            pass the same value as for the substitution cost.
            Defaults to 1.0.

    Returns:
        Non-negative gap penalty.

    Raises:
        ValueError: If *gap_penalty_metric* is not recognised.
    """
    metric: float
    if gap_penalty_metric == "sinuosity":
        metric = float(morpho.compute_bend_sinuosity(bend.id))
    elif gap_penalty_metric == "curvature":
        curv = cl.get_bend_curvature_filtered(bend.id)
        metric = float(np.mean(np.abs(curv))) * norm_width
    else:
        raise ValueError(
            f"Unknown gap penalty metric: '{gap_penalty_metric}'. "
            "Expected 'sinuosity' or 'curvature'."
        )
    if not np.isfinite(metric):
        metric = 1.0
    return gap_penalty_scale * metric


def align_bend_sequences(
    cl_t1: Centerline,
    cl_t2: Centerline,
    norm_width: float,
    weights: dict[str, float] | None = None,
    gap_penalty_metric: str = "sinuosity",
    gap_penalty_scale: float = DEFAULT_GAP_PENALTY_SCALE,
) -> BendAlignment:
    """Align two bend sequences using Needleman-Wunsch.

    Performs global sequence alignment via dynamic programming to
    find the optimal correspondence between bends at two time steps.

    Matching bend *i* (T1) with bend *j* (T2) costs the substitution
    cost ``c(i, j)`` (see ``_compute_substitution_cost``, a sum of
    dimensionless differences, 0 for identical bends). Leaving them
    unmatched costs one deletion plus one insertion, i.e. the sum of
    their gap penalties ``g(i) + g(j)``. A match is therefore preferred
    whenever ``c(i, j) < g(i) + g(j)``. With the default
    ``"sinuosity"`` metric, ``g = gap_penalty_scale * sinuosity`` with
    sinuosity >= 1, so the default scale of 2.5 tolerates a
    substitution cost of at least 5 (e.g. an apex displacement of
    about 2 channel widths combined with moderate shape changes) before
    preferring a deletion/insertion pair.

    Args:
        cl_t1: First centerline.
        cl_t2: Second centerline.
        norm_width: Normalisation width (e.g. channel width).
        weights: Optional weight dict for substitution cost. Missing
            keys default to 1.0.
        gap_penalty_metric: Metric for gap penalties
            (``"sinuosity"`` or ``"curvature"``).
        gap_penalty_scale: Scale factor for gap penalties.
            Defaults to 2.5.

    Returns:
        A ``BendAlignment`` with matches, deletions, insertions,
        the full cost matrix, the total alignment cost, and the
        substitution cost of each match.
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

    # 2. Pre-compute gap penalties and substitution costs.
    gap_t1 = np.array(
        [
            _compute_gap_penalty(
                bend,
                cl_t1,
                morpho_t1,
                gap_penalty_metric,
                gap_penalty_scale,
                norm_width,
            )
            for bend in bends_t1
        ],
        dtype=np.float64,
    )
    gap_t2 = np.array(
        [
            _compute_gap_penalty(
                bend,
                cl_t2,
                morpho_t2,
                gap_penalty_metric,
                gap_penalty_scale,
                norm_width,
            )
            for bend in bends_t2
        ],
        dtype=np.float64,
    )
    sub = np.zeros((n, m), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            sub[i, j] = _compute_substitution_cost(
                bends_t1[i],
                bends_t2[j],
                cl_t1,
                cl_t2,
                morpho_t1,
                morpho_t2,
                norm_width,
                weights,
            )

    # 3. Initialise DP matrix: first column (deletions) and first row
    # (insertions).
    dp = np.zeros((n + 1, m + 1), dtype=np.float64)
    for i in range(1, n + 1):
        dp[i, 0] = dp[i - 1, 0] + gap_t1[i - 1]
    for j in range(1, m + 1):
        dp[0, j] = dp[0, j - 1] + gap_t2[j - 1]

    # 4. Fill the DP matrix.
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i, j] = min(
                dp[i - 1, j - 1] + sub[i - 1, j - 1],
                dp[i - 1, j] + gap_t1[i - 1],
                dp[i, j - 1] + gap_t2[j - 1],
            )

    # 5. Traceback.
    matches: list[tuple[int, int]] = []
    match_costs: list[float] = []
    deletions: list[int] = []
    insertions: list[int] = []
    i, j = n, m
    while i > 0 or j > 0:
        if (
            i > 0
            and j > 0
            and np.isclose(dp[i, j], dp[i - 1, j - 1] + sub[i - 1, j - 1])
        ):
            matches.append((bends_t1[i - 1].id, bends_t2[j - 1].id))
            match_costs.append(float(sub[i - 1, j - 1]))
            i -= 1
            j -= 1
            continue
        if i > 0 and np.isclose(dp[i, j], dp[i - 1, j] + gap_t1[i - 1]):
            deletions.append(bends_t1[i - 1].id)
            i -= 1
            continue
        if j > 0:
            insertions.append(bends_t2[j - 1].id)
            j -= 1
        else:
            # Numerical safety: only deletions remain.
            deletions.append(bends_t1[i - 1].id)
            i -= 1

    matches.reverse()
    match_costs.reverse()
    deletions.reverse()
    insertions.reverse()

    # 6. Return result.
    return BendAlignment(
        matches=matches,
        deletions=deletions,
        insertions=insertions,
        cost_matrix=dp,
        total_cost=float(dp[n, m]),
        match_costs=match_costs,
    )


def _get_bend_centroid(
    bend: Bend,
    cl: Centerline,
) -> npt.NDArray[np.float64]:
    """Return the centroid of a bend.

    Uses ``bend.pt_centroid`` if available, otherwise falls back
    to the apex point coordinates from the centerline, or to the
    middle point (by index) of the bend if it has no apex.

    Args:
        bend: The bend whose centroid is needed.
        cl: Centerline the bend belongs to.

    Returns:
        2-D coordinate array of the centroid.
    """
    if bend.pt_centroid is not None:
        return bend.pt_centroid
    if _has_apex(bend):
        return cl.cl_points[bend.index_apex].pt
    index_mid = (bend.index_inflex_up + bend.index_inflex_down) // 2
    return cl.cl_points[index_mid].pt


def detect_splits_and_merges(
    alignment: BendAlignment,
    cl_t1: Centerline,
    cl_t2: Centerline,
    norm_width: float,
    spatial_tolerance: float = 2.0,
) -> SplitMergeResult:
    """Detect split (1->N) and merge (N->1) events.

    A **split** occurs when a single deleted bend at T1 has two or
    more inserted bends at T2 within the spatial tolerance.  A
    **merge** is the reverse: a single inserted bend at T2 has two
    or more deleted bends at T1 within the tolerance.  Each deleted
    or inserted bend takes part in at most one event: bends already
    claimed by an earlier split (or merge) are skipped.

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
        candidate_ids: list[int] = []
        for i_bend in inserted:
            # an inserted bend can be the child of a single split
            if i_bend.id in claimed_insertions:
                continue
            i_pt = _get_bend_centroid(i_bend, cl_t2)
            if geom.distance(d_pt[:2], i_pt[:2]) <= threshold:
                candidates.append(i_bend.uid)
                candidate_ids.append(i_bend.id)
        if len(candidates) >= 2:
            splits.append((d_bend.uid, candidates))
            claimed_deletions.add(d_bend.id)
            claimed_insertions.update(candidate_ids)

    # --- Merge detection (N deleted -> 1 inserted) ---
    merges: list[tuple[list[int], int]] = []
    for i_bend in inserted:
        if i_bend.id in claimed_insertions:
            continue
        i_pt = _get_bend_centroid(i_bend, cl_t2)
        merge_candidates: list[int] = []
        merge_candidate_ids: list[int] = []
        for d_bend in deleted:
            if d_bend.id in claimed_deletions:
                continue
            d_pt = _get_bend_centroid(d_bend, cl_t1)
            if geom.distance(d_pt[:2], i_pt[:2]) <= threshold:
                merge_candidates.append(d_bend.uid)
                merge_candidate_ids.append(d_bend.id)
        if len(merge_candidates) >= 2:
            merges.append((merge_candidates, i_bend.uid))
            claimed_insertions.add(i_bend.id)
            claimed_deletions.update(merge_candidate_ids)

    return SplitMergeResult(splits=splits, merges=merges)
