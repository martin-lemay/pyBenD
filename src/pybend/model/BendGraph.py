# SPDX-FileCopyrightText: Copyright 2025 Martin Lemay
# SPDX-FileContributor: Martin Lemay

"""Graph construction for meander bends.

Builds spatial and temporal graphs where each node represents a
bend and edges connect consecutive or temporally linked bends.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

import networkx as nx

from pybend.model.Bend import parse_bend_uid
from pybend.model.BendEvolution import BendEvolution

if TYPE_CHECKING:
    from pybend.model.Centerline import Centerline
    from pybend.model.CenterlineCollection import (
        CenterlineCollection,
    )


def build_spatial_graph(centerline: Centerline) -> nx.Graph:
    """Build an undirected path graph of bends along a centerline.

    Each bend is a node (ID = bend.uid, data = Bend reference).
    Edges connect consecutive bends in the ordered bend list.

    Args:
        centerline: a Centerline with detected bends.

    Returns:
        nx.Graph: undirected path graph.
    """
    g: nx.Graph = nx.Graph()
    bends = centerline.bends
    for bend in bends:
        g.add_node(bend.uid, bend=bend)
    for i in range(len(bends) - 1):
        g.add_edge(bends[i].uid, bends[i + 1].uid)
    return g


def build_temporal_graph(
    centerline_collection: "CenterlineCollection",
    norm_width: float,
    weights: dict[str, float] | None = None,
    gap_penalty_metric: str = "sinuosity",
    gap_penalty_scale: float = 1.0,
    spatial_tolerance: float = 2.0,
) -> nx.DiGraph:
    """Build a directed temporal graph linking bends across ages.

    Nodes are all bends from every centerline in the collection.
    Directed edges go from older to younger bends based on
    sequence alignment and split/merge detection.

    Args:
        centerline_collection: Collection of centerlines.
        norm_width: Normalisation width (e.g. channel width).
        weights: Optional weight dict for substitution cost.
        gap_penalty_metric: Metric for gap penalties
            (``"sinuosity"`` or ``"curvature"``).
        gap_penalty_scale: Scale factor for gap penalties.
        spatial_tolerance: Distance threshold expressed as a
            multiple of *norm_width* for split/merge detection.

    Returns:
        nx.DiGraph: directed temporal graph.
    """
    from pybend.algorithms.bend_alignment import (
        align_bend_sequences as _align,
    )
    from pybend.algorithms.bend_alignment import (
        detect_splits_and_merges as _detect_sm,
    )

    g: nx.DiGraph = nx.DiGraph()
    ages = centerline_collection.get_all_ages()

    # 1. Add ALL bends as nodes.
    for age in ages:
        cl = centerline_collection.centerlines[age]
        for bend in cl.bends:
            g.add_node(bend.uid, bend=bend)

    # 2. For each consecutive pair of ages, align and link.
    for idx in range(len(ages) - 1):
        age_t1 = int(ages[idx])
        age_t2 = int(ages[idx + 1])
        cl_t1 = centerline_collection.centerlines[age_t1]
        cl_t2 = centerline_collection.centerlines[age_t2]

        alignment = _align(
            cl_t1,
            cl_t2,
            norm_width,
            weights,
            gap_penalty_metric,
            gap_penalty_scale,
        )

        # 2a. Matched pairs.
        for bid_t1, bid_t2 in alignment.matches:
            bend_t1 = cl_t1.bends[bid_t1]
            bend_t2 = cl_t2.bends[bid_t2]
            g.add_edge(
                bend_t1.uid,
                bend_t2.uid,
                cost=alignment.total_cost,
            )
            bend_t1.add_bend_connection_next(bend_t2.uid)
            bend_t2.add_bend_connection_prev(bend_t1.uid)

        # 2b. Detect splits and merges.
        sm = _detect_sm(
            alignment,
            cl_t1,
            cl_t2,
            norm_width,
            spatial_tolerance,
        )

        # 2c. Splits: one bend at T1 -> multiple at T2.
        for src_uid, dst_uids in sm.splits:
            _, src_id = parse_bend_uid(src_uid)
            src_bend = cl_t1.bends[src_id]
            for dst_uid in dst_uids:
                _, dst_id = parse_bend_uid(dst_uid)
                dst_bend = cl_t2.bends[dst_id]
                g.add_edge(src_uid, dst_uid, split=True)
                src_bend.add_bend_connection_next(dst_uid)
                dst_bend.add_bend_connection_prev(src_uid)

        # 2d. Merges: multiple bends at T1 -> one at T2.
        for src_uids, dst_uid in sm.merges:
            _, dst_id = parse_bend_uid(dst_uid)
            dst_bend = cl_t2.bends[dst_id]
            for src_uid in src_uids:
                _, src_id = parse_bend_uid(src_uid)
                src_bend = cl_t1.bends[src_id]
                g.add_edge(src_uid, dst_uid, merge=True)
                src_bend.add_bend_connection_next(dst_uid)
                dst_bend.add_bend_connection_prev(src_uid)

    return g


def build_bend_evolutions_from_graph(
    graph: nx.DiGraph,
    centerline_collection: "CenterlineCollection",
    bend_evol_validity: int = 2,
) -> list[BendEvolution]:
    """Build bend evolutions by tracing ancestors in the graph.

    Starting from bends in the youngest centerline, traverses
    backward through predecessors to collect all ancestor bends,
    then groups them by age into ``BendEvolution`` objects.

    Args:
        graph: Directed temporal graph from
            ``build_temporal_graph``.
        centerline_collection: Collection of centerlines.
        bend_evol_validity: Minimum number of ages for a
            valid evolution. Defaults to ``2``.

    Returns:
        list[BendEvolution]: one evolution per bend in the
            youngest centerline.
    """
    ages = centerline_collection.get_all_ages()
    youngest_age = int(ages[-1])
    youngest_cl = centerline_collection.centerlines[youngest_age]

    evolutions: list[BendEvolution] = []

    for i, bend in enumerate(youngest_cl.bends):
        # Collect all ancestors via BFS backward.
        visited: set[int] = set()
        queue: list[int] = [bend.uid]
        ancestors: list[int] = [bend.uid]
        visited.add(bend.uid)

        while queue:
            current = queue.pop(0)
            for pred in graph.predecessors(current):
                if pred not in visited:
                    visited.add(pred)
                    ancestors.append(pred)
                    queue.append(pred)

        # Group by age.
        bend_indexes: dict[int, list[int]] = {}
        for uid in ancestors:
            age, bid = parse_bend_uid(uid)
            if age not in bend_indexes:
                bend_indexes[age] = []
            bend_indexes[age].append(bid)

        order = max(len(ids) for ids in bend_indexes.values())

        evol = BendEvolution(
            bend_indexes=bend_indexes,
            ide=i,
            order=order,
        )
        evol.set_is_valid(bend_evol_validity)
        evolutions.append(evol)

    return evolutions


# ------------------------------------------------------------------
# Convenience helpers
# ------------------------------------------------------------------


def get_lineage(graph: nx.DiGraph, bend_uid: int) -> list[int]:
    """Trace full lineage of a bend.

    Collects ancestors (via predecessors), the bend itself, and
    descendants (via successors).

    Args:
        graph: Directed temporal graph.
        bend_uid: UID of the bend to trace.

    Returns:
        List of UIDs ordered by age (oldest to youngest).
    """
    visited: set[int] = {bend_uid}

    # BFS backward for ancestors.
    queue: deque[int] = deque([bend_uid])
    while queue:
        node = queue.popleft()
        for pred in graph.predecessors(node):
            if pred not in visited:
                visited.add(pred)
                queue.append(pred)

    # BFS forward for descendants.
    queue = deque([bend_uid])
    while queue:
        node = queue.popleft()
        for succ in graph.successors(node):
            if succ not in visited:
                visited.add(succ)
                queue.append(succ)

    return sorted(
        visited,
        key=lambda u: graph.nodes[u]["bend"].age,
    )


def get_births(graph: nx.DiGraph) -> list[int]:
    """Return UIDs of all root nodes (in-degree 0).

    Args:
        graph: Directed temporal graph.

    Returns:
        List of UIDs with no predecessors.
    """
    return [n for n in graph.nodes() if graph.in_degree(n) == 0]


def get_deaths(graph: nx.DiGraph) -> list[int]:
    """Return UIDs of all terminal nodes (out-degree 0).

    Args:
        graph: Directed temporal graph.

    Returns:
        List of UIDs with no successors.
    """
    return [n for n in graph.nodes() if graph.out_degree(n) == 0]


def get_split_bends(graph: nx.DiGraph) -> list[int]:
    """Return UIDs of bends that split (out-degree > 1).

    Args:
        graph: Directed temporal graph.

    Returns:
        List of UIDs whose out-degree exceeds 1.
    """
    return [n for n in graph.nodes() if graph.out_degree(n) > 1]


def get_merge_bends(graph: nx.DiGraph) -> list[int]:
    """Return UIDs of bends resulting from a merge.

    A merge bend has in-degree > 1, meaning multiple older
    bends converge into it.

    Args:
        graph: Directed temporal graph.

    Returns:
        List of UIDs whose in-degree exceeds 1.
    """
    return [n for n in graph.nodes() if graph.in_degree(n) > 1]


def is_single_lobe_lineage(graph: nx.DiGraph, bend_uid: int) -> bool:
    """Check whether a bend's lineage is single-lobe.

    A single-lobe lineage has every node with in-degree <= 1
    and out-degree <= 1 (i.e. no splits or merges).

    Args:
        graph: Directed temporal graph.
        bend_uid: UID of the bend to check.

    Returns:
        True if the full lineage is a simple chain.
    """
    lineage = get_lineage(graph, bend_uid)
    return all(
        graph.in_degree(n) <= 1 and graph.out_degree(n) <= 1 for n in lineage
    )


def get_single_lobe_lineages(
    graph: nx.DiGraph,
) -> list[list[int]]:
    """Return all single-lobe lineages starting from births.

    A single-lobe lineage is one where every node has
    in-degree <= 1 and out-degree <= 1.

    Args:
        graph: Directed temporal graph.

    Returns:
        List of lineages (each a list of UIDs sorted by age).
    """
    result: list[list[int]] = []
    for birth in get_births(graph):
        lineage = get_lineage(graph, birth)
        if all(
            graph.in_degree(n) <= 1 and graph.out_degree(n) <= 1
            for n in lineage
        ):
            result.append(lineage)
    return result


def get_multilobate_lineages(
    graph: nx.DiGraph,
) -> list[list[int]]:
    """Return all lineages containing at least one split.

    A multilobate lineage has at least one node with
    out-degree > 1.

    Args:
        graph: Directed temporal graph.

    Returns:
        List of lineages (each a list of UIDs sorted by age).
    """
    result: list[list[int]] = []
    for birth in get_births(graph):
        lineage = get_lineage(graph, birth)
        if any(graph.out_degree(n) > 1 for n in lineage):
            result.append(lineage)
    return result
