# SPDX-FileCopyrightText: Copyright 2025 Martin Lemay <martin.lemay@mines-paris.org>
# SPDX-FileContributor: Martin Lemay
# ruff: noqa: E402 # disable Module level import not at top of file

"""Tests for bend_alignment dataclasses and cost/penalty functions."""

import copy
import inspect
import unittest
from types import SimpleNamespace
from typing import Self
from unittest import mock

import networkx as nx
import numpy as np
import pandas as pd

from pybend.algorithms.bend_alignment import (
    NAN_METRIC_PENALTY,
    BendAlignment,
    SplitMergeResult,
    _compute_gap_penalty,
    _compute_substitution_cost,
    align_bend_sequences,
    detect_splits_and_merges,
)
from pybend.model.Bend import get_bend_uid
from pybend.model.BendGraph import (
    build_bend_evolutions_from_graph,
    build_temporal_graph,
)
from pybend.model.Centerline import Centerline
from pybend.model.Morphometry import Morphometry
from pybend.utils.globalParameters import set_nb_procs
from pybend.utils.logging import ERROR, logger

logger.setLevel(ERROR)
set_nb_procs(1)


def _make_sinuous_centerline_data(n_bends: int, age: int) -> pd.DataFrame:
    """Generate a sine-wave DataFrame for Centerline construction.

    Args:
        n_bends: Number of bends to generate.
        age: Age value (unused in data, kept for API symmetry).

    Returns:
        DataFrame with Cart_abscissa and Cart_ordinate columns.
    """
    n_pts = max(500, n_bends * 100)
    x = np.linspace(0, n_bends * np.pi * 100, n_pts)
    y = 200.0 * np.sin(x / (np.pi * 100) * np.pi)
    return pd.DataFrame(
        {
            "Cart_abscissa": x,
            "Cart_ordinate": y,
            "Elevation": np.zeros_like(x),
        }
    )


def _make_centerline(n_bends: int, age: int) -> Centerline:
    """Build a Centerline from synthetic sinusoidal data.

    Args:
        n_bends: Number of bends.
        age: Age of the centerline.

    Returns:
        Centerline with bends detected.
    """
    dataset = _make_sinuous_centerline_data(n_bends, age)
    return Centerline(
        age,
        dataset,
        spacing=50.0,
        smooth_distance=100.0,
        sinuo_thres=1.0,
        n=2,
    )


class TestBendAlignmentDataclass(unittest.TestCase):
    """Tests for BendAlignment dataclass."""

    def test_construction(self: Self) -> None:
        """BendAlignment stores all fields correctly."""
        mat = np.zeros((3, 4))
        ba = BendAlignment(
            matches=[(0, 0), (1, 1)],
            deletions=[2],
            insertions=[3],
            cost_matrix=mat,
            total_cost=5.0,
        )
        self.assertEqual(ba.matches, [(0, 0), (1, 1)])
        self.assertEqual(ba.deletions, [2])
        self.assertEqual(ba.insertions, [3])
        self.assertIs(ba.cost_matrix, mat)
        self.assertEqual(ba.total_cost, 5.0)

    def test_empty(self: Self) -> None:
        """BendAlignment works with empty lists."""
        ba = BendAlignment(
            matches=[],
            deletions=[],
            insertions=[],
            cost_matrix=np.array([]),
            total_cost=0.0,
        )
        self.assertEqual(len(ba.matches), 0)
        self.assertEqual(ba.total_cost, 0.0)


class TestSplitMergeResultDataclass(unittest.TestCase):
    """Tests for SplitMergeResult dataclass."""

    def test_construction(self: Self) -> None:
        """SplitMergeResult stores splits and merges."""
        smr = SplitMergeResult(
            splits=[(1, [10, 11])],
            merges=[([20, 21], 2)],
        )
        self.assertEqual(smr.splits, [(1, [10, 11])])
        self.assertEqual(smr.merges, [([20, 21], 2)])

    def test_empty(self: Self) -> None:
        """SplitMergeResult works with empty lists."""
        smr = SplitMergeResult(splits=[], merges=[])
        self.assertEqual(len(smr.splits), 0)
        self.assertEqual(len(smr.merges), 0)


class TestComputeSubstitutionCost(unittest.TestCase):
    """Tests for _compute_substitution_cost function."""

    cl_1: Centerline
    cl_2: Centerline
    morpho_1: Morphometry
    morpho_2: Morphometry
    norm_width: float

    @classmethod
    def setUpClass(cls) -> None:
        """Create two centerlines with bends for cost tests."""
        cls.cl_1 = _make_centerline(6, age=0)
        cls.cl_2 = _make_centerline(6, age=1)
        cls.morpho_1 = Morphometry(cls.cl_1)
        cls.morpho_2 = Morphometry(cls.cl_2)
        cls.norm_width = 100.0

    def test_identical_bends_zero_cost(self: Self) -> None:
        """Cost of aligning a bend with itself should be zero."""
        cl = self.__class__.cl_1
        morpho = self.__class__.morpho_1
        nw = self.__class__.norm_width
        bend = cl.bends[0]
        cost = _compute_substitution_cost(
            bend, bend, cl, cl, morpho, morpho, nw
        )
        self.assertAlmostEqual(cost, 0.0, places=6)

    def test_cost_non_negative(self: Self) -> None:
        """Substitution cost must always be non-negative."""
        cl1 = self.__class__.cl_1
        cl2 = self.__class__.cl_2
        m1 = self.__class__.morpho_1
        m2 = self.__class__.morpho_2
        nw = self.__class__.norm_width
        for i in range(min(len(cl1.bends), 3)):
            for j in range(min(len(cl2.bends), 3)):
                cost = _compute_substitution_cost(
                    cl1.bends[i],
                    cl2.bends[j],
                    cl1,
                    cl2,
                    m1,
                    m2,
                    nw,
                )
                self.assertGreaterEqual(cost, 0.0)

    def test_symmetry(self: Self) -> None:
        """Cost(i,j) should equal cost(j,i)."""
        cl1 = self.__class__.cl_1
        cl2 = self.__class__.cl_2
        m1 = self.__class__.morpho_1
        m2 = self.__class__.morpho_2
        nw = self.__class__.norm_width
        b1 = cl1.bends[0]
        b2 = cl2.bends[1]
        c_ij = _compute_substitution_cost(b1, b2, cl1, cl2, m1, m2, nw)
        c_ji = _compute_substitution_cost(b2, b1, cl2, cl1, m2, m1, nw)
        self.assertAlmostEqual(c_ij, c_ji, places=6)

    def test_custom_weights(self: Self) -> None:
        """Zero weights should give zero cost."""
        cl = self.__class__.cl_1
        cl2 = self.__class__.cl_2
        m1 = self.__class__.morpho_1
        m2 = self.__class__.morpho_2
        nw = self.__class__.norm_width
        zero_weights = {
            "apex_distance": 0.0,
            "amplitude": 0.0,
            "wavelength": 0.0,
            "curvature": 0.0,
            "asymmetry": 0.0,
            "sinuosity": 0.0,
        }
        cost = _compute_substitution_cost(
            cl.bends[0],
            cl2.bends[1],
            cl,
            cl2,
            m1,
            m2,
            nw,
            weights=zero_weights,
        )
        self.assertAlmostEqual(cost, 0.0, places=6)

    def test_single_weight(self: Self) -> None:
        """Only one non-zero weight should produce a partial cost."""
        cl1 = self.__class__.cl_1
        cl2 = self.__class__.cl_2
        m1 = self.__class__.morpho_1
        m2 = self.__class__.morpho_2
        nw = self.__class__.norm_width
        w = {
            "apex_distance": 1.0,
            "amplitude": 0.0,
            "wavelength": 0.0,
            "curvature": 0.0,
            "asymmetry": 0.0,
            "sinuosity": 0.0,
        }
        cost_apex_only = _compute_substitution_cost(
            cl1.bends[0],
            cl2.bends[0],
            cl1,
            cl2,
            m1,
            m2,
            nw,
            weights=w,
        )
        cost_all = _compute_substitution_cost(
            cl1.bends[0],
            cl2.bends[0],
            cl1,
            cl2,
            m1,
            m2,
            nw,
        )
        self.assertLessEqual(cost_apex_only, cost_all + 1e-9)


class TestComputeGapPenalty(unittest.TestCase):
    """Tests for _compute_gap_penalty function."""

    cl: Centerline
    morpho: Morphometry

    @classmethod
    def setUpClass(cls) -> None:
        """Create a centerline for gap penalty tests."""
        cls.cl = _make_centerline(6, age=0)
        cls.morpho = Morphometry(cls.cl)

    def test_sinuosity_metric(self: Self) -> None:
        """Gap penalty with sinuosity metric returns positive value."""
        cl = self.__class__.cl
        morpho = self.__class__.morpho
        bend = cl.bends[0]
        penalty = _compute_gap_penalty(bend, cl, morpho, "sinuosity", 1.0)
        self.assertGreater(penalty, 0.0)

    def test_curvature_metric(self: Self) -> None:
        """Gap penalty with curvature metric returns positive value."""
        cl = self.__class__.cl
        morpho = self.__class__.morpho
        bend = cl.bends[0]
        penalty = _compute_gap_penalty(bend, cl, morpho, "curvature", 1.0)
        self.assertGreater(penalty, 0.0)

    def test_scale_factor(self: Self) -> None:
        """Doubling scale should double the penalty."""
        cl = self.__class__.cl
        morpho = self.__class__.morpho
        bend = cl.bends[0]
        p1 = _compute_gap_penalty(bend, cl, morpho, "sinuosity", 1.0)
        p2 = _compute_gap_penalty(bend, cl, morpho, "sinuosity", 2.0)
        self.assertAlmostEqual(p2, 2.0 * p1, places=6)

    def test_unknown_metric_raises(self: Self) -> None:
        """Unknown metric should raise ValueError."""
        cl = self.__class__.cl
        morpho = self.__class__.morpho
        bend = cl.bends[0]
        with self.assertRaises(ValueError):
            _compute_gap_penalty(bend, cl, morpho, "unknown_metric", 1.0)


class TestAlignBendSequences(unittest.TestCase):
    """Tests for align_bend_sequences function."""

    cl_1: Centerline
    cl_2: Centerline
    cl_few: Centerline
    norm_width: float

    @classmethod
    def setUpClass(cls) -> None:
        """Create centerlines for alignment tests."""
        cls.cl_1 = _make_centerline(6, age=0)
        cls.cl_2 = _make_centerline(6, age=1)
        cls.cl_few = _make_centerline(3, age=2)
        cls.norm_width = 100.0

    def test_identical_sequences(self: Self) -> None:
        """Aligning a sequence with itself gives all matches."""
        cl = self.__class__.cl_1
        nw = self.__class__.norm_width
        bends = cl.bends
        result = align_bend_sequences(cl, cl, nw)
        self.assertEqual(len(result.matches), len(bends))
        self.assertEqual(len(result.deletions), 0)
        self.assertEqual(len(result.insertions), 0)
        for b_id_t1, b_id_t2 in result.matches:
            self.assertEqual(b_id_t1, b_id_t2)

    def test_shifted_sequence_produces_matches(self: Self) -> None:
        """Two similar sine centerlines still produce matches."""
        cl1 = self.__class__.cl_1
        cl2 = self.__class__.cl_2
        nw = self.__class__.norm_width
        result = align_bend_sequences(cl1, cl2, nw)
        self.assertGreater(len(result.matches), 0)

    def test_deletion_when_fewer_bends_at_t2(self: Self) -> None:
        """T1 with more bends than T2 produces deletions."""
        cl1 = self.__class__.cl_1
        cl2 = self.__class__.cl_few
        nw = self.__class__.norm_width
        result = align_bend_sequences(cl1, cl2, nw)
        self.assertGreater(len(result.deletions), 0)

    def test_cost_matrix_shape(self: Self) -> None:
        """Cost matrix has shape (n+1, m+1)."""
        cl1 = self.__class__.cl_1
        cl2 = self.__class__.cl_2
        nw = self.__class__.norm_width
        result = align_bend_sequences(cl1, cl2, nw)
        n = len(cl1.bends)
        m = len(cl2.bends)
        self.assertEqual(result.cost_matrix.shape, (n + 1, m + 1))


class TestSplitMergeDetection(unittest.TestCase):
    """Tests for detect_splits_and_merges function."""

    cl_1: Centerline
    cl_2: Centerline
    norm_width: float

    @classmethod
    def setUpClass(cls) -> None:
        """Create centerlines for split/merge tests."""
        cls.cl_1 = _make_centerline(6, age=0)
        cls.cl_2 = _make_centerline(6, age=1)
        cls.norm_width = 100.0

    def test_no_splits_or_merges_on_identical_sequences(
        self: Self,
    ) -> None:
        """Identical sequences have no deletions/insertions."""
        cl = self.__class__.cl_1
        nw = self.__class__.norm_width
        alignment = align_bend_sequences(cl, cl, nw)
        result = detect_splits_and_merges(alignment, cl, cl, nw)
        self.assertEqual(len(result.splits), 0)
        self.assertEqual(len(result.merges), 0)

    def test_empty_result_type(self: Self) -> None:
        """Result is SplitMergeResult even when empty."""
        cl = self.__class__.cl_1
        nw = self.__class__.norm_width
        alignment = align_bend_sequences(cl, cl, nw)
        result = detect_splits_and_merges(alignment, cl, cl, nw)
        self.assertIsInstance(result, SplitMergeResult)


def _make_migrated_centerline(
    age: int,
    shift_x: float = 0.0,
    shift_y: float = 0.0,
    amplitude: float = 200.0,
    n_bends: int = 6,
) -> Centerline:
    """Build a sine centerline translated by (shift_x, shift_y).

    Args:
        age: Age of the centerline.
        shift_x: Translation along x (m).
        shift_y: Translation along y (m).
        amplitude: Sine amplitude (m).
        n_bends: Number of bends.

    Returns:
        Centerline with bends detected.
    """
    x = np.linspace(0, n_bends * np.pi * 100, 600)
    y = amplitude * np.sin(x / 100.0)
    dataset = pd.DataFrame(
        {
            "Cart_abscissa": x + shift_x,
            "Cart_ordinate": y + shift_y,
            "Elevation": np.zeros_like(x),
        }
    )
    return Centerline(
        age,
        dataset,
        spacing=50.0,
        smooth_distance=100.0,
        sinuo_thres=1.0,
        n=2,
    )


class TestSubstitutionCostRobustness(unittest.TestCase):
    """Tests for NaN handling and weight merging in substitution cost."""

    cl_1: Centerline
    cl_2: Centerline
    morpho_1: Morphometry
    morpho_2: Morphometry
    norm_width: float

    @classmethod
    def setUpClass(cls) -> None:
        """Create two centerlines."""
        cls.cl_1 = _make_centerline(6, age=0)
        cls.cl_2 = _make_migrated_centerline(1, shift_y=50.0)
        cls.morpho_1 = Morphometry(cls.cl_1)
        cls.morpho_2 = Morphometry(cls.cl_2)
        cls.norm_width = 100.0

    def _cost(
        self: Self,
        bend_i: object,
        bend_j: object,
        weights: dict[str, float] | None = None,
    ) -> float:
        """Substitution cost helper between cl_1 and cl_2 bends.

        Args:
            bend_i: Bend of cl_1.
            bend_j: Bend of cl_2.
            weights: Optional weights.

        Returns:
            Substitution cost.
        """
        cls = self.__class__
        return _compute_substitution_cost(
            bend_i,  # type: ignore[arg-type]
            bend_j,  # type: ignore[arg-type]
            cls.cl_1,
            cls.cl_2,
            cls.morpho_1,
            cls.morpho_2,
            cls.norm_width,
            weights,
        )

    def test_partial_weights_merged_with_defaults(self: Self) -> None:
        """Missing weight keys default to 1.0, not 0.0."""
        b1 = self.__class__.cl_1.bends[1]
        b2 = self.__class__.cl_2.bends[1]
        cost_default = self._cost(b1, b2)
        cost_partial = self._cost(b1, b2, {"apex_distance": 1.0})
        self.assertGreater(cost_default, 0.0)
        self.assertAlmostEqual(cost_partial, cost_default, places=9)
        cost_double = self._cost(b1, b2, {"apex_distance": 2.0})
        cost_apex = self._cost(
            b1,
            b2,
            {
                "apex_distance": 1.0,
                "amplitude": 0.0,
                "wavelength": 0.0,
                "curvature": 0.0,
                "asymmetry": 0.0,
                "sinuosity": 0.0,
            },
        )
        self.assertAlmostEqual(cost_double, cost_default + cost_apex, places=9)

    def test_missing_apex_gives_finite_penalty(self: Self) -> None:
        """A bend without apex gives a finite, penalised cost."""
        b1 = copy.copy(self.__class__.cl_1.bends[1])
        b1.index_apex = -1
        b2 = self.__class__.cl_2.bends[1]
        cost = self._cost(b1, b2)
        self.assertTrue(np.isfinite(cost))
        self.assertGreaterEqual(cost, NAN_METRIC_PENALTY)
        # apex distance term skipped when its weight is 0
        cost_no_apex = self._cost(b1, b2, {"apex_distance": 0.0})
        self.assertTrue(np.isfinite(cost_no_apex))
        self.assertLess(cost_no_apex, NAN_METRIC_PENALTY)

    def test_nan_metric_replaced_by_penalty(self: Self) -> None:
        """NaN metric differences are replaced by a finite penalty."""
        cls = self.__class__
        b1 = cls.cl_1.bends[1]
        b2 = cls.cl_2.bends[1]
        ref = self._cost(b1, b2, {"sinuosity": 0.0})
        with mock.patch.object(
            cls.morpho_1, "compute_bend_sinuosity", return_value=np.nan
        ):
            cost = self._cost(b1, b2)
        self.assertAlmostEqual(cost, ref + NAN_METRIC_PENALTY, places=6)

    def test_zero_weight_term_not_computed(self: Self) -> None:
        """Terms with zero weight are skipped (NaN cannot leak in)."""
        cls = self.__class__
        b1 = cls.cl_1.bends[1]
        b2 = cls.cl_2.bends[1]
        ref = self._cost(b1, b2, {"sinuosity": 0.0})
        with mock.patch.object(
            cls.morpho_1,
            "compute_bend_sinuosity",
            side_effect=AssertionError("should not be called"),
        ):
            cost = self._cost(b1, b2, {"sinuosity": 0.0})
        self.assertAlmostEqual(cost, ref, places=9)

    def test_alignment_finite_with_missing_apex(self: Self) -> None:
        """DP matrix stays finite when a bend has no apex."""
        cl = _make_centerline(6, age=0)
        cl.bends[2].index_apex = -1
        result = align_bend_sequences(cl, self.__class__.cl_2, 100.0)
        self.assertTrue(np.all(np.isfinite(result.cost_matrix)))
        self.assertNotIn(2, [m[0] for m in result.matches])


class TestGapPenaltyScaling(unittest.TestCase):
    """Tests for gap penalty normalisation and defaults."""

    def test_curvature_gap_is_dimensionless(self: Self) -> None:
        """Curvature gap penalty scales with norm_width."""
        cl = _make_centerline(6, age=0)
        morpho = Morphometry(cl)
        bend = cl.bends[1]
        p1 = _compute_gap_penalty(bend, cl, morpho, "curvature", 2.0, 1.0)
        p100 = _compute_gap_penalty(bend, cl, morpho, "curvature", 2.0, 100.0)
        self.assertAlmostEqual(p100, 100.0 * p1, places=9)
        curv = cl.get_bend_curvature_filtered(bend.id)
        expected = 2.0 * float(np.mean(np.abs(curv))) * 100.0
        self.assertAlmostEqual(p100, expected, places=9)

    def test_default_gap_penalty_scale(self: Self) -> None:
        """Default gap penalty scale is 2.5 everywhere."""
        for func in (align_bend_sequences, build_temporal_graph):
            param = inspect.signature(func).parameters["gap_penalty_scale"]
            self.assertEqual(param.default, 2.5)

    def test_default_spatial_tolerance(self: Self) -> None:
        """Default spatial tolerance is 2.0 as documented."""
        for func in (detect_splits_and_merges, build_temporal_graph):
            param = inspect.signature(func).parameters["spatial_tolerance"]
            self.assertEqual(param.default, 2.0)

    def test_migrated_bends_are_matched(self: Self) -> None:
        """Bends migrated by 2 widths with +10% amplitude are matched."""
        width = 25.0
        cl1 = _make_migrated_centerline(0)
        cl2 = _make_migrated_centerline(
            1, shift_x=2.0 * width, amplitude=220.0
        )
        self.assertEqual(len(cl1.bends), len(cl2.bends))
        result = align_bend_sequences(cl1, cl2, width)
        self.assertEqual(result.matches, [(b.id, b.id) for b in cl1.bends])
        self.assertEqual(result.deletions, [])
        self.assertEqual(result.insertions, [])
        # the former default scale (1.0) loses most of the matches
        result_old = align_bend_sequences(
            cl1, cl2, width, gap_penalty_scale=1.0
        )
        self.assertLess(len(result_old.matches), len(cl1.bends))

    def test_match_costs(self: Self) -> None:
        """Each match stores its substitution cost."""
        cl1 = _make_migrated_centerline(0)
        cl2 = _make_migrated_centerline(1, shift_y=50.0)
        result = align_bend_sequences(cl1, cl2, 100.0)
        self.assertEqual(len(result.match_costs), len(result.matches))
        dp = result.cost_matrix
        for (i, j), cost in zip(
            result.matches, result.match_costs, strict=True
        ):
            self.assertAlmostEqual(cost, dp[i + 1, j + 1] - dp[i, j], places=6)


class TestSplitClaims(unittest.TestCase):
    """Tests that inserted bends belong to a single split."""

    def test_split_children_not_shared(self: Self) -> None:
        """Two deletions near the same insertions do not share them."""
        cl = _make_centerline(6, age=0)
        nw = 100.0
        alignment = BendAlignment(
            matches=[(0, 0), (5, 5)],
            deletions=[2, 3],
            insertions=[1, 2, 3, 4],
            cost_matrix=np.zeros((1, 1)),
            total_cost=0.0,
        )
        # tolerance (550 m) large enough for each deleted bend to reach 3
        # inserted bends
        result = detect_splits_and_merges(
            alignment, cl, cl, nw, spatial_tolerance=5.5
        )
        children = [uid for _, dst in result.splits for uid in dst]
        self.assertGreater(len(result.splits), 0)
        self.assertEqual(len(children), len(set(children)))
        merged_children = {dst for _, dst in result.merges}
        self.assertEqual(merged_children & set(children), set())


class TestBendGraph(unittest.TestCase):
    """Tests for BendGraph edge costs and bend evolution indexes."""

    def test_edge_cost_is_pair_cost(self: Self) -> None:
        """Matched edges store pair costs; total cost at graph level."""
        nw = 100.0
        cl1 = _make_migrated_centerline(0)
        cl2 = _make_migrated_centerline(1, shift_y=50.0)
        collection = SimpleNamespace(
            centerlines={0: cl1, 1: cl2},
            get_all_ages=lambda: np.array([0, 1]),
        )
        graph = build_temporal_graph(collection, nw)  # type: ignore
        alignment = align_bend_sequences(cl1, cl2, nw)
        self.assertAlmostEqual(
            graph.graph["alignment_costs"][(0, 1)], alignment.total_cost
        )
        self.assertGreater(len(alignment.matches), 1)
        for (i, j), cost in zip(
            alignment.matches, alignment.match_costs, strict=True
        ):
            edge = graph.edges[cl1.bends[i].uid, cl2.bends[j].uid]
            self.assertAlmostEqual(edge["cost"], cost)
        total_edges = sum(
            d["cost"] for _, _, d in graph.edges(data=True) if "cost" in d
        )
        self.assertLessEqual(total_edges, alignment.total_cost + 1e-9)

    def test_bend_indexes_sorted(self: Self) -> None:
        """bend_indexes ages and ids are sorted in ascending order."""
        graph = nx.DiGraph()
        young = get_bend_uid(0, 2)
        mid = [get_bend_uid(3, 1), get_bend_uid(1, 1)]
        old = [get_bend_uid(5, 0), get_bend_uid(2, 0)]
        graph.add_edge(mid[0], young)
        graph.add_edge(mid[1], young)
        graph.add_edge(old[0], mid[0])
        graph.add_edge(old[1], mid[1])
        collection = SimpleNamespace(
            centerlines={
                2: SimpleNamespace(bends=[SimpleNamespace(uid=young)])
            },
            get_all_ages=lambda: np.array([0, 1, 2]),
        )
        evols = build_bend_evolutions_from_graph(
            graph,
            collection,  # type: ignore[arg-type]
        )
        self.assertEqual(len(evols), 1)
        bend_indexes = evols[0].bend_indexes
        self.assertEqual(list(bend_indexes.keys()), [0, 1, 2])
        self.assertEqual(bend_indexes[0], [2, 5])
        self.assertEqual(bend_indexes[1], [1, 3])
        self.assertEqual(bend_indexes[2], [0])


if __name__ == "__main__":
    unittest.main()
