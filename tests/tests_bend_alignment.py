# SPDX-FileCopyrightText: Copyright 2025 Martin Lemay <martin.lemay@mines-paris.org>
# SPDX-FileContributor: Martin Lemay
# ruff: noqa: E402 # disable Module level import not at top of file

"""Tests for bend_alignment dataclasses and cost/penalty functions."""

import unittest
from typing import Self

import numpy as np
import pandas as pd

from pybend.algorithms.bend_alignment import (
    BendAlignment,
    SplitMergeResult,
    _compute_gap_penalty,
    _compute_substitution_cost,
    align_bend_sequences,
    detect_splits_and_merges,
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


if __name__ == "__main__":
    unittest.main()
