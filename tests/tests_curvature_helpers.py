# SPDX-FileCopyrightText: Copyright 2025 Martin Lemay
# SPDX-FileContributor: Martin Lemay
# ruff: noqa: E402

"""Tests for low-level curvature helper functions."""

import unittest
from typing import Self

import numpy as np

from pybend.algorithms.centerline_process_functions import (
    compute_bend_apex_from_curvature,
    compute_bend_side_from_curvature,
)
from pybend.model.enumerations import BendSide


class TestComputeBendSideFromCurvature(unittest.TestCase):
    """Tests for compute_bend_side_from_curvature."""

    def test_positive_curvature_returns_up(self: Self) -> None:
        """Positive curvature sum with valid sinuosity returns UP."""
        curvature = np.array([0.1, 0.3, 0.5, 0.3, 0.1])
        result = compute_bend_side_from_curvature(
            curvature, sinuosity=1.5, sinuo_thres=1.05,
        )
        self.assertEqual(result, BendSide.UP)

    def test_negative_curvature_returns_down(self: Self) -> None:
        """Negative curvature sum with valid sinuosity returns DOWN."""
        curvature = np.array([-0.1, -0.3, -0.5, -0.3, -0.1])
        result = compute_bend_side_from_curvature(
            curvature, sinuosity=1.5, sinuo_thres=1.05,
        )
        self.assertEqual(result, BendSide.DOWN)

    def test_low_sinuosity_returns_straight(self: Self) -> None:
        """Sinuosity below threshold returns STRAIGHT."""
        curvature = np.array([0.1, 0.3, 0.5, 0.3, 0.1])
        result = compute_bend_side_from_curvature(
            curvature, sinuosity=1.01, sinuo_thres=1.05,
        )
        self.assertEqual(result, BendSide.STRAIGHT)

    def test_zero_curvature_returns_down(self: Self) -> None:
        """Zero curvature sum (not > 0) returns DOWN."""
        curvature = np.array([0.1, -0.1, 0.0])
        result = compute_bend_side_from_curvature(
            curvature, sinuosity=1.5, sinuo_thres=1.05,
        )
        self.assertEqual(result, BendSide.DOWN)


class TestComputeBendApexFromCurvature(unittest.TestCase):
    """Tests for compute_bend_apex_from_curvature."""

    def test_single_peak(self: Self) -> None:
        """Single-peak curvature returns the peak index."""
        curvature = np.array(
            [0.01, 0.05, 0.1, 0.5, 0.1, 0.05, 0.01],
        )
        result = compute_bend_apex_from_curvature(curvature, n=2.0)
        self.assertEqual(result, 3)

    def test_symmetric_curvature(self: Self) -> None:
        """Symmetric curvature returns middle index."""
        curvature = np.array([0.2, 0.5, 0.5, 0.2])
        result = compute_bend_apex_from_curvature(curvature, n=2.0)
        # Median of symmetric dist falls at index 1 or 2
        self.assertIn(result, [1, 2])

    def test_left_skewed(self: Self) -> None:
        """Left-skewed curvature returns index left of center."""
        curvature = np.array([0.5, 0.3, 0.1, 0.05, 0.01])
        result = compute_bend_apex_from_curvature(curvature, n=2.0)
        self.assertLess(result, 2)


if __name__ == "__main__":
    unittest.main()
