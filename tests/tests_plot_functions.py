# SPDX-FileCopyrightText: Copyright 2025 Martin Lemay <martin.lemay@mines-paris.org>
# SPDX-FileContributor: Martin Lemay
# ruff: noqa: E402 # disable Module level import not at top of file

"""Smoke tests for plot_functions (Agg backend)."""

import copy
import unittest
from types import SimpleNamespace
from typing import Self

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pybend.algorithms.plot_functions import (
    _plot_warping,
    plot_bends,
    plot_section,
    plot_versus_curvilinear,
)
from pybend.model.Centerline import Centerline
from pybend.utils.logging import ERROR, logger

logger.setLevel(ERROR)


def _make_centerline(age: int = 0) -> Centerline:
    """Build a sine centerline with bends.

    Args:
        age: Age of the centerline.

    Returns:
        Centerline with bends detected.
    """
    x = np.linspace(0, 3 * np.pi * 100, 500)
    y = 200.0 * np.sin(x / 100.0)
    dataset = pd.DataFrame(
        {
            "Cart_abscissa": x,
            "Cart_ordinate": y,
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


class TestsPlotFunctions(unittest.TestCase):
    """Smoke tests of plotting helpers."""

    cl: Centerline

    @classmethod
    def setUpClass(cls) -> None:
        """Create a centerline."""
        cls.cl = _make_centerline()

    def tearDown(self: Self) -> None:
        """Close figures."""
        plt.close("all")

    def test_plot_bends_draws_on_given_axes(self: Self) -> None:
        """Arrows and texts are drawn on ax with annot_text_size."""
        cl = self.__class__.cl
        bends = [copy.copy(bend) for bend in cl.bends]
        bends[0].index_apex = -1  # no apex: must be skipped
        _, (ax_other, ax) = plt.subplots(1, 2)
        plt.sca(ax_other)
        plot_bends(
            ax,
            (cl.cl_points,),
            bends,
            domain=((-1e6, 1e6), (-1e6, 1e6)),
            annotate=True,
            plot_inflex=True,
            plot_normal=True,
            annot_text_size=7,
        )
        self.assertEqual(len(ax_other.texts), 0)
        self.assertEqual(len(ax_other.patches), 0)
        self.assertEqual(len(ax.texts), len(bends) - 1)
        for text in ax.texts:
            self.assertEqual(text.get_fontsize(), 7)
        self.assertGreater(len(ax.patches), 0)

    def test_plot_bends_rotate_centered_on_chord(self: Self) -> None:
        """Rotated bends are centred on the chord middle point."""
        cl = self.__class__.cl
        bend = cl.bends[1]
        _, ax = plt.subplots()
        plot_bends(ax, (cl.cl_points,), [bend], rotate=True)
        xdata = np.asarray(ax.lines[0].get_xdata())
        ydata = np.asarray(ax.lines[0].get_ydata())
        mid = (
            np.array([xdata[0], ydata[0]]) + np.array([xdata[-1], ydata[-1]])
        ) / 2.0
        self.assertTrue(np.allclose(mid, 0.0, atol=1e-6))
        # inflection points aligned along the horizontal axis
        self.assertAlmostEqual(ydata[0], ydata[-1], places=6)

    def test_plot_section_colormap(self: Self) -> None:
        """plot_section with color_same_bend=False uses a real colormap."""
        cl_pt = SimpleNamespace(get_property=lambda name: 1.0)
        isoline = SimpleNamespace(
            points=[(0.0, 0.0), (1.0, 1.0), (2.0, 0.0)], cl_pt_ref=cl_pt
        )
        section = SimpleNamespace(
            isolines=[isoline, isoline], isolines_origin=[(0, 0), (1, 1)]
        )
        _, ax = plt.subplots()
        plot_section(section, ax, color_same_bend=False)  # type: ignore
        plot_section(
            section,  # type: ignore[arg-type]
            ax,
            color_same_bend=False,
            cmap_name="viridis",
        )
        self.assertEqual(len(ax.patches), 4)

    def test_plot_versus_curvilinear(self: Self) -> None:
        """Valid number of curves is accepted, ylim is not forced."""
        abscissa = np.linspace(0, 10, 20)
        curves = [10.0 * np.sin(abscissa), 5.0 * np.cos(abscissa)]
        captured: dict[str, tuple[float, float]] = {}
        orig_close = plt.close

        def _close(*args: object) -> None:
            fig = plt.gcf()
            if fig.axes:
                captured["ylim"] = fig.axes[0].get_ylim()
            orig_close(*args)  # type: ignore[arg-type]

        plt.close = _close  # type: ignore[assignment]
        try:
            plot_versus_curvilinear(
                "", abscissa, curves, ["a", "b"], [curves[0]], ["c"]
            )
        finally:
            plt.close = orig_close  # type: ignore[assignment]
        self.assertGreater(captured["ylim"][1], 1.0)
        with self.assertRaises(AssertionError):
            plot_versus_curvilinear(
                "",
                abscissa,
                [curves[0]] * 8,
                ["a"] * 8,
                [],
                [],
            )

    def test_plot_warping_indexes(self: Self) -> None:
        """Missing ages are skipped and last index is included."""

        def _cl(nb: int, prev: list[int]) -> SimpleNamespace:
            pts = [SimpleNamespace(pt=np.array([i, 0.0])) for i in range(nb)]
            return SimpleNamespace(
                cl_points=pts,
                index_cl_pts_prev_centerline=prev,
                get_nb_points=lambda: nb,
            )

        centerlines = {
            0: _cl(5, []),
            1: _cl(5, [0, 1, 2, 3, 4]),
            2: _cl(5, [0, 1, 2, 3, 4]),
        }
        collec = SimpleNamespace(
            centerlines=centerlines,
            get_all_ages=lambda: np.array([0, 1, 2]),
        )
        _, ax = plt.subplots()
        # age 2 is missing from indexes: skipped without error
        _plot_warping(ax, collec, {1: (1, 3)})  # type: ignore[arg-type]
        self.assertEqual(len(ax.lines), 3)  # indexes 1, 2, 3 inclusive
        _, ax2 = plt.subplots()
        _plot_warping(ax2, collec)  # type: ignore[arg-type]
        self.assertEqual(len(ax2.lines), 10)


if __name__ == "__main__":
    unittest.main()
