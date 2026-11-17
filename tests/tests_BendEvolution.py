# SPDX-FileCopyrightText: Copyright 2025 Martin Lemay <martin.lemay@mines-paris.org>
# SPDX-FileContributor: Martin Lemay
# ruff: noqa: E402 # disable Module level import not at top of file

__doc__ = """
Tests functions for Bend class.
"""

import unittest
from typing import Self, Sequence

from pybend.model.BendEvolution import BendEvolution

# inputs
bend_indexes: dict[int, list[int]] = {10: [5, 6], 20: [7], 30: [4, 5]}
ide: int = 11
order: int = 1


class TestsBendEvolution(unittest.TestCase):
    def test_Bend_evolution(self: Self) -> None:
        """Test of BendEvolution instanciation."""
        bend_evol: BendEvolution
        try:
            bend_evol = BendEvolution(bend_indexes, ide, order, False)
        except Exception as err:
            print(err)
            self.fail("Enable to create Bend object.")

        for key, val in bend_evol.bend_indexes.items():
            self.assertIn(key, bend_indexes.keys())
            indexes: Sequence[int] = bend_indexes[key]
            self.assertSequenceEqual(indexes, val)

        self.assertEqual(bend_evol.id, ide)
        self.assertEqual(bend_evol.order, order)
        self.assertFalse(bend_evol.isvalid)
        self.assertSequenceEqual(
            list(bend_indexes.keys()), list(bend_evol.get_all_ages())
        )

    def test_set_is_valid(self: Self) -> None:
        """Test of BendEvolution.set_is_valid() method."""
        bend_evol: BendEvolution = BendEvolution(
            bend_indexes, ide, order, False
        )
        if bend_evol is None:
            self.skipTest("BendEvolution object was not created.")
        bend_evol.set_is_valid(1)
        self.assertTrue(bend_evol.isvalid)

        bend_evol.set_is_valid(10)
        self.assertFalse(bend_evol.isvalid)

    def test_is_valid(self: Self) -> None:
        """Test of BendEvolution._check_is_valid() method."""
        bend_evol: BendEvolution = BendEvolution(
            bend_indexes, ide, order, False
        )
        if bend_evol is None:
            self.skipTest("BendEvolution object was not created.")

        ret: bool = bend_evol._check_is_valid(1)
        self.assertTrue(ret)

        ret = bend_evol._check_is_valid(10)
        self.assertFalse(ret)

    def test_get_number_of_bends(self: Self) -> None:
        """Test of BendEvolution.get_number_of_bends() method."""
        bend_evol: BendEvolution = BendEvolution(
            bend_indexes, ide, order, False
        )
        self.assertEqual(bend_evol.get_number_of_bends(), 5)
        self.assertEqual(len(bend_evol.get_all_ages()), 3)

    def test_bend_indexes_sorted(self: Self) -> None:
        """Test that bend indexes are sorted by age and by bend index."""
        unsorted: dict[int, list[int]] = {30: [5, 4], 10: [6, 5], 20: [7]}
        bend_evol: BendEvolution = BendEvolution(unsorted, ide, order)
        self.assertListEqual(list(bend_evol.bend_indexes.keys()), [10, 20, 30])
        self.assertDictEqual(
            bend_evol.bend_indexes, {10: [5, 6], 20: [7], 30: [4, 5]}
        )
        # input is not modified
        self.assertListEqual(unsorted[30], [5, 4])


if __name__ == "__main__":
    unittest.main()
