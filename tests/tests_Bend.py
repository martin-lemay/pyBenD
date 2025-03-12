# SPDX-FileCopyrightText: Copyright 2025 Martin Lemay <martin.lemay@mines-paris.org>
# SPDX-FileContributor: Martin Lemay
# ruff: noqa: E402 # disable Module level import not at top of file

__doc__ = """
Tests functions for Bend class.
"""

import unittest
from typing import Self

from pybend.model.Bend import Bend, get_bend_uid, parse_bend_uid
from pybend.model.enumerations import BendSide

# input
bend_id1: int = 5
age1: int = 1500
index_inflex_up1: int = 12
index_inflex_down1: int = 25
side1: BendSide = BendSide.UP

bend_id2: int = 3
age2: int = 2800
index_inflex_up2: int = 38
index_inflex_down2: int = 56
side2: BendSide = BendSide.DOWN

# expected outputs
uid_out1: int = 15000005
uid_out2: int = 28000003


class TestsBend(unittest.TestCase):
    def test_get_bend_uid(self: Self) -> None:
        """Test of get_bend_uid function."""
        uid: int = get_bend_uid(bend_id1, age1)
        self.assertEqual(uid, uid_out1, "uid1 is wrong.")

        uid = get_bend_uid(bend_id2, age2)
        self.assertEqual(uid, uid_out2, "uid2 is wrong.")

    def test_parse_bend_uid(self: Self) -> None:
        """Test of parse_bend_uid function."""
        age_obs, bend_id_obs = parse_bend_uid(uid_out1)
        self.assertEqual(age1, age_obs, "Age1 is wrong.")
        self.assertEqual(bend_id1, bend_id_obs, "Bend_id1 is wrong")

        age_obs, bend_id_obs = parse_bend_uid(uid_out2)
        self.assertEqual(age2, age_obs, "Age2 is wrong.")
        self.assertEqual(bend_id2, bend_id_obs, "Bend_id2 is wrong")

    def test_Bend(self: Self) -> None:
        """Test of Bend instanciation."""
        for bend_id, index_inflex_up, index_inflex_down, age, side, uid in (
            (
                bend_id1,
                index_inflex_up1,
                index_inflex_down1,
                age1,
                side1,
                uid_out1,
            ),
            (
                bend_id2,
                index_inflex_up2,
                index_inflex_down2,
                age2,
                side2,
                uid_out2,
            ),
        ):
            bend: Bend
            try:
                bend = Bend(
                    bend_id,
                    index_inflex_up,
                    index_inflex_down,
                    age,
                    side,
                    True,
                )
            except Exception as err:
                print(err)
                self.fail("Enable to create Bend object.")

            self.assertEqual(bend.id, bend_id)
            self.assertEqual(bend.age, age)
            self.assertEqual(bend.uid, uid)
            self.assertTrue(bend.isvalid)
            self.assertEqual(bend.side, side)
            self.assertEqual(bend.index_inflex_up, index_inflex_up)
            self.assertEqual(bend.index_inflex_down, index_inflex_down)

    def test_bend_get_nb_points(self: Self) -> None:
        """Test of Bend.get_nb_points() method."""
        bend: Bend = Bend(
            bend_id1, index_inflex_up1, index_inflex_down1, age1, side1, True
        )
        self.assertEqual(
            bend.get_nb_points(), index_inflex_down1 - index_inflex_up1 + 1
        )

    def test_bend_repr(self: Self) -> None:
        """Test of Bend.__repr__() method."""
        for bend_id, index_inflex_up, index_inflex_down, age, side in (
            (bend_id1, index_inflex_up1, index_inflex_down1, age1, side1),
            (bend_id2, index_inflex_up2, index_inflex_down2, age2, side2),
        ):
            bend: Bend = Bend(
                bend_id, index_inflex_up, index_inflex_down, age, side, True
            )
            if bend is None:
                self.skipTest("Bend object was not created.")
            self.assertSequenceEqual(str(bend), str(age) + "-" + str(bend_id))

    def test_bend_add(self: Self) -> None:
        """Test of Bend.__add__() method."""
        bend1 = Bend(
            bend_id1, index_inflex_up1, index_inflex_down1, age1, side1, True
        )
        bend2 = Bend(
            bend_id2, index_inflex_up2, index_inflex_down2, age2, side2, True
        )
        if (bend1 is None) or (bend2 is None):
            self.skipTest("Bend object was not created.")
        bend: Bend = bend1 + bend2

        self.assertEqual(bend.id, bend_id1)
        self.assertEqual(bend.age, age1)
        self.assertEqual(bend.uid, uid_out1)
        self.assertTrue(bend.isvalid)
        self.assertEqual(bend.side, side1)
        self.assertEqual(bend.index_inflex_up, index_inflex_up1)
        self.assertEqual(bend.index_inflex_down, index_inflex_down2)
        self.assertEqual(
            bend.get_nb_points(), index_inflex_down2 - index_inflex_up1 + 1
        )

    def test_equality(self: Self) -> None:
        """Test of Bend.__eq__() method."""
        bend1 = Bend(
            bend_id1, index_inflex_up1, index_inflex_down1, age1, side1, True
        )
        bend2 = Bend(
            bend_id2, index_inflex_up2, index_inflex_down2, age2, side2, True
        )
        if (bend1 is None) or (bend2 is None):
            self.skipTest("Bend object was not created.")
        self.assertEqual(bend1, bend1)
        self.assertNotEqual(bend1, bend2)

    def test_add_bend_connection_next(self: Self) -> None:
        """Test of Bend.add_bend_connection_next() method."""
        for bend_id, index_inflex_up, index_inflex_down, age, side in (
            (bend_id1, index_inflex_up1, index_inflex_down1, age1, side1),
            (bend_id2, index_inflex_up2, index_inflex_down2, age2, side2),
        ):
            bend: Bend = Bend(
                bend_id, index_inflex_up, index_inflex_down, age, side, True
            )
            if bend is None:
                self.skipTest("Bend object was not created.")
            bend.add_bend_connection_next(uid_out2)
            self.assertTrue(bend.bend_uid_next is not None)
            assert bend.bend_uid_next is not None  # to avoid type warning
            self.assertEqual(len(bend.bend_uid_next), 1)
            self.assertEqual(bend.bend_uid_next[0], uid_out2)

    def test_add_bend_connection_prev(self: Self) -> None:
        """Test of Bend.add_bend_connection_prev() method."""
        for bend_id, index_inflex_up, index_inflex_down, age, side in (
            (bend_id1, index_inflex_up1, index_inflex_down1, age1, side1),
            (bend_id2, index_inflex_up2, index_inflex_down2, age2, side2),
        ):
            bend: Bend = Bend(
                bend_id, index_inflex_up, index_inflex_down, age, side, True
            )
            if bend is None:
                self.skipTest("Bend object was not created.")

            bend.add_bend_connection_prev(uid_out2)
            self.assertTrue(bend.bend_uid_prev is not None)
            assert bend.bend_uid_prev is not None  # to avoid type warning
            self.assertEqual(len(bend.bend_uid_prev), 1)
            self.assertEqual(bend.bend_uid_prev[0], uid_out2)


if __name__ == "__main__":
    unittest.main()
