# SPDX-FileCopyrightText: Copyright 2025 Martin Lemay <martin.lemay@mines-paris.org>
# SPDX-FileContributor: Martin Lemay
# ruff: noqa: E402 # disable Module level import not at top of file

"""Tests for pybend.utils (global parameters and logging)."""

import unittest
from typing import Self

import pybend.utils.globalParameters as gp
from pybend.utils.logging import getLogger


class TestsGlobalParameters(unittest.TestCase):
    """Tests of the number of processors setter."""

    def tearDown(self: Self) -> None:
        """Reset the number of processors."""
        gp.set_nb_procs(1)

    def test_set_nb_procs_valid(self: Self) -> None:
        """Valid values are stored."""
        gp.set_nb_procs(1)
        self.assertEqual(gp.get_nb_procs(), 1)

    def test_set_nb_procs_invalid(self: Self) -> None:
        """Values lower than 1 raise ValueError."""
        for nb in (0, -1):
            with self.assertRaises(ValueError):
                gp.set_nb_procs(nb)
        self.assertGreaterEqual(gp.get_nb_procs(), 1)


class TestsLogging(unittest.TestCase):
    """Tests of the logger factory."""

    def test_handlers_not_duplicated(self: Self) -> None:
        """Repeated getLogger calls do not add handlers."""
        log1 = getLogger("pybend-test-logger")
        nb_handlers = len(log1.handlers)
        log2 = getLogger("pybend-test-logger")
        self.assertIs(log1, log2)
        self.assertEqual(nb_handlers, 1)
        self.assertEqual(len(log2.handlers), 1)
        self.assertFalse(log2.propagate)


if __name__ == "__main__":
    unittest.main()
