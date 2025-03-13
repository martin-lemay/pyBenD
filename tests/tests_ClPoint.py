# SPDX-FileCopyrightText: Copyright 2025 Martin Lemay <martin.lemay@mines-paris.org>
# SPDX-FileContributor: Martin Lemay
# ruff: noqa: E402 # disable Module level import not at top of file

__doc__ = """
Tests functions for ClPoint class
"""
import unittest

import numpy as np
import numpy.typing as npt
import pandas as pd  # type: ignore[import-untyped]
from typing_extensions import Self

from pybend.model.ClPoint import ClPoint
from pybend.model.enumerations import PropertyNames

# inputs
columns = (
    PropertyNames.CURVILINEAR_ABSCISSA.value,
    PropertyNames.CARTESIAN_ABSCISSA.value,
    PropertyNames.CARTESIAN_ORDINATE.value,
    PropertyNames.ELEVATION.value,
    PropertyNames.CURVATURE.value,
    PropertyNames.VELOCITY.value,
    PropertyNames.DEPTH_MEAN.value,
    PropertyNames.WIDTH.value,
    PropertyNames.VELOCITY_PERTURBATION.value,
    "Property1",
    "Property2",
    "Property3",
)
ide1: str = "1"
age1: int = 100
data1: npt.NDArray[np.float64] = np.array(
    [1.2, 152.0, 652.0, 2.2, 0.31, 0.8, 1.1, 30.2, 1.3, 2.0, 3.0, 4.0]
)
dataset1: pd.Series = pd.Series(data1, index=columns)

ide2: str = "2"
age2: int = 200
data2: npt.NDArray[np.float64] = np.array(
    [1.8, 160.0, 648.0, 1.9, 0.40, 0.6, 1.2, 35.1, 1.8, 7.0, 8.0, 9.0]
)
dataset2: pd.Series = pd.Series(data2, index=columns)

ide3: str = "3"
age3: int = 300
data3: npt.NDArray[np.float64] = np.array(
    [1.5, 163.0, 661.0, 2.5, 0.25, 0.3, 2.0, 25.8, -1.5, 5.0, 4.0, 7.0]
)
dataset3: pd.Series = pd.Series(data3, index=columns)


def create_cl_point(dataset: pd.Series, ide: str, age: int) -> ClPoint:
    """Create ClPoint object.

    Args:
        dataset (pd.Series): data
        ide (str): id
        age (int): age

    Returns:
        ClPoint: ClPoint object
    """
    return ClPoint(ide, age, dataset)


class TestsClPoint(unittest.TestCase):
    def test_create_cl_point(self: Self) -> None:
        """Test of create_cl_point function."""
        for ide, age, dataset in zip(
            (ide1, ide2, ide3),
            (age1, age2, age3),
            (dataset1, dataset2, dataset3),
            strict=True,
        ):
            cl_pt: ClPoint = create_cl_point(dataset, ide, age)
            self.assertTrue(
                cl_pt is not None,
                "Channel_point must be instantiated and not null",
            )

    def test_id(self: Self) -> None:
        """Test of cl_point if."""
        for ide, age, dataset in zip(
            (ide1, ide2, ide3),
            (age1, age2, age3),
            (dataset1, dataset2, dataset3),
            strict=True,
        ):
            cl_pt: ClPoint = create_cl_point(dataset, ide, age)
            self.assertEqual(
                cl_pt._id, ide, f"Id test: Channel point id must be {ide}"
            )

    def test_age(self: Self) -> None:
        """Test of create_cl_point age."""
        for ide, age, dataset in zip(
            (ide1, ide2, ide3),
            (age1, age2, age3),
            (dataset1, dataset2, dataset3),
            strict=True,
        ):
            cl_pt: ClPoint = create_cl_point(dataset, ide, age)
            self.assertEqual(
                cl_pt._age, age, f"Age test: Channel point age must be {age}"
            )

    def test_coordinates(self: Self) -> None:
        """Test of create_cl_point coordinates."""
        for ide, age, dataset in zip(
            (ide1, ide2, ide3),
            (age1, age2, age3),
            (dataset1, dataset2, dataset3),
            strict=True,
        ):
            cl_pt = create_cl_point(dataset, ide, age)
            self.assertEqual(
                cl_pt._s,
                dataset[PropertyNames.CURVILINEAR_ABSCISSA.value],
                "Coordinate test: Curv_abscissa must be equal to %.3f"
                % dataset[PropertyNames.CURVILINEAR_ABSCISSA.value],
            )
            self.assertEqual(
                cl_pt.pt[0],
                dataset[PropertyNames.CARTESIAN_ABSCISSA.value],
                "Coordinate test: Cart_abscissa must be equal to %.3f"
                % dataset[PropertyNames.CARTESIAN_ABSCISSA.value],
            )
            self.assertEqual(
                cl_pt.pt[1],
                dataset[PropertyNames.CARTESIAN_ORDINATE.value],
                "Coordinate test: Cart_ordinate must be equal to %.3f"
                % dataset[PropertyNames.CARTESIAN_ORDINATE.value],
            )
            self.assertEqual(
                cl_pt.pt[2],
                dataset[PropertyNames.ELEVATION.value],
                "Coordinate test: Elevation must be equal to %.3f"
                % dataset[PropertyNames.ELEVATION.value],
            )

    def test_properties(self: Self) -> None:
        """Test of create_cl_point properties."""
        for ide, age, dataset in zip(
            (ide1, ide2, ide3),
            (age1, age2, age3),
            (dataset1, dataset2, dataset3),
            strict=True,
        ):
            cl_pt = create_cl_point(dataset, ide, age)
            self.assertEqual(
                cl_pt.curvature(),
                dataset[PropertyNames.CURVATURE.value],
                "Properties test: Curvature must be equal to %.3f"
                % dataset[PropertyNames.CURVATURE.value],
            )
            self.assertEqual(
                cl_pt.velocity(),
                dataset[PropertyNames.VELOCITY.value],
                "Properties test: Velocity must be equal to %.3f"
                % dataset[PropertyNames.VELOCITY.value],
            )
            self.assertEqual(
                cl_pt.depth_mean(),
                dataset[PropertyNames.DEPTH_MEAN.value],
                "Properties test: Mean_depth must be equal to %.3f"
                % dataset[PropertyNames.DEPTH_MEAN.value],
            )
            self.assertEqual(
                cl_pt.width(),
                dataset[PropertyNames.WIDTH.value],
                "Properties test: Width must be equal to %.3f"
                % dataset[PropertyNames.WIDTH.value],
            )
            self.assertEqual(
                cl_pt.velocity_perturbation(),
                dataset[PropertyNames.VELOCITY_PERTURBATION.value],
                "Properties test: Vel_perturb must be equal to %.3f"
                % dataset[PropertyNames.VELOCITY_PERTURBATION.value],
            )
            self.assertEqual(
                cl_pt.get_property("Property1"),
                dataset["Property1"],
                "Properties test: Property1 must be equal to %.3f"
                % dataset["Property1"],
            )
            self.assertFalse(
                np.isfinite(cl_pt.curvature_filtered()),
                "Properties test: curvature_filtered function must return "
                + "finite values.",
            )

    def test_repr(self: Self) -> None:
        """Test of cl point repr function."""
        for ide, age, dataset in zip(
            (ide1, ide2, ide3),
            (age1, age2, age3),
            (dataset1, dataset2, dataset3),
            strict=True,
        ):
            cl_pt = create_cl_point(dataset, ide, age)
            rep = str(
                np.array(
                    [
                        dataset[PropertyNames.CARTESIAN_ABSCISSA.value],
                        dataset[PropertyNames.CARTESIAN_ORDINATE.value],
                        dataset[PropertyNames.ELEVATION.value],
                    ]
                )
            )
            self.assertTrue(
                str(cl_pt.pt) == rep,
                "Repr test: channel point must be printed: %s" % rep,
            )

    def test_add(self: Self) -> None:
        """Test of cl point add function."""
        cl_pt1 = create_cl_point(dataset1, ide1, age1)
        cl_pt2 = create_cl_point(dataset2, ide2, age2)
        cl_pt12 = cl_pt1 + cl_pt2
        self.assertEqual(cl_pt12._id, cl_pt1._id, "Add test: id must be equal")
        self.assertEqual(
            cl_pt12._age, cl_pt1._age, "Add test: age must be equal"
        )
        expected = cl_pt1.width() + cl_pt2.width()
        self.assertEqual(
            cl_pt12.width(),
            expected,
            "Add test: width must be equal to %.2f" % expected,
        )

    def test_multiply(self: Self) -> None:
        """Test of cl point multiply function."""
        for ide, age, dataset in zip(
            (ide1, ide2, ide3),
            (age1, age2, age3),
            (dataset1, dataset2, dataset3),
            strict=True,
        ):
            cl_pt = create_cl_point(dataset, ide, age)
            cl_pt11 = cl_pt * 2.0
            self.assertEqual(
                cl_pt11._id,
                cl_pt._id,
                "Multiply test - cl_pt*n: id must be unchanged",
            )
            self.assertEqual(
                cl_pt11._age,
                cl_pt._age,
                "Multiply test - cl_pt*n: age must be unchanged",
            )
            expected = cl_pt.width() * 2.0
            self.assertEqual(
                cl_pt11.width(),
                expected,
                "Multiply test - cl_pt*n: Width must be equal to %.2f"
                % expected,
            )

            cl_pt12 = 5.0 * cl_pt
            self.assertEqual(
                cl_pt11._id,
                cl_pt._id,
                "Multiply test - n*cl_pt: id must be unchanged",
            )
            self.assertEqual(
                cl_pt11._age,
                cl_pt._age,
                "Multiply test - n*cl_pt: age must be unchanged",
            )
            expected = cl_pt.width() * 5.0
            self.assertEqual(
                cl_pt12.width(),
                expected,
                "Multiply test - n*cl_pt: Width must be equal to %.2f"
                % expected,
            )

    def test_divide(self: Self) -> None:
        """Test of cl point divide function."""
        for ide, age, dataset in zip(
            (ide1, ide2, ide3),
            (age1, age2, age3),
            (dataset1, dataset2, dataset3),
            strict=True,
        ):
            cl_pt = create_cl_point(dataset, ide, age)
            cl_pt11 = cl_pt / 2.0
            self.assertEqual(
                cl_pt11._id, cl_pt._id, "Divide test: id must be unchanged"
            )
            self.assertEqual(
                cl_pt11._age, cl_pt._age, "Divide test: age must be unchanged"
            )
            expected = cl_pt.width() / 2.0
            self.assertEqual(
                cl_pt11.width(),
                expected,
                "Divide test: Width must be equal to %.2f" % expected,
            )

    def test_eq(self: Self) -> None:
        """Test of cl point eq function."""
        cl_pt1 = create_cl_point(dataset1, ide1, age1)
        cl_pt2 = create_cl_point(dataset2, ide2, age2)
        self.assertEqual(
            cl_pt1, cl_pt1, "Equality test: cl_pt1 must be equal to cpt1"
        )
        self.assertNotEqual(
            cl_pt1,
            cl_pt2,
            "Equality test: cl_pt1 must be different from cl_pt2",
        )


if __name__ == "__main__":
    unittest.main()
