# SPDX-FileCopyrightText: Copyright 2025 Martin Lemay <martin.lemay@mines-paris.org>
# SPDX-FileContributor: Martin Lemay
# ruff: noqa: E402 # disable Module level import not at top of file

from __future__ import annotations

from typing import Self

import numpy as np
import numpy.typing as npt
import pandas as pd  # type: ignore[import-untyped]

from pybend.model.enumerations import PropertyNames
from pybend.utils.logging import logger

__doc__ = r"""
Let's consider a channel centerline discretized into successive channel points.
This module defines channel point object that stores channel point properties.
These properties include channel point cartesian and curvilinear coordinates,
channel geometry (width, mean and maximal depth, curvature, etc.), and flow
properties (velocity, velocity perturbation, etc.).

             .                         .                        .
          .     .                    .    .                   .    .
        .         .                .        .               .        .
       .           .              .          .             .          .
      .             .            .            .           .            .
                     .          .              .         .
                      .        .                .       .
                        .    .                    .   .
                           .                        .

*Sinuous channel centerline discretized into a series avec channel points (.).*


To use it:

..code-block::
    ide :int
    age :int
    data :pd.Series
    clPt :ClPoint = ClPoint(ide, age, data)
"""


class ClPoint:
    def __init__(self: Self, ide: str, age: int, dataset: pd.Series) -> None:
        """Centerline point stores coordinates and associated variables.

        Channel point coordinates must be present in the dataset.

        Parameters:
        ----------
            ide (str): channel point id
            age (int): channel point age
            dataset (pd.Series): data associated to the channel point
        """
        #: Channel point id
        self._id: str = ide
        #: channel point age
        self._age: int = age
        # channel point curvilinear abscissa

        self._s: float = dataset[PropertyNames.CURVILINEAR_ABSCISSA.value]
        #: channel point cartesian coordinates
        self.pt: npt.NDArray[np.float64] = np.array(
            [
                dataset[PropertyNames.CARTESIAN_ABSCISSA.value],
                dataset[PropertyNames.CARTESIAN_ORDINATE.value],
                dataset[PropertyNames.ELEVATION.value],
            ]
        )
        #: channel point data (curvature, height, velocity, etc.)
        self._data: pd.Series = dataset

        #: list of channel point index in the centerline at previous time step
        self.cl_pt_index_prev: list[int] = []
        #: list of channel point index in the centerline at next time step
        self.cl_pt_index_next: list[int] = []

    def __repr__(self: Self) -> str:
        """Return the representatin of ClPoint.

        Returns:
        ----------
            str: representatin of ClPoint

        """
        return self.pt.__repr__()

    def __add__(self: Self, cl_point: ClPoint) -> ClPoint:
        """Add the properties of self with those of another channel point.

        Parameters:
        ----------
            cl_point (ClPoint): another channel point

        Returns:
        ----------
            ClPoint: new channel point with updated properties
        """
        array = [
            self._data[col] + cl_point._data[col] for col in self._data.index
        ]
        data = pd.Series(array, index=self._data.index)
        return ClPoint(self._id, self._age, data)

    def __mul__(self: Self, n: float) -> ClPoint:
        """Multiply the properties of self by a scalar n.

        Parameters:
        ----------
            n (float): multiplication value

        Returns:
        ----------
            ClPoint: new channel point with updated properties
        """
        array = [n * self._data[col] for col in self._data.index]
        data = pd.Series(array, index=self._data.index)
        return ClPoint(self._id, self._age, data)

    def __rmul__(self: Self, n: float) -> ClPoint:
        """Multiply the properties of self by a scalar n.

        Parameters:
        ----------
            n (float): multiplication value

        Returns:
        ----------
            ClPoint: new channel point with updated properties
        """
        return self.__mul__(n)

    def __truediv__(self: Self, n: float) -> ClPoint:
        """Divide the properties of self by a scalar n.

        Parameters:
        ----------
            n (float): division value

        Returns:
        ----------
            ClPoint: new channel point with updated properties
        """
        array = [self._data[col] / n for col in self._data.index]
        data = pd.Series(array, index=self._data.index)
        return ClPoint(self._id, self._age, data)

    def __eq__(self: Self, other: object) -> bool:
        """Test for the equality of channel point.

        Compares channel point ids only.

        Parameters:
        ----------
            other (object): another object

        Returns:
        ----------
            bool: True if channel point ids are the same.
        """
        if not isinstance(other, ClPoint):
            return NotImplemented
        return self._id == other._id

    def set_curvature(self: Self, curv: float) -> None:
        """Set channel point curvature property.

        Parameters:
        ----------
            curv (float): curvature
        """
        self.set_property(PropertyNames.CURVATURE.value, curv)

    def curvature(self: Self) -> float:
        """Get channel point curvature property.

        Returns:
        ----------
            float: curvature
        """
        return self.get_property(PropertyNames.CURVATURE.value)

    def curvature_filtered(self: Self) -> float:
        """Get channel point filtered curvature property.

        Returns:
        ----------
            float: smoothed curvature
        """
        return self.get_property(PropertyNames.CURVATURE_FILTERED.value)

    def velocity(self: Self) -> float:
        """Get channel point velocity property.

        Returns:
        ----------
            float: velocity
        """
        return self.get_property(PropertyNames.VELOCITY.value)

    def depth_mean(self: Self) -> float:
        """Get channel point mean depth property.

        Returns:
        ----------
            float: mean depth
        """
        return self.get_property(PropertyNames.DEPTH_MEAN.value)

    def depth_max(self: Self) -> float:
        """Get channel point mean depth property.

        Returns:
        ----------
            float: mean depth
        """
        return self.get_property(PropertyNames.DEPTH_MAX.value)

    def width(self: Self) -> float:
        """Get channel point width property.

        Returns:
        ----------
            float: width
        """
        return self.get_property(PropertyNames.WIDTH.value)

    def velocity_perturbation(self: Self) -> float:
        """Get channel point velocity perturbation property.

        Returns:
        ----------
            float: velocity perturbation
        """
        return self.get_property(PropertyNames.VELOCITY_PERTURBATION.value)

    def get_data(self: Self) -> pd.Series:
        """Get the channel poitn properties.

        Returns:
        ----------
            pd.Series: channel poitn properties
        """
        return self._data

    def get_property(self: Self, name: str) -> float:
        """Get channel point property from input name.

        Parameters:
        ----------
            name (str): name of the property

        Returns:
        ----------
            float: property values if it exists, or np.nan otherwise
        """
        if name in self._data.index:
            return self._data[name]
        logger.warning(f"Property {name} is not stored on channel points.")
        return np.nan

    def set_property(self: Self, name: str, value: float) -> None:
        """Set property value.

        Parameters:
        ----------
            name (str): name of the property to set
            value (float): value of the property to set
        """
        self._data[name] = value
