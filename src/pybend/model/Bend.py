# SPDX-FileCopyrightText: Copyright 2025 Martin Lemay <martin.lemay@mines-paris.org>
# SPDX-FileContributor: Martin Lemay
# ruff: noqa: E402 # disable Module level import not at top of file

from __future__ import annotations

from typing import Optional

import numpy as np
import numpy.typing as npt
import pandas as pd  # type: ignore[import-untyped]
from shapely.geometry import LineString, Polygon  # type: ignore
from typing_extensions import Self

from pybend.model.enumerations import BendSide

__doc__ = r"""
Bend module defines Bend object and associated utils.

Let's Suppose a sinuous channel centerline. Bends are defined as the channel
path comprised between 2 consecutive inflection points (o). A Bend contains an
apex whose definition may vary according to bend shape:
    - kinoshita-like bends: maximum curvature (see
        Kinoshita (1961);
        `Parker et al. (1983) <https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/abs/on-the-time-development-of-meander-bends/2E90F22506BAB77771E1E54126B95D40>`_
        `Abad and Garcia (2009) <https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2008WR007016>`_
    - circular bend (constant curvature): equidistance from inflection points

By convention Bend is UP if curvature is positive -clockwise rotation along
flow direction- and is DOWN if curvature is negative -counter-clockwise
rotation along flow direction.


                         UP                    DOWN
                         x
                       .   .               o     m     o
      Flow           .       .              .    b    .
       -->          .    b    .              .       .
    Direction      o     m     o               .   .
                                                 x

*Elementary Bends are defined by upstream and downstream inflection points (o).
Bends contains distinctive points including the apex (x), the middle (m) and
the barucenter (b). By convention Bend is UP if curvature is positive and DOWN
if curvature is negative*


"""

#: modulo value for bend unique ids
uid_module: int = int(1e4)


def get_bend_uid(bend_id: int, age: int) -> int:
    """Get bend unique id from bend id and age.

    Parameters:
        ----------
        bend_id (int): bend index
        age (int): age

    Returns:
        ----------
        int: unique id
    """
    return int(uid_module * age + bend_id)


def parse_bend_uid(uid: int) -> tuple[int, int]:
    """Parse bend unique id to get back bend index and age.

    Parameters:
        ----------
        uid (int): bend unique id

    Returns:
        ----------
        tuple[float, int]: tuple containing bend age and index.
    """
    ide: int = uid % uid_module
    age: int = uid // uid_module
    return (age, ide)


class Bend:
    def __init__(
        self: Self,
        bend_id: int,
        index_inflex_up: int,
        index_inflex_down: int,
        age: int = 0,
        side: BendSide = BendSide.UNKNWON,
        isvalid: bool = False,
    ) -> None:
        """Store bend parameters associated to a Centerline object.

        Parameters:
        ----------
            bend_id (int): bend id
            index_inflex_up (int): index of the upstream inflection point along
                the centerline
            index_inflex_down (int): index of the downstream inflection point
                along the centerline
            age (int, optional): age of the bend.

                Defaults to 0..
            side (Bend_side, optional): bend side (Bend_side.UP, Bend_side.DOWN,
                Bend_side.UNKNWON).

                Defaults to Bend_side.UNKNWON.
            isvalid (bool, optional): bend is valid if its sinuosity is greater
                than a user defined threshold.

                Defaults to False.

        """
        #: bend id
        self.id: int = bend_id
        #: bend age
        self.age: int = age
        #: bend unique id
        self.uid: int = get_bend_uid(bend_id, age)
        #: bend is valid
        self.isvalid: bool = isvalid
        #: bend side
        self.side: BendSide = side

        #: index of upstream inflection point
        self.index_inflex_up: int = index_inflex_up
        #: index of downstream inflection point
        self.index_inflex_down: int = index_inflex_down
        #: index of apex point
        self.index_apex: int = -1
        #: index of maximal curvature point
        self.index_max_curv: Optional[int] = False

        #: apex probability values for each point of the bend
        self.apex_probability: Optional[npt.NDArray[np.float64]] = None
        #: smoothed apex probability values for each point of the bend
        self.apex_probability_smooth: Optional[npt.NDArray[np.float64]] = None

        #: middle point coordinates. Middle point is defined as the point at equal
        #: distance from inflection points.
        self.pt_middle: Optional[npt.NDArray[np.float64]] = None
        #: Bend centroid is the barycenter of the polygon defined by the centerline
        #: between upstream and downstream inflection points and is closed between
        #: these points.
        self.pt_centroid: Optional[npt.NDArray[np.float64]] = None
        #: polygon defined by the centerline between upstream and downstream
        #: inflection points and is closed between these points
        self.polygon: Optional[Polygon | LineString] = None

        #: indexes of connected bend in previous centerline
        self.bend_uid_prev: Optional[list[int]] = None
        #: indexes of connected bend in next centerline
        self.bend_uid_next: Optional[list[int]] = None
        #: id of the BendEvolution object the bend belongs to
        self.bend_evol_id: Optional[int] = False

        self.intersected_section_indexes: Optional[list[int]] = None

        # Sinuosity, Length, half-wavelength, Amplitude perpendicular, Amplitude middle
        # individual meander geometry
        self.params: Optional[pd.Series] = None

        # meander geometry averaged over a given window (computed later)
        self.params_averaged: Optional[pd.DataFrame] = None

    def __repr__(self: Self) -> str:
        """Returned string.

        Returns:
        --------
            str: description of the object.
        """
        return str(self.age) + "-" + str(self.id)

    # add properties of self and another bend
    # return a new bend with the same id as self
    def __add__(self: Self, bend: Bend) -> Bend:
        """Add current bend to another bend.

        Parameters:
        ----------
            bend (Bend): another Bend object

        Returns:
        ----------
            Bend: new bend
        """
        new_bend = Bend(
            self.id,
            self.index_inflex_up,
            bend.index_inflex_down,
            self.age,
            self.side,
            self.isvalid,
        )
        return new_bend

    def __eq__(self: Self, other: object) -> bool:
        """Equality method.

        Parameters:
        ----------
            other (object): another object

        Returns:
        ----------
            bool: True if bend unique id are equal.
        """
        if not isinstance(other, Bend):
            return NotImplemented
        return other.uid == self.uid

    def __hash__(self: Self) -> int:
        """Hash method.

        Returns:
        ----------
            int: hash
        """
        return int(self.uid)

    def get_nb_points(self: Self) -> int:
        """Get the number of points of the bend.

        Returns:
        ----------
            int: number of points

        """
        return self.index_inflex_down - self.index_inflex_up + 1

    def add_bend_connection_next(self: Self, bend_uid: int) -> None:
        """Add bend connection with bend in the next centerline.

        Parameters:
        ----------
            bend_uid (int): unique index of the bend connected to itself.
        """
        if self.bend_uid_next is None:
            self.bend_uid_next = [bend_uid]
        else:
            if bend_uid not in self.bend_uid_next:
                self.bend_uid_next += [bend_uid]

    def add_bend_connection_prev(self: Self, bend_uid: int) -> None:
        """Add bend connection with bend in the previous centerline.

        Parameters:
        ----------
            bend_uid (int): unique index of the bend connected to itself.
        """
        if self.bend_uid_prev is None:
            self.bend_uid_prev = [bend_uid]
        else:
            if bend_uid not in self.bend_uid_prev:
                self.bend_uid_prev += [bend_uid]

    def add_intersected_section_index(self: Self, i: int) -> None:
        """Add the section index of intersected section with itself.

        Parameters:
        ----------
            i (int): section index
        """
        if not self.intersected_section_indexes:
            self.intersected_section_indexes = []
        self.intersected_section_indexes += [i]


class BendClPointIndexIter:
    def __init__(self: Self, bend: Bend) -> None:
        """Itetator on bend channel point indexes.

        Parameters:
        ----------
            bend (Bend): Bend to iterate to.
        """
        #: bend
        self.bend = bend
        #: index
        self.index: int = 0

    def __iter__(self: Self) -> BendClPointIndexIter:
        """Iterator.

        Returns:
        ----------
            BendClPointIndexIter: self
        """
        self.index = self.bend.index_inflex_up
        return self

    def __next__(self: Self) -> int:
        """Next method.

        Raises:
            StopIteration: stop when index_inflex_down is reached.

        Returns:
        ----------
            int: channel point index
        """
        if self.index <= self.bend.get_nb_points():
            x: int = self.index
            self.index += 1
            return x
        else:
            raise StopIteration
