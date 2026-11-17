# SPDX-FileCopyrightText: Copyright 2025 Martin Lemay <martin.lemay@mines-paris.org>
# SPDX-FileContributor: Martin Lemay
# ruff: noqa: E402 # disable Module level import not at top of file

"""Bend model.

This module defines the `Bend` object and helpers to encode/decode bend unique
identifiers.

Let's suppose a sinuous channel centerline. Bends are defined as the channel
path comprised between two consecutive inflection points (o). A bend contains
an apex whose definition may vary according to bend shape:

- Kinoshita-like bends: maximum curvature (see Kinoshita (1961);
  `Parker et al. (1983) <https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/abs/on-the-time-development-of-meander-bends/2E90F22506BAB77771E1E54126B95D40>`_
  `Abad and Garcia (2009) <https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2008WR007016>`_)
- Circular bend (constant curvature): equidistance from inflection points.

By convention, a bend is UP if curvature is positive (clockwise rotation along
flow direction), and DOWN if curvature is negative (counter-clockwise rotation
along flow direction).

.. code-block:: bash

                         UP                    DOWN
                         x
                       .   .               o     m     o
      Flow           .       .              .    b    .
       -->          .    b    .              .       .
    Direction      o     m     o               .   .
                                                 x

Elementary bends are defined by upstream and downstream inflection points (o).
Bends contain distinctive points including the apex (x), the middle (m) and
the barycenter (b).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import numpy.typing as npt
import pandas as pd  # type: ignore[import-untyped]
from shapely.geometry import LineString, Polygon  # type: ignore
from typing_extensions import Self

from pybend.model.enumerations import BendSide

#: modulo value for bend unique ids
uid_module: int = int(1e4)


def _as_integer(value: float, name: str) -> int:
    """Convert an integer-valued number to int.

    Args:
        value (float): value to convert.
        name (str): name of the value, used in the error message.

    Raises:
        ValueError: if value is not finite or not integer-valued.

    Returns:
        int: integer value.
    """
    try:
        value_f = float(value)
    except (TypeError, ValueError) as err:
        raise ValueError(f"{name} must be a number, got {value!r}.") from err
    if not np.isfinite(value_f) or not value_f.is_integer():
        raise ValueError(
            f"{name} must be an integer-valued number, got {value!r}."
        )
    return int(value_f)


def get_bend_uid(bend_id: int, age: int | float) -> int:
    """Get bend unique id from bend id and age.

    Unique id is computed as ``uid_module * age + bend_id``, it is then
    required that ``0 <= bend_id < uid_module``.

    Args:
        bend_id (int): bend index
        age (int | float): age. Must be integer-valued (e.g., 2500 or
            2500.0).

    Raises:
        ValueError: if bend_id is not in [0, uid_module[ or if bend_id or
            age is not integer-valued.

    Returns:
        int: unique id
    """
    bend_id_i: int = _as_integer(bend_id, "Bend id")
    age_i: int = _as_integer(age, "Age")
    if bend_id_i < 0 or bend_id_i >= uid_module:
        raise ValueError(
            f"Bend id {bend_id_i} is out of range [0, {uid_module}[: bend "
            + "unique ids would collide. Too many bends along the centerline."
        )
    return uid_module * age_i + bend_id_i


def parse_bend_uid(uid: int | float) -> tuple[int, int]:
    """Parse bend unique id to get back bend index and age.

    Args:
        uid (int | float): bend unique id. Must be integer-valued.

    Raises:
        ValueError: if uid is not integer-valued.

    Returns:
        tuple[int, int]: tuple containing bend age and index.
    """
    uid_i: int = _as_integer(uid, "Bend unique id")
    ide: int = uid_i % uid_module
    age: int = uid_i // uid_module
    return (age, ide)


class Bend:
    """A meander bend delimited by two consecutive inflection points."""

    def __init__(
        self: Self,
        bend_id: int,
        index_inflex_up: int,
        index_inflex_down: int,
        age: int = 0,
        side: BendSide = BendSide.UNKNOWN,
    ) -> None:
        """Store bend parameters associated to a Centerline object.

        Args:
            bend_id (int): Bend id.
            index_inflex_up (int): index of the upstream inflection point along
                the centerline.
            index_inflex_down (int): index of the downstream inflection point
                along the centerline.
            age (int, optional): Age of the bend.
                Defaults to 0.
            side (BendSide, optional): Bend side (UP, DOWN, or UNKNOWN).
                Defaults to BendSide.UNKNOWN.

        Raises:
            ValueError: if bend unique id cannot be computed from bend_id
                and age (see :func:`get_bend_uid`).

        """
        #: bend id
        self.id: int = bend_id
        #: bend age
        self.age: int = age
        #: bend unique id
        self.uid: int = get_bend_uid(bend_id, age)
        #: bend side
        self.side: BendSide = side

        #: index of upstream inflection point
        self.index_inflex_up: int = index_inflex_up
        #: index of downstream inflection point
        self.index_inflex_down: int = index_inflex_down
        #: index of apex point along the centerline. -1 (or None) means that
        #: the apex is undefined.
        self.index_apex: int = -1
        #: index of maximal curvature point from upstream inflection point,
        #: None if not computed
        self.index_max_curv: Optional[int] = None

        #: apex probability values for each point of the bend
        self.apex_probability: Optional[npt.NDArray[np.float64]] = None
        #: smoothed apex probability values for each point of the bend
        self.apex_probability_smooth: Optional[npt.NDArray[np.float64]] = None

        #: center point coordinates. Center point is defined as the point at
        #: equal distance from inflection points.
        self.pt_center: Optional[npt.NDArray[np.float64]] = None
        #: Bend centroid is the barycenter of the polygon defined by the
        #: centerline between upstream and downstream inflection points and is
        #: closed between these points.
        self.pt_centroid: Optional[npt.NDArray[np.float64]] = None
        #: polygon defined by the centerline between upstream and downstream
        #: inflection points and is closed between these points
        self.polygon: Optional[Polygon | LineString] = None

        #: indexes of connected bend in previous centerline
        self.bend_uid_prev: Optional[list[int]] = None
        #: indexes of connected bend in next centerline
        self.bend_uid_next: Optional[list[int]] = None
        #: id of the BendEvolution object the bend belongs to, None if the
        #: bend does not belong to any BendEvolution
        self.bend_evol_id: Optional[int] = None

        self.intersected_section_indexes: Optional[list[int]] = None

        # Sinuosity, Length, half-wavelength, Amplitude perpendicular,
        # Amplitude middle
        # individual meander geometry
        self.params: Optional[pd.Series] = None

        # meander geometry averaged over a given window (computed later)
        self.params_averaged: Optional[pd.DataFrame] = None

    @property
    def is_valid(self: Self) -> bool:
        """Whether the bend is valid (not straight).

        Returns:
            bool: True if side is not STRAIGHT.
        """
        return self.side != BendSide.STRAIGHT

    def __repr__(self: Self) -> str:
        """Return a concise string representation.

        Returns:
            str: Description of the object.
        """
        return str(self.age) + "-" + str(self.id)

    # add properties of self and another bend
    # return a new bend with the same id as self
    def __add__(self: Self, bend: Bend) -> Bend:
        """Add current bend to another bend.

        Args:
            bend (Bend): another Bend object

        Returns:
            Bend: new bend
        """
        new_bend = Bend(
            self.id,
            self.index_inflex_up,
            bend.index_inflex_down,
            self.age,
            self.side,
        )
        return new_bend

    def __eq__(self: Self, other: object) -> bool:
        """Equality method.

        Args:
            other (object): another object

        Returns:
            bool: True if bend unique id are equal.
        """
        if not isinstance(other, Bend):
            return NotImplemented
        return other.uid == self.uid

    def __hash__(self: Self) -> int:
        """Hash method.

        Returns:
            int: hash
        """
        return int(self.uid)

    def get_nb_points(self: Self) -> int:
        """Get the number of points of the bend.

        Returns:
            int: number of points

        """
        return self.index_inflex_down - self.index_inflex_up + 1

    def add_bend_connection_next(self: Self, bend_uid: int) -> None:
        """Add bend connection with bend in the next centerline.

        Args:
            bend_uid (int): unique index of the bend connected to itself.
        """
        if self.bend_uid_next is None:
            self.bend_uid_next = [bend_uid]
        else:
            if bend_uid not in self.bend_uid_next:
                self.bend_uid_next += [bend_uid]

    def add_bend_connection_prev(self: Self, bend_uid: int) -> None:
        """Add bend connection with bend in the previous centerline.

        Args:
            bend_uid (int): unique index of the bend connected to itself.
        """
        if self.bend_uid_prev is None:
            self.bend_uid_prev = [bend_uid]
        else:
            if bend_uid not in self.bend_uid_prev:
                self.bend_uid_prev += [bend_uid]

    def add_intersected_section_index(self: Self, i: int) -> None:
        """Add the section index of intersected section with itself.

        Args:
            i (int): section index
        """
        if not self.intersected_section_indexes:
            self.intersected_section_indexes = []
        self.intersected_section_indexes += [i]


class BendClPointIndexIter:
    """Iterator over channel-point indices belonging to a `Bend`.

    Iterates over absolute channel-point indices from
    ``bend.index_inflex_up`` to ``bend.index_inflex_down`` (both included).
    """

    def __init__(self: Self, bend: Bend) -> None:
        """Create an iterator over bend channel-point indices.

        Args:
            bend (Bend): Bend to iterate to.
        """
        #: bend
        self.bend = bend
        #: index
        self.index: int = bend.index_inflex_up

    def __iter__(self: Self) -> BendClPointIndexIter:
        """Return the iterator.

        Returns:
            BendClPointIndexIter: self
        """
        self.index = self.bend.index_inflex_up
        return self

    def __next__(self: Self) -> int:
        """Return the next channel-point index.

        Raises:
            StopIteration: Raised when the iterator is exhausted.

        Returns:
            int: Channel-point index.
        """
        if self.index <= self.bend.index_inflex_down:
            x: int = self.index
            self.index += 1
            return x
        else:
            raise StopIteration
