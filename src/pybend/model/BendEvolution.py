# SPDX-FileCopyrightText: Copyright 2025 Martin Lemay <martin.lemay@mines-paris.org>
# SPDX-FileContributor: Martin Lemay
# ruff: noqa: E402 # disable Module level import not at top of file

import numpy as np
import numpy.typing as npt
from typing_extensions import Self

__doc__ = r"""
BendEvolution module defines BendEvolution object that stores the successive
indexes of a same bend in a CenterlineCollection.

A bend at a give time may progressively become multilobate, including multiple
smaller order bends. This is why Bend collection stores a list of bend index at
each time step. In addition BendEvolution order increases with the number of
individual bends it includes.



                                   X
                              x °      °
                          .   °   .      °
                        .          .
                            °               °
                      .               .
                          °                   °
                    .                   .
                        O                      O
                   o                     o
*A BendEvolution contains the successvive state of a same bend that evolves
through time, for instance here the bend has tranlated between age t (o..x..o)
and t+1 (O°°X°°O)*

"""


class BendEvolution:
    def __init__(
        self: Self,
        bend_indexes: dict[int, list[int]],
        ide: int,
        order: int,
        isvalid: bool = False,
    ) -> None:
        """Store bend indexes in each Centerline that belongs to BendEvolution.

        Parameters:
        ----------
            bend_indexes (dict[int, list[int]]): dictionnary that contains
                indexes of bends that belongs to this BendEvolution in each
                centerline
            ide (int): id of bend evolution
            order (int): order of bend evolution
            isvalid (bool, optional): bend evolution is valid.

                Defaults to False.

        """
        #: indexes of bends in each centerline that belongs to bend evolution
        self.bend_indexes: dict[int, list[int]] = bend_indexes
        #: bend evolution id
        self.id: int = ide
        #: bend evolution order
        self.order: int = order
        #: bend evolution validity
        self.isvalid: bool = isvalid

        #: list of smoothed apex point location
        self.apex_trajec_smooth: list[npt.NDArray[np.float64]] = []
        #: list of smoothed upstream inflection point location
        self.inflex_up_trajec_smooth: list[npt.NDArray[np.float64]] = []
        #: list of smoothed downstream inflection point location
        self.inflex_down_trajec_smooth: list[npt.NDArray[np.float64]] = []
        #: list of smoothed middle point location
        self.middle_trajec_smooth: list[npt.NDArray[np.float64]] = []
        #: list of smoothed centroid point location
        self.centroid_trajec_smooth: list[npt.NDArray[np.float64]] = []

        # TODO: to move to kinematics
        self.extension = None
        self.translation = None
        self.rotation = None
        self.expansion = None

    def __repr__(self: Self) -> str:
        """Representation of bend evolution.

        Returns:
        ----------
            str: representation
        """
        to_return = "last bend id: {} \n".format(self.id)
        to_return += "First iter: {} \n".format(self.get_all_ages()[0])
        to_return += "Last iter: {} \n".format(self.get_all_ages()[-1])
        return to_return

    def set_is_valid(self: Self, nb: int) -> None:
        """Update isvalid according to input number of bends.

        Parameters:
        ----------
            nb (int): minimum number of bends

        """
        self.isvalid = self._check_is_valid(nb)

    def _check_is_valid(self: Self, nb: int) -> bool:
        """Check bend validity according to input number of bends.

        Parameters:
        ----------
            nb (int): minimum number of bends

        """
        return self.get_all_ages().size >= nb

    def get_all_ages(self: Self) -> npt.NDArray[np.int64]:
        """Get the ages of all the bends that belong to the bend evolution.

        Returns:
        ----------
            npt.NDArray[np.int64]: ages of bends

        """
        return np.sort(list(self.bend_indexes.keys()))

    def get_number_of_bends(self: Self) -> int:
        """Get the number of bends.

        Returns:
        ----------
            int: number of bends

        """
        return len(self.bend_indexes)
