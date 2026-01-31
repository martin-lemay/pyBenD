# SPDX-FileCopyrightText: Copyright 2025 Martin Lemay <martin.lemay@mines-paris.org>
# SPDX-FileContributor: Martin Lemay
# ruff: noqa: E402 # disable Module level import not at top of file

"""Synthetic bend generators."""

import numpy as np
import numpy.typing as npt

import pybend.algorithms.centerline_process_functions as cpf


def mirror(
    coords: npt.NDArray[np.float64], nb_pts: int
) -> npt.NDArray[np.float64]:
    """Function to add points at the beginning and end of the list.

    Args:
        coords (npt.NDArray[np.float64]): input coordinates
        nb_pts (int): number of points to add

    Returns:
        npt.NDArray[np.float64]: new coordinates with added points

    """
    coords_new: npt.NDArray[np.float64] = np.zeros(
        (coords.shape[0] + 2 * nb_pts, coords.shape[1])
    )
    coords_new[nb_pts:-nb_pts] = coords
    for i in range(nb_pts):
        # add points at the beginning
        c1: float = coords[i]
        c2: float = coords[i + 1]
        dc: float = c2 - c1
        coords_new[nb_pts - i - 1] = coords_new[nb_pts - i] - dc

        # add point at the end
        c1 = coords[-1 - i]
        c2 = coords[-1 - i - 1]
        dc = c2 - c1
        coords_new[-nb_pts + i] = coords_new[-nb_pts + i - 1] - dc
    return coords_new


def circular_bend(nb_pts: int, ampl: float = 1.0) -> npt.NDArray[np.float64]:
    """Create a circular bend.

    Args:
        nb_pts (int): number of points along bend centerline
        ampl (float, optional): amplitude of bends.
            Defaults to 1.

    Returns:
        npt.NDArray[np.float64]: point coordinates

    """
    coords_x: list[float] = []
    coords_y: list[float] = []
    i: int = 0
    while i < nb_pts:
        t: float = np.pi * i / (nb_pts - 1)
        coords_x += [np.cos(t + np.pi)]
        coords_y += [np.sin(t)]
        i += 1

    coords: npt.NDArray[np.float64] = np.column_stack((coords_x, coords_y))
    return ampl * coords


def kinoshita_bend(
    nb_pts: int, teta_max: float, Js: float, Jf: float
) -> npt.NDArray[np.float64]:
    r"""Create a Kinoshita bend.

    Bend centerline follows the Kinoshita curve (Kinoshita, 1961):

    .. math::

        \Theta = \Theta_0 \cos\left(\frac{2\pi s}{\lambda}\right)
        + \Theta_0^3\left(J_s \sin\left(3\frac{2\pi s}{\lambda}\right)
        - J_f \cos\left(3\frac{2\pi s}{\lambda}\right)\right)

    where :math:`\Theta` is the local angle from x axis, :math:`\Theta_0`
    the maximum angle, :math:`s` the curvilinear coordinate,
    :math:`\lambda` the wavelength, :math:`J_s` the skewness coefficient,
    and :math:`J_f` the flattening coefficient.

    Inflection point may be downstream the first point at
    :math:`\Theta = \Theta_0`, then the bend between inflection points is
    determined from:

        1. compute point coordinates over a bit more than a wavelength,
        2. find inflection points
        3. return coords in-between inflection points

    Args:
        nb_pts (int): number of points along bend centerline
        teta_max (float): maximum angle (rad) from horizontal axis
        Js (float): skewness coefficient. If positive, bends are left skewed,
            if negative bends are right skewed.
        Jf (float): flatness coefficient. If positive, bends are more
            elongated, if negative bends are more flat.

    Returns:
        npt.NDArray[np.float64]: point coordinates

    """
    coords_x: list[float] = [0.0]
    coords_y: list[float] = [0.0]
    teta: float = teta_max
    ds: float = 1.0
    for i in range(int(round(1.3 * nb_pts))):
        i1 = i - (nb_pts / 6)
        t: float = np.pi * i1 / (nb_pts - 1)
        teta = teta_max * np.cos(t) + teta_max**3 * (
            Js * np.sin(3 * t) - Jf * np.cos(3 * t)
        )
        coords_x += [coords_x[i] + ds * np.cos(teta)]
        coords_y += [coords_y[i] + ds * np.sin(teta)]

    # remove first point that may not be accurate
    coords_x = coords_x[1:]
    coords_y = coords_y[1:]

    # get coords of the bend where curvature is negative
    coords: npt.NDArray[np.float64] = np.column_stack((coords_x, coords_y))
    curvature: npt.NDArray[np.float64] = cpf.compute_curvature(coords)
    coords1 = coords[np.nonzero(curvature >= 0)]

    # normalize bend x coordinates between 0 and 1
    xmin = np.min(coords1[:, 0])
    xmax = np.max(coords1[:, 0])
    coords1 = (coords1 - xmin) / (xmax - xmin)

    # return coordinates between inflection points
    return coords1
