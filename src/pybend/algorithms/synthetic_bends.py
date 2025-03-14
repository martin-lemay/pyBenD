# SPDX-FileCopyrightText: Copyright 2025 Martin Lemay <martin.lemay@mines-paris.org>
# SPDX-FileContributor: Martin Lemay
# ruff: noqa: E402 # disable Module level import not at top of file

import numpy as np
import numpy.typing as npt

import pybend.algorithms.centerline_process_function as cpf

__doc__ = """
Set of methods to create synthetic bends.
"""


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

    :math: `$\Theta=\Theta_0.\cos(\frac{2\pi.s}{\lambda})+\Theta_0^3.(Js.\sin(3\frac{2\pi.s}{\lambda})-Jf.\cos(3\frac{2\pi.s}{\lambda}))$`

    where :math:`$\Theta$` is the local angle from x axis, :math:`$\Theta_0$`
    the maximum angle, :math:`$s$` the curvilinear coordinate,
    :math:`$\lambda$` the wavelength, :math:`$Js$` the skewness coefficient,
    and :math:`$Jf$` the flattening coefficient.

    Inflection point may be downstream the first point at
    :math:`$\Theta=\Theta_0$`, then the bend between inflection points is
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

    """  # noqa: E501
    coords_x: list[float] = [0.0]
    coords_y: list[float] = [0.0]
    teta: float = teta_max
    ds: float = 1.0
    for i in range(int(round(1.2 * nb_pts))):
        t: float = np.pi * i / (nb_pts - 1)
        teta = teta_max * np.cos(t) + teta_max**3 * (
            Js * np.sin(3 * t) - Jf * np.cos(3 * t)
        )
        coords_x += [coords_x[i] + ds * np.cos(teta)]
        coords_y += [coords_y[i] + ds * np.sin(teta)]

    # get minimum curvature indexes
    coords: npt.NDArray[np.float64] = np.column_stack((coords_x, coords_y))
    curvature: npt.NDArray[np.float64] = np.abs(cpf.compute_curvature(coords))

    # find inflection points
    curv1: npt.NDArray[np.float64] = -1.0 * curvature
    peak_indexes: npt.NDArray[np.int64] = (
        cpf.find_inflection_points_from_peaks(curv1, 0.1)
    )
    i_up: int = 0
    i_down: int = len(coords)
    if len(peak_indexes) == 1:
        i_down = peak_indexes[0]
    elif len(peak_indexes) == 2:
        i_up, i_down = peak_indexes
    elif len(peak_indexes) > 2:
        i_up = peak_indexes[0]
        i_down = peak_indexes[-1]
    # return coordinates between inflection points
    return coords[i_up : i_down + 1]
