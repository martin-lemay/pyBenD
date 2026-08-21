# SPDX-FileCopyrightText: Copyright 2025 Martin Lemay <martin.lemay@mines-paris.org>
# SPDX-FileContributor: Martin Lemay

"""Enumerations used throughout pybend."""

from enum import StrEnum


# TODO replace property name in the code with this enum
class PropertyNames(StrEnum):
    """Enumeration of usual ChannelPoint property names."""

    AGE = "Age"
    CURVILINEAR_ABSCISSA = "Curv_abscissa"
    CARTESIAN_ABSCISSA = "Cart_abscissa"
    CARTESIAN_ORDINATE = "Cart_ordinate"
    NORMAL_X = "Normal_x"
    NORMAL_Y = "Normal_y"
    ELEVATION = "Elevation"
    CURVATURE = "Curvature"
    CURVATURE_FILTERED = "Curvature_filtered"
    VELOCITY = "Velocity"
    WIDTH = "Width"
    DEPTH_MEAN = "Mean_depth"
    DEPTH_MAX = "Max_depth"
    VELOCITY_PERTURBATION = "Vel_perturb"
    APEX_PROBABILITY = "Apex_probability"


class AmplitudeType(StrEnum):
    """Method to compute amplitude."""

    #: use the distance between the given point and bend center.
    MIDDLE = "Middle"
    #: use the orthogonal distance between the given point and bend chord.
    ORTHOGONAL = "Orthogonal"


class BendConnectionMethod(StrEnum):
    """Enumeration defining bend connection method."""

    #: Connected bends: same side and shortest distance between apexes.
    APEX = "From Apex"
    #: Connected bends: same side and shortest distance between centroids.
    CENTROID = "From Centroid"
    #: Connected bends: greatest number of cnnected channel points.
    MATCHING = "From Matching"
    #: Connected bends: Needleman-Wunsch sequence alignment.
    SEQUENCE_ALIGNMENT = "From Sequence Alignment"
    #: Connected bends: Dynamic Time Warping alignment.
    DTW = "From DTW"
    #: Connected bends: Generalized NW alignment with
    #: merge/split transitions.
    GENERALIZED_ALIGNMENT = "From Generalized Alignment"


class BendSide(StrEnum):
    """Enum for bend side.

    Bend is UP if curvature is positive, DOWN if curvature is negative,
    and STRAIGHT if the bend sinuosity is below the validity threshold.
    """

    #: positive curvature bend
    UP = "up"
    #: negative curvature bend
    DOWN = "down"
    #: bend below sinuosity threshold (invalid)
    STRAIGHT = "straight"
    #: undefined side
    UNKNOWN = "unknown"


class CreateSectionMethod(StrEnum):
    """Enumeration of methods to use to automatically create cross-sections."""

    #: Section goes by the middle point of the last bend of BendEvolution
    MIDDLE = "From middle"
    #: Section goes by the centroid point of the last bend of BendEvolution
    CENTROID = "From centroid"
    #: Section goes by the apex point of the last bend of BendEvolution
    APEX = "From neighboring apex"


class FilterName(StrEnum):
    """Enumeration for filter names."""

    #: Uniform filter
    UNIFORM = "Uniform filter"
    #: Savitsky-Golay filter
    SAVITSKY = "Savitsky-Golay filter"


class MorphometricNames(StrEnum):
    """Enumeration for morphometric names."""

    ARC_LENGTH = "Arc_length"
    WAVELENGTH = "Wavelength"
    SINUOSITY = "Sinuosity"
    AMPLITUDE = "Amplitude"
    EXTENSION = "Extension"
    RADIUS_CURVATURE = "RadiusCurvature"
    ASYMMETRY = "Asymmetry"
    ASYMMETRY_CENTROID = "Asymmetry_Centroid"
    ROUNDNESS = "Roundness"
    SKEWNESS = "Skewness"
    MAX_AMPLITUDE = "Max_Amplitude"
    MAX_EXTENSION = "Max_Extension"
    MATURITY_INDEX = "Maturity_Index"
    WAVELENGTH_LEOPOLD = "Wavelength_Leopold"
    AMPLITUDE_LEOPOLD = "Amplitude_Leopold"
