###############################################################################
Glossary
###############################################################################

This page contains a glossary of terms and concepts used in pyBenD.

Meandering Channel Morphology
------------------------------

.. _glossary-centerline:

- **Centerline** (:class:`~pybend.model.Centerline.Centerline`): The centerline of a channel is the curve that runs along the middle of the
    channel's cross-section, right in-between channels banks, representing its neutral axis.
    In pyBenD, the centerline is descretized as a series of connected points (ChannelPoint)
    that define the channel path.

.. _glossary-meander-bend:

- **Meander Bend** (or half-meander; :class:`~pybend.model.Bend.Bend`): A meander bend is a curved section of a river or stream
    that forms part of a larger meandering pattern. In pyBenD, a meander bend is defined as the
    portion of the channel centerline between two consecutive inflection points, where the
    channel changes its curvature sign (from concave to convex or vice versa).

.. _glossary-inflection-point:

- **Inflection Point**: An inflection point is a point along the channel centerline where the
    curvature changes sign. In pyBenD, inflection points are used to delineate meander bends.

.. _glossary-bend-chord:

- **Bend Chord**: The bend chord is the straight line connecting the two inflection points
    defining a meander bend.

.. _glossary-bend-center:

- **Bend Center**: The bend center is the midpoint of the bend chord.

.. _glossary-bend-centroid:

- **Bend Centoid**: The bend centroid is the geometric center of the area enclosed between
    the bend arc and the bend chord.

.. _glossary-bend-apex:

- **Bend Apex**: . In pyBenD, bend apex can be determined from various methods, including:

    - Curvature Cmulative Spatial Distribution Method: The median abscissa of the cumulative
        distribution of curvature along the bend. This is the default method used in pyBenD as
        it has numerous advantages (see Lemay et al., 2026).
    - Maximum Curvature Method: The point along the bend where the curvature is at its maximum
    - Maximum Amplitude Method: The point along the bend that is farthest from the straight
        line connecting the two inflection points (i.e., perpendicular distance to the chord
        is greatest).
    - Maximum Extension Method: The point along the bend where the perpendicular distance
        to the chord connecting the two inflection points is greatest.
    - Midpoint Method: The point that is equidistant from the two inflection points defining
        the bend.
    - Weighting Method: A weighted average of the above methods (Maximum Curvature, maximum
        Amplitude, and Midpoint methods) to provide a more robust estimate of the bend apex location.
    - Azimuth Change Method: The point along the bend where the change in azimuth (direction)
        is (see Limaye, 2025):

        - half the total change of azimuth between the two inflection points, if the total
            change is less than 180°.
        - 90° of change compared to upstream inflection point, if the total change of azimuth
            is greater than 180°.


Morphometric Parameters
------------------------

.. _glossary-curvature:

- **Curvature** (k): The curvature of a curve at a given point is a measure of how sharply the
    curve bends at that point. It is defined as the reciprocal of the radius of the osculating 
    circle at that point. In pyBenD, curvature is computed along the channel centerline to
    identify inflection points.

.. _glossary-sinuosity:

- **Sinuosity** (S): The sinuosity is a measure of how much a river meanders. It is defined
    as the ratio of the channel centerline length to the straight-line distance between two
    points along the channel.

.. _glossary-bend-arc-length:

- **Bend Arc Length** (L_b): The bend length is the length of the channel centerline between
    two consecutive inflection points defining a meander bend.

.. _glossary-bend-wavelength:

- **Bend Wavelength** (λ_b): The bend wavelength is the cartesian distance between two
    consecutive inflection points along the channel centerline.

.. _glossary-bend-amplitude:

- **Bend Amplitude** (A_b): The bend amplitude is the maximum perpendicular distance from the
    bend apex to the bend chord.

.. _glossary-bend-extension:

- **Bend Extension** (E_b): The bend extension is the distance from bend center to the bend apex.

.. _glossary-bend-radius-of-curvature:

- **Bend Radius of Curvature**: The bend radius of curvature is the radius of the osculating
    circle at the bend apex. It is computed as the inverse of the curvature at the bend apex.

.. _glossary-meander-wavelength:

- **Meander Wavelength** (λ_m): The meander wavelength is the cartesian distance between two
    consecutive apex points along the channel centerline. It corresponds to the definition
    of Leopold and Wolman (1960).

.. _glossary-meander-amplitude:

- **Meander Amplitude** (A_m): The meander amplitude is the maximum distance between two
    consecutive apex points measured perpendicular to the main flow direction along the
    channel centerline. It corresponds to the definition of Leopold and Wolman (1960).

.. _glossary-asymmetry-coefficient:

- **Asymmetry Coefficient** (AC): The asymmetry coefficient is a measure of the asymmetry of
    a meander bend. It is defined as the ratio of the difference of lengths between apex and
    upstream and downstream inflection points to the bend arc length. An AC value of 0 
    indicates a symmetric bend, while values greater than 0 indicate a bend with a
    downstream-shifted apex, and values less than 0 indicate an upstream-shifted apex 
    (See Howard and Hemberger, 1991; Finotello et al 2024).

.. _glossary-bend-roundness:

- **Bend Roundness**: The roundness is the ratio of maximum to mean curvature along the bend
    (see Schwenk et al., 2016).


Kinematic Parameters
------------------------

.. _glossary-lateral-migration-rate:

- **Lateral Migration Rate** (M): The lateral migration rate is the rate at which a channel
    migrates laterally over time. In pyBenD, lateral migration rates are computed by tracking
    the movement of channel points between successive centerlines in a centerline collection.

.. _glossary-mobility-number:

- **Mobility Number**: The mobility number is a dimensionless parameter that quantifies the
    relative mobility of a meander bend. It was initially defined as the ratio of the lateral
    migration rate over channel aggradation rate (see Jerolmack and Mohrig, 2007).

.. _glossary-stratigraphic-mobility-number:

- **Stratigraphic Mobility Number**: The stratigraphic mobility number is a dimensionless
    parameter that quantifies the relative mobility of a meander bend after the stratigraphic
    architecture. It is analogous to the Mobility Number but defined from channel displacements
    instead of migration rates. It is defined from cross-sections as the ratio of the lateral
    displacement over the vertical displacement of the channel centerline between two time steps
    (see Jobe et al., 2016; Lemay et al., 2024).


pyBenD terms
------------------------

.. _glossary-centerline-collection:

- **Centerline Collection** (:class:`~pybend.model.CenterlineCollection.CenterlineCollection`):
    A centerline collection is a set of channel centerlines that represent 
    the same channel at different time steps. In pyBenD, centerline collections are
    used to analyze channel migration and meander bend evolution over time.


.. _glossary-bend-evolution:

- **Bend Evolution** (:class:`~pybend.model.BendEvolution.BendEvolution`): A
    bend evolution if a set of meander bends that represent the same bend at different
    time steps. In pyBenD, bend evolutions are used to analyze the morphological
    and kinematic changes of meander bends over time.

.. _glossary-channel-point:

- **Channel Point** (:class:`~pybend.model.ClPoint.ClPoint`): A channel point
    is a point along the discretized channel centerline.


References
-----------

- Finotello, A., Ielpi, A., Lapôtre, M. G., Lazarus, E. D., Ghinassi, M., Carniello, L.,
    ... and D’Alpaos, A. (2024). Vegetation enhances curvature-driven dynamics in meandering
    rivers. Nature Communications, 15(1), 1968. https://doi.org/10.1038/s41467-024-46292-x 
- Howard, A. D., and Hemberger, A. T. (1991). Multivariate characterization of
    meandering. https://doi.org/10.1016/0169-555X(91)90002-R 
- Jerolmack, D.J., and Mohrig, D. (2007). Conditions for branching in depositional
    rivers: Geology, v. 35, p. 463–466. https://doi.org/10.1130/G23308A.1
- Jobe, Z.R., Howes, N.C., Auchter, N.C., 2016. Comparing submarine and fluvial channel
    kinematics: implications for stratigraphic architecture. Geology 44 (11), 931–934.
    https://doi.org/10.1130/G38158.1
- Limaye, A. B. (2025). A geometric algorithm to identify river meander bends: 1.
    Effect of perspective. Journal of Geophysical Research: Earth Surface, 130(3),
    e2024JF007908. https://doi.org/10.1029/2024JF007908 
- Lemay, M., Grimaud, J. L., Cojan, I., Rivoirard, J., and Ors, F. (2024). Submarine
    channel stacking patterns controlled by the 3D kinematics of meander bends. Geological
    Society, London, Special Publications, SP540-2022-143. https://doi.org/10.1144/SP540-2022-143. 
- Leopold, L.B. and Wolman, M.G. (1960). River meanders. Geological Society of America
    Bulletin, 71, 769–793, https://doi.org/10.1130/0016-7606%281960%2971%5B769:RM%5D2.0.CO;2 
- Schwenk, J., Lanzoni, S., and Foufoula-Georgiou, E. (2015). The life of a meander bend:
    Connecting shape and dynamics via analysis of a numerical model. Journal of Geophysical
    Research: Earth Surface, 120(4), 690–710. https://doi.org/10.1002/2014JF003252 