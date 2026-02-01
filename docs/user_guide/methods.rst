Methods
=======

The page describes the methods used in pyBenD to analyze river meanders. It includes
details on how pyBenD create :ref:`Centerline <glossary-centerline>` objects, identifies
:ref:`meander bends <glossary-meander-bend>` and their
characteristic points, track centerlines and bends through time, and calculates
morphometric and kinematic parameters.


.. contents:: On this page
	:local:
	:depth: 3


Centerline Creation
--------------------

pyBenD provides methods to create :ref:`Centerline <glossary-centerline>` objects from various input data formats
(see Inputs section for more details). The :ref:`Centerline <glossary-centerline>` class
stores the channel centerline as a series of connected :ref:`channel points <glossary-channel-point>`
(:class:`~pybend.model.ClPoint.ClPoint`). For subsequent analyzes, the centerline
is optionally resampled and smoothed to reduce digitizing noise and to provide a more
uniform point spacing prior to computing geometric attributes.

The main steps are:

Resampling and path smoothing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. **Curvilinear abscissa computation**

	The centerline cumulative distance (curvilinear abscissa) is computed from the input
	XY coordinates. This provides an estimate of total centerline length and is used to
	determine how many points are required after resampling.

2. **Resampling with a parametric spline**

	If ``spacing > 0``, the centerline is resampled with a parametric spline so that the
	output contains either:

	* a *target spacing* between consecutive points (default behavior), or
	* a *fixed number of points* when ``use_fix_nb_points=True``. In this case, the
	  number of points can be set with the ``spacing`` parameter (integer value).

	If ``spacing == 0``, the centerline is not resampled and the original point
	positions are used.

3. **Smoothing of the resampled XY path**

	When resampling is enabled (``spacing > 0``), the resampled X and Y coordinates are
	smoothed using a Savitzky-Golay filter (polynomial order 3, ``mode='nearest'``).
	The smoothing window is derived from the requested smoothing distance:

	- window size (in points) is ``int(smooth_distance / spacing)``,
	- the window is forced to be odd and at least 5 points.

4. **Recompute derived geometry on the new points**

	After resampling/smoothing, pyBenD recomputes (i) the curvilinear abscissa of the
	updated point set and (ii) the local normal vector components (``NORMAL_X``,
	``NORMAL_Y``) from neighboring points.

Property interpolation
^^^^^^^^^^^^^^^^^^^^^^

If ``interpol_props=True``, existing properties stored in the input DataFrame are
transferred to the resampled points.

For each resampled point, pyBenD:

- finds the two closest points on the original centerline (in the XY plane),
- computes their distances to the resampled point,
- interpolates each property using a distance-weighted average.

Some fields are intentionally excluded from interpolation because they are recomputed
for the resampled geometry: curvilinear abscissa and XY coordinates.

Curvature calculation and curvature smoothing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If ``compute_curvature=True``, curvature is computed along the (resampled and smoothed)
centerline using a local 3-point method: the curvature at point *i* depends on the
coordinates of points *(i-1, i, i+1)*.

(See :ref:`Curvature <glossary-curvature>` in the Glossary.)

To reduce local variability, a second curvature series is then computed by filtering the
raw curvature. Two filters are available:

- a moving uniform filter (window size in points), or
- a Savitzky-Golay filter (polynomial order 2).

For the Savitzky-Golay curvature filter, the window size is forced to be odd and at
least 5 points.

Meander Bend Identification
----------------------------

In pyBenD, :ref:`meander bends <glossary-meander-bend>` are identified directly from a
:class:`~pybend.model.Centerline.Centerline`
object once a *filtered curvature* series has been computed on the centerline points
(``Curvature_filtered`` property).

The workflow implemented in :meth:`~pybend.model.Centerline.Centerline.find_bends` is:

1. **Detect inflection points from curvature sign changes**

	:ref:`Inflection points <glossary-inflection-point>` are extracted from the filtered curvature array using
	:func:`~pybend.algorithms.centerline_process_functions.find_inflection_points`.
	A point at index *i* is considered an inflection point when the curvature changes
	sign in its neighborhood (using a 3-point condition based on sums of consecutive
	curvatures):

	.. math::

		\mathrm{sign}(C_{i-1}+C_i) \neq \mathrm{sign}(C_i+C_{i+1})

	Very close inflection points are then filtered so that two consecutive inflection
	indices are separated by at least a small ``lag`` (in points). In the current
	implementation, ``lag=2`` is used.

	The first and last centerline points are always added as bend delimiters if they
	were not detected as inflection points.

2. **Create bends as segments between consecutive inflection points**

	Each bend is defined as the set of centerline points between two consecutive
	:ref:`inflection points <glossary-inflection-point>` (upstream and downstream). Internally, this creates one
	:class:`~pybend.model.Bend.Bend` object per interval.

3. **Compute bend properties (side, validity, apex, and middle point)**

	For each bend, pyBenD computes:

	- **Bend side** (:class:`~pybend.model.enumerations.BendSide`): based on the
	  sign of the *sum* of filtered curvature along the bend. Positive sum
	  gives ``BendSide.UP``, negative gives ``BendSide.DOWN``.

	- **Bend validity**: based on a sinuosity criterion computed between the two
	  :ref:`inflection points <glossary-inflection-point>`.

	  (See :ref:`Sinuosity <glossary-sinuosity>` in the Glossary.)

	  - bend arc-length *L* is measured as the curvilinear abscissa difference,
	  - bend chord length *D* is the Euclidean distance between inflection-point
		 coordinates.

	  The bend sinuosity is then:

	  .. math::

		  S = \frac{L}{D}

	  A bend is marked valid if ``S >= sinuo_thres`` and ``S < 10`` (the upper bound is
	  currently hard-coded).

	- **Bend middle point** (``Bend.pt_center``): the midpoint of the
	  inflection-point chord (the :ref:`bend chord <glossary-bend-chord>`)
	  (XY midpoint), with Z set to the average Z of the two inflection points.

	- **Bend apex index** (``Bend.index_apex``): by default, the
	  :ref:`bend apex <glossary-bend-apex>` is determined
	  from the filtered curvature distribution (see below).

When multiple CPU cores are available, bend creation and per-bend property computation
can be parallelized using multiprocessing.

Bend apex identification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

pyBenD provides multiple methods to locate the apex of each bend.
All methods return an index along the full centerline (not a local bend index).

Curvature-distribution method (default)
"""""""""""""""""""""""""""""""""""""""

The default apex definition used in :meth:`~pybend.model.Centerline.Centerline.find_bend_apex`
is based on the *median* of the cumulative filtered curvature distribution within the bend.

Let :math:`\kappa_i` be the filtered curvature magnitude within the bend. The cumulative weighting
function is built from :math:`\kappa_i^n` (``n`` is an exponent parameter):

.. math::

	F(j) = \frac{\sum_{i \le j} |\kappa_i|^n}{\sum_i |\kappa_i|^n}

The apex is the first index where :math:`F(j) > 0.5`. Larger values of ``n`` put more emphasis
on the highest-curvature part of the bend.

Probability-from-weights method
"""""""""""""""""""""""""""""""""""

Alternatively, an *apex probability* can be computed with
:meth:`~pybend.model.Centerline.Centerline.set_bend_apex_probability_user_weights` and used
to detect the apex with :meth:`~pybend.model.Centerline.Centerline.find_bend_apex_from_weights`.

The apex probability is a weighted combination of three normalized components:

- filtered curvature magnitude,
- bend amplitude (distance to the bend chord),
- a length term that favors the center of the bend over its ends.

User weights ``(curvature_weight, amplitude_weight, length_weight)`` are renormalized so that
their sum equals 1. Inflection points are forced to have zero probability.

To turn probability into an apex index, local maxima of the probability curve are detected
(``scipy.signal.find_peaks`` with a peak height threshold in ``[0.6, 1.0]``). If multiple peaks
are found, the apex is taken as the mean index of the peaks.

Azimuth change method
"""""""""""""""""""""""

The method implemented in :meth:`~pybend.model.Centerline.Centerline.find_bend_apex_limaye`
follows Limaye (2025) and defines the apex from half of the total channel path angle variation
between bend endpoints if total change if less than 90°. If the total change is greater than 180°,
the apex is located at 90° of change from the upstream inflection point.

Practically, the algorithm uses the local normal vectors along the bend (``Normal_x`` and
``Normal_y`` properties) and finds the index where the angular deviation from the first normal
reaches half of the overall deviation.

Morphometric Parameter Calculation
------------------------------------------------

Morphometric parameters are computed once bends and their characteristic points
(inflection points, apex, and chord midpoint) have been identified on a
:class:`~pybend.model.Centerline.Centerline`.
In practice, the computations are implemented in
:class:`~pybend.model.Morphometry.Morphometry` and exposed through
:meth:`~pybend.model.Morphometry.Morphometry.compute_bends_morphometry`.

Unless stated otherwise, all bend-scale metrics are computed on the subset of
*valid* bends (see bend validity in the previous section). Internally, each bend
stores indices to the upstream and downstream inflection points
(``Bend.index_inflex_up``, ``Bend.index_inflex_down``), the apex index
(``Bend.index_apex``), and the chord midpoint (``Bend.pt_center``).

Morphometric parameters (bend scale)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let *o* be the two inflection points delimiting a bend and *x* the bend apex
(:ref:`inflection points <glossary-inflection-point>` and :ref:`bend apex <glossary-bend-apex>`).
Let :math:`s` denote the curvilinear abscissa along the
centerline.

- **Bend arc length** (:class:`~pybend.model.enumerations.MorphometricNames` ``ARC_LENGTH``):

	The :ref:`bend arc length <glossary-bend-arc-length>` :math:`L_b` is the curvilinear distance between the two inflection
	points:

	.. math::

			L_b = s_{\mathrm{down}} - s_{\mathrm{up}}

	This is computed from the ``Curvilinear_abscissa`` property stored on the
	centerline points within the bend.

- **Bend wavelength** (:class:`~pybend.model.enumerations.MorphometricNames` ``WAVELENGTH``):

	The :ref:`bend wavelength <glossary-bend-wavelength>` :math:`\lambda_b` is the Euclidean distance between the upstream and
	downstream inflection points:

	.. math::

			\lambda_b = \|\mathbf{p}_{\mathrm{down}} - \mathbf{p}_{\mathrm{up}}\|

- **Bend sinuosity** (:class:`~pybend.model.enumerations.MorphometricNames` ``SINUOSITY``):

	Computed as :ref:`sinuosity <glossary-sinuosity>` :math:`S = L_b / \lambda_b` when :math:`\lambda_b > 0`.

- **Bend amplitude** (:class:`~pybend.model.enumerations.MorphometricNames` ``AMPLITUDE``):
	The :ref:`bend amplitude <glossary-bend-amplitude>` :math:`A_b` is the orthogonal (perpendicular) distance from the apex
	point to the bend chord (the line through the two inflection points):

	.. math::

			A_b = d_\perp\left(\mathbf{p}_{\mathrm{apex}},\ \mathbf{p}_{\mathrm{up}},\ \mathbf{p}_{\mathrm{down}}\right)

	In code this is computed with an orthogonal-distance routine applied to the
	apex and the two inflection-point coordinates.

- **Bend extension** (:class:`~pybend.model.enumerations.MorphometricNames` ``EXTENSION``):

	The :ref:`bend extension <glossary-bend-extension>` :math:`E_b` is the distance from the apex to the midpoint of the chord
	(:ref:`bend center <glossary-bend-center>`):

	.. math::

			E_b = \|\mathbf{p}_{\mathrm{apex}} - \mathbf{p}_{\mathrm{center}}\|

	where :math:`\mathbf{p}_{\mathrm{center}}` is the chord midpoint (XY midpoint; Z set
	to the mean Z of the two inflection points).

- **Radius of curvature** (:class:`~pybend.model.enumerations.MorphometricNames` ``RADIUS_CURVATURE``):

	The :ref:`bend radius of curvature <glossary-bend-radius-of-curvature>` is computed from the *filtered curvature* at the apex
	(see curvature filtering above):

	.. math::

			R = \frac{1}{|\kappa_{\mathrm{apex}}|}

	If :math:`|\kappa_{\mathrm{apex}}| = 0`, the radius is reported as NaN.

- **Asymmetry coefficient** (:class:`~pybend.model.enumerations.MorphometricNames` ``ASYMMETRY``):

	Following Howard and Hemberger (1991), the :ref:`asymmetry coefficient <glossary-asymmetry-coefficient>` compares the
	curvilinear distances from the apex to the upstream and downstream inflection
	points:

	.. math::

			AC = \frac{L_{\mathrm{up}} - L_{\mathrm{down}}}{L_b}

	where :math:`L_{\mathrm{up}}` is the arc length from the upstream inflection point
	to the apex, and :math:`L_{\mathrm{down}}` is the arc length from the apex to the
	downstream inflection point.

- **Roundness** (:class:`~pybend.model.enumerations.MorphometricNames` ``ROUNDNESS``):

	The :ref:`roundness <glossary-bend-roundness>` coefficient (Schwenk et al., 2015) is computed from the
	distribution of *filtered curvature magnitude* :math:`|\kappa|` within the bend:

	.. math::

			\mathrm{Roundness} = \frac{\max(|\kappa|)}{\mathrm{mean}(|\kappa|)}

- **Skewness** (:class:`~pybend.model.enumerations.MorphometricNames` ``SKEWNESS``):

	The skewness corresponds to Pearson's skewness coefficient of the curvature
	spatial distribution. The curvilinear abscissa is first normalized to
	:math:`s' \in [0, 1]` within the bend, and the skewness is evaluated as:

	.. math::

			\mathrm{Skewness} = \frac{3\,(\mu - s'_{\mathrm{apex}})}{\sigma}

	where :math:`\mu` and :math:`\sigma` are the mean and standard deviation of the curvature
	spatial distribution computed from :math:`|\kappa|` and :math:`s'`, and :math:`s'_{\mathrm{apex}}`
	is the normalized curvilinear abscissa at the apex index.

Morphometric parameters (meander scale, Leopold & Wolman)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For interior bends (i.e., bends that have both a preceding and a following bend),
pyBenD also provides two meander-scale metrics consistent with Leopold and
Wolman (1960). These use apex points of three consecutive bends:

- **Meander wavelength** (``WAVELENGTH_LEOPOLD``): Euclidean distance between the
  apex of the previous bend and the apex of the next bend.
- **Meander amplitude** (``AMPLITUDE_LEOPOLD``): orthogonal distance from the
  current bend apex to the straight line joining the previous and next bend apexes.


Centerline and Bend tracking
-----------------------------

In pyBenD, kinematic metrics require linking geometric entities through time.
This is done in two stages implemented in
:class:`~pybend.model.CenterlineCollection.CenterlineCollection` (a :ref:`centerline collection <glossary-centerline-collection>`):

1. **Centerline matching**: associate channel points between consecutive
	 centerlines (point-to-point tracking).
2. **Bend connection**: associate bends between consecutive centerlines and
	 group them into :class:`~pybend.model.BendEvolution.BendEvolution` objects
	 (bend-to-bend tracking; see :ref:`bend evolution <glossary-bend-evolution>`).

Centerline matching (channel-point tracking)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Channel-point tracking is computed with
:meth:`~pybend.model.CenterlineCollection.CenterlineCollection.match_centerlines`.
For each pair of consecutive centerlines (age ``prev_key`` and ``key``), pyBenD
builds a *pairwise cost matrix* and solves a Dynamic Time Warping (DTW) problem
to obtain an alignment between point indices. Centerlines are tracked sequentially from
the oldest to the youngest.

The methodology for channel point tracking is inspired from
`zsylvester/meanderpy <https://github.com/zsylvester/meanderpy>`_.

**Cost matrix components**

For points :math:`i` on the current centerline and :math:`j` on the previous one, three
non-negative distances can be combined:

- **Planform distance**: Euclidean distance between :math:`(x_i, y_i)` and
	:math:`(x_j, y_j)`. If the distance exceeds ``dmax``, it is set to a very large
  penalty so that DTW avoids matching distant points. ``dmax`` is usually set to
  a meander wavelength.
- **Curvature distance**: difference in *filtered curvature magnitude*
  :math:`\left|\,|\kappa_i| - |\kappa_j|\,\right|`.
- **Velocity-perturbation distance** (optional): absolute difference of the
  ``VELOCITY_PERTURBATION`` property after Savitzky-Golay filtering
  (``window_length=window``, ``polyorder=2``), when the property exists.

Each component is normalized by its maximum (when non-zero) so that the weights
can be interpreted consistently. The final DTW cost is:

.. math::

	 C = w_v\,C_v + w_\kappa\,C_\kappa + w_d\,C_d

with user weights ``vel_perturb_weight``, ``curvature_weight`` and
``distance_weight``.

**DTW alignment and stored links**

The DTW implementation is provided by the ``dtw-python`` package. DTW is applied to
the cost matrix (``dtw.dtw``) with a configurable step pattern (``pattern``, default
``"asymmetric"``). The resulting warping path is converted into a list of indices in
the previous centerline (``dtw.warp``).

pyBenD then stores the matching in both directions:

- On the *current* centerline: ``index_cl_pts_prev_centerline`` stores, for
  each current point index, the matched index in the previous centerline (or
  ``-1`` when no acceptable match exists).
- On the *previous* centerline: ``index_cl_pts_next_centerline`` stores, for
  each previous point index, the list of current indices that map to it
  (one-to-many links may occur due to DTW).

In addition, the per-point link lists on each channel point ``ClPoint`` are updated:
``cl_pt_index_prev`` and ``cl_pt_index_next``.

Centerline tracking is generally time-consuming. When multiple CPU cores are available,
matching of multiple consecutive pairs can be parallelized.

Bend connection (bend-to-bend tracking)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once bends have been identified on each centerline (see previous sections),
pyBenD can connect bends through time and build
:class:`~pybend.model.BendEvolution.BendEvolution` objects using
:meth:`~pybend.model.CenterlineCollection.CenterlineCollection.connect_bends`.
Three strategies are implemented (``method`` argument):

1) Apex-distance connection
""""""""""""""""""""""""""""

With ``method=BendConnectionMethod.APEX`` (:class:`~pybend.model.enumerations.BendConnectionMethod`),
bends are connected by minimizing the Euclidean distance between successive *apex points*
(``Bend.index_apex``). The search is performed backward through time. For each valid bend
at the current age:

- candidate trajectories are restricted to those whose last bend belongs to the
  immediately previous age and has the same bend side (``Bend.side``),
- the closest apex is selected if its distance is below ``dmax``;
  otherwise a new trajectory is started.

2) Centroid-distance connection
""""""""""""""""""""""""""""""""

With ``method=BendConnectionMethod.CENTROID`` (:class:`~pybend.model.enumerations.BendConnectionMethod`),
the same logic is applied but the distance is computed between bend centroids
(``Bend.pt_centroid``) rather than apex points.

3) Connection from point matching
""""""""""""""""""""""""""""""""""

With ``method=BendConnectionMethod.MATCHING`` (:class:`~pybend.model.enumerations.BendConnectionMethod`),
bend association is derived from the channel-point tracking result and therefore requires that
:meth:`~pybend.model.CenterlineCollection.CenterlineCollection.match_centerlines`
has been run first.

For each bend of age ``key`` and its successive centerline ``next_key``, pyBenD
counts how many channel points inside the bend (inflection points excluded)
connect to each bend in ``next_key`` via the DTW links (``cl_pt_index_next``).
Connected bends are retained when the fraction of connected points exceeds a
threshold (currently 0.2).

Reciprocal connections are stored on bends as lists of unique identifiers
(``bend_uid_next`` and ``bend_uid_prev``). Finally, connected components of this
directed/undirected connection graph are collected recursively to build the
:class:`~pybend.model.BendEvolution.BendEvolution` list.

Validity of BendEvolution objects
""""""""""""""""""""""""""""""""""

After bend connection, each :class:`~pybend.model.BendEvolution.BendEvolution` contains
a mapping from ages to bend indices (``bend_indexes``). A :class:`~pybend.model.BendEvolution.BendEvolution`
can be flagged as valid if it contains enough time steps; this is controlled by
``bend_evol_validity``.

Section across channel belt
----------------------------

pyBenD can build 2D stratigraphic sections across a channel belt by intersecting a
set of user-defined (or automatically generated) section lines with all centerlines
stored in a :class:`~pybend.model.CenterlineCollection.CenterlineCollection`.

The workflow is implemented in
:meth:`~pybend.model.CenterlineCollection.CenterlineCollection.find_points_on_sections`
and produces a list of :class:`~pybend.model.Section.Section` objects, each storing
the channel position through time in a section-aligned coordinate system.

Defining section lines
^^^^^^^^^^^^^^^^^^^^^^

Sections are defined in plan view as straight line segments (``shapely.LineString``)
stored in ``CenterlineCollection.section_lines``. Two approaches are available:

- **Manual definition** with
  :meth:`~pybend.model.CenterlineCollection.CenterlineCollection.set_section_lines`,
  by providing lists of start and end coordinates (``pts_start`` and ``pts_end``).
  This is the approach used in ``notebooks/bend_kinematics_analysis.ipynb`` where
  section endpoints are chosen so that each section intersects the centerlines of a
  given channel belt.
- **Automatic definition** with
  :meth:`~pybend.model.CenterlineCollection.CenterlineCollection.create_section_lines`.
  The method uses the youngest (last) centerline and creates one section per
  *valid* interior bend (first and last bends are skipped). The section line
  depends on the selected :class:`~pybend.model.enumerations.CreateSectionMethod`:

  - ``MIDDLE``: line from bend apex to chord midpoint (``Bend.pt_center``).
  - ``CENTROID``: line from bend apex to bend centroid (``Bend.pt_centroid``).
  - ``APEX``: line from the bend apex to the midpoint between neighboring bend
	 apexes (fallbacks use bend midpoints when apexes are undefined).

Finding channel location on a section
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Given a section line, pyBenD searches for its intersections with every centerline
polyline. For each age:

1. A local search window is built around the section line to avoid testing all
	segments.
2. Each candidate centerline segment (two consecutive
	:class:`~pybend.model.ClPoint.ClPoint`) is intersected with the section line.
3. When an intersection exists, channel-point properties are interpolated at the
	intersection location using the relative distance along the segment.

Each intersection produces an :class:`~pybend.model.Isoline.ChannelCrossSection`
isoline (type ``CHANNEL``) whose reference point is the interpolated
:class:`~pybend.model.ClPoint.ClPoint`. An idealized cross-section geometry is then
generated with :meth:`~pybend.model.Isoline.ChannelCrossSection.complete_channel_shape`
(parabolic shape whose width and depth come from the reference point properties).

Finally, if a section line intersects enough centerlines (controlled by the
``thres`` argument), a :class:`~pybend.model.Section.Section` object is created and
stored in ``CenterlineCollection.sections``. During this process, the bend
intersected by each channel position is also identified from the intersected point
index and the bend is tagged with the section index (used later for kinematic
summaries).

Section coordinate system
^^^^^^^^^^^^^^^^^^^^^^^^^

Within a :class:`~pybend.model.Section.Section`, the channel position at each age is
stored relative to the first intersected channel position according to the
coordinates ``(d, dz)``, where:

- :math:`d` is the signed lateral position along the section, measured relative to the
  first channel occurrence on the section.
- :math:`dz` is the vertical offset relative to that first occurrence.

The sign of :math:`d` is determined from the user-provided flow direction (``flow_dir``):
the migration vector between the reference channel position and the current one is
projected onto the direction perpendicular to ``flow_dir`` to assign the left/right
side of migration. These :math:`(d, dz)` coordinates form ``Section.isolines_origin`` and
are used to compute stacking-pattern classes
(:meth:`~pybend.model.Section.Section.get_stacking_pattern_type`),
apparent channel displacements
(:meth:`~pybend.model.Section.Section.channel_apparent_displacements`), and
section-averaged mobility metrics
(:meth:`~pybend.model.Section.Section.section_averaged_channel_displacements`).

Stacking pattern classification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When a channel belt is observed on a 2D stratigraphic section, the successive
intersections of the channel with the section can be interpreted as a trajectory
of the channel (here, represented by the channel reference point of each
intersected :class:`~pybend.model.Isoline.ChannelCrossSection`).

pyBenD classifies this trajectory into stacking-pattern types using
:meth:`~pybend.model.Section.Section.get_stacking_pattern_type`. The implemented
classification follows the workflow used in:

Lemay, M., Grimaud, J. L., Cojan, I., Rivoirard, J., and Ors, F. (2024).
*Submarine channel stacking patterns controlled by the 3D kinematics of meander
bends*. Geological Society, London, Special Publications, SP540-2022-143.
`doi.org/10.1144/SP540-2022-143 <https://doi.org/10.1144/SP540-2022-143>`_.

Step classification
"""""""""""""""""""

The classification operates on the ordered list ``Section.isolines_origin`` of
channel points for each time step containing pairs ``(d, dz)``.
For each consecutive pair of time steps, pyBenD computes a signed lateral
migration increment:

.. math::

	 \Delta d_k = d_{k} - d_{k-1}

and discretizes it into a ternary sequence:

- ``0`` if :math:`|\Delta d_k| < \texttt{mig_threshold}` (treated as aggradation-only),
- ``+1`` if :math:`\Delta d_k \ge \texttt{mig_threshold}`,
- ``-1`` if :math:`\Delta d_k \le -\texttt{mig_threshold}`.

The parameter ``mig_threshold`` therefore controls when lateral motion is
considered significant. In ``notebooks/bend_kinematics_analysis.ipynb`` it is set
proportionally to channel width (``mig_threshold = 0.0125 * width``).

From steps to stacking-pattern types
""""""""""""""""""""""""""""""""""""

Let :math:`f_0`, :math:`f_{+}`, and :math:`f_{-}` be the fractions of steps classified as
``0``, ``+1`` and ``-1`` respectively. A first-order decision is made from these
fractions using ``frac_threshold`` (default 0.95; also used in the notebook):

- **Aggradation Only** (``StackingPatternType.AGGRADATION``) if :math:`f_0` exceeds
	``frac_threshold``.
- **One Way** (``StackingPatternType.ONE_WAY``) if either :math:`f_{+}` or :math:`f_{-}`
	exceeds ``frac_threshold``.

When the trajectory contains both aggradation steps and a dominant migration
direction such that :math:`(f_0 + f_{+})` or :math:`(f_0 + f_{-})` exceeds
``frac_threshold``, pyBenD distinguishes between:

- **Aggradation then One Way** (``StackingPatternType.AGGRAD_ONE_WAY``): a
	sustained early aggradation phase followed by a sustained migration phase.
- **One Way** (``StackingPatternType.ONE_WAY``): migration dominates from early
	time steps.

In practice, the implementation groups consecutive steps of the dominant sign and
of ``0`` into runs, and compares the run lengths to a small fraction of the
record length (``begin_threshold``, default 0.1). This is the criterion used to
decide whether the initial part of the trajectory is long enough to be treated as
a distinct aggradation phase.

When neither direction dominates and aggradation is not overwhelming, the
classification is based on the number of direction reversals after removing
aggradation steps (``0``):

- **Two Way** (``StackingPatternType.TWO_WAYS``) when the non-zero migration
	steps can be reduced to two main runs of opposite sign.
- **Multiple Ways** (``StackingPatternType.MULTIPLE_WAYS``) when more than two
	alternating runs are required.

The resulting class is stored in ``Section.stacking_pattern_type`` and can be
used downstream to compute “bend-scale” vs “whole-trajectory” section-averaged
mobility metrics (see
:meth:`~pybend.model.Section.Section.section_averaged_channel_displacements`).


Kinematic Parameter Calculation
--------------------------------

In pyBenD, **kinematic parameters** quantify how the channel (or parts of it)
move through time, once geometric entities have been linked between successive
centerlines.

Two complementary families of kinematic measurements are commonly used:

- **Real (planform) kinematics**: measured from successive mapped/simulated
  centerlines, after point-to-point tracking (DTW matching).
- **Apparent (section-based) kinematics**: measured from channel positions
  recorded on stratigraphic sections, in a section-aligned
  coordinate system.

The notebooks
``notebooks/seine_river_migration_analysis.ipynb`` and
``notebooks/bend_kinematics_analysis.ipynb`` illustrate these computations.

Prerequisites
^^^^^^^^^^^^^

Real kinematics require:

1. A :class:`~pybend.model.CenterlineCollection.CenterlineCollection` built
	from multiple centerlines (ages/iterations).
2. Channel-point tracking computed with
	:meth:`~pybend.model.CenterlineCollection.CenterlineCollection.match_centerlines`.

Depending on the metric, you may also need bends to be detected on each
centerline (``find_bends=True``) and/or cross-sections to be defined.

Channel-point migration (Seine river example)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The simplest kinematic metric is the **migration distance** of each
:ref:`channel point <glossary-channel-point>` between two consecutive centerlines.

After DTW matching, each centerline of age :math:`t` stores an index mapping
``index_cl_pts_prev_centerline`` such that for a point :math:`i` on the current
centerline, the matched point index on the previous centerline is :math:`j`.

Let :math:`\mathbf{p}_t(i)` be the XY position of point :math:`i` at age :math:`t` and
:math:`\mathbf{p}_{t-1}(j)` the matched XY position at age :math:`t-1`. The planform
migration vector and distance are:

.. math::

	 \Delta \mathbf{p}(i) = \mathbf{p}_t(i) - \mathbf{p}_{t-1}(j)

.. math::

	 d_{\mathrm{mig}}(i) = \|\Delta \mathbf{p}(i)\|

In ``notebooks/seine_river_migration_analysis.ipynb``, this is implemented by
looping over the matched indices and computing
``cpf.distance(pt1, pt2)`` for each valid match.

This point-wise migration can then be summarized (mean, median, histogram) or
mapped along the centerline.

From trajectories to kinematic components
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Many bend-scale metrics are computed by first building a **trajectory**: a time
ordered list of positions of a tracked entity (apex, bend-average position,
any point index, etc.).

Given a list of points ``l_pt`` (2D or 3D),
:func:`~pybend.algorithms.centerline_process_functions.compute_point_displacements`
computes:

- ``local_disp``: incremental displacements between successive time steps,
  with columns ``(dX, dY, dZ, dMig)``.

- ``whole_disp``: displacement between first and last trajectory points,
  again as ``(DX, DY, DZ, DMig)``.

Internally, the XY displacement is projected into a user-defined local basis
(a change of coordinates). This is what allows the notebooks to express
displacements *across* or *along* a chosen direction.

Bend-average migration (bend_kinematics_analysis example)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The bend-averaged kinematics shown in
``notebooks/bend_kinematics_analysis.ipynb`` follow this logic:

1. Select a bend on a reference centerline (often the youngest one).
2. Build the list of point indices that belong to the bend
	(from upstream to downstream inflection points).
3. Step through ages and use ``index_cl_pts_prev_centerline`` to follow each
	point to its match on the next centerline.
4. For each consecutive pair of ages, compute the mean migration vector across
	all bend points (excluding unmatched points):

	.. math::

		 \Delta \bar{\mathbf{p}} = \frac{1}{N}\sum_{k=1}^N
		 \left(\mathbf{p}_t(k) - \mathbf{p}_{t-1}(k')\right)

	This produces one mean vector per time step; integrating those vectors over
	time yields a bend-average trajectory.

5. Optionally smooth/resample that trajectory (the notebook uses a
	resampling step followed by a Savitzky-Golay filter).
6. Compute incremental and total displacements with
	:func:`~pybend.algorithms.centerline_process_functions.compute_point_displacements`.

The notebook stores bend-scale metrics from the resulting ``whole_disp``:

- ``DxBend``, ``DyBend``: absolute values of the displacement components in the chosen local basis.
- ``DmigBend``: total planform migration magnitude (``DMig``).

Apex migration (bend_kinematics_analysis example)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The apex trajectory is computed similarly, but instead of averaging over all
points of the bend, the notebook tracks only the apex index
(``Bend.index_apex``) through time using the DTW index mapping. The resulting
trajectory is then processed with the same displacement routine, yielding
``DxApex``, ``DyApex``, and ``DmigApex``.

Section-based (apparent) kinematics and mobility numbers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When stratigraphic sections are available, pyBenD can compute *apparent*
displacements along a section using
:meth:`~pybend.model.Section.Section.channel_apparent_displacements`.

Given successive channel positions on the section (stored as
``Section.isolines_origin``), the method computes incremental lateral and
vertical displacements:

.. math::

	 \Delta x = x_{t} - x_{t-1},\qquad \Delta z = z_{t} - z_{t-1}

and the local (dimensionless) :ref:`stratigraphic mobility number <glossary-stratigraphic-mobility-number>`:

.. math::

	 M_{\mathrm{sb}} = \frac{\Delta x}{\Delta z}\,\frac{H}{W}

where :math:`W` is channel width and :math:`H` is channel depth (see
:meth:`~pybend.model.Section.Section.section_averaged_channel_displacements`,
after Jobe et al., 2016).

In ``notebooks/bend_kinematics_analysis.ipynb``, analogous :ref:`mobility numbers <glossary-mobility-number>` are
also reported for real planform kinematics by combining a planform migration
magnitude (``DmigApex`` or ``DmigBend``) with a section-derived thickness
estimate (``DzApex`` / ``DzBend``):

.. math::

	 M = \frac{D_{\mathrm{mig}}}{D_z}\,\frac{H}{W}

This allows comparing planform-derived kinematics ("real") to section-derived
kinematics ("recorded") on the same dimensionless scale.

