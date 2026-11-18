###############################################################################
How-To
###############################################################################

Short recipes for the most common pyBenD tasks. Complete workflows on real
datasets are available in the :doc:`../examples` notebooks.

All snippets assume that pyBenD is installed (see :doc:`../index`) and that
coordinates are expressed in meters. Lengths given to pyBenD (spacing,
smoothing distance, thresholds) use the same unit as the coordinates.

Create a Centerline object
***************************

A :class:`~pybend.model.Centerline.Centerline` is built from an age and a
:class:`pandas.DataFrame` of ordered channel points (upstream to downstream).
Loaders from :mod:`pybend.io` return this DataFrame. When the centerline is
created, points are resampled every ``spacing`` meters, smoothed over
``smooth_distance`` meters, curvature is computed, and bends are detected.

From csv file
----------------------

The csv file must contain at least the X and Y coordinates of channel points,
in flow order. Elevation and any additional column (e.g., width, depth) are
optional; additional columns are kept as channel point properties and
interpolated on resampled points.

.. code-block:: text

    X;Y;Z;Width
    0.0;0.0;0.0;50.0
    0.98;0.17;0.0;50.0
    ...

.. code-block:: python

    import pybend.io as io
    from pybend.model.Centerline import Centerline

    age, dataset = io.load_centerline_from_file(
        "centerline.csv",
        kind=io.CenterlineIOFormat.CSV,
        x_prop="X",  # name of the X column
        y_prop="Y",  # name of the Y column
        z_prop="Z",  # name of the elevation column (optional)
        sep=";",
    )

    centerline = Centerline(
        age,
        dataset,
        spacing=10.0,  # distance between resampled points (m)
        smooth_distance=50.0,  # smoothing distance (m)
        sinuo_thres=1.05,  # bends below this sinuosity are invalid
        n=2,  # exponent used for bend apex detection
    )
    print(centerline.get_nb_bends(), centerline.get_nb_valid_bends())

Columns that are not needed can be removed with
``drop_columns=("col1", "col2")``. Flumy outputs are read with
``kind=io.CenterlineIOFormat.FLUMY_CSV``, which also returns the centerline
age stored in the file.

From kml file
----------------------

KML files (e.g., digitized in Google Earth) store longitude and latitude in
degrees. Set ``to_metric=True`` to convert them to local metric coordinates
(equirectangular projection centered on the first point, accurate for
centerlines spanning a few tens of kilometers). Otherwise, all distances are
expressed in degrees.

.. code-block:: python

    import pybend.io as io
    from pybend.model.Centerline import Centerline

    age, dataset = io.load_centerline_from_file(
        "centerline.kml", kind=io.CenterlineIOFormat.KML, to_metric=True
    )
    centerline = Centerline(age, dataset, spacing=10.0, smooth_distance=50.0)

The age returned for a KML file is always 0. Points are read from every
``<coordinates>`` element in document order; use ``keyword=...`` to read
another tag.

From inline code
----------------------

Any set of coordinates can be used by building the DataFrame directly. Column
names must follow :class:`~pybend.model.enumerations.PropertyNames`, and the
curvilinear abscissa must be provided.

.. code-block:: python

    import numpy as np
    import pandas as pd

    import pybend.algorithms.centerline_process_functions as cpf
    from pybend.model.Centerline import Centerline
    from pybend.model.enumerations import PropertyNames

    s = np.linspace(0.0, 4.0 * np.pi, 400)
    x = 100.0 * s
    y = 50.0 * np.sin(s)

    dataset = pd.DataFrame(
        {
            PropertyNames.CARTESIAN_ABSCISSA.value: x,
            PropertyNames.CARTESIAN_ORDINATE.value: y,
            PropertyNames.ELEVATION.value: 0.0,
        }
    )
    dataset[PropertyNames.CURVILINEAR_ABSCISSA.value] = (
        cpf.compute_cuvilinear_abscissa(np.column_stack((x, y)))
    )

    centerline = Centerline(0, dataset, spacing=10.0, smooth_distance=50.0)

Synthetic meander shapes (circular or Kinoshita bends) can be generated with
:mod:`pybend.algorithms.synthetic_bends`.

Compute Centerline morphometric parameters
*******************************************

:class:`~pybend.model.Morphometry.Morphometry` computes bend metrics
(arc length, wavelength, sinuosity, amplitude, extension, radius of curvature,
asymmetry, roundness, skewness, maturity index, Leopold wavelength and
amplitude) on a centerline whose bends were detected.

.. code-block:: python

    from pybend.model.Morphometry import Morphometry

    morph = Morphometry(centerline)

    # one row per bend, indexed by bend id
    metrics = morph.compute_bends_morphometry(valid_bends=True)
    print(metrics[["Wavelength", "Amplitude", "Sinuosity"]].describe())

    # a single metric on a single bend
    bend_id = centerline.get_valid_bend_indexes()[0]
    sinuosity = morph.compute_bend_sinuosity(bend_id)

Set ``valid_bends=False`` to include the bends whose sinuosity is below
``sinuo_thres``. Leopold wavelength and amplitude use neighboring bends and
are therefore NaN for the first and last bends.

Apex-dependent metrics (amplitude, extension, asymmetry) depend on the apex
detection method. The apex can be recomputed before computing the metrics,
for instance with ``centerline.find_all_bend_apex(n=3)``. See the
:doc:`../notebooks/bend_apex_detection` notebook for a comparison of methods.

Create a CenterlineCollection object
**************************************

A :class:`~pybend.model.CenterlineCollection.CenterlineCollection` stores the
successive centerlines of the same channel. Loaders return a dictionary that
maps each age to a centerline DataFrame; each centerline is then processed as
a :class:`~pybend.model.Centerline.Centerline` (multiprocessing is used when
several CPUs are available).

From a csv file
----------------------

All centerlines are stored in the same file, with a column giving the age of
each point:

.. code-block:: text

    Age;X;Y;Z
    10;0.0;0.0;0.0
    10;0.98;0.17;0.0
    ...
    20;0.0;0.0;0.0
    ...

.. code-block:: python

    import pybend.io as io
    from pybend.model.CenterlineCollection import CenterlineCollection

    map_dataset = io.load_centerline_collection_from_a_file(
        "centerlines.csv",
        kind=io.CenterlineIOFormat.CSV,
        x_prop="X",
        y_prop="Y",
        z_prop="Z",
        age_prop="Age",
        sep=";",
    )

    cl_collec = CenterlineCollection(
        map_dataset,
        spacing=10.0,
        smooth_distance=50.0,
        use_fix_nb_points=False,
        sinuo_thres=1.05,
        n=2,
    )
    print(cl_collec.get_all_ages())

Flumy outputs containing several iterations are read with
``kind=io.CenterlineIOFormat.FLUMY_CSV``.

From multiple csv files
------------------------

Each centerline is stored in its own file. Give a dictionary that maps each
age to its file; all files share the same column names.

.. code-block:: python

    map_file = {
        1850: "centerline_1850.csv",
        1900: "centerline_1900.csv",
        1950: "centerline_1950.csv",
    }
    map_dataset = io.load_centerline_collection_from_multiple_files(
        map_file, kind=io.CenterlineIOFormat.CSV, x_prop="X", y_prop="Y"
    )
    cl_collec = CenterlineCollection(
        map_dataset, spacing=10.0, smooth_distance=50.0,
        use_fix_nb_points=False,
    )

KML files are loaded the same way with ``kind=io.CenterlineIOFormat.KML``,
``directory=...`` (folder containing the files) and ``to_metric=True``.

Compute channel migration rates
*********************************

Migration is measured between equivalent channel points of successive
centerlines. These points are first connected with
:meth:`~pybend.model.CenterlineCollection.CenterlineCollection.match_centerlines`
(Dynamic Time Warping). The cost of connecting two points combines their
distance, velocity perturbation and curvature according to the given weights;
``dmax`` is the maximum distance allowed between connected points.

.. code-block:: python

    import numpy as np
    import pandas as pd

    width = 100.0  # channel width (m)
    cl_collec.match_centerlines(
        dmax=3.0 * width,
        distance_weight=1.0,
        vel_perturb_weight=0.0,
        curvature_weight=0.0,
    )

    rows = []
    ages = cl_collec.get_all_ages()
    for age_prev, age_next in zip(ages[:-1], ages[1:]):
        cl_prev = cl_collec.centerlines[age_prev]
        cl_next = cl_collec.centerlines[age_next]
        # index of the connected point in the previous centerline (-1: none)
        for i_next, i_prev in enumerate(cl_next.index_cl_pts_prev_centerline):
            if i_prev < 0:
                continue
            pt_next = cl_next.cl_points[i_next].pt
            pt_prev = cl_prev.cl_points[int(i_prev)].pt
            distance = np.linalg.norm(pt_next[:2] - pt_prev[:2])
            rate = distance / (age_next - age_prev)
            rows.append((age_next, i_next, distance, rate))

    migration = pd.DataFrame(
        rows, columns=["Age", "Point", "Distance", "Rate"]
    )
    print(migration.groupby("Age")["Rate"].median())

Rates are expressed in coordinate units per age unit (e.g., m/yr if ages are
years). See the :doc:`../notebooks/seine_river_migration_analysis` notebook
for a complete example.

Compute bend kinematics parameters
***********************************

Bend kinematics follow the same bend through time. Bends of successive
centerlines are connected into
:class:`~pybend.model.BendEvolution.BendEvolution` objects with
:meth:`~pybend.model.CenterlineCollection.CenterlineCollection.connect_bends`.
The apex trajectory of each BendEvolution is then computed, and its
displacements are measured with
:func:`~pybend.algorithms.centerline_process_functions.compute_point_displacements`.

.. code-block:: python

    import pybend.algorithms.centerline_process_functions as cpf
    from pybend.model.enumerations import BendConnectionMethod

    # connect bends: same side and closest apex in successive centerlines.
    # A BendEvolution is valid if it contains more than 2 bends.
    cl_collec.connect_bends(
        bend_evol_validity=2, method=BendConnectionMethod.APEX
    )

    # compute bend apex and smooth apex trajectories
    cl_collec.find_all_bend_apex(n=2, smooth_trajectory=True)

    for bend_evol in cl_collec.bends_evol:
        if not bend_evol.isvalid:
            continue
        ages = bend_evol.get_all_ages()
        local_disp, whole_disp = cpf.compute_point_displacements(
            bend_evol.apex_trajec_smooth
        )
        # columns: dX, dY, dZ, horizontal displacement
        apex_rate = whole_disp[3] / (ages[-1] - ages[0])
        print(f"Bend {bend_evol.id}: apex migration rate {apex_rate:.2f}")

``local_disp`` contains the displacement between each pair of successive
ages, and ``whole_disp`` the displacement between the first and last ages.
Displacements can be projected on a local frame by passing a direction to
``compute_point_displacements`` (``dir_trans``), for instance the direction
perpendicular to a section. Other connection methods are listed in
:class:`~pybend.model.enumerations.BendConnectionMethod`
(``CENTROID``, ``MATCHING``). The
:doc:`../notebooks/bend_kinematics_analysis` notebook shows a complete
analysis.

Create sections across channel paths
*************************************

A :class:`~pybend.model.Section.Section` gathers the successive channel
positions crossed by a section line. They are used to measure apparent
channel displacements and stacking patterns (2D view of the 3D bend
kinematics).

Section lines are either created automatically from the bends of the last
centerline, or set manually from start and end points:

.. code-block:: python

    import numpy as np

    from pybend.model.enumerations import CreateSectionMethod

    # automatic: one line per valid bend of the last centerline
    cl_collec.create_section_lines(CreateSectionMethod.APEX)

    # or manual: one line per (start, end) pair of (x, y) coordinates
    cl_collec.set_section_lines(
        pts_start=[(1000.0, 500.0), (3000.0, 800.0)],
        pts_end=[(1200.0, -500.0), (3100.0, -300.0)],
    )

    # create Section objects crossing more than 10 channels
    cl_collec.find_points_on_sections(
        thres=10, flow_dir=np.array([1.0, 0.0])
    )

Each section then computes apparent displacements and stacking pattern:

.. code-block:: python

    width, depth = 100.0, 5.0  # channel width and depth (m)

    for section in cl_collec.sections:
        # stacking pattern (one-way, two-ways, aggradation...); lateral
        # migration below mig_threshold is considered null
        pattern = section.get_stacking_pattern_type(
            mig_threshold=0.0125 * width
        )
        # lateral and vertical displacements between successive channels
        section.channel_apparent_displacements(norm_hor=width,
                                               norm_vert=depth)
        # section-averaged metrics (Dx, Dz, belt width and thickness,
        # stratigraphic mobility number)
        section.section_averaged_channel_displacements(norm_hor=width,
                                                       norm_vert=depth)
        print(section.id, pattern, section.averaged_disp["full"])

``averaged_disp["full"]`` is computed from the first to the last channel
crossed by the section. ``averaged_disp["bend"]`` covers only the last
migration phase when the stacking pattern is not one-way, which is why the
stacking pattern is computed first. The available automatic methods are
listed in
:class:`~pybend.model.enumerations.CreateSectionMethod` (``APEX``,
``MIDDLE``, ``CENTROID``).
