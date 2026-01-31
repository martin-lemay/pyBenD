Input data
==========

pyBenD primarily works with channel centerline data, which can be provided in various
formats, including CSV and KML. The discretized centerline data should include point
coordinates (x, y) ordered according to flow direction, and can optionally include
elevation (z) and other attributes.

pyBenD defines two main objects:

* ``Centerline``: represents a single channel centerline at a specific time.
* ``CenterlineCollection``: represents a series of channel centerlines over time,
    allowing analysis of channel kinematics.

.. NOTE::

    Centerline data can be obtained from manual digitization using GIS software,
    or extracted from Remote Sensing data using specialized tools, such as:

    * Python tools:
        - `RivaMap <https://github.com/isikdogan/rivamap>`_
        - `PyRIS <https://github.com/hchicchon/pyris>`_

    * Matlab tool: `RivMap <https://www.mathworks.com/matlabcentral/fileexchange/58264-rivmap-river-morphodynamics-from-analysis-of-planforms>`_
    * Google Earth Engine: `RivWidthCloud <https://ieeexplore.ieee.org/document/8752013>`_


Loading Centerline Data
------------------------

The main entry point for loading a single centerline is the 
:meth:`~pybend.io.centerline_io.load_centerline_from_file` function in the ``pybend.io`` module.
This function can read centerline data from different file formats and convert them
into a ``pandas.DataFrame``, from which a :class:`~pybend.model.Centerline.Centerline`
object can be instantiated.

**CSV format**

The CSV file should contain columns for x and y coordinates. Optionally, it can include
a z column for elevation data. Additional columns can be included for other attributes.

See for instance the file in test data: ``tests/data/centerline_xyz_data.csv``.

*Example of loading centerline data from a CSV file:*

.. code-block:: python
        
    import os
    import pandas as pd
    from pybend import load_centerline_from_file, CenterlineIOFormat

    # filepath needs to be adapted to your local setup
    filepath = "tests/data/centerline_xyz_data.csv"
    dataset: pd.DataFrame = load_centerline_from_file(
                filepath, kind=CenterlineIOFormat.CSV,
                x_prop="X", y_prop="Y", z_prop="Z"
            )


.. NOTE::

    The ``x_prop``, ``y_prop``, and ``z_prop`` parameters specify the column names
    for the x, y, and z coordinates in the CSV file. If the z column is not present,
    the ``z_prop`` parameter can be omitted.

    The separator for CSV file can be specified using the ``sep`` parameter in the
    :meth:`~pybend.io.load_centerline_from_file` function. By default, it is set to
    a semi-comma (`;`).

    Additional attributes are loaded by default if present in the CSV file, but can be
    dropped by specifying the ``drop_columns`` parameter as a list of column names to drop.

**FLUMY CSV format**

The FLUMY CSV format is a specific structure used by the 
`FLUMY software <https://flumy.minesparis.psl.eu/>`_. It typically includes columns for
x, y, z coordinates, and may also contain additional metadata, separated by a ';'.
The exact column names and order should follow the FLUMY specifications:

- Iteration: age of centerline point
- Dist_previous: distance to previous point
- Curv_abscissa: curvilinear abscissa
- Cart_abscissa: cartesian abscissa
- Cart_ordinate: cartesian ordinate
- Elevation: elevation of point (without regional slope)
- Curvature: curvature at point
- Vel_perturb: velocity perturbation
- Velocity: mean velocity
- Mean_depth: mean water depth
- True_elevation: true elevation taking into account the regional slope

See for instance the file in test data: ``tests/data/centerline_flumy_data.csv``.

*Example of loading centerline data from a FLUMY CSV file:*

.. code-block:: python

    import os
    import pandas as pd
    from pybend import load_centerline_from_file, CenterlineIOFormat

    # filepath needs to be adapted to your local setup
    filepath = "tests/data/centerline_flumy_data.csv"
    dataset: pd.DataFrame = load_centerline_from_file(
                filepath, kind=CenterlineIOFormat.FLUMY_CSV
            )


.. NOTE::

    The separator for CSV file can be specified using the ``sep`` parameter in the
    ``load_centerline_from_file`` function. By default, it is set to a semi-comma (`;`).

**KML format**

KML files should contain LineString elements representing the channel centerline. The
coordinates should be in the order of longitude, latitude, and optionally altitude.

See for instance the file in test data: ``tests/data/centerline_kml.kml``.

*Example of loading centerline data from a KML file:*

.. code-block:: python

    import os
    import pandas as pd
    from pybend import load_centerline_from_file, CenterlineIOFormat

    # filepath needs to be adapted to your local setup
    filepath = "tests/data/centerline_kml.kml"
    dataset: pd.DataFrame = load_centerline_from_file(
                filepath, kind=CenterlineIOFormat.KML
            )


**Creating a ``Centerline`` object from ``DataFrame``**

Once the centerline data is loaded into a ``pandas.DataFrame``, a ``Centerline`` object
can be created by passing the DataFrame to the ``Centerline`` class constructor.
Additional parameters are needed to correctly sample the centerline, including:

* ``age`` (int): Age of the centerline.
* ``dataset`` (pd.DataFrame): DataFrame that contains channel point
    coordinates and properties. Points must be ordered according
    to flow direction.

* ``spacing`` (float): Target distance (m) between channel points after
    resampling. At first approximation, the spacing must be around channel half-width.
    If spacing equals 0, centerline is not resampled. If ``use_fix_nb_points`` is True,
    spacing becomes the number of points of the resampled centerline.

* ``smooth_distance`` (float): Smoothing distance (m) for Savitsky-Golay
    filter applied on channel path. A reasonable value is 5 times the channel width,
    corresponding to approximatively a meander wavelength.

* ``use_fix_nb_points`` (bool, optional): If True, the resampled
    centerline will contains exactly spacing points, otherwise,
    spacing is the targeted distance between 2 consecutive points. Defaults to False.

* ``curvature_filtering_window`` (int, optional): Number of points used
    for filtering curvature. Defaults to 5.

* ``sinuo_thres`` (float, optional): Sinuosity threshold used to
    discriminate valid bends. Bends whom sinuosity is below this threshold are considered
    invalid. Defaults to 1.05.

* ``n`` (float): exponent value for bend apex detection using curvature cumulative 
    spatial distribution method. Defaults to 2.

* ``compute_curvature`` (bool, optional): If True, recompute and filter
    curvature along channel points. Defaults to True.

* ``interpol_props`` (bool, optional): If True, interpolate channel point
    properties along channel points after resampling. Defaults to True.

* ``find_bends`` (bool, optional): If True, automatically compute
    curvature, interpolate properties, and detect meander bends
    along channel centerline. Defaults to True.


*Example of creating a Centerline object from a DataFrame:*

.. code-block:: python

    import os
    import pandas as pd
    from pybend import load_centerline_from_file, CenterlineIOFormat
    from pybend import Centerline
    # filepath needs to be adapted to your local setup
    filepath = "tests/data/centerline_xyz_data.csv"
    dataset: pd.DataFrame = load_centerline_from_file(
                filepath, kind=CenterlineIOFormat.CSV,
                x_prop="X", y_prop="Y", z_prop="Z"
            )

    centerline = Centerline(
        age=0,
        dataset=dataset,
        spacing=1.0,
        smooth_distance=5.0,
        use_fix_nb_points=False,
        curvature_filtering_window=5,
        sinuo_thres=1.05,
        n=2,
        compute_curvature=True,
        interpol_props=True,
        find_bends=True,
    )


Loading ``CenterlineCollection`` Data
-------------------------------------

Centerline collection data consists of multiple channel centerlines at different ages,
representing the same channel migrating over time. Centerline collection data can be
provided either as a single file containing multiple centerlines or as multiple files,
each representing a centerline at a specific age. In both cases, data are loaded as a 
dictionnary of ``pandas.DataFrame``, where keys are centerline ages, and values are the
corresponding centerline data, from which a :class:`~pybend.model.CenterlineCollection.CenterlineCollection`
object can be instantiated.

**Single file**

The main entry point for loading a centerline collection from a single file is the
:meth:`~pybend.io.centerline_collection_io.load_centerline_collection_from_a_file` function in the ``pybend.io``
module. The file must be a csv file containing a column indicating the age of each
centerline point, in addition to the x and y coordinates. Optionally, a z column for
elevation data and other attribute columns can be included. The file can be a user
defined csv file or a FLUMY csv file.

See for instance in test data the csv file: ``tests/data/centerline_collection_test_data.csv``,
and the FLUMY file ``tests/data/centerlines.csv``.

*Example of loading an evolving centerline from a FLUMY CSV file:*

.. code-block:: python

    import os
    import pandas as pd
    from pybend import load_centerline_collection_from_a_file, CenterlineIOFormat

    # filepath needs to be adapted to your local setup
    filepath = "tests/data/centerlines.csv"
    dataset: pd.DataFrame = load_centerline_collection_from_a_file(
                filepath, kind=CenterlineIOFormat.FLUMY_CSV
            )


.. NOTE::

    The separator for CSV file can be specified using the ``sep`` parameter in the
    ``load_centerline_from_file`` function. By default, it is set to a semi-comma (`;`).

*Example of loading an evolving centerline from a CSV file:*

.. code-block:: python
        
    import os
    import pandas as pd
    from pybend import load_centerline_collection_from_a_file, CenterlineIOFormat

    # filepath needs to be adapted to your local setup
    filepath = "tests/data/centerline_collection_test_data.csv"
    dataset: pd.DataFrame = load_centerline_collection_from_a_file(
                filepath, kind=CenterlineIOFormat.CSV,
                age_prop="Iteration", x_prop="Cart_abscissa", y_prop="Cart_ordinate", z_prop="Elevation"
            )


.. NOTE::

    The ``x_prop``, ``y_prop``, and ``z_prop`` parameters specify the column names
    for the x, y, and z coordinates in the CSV file. If the z column is not present,
    the ``z_prop`` parameter can be omitted.

    The separator for CSV file can be specified using the ``sep`` parameter in the
    ``load_centerline_from_file`` function. By default, it is set to a semi-comma (`;`).

    Additional attributes are loaded by default if present in the CSV file, but can be
    dropped by specifying the ``drop_columns`` parameter as a list of column names to drop.
    The age column is automatically dropped since information is saved.


**Multiple files**

The main entry point for loading a centerline collection from multiple files is the
:meth:`~pybend.io.centerline_collection_io.load_centerline_collection_from_multiple_files` function in the
``pybend.io`` module. The files can be either a csv file or a KML file. Each file define
a centerline at a specific age. The function requires as input a dictionnary where keys
are centerline ages, and values are the corresponding file paths.

*Example of loading an evolving centerline from multiple CSV files:*

.. code-block:: python

    import os
    import pandas as pd
    from pybend import load_centerline_collection_from_multiple_files, CenterlineIOFormat

    # filepaths need to be adapted to your local setup
    map = {
        10: "tests/data/centerline_collection_test_data10.csv",
        40: "tests/data/centerline_collection_test_data40.csv",
        70: "tests/data/centerline_collection_test_data70.csv",
    }
    dataset: pd.DataFrame = load_centerline_collection_from_multiple_files(
                map, kind=CenterlineIOFormat.CSV,
                x_prop="Cart_abscissa", y_prop="Cart_ordinate", z_prop="Elevation"
            )


.. NOTE::

    The ``x_prop``, ``y_prop``, and ``z_prop`` parameters specify the column names
    for the x, y, and z coordinates in the CSV file. If the z column is not present,
    the ``z_prop`` parameter can be omitted.

    The separator for CSV file can be specified using the ``sep`` parameter in the
    ``load_centerline_from_file`` function. By default, it is set to a semi-comma (`;`).

    Additional attributes are loaded by default if present in the CSV file, but can be
    dropped by specifying the ``drop_columns`` parameter as a list of column names to drop.


**Creating a ``CenterlineCollection`` object from a dictionnary of ``DataFrame``**

Once the centerline collection data is loaded into a dictionnary of ``pandas.DataFrame``, a
:class:`~pybend.model.CenterlineCollection.CenterlineCollection` object can be created by passing the dictionnary to the
:class:`~pybend.model.CenterlineCollection.CenterlineCollection` class constructor.

Additional parameters are needed to correctly sample the centerlines, similar to those used
for creating a single :class:`~pybend.model.Centerline.Centerline` object, including:

* ``map_centerline_data`` (dict[int, pd.DataFrame]): dictionnary
    containing for each age a ``pandas.DataFrame`` containing centerline data

* ``spacing`` (float): Target distance (m) between channel points after
    resampling. At first approximation, the spacing must be around channel half-width.
    If spacing equals 0, centerline is not resampled. If ``use_fix_nb_points`` is True,
    spacing becomes the number of points of the resampled centerline.

* ``smooth_distance`` (float): Smoothing distance (m) for Savitsky-Golay
    filter applied on channel path. A reasonable value is 5 times the channel width,
    corresponding to approximatively a meander wavelength.

* ``use_fix_nb_points`` (bool, optional): If True, the resampled
    centerline will contains exactly spacing points, otherwise,
    spacing is the targeted distance between 2 consecutive points. Defaults to False.

* ``curvature_filtering_window`` (int, optional): Number of points used
    for filtering curvature. Defaults to 5.

* ``sinuo_thres`` (float, optional): Sinuosity threshold used to
    discriminate valid bends. Bends whom sinuosity is below this threshold are considered
    invalid. Defaults to 1.05.

* ``n`` (float): exponent value for bend apex detection using curvature cumulative 
    spatial distribution method. Defaults to 2.

* ``compute_curvature`` (bool, optional): If True, recompute and filter
    curvature along channel points. Defaults to True.

* ``interpol_props`` (bool, optional): If True, interpolate channel point
    properties along channel points after resampling. Defaults to True.

* ``find_bends`` (bool, optional): If True, automatically compute
    curvature, interpolate properties, and detect meander bends
    along channel centerline. Defaults to True.


*Example of creating a CenterlineColection object from multiple CSV files:*

.. code-block:: python
        
    import os
    import pandas as pd
    from pybend import load_centerline_collection_from_multiple_files, CenterlineIOFormat
    from pybend import CenterlineCollection

    # filepaths need to be adapted to your local setup
    map = {
        10: "tests/data/centerline_collection_test_data10.csv",
        40: "tests/data/centerline_collection_test_data40.csv",
        70: "tests/data/centerline_collection_test_data70.csv",
    }
    dataset: pd.DataFrame = load_centerline_collection_from_multiple_files(
                map, kind=CenterlineIOFormat.CSV,
                x_prop="Cart_abscissa", y_prop="Cart_ordinate", z_prop="Elevation"
            )

    centerline_collection = CenterlineCollection(
        map_centerline_data=dataset,
        spacing=1.0,
        smooth_distance=5.0,
        use_fix_nb_points=False,
        curvature_filtering_window=5,
        sinuo_thres=1.05,
        n=2,
        compute_curvature=True,
        interpol_props=True,
        find_bends=True,
    )


.. WARNING::

    The main pitfall is that pyBenD does currently not include CRS neither unit management. 
    pyBenD works under the assumption that all centerlines in a collection are in the same
    CRS and units are in meters. Therefore, it is the user's responsibility to ensure loaded
    data meets these criteria.