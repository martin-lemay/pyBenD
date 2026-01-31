###############################################################################
pyBenD Python Package
###############################################################################

pyBenD project contains:

* docs: documentation source
* notebooks: Jupyter notebooks used for related publications that show how to use pyBenD
   features
* tests: unit and integrated tests. Tests files contain numerous examples on how to use
   pyBenD.
* src: pyBenD source code


Main API
----------

The main API can be imported directly from pybend package:

.. code-block:: python

    from pybend import (
        AmplitudeType,
        Bend,
        BendConnectionMethod,
        BendEvolution,
        BendSide,
        Centerline,
        CenterlineCollection,
        CreateSectionMethod,
        FilterName,
        MorphometricNames,
        Morphometry,
        Section,
        CenterlineIOFormat,
        load_centerline_from_file,
        load_centerline_collection_from_a_file,
        load_centerline_collection_from_multiple_files,
    )

    # do anything with the imported classes and functions


Core model
^^^^^^^^^^^^

* :class:`~pybend.model.Centerline.Centerline`
* :class:`~pybend.model.CenterlineCollection.CenterlineCollection`
* :class:`~pybend.model.Bend.Bend`
* :class:`~pybend.model.BendEvolution.BendEvolution`
* :class:`~pybend.model.ClPoint.ClPoint`
* :class:`~pybend.model.Morphometry.Morphometry`
* :class:`~pybend.model.Section.Section`


Method & Types
^^^^^^^^^^^^^^^^^^

* Bend side: :class:`~pybend.model.enumerations.BendSide`
* Bend tracking method: :class:`~pybend.model.enumerations.BendConnectionMethod`
* Amplitude type: :class:`~pybend.model.enumerations.AmplitudeType`
* Morphometric names: :class:`~pybend.model.enumerations.MorphometricNames`
* Section creation method: :class:`~pybend.model.enumerations.CreateSectionMethod`
* Filter names: :class:`~pybend.model.enumerations.FilterName`


IO
^^^^^^

* supported formats: :class:`~pybend.io.common.CenterlineIOFormat` 
* Centerline importers: :meth:`~pybend.io.centerline_io.load_centerline_from_file`
* CenterlineCollection importers
   - :meth:`~pybend.io.centerline_collection_io.load_centerline_collection_from_a_file`
   - :meth:`~pybend.io.centerline_collection_io.load_centerline_collection_from_multiple_files`


Packages
-----------

.. toctree::
   :maxdepth: 1

   pybend.algorithms
   pybend.io
   pybend.model
   pybend.utils
