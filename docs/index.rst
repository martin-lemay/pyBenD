Python Bend Dynamics - pyBenD
=============================

``pyBenD``, which stands for Python Bend Dynamics, is a ``Python`` package dedicated to
meandering system morphodynamic analysis. 

``pyBenD`` consists in:

- a data structure that stores individual channel centerlines and successive centerlines
  of a same channel migrating over time from various input file format (csv, kml, etc.)
- tools that automatically detect meander bends and characteristics points such as
  inflection points, bend apex, or bend center
- tools to compute meander bend morphometric parameters
- tools to compute channel lateral migration rates
- tools to compute meander bend kinematics parameters.


.. NOTE::

   If you use pyBenD, please cite one of the related papers and/or its reference on Zenodo:

   - pyBenD v1.0.0: Lemay, M. (2026). pyBenD: Python Bend Dynamics (v1.0.0).
     Zenodo. `doi.org/10.5281/zenodo.18442370 <https://doi.org/10.5281/zenodo.18442370>`_
   - pyBenD develop version: Lemay, M. (2026). pyBenD: Python Bend Dynamics (v1.0.0).
     Zenodo. `doi.org/10.5281/zenodo.18442369 <https://doi.org/10.5281/zenodo.18442369>`_


Installation
-------------

To install pyBenD, you may either clone the ``GitHub`` `repository <https://github.com/martin-lemay/pyBenD.git>`_
or download an archive in
:download:`zip <https://github.com/martin-lemay/pyBenD/archive/refs/tags/v1.0.0.zip>` or
:download:`tar.gz <https://github.com/martin-lemay/pyBenD/archive/refs/tags/v1.0.0.tar.gz>` 
format. In both cases, it is recommended to use a 
`virtual Python environment <https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#create-and-use-virtual-environments>`_.

From pyBenD GitHub repository, run the following commands:

- using pip and a virtual environment:

.. code-block:: bash
   
   cd path/to/install/dir/
   git clone https://github.com/martin-lemay/pyBenD.git
   cd pyBenD
   source .venv/bin/activate
   pip install -e ./

- using conda:

.. code-block:: bash
   
   cd path/to/install/dir/
   git clone https://github.com/martin-lemay/pyBenD.git
   cd pyBenD
   conda activate venv
   conda install ./


If you downloaded an archive, install the package from the following commands
(replace `PATH/TO/pyBenD-1.0.0.zip` to the actual path to the downloaded archive):

- using pip (where .venv is your virtual environment directory):

.. code-block:: bash
   
   source .venv/bin/activate
   pip install PATH/TO/pyBenD-1.0.0.zip

- using conda (where venv is the name of an existing conda virtual environment):

.. code-block:: bash
   
   conda activate venv
   conda install PATH/TO/pyBenD-1.0.0.zip


Testing
---------

You can test pyBenD package using ``pytest`` (see the `homepage <https://pytest.org/>`_).

* To test the source distribution, run the following commands from pyBenD root directory:

.. code-block:: bash

   pytest ./

* To test the installed package, run the following commands:

.. code-block:: bash

   pytest --pyargs pybend


Contributing
--------------

Contributions are welcome — bug reports, feature requests, docs improvements, and code
changes.

Workflow (issues + PR/MR)
^^^^^^^^^^^^^^^^^^^^^^^^^

- Create an **issue** first to describe the bug / enhancement (with a minimal
   reproducible example when relevant).
- Create a **Pull Request / Merge Request** that **addresses one issue**.

  - Reference the issue in the PR description (e.g. ``Fixes #123``).
  - Keep changes focused and include tests/docs updates when applicable.

Local setup
^^^^^^^^^^^

.. code-block:: bash

   pip install -e .[dev,test]

If you plan to build the docs locally, install the doc build dependencies as well:

.. code-block:: bash

   pip install -r requirements.txt

Formatting, linting, typing, tests
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run these from the repository root:

.. code-block:: bash

   # Format
   ruff format .

   # Lint (optionally auto-fix)
   ruff check .
   ruff check --fix .

   # Type-check
   mypy .

   # Tests
   pytest

   # To mirror CI more closely (includes doctests)
   pytest ./ --doctest-modules

Build the docs locally
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   python -m sphinx -b html docs docs/_build/html

Then open ``docs/_build/html/index.html`` in your browser.

What is checked in CI
^^^^^^^^^^^^^^^^^^^^^

On each Pull Request, GitHub Actions runs:

- ``ruff check`` (lint; currently non-blocking in CI)
- ``mypy`` (static type checks)
- ``pytest`` (tests + doctests)
- ``sphinx`` (documentation build)

Contents
----------

.. toctree::
   :maxdepth: 1

   user_guide/user_guide
   examples
   howto/howto
   pybend/pybend   
   glossary
   publications
