Python Bend Dynamics - pyBenD
=============================

``pyBenD``, which stands for ``Python`` Bend Dynamics, is a Python package dedicated to meandering system morphodynamic analysis. 

``pyBenD`` consists in:

* a data structure that stores individual channel centerlines and successive centerlines of a same channel 
migrating over time from various input file format (csv, kml, etc.)
* tools that automatically detect meander bends and characteristics points such as inflection points, bend apex, or bend center
* tools to compute meander bend morphometric parameters
* tools to compute channel lateral migration rates
* tools to compute meander bend kinematics parameters.


Installation
-------------

To install pyBenD, you may either clone the ``GitHub`` `repository`<https://github.com/martin-lemay/pyBenD.git>   
or download the `wheel`<>. In both cases, it is recommended to use a 
`virtual Python environment`<https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#create-and-use-virtual-environments>.

If you downloaded the wheel, install the package from the following commands:

* using pip (where .venv is your virtual environment directory):

.. code-block:: bash
   
   source .venv/bin/activate
   pip install pybend-1.0.0-py3-none-any.whl

* using conda (where venv is the name of an existing conda virtual environment):

.. code-block:: bash
   
   conda activate venv
   conda install pybend-1.0.0-py3-none-any.whl


From pyBenD source, run the following commands:

* using pip and a virtual environment:

.. code-block:: bash
   
   cd path/to/install/dir/
   git clone https://github.com/martin-lemay/pyBenD.git
   cd pyBenD
   source .venv/bin/activate
   pip install ./

* using conda:

.. code-block:: bash
   
   cd path/to/install/dir/
   git clone https://github.com/martin-lemay/pyBenD.git
   cd pyBenD
   conda activate venv
   conda install ./



Packages
-------------

.. toctree::
   :maxdepth: 1

   pybend
   publications



Related publications
---------------------

.. toctree::
   :maxdepth: 1

   publications
