###############################################################################
Examples
###############################################################################

Bend apex detection
*********************

The following notebook compare different methods to detect the apex of a meander bend
and compare morphometric parameters computed using these different methods. 
This notebook include the following features:

* create synthetic bend centerlines from Kinoshita curve
* import real meandering river centerlines from csv files
* detect meander bends and bend apex using different methods
* compute meander bend morphometric parameters

Related publication is:

Lemay, M., Grimaud, J. L., Limaye, A. B. (2026). Automated detection
of the apex of a meander bend using the cumulative spatial distribution of curvature.
Submitted to *Earth Surface Processes and Landforms*.

Download the notebook: `bend_apex_detection.ipynb <notebooks/bend_apex_detection.ipynb>`_


Bend kinematic analysis
************************

The following notebook compute 3D bend kinematics from successive centerlines of a
migrating meandering channel. Bend kinematics include apex and bend-average migration
rates, and apparent channel displacement on 2D cross-sections.

This notebook include the following features:

* import successive channel centerlines from csv files
* detect meander bends and bend apex over time
* compute bend kinematic parameters
* compute and display sections across channel paths
* compute apparent channel displacement on sections

Related publication is:

Lemay, M., Grimaud, J. L., Cojan, I., Rivoirard, J., & Ors, F. (2024). Submarine channel stacking patterns
controlled by the 3D kinematics of meander bends. Geological Society, London, Special Publications, SP540-2022-143.
`doi.org/10.1144/SP540-2022-143 <https://doi.org/10.1144/SP540-2022-143>`_. 

Download the notebook: `bend_kinematics_analysis.ipynb <notebooks/bend_kinematics_analysis.ipynb>`_


Meander morphometry analysis
*******************************

The following notebook compute morphometric parameters of meanders from centerlines
extracted from real submarine flow pathways. Parameters are compared together and with
fluvial meander characteristics.

This notebook include the following features:

* import channel centerlines from csv files
* detect meander bends and bend apex over time
* compute bend morphometric parameters
* statistical analysis of meander morphometric parameters

Related publication is:

Lemay, M., Grimaud, J. L., Cojan, I., Rivoirard, J., & Ors, F. (2020). Geomorphic variability of submarine channelized systems
along continental margins: Comparison with fluvial meandering channels. Marine and Petroleum Geology, 115, 104295.
`doi.org/10.1016/j.marpetgeo.2020.104295 <https://doi.org/10.1016/j.marpetgeo.2020.104295>`_

Download the notebook: `meander_morphometry_analysis.ipynb <notebooks/meander_morphometry_analysis.ipynb>`_


Seine River migration analysis
******************************

The following notebook analyze the migration of the Seine River (France) over
the last two centuries using historical maps. The analysis include
the computation of lateral migration rates and meander bend kinematics.

This notebook include the following features:
* import successive channel centerlines from csv files
* compute channel lateral migration rates

Related publication is:

Grimaud, J. L., Gouge, P., Huyghe, D., Petit, C., Lestel, L., Eschbach, D., Lemay, M.,
Catry, J., Quaisse, I., Imperor, A., Szewczyk, L., Mordant, D. Lateral river erosion
impacts the preservation of Neolithic enclosures in alluvial plains. 
Sci Rep 13, 16566 (2023). `doi.org/10.1038/s41598-023-43849-6 <https://doi.org/10.1038/s41598-023-43849-6>`_

Download the notebook: `seine_river_migration_analysis.ipynb <notebooks/seine_river_migration_analysis.ipynb>`_
