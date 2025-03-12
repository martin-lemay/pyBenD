# pyBenD
pyBenD stands for Python Bend Dynamics. This repository replaces [ChannelPy](https://github.com/martin-lemay/ChannelPy) repository.

This project sets the data structure to store channel centreline properties and centreline evolution (a set of successive position if a river centerline) through time. It provides methods to create cross-sections and measure meander bend geomorphic and kinematic parameters.

The methodology for channel point tracking algorithm is inspired from [zsylvester/meanderpy] (https://github.com/zsylvester/meanderpy).

## Related publications
If you use the code to measure channel geomorphic parameters (see this [Jupyter notebook](https://github.com/martin-lemay/pyBenD/notebooks/bend_apex_detection.ipynb)), please refer to the following publication:
- Lemay, M., Grimaud, J. L., Cojan, I., Rivoirard, J., & Ors, F. (2020). Geomorphic variability of submarine channelized systems along continental margins: Comparison with fluvial meandering channels. Marine and Petroleum Geology, 115, 104295. https://doi.org/10.1016/j.marpetgeo.2020.104295

If you use the code to measure channel lateral migration (see this [Jupyter notebook](https://github.com/martin-lemay/pyBenD/notebooks/seine_river_migration.ipynb)), please refer to:
- Grimaud, J. L., Gouge, P., Huyghe, D., Petit, C., Lestel, L., Eschbach, D., Lemay, M., Catry, J., Quaisse, I., Imperor, A., Szewczyk, L., Mordant, D. Lateral river erosion impacts the preservation of Neolithic enclosures in alluvial plains. Sci Rep 13, 16566 (2023). https://doi.org/10.1038/s41598-023-43849-6

If you use the code to analyse channel evolution and measure kinematics (see this [Jupyter notebook](https://github.com/martin-lemay/pyBenD/notebooks/bend_kinematics_analysis.ipynb)), please refer to the following publication:
- Lemay, M., Grimaud, J. L., Cojan, I., Rivoirard, J., & Ors, F. (2024). Submarine channel stacking patterns controlled by the 3D kinematics of meander bends. Geological Society, London, Special Publications, SP540-2022-143. https://doi.org/10.1144/SP540-2022-143. 

If you use the meander bend apex detection algorithm based on the entire curvtaure distribution (see this [Jupyter notebook](https://github.com/martin-lemay/pyBenD/notebooks/bend_apex_detection.ipynb)), please refer to:
- Lemay, M., Grimaud, J. L. (submitted to JGR Earth Surface) Where is meander bend apex? A new robust method for automatic detection of bend apex from curvature spatial distribution.

## License
pyBenD is licensed under [MIT license](https://mit-license.org/).
