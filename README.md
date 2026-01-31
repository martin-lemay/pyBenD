[![CI](https://github.com/martin-lemay/pyBenD/actions/workflows/python-package.yml/badge.svg)](https://github.com/martin-lemay/pyBenD/actions)
[![docs](https://readthedocs.com/projects/mlemay-pybend/badge/?version=latest)](https://mlemay-pybend.readthedocs.io/en/latest/)

# pyBenD (Python Bend Dynamics)
pyBenD is a Python library to process channel centerlines of meandering systems and quantify meander morphology and dynamics (bends, cross-sections, morphometry, kinematics, and apex detection). It replaces the legacy [ChannelPy](https://github.com/martin-lemay/ChannelPy) repository.

This project sets the data structure to store channel centerline properties and centerline evolution (a set of successive position of a river centerline) through time. It provides methods to create cross-sections and measure meander bend geomorphic and kinematic parameters.

The methodology for channel point tracking is inspired from [zsylvester/meanderpy](https://github.com/zsylvester/meanderpy).

A full documentation of the code can be found here: https://mlemay-pybend.readthedocs.io/en/latest/


If you use pyBenD, please cite one of the related papers and/or its reference on Zenodo:

* pyBenD v1.0.0: Lemay, M. (2026). pyBenD: Python Bend Dynamics (v1.0.0). Zenodo. https://doi.org/10.5281/zenodo.18442370 
* pyBenD develop version: Lemay, M. (2026). pyBenD: Python Bend Dynamics (v1.0.0). Zenodo. https://doi.org/10.5281/zenodo.18442369 

## Quickstart

Install from GitHub:

```bash
pip install git+https://github.com/martin-lemay/pyBenD.git
```

Minimal example (single centerline from CSV and morphometry analysis):

```python
from pybend.io import load_centerline_dataset_from_csv
from pybend import Centerline
from pybend import Morphometry

# 1) Load centerline points (must be ordered in flow direction)
# Default separator is ';' and default columns are X, Y (and optional Z)
dataset = load_centerline_dataset_from_csv(
  "path/to/centerline.csv",
  x_prop="X",
  y_prop="Y",
  z_prop="Z",
  sep=";",
)

# 2) Build a Centerline object (resampling + smoothing + bend detection)
cl = Centerline(
  age=0,
  dataset=dataset,
  spacing=50.0,         # target distance between points (m)
  smooth_distance=200., # smoothing distance (m)
  sinuo_thres=1.05,     # validity threshold for bends
  n=2,                  # exponent for curvature-distribution apex detection
  find_bends=True,
)

# 3) Morphometry analysis
morpho = Morphometry(cl)
# returns pandas DataFrame
metrics = morpho.compute_bends_morphometry(valid_bends=False) # morphometry computed only on valid bends if valid_bends is True

# 4) (Optional) Print average metrics
print("Centerline-averaged bend geometry")
print(metrics.mean(axis=0))
print()

```

Supported input formats:

- CSV: use `load_centerline_dataset_from_csv()` (expects at least X/Y columns; optional Z). Default separator is `;`.
- KML: use `load_centerline_dataset_from_kml()`.

Tip: example datasets are available in `notebooks/data/` and `tests/data/`.

## Installation

Requirements:

- Python >= 3.12

Install from GitHub:

```bash
pip install git+https://github.com/martin-lemay/pyBenD.git
```

For development and tests:

```bash
pip install -e .[dev,test]
```

## Contributing

Contributions are welcome — bug reports, feature requests, docs improvements, and code changes.

### Workflow (issues + PR/MR)

- Create an **issue** first to describe the bug / enhancement (with minimal reproducible example when relevant).
- Create a **Pull Request / Merge Request** that **addresses one issue**.
  - Reference the issue in the PR description (e.g. `Fixes #123`).
  - Keep changes focused and include tests/docs updates when applicable.

### Local setup

```bash
pip install -e .[dev,test]
```

If you plan to build the docs locally, install the doc build dependencies as well:

```bash
pip install -r requirements.txt
```

### Formatting, linting, typing, tests

Run these from the repository root:

```bash
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
```

### Build the docs locally

```bash
python -m sphinx -b html docs docs/_build/html
```

Then open `docs/_build/html/index.html` in your browser.

### What is checked in CI

On each Pull Request, GitHub Actions runs:

- `ruff check` (lint; currently non-blocking in CI)
- `mypy` (static type checks)
- `pytest` (tests + doctests)

## Notebooks and related publications

The `notebooks/` folder contains end-to-end case studies used in publications. If you use pyBenD in your research, please cite one of these articles:

- to measure channel geomorphic parameters (see this [Jupyter notebook](https://github.com/martin-lemay/pyBenD/blob/main/notebooks/meander_morphometry_analysis.ipynb)):

  Lemay, M., Grimaud, J. L., Cojan, I., Rivoirard, J., & Ors, F. (2020). **Geomorphic variability of submarine channelized systems along continental margins: Comparison with fluvial meandering channels**. Marine and Petroleum Geology, 115, 104295. https://doi.org/10.1016/j.marpetgeo.2020.104295

- to measure channel lateral migration (see this [Jupyter notebook](https://github.com/martin-lemay/pyBenD/blob/main/notebooks/seine_river_migration_analysis.ipynb)):

  Grimaud, J. L., Gouge, P., Huyghe, D., Petit, C., Lestel, L., Eschbach, D., Lemay, M., Catry, J., Quaisse, I., Imperor, A., Szewczyk, L., Mordant, D. **Lateral river erosion impacts the preservation of Neolithic enclosures in alluvial plains**. Sci Rep 13, 16566 (2023). https://doi.org/10.1038/s41598-023-43849-6

- to analyse channel evolution and measure kinematics (see this [Jupyter notebook](https://github.com/martin-lemay/pyBenD/blob/main/notebooks/bend_kinematics_analysis.ipynb)):

  Lemay, M., Grimaud, J. L., Cojan, I., Rivoirard, J., & Ors, F. (2024). **Submarine channel stacking patterns controlled by the 3D kinematics of meander bends**. Geological Society, London, Special Publications, SP540-2022-143. https://doi.org/10.1144/SP540-2022-143. 

- meander bend apex detection algorithm based on the cumulative curvature spatial distribution (see this notebook: [bend_apex_detection.ipynb](https://github.com/martin-lemay/pyBenD/blob/main/notebooks/bend_apex_detection.ipynb)):

  Lemay, M., Grimaud, J. L., & Limaye, A. B. (submitted to Earth Surface Processes and Landforms) **Automated detection of the apex of a meander bend using the cumulative spatial distribution of curvature.**

## Future work

- Bend kinematics metrics


## Credits

pyBenD was written by [Martin Lemay](https://github.com/martin-lemay) <br>[![ORCID Badge](https://img.shields.io/badge/ORCID-A6CE39?logo=orcid&logoColor=fff&style=flat-square)](https://orcid.org/my-orcid?orcid=0000-0002-5538-7885)</br>


## License

pyBenD is licensed under [MIT license](https://mit-license.org/).
