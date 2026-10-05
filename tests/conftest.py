# SPDX-FileCopyrightText: Copyright 2025 Martin Lemay <martin.lemay@mines-paris.org>
# SPDX-FileContributor: Martin Lemay

__doc__ = """
Pytest configuration shared by all tests.

Figures are only saved to disk in tests, so use the non-interactive Agg
backend to avoid depending on a Tcl/Tk installation (TkAgg).
"""

import matplotlib

matplotlib.use("Agg")
