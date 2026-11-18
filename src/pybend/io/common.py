# SPDX-FileCopyrightText: Copyright 2025 Martin Lemay <martin.lemay@mines-paris.org>
# SPDX-FileContributor: Martin Lemay

"""Common I/O helpers."""

from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

# pandas >= 3.1 parses floats with correct rounding by default and deprecates
# the `float_precision` argument; older versions need "round_trip" for that.
_PANDAS_CORRECT_FLOAT_PARSING: bool = tuple(
    int(part) for part in pd.__version__.split(".")[:2]
) >= (3, 1)


def read_csv(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Read a csv file with correctly rounded float parsing.

    Floats are parsed identically whatever the pandas version, so that
    results do not depend on it.

    Args:
        path (str | Path): Path to the csv file.
        **kwargs (Any): Keyword arguments passed to pandas.read_csv.

    Returns:
        pd.DataFrame: Loaded data.
    """
    if not _PANDAS_CORRECT_FLOAT_PARSING:
        kwargs.setdefault("float_precision", "round_trip")
    return pd.read_csv(str(path), **kwargs)


class CenterlineIOFormat(StrEnum):
    """Supported centerline file formats.

    Attributes:
        CSV: Comma-separated values file.
        FLUMY_CSV: Flumy-specific CSV file.
        KML: Keyhole Markup Language file.
    """

    CSV = "csv"
    FLUMY_CSV = "flumy_csv"
    KML = "kml"


def resolve_path(*, base_dir: Path | None, raw_url: str, ctx: str) -> Path:
    """Resolve an absolute or base_dir-relative path.

    - Absolute paths are accepted even when base_dir is None.
    - Relative paths are resolved against base_dir when provided, otherwise
      against the current working directory.

    Args:
        base_dir (Path | None): Base directory to resolve relative paths.
        raw_url (str): Raw file path or URL.
        ctx (str): Context string for error messages.

    Returns:
        Path: Resolved absolute path.
    """
    p = Path(raw_url).expanduser()
    if p.is_absolute():
        path = p
    else:
        root = base_dir if base_dir is not None else Path.cwd()
        path = (root / p).resolve()

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return path


def normalize_age(age: float | int | str | np.number) -> int:
    """Convert a centerline age read from a file to an integer.

    Ages are used to build bend unique ids, hence they must be integers.
    Integer-valued ages (e.g. ``10`` or ``10.0``) are cast to ``int``.

    Args:
        age (float | int | str | np.number): age value (number or
            numeric string).

    Returns:
        int: age as an integer.

    Raises:
        ValueError: If the age is not a finite integer value (e.g. 10.5).
    """
    try:
        value = float(age)
    except (TypeError, ValueError) as err:
        raise ValueError(f"Centerline age '{age}' is not a number.") from err
    if not np.isfinite(value) or not value.is_integer():
        raise ValueError(
            f"Centerline age {age} is not an integer value. Ages must be "
            "integers since they are used to build bend unique ids; rescale "
            "ages (e.g. multiply by 10) or renumber centerlines before "
            "loading."
        )
    return int(value)
