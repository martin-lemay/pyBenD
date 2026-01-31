# SPDX-FileCopyrightText: Copyright 2025 Martin Lemay <martin.lemay@mines-paris.org>
# SPDX-FileContributor: Martin Lemay

"""Common I/O helpers."""

from enum import StrEnum
from pathlib import Path


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
