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
    - Relative paths require base_dir.

    Args:
        base_dir (Path | None): Base directory to resolve relative paths.
        raw_url (str): Raw file path or URL.
        ctx (str): Context string for error messages.

    Returns:
        Path: Resolved absolute path.
    """
    p = Path(raw_url)
    if p.is_absolute():
        return p
    if base_dir is None:
        raise ValueError(
            f"Base directory must be provided to resolve relative path for {ctx}."
        )
    path = (base_dir / p).resolve()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return path
