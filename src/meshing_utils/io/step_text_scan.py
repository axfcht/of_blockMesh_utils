"""STEP file text scanner for BREP-solid entity name extraction.

Provides pure-Python, OCC-independent parsing of STEP (ISO 10303-21) files
to extract solid geometry entity definitions (MANIFOLD_SOLID_BREP,
BREP_WITH_VOIDS, FACETED_BREP, SHELL_BASED_SURFACE_MODEL) and their names.

This module is intended as a shared utility for all tools in meshing_utils
that need to map OCC-loaded solids back to their STEP file names.

Functions
---------
parse_brep_solid_entries(step_path)
    Parse a STEP file and return all BREP-solid definitions in file order.

build_brep_name_map_by_file_id(step_path)
    Return a mapping {file_id (int): name (str)} for BREP-solid entities.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class BrepSolidEntry:
    """One BREP-solid entity definition parsed from a STEP file.

    Attributes
    ----------
    file_id:
        The integer entity number as written in the STEP file (e.g. ``57``
        for ``#57 = MANIFOLD_SOLID_BREP(...)``).
    type_token:
        The entity type keyword, upper-cased (e.g. ``"MANIFOLD_SOLID_BREP"``).
    name:
        The first string argument of the entity definition, i.e. the solid
        name.  Empty string when the argument was present but empty.
    """

    file_id: int
    type_token: str
    name: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# The four STEP entity type keywords that represent solid BREP geometry.
_BREP_TYPE_PATTERN = re.compile(
    r"(MANIFOLD_SOLID_BREP|BREP_WITH_VOIDS|FACETED_BREP|SHELL_BASED_SURFACE_MODEL)",
    re.IGNORECASE,
)

# Pattern to match a full entity definition line in the DATA section.
# Captures: (1) entity number, (2) type keyword, (3) first string argument.
# Handles optional whitespace around tokens and between keyword and '('.
_ENTRY_PATTERN = re.compile(
    r"#(\d+)\s*=\s*"
    r"(MANIFOLD_SOLID_BREP|BREP_WITH_VOIDS|FACETED_BREP|SHELL_BASED_SURFACE_MODEL)"
    r"\s*\(\s*'([^']*)'",
    re.IGNORECASE,
)


def _extract_data_section(raw: str) -> str:
    """Return the content of the DATA section from a STEP file string.

    The DATA section is delimited by ``DATA;`` and ``ENDSEC;``.  If no
    DATA section is found the full raw string is returned as a fallback so
    that callers can still attempt to parse whatever is present.

    Parameters
    ----------
    raw:
        Full content of a STEP file as a string.

    Returns
    -------
    str
        Content between ``DATA;`` and ``ENDSEC;`` (exclusive), or *raw*
        when the delimiters are not found.
    """
    # Locate DATA; marker (case-insensitive, allow surrounding whitespace)
    data_match = re.search(r"\bDATA\s*;", raw, re.IGNORECASE)
    if data_match is None:
        return raw

    data_start = data_match.end()

    # Locate ENDSEC; after the DATA; marker
    endsec_match = re.search(r"\bENDSEC\s*;", raw[data_start:], re.IGNORECASE)
    if endsec_match is None:
        return raw[data_start:]

    return raw[data_start : data_start + endsec_match.start()]


def _normalise_whitespace(text: str) -> str:
    """Collapse all runs of whitespace (including newlines) to single spaces.

    This converts multi-line entity definitions into single logical lines so
    that a simple line-oriented regex can match them reliably.

    Parameters
    ----------
    text:
        Input text, possibly spanning multiple lines.

    Returns
    -------
    str
        Text with all whitespace runs replaced by a single space.
    """
    return re.sub(r"\s+", " ", text)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_brep_solid_entries(step_path: Path) -> list[BrepSolidEntry]:
    """Parse a STEP file and return all BREP-solid definitions in file order.

    Reads the DATA section of the STEP file and extracts all entity
    definitions for the four solid geometry types:

    * ``MANIFOLD_SOLID_BREP``
    * ``BREP_WITH_VOIDS``
    * ``FACETED_BREP``
    * ``SHELL_BASED_SURFACE_MODEL``

    Multi-line entity definitions (whitespace or newlines within the
    definition) are handled by normalising whitespace before matching.

    Any read or parse error silently returns an empty list rather than
    raising.

    Parameters
    ----------
    step_path:
        Path to the STEP file.

    Returns
    -------
    list[BrepSolidEntry]
        All matched BREP-solid definitions in the order they appear in the
        file.  Entries with empty names are included; callers decide whether
        to filter them.
    """
    try:
        raw = step_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []

    data_section = _extract_data_section(raw)
    normalised = _normalise_whitespace(data_section)

    entries: list[BrepSolidEntry] = []
    for match in _ENTRY_PATTERN.finditer(normalised):
        file_id = int(match.group(1))
        type_token = match.group(2).upper()
        name = match.group(3)
        entries.append(BrepSolidEntry(file_id=file_id, type_token=type_token, name=name))

    return entries


def build_brep_name_map_by_file_id(step_path: Path) -> dict[int, str]:
    """Return a mapping ``{file_id (int): name (str)}`` for BREP-solid entities.

    Convenience wrapper around :func:`parse_brep_solid_entries` that filters
    out entries with empty names and returns a plain dict keyed by the integer
    STEP file entity number.

    Parameters
    ----------
    step_path:
        Path to the STEP file.

    Returns
    -------
    dict[int, str]
        Mapping from entity number to name.  Only entries with non-empty
        names are included.  Returns an empty dict when the file cannot be
        read or contains no matching entities.
    """
    entries = parse_brep_solid_entries(step_path)
    return {e.file_id: e.name for e in entries if e.name}
