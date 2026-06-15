"""Exception hierarchy for extrusion errors."""

from __future__ import annotations

from meshing_utils.exceptions import MeshingUtilsError


class ExtrusionError(MeshingUtilsError):
    """Base class for all extrusion-related errors."""


class NonCoplanarVerticesError(ExtrusionError):
    """Raised when the marked vertices are not coplanar."""


class NoMarkersFoundError(ExtrusionError):
    """Raised when no //* markers are found in the source file."""


class AmbiguousFaceError(ExtrusionError):
    """Raised when a block has more or fewer than exactly one marked face."""
