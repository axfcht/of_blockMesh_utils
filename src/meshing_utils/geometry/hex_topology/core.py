"""Core hex-topology types, constants, and geometry helpers.

Holds the data models (``CurveInfo``, ``HexCandidate``), the
:class:`PointPool` snap-to-grid store, the OpenFOAM face-index constants
``HEX_FACE_INDICES`` / ``HEX_FACE_NAMES``, the exceptions, and the
internal vector helpers used by the algorithms in :mod:`.algorithms`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from meshing_utils.exceptions import MeshingUtilsError
from meshing_utils.geometry.vectors import cross as _cross
from meshing_utils.geometry.vectors import sub as _sub


class HexValidationError(MeshingUtilsError):
    """Raised when a HexCandidate fails structural validation."""


class OrderingConsistencyError(MeshingUtilsError):
    """Raised when two adjacent blocks have incompatible face orientations."""


@dataclass
class CurveInfo:
    """Describes the curve type and control points for one block edge."""

    kind: Literal["line", "arc", "bspline"]
    support_points: list[tuple[float, float, float]]
    arc_midpoint: tuple[float, float, float] | None = None
    arc_angle: float = 0.0


@dataclass
class HexCandidate:
    """Unordered topological description of one hex block candidate."""

    vertex_indices: list[int]
    faces: list[tuple[int, ...]]
    edges: list[tuple[int, int]]
    edge_curves: dict[frozenset[int], CurveInfo]
    label: str | None = None


class PointPool:
    """Snap-to-grid point de-duplication store.

    Points are keyed by ``(round(x/tol), round(y/tol), round(z/tol))`` so
    that any two coordinates within *tol* of each other in each dimension
    map to the same integer grid cell and therefore share an index.
    """

    def __init__(self, tol: float) -> None:
        self._tol: float = tol
        self._grid: dict[tuple[int, int, int], int] = {}
        self._coords: list[tuple[float, float, float]] = []

    def _grid_key(self, coord: tuple[float, float, float]) -> tuple[int, int, int]:
        x, y, z = coord
        tol = self._tol
        return (round(x / tol), round(y / tol), round(z / tol))

    def add_or_get(self, coord: tuple[float, float, float]) -> int:
        """Return the index for *coord*, inserting it if new."""
        key = self._grid_key(coord)
        if key in self._grid:
            return self._grid[key]
        idx = len(self._coords)
        self._coords.append(coord)
        self._grid[key] = idx
        return idx

    def coord_at(self, idx: int) -> tuple[float, float, float]:
        """Return the coordinate stored at *idx*."""
        if idx < 0 or idx >= len(self._coords):
            raise IndexError(f"PointPool index {idx} out of range (size={len(self._coords)})")
        return self._coords[idx]

    def __len__(self) -> int:
        return len(self._coords)


EPSILON_ABS: float = 1e-12
EPSILON_REL: float = 1e-9

# OpenFOAM outward face definitions for a standard hex block.
# Each tuple lists the 4 local vertex indices (within the 8-vertex ordering)
# whose cyclic traversal produces an outward-pointing face normal.
HEX_FACE_INDICES: tuple[tuple[int, int, int, int], ...] = (
    (0, 4, 7, 3),  # imin  outward = -x1
    (1, 2, 6, 5),  # imax  outward = +x1
    (0, 1, 5, 4),  # jmin  outward = -x2
    (3, 7, 6, 2),  # jmax  outward = +x2
    (0, 3, 2, 1),  # kmin  outward = -x3
    (4, 5, 6, 7),  # kmax  outward = +x3
)

HEX_FACE_NAMES: tuple[str, ...] = (
    "imin", "imax", "jmin", "jmax", "kmin", "kmax",
)


def _face_normal(face: tuple[int, ...], pool: PointPool) -> tuple[float, float, float]:
    """Compute the face normal via the cross product of the first two edges."""
    p0 = pool.coord_at(face[0])
    p1 = pool.coord_at(face[1])
    p2 = pool.coord_at(face[2])
    return _cross(_sub(p1, p0), _sub(p2, p0))


def _face_centroid(face: tuple[int, ...], pool: PointPool) -> tuple[float, float, float]:
    """Return the arithmetic centroid of a face's vertices."""
    coords = [pool.coord_at(v) for v in face]
    n = len(coords)
    return (
        sum(c[0] for c in coords) / n,
        sum(c[1] for c in coords) / n,
        sum(c[2] for c in coords) / n,
    )


def _solid_centroid(candidate: HexCandidate, pool: PointPool) -> tuple[float, float, float]:
    """Return the arithmetic centroid of a hex's 8 vertices."""
    coords = [pool.coord_at(v) for v in candidate.vertex_indices]
    n = len(coords)
    return (
        sum(c[0] for c in coords) / n,
        sum(c[1] for c in coords) / n,
        sum(c[2] for c in coords) / n,
    )
