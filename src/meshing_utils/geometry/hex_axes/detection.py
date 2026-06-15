"""Geometry-only hex topology detection from raw 8-vertex coordinate sets.

Holds :class:`TopologyError`, the coplanarity and quad-ordering helpers,
:func:`detect_hex_faces`, and the per-edge axis classification helpers
(:func:`compute_axis_class_per_edge`, :func:`map_class_to_local_axis_index`).
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence

from meshing_utils.exceptions import MeshingUtilsError
from meshing_utils.geometry.vectors import cross as _cross
from meshing_utils.geometry.vectors import dot as _dot
from meshing_utils.geometry.vectors import norm as _norm

# Relative tolerance levels for coplanarity test: scaled by the block diagonal.
# Each level is tried in sequence; the first that yields a valid hex topology
# is used. Higher tolerances handle blocks with arc edges whose vertices
# deviate slightly from perfect planarity.
_TOLERANCE_LEVELS = (1e-6, 1e-4, 1e-2, 1e-1)


class TopologyError(MeshingUtilsError, ValueError):
    """Raised when no valid hex face topology can be detected from coordinates."""


def _dominant_axis(dx: float, dy: float, dz: float) -> int:
    """Return the index (0, 1, 2) of the largest absolute component.

    Tie-breaking: smallest index wins (deterministic for diagonals).
    """
    abs_vals = (abs(dx), abs(dy), abs(dz))
    max_val = max(abs_vals)
    for i, v in enumerate(abs_vals):
        if v == max_val:
            return i
    return 0  # unreachable


def _max_diagonal(coords: Sequence[tuple[float, float, float]]) -> float:
    """Return the maximum pairwise Euclidean distance among the given points."""
    max_dist = 0.0
    pts = list(coords)
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            dx = pts[j][0] - pts[i][0]
            dy = pts[j][1] - pts[i][1]
            dz = pts[j][2] - pts[i][2]
            d = math.sqrt(dx * dx + dy * dy + dz * dz)
            if d > max_dist:
                max_dist = d
    return max_dist


def _is_coplanar(
    pts: list[tuple[float, float, float]],
    tol: float,
) -> bool:
    """Return ``True`` if the four points are coplanar within absolute tolerance *tol*."""
    p0, p1, p2, p3 = pts[0], pts[1], pts[2], pts[3]
    v1 = (p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2])
    v2 = (p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2])
    normal = _cross(v1, v2)
    n_len = _norm(normal)
    if n_len < 1e-15:
        return True
    v3 = (p3[0] - p0[0], p3[1] - p0[1], p3[2] - p0[2])
    dist = abs(_dot(normal, v3)) / n_len
    return dist <= tol


def _order_quad_ccw(
    indices: tuple[int, int, int, int],
    coords: Sequence[tuple[float, float, float]],
) -> tuple[int, int, int, int] | None:
    """Sort four coplanar point indices into CCW order w.r.t. their outward normal.

    Returns the reordered tuple of indices, or ``None`` if the four points
    do not form a convex quadrilateral.
    """
    pts = [coords[i] for i in indices]

    cx = sum(p[0] for p in pts) / 4.0
    cy = sum(p[1] for p in pts) / 4.0
    cz = sum(p[2] for p in pts) / 4.0

    v01 = (pts[1][0] - pts[0][0], pts[1][1] - pts[0][1], pts[1][2] - pts[0][2])
    v02 = (pts[2][0] - pts[0][0], pts[2][1] - pts[0][1], pts[2][2] - pts[0][2])
    normal = _cross(v01, v02)
    n_len = _norm(normal)

    if n_len < 1e-15:
        v03 = (pts[3][0] - pts[0][0], pts[3][1] - pts[0][1], pts[3][2] - pts[0][2])
        normal = _cross(v01, v03)
        n_len = _norm(normal)
    if n_len < 1e-15:
        return None

    normal = (normal[0] / n_len, normal[1] / n_len, normal[2] / n_len)

    u_raw = (pts[1][0] - pts[0][0], pts[1][1] - pts[0][1], pts[1][2] - pts[0][2])
    u_len = _norm(u_raw)
    if u_len < 1e-15:
        return None
    u = (u_raw[0] / u_len, u_raw[1] / u_len, u_raw[2] / u_len)
    v_vec = _cross(normal, u)

    angles: list[tuple[float, int]] = []
    for _local_i, (orig_idx, pt) in enumerate(zip(indices, pts, strict=False)):
        rel = (pt[0] - cx, pt[1] - cy, pt[2] - cz)
        pu = _dot(rel, u)
        pv = _dot(rel, v_vec)
        angle = math.atan2(pv, pu)
        angles.append((angle, orig_idx))

    angles.sort(key=lambda t: (t[0], t[1]))
    ordered = tuple(idx for _, idx in angles)

    ordered_pts = [coords[i] for i in ordered]
    signs = []
    for k in range(4):
        p_cur = ordered_pts[k]
        p_next = ordered_pts[(k + 1) % 4]
        p_next2 = ordered_pts[(k + 2) % 4]
        e1 = (p_next[0] - p_cur[0], p_next[1] - p_cur[1], p_next[2] - p_cur[2])
        e2 = (p_next2[0] - p_next[0], p_next2[1] - p_next[1], p_next2[2] - p_next[2])
        cross_z = _dot(_cross(e1, e2), normal)
        signs.append(cross_z)

    if not all(s >= 0 for s in signs) and not all(s <= 0 for s in signs):
        return None

    return ordered  # type: ignore[return-value]


def _all_points_coplanar(
    coords: Sequence[tuple[float, float, float]],
    tol: float,
) -> bool:
    """Return ``True`` if all points lie in a common plane within *tol*."""
    pts = list(coords)
    n = len(pts)
    if n < 4:
        return True

    normal = None
    p0 = pts[0]
    for i in range(1, n):
        for j in range(i + 1, n):
            v1 = (pts[i][0] - p0[0], pts[i][1] - p0[1], pts[i][2] - p0[2])
            v2 = (pts[j][0] - p0[0], pts[j][1] - p0[1], pts[j][2] - p0[2])
            c = _cross(v1, v2)
            c_len = _norm(c)
            if c_len > 1e-15:
                normal = (c[0] / c_len, c[1] / c_len, c[2] / c_len)
                break
        if normal is not None:
            break

    if normal is None:
        return True

    for pt in pts:
        v = (pt[0] - p0[0], pt[1] - p0[1], pt[2] - p0[2])
        dist = abs(_dot(v, normal))
        if dist > tol:
            return False
    return True


def _validate_hex_topology(faces: list[tuple[int, int, int, int]]) -> bool:
    """Return ``True`` when *faces* (exactly 6) form a valid closed hex topology."""
    if len(faces) != 6:
        return False

    edge_count: dict[frozenset[int], int] = {}
    vertex_count: dict[int, int] = {}

    for face in faces:
        for vi in face:
            vertex_count[vi] = vertex_count.get(vi, 0) + 1
        for k in range(4):
            edge = frozenset({face[k], face[(k + 1) % 4]})
            edge_count[edge] = edge_count.get(edge, 0) + 1

    if len(edge_count) != 12:
        return False
    if any(c != 2 for c in edge_count.values()):
        return False
    if len(vertex_count) != 8:
        return False
    return not any(c != 3 for c in vertex_count.values())


def detect_hex_faces(
    coords: Sequence[tuple[float, float, float]],
) -> list[tuple[int, int, int, int]]:
    """Detect the 6 faces of a hex block from its 8 vertex coordinates.

    Uses geometry exclusively (no assumption about vertex ordering).
    Tries multiple coplanarity tolerance levels; the first that yields a
    valid hex topology is used. Raises :class:`TopologyError` if no
    valid topology can be found at any level.
    """
    max_diag = _max_diagonal(coords)
    if max_diag == 0.0:
        raise TopologyError(
            f"could not detect 6 hex faces from coordinates "
            f"(best attempt found 0 coplanar convex quads); coords={list(coords)}"
        )

    tightest_tol = max(_TOLERANCE_LEVELS[0] * max_diag, 1e-12)
    if _all_points_coplanar(coords, tightest_tol):
        raise TopologyError(
            "could not detect 6 hex faces from coordinates; all 8 points are coplanar"
        )

    best_count = 0

    for tol_rel in _TOLERANCE_LEVELS:
        tol_plane = tol_rel * max_diag
        tol_plane = max(tol_plane, 1e-12)

        candidates: list[tuple[int, int, int, int]] = []
        for combo in itertools.combinations(range(8), 4):
            pts = [coords[i] for i in combo]
            if not _is_coplanar(pts, tol_plane):
                continue
            ordered = _order_quad_ccw(combo, coords)
            if ordered is None:
                continue
            candidates.append(ordered)

        if len(candidates) > best_count:
            best_count = len(candidates)

        if len(candidates) == 6 and _validate_hex_topology(candidates):
            return candidates

        if len(candidates) >= 6:
            for subset in itertools.combinations(candidates, 6):
                if _validate_hex_topology(list(subset)):
                    return list(subset)

    raise TopologyError(
        f"could not detect 6 hex faces from coordinates "
        f"(best attempt found {best_count} coplanar convex quads); coords={list(coords)}"
    )


class _SimpleUnionFind:
    """Minimal Union-Find over frozenset keys."""

    def __init__(self) -> None:
        self._parent: dict[frozenset[int], frozenset[int]] = {}

    def _add(self, x: frozenset[int]) -> None:
        if x not in self._parent:
            self._parent[x] = x

    def find(self, x: frozenset[int]) -> frozenset[int]:
        self._add(x)
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, a: frozenset[int], b: frozenset[int]) -> None:
        self._add(a)
        self._add(b)
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[rb] = ra


def compute_axis_class_per_edge(
    faces: list[tuple[int, int, int, int]],
) -> dict[frozenset[int], frozenset[int]]:
    """Classify the 12 block edges into 3 axis classes via Union-Find.

    For each face two pairs of parallel edges are unioned. Postcondition:
    exactly 3 distinct class roots. Returns a dict mapping every edge
    frozenset to its class root (also a frozenset).
    """
    uf = _SimpleUnionFind()

    for face in faces:
        e0 = frozenset({face[0], face[1]})
        e1 = frozenset({face[1], face[2]})
        e2 = frozenset({face[2], face[3]})
        e3 = frozenset({face[3], face[0]})
        uf.union(e0, e2)
        uf.union(e1, e3)

    all_edges: dict[frozenset[int], frozenset[int]] = {}
    for face in faces:
        for k in range(4):
            edge = frozenset({face[k], face[(k + 1) % 4]})
            all_edges[edge] = uf.find(edge)

    return all_edges


def map_class_to_local_axis_index(
    edge_to_class: dict[frozenset[int], frozenset[int]],
    faces: list[tuple[int, int, int, int]],
) -> dict[frozenset[int], int]:
    """Map each axis class root to a local axis index 0/1/2 (Variant 2).

    Finds the three edges incident to V0, sorts by neighbour vertex
    index, and assigns axis indices 0, 1, 2 in that order.
    """
    v0_edges: list[tuple[int, frozenset[int]]] = []
    for edge in edge_to_class:
        if 0 in edge:
            other = next(v for v in edge if v != 0)
            v0_edges.append((other, edge))

    v0_edges.sort(key=lambda t: t[0])

    class_to_axis: dict[frozenset[int], int] = {}
    for axis_idx, (_, edge) in enumerate(v0_edges):
        root = edge_to_class[edge]
        class_to_axis[root] = axis_idx

    return class_to_axis
