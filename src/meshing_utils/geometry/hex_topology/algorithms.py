"""Validation, ordering, face-convention, and outward-normal algorithms."""

from __future__ import annotations

import math

from meshing_utils.geometry.hex_topology.core import (
    EPSILON_ABS,
    EPSILON_REL,
    HEX_FACE_INDICES,
    HEX_FACE_NAMES,
    HexCandidate,
    HexValidationError,
    OrderingConsistencyError,
    PointPool,
    _face_centroid,
    _face_normal,
    _solid_centroid,
)
from meshing_utils.geometry.vectors import cross as _cross
from meshing_utils.geometry.vectors import dot as _dot
from meshing_utils.geometry.vectors import sub as _sub


def validate_hex(candidate: HexCandidate) -> None:
    """Validate the structural integrity of *candidate*.

    Enforces: 8 vertices, 12 edges, 6 faces of 4 vertices each, vertex
    valence == 3, every edge in exactly 2 faces, every face shares an
    edge with exactly 4 other faces. Raises ``HexValidationError`` with
    a descriptive message including ``candidate.label`` if set.
    """
    label_str = f" (label={candidate.label!r})" if candidate.label else ""

    if len(candidate.vertex_indices) != 8:
        raise HexValidationError(
            f"Hex candidate{label_str} must have exactly 8 vertices, "
            f"got {len(candidate.vertex_indices)}."
        )

    if len(candidate.edges) != 12:
        raise HexValidationError(
            f"Hex candidate{label_str} must have exactly 12 edges, "
            f"got {len(candidate.edges)}."
        )

    if len(candidate.faces) != 6:
        raise HexValidationError(
            f"Hex candidate{label_str} must have exactly 6 faces, "
            f"got {len(candidate.faces)}."
        )
    for i, face in enumerate(candidate.faces):
        if len(face) != 4:
            raise HexValidationError(
                f"Hex candidate{label_str}: face {i} must have exactly 4 vertices, "
                f"got {len(face)}."
            )

    valence: dict[int, int] = {}
    for v in candidate.vertex_indices:
        valence[v] = 0
    for a, b in candidate.edges:
        valence[a] = valence.get(a, 0) + 1
        valence[b] = valence.get(b, 0) + 1
    for v, deg in valence.items():
        if deg != 3:
            raise HexValidationError(
                f"Hex candidate{label_str}: vertex {v} is endpoint of {deg} edges, "
                f"expected exactly 3 edges."
            )

    edge_set: dict[frozenset[int], int] = {}
    for i, (a, b) in enumerate(candidate.edges):
        edge_set[frozenset({a, b})] = i

    edge_face_count: dict[frozenset[int], int] = {k: 0 for k in edge_set}
    for face in candidate.faces:
        n = len(face)
        for j in range(n):
            key = frozenset({face[j], face[(j + 1) % n]})
            if key in edge_face_count:
                edge_face_count[key] += 1
            else:
                edge_face_count[key] = 1
    for key, cnt in edge_face_count.items():
        if cnt != 2:
            raise HexValidationError(
                f"Hex candidate{label_str}: edge {tuple(key)} belongs to {cnt} faces, "
                f"expected exactly 2 faces."
            )

    face_edges: list[frozenset[frozenset[int]]] = []
    for face in candidate.faces:
        n = len(face)
        fe = frozenset(
            frozenset({face[j], face[(j + 1) % n]}) for j in range(n)
        )
        face_edges.append(fe)

    n_faces = len(candidate.faces)
    for i in range(n_faces):
        shared_with = 0
        for j in range(n_faces):
            if i == j:
                continue
            common = face_edges[i] & face_edges[j]
            if len(common) >= 1:
                shared_with += 1
        if shared_with != 4:
            raise HexValidationError(
                f"Hex candidate{label_str}: face {i} shares an edge with {shared_with} "
                f"other faces, expected exactly 4."
            )


def order_hex_vertices(candidate: HexCandidate, pool: PointPool) -> list[int]:
    """Return a canonical OpenFOAM-style vertex ordering for *candidate*.

    1. Select the bottom face (outward normal most opposed to +z).
    2. Identify the top face (opposite to bottom).
    3. Pick v0 = bottom vertex with lex-min ``(z, y, x)``.
    4. Pick v1/v3 by ``Δx + Δy - |Δz|`` heuristic; v2 = remainder.
    5. Pick v4..v7 = vertical pendants of v0..v3 on the top face.
    """
    solid_centroid = _solid_centroid(candidate, pool)

    def _face_score(face):
        n = _face_normal(face, pool)
        length = math.sqrt(n[0] ** 2 + n[1] ** 2 + n[2] ** 2)
        if length < 1e-15:
            unit_n = (0.0, 0.0, 0.0)
        else:
            unit_n = (n[0] / length, n[1] / length, n[2] / length)

        centroid = _face_centroid(face, pool)
        to_centroid = _sub(centroid, solid_centroid)
        if _dot(unit_n, to_centroid) < 0:
            unit_n = (-unit_n[0], -unit_n[1], -unit_n[2])

        dot_neg_z = -unit_n[2]
        fc = centroid
        return (dot_neg_z, -fc[2], -fc[1], -fc[0])

    bottom_face = max(candidate.faces, key=_face_score)
    bottom_set = set(bottom_face)

    top_face = None
    for face in candidate.faces:
        if not set(face) & bottom_set:
            top_face = face
            break

    if top_face is None:
        raise HexValidationError("Could not identify top face opposite to bottom face.")

    bottom_list = list(bottom_face)
    bottom_adj: dict[int, list[int]] = {v: [] for v in bottom_list}
    for a, b in candidate.edges:
        if a in bottom_adj and b in bottom_adj:
            bottom_adj[a].append(b)
            bottom_adj[b].append(a)

    v0 = min(
        bottom_list,
        key=lambda v: (pool.coord_at(v)[2], pool.coord_at(v)[1], pool.coord_at(v)[0]),
    )

    neighbours_v0 = bottom_adj[v0]
    c0 = pool.coord_at(v0)

    def _v1_score(v):
        c = pool.coord_at(v)
        dx = c[0] - c0[0]
        dy = c[1] - c0[1]
        dz = c[2] - c0[2]
        return dx + dy - abs(dz)

    v1 = max(neighbours_v0, key=_v1_score)
    v3 = next(v for v in neighbours_v0 if v != v1)
    v2 = next(v for v in bottom_list if v not in {v0, v1, v3})

    top_set = set(top_face)

    def _top_pendant(v_bottom):
        for a, b in candidate.edges:
            if a == v_bottom and b in top_set:
                return b
            if b == v_bottom and a in top_set:
                return a
        raise HexValidationError(f"No vertical edge found for bottom vertex {v_bottom}.")

    v4 = _top_pendant(v0)
    v5 = _top_pendant(v1)
    v6 = _top_pendant(v2)
    v7 = _top_pendant(v3)

    return [v0, v1, v2, v3, v4, v5, v6, v7]


def ensure_right_handed(ordering: list[int], pool: PointPool) -> list[int]:
    """Ensure the vertex ordering produces a right-handed (positive) hex.

    Computes the scalar triple product
    ``((v1-v0) x (v3-v0)) . (v4-v0)`` and, when ≤ 0, swaps v1↔v3 and
    v5↔v7 to correct handedness. Returns the (possibly corrected)
    ordering.
    """
    v = [pool.coord_at(i) for i in ordering]
    e1 = _sub(v[1], v[0])
    e3 = _sub(v[3], v[0])
    e4 = _sub(v[4], v[0])
    cross = _cross(e1, e3)
    triple = _dot(cross, e4)

    if triple <= 0:
        result = list(ordering)
        result[1], result[3] = result[3], result[1]
        result[5], result[7] = result[7], result[5]
        return result
    return list(ordering)


def enforce_openfoam_face_convention(
    ordering: list[int],
    pool: PointPool,
) -> list[int]:
    """Fix the bottom-face winding to satisfy the OpenFOAM outward convention.

    Prerequisite: *ordering* is already right-handed. After this call
    the kmin face normal (cross product via indices 0,3,2,1) points
    away from the block centroid. Raises ``OrderingConsistencyError``
    when the hex is geometrically degenerate.
    """
    p = [pool.coord_at(ordering[i]) for i in range(8)]

    c_bottom = (
        (p[0][0]+p[1][0]+p[2][0]+p[3][0]) / 4.0,
        (p[0][1]+p[1][1]+p[2][1]+p[3][1]) / 4.0,
        (p[0][2]+p[1][2]+p[2][2]+p[3][2]) / 4.0,
    )
    c_top = (
        (p[4][0]+p[5][0]+p[6][0]+p[7][0]) / 4.0,
        (p[4][1]+p[5][1]+p[6][1]+p[7][1]) / 4.0,
        (p[4][2]+p[5][2]+p[6][2]+p[7][2]) / 4.0,
    )
    dir_to_top = _sub(c_top, c_bottom)

    n_bottom = _cross(_sub(p[3], p[0]), _sub(p[2], p[0]))

    max_dist_sq = 0.0
    for i in range(8):
        for j in range(i + 1, 8):
            dx = p[i][0] - p[j][0]
            dy = p[i][1] - p[j][1]
            dz = p[i][2] - p[j][2]
            d2 = dx*dx + dy*dy + dz*dz
            if d2 > max_dist_sq:
                max_dist_sq = d2
    L = math.sqrt(max_dist_sq)

    eps = max(EPSILON_ABS, EPSILON_REL * L ** 3)
    d = _dot(n_bottom, dir_to_top)

    if abs(d) < eps:
        raise OrderingConsistencyError(
            f"Degenerate hex: bottom-face normal is perpendicular to the "
            f"top-bottom axis (dot={d:.3e}, eps={eps:.3e}). "
            "The block may be coplanar or collapsed."
        )

    if d < 0:
        return list(ordering)

    fixed = list(ordering)
    fixed[1], fixed[3] = fixed[3], fixed[1]
    fixed[5], fixed[7] = fixed[7], fixed[5]
    return fixed


def assert_block_face_normals_outward(
    ordering: list[int],
    pool: PointPool,
    block_label: str = "",
) -> None:
    """Verify that all 6 faces of the hex block have outward-pointing normals.

    Uses ``HEX_FACE_INDICES`` to define the expected local vertex
    winding for each face. Raises ``OrderingConsistencyError`` if any
    face normal points inward.
    """
    p = [pool.coord_at(ordering[i]) for i in range(8)]
    block_centroid = (
        sum(q[0] for q in p) / 8.0,
        sum(q[1] for q in p) / 8.0,
        sum(q[2] for q in p) / 8.0,
    )

    max_dist_sq = 0.0
    for i in range(8):
        for j in range(i + 1, 8):
            dx = p[i][0] - p[j][0]
            dy = p[i][1] - p[j][1]
            dz = p[i][2] - p[j][2]
            max_dist_sq = max(max_dist_sq, dx*dx + dy*dy + dz*dz)
    L = math.sqrt(max_dist_sq)
    eps = max(EPSILON_ABS, EPSILON_REL * L ** 3)

    label_str = f" {block_label!r}" if block_label else ""
    for face_indices, face_name in zip(HEX_FACE_INDICES, HEX_FACE_NAMES, strict=False):
        _i0, _i1, _i2, _i3 = face_indices
        fp = [p[k] for k in face_indices]
        n = _cross(_sub(fp[1], fp[0]), _sub(fp[2], fp[0]))
        fc = (
            (fp[0][0]+fp[1][0]+fp[2][0]+fp[3][0]) / 4.0,
            (fp[0][1]+fp[1][1]+fp[2][1]+fp[3][1]) / 4.0,
            (fp[0][2]+fp[1][2]+fp[2][2]+fp[3][2]) / 4.0,
        )
        outward = _sub(fc, block_centroid)
        d = _dot(n, outward)
        if d <= eps:
            raise OrderingConsistencyError(
                f"Block{label_str} face {face_name!r} points inwards "
                f"(dot={d:.6e}, eps={eps:.3e})."
            )


def assert_hex_outward_from_coords(
    coords: list[tuple[float, float, float]],
    block_label: str = "",
) -> None:
    """Verify outward face normals for a hex given as 8 raw coordinates.

    Used by ``operations.extrusion`` where a ``PointPool`` is not
    available at the point of block construction. Raises
    ``OrderingConsistencyError`` if any face normal points inward.
    """
    p = list(coords)
    block_centroid = (
        sum(q[0] for q in p) / 8.0,
        sum(q[1] for q in p) / 8.0,
        sum(q[2] for q in p) / 8.0,
    )

    max_dist_sq = 0.0
    for i in range(8):
        for j in range(i + 1, 8):
            dx = p[i][0] - p[j][0]
            dy = p[i][1] - p[j][1]
            dz = p[i][2] - p[j][2]
            max_dist_sq = max(max_dist_sq, dx*dx + dy*dy + dz*dz)
    L = math.sqrt(max_dist_sq)
    eps = max(EPSILON_ABS, EPSILON_REL * L ** 3)

    label_str = f" {block_label!r}" if block_label else ""
    for face_indices, face_name in zip(HEX_FACE_INDICES, HEX_FACE_NAMES, strict=False):
        fp = [p[k] for k in face_indices]
        n = _cross(_sub(fp[1], fp[0]), _sub(fp[2], fp[0]))
        fc = (
            (fp[0][0]+fp[1][0]+fp[2][0]+fp[3][0]) / 4.0,
            (fp[0][1]+fp[1][1]+fp[2][1]+fp[3][1]) / 4.0,
            (fp[0][2]+fp[1][2]+fp[2][2]+fp[3][2]) / 4.0,
        )
        outward = _sub(fc, block_centroid)
        d = _dot(n, outward)
        if d <= eps:
            raise OrderingConsistencyError(
                f"Block{label_str} face {face_name!r} points inwards "
                f"(dot={d:.6e}, eps={eps:.3e})."
            )


def _cyclic_variants(seq: tuple) -> list[tuple]:
    """Return all 4 cyclic rotations of a 4-element sequence."""
    n = len(seq)
    return [tuple(seq[(i + k) % n] for i in range(n)) for k in range(n)]


def _reversed_cyclic_variants(seq: tuple) -> list[tuple]:
    """Return all 4 cyclic rotations of the reversed sequence."""
    rev = tuple(reversed(seq))
    return _cyclic_variants(rev)


def check_global_face_consistency(
    orderings: list[list[int]],
    faces_per_block: list[list[tuple]],
) -> None:
    """Check that adjacent blocks use compatible face orientations.

    For every pair of blocks that share a 4-vertex face, the cyclic
    vertex order in block A must equal either a cyclic rotation, or a
    reversed-cyclic rotation, of the order in block B (OpenFOAM
    requires adjacent faces to be traversed in opposite directions).
    Raises ``OrderingConsistencyError`` on incompatible orientation.
    """
    n_blocks = len(orderings)

    block_face_tuples: list[dict[frozenset[int], tuple]] = []
    for block_faces in faces_per_block:
        mapping: dict[frozenset[int], tuple] = {}
        for face in block_faces:
            key = frozenset(face)
            mapping[key] = tuple(face)
        block_face_tuples.append(mapping)

    for i in range(n_blocks):
        for j in range(i + 1, n_blocks):
            shared_keys = set(block_face_tuples[i].keys()) & set(block_face_tuples[j].keys())
            for key in shared_keys:
                face_i = block_face_tuples[i][key]
                face_j = block_face_tuples[j][key]

                fwd_i = set(_cyclic_variants(face_i))
                rev_i = set(_reversed_cyclic_variants(face_i))

                face_j_tuple = tuple(face_j)
                if face_j_tuple not in fwd_i and face_j_tuple not in rev_i:
                    raise OrderingConsistencyError(
                        f"Blocks {i} and {j} share face {tuple(key)} but have "
                        f"incompatible cyclic orientations: "
                        f"block {i} uses {face_i}, block {j} uses {face_j}."
                    )
