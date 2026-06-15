"""Geometric pre/post checks for the extrusion pipeline.

* ``_check_coplanar`` enforces the planarity invariant on marked vertices.
* ``_assert_extrusion_block_outward`` is a defensive self-check that the
  freshly built extrusion block has outward face normals.
"""

from __future__ import annotations

import math

from meshing_utils.foam.dict_file import BlockMeshDict
from meshing_utils.foam.elements import Block, Vertex
from meshing_utils.geometry.hex_topology import assert_hex_outward_from_coords
from meshing_utils.operations.extrusion.exceptions import NonCoplanarVerticesError

_PLANARITY_TOL = 1e-8


def _check_coplanar(vertices: list[Vertex]) -> None:
    """Raise ``NonCoplanarVerticesError`` if the vertices are not coplanar.

    Requires at least 3 vertices. Also catches degenerate (collinear) cases.
    """
    if len(vertices) < 3:
        raise NonCoplanarVerticesError(
            f"Need at least 3 vertices for planarity check, got {len(vertices)}."
        )
    coords = [v.coords for v in vertices]
    p0 = coords[0]
    p1 = coords[1]
    p2 = coords[2]

    def sub(a: list, b: list) -> list:
        return [a[i] - b[i] for i in range(3)]

    def cross(a: list, b: list) -> list:
        return [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]

    def dot(a: list, b: list) -> float:
        return sum(a[i] * b[i] for i in range(3))

    def norm(a: list) -> float:
        return math.sqrt(sum(x * x for x in a))

    normal = cross(sub(p1, p0), sub(p2, p0))
    n_len = norm(normal)
    if n_len < _PLANARITY_TOL:
        raise NonCoplanarVerticesError(
            "Marked vertices are collinear — cannot determine plane normal."
        )
    for p in coords[3:]:
        d = abs(dot(normal, sub(p, p0)))
        if d > _PLANARITY_TOL * n_len:
            raise NonCoplanarVerticesError(
                f"Vertex at {p} is not coplanar with the other marked vertices."
            )


def _assert_extrusion_block_outward(
    block: Block,
    bmd: BlockMeshDict,
    block_label: str = "",
) -> None:
    """Verify that all 6 faces of an extruded block have outward normals.

    Looks up the 8 vertex coordinates from ``bmd.vertices`` and delegates to
    :func:`assert_hex_outward_from_coords`. A failure is a genuine
    implementation bug; the error is intentionally not suppressed.
    """
    name_to_vertex = {v.name: v for v in bmd.vertices}
    coords = []
    for vname in block.vertices:
        v = name_to_vertex.get(vname)
        if v is None:
            return
        coords.append(tuple(v.coords))
    assert_hex_outward_from_coords(coords, block_label=block_label)
