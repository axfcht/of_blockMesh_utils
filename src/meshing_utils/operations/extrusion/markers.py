"""Marker collection, face indexing, face matching, and marker stripping.

Two hex-face conventions
------------------------
``EXTRUSION_LOCAL_FACE_INDICES`` (defined here) and
``geometry.hex_topology.HEX_FACE_INDICES`` are **not** the same constant
even though both describe "the 6 faces of a hex". They use different
vertex windings for different purposes:

``geometry.hex_topology.HEX_FACE_INDICES`` is the **canonical**
project-wide convention. Windings are chosen so that for the standard
OpenFOAM hex layout ``[v0..v7]`` the cyclic traversal of each tuple
produces a normal pointing OUTWARD from the block — used everywhere
the project needs to check or construct face normals on an already-
ordered hex.

``EXTRUSION_LOCAL_FACE_INDICES`` (this module) is **local to extrusion**.
Each face's winding is chosen so that ``(v_a -> v_b) x (v_b -> v_c)``
yields the OUTWARD normal of the *source* block's face. When this
tuple is used as the bottom face of the newly extruded block, the
resulting hex is right-handed with its +x3' axis aligned to the
extrusion direction. Do not import this constant from outside
``operations.extrusion``.
"""

from __future__ import annotations

import warnings

from meshing_utils.foam.dict_file import BlockMeshDict
from meshing_utils.foam.elements import Block, Vertex
from meshing_utils.operations.extrusion.exceptions import AmbiguousFaceError

EXTRUSION_LOCAL_FACE_INDICES: list[tuple[int, int, int, int]] = [
    (0, 3, 2, 1),  # face 0: bottom (-k)  outward = -x3
    (4, 5, 6, 7),  # face 1: top    (+k)  outward = +x3
    (0, 1, 5, 4),  # face 2: front  (-j)  outward = -x2
    (2, 3, 7, 6),  # face 3: back   (+j)  outward = +x2
    (1, 2, 6, 5),  # face 4: right  (+i)  outward = +x1
    (0, 4, 7, 3),  # face 5: left   (-i)  outward = -x1
]

_LOCAL_COORDS: list[tuple[int, int, int]] = [
    (0, 0, 0),  # v0
    (1, 0, 0),  # v1
    (1, 1, 0),  # v2
    (0, 1, 0),  # v3
    (0, 0, 1),  # v4
    (1, 0, 1),  # v5
    (1, 1, 1),  # v6
    (0, 1, 1),  # v7
]


def _source_axis_of_diff(a_idx: int, b_idx: int) -> int:
    """Return the source-block axis (0=i, 1=j, 2=k) along which the segment
    from local vertex ``a_idx`` to ``b_idx`` runs.

    Raises ``ValueError`` when the vertices are not edge-adjacent on the hex.
    """
    a = _LOCAL_COORDS[a_idx]
    b = _LOCAL_COORDS[b_idx]
    diffs = [b[i] - a[i] for i in range(3)]
    nonzero = [i for i, d in enumerate(diffs) if d != 0]
    if len(nonzero) != 1:
        raise ValueError(
            f"Local indices {a_idx} and {b_idx} are not edge-adjacent on the hex."
        )
    return nonzero[0]


def _collect_marked_vertices(bmd: BlockMeshDict) -> list[Vertex]:
    """Return all vertices carrying a //* marker."""
    return [v for v in bmd.vertices if v.marker is not None]


def _collect_marked_blocks(bmd: BlockMeshDict) -> list[Block]:
    """Return all blocks carrying a //* marker."""
    return [b for b in bmd.blocks if b.marker is not None]


def _find_face_index_in_block(block: Block, marked_names: set) -> int:
    """Return the index (0-5) of the unique hex face whose 4 vertices are all
    in ``marked_names``. Raises ``AmbiguousFaceError`` if not exactly one.
    """
    block_verts = block.vertices
    matching: list[int] = []
    for face_idx, local_indices in enumerate(EXTRUSION_LOCAL_FACE_INDICES):
        face_verts = {block_verts[i] for i in local_indices}
        if face_verts <= marked_names:
            matching.append(face_idx)
    if len(matching) != 1:
        raise AmbiguousFaceError(
            f"Block {block.name!r}: expected exactly 1 marked face, "
            f"found {len(matching)}."
        )
    return matching[0]


def _discover_blocks_with_marked_face(
    blocks: list[Block],
    marked_vertex_names: set,
) -> list[tuple[Block, int]]:
    """Find all ``(block, face_idx)`` pairs where all 4 face vertices are marked."""
    result: list[tuple[Block, int]] = []
    for block in blocks:
        matching: list[int] = []
        for face_idx, local_indices in enumerate(EXTRUSION_LOCAL_FACE_INDICES):
            face_names = {block.vertices[i] for i in local_indices}
            if face_names <= marked_vertex_names:
                matching.append(face_idx)
        if len(matching) == 0:
            continue
        if len(matching) > 1:
            warnings.warn(
                f"Block {block.name!r}: {len(matching)} faces fully marked; "
                "all will be extruded.",
                UserWarning,
                stacklevel=3,
            )
        for face_idx in matching:
            result.append((block, face_idx))
    return result


def _strip_all_markers(bmd: BlockMeshDict) -> None:
    """Remove all //* markers from every element in the BlockMeshDict."""
    for v in bmd.vertices:
        v.marker = None
    for e in bmd.edges:
        e.marker = None
    for b in bmd.blocks:
        b.marker = None
    bmd.default_patch.marker = None
    for p in bmd.boundary:
        p.marker = None
        for f in p.faces:
            f.marker = None
