"""Block-face geometry: the :class:`BlockFace` model and pure-Python helpers.

Holds the dataclass returned by :func:`extract_block_faces` plus the
Newell polygon-normal helper. No OCC dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from meshing_utils.geometry.hex_topology import HEX_FACE_INDICES, HEX_FACE_NAMES


@dataclass
class BlockFace:
    """One face of a hex block in a blockMeshDict.

    ``face_index`` is into ``HEX_FACE_INDICES`` (0-5),
    ``vertex_names`` lists the 4 vertices in OpenFOAM winding order,
    ``support_points`` are additional sample points from arc/BSpline edges
    of the 4 face edges (used to verify curved-face matches).
    """

    block_name: str
    face_index: int
    face_name: str
    vertex_names: list[str]
    vertex_coords: list[tuple[float, float, float]]
    support_points: list[tuple[float, float, float]] = field(default_factory=list)


def compute_outward_normal(
    coords: list[tuple[float, float, float]],
) -> tuple[float, float, float]:
    """Compute the polygon normal using the Newell method.

    Numerically stable for non-planar polygons. Returns an un-normalised
    vector; a zero vector indicates a degenerate polygon.
    """
    n = len(coords)
    nx = ny = nz = 0.0
    for i in range(n):
        c = coords[i]
        d = coords[(i + 1) % n]
        nx += (c[1] - d[1]) * (c[2] + d[2])
        ny += (c[2] - d[2]) * (c[0] + d[0])
        nz += (c[0] - d[0]) * (c[1] + d[1])
    return (nx, ny, nz)


def extract_block_faces(block, bmd) -> list[BlockFace]:
    """Extract the 6 hex faces for *block* from *bmd*.

    Also collects support points from arc/BSpline edges incident on each
    face. Line edges contribute no support points. Returns an empty list
    when *block* has fewer than 8 named vertices.
    """
    if len(block.vertices) < 8:
        return []

    name_to_coord: dict[str, tuple[float, float, float]] = {}
    for v in bmd.vertices:
        name_to_coord[v.name] = (v.coords[0], v.coords[1], v.coords[2])

    edge_support: dict[frozenset[str], list[tuple[float, float, float]]] = {}
    for e in bmd.edges:
        key: frozenset[str] = frozenset({e.v_start, e.v_end})
        pts: list[tuple[float, float, float]] = [
            (p[0], p[1], p[2]) for p in e.points
        ]
        edge_support[key] = pts

    vertex_names = block.vertices

    faces: list[BlockFace] = []
    for fi, local_indices in enumerate(HEX_FACE_INDICES):
        face_vnames = [vertex_names[k] for k in local_indices]
        face_vcoords: list[tuple[float, float, float]] = []
        for vn in face_vnames:
            coord = name_to_coord.get(vn)
            if coord is None:
                face_vcoords.append((0.0, 0.0, 0.0))
            else:
                face_vcoords.append(coord)

        support: list[tuple[float, float, float]] = []
        n = len(face_vnames)
        for j in range(n):
            edge_key: frozenset[str] = frozenset({face_vnames[j], face_vnames[(j + 1) % n]})
            pts = edge_support.get(edge_key)
            if pts:
                support.extend(pts)

        faces.append(BlockFace(
            block_name=block.name,
            face_index=fi,
            face_name=HEX_FACE_NAMES[fi],
            vertex_names=face_vnames,
            vertex_coords=face_vcoords,
            support_points=support,
        ))

    return faces
