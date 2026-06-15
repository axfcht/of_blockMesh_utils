"""OCC-backed face-matching: distance queries, voting, dominant-face selection.

Contains all the heavy OCC helpers used to project block faces onto STEP
solid surfaces and pick a unique winning ``TopoDS_Face`` per block face.
"""

from __future__ import annotations

import logging
import math

from meshing_utils.cad.face_matching.geometry import BlockFace

logger = logging.getLogger(__name__)


def _centroid(
    coords: list[tuple[float, float, float]],
) -> tuple[float, float, float]:
    """Compute the arithmetic centroid of a list of points."""
    n = len(coords)
    if n == 0:
        return (0.0, 0.0, 0.0)
    return (
        sum(c[0] for c in coords) / n,
        sum(c[1] for c in coords) / n,
        sum(c[2] for c in coords) / n,
    )


def _try_get_face(support_shape) -> object | None:
    """Try to extract a ``TopoDS_Face`` from an ExtremaDistShapeShape support shape."""
    try:
        from OCP.TopAbs import TopAbs_FACE
        from OCP.TopoDS import TopoDS
        if support_shape.ShapeType() == TopAbs_FACE:
            return TopoDS.Face_s(support_shape)
    except Exception:
        pass
    return None


def _build_face_ancestor_maps(shell) -> tuple:
    """Build edge->faces and vertex->faces ancestor maps for *shell*.

    Returns ``(edge_map, vertex_map)``, two
    ``TopTools_IndexedDataMapOfShapeListOfShape`` instances mapping every
    edge / vertex of *shell* to the faces incident on it. Used by
    :func:`_recover_faces` to recover the adjacent faces when a distance
    query lands exactly on a shared boundary edge or vertex.
    """
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_VERTEX
    from OCP.TopExp import TopExp
    from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape

    edge_map = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(shell, TopAbs_EDGE, TopAbs_FACE, edge_map)
    vertex_map = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(shell, TopAbs_VERTEX, TopAbs_FACE, vertex_map)
    return edge_map, vertex_map


def _recover_faces(support_shape, edge_map, vertex_map) -> list:
    """Return the candidate ``TopoDS_Face`` list for a distance support shape.

    ``BRepExtrema_DistShapeShape`` reports the nearest support sub-shape on
    the shell. When that sub-shape is a ``FACE`` it is the single candidate.
    When the query point lies exactly on a shared ``EDGE`` or ``VERTEX``
    between adjacent faces — which happens whenever the STEP patch
    tessellation aligns with the block-face boundaries — OCC returns that
    edge / vertex instead of either incident face. In that case all incident
    faces from the ancestor maps are returned (deduplicated) so the
    downstream majority vote and normal check can disambiguate. Returns an
    empty list when no face is recoverable.
    """
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_VERTEX
    from OCP.TopoDS import TopoDS

    stype = support_shape.ShapeType()
    if stype == TopAbs_FACE:
        return [TopoDS.Face_s(support_shape)]

    if stype == TopAbs_EDGE:
        ancestor_map = edge_map
    elif stype == TopAbs_VERTEX:
        ancestor_map = vertex_map
    else:
        return []

    if ancestor_map is None or not ancestor_map.Contains(support_shape):
        return []

    faces: list = []
    seen: set[int] = set()
    for shape in ancestor_map.FindFromKey(support_shape):
        if shape.ShapeType() == TopAbs_FACE:
            face = TopoDS.Face_s(shape)
            face_hash = hash(face)
            if face_hash not in seen:
                seen.add(face_hash)
                faces.append(face)
    return faces


def _majority_vote_face(faces_per_point: list[list[object]]) -> object | None:
    """Pick a face by weighted majority vote across per-point face buckets.

    Each sample point contributes 1.0 total vote, distributed evenly
    across the faces in its bucket. Tie-breaking is deterministic via
    ``hash`` (smaller hash wins). Returns ``None`` when no bucket
    contains any face.
    """
    votes: dict[int, list] = {}
    for bucket in faces_per_point:
        if not bucket:
            continue
        weight = 1.0 / len(bucket)
        for face in bucket:
            face_hash = hash(face)
            if face_hash not in votes:
                votes[face_hash] = [face, 0.0]
            votes[face_hash][1] += weight

    if not votes:
        return None

    best_hash = max(votes.keys(), key=lambda h: (votes[h][1], -h))
    return votes[best_hash][0]


def pick_adjacent_face(candidates: list) -> object:
    """Deterministically pick one face from a list of candidate faces.

    When a point lies exactly on an edge shared by two faces the distance
    solver may return multiple solutions. This function picks the face
    with the smallest hash value (stable within a single Python session).
    """
    return min(candidates, key=lambda f: hash(f))


def nearest_face_within_tol(
    block_face: BlockFace,
    solid,
    tol: float,
) -> object | None:
    """Test whether all sample points of *block_face* lie on the outer shell of *solid*.

    Uses ``BRepExtrema_DistShapeShape`` to compute the distance from each
    vertex and support point to the outer shell of *solid*. If every
    point is within *tol*, returns the closest ``TopoDS_Face`` found for
    the centroid (deterministic tie-breaking via :func:`pick_adjacent_face`).
    Returns ``None`` when any point is further than *tol* from the shell.
    """
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
    from OCP.BRepExtrema import BRepExtrema_DistShapeShape
    from OCP.gp import gp_Pnt
    from OCP.TopAbs import TopAbs_SHELL
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    shell_exp = TopExp_Explorer(solid, TopAbs_SHELL)
    outer_shell = None
    while shell_exp.More():
        outer_shell = TopoDS.Shell_s(shell_exp.Current())
        break

    if outer_shell is None:
        outer_shell = solid

    all_points: list[tuple[float, float, float]] = (
        list(block_face.vertex_coords) + list(block_face.support_points)
    )

    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    bbox = Bnd_Box()
    BRepBndLib.Add_s(solid, bbox)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    pad = tol

    for pt in all_points:
        x, y, z = pt
        if (x < xmin - pad or x > xmax + pad or
                y < ymin - pad or y > ymax + pad or
                z < zmin - pad or z > zmax + pad):
            return None

    edge_map, vertex_map = _build_face_ancestor_maps(outer_shell)

    candidate_face = None
    centroid = _centroid(block_face.vertex_coords)
    faces_per_point: list[list[object]] = []

    for pt in all_points:
        vertex_shape = BRepBuilderAPI_MakeVertex(gp_Pnt(*pt)).Vertex()
        dist_calc = BRepExtrema_DistShapeShape(vertex_shape, outer_shell)
        if not dist_calc.IsDone():
            return None
        dist = dist_calc.Value()
        if dist > tol:
            return None

        bucket: list[object] = []
        seen_bucket: set[int] = set()
        for i in range(1, dist_calc.NbSolution() + 1):
            for face in _recover_faces(
                dist_calc.SupportOnShape2(i), edge_map, vertex_map
            ):
                face_hash = hash(face)
                if face_hash not in seen_bucket:
                    seen_bucket.add(face_hash)
                    bucket.append(face)
        faces_per_point.append(bucket)

    cx, cy, cz = centroid
    centroid_vertex = BRepBuilderAPI_MakeVertex(gp_Pnt(cx, cy, cz)).Vertex()
    dist_calc_centroid = BRepExtrema_DistShapeShape(centroid_vertex, outer_shell)
    if dist_calc_centroid.IsDone():
        candidates: list = []
        seen_centroid: set[int] = set()
        for i in range(1, dist_calc_centroid.NbSolution() + 1):
            for face in _recover_faces(
                dist_calc_centroid.SupportOnShape2(i), edge_map, vertex_map
            ):
                face_hash = hash(face)
                if face_hash not in seen_centroid:
                    seen_centroid.add(face_hash)
                    candidates.append(face)
        if candidates:
            if len(candidates) == 1:
                candidate_face = candidates[0]
            else:
                candidate_face = pick_adjacent_face(candidates)

    if candidate_face is None:
        candidate_face = _majority_vote_face(faces_per_point)
        if candidate_face is not None:
            logger.debug(
                "nearest_face_within_tol: centroid lookup yielded no face "
                "for block face '%s/%s'; resolved via per-point majority vote.",
                block_face.block_name,
                block_face.face_name,
            )

    return candidate_face


def find_dominant_face(
    solid,
    sample_points: list[tuple[float, float, float]],
    support_flags: list[bool],
    tol: float,
) -> tuple[object, list[tuple[float, float, float]]] | None:
    """Determine the dominant ``TopoDS_Face`` on *solid* by majority voting.

    For each sample point a ``BRepExtrema_DistShapeShape`` query is
    performed against the outer shell. Every nearest face receives a
    vote; support points (``support_flags`` True) are weighted four
    times as heavily as vertex points. Tie-breaking is deterministic
    via ``hash``. Returns ``(face, contributing_points)`` or ``None``
    when no face received any votes.
    """
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
    from OCP.BRepExtrema import BRepExtrema_DistShapeShape
    from OCP.gp import gp_Pnt
    from OCP.TopAbs import TopAbs_SHELL
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    shell_exp = TopExp_Explorer(solid, TopAbs_SHELL)
    outer_shell = None
    while shell_exp.More():
        outer_shell = TopoDS.Shell_s(shell_exp.Current())
        break

    if outer_shell is None:
        outer_shell = solid

    edge_map, vertex_map = _build_face_ancestor_maps(outer_shell)

    face_votes: dict[int, list] = {}

    for pt, is_support in zip(sample_points, support_flags, strict=False):
        x, y, z = pt
        vertex_shape = BRepBuilderAPI_MakeVertex(gp_Pnt(x, y, z)).Vertex()
        dist_calc = BRepExtrema_DistShapeShape(vertex_shape, outer_shell)
        if not dist_calc.IsDone():
            continue
        if dist_calc.Value() > tol:
            continue

        weight = 4 if is_support else 1

        n_sol = dist_calc.NbSolution()
        nearest_faces: list[object] = []
        seen_faces: set[int] = set()
        for i in range(1, n_sol + 1):
            support_shape = dist_calc.SupportOnShape2(i)
            for face in _recover_faces(support_shape, edge_map, vertex_map):
                face_hash = hash(face)
                if face_hash not in seen_faces:
                    seen_faces.add(face_hash)
                    nearest_faces.append(face)

        if not nearest_faces:
            continue
        per_face_weight = weight / len(nearest_faces)

        for face in nearest_faces:
            face_hash = hash(face)

            if face_hash not in face_votes:
                face_votes[face_hash] = [face, 0.0, []]
            face_votes[face_hash][1] += per_face_weight
            face_votes[face_hash][2].append(pt)

    if not face_votes:
        return None

    best_hash = max(
        face_votes.keys(),
        key=lambda h: (face_votes[h][1], -h),
    )
    best_face, _score, contributing_points = face_votes[best_hash]
    return best_face, contributing_points


def local_surface_normal(
    face,
    point: tuple[float, float, float],
) -> tuple[float, float, float] | None:
    """Evaluate the outward surface normal of *face* at the given 3D *point*.

    Projects *point* onto the face's surface via
    ``ShapeAnalysis_Surface.ValueOfUV`` and evaluates the first
    derivatives. The face orientation is taken into account so the
    returned normal always points outward from the solid. Returns
    ``None`` on a UV singularity or degenerate surface.
    """
    try:
        from OCP.BRep import BRep_Tool
        from OCP.gp import gp_Pnt, gp_Vec
        from OCP.ShapeAnalysis import ShapeAnalysis_Surface
        from OCP.TopAbs import TopAbs_REVERSED

        surface = BRep_Tool.Surface_s(face)
        if surface is None:
            return None

        sa = ShapeAnalysis_Surface(surface)
        px, py, pz = point
        uv = sa.ValueOfUV(gp_Pnt(px, py, pz), 1e-4)
        u = uv.X()
        v = uv.Y()

        p_on_surf = gp_Pnt()
        du = gp_Vec()
        dv = gp_Vec()
        surface.D1(u, v, p_on_surf, du, dv)

        nx = du.Y() * dv.Z() - du.Z() * dv.Y()
        ny = du.Z() * dv.X() - du.X() * dv.Z()
        nz = du.X() * dv.Y() - du.Y() * dv.X()

        length = math.sqrt(nx * nx + ny * ny + nz * nz)
        if length < 1e-15:
            return None

        nx /= length
        ny /= length
        nz /= length

        if face.Orientation() == TopAbs_REVERSED:
            nx, ny, nz = -nx, -ny, -nz

        return (nx, ny, nz)

    except Exception:
        return None
