"""STEP file loading: solid traversal, hex topology filter, ``load_step``."""

from __future__ import annotations

import logging
from pathlib import Path

from meshing_utils.cad.step_loader import (
    explore_solids,
    load_solids_with_names,
    read_step_solid_names,
    read_step_unit,
    read_step_xcaf,
)
from meshing_utils.geometry.hex_topology import HexCandidate, PointPool
from meshing_utils.operations.stp_pipeline.edges import classify_edge

logger = logging.getLogger(__name__)


def _read_step_solid_names(path: Path) -> list:
    """Wrapper — see :func:`meshing_utils.cad.step_loader.read_step_solid_names`."""
    return read_step_solid_names(path)


def _explore_solids(shape) -> list:
    """Wrapper — see :func:`meshing_utils.cad.step_loader.explore_solids`."""
    return explore_solids(shape)


def _is_hex_topology(solid) -> bool:
    """Return ``True`` when *solid* has exactly 6 faces, 12 edges, 8 vertices."""
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_VERTEX
    from OCP.TopExp import TopExp_Explorer

    def _count(shape, topo_type) -> int:
        seen: set = set()
        exp = TopExp_Explorer(shape, topo_type)
        while exp.More():
            seen.add(hash(exp.Current()))
            exp.Next()
        return len(seen)

    n_faces = _count(solid, TopAbs_FACE)
    if n_faces != 6:
        return False
    n_edges = _count(solid, TopAbs_EDGE)
    if n_edges != 12:
        return False
    n_verts = _count(solid, TopAbs_VERTEX)
    return n_verts == 8


def _read_step_xcaf(path: Path) -> list:
    """Wrapper — see :func:`meshing_utils.cad.step_loader.read_step_xcaf`."""
    return read_step_xcaf(path)


def _ordered_edge_vertices(edge, orient):
    """Return ``(first_vertex, last_vertex)`` respecting edge orientation."""
    from OCP.TopExp import TopExp as _TopExp
    from OCP.TopoDS import TopoDS

    oriented_edge = TopoDS.Edge_s(edge.Oriented(orient))
    v_first = _TopExp.FirstVertex_s(oriented_edge, True)
    v_last = _TopExp.LastVertex_s(oriented_edge, True)
    return v_first, v_last


def _solid_to_hex_candidate(solid, label: str | None, pool: PointPool) -> HexCandidate:
    """Convert an OCC solid (validated as hex topology) to a ``HexCandidate``."""
    from OCP.BRep import BRep_Tool
    from OCP.BRepTools import BRepTools as _BRepTools
    from OCP.BRepTools import BRepTools_WireExplorer
    from OCP.TopAbs import (
        TopAbs_FACE,
        TopAbs_REVERSED,
        TopAbs_VERTEX,
    )
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    vert_exp = TopExp_Explorer(solid, TopAbs_VERTEX)
    seen_hashes: set = set()
    while vert_exp.More():
        v = TopoDS.Vertex_s(vert_exp.Current())
        h = hash(v)
        if h not in seen_hashes:
            seen_hashes.add(h)
            pnt = BRep_Tool.Pnt_s(v)
            pool.add_or_get((pnt.X(), pnt.Y(), pnt.Z()))
        vert_exp.Next()

    def _vertex_index(v) -> int:
        pnt = BRep_Tool.Pnt_s(v)
        return pool.add_or_get((pnt.X(), pnt.Y(), pnt.Z()))

    face_vertex_lists: list[list[int]] = []
    edge_set_dedup: dict = {}
    edge_curves: dict = {}

    face_exp = TopExp_Explorer(solid, TopAbs_FACE)
    while face_exp.More():
        face = TopoDS.Face_s(face_exp.Current())
        face_orient = face.Orientation()

        outer_wire = _BRepTools.OuterWire_s(face)
        wire_exp = BRepTools_WireExplorer(outer_wire, face)
        face_verts: list[int] = []
        while wire_exp.More():
            edge = TopoDS.Edge_s(wire_exp.Current())
            edge_orient = wire_exp.Orientation()

            v_first, v_last = _ordered_edge_vertices(edge, edge_orient)
            idx_first = _vertex_index(v_first)
            idx_last = _vertex_index(v_last)

            key = frozenset({idx_first, idx_last})
            if key not in edge_set_dedup:
                edge_set_dedup[key] = (idx_first, idx_last)
                try:
                    curve_info = classify_edge(edge)
                    if curve_info.kind != "line":
                        edge_curves[key] = curve_info
                except Exception:
                    # A single malformed edge must not abort the whole STEP
                    # load, but the failure must be visible: the edge is
                    # silently downgraded to a straight line, which can
                    # distort the resulting mesh.
                    logger.warning(
                        "Failed to classify edge between vertices %d and %d; "
                        "treating it as a straight line.",
                        idx_first,
                        idx_last,
                        exc_info=True,
                    )

            if not face_verts:
                face_verts.append(idx_first)
            face_verts.append(idx_last)

            wire_exp.Next()

        if face_verts and face_verts[0] == face_verts[-1]:
            face_verts = face_verts[:-1]

        if face_orient == TopAbs_REVERSED:
            face_verts = list(reversed(face_verts))

        face_vertex_lists.append(face_verts)
        face_exp.Next()

    all_vertex_indices_set: set = set()
    for fv in face_vertex_lists:
        all_vertex_indices_set.update(fv)
    vertex_indices = sorted(all_vertex_indices_set)

    edges_list = [(a, b) for (a, b) in edge_set_dedup.values()]
    faces = [tuple(fv) for fv in face_vertex_lists]

    return HexCandidate(
        vertex_indices=vertex_indices,
        faces=faces,
        edges=edges_list,
        edge_curves=edge_curves,
        label=label,
    )


def load_step(path: Path) -> tuple[list[HexCandidate], PointPool, str]:
    """Load a STEP file and return ``(candidates, pool, unit)``.

    Only solids with exactly 6 faces, 12 edges, and 8 vertices are included.
    Non-hexahedral solids are silently skipped (and counted in a debug log).
    Requires OCP; raises ``ImportError`` with a helpful message when absent.
    """
    try:
        from OCP.IFSelect import IFSelect_RetDone  # noqa: F401
        from OCP.STEPControl import STEPControl_Reader  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "OCP (cadquery-ocp) is required for load_step. "
            "Install it via: pip install cadquery-ocp"
        ) from exc

    path = Path(path)
    unit = read_step_unit(path)
    pool = PointPool(tol=1e-6)

    named_solids = load_solids_with_names(path)
    solid_label_pairs = [(ns.solid, ns.name) for ns in named_solids]
    logger.info("Loaded %d solid(s)", len(named_solids))
    logger.debug(
        "Name sources: %s",
        ", ".join(ns.source for ns in named_solids),
    )

    candidates: list[HexCandidate] = []
    skipped = 0
    for solid, label in solid_label_pairs:
        if not _is_hex_topology(solid):
            skipped += 1
            continue
        candidates.append(_solid_to_hex_candidate(solid, label, pool))

    if skipped > 0:
        logger.info("Skipped %d non-hexahedral solid(s) in %s", skipped, path.name)

    if not candidates:
        raise RuntimeError(f"No hexahedral solids found in STEP file: {path}")

    return candidates, pool, unit
