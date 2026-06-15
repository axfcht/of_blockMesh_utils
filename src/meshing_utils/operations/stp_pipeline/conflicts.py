"""Edge-conflict detection when registering curved edges into a ``BlockMeshDict``."""

from __future__ import annotations

import logging

from meshing_utils.foam.dict_file import BlockMeshDict
from meshing_utils.foam.elements import Edge
from meshing_utils.geometry.hex_topology import CurveInfo


def _squared_distance(
    p: tuple[float, float, float],
    q: tuple[float, float, float],
) -> float:
    """Squared Euclidean distance between two 3D points."""
    dx = p[0] - q[0]
    dy = p[1] - q[1]
    dz = p[2] - q[2]
    return dx * dx + dy * dy + dz * dz


def _format_line_vs_curved_conflict(
    name_a: str,
    name_b: str,
    coord_a: tuple[float, float, float],
    coord_b: tuple[float, float, float],
    existing_type: str,
    block_name: str,
    owner_block: str,
) -> str:
    """Return a human-readable message for a line-vs-existing-curved conflict."""
    ca = f"({coord_a[0]:.4f},{coord_a[1]:.4f},{coord_a[2]:.4f})"
    cb = f"({coord_b[0]:.4f},{coord_b[1]:.4f},{coord_b[2]:.4f})"
    return (
        f"Edge conflict at ({name_a}@{ca}, {name_b}@{cb}): "
        f"block '{block_name}' uses a STRAIGHT edge here, "
        f"but edge already registered as '{existing_type}' by block '{owner_block}'. "
        f"-> '{block_name}' will silently inherit the curvature in OpenFOAM blockMesh."
    )


def _format_curved_vs_curved_conflict(
    name_a: str,
    name_b: str,
    coord_a: tuple[float, float, float],
    coord_b: tuple[float, float, float],
    existing_type: str,
    new_kind: str,
    block_name: str,
    owner_block: str,
) -> str:
    """Return a human-readable message for a curved-vs-existing-curved conflict."""
    ca = f"({coord_a[0]:.4f},{coord_a[1]:.4f},{coord_a[2]:.4f})"
    cb = f"({coord_b[0]:.4f},{coord_b[1]:.4f},{coord_b[2]:.4f})"
    return (
        f"Edge conflict at ({name_a}@{ca}, {name_b}@{cb}): "
        f"block '{block_name}' tries to add a '{new_kind}' edge, "
        f"but edge already registered as '{existing_type}' by block '{owner_block}'. "
        f"-> New edge will NOT be added; existing edge is kept."
    )


def add_edge_with_conflict_check(
    bmd: BlockMeshDict,
    name_a: str,
    name_b: str,
    coord_a: tuple[float, float, float],
    coord_b: tuple[float, float, float],
    curve_info: CurveInfo,
    block_name: str,
    edge_origin: dict[frozenset[str], str],
    strict: bool,
    logger: logging.Logger,
) -> None:
    """Add a curved edge to *bmd*, checking for conflicts.

    Also detects conflicts where a block uses a straight (implicit) edge at a
    location where a curved edge has already been registered by another block.
    In that case the new block would silently inherit the curvature in
    OpenFOAM ``blockMesh`` — this function makes the situation visible via a
    warning (or raises ``ValueError`` when *strict* is ``True``).

    *edge_origin* tracks which block first registered each curved edge.
    Pre-existing edges from an append-mode load should be pre-populated with
    the sentinel string ``"<pre-existing>"``.
    """
    key: frozenset[str] = frozenset({name_a, name_b})
    existing = bmd.find_edge(name_a, name_b)

    if curve_info.kind == "line":
        if existing is not None:
            owner = edge_origin.get(key, "<unknown>")
            msg = _format_line_vs_curved_conflict(
                name_a=name_a,
                name_b=name_b,
                coord_a=coord_a,
                coord_b=coord_b,
                existing_type=existing.type,
                block_name=block_name,
                owner_block=owner,
            )
            if strict:
                raise ValueError(msg)
            logger.warning(msg)
        return

    if existing is not None:
        owner = edge_origin.get(key, "<unknown>")
        msg = _format_curved_vs_curved_conflict(
            name_a=name_a,
            name_b=name_b,
            coord_a=coord_a,
            coord_b=coord_b,
            existing_type=existing.type,
            new_kind=curve_info.kind,
            block_name=block_name,
            owner_block=owner,
        )
        if strict:
            raise ValueError(msg)
        logger.warning(msg)
        return

    if curve_info.kind == "arc":
        mp = curve_info.arc_midpoint
        if mp is None:
            raise ValueError(f"Arc edge ({name_a}, {name_b}) has no arc_midpoint.")
        edge = Edge(
            type_or_string="arc",
            v_start=name_a,
            v_end=name_b,
            coords=list(mp),
        )
    else:
        if not curve_info.support_points:
            return
        sp = list(curve_info.support_points)
        d_first_to_a = _squared_distance(sp[0], coord_a)
        d_first_to_b = _squared_distance(sp[0], coord_b)
        if d_first_to_b < d_first_to_a:
            sp.reverse()
        edge = Edge(
            type_or_string="BSpline",
            v_start=name_a,
            v_end=name_b,
            points=[list(p) for p in sp],
        )

    bmd.edges.add(edge)
    edge_origin[key] = block_name
