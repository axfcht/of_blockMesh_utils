"""Per-layer vertex, edge, and block builders used by ``extrude_with_steps``."""

from __future__ import annotations

from meshing_utils.foam.elements import Block, Edge, Face, Vertex
from meshing_utils.foam.elements.patch import Boundary
from meshing_utils.operations.extrusion.exceptions import ExtrusionError
from meshing_utils.operations.extrusion.markers import (
    EXTRUSION_LOCAL_FACE_INDICES,
    _source_axis_of_diff,
)
from meshing_utils.operations.extrusion.parsing import LayerStep, _vec_add


def _layer_vertex_name(original_name: str, layer_index: int) -> str:
    """Build the name for an extruded vertex: ``<original>e<layer>``."""
    return f"{original_name}e{layer_index}"


def _build_skip_aware_layer_vertices(
    source_vertices: list[Vertex],
    steps: list[LayerStep],
) -> tuple[
    list[tuple[int, int]],
    dict,
    set[int],
]:
    """Build vertices for skip-aware extrusion.

    Returns ``(block_layer_pairs, layer_vertices, used_layer_indices)``.
    Layer index 0 always refers to the original source vertices.
    """
    running: tuple[float, float, float] = (0.0, 0.0, 0.0)
    used_offsets_ordered: list[tuple[float, float, float]] = []
    block_bottom_top: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = []

    for step in steps:
        bottom_abs = _vec_add(running, step.skip_offset)
        top_abs = _vec_add(bottom_abs, step.block_delta)
        block_bottom_top.append((bottom_abs, top_abs))
        if bottom_abs not in used_offsets_ordered:
            used_offsets_ordered.append(bottom_abs)
        if top_abs not in used_offsets_ordered:
            used_offsets_ordered.append(top_abs)
        running = top_abs

    _ZERO: tuple[float, float, float] = (0.0, 0.0, 0.0)
    offset_to_layer: dict = {}
    next_idx = 1
    for off in used_offsets_ordered:
        if off == _ZERO:
            offset_to_layer[off] = 0
        else:
            if off not in offset_to_layer:
                offset_to_layer[off] = next_idx
                next_idx += 1

    layer_vertices: dict = {}
    for off, layer_idx in offset_to_layer.items():
        if layer_idx == 0:
            continue
        dx, dy, dz = off
        layer: list[Vertex] = []
        for v in source_vertices:
            new_name = _layer_vertex_name(v.name, layer_idx)
            new_coords = [v.coords[0] + dx, v.coords[1] + dy, v.coords[2] + dz]
            layer.append(Vertex(new_name, new_coords))
        layer_vertices[layer_idx] = layer

    block_layer_pairs: list[tuple[int, int]] = []
    for bottom_abs, top_abs in block_bottom_top:
        block_layer_pairs.append(
            (offset_to_layer[bottom_abs], offset_to_layer[top_abs])
        )

    used_layer_indices = set(offset_to_layer.values())
    return block_layer_pairs, layer_vertices, used_layer_indices


def _build_skip_aware_edges_with_offsets(
    source_edges: list[Edge],
    source_vertex_names: set,
    block_layer_pairs: list[tuple[int, int]],
    layer_offsets: dict[int, tuple[float, float, float]],
) -> list[Edge]:
    """Build edges for skip-aware extrusion (no edges in skip gaps).

    Only edges between marked vertices are copied. Each edge is emitted once
    per layer that belongs to at least one real block (bottom or top).
    """
    new_edges: list[Edge] = []
    emitted: set = set()

    for bottom_idx, top_idx in block_layer_pairs:
        for layer_idx in (bottom_idx, top_idx):
            if layer_idx == 0 or layer_idx in emitted:
                continue
            if layer_idx not in layer_offsets:
                continue
            emitted.add(layer_idx)
            dx, dy, dz = layer_offsets[layer_idx]
            for e in source_edges:
                if e.v_start in source_vertex_names and e.v_end in source_vertex_names:
                    new_v_start = _layer_vertex_name(e.v_start, layer_idx)
                    new_v_end = _layer_vertex_name(e.v_end, layer_idx)
                    new_points = [
                        [pt[0] + dx, pt[1] + dy, pt[2] + dz]
                        for pt in e.points
                    ]
                    new_edges.append(
                        Edge(e.type, new_v_start, new_v_end, points=new_points)
                    )
    return new_edges


def _build_extrusion_block_BLOCK_DRIVEN(
    source_block: Block,
    face_idx: int,
    layer_bottom: int,
    layer_top: int,
    extrusion_block_index: int,
) -> Block:
    """Build a single hex block for the extrusion between two consecutive layers.

    The bottom face comes from ``layer_bottom`` (0 = original) and the top from
    ``layer_top``. Vertex order on the bottom face is taken from
    ``EXTRUSION_LOCAL_FACE_INDICES[face_idx]``; cell counts and grading are mapped from the
    source block's axes onto the new block's local axes.
    """
    local_face_indices = EXTRUSION_LOCAL_FACE_INDICES[face_idx]
    block_verts = source_block.vertices

    bottom_names = [block_verts[i] for i in local_face_indices]
    top_names = [
        _layer_vertex_name(name, layer_top) for name in bottom_names
    ]
    if layer_bottom > 0:
        bottom_names = [
            _layer_vertex_name(name, layer_bottom) for name in bottom_names
        ]

    hex_verts = bottom_names + top_names

    i0, i1, i2, _i3 = local_face_indices
    x1_axis = _source_axis_of_diff(i0, i1)
    x2_axis = _source_axis_of_diff(i1, i2)
    in_plane = {x1_axis, x2_axis}
    if len(in_plane) != 2:
        raise ExtrusionError(
            f"Degenerate face {face_idx}: x1' and x2' map to the same source axis."
        )

    src_cells = source_block.cells
    src_grading = source_block.grading_def

    new_cells = [src_cells[x1_axis], src_cells[x2_axis], 1]
    new_grading = [src_grading[x1_axis], src_grading[x2_axis], 1.0]

    block_name = f"{source_block.name}e{extrusion_block_index}"
    return Block(
        block_name,
        vertices=hex_verts,
        cells=new_cells,
        grading_type=source_block.grading_type,
        grading_def=new_grading,
        zone=source_block.zone,
    )


def _update_boundary_patches(
    boundary: Boundary,
    source_vertex_names: set,
    top_layer_idx: int,
) -> None:
    """Replace faces on the marked layer with top-layer faces in every patch."""
    for patch in boundary:
        new_faces: list[Face] = []
        for face in patch.faces:
            if set(face.vertices) <= source_vertex_names:
                top_verts = [
                    _layer_vertex_name(name, top_layer_idx) for name in face.vertices
                ]
                new_faces.append(Face(top_verts))
            else:
                new_faces.append(face)
        patch.faces = new_faces
