"""Public ``extrude`` / ``extrude_with_steps`` entry points."""

from __future__ import annotations

import warnings

from meshing_utils.foam.dict_file import BlockMeshDict
from meshing_utils.operations.extrusion.builders import (
    _build_extrusion_block_BLOCK_DRIVEN,
    _build_skip_aware_edges_with_offsets,
    _build_skip_aware_layer_vertices,
    _update_boundary_patches,
)
from meshing_utils.operations.extrusion.exceptions import NoMarkersFoundError
from meshing_utils.operations.extrusion.markers import (
    EXTRUSION_LOCAL_FACE_INDICES,
    _collect_marked_blocks,
    _collect_marked_vertices,
    _discover_blocks_with_marked_face,
    _find_face_index_in_block,
    _strip_all_markers,
)
from meshing_utils.operations.extrusion.parsing import LayerStep
from meshing_utils.operations.extrusion.validation import (
    _assert_extrusion_block_outward,
    _check_coplanar,
)


def extrude(bmd: BlockMeshDict, offsets: list[tuple[float, float, float]]) -> BlockMeshDict:
    """Extrude the marked face(s) in ``bmd`` along the given incremental
    offset vectors. Each offset is a NORMAL displacement that produces one
    block; use :func:`extrude_with_steps` for skip markers.

    Returns a *new* ``BlockMeshDict``; the original is not modified.
    """
    steps = [LayerStep(skip_offset=(0.0, 0.0, 0.0), block_delta=off) for off in offsets]
    return extrude_with_steps(bmd, steps)


def extrude_with_steps(bmd: BlockMeshDict, steps: list[LayerStep]) -> BlockMeshDict:
    """Extrude the marked face(s) in ``bmd`` using skip-aware :class:`LayerStep` objects.

    Implements the full skip-marker semantics: vertices and edges are only
    created for layers that are the bottom or top of at least one real block
    (R3), ``cells`` / ``grading`` per block are derived exclusively from
    ``step.block_delta`` (R1), and skip gaps receive no boundary patches (R2).

    Returns a *new* ``BlockMeshDict``; the original is not modified.
    """
    import copy
    result = copy.deepcopy(bmd)

    marked_vertices = _collect_marked_vertices(result)
    if not marked_vertices:
        raise NoMarkersFoundError(
            "No marked vertices (//*) found in blockMeshDict. "
            "Vertex lines must end with '//*' or '//* <label>'."
        )

    _check_coplanar(marked_vertices)

    marked_vertex_names = {v.name for v in marked_vertices}

    block_layer_pairs, layer_vertices_dict, _used_layer_indices = (
        _build_skip_aware_layer_vertices(marked_vertices, steps)
    )

    for layer_idx in sorted(layer_vertices_dict.keys()):
        for v in layer_vertices_dict[layer_idx]:
            result.vertices.add(v)

    layer_offsets: dict = {}
    for layer_idx, vlist in layer_vertices_dict.items():
        first_new = vlist[0]
        suffix = f"e{layer_idx}"
        src_name = first_new.name[: -len(suffix)]
        src_vertex = None
        for mv in marked_vertices:
            if mv.name == src_name:
                src_vertex = mv
                break
        if src_vertex is not None:
            layer_offsets[layer_idx] = (
                first_new.coords[0] - src_vertex.coords[0],
                first_new.coords[1] - src_vertex.coords[1],
                first_new.coords[2] - src_vertex.coords[2],
            )

    all_source_edges = list(result.edges)
    new_edges = _build_skip_aware_edges_with_offsets(
        all_source_edges, marked_vertex_names, block_layer_pairs, layer_offsets
    )
    for e in new_edges:
        result.edges.add(e)

    marked_blocks = _collect_marked_blocks(result)

    if marked_blocks:
        for source_block in marked_blocks:
            face_idx = _find_face_index_in_block(source_block, marked_vertex_names)
            for block_num, (layer_bottom, layer_top) in enumerate(block_layer_pairs, start=1):
                new_block = _build_extrusion_block_BLOCK_DRIVEN(
                    source_block,
                    face_idx,
                    layer_bottom,
                    layer_top,
                    block_num,
                )
                _assert_extrusion_block_outward(new_block, result, new_block.name)
                result.blocks.add(new_block)
    else:
        source_pairs = _discover_blocks_with_marked_face(
            list(result.blocks), marked_vertex_names
        )
        if not source_pairs:
            raise NoMarkersFoundError(
                "No block contains a face whose 4 vertices are all marked. "
                "Ensure 4 vertices of an existing hex block share the //* marker."
            )
        used_names: set = set()
        for block, face_idx in source_pairs:
            for local_idx in EXTRUSION_LOCAL_FACE_INDICES[face_idx]:
                used_names.add(block.vertices[local_idx])
        unused = marked_vertex_names - used_names
        if unused:
            warnings.warn(
                f"Marked vertices not part of any discovered face (ignored): "
                f"{sorted(unused)}",
                UserWarning,
                stacklevel=2,
            )
        for source_block, face_idx in source_pairs:
            for block_num, (layer_bottom, layer_top) in enumerate(block_layer_pairs, start=1):
                new_block = _build_extrusion_block_BLOCK_DRIVEN(
                    source_block,
                    face_idx,
                    layer_bottom,
                    layer_top,
                    block_num,
                )
                _assert_extrusion_block_outward(new_block, result, new_block.name)
                result.blocks.add(new_block)

    top_layer_idx = block_layer_pairs[-1][1] if block_layer_pairs else 1

    _update_boundary_patches(result.boundary, marked_vertex_names, top_layer_idx)

    _strip_all_markers(result)

    return result
