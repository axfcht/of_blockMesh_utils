"""Extrusion subpackage.

Re-exports every symbol the previous single-module path
``meshing_utils.operations.extrusion`` exposed so existing imports keep
working unchanged.

Sub-modules:

* :mod:`.exceptions` — exception hierarchy.
* :mod:`.parsing` — ``VectorToken``, ``LayerStep``, ``ExtrusionRequest``,
  offset-string parsers.
* :mod:`.markers` — extrusion-local ``EXTRUSION_LOCAL_FACE_INDICES`` convention plus marker
  discovery and stripping helpers.
* :mod:`.validation` — planarity and outward-normal checks.
* :mod:`.builders` — per-layer vertex / edge / block builders.
* :mod:`.core` — public ``extrude`` / ``extrude_with_steps`` entry points.
"""

from __future__ import annotations

from meshing_utils.operations.extrusion.builders import (
    _build_extrusion_block_BLOCK_DRIVEN,
    _build_skip_aware_edges_with_offsets,
    _build_skip_aware_layer_vertices,
    _layer_vertex_name,
    _update_boundary_patches,
)
from meshing_utils.operations.extrusion.core import extrude, extrude_with_steps
from meshing_utils.operations.extrusion.exceptions import (
    AmbiguousFaceError,
    ExtrusionError,
    NoMarkersFoundError,
    NonCoplanarVerticesError,
)
from meshing_utils.operations.extrusion.markers import (
    EXTRUSION_LOCAL_FACE_INDICES,
    _collect_marked_blocks,
    _collect_marked_vertices,
    _discover_blocks_with_marked_face,
    _find_face_index_in_block,
    _source_axis_of_diff,
    _strip_all_markers,
)
from meshing_utils.operations.extrusion.parsing import (
    ExtrusionRequest,
    LayerStep,
    ParseError,
    VectorToken,
    build_layer_steps,
    cumulative_offsets,
    parse_layer_steps,
    parse_offsets,
)
from meshing_utils.operations.extrusion.validation import (
    _assert_extrusion_block_outward,
    _check_coplanar,
)

__all__ = [
    "AmbiguousFaceError",
    "ExtrusionError",
    "ExtrusionRequest",
    "LayerStep",
    "NoMarkersFoundError",
    "NonCoplanarVerticesError",
    "ParseError",
    "VectorToken",
    "build_layer_steps",
    "cumulative_offsets",
    "extrude",
    "extrude_with_steps",
    "parse_layer_steps",
    "parse_offsets",
]
