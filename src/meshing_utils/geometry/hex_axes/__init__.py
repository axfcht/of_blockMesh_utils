"""Hex-axes subpackage.

Re-exports every symbol previously exposed at
``meshing_utils.geometry.hex_axes``.

Sub-modules:

* :mod:`.detection` — pure-geometry hex face detection, axis
                      classification, ``TopologyError``,
                      ``_SimpleUnionFind``.
* :mod:`.axes`      — :class:`BlockAxis`, per-block axis construction,
                      longest-edges helper, :class:`AxisEquivalenceClasses`
                      and its weighted ``_UnionFind``.
"""

from __future__ import annotations

from meshing_utils.geometry.hex_axes.axes import (
    AxisEquivalenceClasses,
    BlockAxis,
    _UnionFind,
    build_block_axes,
    compute_longest_edges_per_axis,
)
from meshing_utils.geometry.hex_axes.detection import (
    _TOLERANCE_LEVELS,
    TopologyError,
    _all_points_coplanar,
    _dominant_axis,
    _is_coplanar,
    _max_diagonal,
    _order_quad_ccw,
    _SimpleUnionFind,
    _validate_hex_topology,
    compute_axis_class_per_edge,
    detect_hex_faces,
    map_class_to_local_axis_index,
)

__all__ = [
    "AxisEquivalenceClasses",
    "BlockAxis",
    "TopologyError",
    "build_block_axes",
    "compute_axis_class_per_edge",
    "compute_longest_edges_per_axis",
    "detect_hex_faces",
    "map_class_to_local_axis_index",
]
