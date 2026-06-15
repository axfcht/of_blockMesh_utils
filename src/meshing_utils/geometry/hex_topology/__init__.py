"""Hex-topology subpackage.

Re-exports every symbol previously exposed at
``meshing_utils.geometry.hex_topology``.

Sub-modules:

* :mod:`.core`       — data models (:class:`CurveInfo`,
                       :class:`HexCandidate`), :class:`PointPool`,
                       exceptions, ``HEX_FACE_INDICES`` /
                       ``HEX_FACE_NAMES``, vector helpers.
* :mod:`.algorithms` — validation, ordering, OpenFOAM face convention,
                       outward-normal checks, global face consistency.
"""

from __future__ import annotations

from meshing_utils.geometry.hex_topology.algorithms import (
    _cyclic_variants,
    _reversed_cyclic_variants,
    assert_block_face_normals_outward,
    assert_hex_outward_from_coords,
    check_global_face_consistency,
    enforce_openfoam_face_convention,
    ensure_right_handed,
    order_hex_vertices,
    validate_hex,
)
from meshing_utils.geometry.hex_topology.core import (
    EPSILON_ABS,
    EPSILON_REL,
    HEX_FACE_INDICES,
    HEX_FACE_NAMES,
    CurveInfo,
    HexCandidate,
    HexValidationError,
    OrderingConsistencyError,
    PointPool,
    _face_centroid,
    _face_normal,
    _solid_centroid,
)

__all__ = [
    "HEX_FACE_INDICES",
    "HEX_FACE_NAMES",
    "CurveInfo",
    "HexCandidate",
    "HexValidationError",
    "OrderingConsistencyError",
    "PointPool",
    "assert_block_face_normals_outward",
    "assert_hex_outward_from_coords",
    "check_global_face_consistency",
    "enforce_openfoam_face_convention",
    "ensure_right_handed",
    "order_hex_vertices",
    "validate_hex",
]
