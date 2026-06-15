"""Block-face / STEP-surface matching subpackage.

Re-exports every symbol previously exposed at the module path
``meshing_utils.cad.face_matching`` so existing imports keep working.

Sub-modules:

* :mod:`.geometry`   — :class:`BlockFace`, ``compute_outward_normal``,
                       ``extract_block_faces``.
* :mod:`.matching`   — OCC distance queries, voting, dominant-face
                       selection, ``local_surface_normal``.
* :mod:`.tolerances` — surface-type queries, adaptive normal tolerance,
                       ``normals_consistent``.
"""

from __future__ import annotations

from meshing_utils.cad.face_matching.geometry import (
    BlockFace,
    compute_outward_normal,
    extract_block_faces,
)
from meshing_utils.cad.face_matching.matching import (
    _centroid,
    _majority_vote_face,
    _try_get_face,
    find_dominant_face,
    local_surface_normal,
    nearest_face_within_tol,
    pick_adjacent_face,
)
from meshing_utils.cad.face_matching.tolerances import (
    _SURFACE_TYPE_MAP,
    effective_normal_tolerance,
    normals_consistent,
    surface_type_of,
)

__all__ = [
    "BlockFace",
    "compute_outward_normal",
    "effective_normal_tolerance",
    "extract_block_faces",
    "find_dominant_face",
    "local_surface_normal",
    "nearest_face_within_tol",
    "normals_consistent",
    "pick_adjacent_face",
    "surface_type_of",
]
