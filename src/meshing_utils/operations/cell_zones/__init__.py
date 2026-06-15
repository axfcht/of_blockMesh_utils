"""Cell-zone assignment subpackage.

Re-exports every symbol previously exposed at the module path
``meshing_utils.operations.cell_zones``.

Sub-modules:

* :mod:`.constants`      — sampling/vote strings and :class:`BlockSolidCounts`.
* :mod:`.naming`         — zone-name sanitisation and uniqueness allocation.
* :mod:`.classification` — sample generation, per-solid classification,
                          majority vote, perturbation fallback.
* :mod:`.resolution`     — Stage A (vertex-dominant) and Stage B
                          (interior-fallback) winner selection.
* :mod:`.core`           — :func:`assign_cell_zones` orchestrator.
"""

from __future__ import annotations

from meshing_utils.operations.cell_zones.classification import (
    _classify_block_two_pass,
    _classify_samples_against_solid,
    _compute_samples_for_block,
    _majority_vote,
    _perturbation_fallback,
)
from meshing_utils.operations.cell_zones.constants import (
    SAMPLING_CENTROID,
    SAMPLING_INSET,
    STAGE_A_NOT_APPLICABLE,
    VALID_SAMPLING_STRATEGIES,
    VALID_VOTE_POLICIES,
    VOTE_MAJORITY,
    BlockSolidCounts,
    _StageANotApplicable,
)
from meshing_utils.operations.cell_zones.core import assign_cell_zones
from meshing_utils.operations.cell_zones.naming import (
    _assign_unique_zone_names,
    _sanitize_zone_name,
)
from meshing_utils.operations.cell_zones.resolution import (
    _resolve_interior_candidates,
    _resolve_vertex_dominant,
)

__all__ = [
    "SAMPLING_CENTROID",
    "SAMPLING_INSET",
    "VOTE_MAJORITY",
    "BlockSolidCounts",
    "assign_cell_zones",
]
