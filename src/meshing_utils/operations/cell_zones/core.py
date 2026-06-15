"""Public :func:`assign_cell_zones` orchestrator."""

from __future__ import annotations

import logging

from meshing_utils.geometry.containment import (
    AABB,
    DEFAULT_INSET_FACTOR,
    bmd_bbox_diagonal,
    compute_solid_aabb,
    make_solid_classifier,
    resolve_block_coords,
)
from meshing_utils.operations.cell_zones.classification import (
    _classify_block_two_pass,
    _compute_samples_for_block,
)
from meshing_utils.operations.cell_zones.constants import (
    SAMPLING_INSET,
    STAGE_A_NOT_APPLICABLE,
    VALID_SAMPLING_STRATEGIES,
    VALID_VOTE_POLICIES,
    VOTE_MAJORITY,
)
from meshing_utils.operations.cell_zones.naming import _assign_unique_zone_names
from meshing_utils.operations.cell_zones.resolution import (
    _resolve_interior_candidates,
    _resolve_vertex_dominant,
)

logger = logging.getLogger(__name__)


def assign_cell_zones(
    bmd,
    solid_label_pairs: list[tuple],
    *,
    tol: float = 1e-7,
    strict: bool = False,
    epsilon: float | None = None,
    sampling_strategy: str = SAMPLING_INSET,
    inset_factor: float = DEFAULT_INSET_FACTOR,
    vote_policy: str = VOTE_MAJORITY,
    use_aabb_filter: bool = True,
) -> dict[str, str]:
    """Assign zone names to blocks by testing sample-point containment in solids.

    See :mod:`meshing_utils.operations.cell_zones` for a description of
    the two sampling strategies (``"centroid"`` / ``"inset"``) and the
    two-stage decision used by the ``"inset"`` strategy.

    Updates ``block.zone`` in place and returns a mapping
    ``block_name -> zone_name``. Blocks without a match are absent from
    the mapping and remain unzoned (no exception even in strict mode).

    Raises :class:`ValueError` for invalid *sampling_strategy*,
    *vote_policy*, or *inset_factor*, and :class:`RuntimeError` when
    ``strict=True`` and an ambiguous assignment is encountered.
    """
    if sampling_strategy not in VALID_SAMPLING_STRATEGIES:
        raise ValueError(
            f"sampling_strategy must be one of {VALID_SAMPLING_STRATEGIES!r}, "
            f"got {sampling_strategy!r}"
        )
    if vote_policy not in VALID_VOTE_POLICIES:
        raise ValueError(
            f"vote_policy must be one of {VALID_VOTE_POLICIES!r}, "
            f"got {vote_policy!r}"
        )

    if epsilon is None:
        epsilon = max(bmd_bbox_diagonal(bmd) * 1e-9, 1e-12)

    if sampling_strategy == SAMPLING_INSET:
        logger.info(
            "Sampling: inset (factor=%g, raw vertices=8), vote: %s",
            inset_factor,
            vote_policy,
        )
    else:
        logger.info(
            "Sampling: %s (factor=N/A), vote: %s",
            sampling_strategy,
            vote_policy,
        )

    zone_names = _assign_unique_zone_names(solid_label_pairs)

    if use_aabb_filter and solid_label_pairs:
        solid_aabbs: list[AABB] | None = [
            compute_solid_aabb(solid, tol) for solid, _ in solid_label_pairs
        ]
    else:
        solid_aabbs = None

    classifiers: list = [
        make_solid_classifier(solid) for solid, _ in solid_label_pairs
    ]

    mapping: dict[str, str] = {}

    for block in bmd.blocks:
        if len(block.vertices) != 8:
            logger.warning(
                "Block %r has %d vertices — skipped.",
                block.name,
                len(block.vertices),
            )
            continue

        coords = resolve_block_coords(block, bmd)
        interior_samples, vertex_samples = _compute_samples_for_block(
            coords, sampling_strategy, inset_factor
        )
        counts = _classify_block_two_pass(
            interior_samples,
            vertex_samples,
            solid_label_pairs,
            tol,
            solid_aabbs=solid_aabbs,
            classifiers=classifiers,
        )

        logger.debug(
            "Block %r: per-solid BlockSolidCounts = %s",
            block.name,
            counts,
        )

        chosen_idx = _resolve_vertex_dominant(block, counts, zone_names, strict)

        if chosen_idx is STAGE_A_NOT_APPLICABLE:
            chosen_idx = _resolve_interior_candidates(
                block,
                interior_samples,
                counts,
                solid_label_pairs,
                zone_names,
                tol,
                epsilon,
                strict,
            )

        if chosen_idx is None:
            logger.info(
                "Block %r: no containing solid — leaving unzoned", block.name
            )
            continue

        logger.debug(
            "Block %r: chosen solid index %d (%s)",
            block.name,
            chosen_idx,
            zone_names[chosen_idx],
        )
        block.zone = zone_names[chosen_idx]
        block_key = block.name if block.name else f"<block@{id(block)}>"
        mapping[block_key] = block.zone

    return mapping
